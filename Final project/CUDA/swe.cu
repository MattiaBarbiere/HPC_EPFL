#include "swe.hh"
#include "xdmf_writer.hh"
#include <iostream>
#include <cstddef>
#include <vector>
#include <string>
#include <cassert>
#include <hdf5.h>
#include <hdf5_hl.h>
#include <cstdio>
#include <cmath>
#include <memory>

// A struct to keep the data organized
struct SWEData {
    double* h0;
    double* hu0;
    double* hv0;
    double* h1;
    double* hu1;
    double* hv1;
    double* z;
    double* zdx;
    double* zdy;
    size_t nx;
    size_t ny;
    double size_x;
    double size_y;
};

// %%%%%%%%%%%%%%%%%%%%%%%%
// %%% Device functions %%%
// %%%%%%%%%%%%%%%%%%%%%%%%

// Function to get the row ID in a 2D block
__device__
int get_row_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

// Function to get the column ID in a 2D block
__device__
int get_col_id() {
  return blockIdx.y * blockDim.y + threadIdx.y;
}

// Function to get the unique thread ID across the entire grid
__device__
int get_id() {
  int id_in_block = threadIdx.y * blockDim.x + threadIdx.x;
  int offset = blockIdx.y * gridDim.x + blockIdx.x;
  return offset * blockDim.x * blockDim.y + id_in_block;
}

// Function identical to at() but for device code
__device__ 
double& at_device(SWEData* data, std::size_t i, std::size_t j, double* arr)
{
    return arr[j * data->nx + i];
}

// Function to initialize dx and dy arrays based on the grid size
__device__
void init_dx_dy(SWEData* data)
{
    // Compute grid spacing in x and y directions
    const double dx = data->size_x / data->nx;
    const double dy = data->size_y / data->ny;

    // Loop over interior grid points (excluding boundaries)
    for (std::size_t j = 1; j < data->ny - 1; ++j)
    {
        for (std::size_t i = 1; i < data->nx - 1; ++i)
        {
            // Central difference for topography gradient in x
            at_device(data, i, j, data->zdx) = 0.5 * (at_device(data, i + 1, j, data->z) - at_device(data, i - 1, j, data->z)) / dx;
            // Central difference for topography gradient in y
            at_device(data, i, j, data->zdy) = 0.5 * (at_device(data, i, j + 1, data->z) - at_device(data, i, j - 1, data->z)) / dy;
        }
    }
}

// Function to compute the kernel for a single cell in the SWE solver
__device__
void compute_kernel_device(SWEData* data, const std::size_t i, const std::size_t j, const double dt)
{  
    // Gravitational acceleration
    static constexpr double g = 127267.20000000;
    
    // Compute grid spacing and coefficients for the update
    const double dx = data->size_x / data->nx;
    const double dy = data->size_y / data->ny;
    const double C1x = 0.5 * dt / dx;
    const double C1y = 0.5 * dt / dy;
    const double C2 = dt * g;
    constexpr double C3 = 0.5 * g;

    // Compute updated water height using a 2D stencil and fluxes
    double hij = 0.25 * (at_device(data, i, j - 1, data->h0) + at_device(data, i, j + 1, data->h0) + at_device(data, i - 1, j, data->h0) + at_device(data, i + 1, j, data->h0))
                + C1x * (at_device(data, i - 1, j, data->hu0) - at_device(data, i + 1, j, data->hu0)) + C1y * (at_device(data, i, j - 1, data->hv0) - at_device(data, i, j + 1, data->hv0));
    
    // Ensure non-negative water height
    if (hij < 0.0)
    {
        hij = 1.0e-5;
    }

    // Update water height in the next time step
    at_device(data, i, j, data->h1) = hij;

    // Update velocity components if water height is above threshold
    if (hij > 0.0001)
    {
        at_device(data, i, j, data->hu1) =
          0.25 * (at_device(data, i, j - 1, data->hu0) + at_device(data, i, j + 1, data->hu0) + at_device(data, i - 1, j, data->hu0) + at_device(data, i + 1, j, data->hu0)) - C2 * hij * at_device(data, i, j, data->zdx)
          + C1x
              * (at_device(data, i - 1, j, data->hu0) * at_device(data, i - 1, j, data->hu0) / at_device(data, i - 1, j, data->h0) + C3 * at_device(data, i - 1, j, data->h0) * at_device(data, i - 1, j, data->h0)
                - at_device(data, i + 1, j, data->hu0) * at_device(data, i + 1, j, data->hu0) / at_device(data, i + 1, j, data->h0) - C3 * at_device(data, i + 1, j, data->h0) * at_device(data, i + 1, j, data->h0))
          + C1y
              * (at_device(data, i, j - 1, data->hu0) * at_device(data, i, j - 1, data->hv0) / at_device(data, i, j - 1, data->h0)
                - at_device(data, i, j + 1, data->hu0) * at_device(data, i, j + 1, data->hv0) / at_device(data, i, j + 1, data->h0));

        at_device(data, i, j, data->hv1) =
          0.25 * (at_device(data, i, j - 1, data->hv0) + at_device(data, i, j + 1, data->hv0) + at_device(data, i - 1, j, data->hv0) + at_device(data, i + 1, j, data->hv0)) - C2 * hij * at_device(data, i, j, data->zdy)
          + C1x
              * (at_device(data, i - 1, j, data->hu0) * at_device(data, i - 1, j, data->hv0) / at_device(data, i - 1, j, data->h0)
                - at_device(data, i + 1, j, data->hu0) * at_device(data, i + 1, j, data->hv0) / at_device(data, i + 1, j, data->h0))
          + C1y
              * (at_device(data, i, j - 1, data->hv0) * at_device(data, i, j - 1, data->hv0) / at_device(data, i, j - 1, data->h0) + C3 * at_device(data, i, j - 1, data->h0) * at_device(data, i, j - 1, data->h0)
                - at_device(data, i, j + 1, data->hv0) * at_device(data, i, j + 1, data->hv0) / at_device(data, i, j + 1, data->h0) - C3 * at_device(data, i, j + 1, data->h0) * at_device(data, i, j + 1, data->h0));
    }
    else
    {
        at_device(data, i, j, data->hu1) = 0.0;
        at_device(data, i, j, data->hv1) = 0.0;
    }
}


// %%%%%%%%%%%%%%%%%%%%%%%%
// %%% Kernel functions %%%
// %%%%%%%%%%%%%%%%%%%%%%%%

// Kernel to initialize dx and dy
__global__
void init_dx_dy_kernel(SWEData* data)
{ 
    // Only one thread calls this since we write to global memory
    if (get_row_id() == 0 && get_col_id() == 0) {
        // Initialize dx and dy arrays
        init_dx_dy(data);
    }
}

// Kernel to initialize the SWEData structure with a water drop test case
__global__
void init_gaussian_kernel(SWEData* data)
{   
    // Get the row and column indices
    size_t i = get_row_id();
    size_t j = get_col_id();
    size_t index_val = i + j * data->nx;

    // Iterate over the grid
    if (i < data->nx && j < data->ny) {
        data->hu0[index_val] = 0.0;
        data->hv0[index_val] = 0.0;

        const double x0_0 = data->size_x / 4.0;
        const double y0_0 = data->size_y / 3.0;
        const double x0_1 = data->size_x / 2.0;
        const double y0_1 = 0.75 * data->size_y;

        const double dx = data->size_x / data->nx;
        const double dy = data->size_y / data->ny;

        double x = dx * (static_cast<double>(i) + 0.5);
        double y = dy * (static_cast<double>(j) + 0.5);
        double gauss_0 = 10.0 * exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 1000.0);
        double gauss_1 = 10.0 * exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 1000.0);

        // Set the water height and velocity values
        data->h0[index_val] = 10.0 + gauss_0 + gauss_1;
        data->h1[index_val] = 0.0;
        data->hu1[index_val] = 0.0;
        data->hv1[index_val] = 0.0;

        // Set the topography value
        data->z[index_val] = 0.0;
    }

    // Only one thread calls this since we write to global memory
    if (i == 0 && j == 0) 
    { 
        // Initialize dx and dy
        init_dx_dy(data);
    }
}

// Kernel to initialize the SWEData structure with a dummy tsunami test case
__global__
void init_dummy_tsunami_kernel(SWEData* data)
{ 
    // Get the row and column indices
    size_t i = get_row_id();
    size_t j = get_col_id();
    size_t idx = i + j * data->nx;

    // Iterate over the grid
    if (i < data->nx && j < data->ny) {
        // Initialize the water height and velocities
        data->hu0[idx] = 0.0;
        data->hv0[idx] = 0.0;
        data->h1[idx] = 0.0;
        data->hu1[idx] = 0.0;
        data->hv1[idx] = 0.0;

        const double x0_0 = 0.6 * data->size_x;
        const double y0_0 = 0.6 * data->size_y;
        const double x0_1 = 0.4 * data->size_x;
        const double y0_1 = 0.4 * data->size_y;
        const double x0_2 = 0.7 * data->size_x;
        const double y0_2 = 0.3 * data->size_y;

        const double dx = data->size_x / data->nx;
        const double dy = data->size_y / data->ny;

        const double x = dx * (static_cast<double>(i) + 0.5);
        const double y = dy * (static_cast<double>(j) + 0.5);

        const double gauss_0 = 2.0 * exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 3000.0);
        const double gauss_1 = 3.0 * exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 10000.0);
        const double gauss_2 = 5.0 * exp(-((x - x0_2) * (x - x0_2) + (y - y0_2) * (y - y0_2)) / 100.0);

        double z_val = -1.0 + gauss_0 + gauss_1;

        // Set the topography value
        data->z[idx] = z_val;

        // Set the water height based on the topography and Gaussian values
        double h0_val = (z_val < 0.0) ? (-z_val + gauss_2) : 0.00001;
        data->h0[idx] = h0_val;
    }

    // Only one thread calls this
    if (i == 0 && j == 0) {
        init_dx_dy(data);
    }
}

// Kernel to compute the time step for a single block
__global__
void compute_time_step_block_kernel(const SWEData* data, double* max_partial) {
    // Shared memory for saving the values for each thread in the block
    extern __shared__ double shared_max[];

    // Get the thread ID and row/column indices
    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;
    int i = get_row_id();
    int j = get_col_id();
    int block_size = blockDim.x * blockDim.y;

    double max_nu_sqr_local = 0.0;
    static constexpr double g = 127267.20000000;

    // Check if the thread is within the valid range of the grid
    if (i > 0 && i < data->nx - 1 && j > 0 && j < data->ny - 1) {
        int idx = j * data->nx + i;
        double h_val = data->h0[idx];
        double hu_val = data->hu0[idx];
        double hv_val = data->hv0[idx];

        const double nu_u = fabs(hu_val) / h_val + sqrt(g * h_val);
        const double nu_v = fabs(hv_val) / h_val + sqrt(g * h_val);

        // Compute the value for a single thread
        max_nu_sqr_local = nu_u * nu_u + nu_v * nu_v;
    }

    // Store the local maximum in shared memory
    shared_max[thread_id] = max_nu_sqr_local;
    __syncthreads();

    // Finding the maximum value over a single block
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (thread_id < s){
            shared_max[thread_id] = fmax(shared_max[thread_id], shared_max[thread_id + s]);
        }
        __syncthreads();
    }

    // Write the maximum value of a block to the global memory
    if (thread_id == 0){
        max_partial[blockIdx.y * gridDim.x + blockIdx.x] = shared_max[0];
    }
}

// Kernel to take the maximum value over all the blocks
__global__
void compute_time_step_kernel(const SWEData* data, double* max_partial, int length_max,
                            const double T, const double Tend, double* dt_out) {

    // Reduce the maximum value across all blocks from the max_partial array
    double max_nu_sqr = 0.0;
    for (int i = 0; i < length_max; ++i) {
        max_nu_sqr = fmax(max_nu_sqr, max_partial[i]);
    }

    // Compute the time step based on the maximum value
    const double dx = data->size_x / data->nx;
    const double dy = data->size_y / data->ny;
    double dt = fmin(dx, dy) / sqrt(2.0 * max_nu_sqr);

    // Return the time step, ensuring it does not exceed the remaining simulation time
    *dt_out = fmin(dt, Tend - T);
}

// Kernel that computes the kernel over the full grid
__global__
void solve_step_kernel(SWEData* data, double* dt)
{
    double dt_val = *dt; 

    // Get row and column indices
    size_t j = get_col_id();
    size_t i = get_row_id();

    // Check if the thread is within the valid range of the grid
    if (i > 0 && i < data->nx - 1 && j > 0 && j < data->ny - 1)
    {
        // Compute the kernel for the current cell
        compute_kernel_device(data, i, j, dt_val);
    }
}

// Kernel to update the boundary conditions
__global__
void update_bc_kernel(SWEData* data, bool* reflective){
    // Set the coefficient based on whether the boundary is reflective or not
    const double coef = *reflective ? -1.0 : 1.0;

    // Single thread computes the boundary conditions for all the sides
    size_t i = get_row_id();

    // Top and bottom boundaries.
    if (i < data->nx){
      at_device(data, i, 0, data->h1) = at_device(data, i, 1, data->h0);
      at_device(data, i, data->ny - 1, data->h1) = at_device(data, i, data->ny - 2, data->h0);

      at_device(data, i, 0, data->hu1) = at_device(data, i, 1, data->hu0);
      at_device(data, i, data->ny - 1, data->hu1) = at_device(data, i, data->ny - 2, data->hu0);

      at_device(data, i, 0, data->hv1) = coef * at_device(data, i, 1, data->hv0);
      at_device(data, i, data->ny - 1, data->hv1) = coef * at_device(data, i, data->ny - 2, data->hv0);
    }

    // Left and right boundaries.
    if (i < data->ny) {
      at_device(data, 0, i, data->h1) = at_device(data, 1, i, data->h0);
      at_device(data, data->nx - 1, i, data->h1) = at_device(data, data->nx - 2, i, data->h0);

      at_device(data, 0, i, data->hu1) = coef * at_device(data, 1, i, data->hu0);
      at_device(data, data->nx - 1, i, data->hu1) = coef * at_device(data, data->nx - 2, i, data->hu0);

      at_device(data, 0, i, data->hv1) = at_device(data, 1, i, data->hv0);
      at_device(data, data->nx - 1, i, data->hv1) = at_device(data, data->nx - 2, i, data->hv0);
    }
}

// Kernel to update the new values with the old ones
__global__
void swap_data_kernel(SWEData* data)
{
    // Swap h0 and h1
    double* temp_h = data->h0;
    data->h0 = data->h1;
    data->h1 = temp_h;
    
    // Swap hu0 and hu1
    double* temp_hu = data->hu0;
    data->hu0 = data->hu1;
    data->hu1 = temp_hu;
    
    // Swap hv0 and hv1
    double* temp_hv = data->hv0;
    data->hv0 = data->hv1;
    data->hv1 = temp_hv;
}


// %%%%%%%%%%%%%%%%%
// %%% Host code %%%
// %%%%%%%%%%%%%%%%%


// Function to split the threads per block into a 2D grid
dim3 divide_threads_2D(int threadsPerBlock) {
    int dimX = (int)sqrtf((float)threadsPerBlock);
    while (threadsPerBlock % dimX != 0) {
        dimX--;
    }
    int dimY = threadsPerBlock / dimX;
    return dim3(dimX, dimY);
}


namespace
{
// Function to read a 2D array from an HDF5 file
void
read_2d_array_from_DF5(const std::string &filename,
                       const std::string &dataset_name,
                       std::vector<double> &data,
                       std::size_t &nx,
                       std::size_t &ny)
{
    hid_t file_id, dataset_id, dataspace_id;
    hsize_t dims[2];
    herr_t status;

    // Open the HDF5 file
    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0)
    {
        std::cerr << "Error opening HDF5 file: " << filename << std::endl;
        return;
    }

    // Open the dataset
    dataset_id = H5Dopen2(file_id, dataset_name.c_str(), H5P_DEFAULT);
    if (dataset_id < 0)
    {
        std::cerr << "Error opening dataset: " << dataset_name << std::endl;
        H5Fclose(file_id);
        return;
    }

    // Get the dataspace
    dataspace_id = H5Dget_space(dataset_id);
    if (dataspace_id < 0)
    {
        std::cerr << "Error getting dataspace" << std::endl;
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }

    // Get the dimensions of the dataset
    status = H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    if (status < 0)
    {
        std::cerr << "Error getting dimensions" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        return;
    }
    nx = dims[0];
    ny = dims[1];

    // Resize the data vector
    data.resize(nx * ny);

    // Read the data
    status = H5Dread(dataset_id, H5T_IEEE_F64LE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    if (status < 0)
    {
        std::cerr << "Error reading data" << std::endl;
        H5Sclose(dataspace_id);
        H5Dclose(dataset_id);
        H5Fclose(file_id);
        data.clear();
        return;
    }

    // Close resources
    H5Sclose(dataspace_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
}

} // namespace


// Constructor for the SWESolver class for the analytical tsunami and water drop test cases
SWESolver::SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny, const int threads_per_block) :
    nx_(nx), ny_(ny), size_x_(500.0), size_y_(500.0), threads_per_block_(threads_per_block) {

    // Compute the block and grid dims
    block_dims_ = divide_threads_2D(threads_per_block);
    grid_dims_ = dim3((nx + block_dims_.x - 1) / block_dims_.x, (ny + block_dims_.y - 1) / block_dims_.y);

    // Initialize variables to store pointers to device memory
    double *h0_device, *hu0_device, *hv0_device, *h1_device, *hu1_device, *hv1_device, *z_device, *zdx_device, *zdy_device;
    size_t numb_entries = nx * ny;

    // Allocate device memory for arrays
    cudaMalloc(&h0_device, numb_entries * sizeof(double));
    cudaMalloc(&hu0_device, numb_entries * sizeof(double));
    cudaMalloc(&hv0_device, numb_entries * sizeof(double));
    cudaMalloc(&h1_device, numb_entries * sizeof(double));
    cudaMalloc(&hu1_device, numb_entries * sizeof(double));
    cudaMalloc(&hv1_device, numb_entries * sizeof(double));
    cudaMalloc(&z_device, numb_entries * sizeof(double));
    cudaMalloc(&zdx_device, numb_entries * sizeof(double));
    cudaMalloc(&zdy_device, numb_entries * sizeof(double));
    cudaMallocManaged(&dt_, sizeof(double));
    cudaMallocManaged(&max_partial_, grid_dims_.x * grid_dims_.y * sizeof(double));

    // Set the values as part of the SWEData struct
    SWEData h_data;
    h_data.h0 = h0_device;
    h_data.hu0 = hu0_device;
    h_data.hv0 = hv0_device;
    h_data.h1 = h1_device;
    h_data.hu1 = hu1_device;
    h_data.hv1 = hv1_device;
    h_data.z = z_device;
    h_data.zdx = zdx_device;
    h_data.zdy = zdy_device;
    h_data.nx = nx;
    h_data.ny = ny;
    h_data.size_x = size_x_;
    h_data.size_y = size_y_;
    
    // Allocate memory for SWEData struct on device
    cudaMalloc(&data_device_, sizeof(SWEData));

    // Copy struct to device
    cudaMemcpy(data_device_, &h_data, sizeof(SWEData), cudaMemcpyHostToDevice);

    // Initialize the vectors depending on the test case
    assert(test_case_id == 1 || test_case_id == 2);
    if (test_case_id == 1)
    {
      this->reflective_ = true;
      this->init_gaussian();
    }
    else if (test_case_id == 2)
    {
      this->reflective_ = false;
      this->init_dummy_tsunami();
    }
    else
    {
      assert(false);
    }

    // Send the reflective_ variable to the device
    cudaMalloc(&reflective_device_, sizeof(bool));
    cudaMemcpy(reflective_device_, &this->reflective_, sizeof(bool), cudaMemcpyHostToDevice);
}


// Constructor for the SWESolver class that reads from an HDF5 file
SWESolver::SWESolver(const std::string &h5_file, const double size_x, const double size_y, const int threads_per_block) :
    size_x_(size_x), size_y_(size_y), reflective_(false), threads_per_block_(threads_per_block) {
    this->init_from_HDF5_file(h5_file);
}

void SWESolver::init_from_HDF5_file(const std::string &h5_file)
{
    // Print when started initializing
    std::cout << "Starting reading from HDF5 file" << std::endl;

    // Read all the data
    read_2d_array_from_DF5(h5_file, "h0", this->h0_, this->nx_, this->ny_);
    read_2d_array_from_DF5(h5_file, "hu0", this->hu0_, this->nx_, this->ny_);
    read_2d_array_from_DF5(h5_file, "hv0", this->hv0_, this->nx_, this->ny_);
    read_2d_array_from_DF5(h5_file, "topography", this->z_, this->nx_, this->ny_);

    // Compute the block and grid dims
    block_dims_ = divide_threads_2D(threads_per_block_);
    grid_dims_ = dim3((this->nx_ + block_dims_.x - 1) / block_dims_.x, (this->ny_ + block_dims_.y - 1) / block_dims_.y);

    // Resize the vectors for the next time step
    this->h1_.resize(this->h0_.size(), 0.0);
    this->hu1_.resize(this->hu0_.size(), 0.0);
    this->hv1_.resize(this->hv0_.size(), 0.0);

    // Initialize device pointers
    double *h0_device, *hu0_device, *hv0_device, *h1_device, *hu1_device, *hv1_device, *z_device, *zdx_device, *zdy_device;
    size_t numb_entries = this->nx_ * this->ny_;

    // Allocate device memory for arrays
    cudaMalloc(&h0_device, numb_entries * sizeof(double));
    cudaMalloc(&hu0_device, numb_entries * sizeof(double));
    cudaMalloc(&hv0_device, numb_entries * sizeof(double));
    cudaMalloc(&h1_device, numb_entries * sizeof(double));
    cudaMalloc(&hu1_device, numb_entries * sizeof(double));
    cudaMalloc(&hv1_device, numb_entries * sizeof(double));
    cudaMalloc(&z_device, numb_entries * sizeof(double));
    cudaMalloc(&zdx_device, numb_entries * sizeof(double));
    cudaMalloc(&zdy_device, numb_entries * sizeof(double));
    cudaMallocManaged(&dt_, sizeof(double));
    cudaMallocManaged(&max_partial_, grid_dims_.x * grid_dims_.y * sizeof(double));

    // Set the values as part of the SWEData struct
    SWEData h_data;
    h_data.h0 = h0_device;
    h_data.hu0 = hu0_device;
    h_data.hv0 = hv0_device;
    h_data.h1 = h1_device;
    h_data.hu1 = hu1_device;
    h_data.hv1 = hv1_device;
    h_data.z = z_device;
    h_data.zdx = zdx_device;
    h_data.zdy = zdy_device;
    h_data.nx = this->nx_;
    h_data.ny = this->ny_;
    h_data.size_x = this->size_x_;
    h_data.size_y = this->size_y_;

    // Allocate memory for SWEData struct on device
    cudaMalloc(&data_device_, sizeof(SWEData));

    // Copy struct to device
    cudaMemcpy(data_device_, &h_data, sizeof(SWEData), cudaMemcpyHostToDevice);

    // Move the data given from the file from host to device
    cudaMemcpy(h0_device, this->h0_.data(), numb_entries * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(hu0_device, this->hu0_.data(), numb_entries * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(hv0_device, this->hv0_.data(), numb_entries * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(z_device, this->z_.data(), numb_entries * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(h1_device, this->h1_.data(), numb_entries * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(hu1_device, this->hu1_.data(), numb_entries * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(hv1_device, this->hv1_.data(), numb_entries * sizeof(double), cudaMemcpyHostToDevice);

    // Send the reflective_ variable to the device
    cudaMalloc(&reflective_device_, sizeof(bool));
    cudaMemcpy(reflective_device_, &this->reflective_, sizeof(bool), cudaMemcpyHostToDevice);


    // Call the init_dx_dy_kernel to initialize dx and dy with only one thread
    init_dx_dy_kernel<<<1, 1>>>(data_device_);
    cudaDeviceSynchronize();

    // Print finished initializing
    std::cout << "Finished reading from HDF5 file." << std::endl;
}

// Host function for the water drop test case
void SWESolver::init_gaussian()
{
    // Print when started initializing
    std::cout << "Starting Gaussian test case." << std::endl;

    // Call the init gaussian kernel
    init_gaussian_kernel<<<grid_dims_, block_dims_>>>(data_device_);
    cudaDeviceSynchronize();

    // Print when finished initializing
    std::cout << "Initialized Gaussian test case." << std::endl;
}

// Host function for the dummy tsunami test case
void SWESolver::init_dummy_tsunami()
{
    // Print when started initializing
    std::cout << "Starting Dummy Tsunami test case." << std::endl;

    // Call the init dummy tsunami kernel
    init_dummy_tsunami_kernel<<<grid_dims_, block_dims_>>>(data_device_);
    cudaDeviceSynchronize();

    // Print when finished initializing
    std::cout << "Initialized Dummy Tsunami test case." << std::endl;
}


// Function to solve the SWE equations
void SWESolver::solve(const double Tend, const bool full_log, const std::size_t output_n, const std::string &fname_prefix)
{
    // Initialize the output writer
    std::shared_ptr<XDMFWriter> writer;
    
    // If needed write the initial data to a file for visualization
    if (output_n > 0)
    {
        // First, copy the SWEData struct from device to host to get the device pointers
        SWEData data_host;
        cudaMemcpy(&data_host, data_device_, sizeof(SWEData), cudaMemcpyDeviceToHost);

        // Initialize host vectors
        this->z_.resize(nx_ * ny_);
        this->h0_.resize(nx_ * ny_);
        
        // Copy the z array data from device to host
        cudaMemcpy(this->z_.data(), data_host.z, this->z_.size() * sizeof(double), cudaMemcpyDeviceToHost);
        
        // Copy initial h0 data for the writer
        cudaMemcpy(this->h0_.data(), data_host.h0, this->h0_.size() * sizeof(double), cudaMemcpyDeviceToHost);

        // Write to file
        writer = std::make_shared<XDMFWriter>(fname_prefix, this->nx_, this->ny_, this->size_x_, this->size_y_, this->z_);
        writer->add_h(h0_, 0.0);

    }

    // Variable to keep track of time
    double T = 0.0;

    // Print initial message
    std::cout << "Solving SWE..." <<  std::endl;
    
    // Start a counter for the number of time steps
    std::size_t nt = 1;

    // Iterate until the end time is reached
    while (T < Tend)
    {
        // Compute the time step size
        *dt_ = 0.0;
        this->compute_time_step(T, Tend);

        // Compute the next time step
        const double T1 = T + *dt_;

        // Print some information about the current step
        printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, *dt_ * 3600, 100 * T1 / Tend);
        std::cout << (full_log ? "\n" : "\r") << std::flush;

        // Update the boundary conditions
        this->update_bcs();

        // Solve the SWE equations for the current time step
        this->solve_step();

        // If needed write the output to a file for visualization
        if (output_n > 0 && nt % output_n == 0)
        {
            // Copy current h data from GPU for output
            SWEData data_host;
            cudaMemcpy(&data_host, data_device_, sizeof(SWEData), cudaMemcpyDeviceToHost);

            // Move the h1 data from device to host
            std::vector<double> h1(nx_ * ny_);
            cudaMemcpy(h1.data(), data_host.h1, h1.size() * sizeof(double), cudaMemcpyDeviceToHost);
            
            // Write to file
            writer->add_h(h1, T1);
        }
        // Update counter
        ++nt;

        // Swap the old and new solutions
        this->swap_data();

        // Update the time
        T = T1;
    }

    // Swap the final data to make sure the last step is ready for output
    this->swap_data();
    
    // If needed write the final output to a file for visualization
    if (output_n > 0)
    {
      // Copy final h data for output
      SWEData data_host;
      cudaMemcpy(&data_host, data_device_, sizeof(SWEData), cudaMemcpyDeviceToHost);

      // Move the h1 data from device to host
      this->h1_.resize(nx_ * ny_);
      cudaMemcpy(this->h1_.data(), data_host.h1, this->h1_.size() * sizeof(double), cudaMemcpyDeviceToHost);

      // Write the final step to file
      writer->add_h(h1_, T);
    }

    // Finish with printing message
    std::cout << "Finished solving SWE." << std::endl;
}


// Host function to compute the time step
void SWESolver::compute_time_step(const double T, const double Tend)
{
    // Allocate memory for max_partial_ on the device's share memory
    size_t shared_memory_size = block_dims_.x * block_dims_.y * sizeof(double);

    // Launch the kernel to compute the time step for each block
    compute_time_step_block_kernel<<<grid_dims_, block_dims_, shared_memory_size>>>(data_device_, max_partial_);
    cudaDeviceSynchronize();

    // Perform reduction to find the maximum value across all blocks
    int max_partial_length = grid_dims_.x * grid_dims_.y;
    compute_time_step_kernel<<<1, 1>>>(data_device_, max_partial_, max_partial_length, T, Tend, dt_);
    cudaDeviceSynchronize();

}

// Host function to solve a single step of the SWE equations
void SWESolver::solve_step() const
{
    // Launch the kernel to compute the next step
    solve_step_kernel<<<grid_dims_, block_dims_>>>(data_device_, dt_);
    cudaDeviceSynchronize();
}

// Host function to update boundary conditions
void SWESolver::update_bcs() const
{
  // Call a single block with enough threads
  int threads_for_bc = std::max(nx_, ny_);
  update_bc_kernel<<<1, threads_for_bc>>>(data_device_, reflective_device_);
  cudaDeviceSynchronize();
}

// Host function to swap the old and new data
void SWESolver::swap_data() const
{
    // Launch the swap kernel on the device to swap pointers in the device struct
    swap_data_kernel<<<1, 1>>>(data_device_);
    cudaDeviceSynchronize();
}

// Destructor for the SWESolver class
SWESolver::~SWESolver()
{
    // Get device pointers first
    SWEData data_host;
    cudaMemcpy(&data_host, data_device_, sizeof(SWEData), cudaMemcpyDeviceToHost);
    
    // Free the device memory for the SWEData struct
    cudaFree(data_host.h0);
    cudaFree(data_host.hu0);
    cudaFree(data_host.hv0);
    cudaFree(data_host.h1);
    cudaFree(data_host.hu1);
    cudaFree(data_host.hv1);
    cudaFree(data_host.z);
    cudaFree(data_host.zdx);
    cudaFree(data_host.zdy);
    cudaFree(data_device_);
    cudaFree(dt_);
    cudaFree(reflective_device_);
}

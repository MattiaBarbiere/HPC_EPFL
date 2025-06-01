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


__device__ 
double& at_device(SWEData* data, std::size_t i, std::size_t j, double* arr)
{
    return arr[j * data->nx + i];
}

__device__
void init_dx_dy_device(SWEData* data)
{
    const double dx = data->size_x / data->nx;
    const double dy = data->size_y / data->ny;

    for (std::size_t j = 1; j < data->ny - 1; ++j)
    {
        for (std::size_t i = 1; i < data->nx - 1; ++i)
        {
            at_device(data, i, j, data->zdx) = 0.5 * (at_device(data, i + 1, j, data->z) - at_device(data, i - 1, j, data->z)) / dx;
            at_device(data, i, j, data->zdy) = 0.5 * (at_device(data, i, j + 1, data->z) - at_device(data, i, j - 1, data->z)) / dy;
        }
    }
}

__device__
void compute_kernel_device(SWEData* data, const std::size_t i, const std::size_t j, const double dt){
  
  static constexpr double g = 127267.20000000;
  const double dx = data->size_x / data->nx;
  const double dy = data->size_y / data->ny;
  const double C1x = 0.5 * dt / dx;
  const double C1y = 0.5 * dt / dy;
  const double C2 = dt * g;
  constexpr double C3 = 0.5 * g;

  double hij = 0.25 * (at_device(data, i, j - 1, data->h0) + at_device(data, i, j + 1, data->h0) + at_device(data, i - 1, j, data->h0) + at_device(data, i + 1, j, data->h0))
               + C1x * (at_device(data, i - 1, j, data->hu0) - at_device(data, i + 1, j, data->hu0)) + C1y * (at_device(data, i, j - 1, data->hv0) - at_device(data, i, j + 1, data->hv0));
  if (hij < 0.0)
  {
    hij = 1.0e-5;
  }

  at_device(data, i, j, data->h1) = hij;

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

__global__
void init_gaussian_kernel(SWEData* data)
{
    // Initialize hu0, hv0 to 0.0
    for (size_t idx = 0; idx < data->nx * data->ny; ++idx) {
        data->hu0[idx] = 0.0;
        data->hv0[idx] = 0.0;
    }

    const double x0_0 = data->size_x / 4.0;
    const double y0_0 = data->size_y / 3.0;
    const double x0_1 = data->size_x / 2.0;
    const double y0_1 = 0.75 * data->size_y;

    const double dx = data->size_x / data->nx;
    const double dy = data->size_y / data->ny;

    for (size_t j = 0; j < data->ny; ++j)
    {
        for (size_t i = 0; i < data->nx; ++i)
        {
            double x = dx * (static_cast<double>(i) + 0.5);
            double y = dy * (static_cast<double>(j) + 0.5);
            double gauss_0 = 10.0 * exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 1000.0);
            double gauss_1 = 10.0 * exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 1000.0);

            data->h0[j * data->nx + i] = 10.0 + gauss_0 + gauss_1;
        }
    }

    for (size_t idx = 0; idx < data->nx * data->ny; ++idx) {
        data->h1[idx] = 0.0;
        data->hu1[idx] = 0.0;
        data->hv1[idx] = 0.0;
    }

    for (size_t idx = 0; idx < data->nx * data->ny; ++idx) {
        data->z[idx] = 0.0;
    }

    init_dx_dy_device(data);
}

__global__
void init_dummy_tsunami_kernel(SWEData* data)
{
    for (size_t idx = 0; idx < data->nx * data->ny; ++idx) {
        data->hu0[idx] = 0.0;
        data->hv0[idx] = 0.0;
        data->h1[idx] = 0.0;
        data->hu1[idx] = 0.0;
        data->hv1[idx] = 0.0;
    }

    const double x0_0 = 0.6 * data->size_x;
    const double y0_0 = 0.6 * data->size_y;
    const double x0_1 = 0.4 * data->size_x;
    const double y0_1 = 0.4 * data->size_y;
    const double x0_2 = 0.7 * data->size_x;
    const double y0_2 = 0.3 * data->size_y;

    const double dx = data->size_x / data->nx;
    const double dy = data->size_y / data->ny;

    for (size_t j = 0; j < data->ny; ++j) {
        for (size_t i = 0; i < data->nx; ++i)
        {
            const double x = dx * (static_cast<double>(i) + 0.5);
            const double y = dy * (static_cast<double>(j) + 0.5);

            const double gauss_0 = 2.0 * exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 3000.0);
            const double gauss_1 = 3.0 * exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 10000.0);
            const double gauss_2 = 5.0 * exp(-((x - x0_2) * (x - x0_2) + (y - y0_2) * (y - y0_2)) / 100.0);

            double z_val = -1.0 + gauss_0 + gauss_1;
            at_device(data, i, j, data->z) = z_val;

            double h0_val = (z_val < 0.0) ? (-z_val + gauss_2) : 0.00001;
            at_device(data, i, j, data->h0) = h0_val;
        }
    }

    init_dx_dy_device(data);
}

__global__
void compute_time_step_kernel(const SWEData* data,  const double T,
                             const double Tend, double* dt_out) {
    double max_nu_sqr = 0.0;
    static constexpr double g = 127267.20000000;

    for (std::size_t j = 1; j < data->ny - 1; ++j) {
        for (std::size_t i = 1; i < data->nx - 1; ++i) {
            double h_val = data->h[j * data->nx + i];
            double hu_val = data->hu[j * data->nx + i];
            double hv_val = data->hv[j * data->nx + i];

            au = fmax(au, fabs(hu_val));
            av = fmax(av, fabs(hv_val));

            const double nu_u = fabs(hu_val) / h_val + sqrt(g * h_val);
            const double nu_v = fabs(hv_val) / h_val + sqrt(g * h_val);

            max_nu_sqr = fmax(max_nu_sqr, nu_u * nu_u + nu_v * nu_v);
        }
    }

    const double dx = data->size_x / data->nx;
    const double dy = data->size_y / data->ny;
    double dt = fmin(dx, dy) / sqrt(2.0 * max_nu_sqr);

    *dt_out = fmin(dt, data->Tend - data->T);
}


__global__
void solve_step_kernel(SWEData* data, const double dt)
{
  for (std::size_t j = 1; j < data->ny - 1; ++j)
  {
    for (std::size_t i = 1; i < data->nx - 1; ++i)
    {
      compute_kernel_device(data, i, j, dt);
    }
  }
}


__global__
void update_bc_kernel(SWEData* data, bool reflective){
  const double coef = reflective ? -1.0 : 1.0;

  // Top and bottom boundaries.
  for (std::size_t i = 0; i < data->nx; ++i)
  {
    at_device(data, i, 0, data->h1) = at_device(data, i, 1, data->h0);
    at_device(data, i, data->ny - 1, data->h1) = at_device(data, i, data->ny - 2, data->h0);

    at_device(data, i, 0, data->hu1) = at_device(data, i, 1, data->hu0);
    at_device(data, i, data->ny - 1, data->hu1) = at_device(data, i, data->ny - 2, data->hu0);

    at_device(data, i, 0, data->hv1) = coef * at_device(data, i, 1, data->hv0);
    at_device(data, i, data->ny - 1, data->hv1) = coef * at_device(data, i, data->ny - 2, data->hv0);
  }

  // Left and right boundaries.
  for (std::size_t j = 0; j < data->ny; ++j)
  {
    at_device(data, 0, j, data->h1) = at_device(data, 1, j, data->h0);
    at_device(data, data->nx - 1, j, data->h1) = at_device(data, data->nx - 2, j, data->h0);

    at_device(data, 0, j, data->hu1) = coef * at_device(data, 1, j, data->hu0);
    at_device(data, data->nx - 1, j, data->hu1) = coef * at_device(data, data->nx - 2, j, data->hu0);

    at_device(data, 0, j, data->hv1) = at_device(data, 1, j, data->hv0);
    at_device(data, data->nx - 1, j, data->hv1) = at_device(data, data->nx - 2, j, data->hv0);
  }
}


// %%%%%%%%%%%%%%%%%
// %%% Host code %%%
// %%%%%%%%%%%%%%%%%


namespace
{

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

  // std::cout << "Successfully read 2D array from HDF5 file: " << filename << ", dataset: " << dataset_name <<
  // std::endl;
}

} // namespace

SWESolver::SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny) :
  nx_(nx), ny_(ny), size_x_(500.0), size_y_(500.0) {

  // Send variables to the device
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
  cudaMalloc(&dt_device_, sizeof(double));

  // Allocate memory for SWEData struct on host and device
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

  cudaMalloc(&data_device_, sizeof(SWEData));

  // Copy struct to device
  cudaMemcpy(data_device_, &h_data, sizeof(SWEData), cudaMemcpyHostToDevice);

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
  bool* reflective_device;
  cudaMalloc(&reflective_device, sizeof(bool));
  cudaMemcpy(reflective_device, &this->reflective_, sizeof(bool), cudaMemcpyHostToDevice);
}

SWESolver::SWESolver(const std::string &h5_file, const double size_x, const double size_y) :
  size_x_(size_x), size_y_(size_y), reflective_(false)
{
  this->init_from_HDF5_file(h5_file);
}

void SWESolver::init_from_HDF5_file(const std::string &h5_file)
{
  read_2d_array_from_DF5(h5_file, "h0", this->h0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hu0", this->hu0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hv0", this->hv0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "topography", this->z_, this->nx_, this->ny_);

  this->h1_.resize(this->h0_.size(), 0.0);
  this->hu1_.resize(this->hu0_.size(), 0.0);
  this->hv1_.resize(this->hv0_.size(), 0.0);

  this->init_dx_dy();
}


void SWESolver::init_gaussian()
{
  // Call the init gaussian kernel
  init_gaussian_kernel<<<1, 1>>>(data_device_);
  cudaDeviceSynchronize();
}


void SWESolver::init_dummy_tsunami()
{
  // Call the init dummy tsunami kernel
  init_dummy_tsunami_kernel<<<1, 1>>>(data_device_);
  cudaDeviceSynchronize();
}

void SWESolver::solve(const double Tend, const bool full_log, const std::size_t output_n, const std::string &fname_prefix)
{
  std::shared_ptr<XDMFWriter> writer;
  if (output_n > 0)
  {
    writer = std::make_shared<XDMFWriter>(fname_prefix, this->nx_, this->ny_, this->size_x_, this->size_y_, this->z_);
    writer->add_h(h0_, 0.0);
  }

  double T = 0.0;

  std::vector<double> &h = h1_;
  std::vector<double> &hu = hu1_;
  std::vector<double> &hv = hv1_;

  std::vector<double> &h0 = h0_;
  std::vector<double> &hu0 = hu0_;
  std::vector<double> &hv0 = hv0_;

  std::cout << "Solving SWE..." << std::endl;

  std::size_t nt = 1;
  while (T < Tend)
  {
    this->compute_time_step(T, Tend);
    
    // Copy dt back from device
    double dt;
    cudaMemcpy(&dt, dt_device_, sizeof(double), cudaMemcpyDeviceToHost);

    const double T1 = T + dt;

    printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
    std::cout << (full_log ? "\n" : "\r") << std::flush;

    this->update_bcs();

    this->solve_step();

    if (output_n > 0 && nt % output_n == 0)
    {
      writer->add_h(h, T1);
    }
    ++nt;

    // Swap the old and new solutions
    std::swap(h, h0);
    std::swap(hu, hu0);
    std::swap(hv, hv0);

    T = T1;
  }

  // Copying last computed values to h1_, hu1_, hv1_ (if needed)
  if (&h0 != &h1_)
  {
    h1_ = h0;
    hu1_ = hu0;
    hv1_ = hv0;
  }

  if (output_n > 0)
  {
    writer->add_h(h1_, T);
  }

  std::cout << "Finished solving SWE." << std::endl;
}

void SWESolver::compute_time_step(const double T, const double Tend)
{
  compute_time_step_kernel<<<1, 1>>>(data_device_, T, Tend, dt_device_);
  cudaDeviceSynchronize();
}

void SWESolver::solve_step() const
{
  // Launch the kernel to compute the next step
  solve_step_kernel<<<1, 1>>>(data_device_, dt_device_);
  cudaDeviceSynchronize();
}

void SWESolver::update_bcs() const
{
  // Call the kernel
  update_bc_kernel<<<1, 1>>>(data_device_, reflective_device_);
};

SWESolver::~SWESolver()
{
  // Free device memory
  cudaFree(h0_.data());
  cudaFree(hu0_.data());
  cudaFree(hv0_.data());
  cudaFree(h1_.data());
  cudaFree(hu1_.data());
  cudaFree(hv1_.data());
  cudaFree(z_.data());
  cudaFree(zdx_.data());
  cudaFree(zdy_.data());
  cudaFree(data_device_);
  cudaFree(dt_device_);
  cudaFree(reflective_device_);
}

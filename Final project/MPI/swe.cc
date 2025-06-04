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

bool DEBUG = false;

namespace
{

// Function to read a 2D array from an HDF5 file into a flat vector, and get its dimensions
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

// Constructor for initializing the SWESolver for the water drop and analytical tsunami test cases
SWESolver::SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny, MPI_Comm comm) :
  nx_(nx), ny_(ny), size_x_(500.0), size_y_(500.0), cart_comm_(MPI_COMM_NULL) {
    
    // Set up MPI Cartesian communicator and domain decomposition
    MPI_Comm_size(comm, &size_);
    MPI_Comm_rank(comm, &rank_);
    dims_[0] = dims_[1] = 0;

    // Make the sure dims_[0]*dims_[1] = size_
    MPI_Dims_create(size_, 2, dims_);            

    // Set the periodicity to false
    int periods[2] = {0, 0};

    // Create the Cartesian communicator
    MPI_Cart_create(comm, 2, dims_, periods, 1, &cart_comm_);
    MPI_Cart_coords(cart_comm_, rank_, 2, coords_);
    MPI_Cart_shift(cart_comm_, 0, 1, &nbr_left_, &nbr_right_);
    MPI_Cart_shift(cart_comm_, 1, 1, &nbr_down_, &nbr_up_);

    // Ensure that the number of points in each direction is divisible by dims_[0] and dims_[1]
    assert(nx_ % dims_[0] == 0 && ny_ % dims_[1] == 0);
    
    // Set the size of the local grid
    nx_local_ = nx_ / dims_[0];
    ny_local_ = ny_ / dims_[1];

    // Set the offsets
    offset_x_ = coords_[0] * nx_local_;
    offset_y_ = coords_[1] * ny_local_;

    // Function that adds the halo cells to the edges of the local grid
    auto alloc = [&](std::vector<double>& v){
      v.resize((nx_local_ + 2) * (ny_local_ + 2), 0.0);
    };

    // Add the halo cells to each variable
    alloc(h0_);
    alloc(h1_);
    alloc(hu0_);
    alloc(hu1_);
    alloc(hv0_);
    alloc(hv1_);
    alloc(z_);
    alloc(zdx_);
    alloc(zdy_);

    if (DEBUG){
        std::cout << "Rank: " << rank_ << std::endl;
        std::cout << "SWESolver initialized with " << size_ << " processes." << std::endl;
        std::cout << rank_ <<"Process coordinates: (" << coords_[0] << ", " << coords_[1] << ")" << std::endl;
    }
    
    // Initialize PDE based on test case
    assert(test_case_id == 1 || test_case_id == 2);
    if (test_case_id == 1)
    {
      this->reflective_ = true;
      this->init_gaussian_local();
    }
    else if (test_case_id == 2)
    {
      this->reflective_ = false;
      this->init_dummy_tsunami_local();
    }
    else
    {
      assert(false);
    }

}

// Constructor for initializing from HDF5 file
SWESolver::SWESolver(const std::string &h5_file, const double size_x, const double size_y, MPI_Comm comm):
  size_x_(size_x), size_y_(size_y), reflective_(false), cart_comm_(MPI_COMM_NULL){
    
    this->init_from_HDF5_file(h5_file, comm);
}

// Initialize solver state from HDF5 file
void SWESolver::init_from_HDF5_file(const std::string &h5_file, MPI_Comm comm)
{
  // Initialize the global variables to read from file
  std::vector<double> h0_global, hu0_global, hv0_global, z_global;

  // Read from the HDF5 file
  read_2d_array_from_DF5(h5_file, "h0", h0_global, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hu0", hu0_global, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hv0", hv0_global, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "topography", z_global, this->nx_, this->ny_);

  // The communicator 
  MPI_Comm_size(comm, &size_);
  MPI_Comm_rank(comm, &rank_);
  dims_[0] = dims_[1] = 0;

  // Make the sure dims_[0]*dims_[1] = size_
  MPI_Dims_create(size_, 2, dims_);            

  // Set the periodicity to false
  int periods[2] = {0, 0};

  // Create the Cartesian communicator
  MPI_Cart_create(comm, 2, dims_, periods, 1, &cart_comm_);
  MPI_Cart_coords(cart_comm_, rank_, 2, coords_);
  MPI_Cart_shift(cart_comm_, 0, 1, &nbr_left_, &nbr_right_);
  MPI_Cart_shift(cart_comm_, 1, 1, &nbr_down_, &nbr_up_);

  // Ensure that the number of points in each direction is divisible by dims_[0] and dims_[1]
  assert(nx_ % dims_[0] == 0 && ny_ % dims_[1] == 0);
  
  // Set the size of the local grid
  nx_local_ = nx_ / dims_[0];
  ny_local_ = ny_ / dims_[1];

  // Set the offsets
  offset_x_ = coords_[0] * nx_local_;
  offset_y_ = coords_[1] * ny_local_;

  // Function that adds the halo cells to the edges of the local grid
  auto alloc = [&](std::vector<double>& v){
    v.resize((nx_local_ + 2) * (ny_local_ + 2), 0.0);
  };

  // Add the halo cells to each variable
  alloc(h0_);
  alloc(h1_);
  alloc(hu0_);
  alloc(hu1_);
  alloc(hv0_);
  alloc(hv1_);
  alloc(z_);
  alloc(zdx_);
  alloc(zdy_);

  // Fill the local grids with only the data they need
  for (std::size_t j = 0; j < ny_local_ + 2; ++j)
  {
      for (std::size_t i = 0; i < nx_local_ + 2; ++i)
      {
          int i_global = offset_x_ + i - 1;
          int j_global = offset_y_ + j - 1;

          // Make sure the global indices are within the correct bounds
          i_global = std::min(std::max(i_global, 0), (int)nx_ - 1);
          j_global = std::min(std::max(j_global, 0), (int)ny_ - 1);
          at(h0_, i, j) = h0_global[j_global * nx_ + i_global];
          at(hu0_, i, j) = hu0_global[j_global * nx_ + i_global];
          at(hv0_, i, j) = hv0_global[j_global * nx_ + i_global];
          at(z_, i, j) = z_global[j_global * nx_ + i_global];
      }
  }

  // Communicate to all neighbors the initial conditions
  this->exchange_data();

  // Initialize dx and dy
  this->init_dx_dy_local();
}

// Initialize Gaussian water drop test case over each processor's local domain
void SWESolver::init_gaussian_local(){
    const double x0_0 = size_x_ / 4.0;
    const double y0_0 = size_y_ / 3.0;
    const double x0_1 = size_x_ / 2.0;
    const double y0_1 = 0.75 * size_y_;

    const double dx = size_x_ / nx_;
    const double dy = size_y_ / ny_;

    // Initialize only on the local grid
    for (std::size_t j = 0; j < ny_local_ + 2; ++j){
        for (std::size_t i = 0; i < nx_local_ + 2; ++i){
            int gi = offset_x_ + i - 1;
            int gj = offset_y_ + j - 1;
            const double x = dx * (static_cast<double>(gi) + 0.5);
            const double y = dy * (static_cast<double>(gj) + 0.5);

            const double gauss_0 = 10.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 1000.0);
            const double gauss_1 = 10.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 1000.0);
            at(z_, i, j) = 0.0;
            at(h0_, i, j) = 10.0 + gauss_0 + gauss_1;
            at(hu0_, i, j) = 0.0;
            at(hv0_, i, j) = 0.0;
        }
    }

    // Communicate initial conditions
    this->exchange_data();

    // Initialize dx and dy
    this->init_dx_dy_local();

}

// Initialize dummy tsunami test case over each processor's local domain
void SWESolver::init_dummy_tsunami_local(){

    const double x0_0 = 0.6 * size_x_;
    const double y0_0 = 0.6 * size_y_;
    const double x0_1 = 0.4 * size_x_;
    const double y0_1 = 0.4 * size_y_;
    const double x0_2 = 0.7 * size_x_;
    const double y0_2 = 0.3 * size_y_;

    const double dx = size_x_ / nx_;
    const double dy = size_y_ / ny_;

    // Initialize only on the local grid
    for(std::size_t j = 0; j < ny_local_ + 2; j++){
      for(std::size_t i = 0; i < nx_local_ + 2; i++){
        int gi = offset_x_ + i - 1;
        int gj = offset_y_ + j - 1;
        const double x = dx * (static_cast<double>(gi) + 0.5);
        const double y = dy * (static_cast<double>(gj) + 0.5);

        const double gauss_0 = 2.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 3000.0);
        const double gauss_1 = 3.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 10000.0);
        const double gauss_2 = 5.0 * std::exp(-((x - x0_2) * (x - x0_2) + (y - y0_2) * (y - y0_2)) / 100.0);
        const double z = -1.0 + gauss_0 + gauss_1;

        at(z_, i, j) = z;

        double h0 = z < 0.0 ? -z + gauss_2 : 0.00001;
        at(h0_, i, j) = h0;
      }
    }

    // Communicate initial conditions and compute topography derivatives
    this->exchange_data();

    // Initialize dx and dy
    this->init_dx_dy_local();
}

// Compute topography derivatives (zdx_, zdy_) using central differences
void SWESolver::init_dx_dy_local()
{
    double dx = size_x_/nx_;
    double dy = size_y_/ny_;

    // Only loop interior cells
    for(int j = 1; j <= (int)ny_local_; ++j) {
        for(int i = 1; i <= (int)nx_local_; ++i) {
          
          // Compute the derivatives
          at(zdx_, i, j) = ( at(z_, i+1, j) - at(z_, i-1, j) ) / (2.0*dx);
          at(zdy_, i, j) = ( at(z_, i, j+1) - at(z_, i, j-1) ) / (2.0*dy);
        }
    }
}

// Exchange halo data with neighboring MPI ranks for all relevant arrays
void SWESolver::exchange_data()
{
    // Create MPI datatypes for a column
    MPI_Datatype column_type;
    MPI_Type_vector(ny_local_, 1, nx_local_+2, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);

    // Create an MPI datatype for a row
    MPI_Datatype row_type;
    MPI_Type_vector(nx_local_, 1, 1, MPI_DOUBLE, &row_type);
    MPI_Type_commit(&row_type);

    // Lambda to exchange halo data for a single array
    auto exch = [&](std::vector<double>& F){
      // Send data from right and receive from left 
      MPI_Sendrecv(&at(F, 1, 1), 1, column_type, nbr_left_, 0,
                  &at(F, nx_local_ + 1, 1), 1, column_type, nbr_right_, 0,
                  cart_comm_, MPI_STATUS_IGNORE);

      // Send data from left and receive from right
      MPI_Sendrecv(&at(F, nx_local_, 1), 1, column_type, nbr_right_, 1,
                  &at(F,0,1), 1, column_type, nbr_left_, 1,
                  cart_comm_, MPI_STATUS_IGNORE);
      
      // Send data from up and receive from down
      MPI_Sendrecv(&at(F, 1, 1), 1, row_type, nbr_down_, 2,
                  &at(F, 1, ny_local_ + 1), 1, row_type, nbr_up_, 2,
                  cart_comm_, MPI_STATUS_IGNORE);

      // Send data from down and receive from up
      MPI_Sendrecv(&at(F, 1, ny_local_), 1, row_type, nbr_up_, 3,
                  &at(F, 1, 0), 1, row_type, nbr_down_, 3,
                  cart_comm_, MPI_STATUS_IGNORE);
  };

  // Exchange data for all variables
  for (auto *F : { &h0_, &hu0_, &hv0_, &zdx_, &zdy_ })
    exch(*F);

  // Free MPI datatypes
  MPI_Type_free(&column_type);
  MPI_Type_free(&row_type);
}

// Function to remove halo cells from a local array and return the interior data
std::vector<double> SWESolver::remove_halo(std::vector<double>& F)
{
    // Remove the halo cells from the vector F
    std::vector<double> F_no_halo(nx_local_ * ny_local_);
    
    // Only iterate in the interior of the local grid without the halos
    for (std::size_t j = 1; j <= ny_local_; ++j)
    {
        for (std::size_t i = 1; i <= nx_local_; ++i)
        {
            F_no_halo[(j - 1) * nx_local_ + (i - 1)] = at(F, i, j);
        }
    }
    
    // Return the vector without the halo cells
    return F_no_halo;
}

// Gather local data (without halos) from all ranks to rank 0 and organize into global array
std::vector<double> SWESolver::gather_data(const std::vector<double>& local_data)
{
    // Remove halo cells from local data
    std::vector<double> local_no_halo = this->remove_halo(const_cast<std::vector<double>&>(local_data));
    
    // Prepare global vector on rank 0
    std::vector<double> gathered_data;
    std::vector<double> organized_data;
    if (rank_ == 0) {
        gathered_data.resize(nx_ * ny_);
        organized_data.resize(nx_ * ny_, 0.0);
    }
    
    // Gather data from all ranks to rank 0
    MPI_Gather(local_no_halo.data(), nx_local_ * ny_local_, MPI_DOUBLE,
              rank_ == 0 ? gathered_data.data() : nullptr, nx_local_ * ny_local_, MPI_DOUBLE,
              0, cart_comm_);
    
    // Only rank 0 organizes the gathered data
    if (rank_ == 0)
    {
        // Organize the data for visuals
        for (int r = 0; r < size_; ++r)
        {
            // Get the coordinates of the rank
            int coords[2];
            MPI_Cart_coords(cart_comm_, r, 2, coords);

            // Compute the offsets for the local grid
            std::size_t offset_x = coords[0] * nx_local_;
            std::size_t offset_y = coords[1] * ny_local_;

            // Organize the data in the correct position for the global array
            for (std::size_t j = 0; j < ny_local_; ++j)
            {
                for (std::size_t i = 0; i < nx_local_; ++i)
                {
                    // Index for the gathered array
                    std::size_t gathered_index = r * (nx_local_ * ny_local_) + j * nx_local_ + i;
                    
                    // Index for the organized array
                    std::size_t global_index = (offset_y + j) * nx_ + (offset_x + i);

                    // Place the gathered data into the organized array
                    organized_data[global_index] = gathered_data[gathered_index];
                }
            }
        }
    }
    // Add a barrier 
    MPI_Barrier(cart_comm_);

    // Return the organized data
    return organized_data;
}

// Main time-stepping loop for the SWE solver
void SWESolver::solve(const double Tend, const bool full_log, const std::size_t output_n, const std::string &fname_prefix)
{
    // Initialize the writer
    std::shared_ptr<XDMFWriter> writer;
    
    // If needed write the initial conditions to the XDMF file
    if (output_n > 0)
    {
      // Gather the initial values and topography from all ranks
      std::vector<double> z_global = this->gather_data(z_);
      std::vector<double> h_global = this->gather_data(h0_);

      // Rank 0 writes to the file
      if (rank_ == 0) {
        // Set the writer
        writer = std::make_shared<XDMFWriter>(fname_prefix, this->nx_, this->ny_, this->size_x_, this->size_y_, z_global);
        
        // Write to the XDMF file
        writer->add_h(h_global, 0.0);
      }
    
    }

    // Variable to keep track of time
    double T = 0.0;

    // Print the start message
    if (rank_ == 0){
      std::cout << "Solving SWE..." << std::endl;
    }

    // Initialize the time step counter
    std::size_t nt = 1;

    // Loop until the end time is reached
    while (T < Tend)
    { 
        // Compute the time step
        const double dt = this->compute_time_step(T, Tend);

        // Update the next time
        const double T1 = T + dt;
        
        // Print some information of the iteration
        if (rank_ == 0){
            printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
            std::cout << (full_log ? "\n" : "\r") << std::flush;
        }
        
        // Exchange the data with neighbors
        this->exchange_data();

        // Update the boundary conditions
        this->update_bcs();

        // Solve the next time step
        this->solve_step(dt);

        // If needed write to files
        if (output_n > 0 && nt % output_n == 0)
        {
            // Perform the gather operation
            std::vector<double> h1_global = this->gather_data(h1_);

            // If rank 0, write the gathered data to the XDMF file
            if (rank_ == 0)
            {
                // Write the solution to the XDMF file
                writer->add_h(h1_global, T1);
            }
        }

        // Update the counter
        ++nt;

        // Swap the old and new solutions
        std::swap(h1_, h0_);
        std::swap(hu1_, hu0_);
        std::swap(hv1_, hv0_);

        // Update the time
        T = T1;
    }

    // Copying last computed values to h1_, hu1_, hv1_ (if needed)
    if (&h0_ != &h1_)
    {
        h1_ = h0_;
        hu1_ = hu0_;
        hv1_ = hv0_;
    }

    // If needed write the final solution to the XDMF file
    if (output_n > 0)
    {
        // Remove the halo cells from h1_ and write to the XDMF file
        std::vector<double> h1_global = this->gather_data(h1_);

        // Print size of h1_global
        if (rank_ == 0)
        {
              // Write the final solution to the XDMF file
              writer->add_h(h1_global, T);
        }
    }

    if (rank_ == 0)
    {
        std::cout << "Finished solving SWE." << std::endl;
    }

}

// Function to compute the time step
double SWESolver::compute_time_step(const double T, const double Tend)
{
    // Init the global maximum of nu_sqr
    double max_nu_sqr = 0.0;
    
    // Init the local maximum of nu_sqr
    double max_nu_sqr_local = 0.0;

    // Iterate over the interior of the local grids
    for(std::size_t j = 1; j <= ny_local_; ++j){
        for(std::size_t i = 1; i <= nx_local_; ++i){

            // Compute the local values
            const double nu_u = std::fabs(at(hu0_, i, j)) / at(h0_, i, j) + sqrt(g * at(h0_, i, j));
            const double nu_v = std::fabs(at(hv0_, i, j)) / at(h0_, i, j) + sqrt(g * at(h0_, i, j));
            max_nu_sqr_local = std::max(max_nu_sqr_local, nu_u * nu_u + nu_v * nu_v);

        }
    }

    // Perform an allreduce operation using MPI_MAX
    MPI_Allreduce(&max_nu_sqr_local, &max_nu_sqr, 1, MPI_DOUBLE, MPI_MAX, cart_comm_);
    
    // Compute the global time step 
    const double dx = size_x_ / nx_;
    const double dy = size_y_ / ny_;
    double dt = std::min(dx, dy) / (sqrt(2.0 * max_nu_sqr));
    return std::min(dt, Tend - T);
}

// Compute the SWE update for a single cell (i, j) for the next time step
void SWESolver::compute_kernel(const std::size_t i,
                          const std::size_t j,
                          const double dt)
{   
    // Compute grid spacing and coefficients for the update
    const double dx = size_x_ / nx_;
    const double dy = size_y_ / ny_;
    const double C1x = 0.5 * dt / dx;
    const double C1y = 0.5 * dt / dy;
    const double C2 = dt * g;
    constexpr double C3 = 0.5 * g;

    // Compute updated water height
    double hij = 0.25 * (at(h0_, i, j - 1) + at(h0_, i, j + 1) + at(h0_, i - 1, j) + at(h0_, i + 1, j))
                + C1x * (at(hu0_, i - 1, j) - at(hu0_, i + 1, j)) + C1y * (at(hv0_, i, j - 1) - at(hv0_, i, j + 1));
    
    // Ensure non-negative water height
    if (hij < 0.0)
    {
        hij  = 1.0e-5;
    }

    // Update water height in the next time step
    at(h1_, i, j) = hij;

    // Update velocity components if water height is above threshold
    if (hij > 0.0001)
    {
        at(hu1_, i, j) =
          0.25 * (at(hu0_, i, j - 1) + at(hu0_, i, j + 1) + at(hu0_, i - 1, j) + at(hu0_, i + 1, j)) - C2 * hij * at(zdx_, i, j)
          + C1x
              * (at(hu0_, i - 1, j) * at(hu0_, i - 1, j) / at(h0_, i - 1, j) + C3 * at(h0_, i - 1, j) * at(h0_, i - 1, j)
                - at(hu0_, i + 1, j) * at(hu0_, i + 1, j) / at(h0_, i + 1, j) - C3 * at(h0_, i + 1, j) * at(h0_, i + 1, j))
          + C1y
              * (at(hu0_, i, j - 1) * at(hv0_, i, j - 1) / at(h0_, i, j - 1)
                - at(hu0_, i, j + 1) * at(hv0_, i, j + 1) / at(h0_, i, j + 1));

        at(hv1_, i, j) =
          0.25 * (at(hv0_, i, j - 1) + at(hv0_, i, j + 1) + at(hv0_, i - 1, j) + at(hv0_, i + 1, j)) - C2 * hij * at(zdy_, i, j)
          + C1x
              * (at(hu0_, i - 1, j) * at(hv0_, i - 1, j) / at(h0_, i - 1, j)
                - at(hu0_, i + 1, j) * at(hv0_, i + 1, j) / at(h0_, i + 1, j))
          + C1y
              * (at(hv0_, i, j - 1) * at(hv0_, i, j - 1) / at(h0_, i, j - 1) + C3 * at(h0_, i, j - 1) * at(h0_, i, j - 1)
                - at(hv0_, i, j + 1) * at(hv0_, i, j + 1) / at(h0_, i, j + 1) - C3 * at(h0_, i, j + 1) * at(h0_, i, j + 1));
    }
    else
    {
        at(hu1_, i, j) = 0.0;
        at(hv1_, i, j) = 0.0;
    }
}

// Function to compute the next time step for all interior cells
void SWESolver::solve_step(const double dt)
{
    for (std::size_t j = 1; j < ny_local_ + 1; ++j)
    {
        for (std::size_t i = 1; i < nx_local_ + 1; ++i)
        {
            // Compute the kernel for each i and j in the interior of the local grid
            this->compute_kernel(i, j, dt);
        }
    }
}

// Update boundary conditions for all edges
void SWESolver::update_bcs()
{ 
    // Coefficient depending on whether the boundary is reflective or not
    const double coef = this->reflective_ ? -1.0 : 1.0;

    // Compute the boundary conditions if the neighbor is MPI_PROC_NULL (i.e., no neighbor in that direction)
    if (nbr_left_ == MPI_PROC_NULL){
        // Left boundary
        for (std::size_t j = 1; j < ny_local_ + 1; ++j)
        {
            at(h1_, 0, j) = at(h0_, 1, j);
            at(hu1_, 0, j) = coef * at(hu0_, 1, j);
            at(hv1_, 0, j) = at(hv0_, 1, j);
        }
    }

    if (nbr_right_ == MPI_PROC_NULL){
        // Right boundary
        for (std::size_t j = 1; j < ny_local_ + 1; ++j)
        {
            at(h1_, nx_local_ + 1, j) = at(h0_, nx_local_, j);
            at(hu1_, nx_local_ + 1, j) = coef * at(hu0_, nx_local_, j);
            at(hv1_, nx_local_ + 1, j) = at(hv0_, nx_local_, j);
        }
    }

    if (nbr_down_ == MPI_PROC_NULL){
        // Bottom boundary
        for (std::size_t i = 1; i < nx_local_ + 1; ++i)
        {
            at(h1_, i, 0) = at(h0_, i, 1);
            at(hu1_, i, 0) = at(hu0_, i, 1);
            at(hv1_, i, 0) = coef * at(hv0_, i, 1);
        }
    }

    if (nbr_up_ == MPI_PROC_NULL){
        // Top boundary
        for (std::size_t i = 1; i < nx_local_ + 1; ++i)
        {
            at(h1_, i, ny_local_ + 1) = at(h0_, i, ny_local_);
            at(hu1_, i, ny_local_ + 1) = at(hu0_, i, ny_local_);
            at(hv1_, i, ny_local_ + 1) = coef * at(hv0_, i, ny_local_);
        }
    }
};

#include "swe.hh"
#include "xdmf_writer.hh"
#include <mpi.h>
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

SWESolver::SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny, MPI_Comm comm) :
  nx_(nx), ny_(ny), size_x_(500.0), size_y_(500.0), cart_comm_(MPI_COMM_NULL)
{
  
  
  // The communicator 
  MPI_Comm_size(comm, &size_);
  MPI_Comm_rank(comm, &rank_);
  dims_[0] = dims_[1] = 0;

  // fill dims_[0]*dims_[1] = size_
  MPI_Dims_create(size_, 2, dims_);            
  
  // Set the periodiciity to false
  int periods[2] = {0,0};

  // Create the Cartesian communicator
  MPI_Cart_create(comm, 2, dims_, periods, 1, &cart_comm_);
  MPI_Cart_coords(cart_comm_, rank_, 2, coords_);
  MPI_Cart_shift(cart_comm_, 0, 1, &nbr_west_, &nbr_east_);
  MPI_Cart_shift(cart_comm_, 1, 1, &nbr_south_, &nbr_north_);
  
  // I assume that the number of points in each direction is divisible by dims_[0] and dims_[1]
  assert(nx_ % dims_[0] == 0 && ny_ % dims_[1] == 0);
  local_nx_ = nx_ / dims_[0];
  local_ny_ = ny_ / dims_[1];
  offset_x_ = coords_[0]*local_nx_;
  offset_y_ = coords_[1]*local_ny_;

  // Add some buffer cells to the edges of the local grid
  auto alloc = [&](std::vector<double>& V){
    V.resize((local_nx_+2)*(local_ny_+2), 0.0);
  };

  // Add the buffer cells to each variable
  alloc(h0_); alloc(h1_);
  alloc(hu0_); alloc(hu1_);
  alloc(hv0_); alloc(hv1_);
  alloc(z_);  alloc(zdx_); alloc(zdy_);
  
  
  assert(test_case_id == 1 || test_case_id == 2);
  if (test_case_id == 1)
  {
    this->reflective_ = true;
    this->local_init_gaussian();
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


}

SWESolver::SWESolver(const std::string &h5_file, const double size_x, const double size_y) :
  size_x_(size_x), size_y_(size_y), reflective_(false)
{
  this->init_from_HDF5_file(h5_file);
}

void
SWESolver::init_from_HDF5_file(const std::string &h5_file)
{
  read_2d_array_from_DF5(h5_file, "h0", this->h0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hu0", this->hu0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "hv0", this->hv0_, this->nx_, this->ny_);
  read_2d_array_from_DF5(h5_file, "topography", this->z_, this->nx_, this->ny_);

  this->h1_.resize(this->h0_.size(), 0.0);
  this->hu1_.resize(this->hu0_.size(), 0.0);
  this->hv1_.resize(this->hv0_.size(), 0.0);

  this->init_dx_dy_local();
}

void SWESolver::local_init_gaussian(){
  const double x0_0 = size_x_ / 4.0;
  const double y0_0 = size_y_ / 3.0;
  const double x0_1 = size_x_ / 2.0;
  const double y0_1 = 0.75 * size_y_;

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;

  for (std::size_t j = 0; j < local_ny_ + 2; ++j){
    for (std::size_t i = 0; i < local_nx_ + 2; ++i){
      int gi = offset_x_ + i - 1;
      int gj = offset_y_ + j - 1;
      const double x = dx * (static_cast<double>(gi) + 0.5);
      const double y = dy * (static_cast<double>(gj) + 0.5);

      const double gauss_0 = 10.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 1000.0);
      const double gauss_1 = 10.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 1000.0);
      z_[j * (local_nx_ + 2) + i] = 0.0;
      h0_[j * (local_nx_ + 2) + i] = 10.0 + gauss_0 + gauss_1;
      hu0_[j * (local_nx_ + 2) + i] = 0.0;
      hv0_[j * (local_nx_ + 2) + i] = 0.0;
    }
  }

  // Communicate to all neighbors the initial conditions


  // Call init_dx_dy to compute the derivatives
  this->init_dx_dy_local();

}


void
SWESolver::init_dummy_tsunami()
{
  hu0_.resize(nx_ * ny_);
  hv0_.resize(nx_ * ny_);
  std::fill(hu0_.begin(), hu0_.end(), 0.0);
  std::fill(hv0_.begin(), hv0_.end(), 0.0);

  h1_.resize(nx_ * ny_);
  hu1_.resize(nx_ * ny_);
  hv1_.resize(nx_ * ny_);
  std::fill(h1_.begin(), h1_.end(), 0.0);
  std::fill(hu1_.begin(), hu1_.end(), 0.0);
  std::fill(hv1_.begin(), hv1_.end(), 0.0);

  const double x0_0 = 0.6 * size_x_;
  const double y0_0 = 0.6 * size_y_;
  const double x0_1 = 0.4 * size_x_;
  const double y0_1 = 0.4 * size_y_;
  const double x0_2 = 0.7 * size_x_;
  const double y0_2 = 0.3 * size_y_;

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;

  // Creating topography and initial water height
  z_.resize(nx_ * ny_);
  h0_.resize(nx_ * ny_);
  for (std::size_t j = 0; j < ny_; ++j)
  {
    for (std::size_t i = 0; i < nx_; ++i)
    {
      const double x = dx * (static_cast<double>(i) + 0.5);
      const double y = dy * (static_cast<double>(j) + 0.5);

      const double gauss_0 = 2.0 * std::exp(-((x - x0_0) * (x - x0_0) + (y - y0_0) * (y - y0_0)) / 3000.0);
      const double gauss_1 = 3.0 * std::exp(-((x - x0_1) * (x - x0_1) + (y - y0_1) * (y - y0_1)) / 10000.0);
      const double gauss_2 = 5.0 * std::exp(-((x - x0_2) * (x - x0_2) + (y - y0_2) * (y - y0_2)) / 100.0);

      const double z = -1.0 + gauss_0 + gauss_1;
      at(z_, i, j) = z;

      double h0 = z < 0.0 ? -z + gauss_2 : 0.00001;
      at(h0_, i, j) = h0;
    }
  }
  this->init_dx_dy_local();
}

void
SWESolver::init_dummy_slope()
{
  hu0_.resize(nx_ * ny_);
  hv0_.resize(nx_ * ny_);
  std::fill(hu0_.begin(), hu0_.end(), 0.0);
  std::fill(hv0_.begin(), hv0_.end(), 0.0);

  h1_.resize(nx_ * ny_);
  hu1_.resize(nx_ * ny_);
  hv1_.resize(nx_ * ny_);
  std::fill(h1_.begin(), h1_.end(), 0.0);
  std::fill(hu1_.begin(), hu1_.end(), 0.0);
  std::fill(hv1_.begin(), hv1_.end(), 0.0);

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;

  const double dz = 10.0;

  // Creating topography and initial water height
  z_.resize(nx_ * ny_);
  h0_.resize(nx_ * ny_);
  for (std::size_t j = 0; j < ny_; ++j)
  {
    for (std::size_t i = 0; i < nx_; ++i)
    {
      const double x = dx * (static_cast<double>(i) + 0.5);
      const double y = dy * (static_cast<double>(j) + 0.5);
      static_cast<void>(y);

      const double z = -10.0 - 0.5 * dz + dz / size_x_ * x;
      at(z_, i, j) = z;

      double h0 = z < 0.0 ? -z : 0.00001;
      at(h0_, i, j) = h0;
    }
  }
  this->init_dx_dy_local();
}

void
SWESolver::init_dx_dy_local()
{
  zdx_.resize(this->z_.size(), 0.0);
  zdy_.resize(this->z_.size(), 0.0);

  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;
  for (std::size_t j = 1; j < ny_ - 1; ++j)
  {
    for (std::size_t i = 1; i < nx_ - 1; ++i)
    {
      at(this->zdx_, i, j) = 0.5 * (at(this->z_, i + 1, j) - at(this->z_, i - 1, j)) / dx;
      at(this->zdy_, i, j) = 0.5 * (at(this->z_, i, j + 1) - at(this->z_, i, j - 1)) / dy;
    }
  }
}

void SWESolver::exchange_halos()
{
  MPI_Datatype column_t;
  MPI_Type_vector(local_ny_, 1, local_nx_+2, MPI_DOUBLE, &column_t);
  MPI_Type_commit(&column_t);

  MPI_Datatype row_t;
  MPI_Type_vector(local_nx_, 1, 1, MPI_DOUBLE, &row_t);
  MPI_Type_commit(&row_t);

  // Anonymous lambda function to exchange 
  auto exch = [&](std::vector<double>& F){
    // west<->east
    MPI_Sendrecv(&at(F, 1, 1, local_nx_), 1, column_t, nbr_west_, 0,
                 &at(F, local_nx_ + 1, 1, local_nx_), 1, column_t, nbr_east_, 0,
                 cart_comm_, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&at(F, local_nx_, 1, local_nx_), 1, column_t, nbr_east_, 1,
                 &at(F,0,1), 1, column_t, nbr_west_, 1,
                 cart_comm_, MPI_STATUS_IGNORE);
    // south<->north
    MPI_Sendrecv(&at(F, 1, 1, local_nx_), 1, row_t, nbr_south_, 2,
                 &at(F, 1, local_ny_ + 1, local_nx_), 1, row_t, nbr_north_, 2,
                 cart_comm_, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&at(F, 1, local_ny_, local_nx_), 1, row_t, nbr_north_, 3,
                 &at(F, 1, 0, local_nx_), 1, row_t, nbr_south_, 3,
                 cart_comm_, MPI_STATUS_IGNORE);
  };

  for (auto *F : { &h0_, &hu0_, &hv0_, &zdx_, &zdy_ })
    exch(*F);

  MPI_Type_free(&column_t);
  MPI_Type_free(&row_t);
}

void
SWESolver::solve(const double Tend, const bool full_log, const std::size_t output_n, const std::string &fname_prefix)
{
  int rank_int, size_int;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_int);
  MPI_Comm_size(MPI_COMM_WORLD, &size_int);
  std::shared_ptr<XDMFWriter> writer;
  if (output_n > 0 && rank_int == 0)
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

  if (rank_int == 0){
    std::cout << "Solving SWE..." << std::endl;
  }

  std::size_t nt = 1;
  while (T < Tend)
  { 
    // 1) gather max local wave‐speed for global CFL
    const double dt = this->compute_time_step(h0, hu0, hv0, T, Tend);

    const double T1 = T + dt;

    
    if (rank_int == 0){
      printf("Computing T: %2.4f hr  (dt = %.2e s) -- %3.3f%%", T1, dt * 3600, 100 * T1 / Tend);
    std::cout << (full_log ? "\n" : "\r") << std::flush;
    }
    

    this->update_bcs(h0, hu0, hv0, h, hu, hv);

    this->solve_step(dt, h0, hu0, hv0, h, hu, hv);

    if (output_n > 0 && nt % output_n == 0 && rank_int == 0)
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

  if (output_n > 0 && rank_int == 0)
  {
    writer->add_h(h1_, T);
  }

  if (rank_int == 0){
    std::cout << "Finished solving SWE." << std::endl;
  }
  
}

// The function to be parallelized
// double
// SWESolver::compute_time_step(const std::vector<double> &h,
//                              const std::vector<double> &hu,
//                              const std::vector<double> &hv,
//                              const double T,
//                              const double Tend) const
// {
//   double max_nu_sqr = 0.0;
//   double au{0.0};
//   double av{0.0};
//   for (std::size_t j = 1; j < ny_ - 1; ++j)
//   {
//     for (std::size_t i = 1; i < nx_ - 1; ++i)
//     {
//       au = std::max(au, std::fabs(at(hu, i, j)));
//       av = std::max(av, std::fabs(at(hv, i, j)));
//       const double nu_u = (at(hu, i, j)) / at(h, i, j) + sqrt(g * at(h, i, j));
//       const double nu_v = std::fabs(at(hv, i, j)) / at(h, i, j) + sqrt(g * at(h, i, j));
//       max_nu_sqr = std::max(max_nu_sqr, nu_u * nu_u + nu_v * nu_v);
//     }
//   }

//   const double dx = size_x_ / nx_;
//   const double dy = size_y_ / ny_;
//   double dt = std::min(dx, dy) / (sqrt(2.0 * max_nu_sqr));
//   return std::min(dt, Tend - T);
// }

double
SWESolver::compute_time_step(const std::vector<double> &h,
                             const std::vector<double> &hu,
                             const std::vector<double> &hv,
                             const double T,
                             const double Tend) const
{

  // Compute the numbers of rows each process should get
  // std::size_t rank = static_cast<std::size_t>(rank_);
  // std::size_t size = static_cast<std::size_t>(size_);

  // std::size_t width = nx_ - 2;
  // std::size_t height = ny_ - 2;

  // int total_entries = width * height;
  // size_t entries_per_pro = total_entries / size;
  // size_t remainder = total_entries % size;

  // // Each rank gets rows_per_pro, and the first 'remainder' ranks get one more
  // size_t start = rank * entries_per_pro + std::min(rank, remainder);
  // size_t end = start + entries_per_pro + (rank < remainder ? 1 : 0);

  // // Test that the above logic is correct by printing the start and end rows for each rank
  // // print the size
  // if (DEBUG)
  // {
  //   std::cout << "Rank " << rank << " of " << size << std::endl;
  //   std::cout << "Rank " << rank << ": start = " << start << ", end = " << end << std::endl;
  // }
    
  double max_nu_sqr = 0.0;

  // Init the local variables
  double max_nu_sqr_local = 0.0;
  for(std::size_t j = 0; j <= local_ny_; ++j){
    for(std::size_t i = 0; i <= local_nx_; ++i){

      // Compute the local values
      const double nu_u = (at(hu, i, j)) / at(h, i, j) + sqrt(g * at(h, i, j));
      const double nu_v = std::fabs(at(hv, i, j)) / at(h, i, j) + sqrt(g * at(h, i, j));
      max_nu_sqr_local = std::max(max_nu_sqr_local, nu_u * nu_u + nu_v * nu_v);

    }
  }

  // Perform an allreduce operation using MPI_MAX
  MPI_Allreduce(&max_nu_sqr_local, &max_nu_sqr, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  
  // Compute the global time step 
  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;
  double dt = std::min(dx, dy) / (sqrt(2.0 * max_nu_sqr));
  return std::min(dt, Tend - T);
}

void
SWESolver::compute_kernel(const std::size_t i,
                          const std::size_t j,
                          const double dt,
                          const std::vector<double> &h0,
                          const std::vector<double> &hu0,
                          const std::vector<double> &hv0,
                          std::vector<double> &h,
                          std::vector<double> &hu,
                          std::vector<double> &hv) const
{
  const double dx = size_x_ / nx_;
  const double dy = size_y_ / ny_;
  const double C1x = 0.5 * dt / dx;
  const double C1y = 0.5 * dt / dy;
  const double C2 = dt * g;
  constexpr double C3 = 0.5 * g;

  double hij = 0.25 * (at(h0, i, j - 1) + at(h0, i, j + 1) + at(h0, i - 1, j) + at(h0, i + 1, j))
               + C1x * (at(hu0, i - 1, j) - at(hu0, i + 1, j)) + C1y * (at(hv0, i, j - 1) - at(hv0, i, j + 1));
  if (hij < 0.0)
  {
    hij = 1.0e-5;
  }

  at(h, i, j) = hij;

  if (hij > 0.0001)
  {
    at(hu, i, j) =
      0.25 * (at(hu0, i, j - 1) + at(hu0, i, j + 1) + at(hu0, i - 1, j) + at(hu0, i + 1, j)) - C2 * hij * at(zdx_, i, j)
      + C1x
          * (at(hu0, i - 1, j) * at(hu0, i - 1, j) / at(h0, i - 1, j) + C3 * at(h0, i - 1, j) * at(h0, i - 1, j)
             - at(hu0, i + 1, j) * at(hu0, i + 1, j) / at(h0, i + 1, j) - C3 * at(h0, i + 1, j) * at(h0, i + 1, j))
      + C1y
          * (at(hu0, i, j - 1) * at(hv0, i, j - 1) / at(h0, i, j - 1)
             - at(hu0, i, j + 1) * at(hv0, i, j + 1) / at(h0, i, j + 1));

    at(hv, i, j) =
      0.25 * (at(hv0, i, j - 1) + at(hv0, i, j + 1) + at(hv0, i - 1, j) + at(hv0, i + 1, j)) - C2 * hij * at(zdy_, i, j)
      + C1x
          * (at(hu0, i - 1, j) * at(hv0, i - 1, j) / at(h0, i - 1, j)
             - at(hu0, i + 1, j) * at(hv0, i + 1, j) / at(h0, i + 1, j))
      + C1y
          * (at(hv0, i, j - 1) * at(hv0, i, j - 1) / at(h0, i, j - 1) + C3 * at(h0, i, j - 1) * at(h0, i, j - 1)
             - at(hv0, i, j + 1) * at(hv0, i, j + 1) / at(h0, i, j + 1) - C3 * at(h0, i, j + 1) * at(h0, i, j + 1));
  }
  else
  {
    at(hu, i, j) = 0.0;
    at(hv, i, j) = 0.0;
  }

  // h(2:nx-1,2:nx-1) = 0.25*(h0(2:nx-1,1:nx-2)+h0(2:nx-1,3:nx)+h0(1:nx-2,2:nx-1)+h0(3:nx,2:nx-1)) ...
  //     + C1*( hu0(2:nx-1,1:nx-2) - hu0(2:nx-1,3:nx) + hv0(1:nx-2,2:nx-1) - hvhv0:nx,2:nx-1) );

  // hu(2:nx-1,2:nx-1) = 0.25*(hu0(2:nx-1,1:nx-2)+hu0(2:nx-1,3:nx)+hu0(1:nx-2,2:nx-1)+hu0(3:nx,2:nx-1)) -
  // C2*H(2:nx-1,2:nx-1).*Zdx(2:nx-1,2:nx-1) ...
  //     + C1*( hu0(2:nx-1,1:nx-2).^2./h0(2:nx-1,1:nx-2) + 0.5*g*h0(2:nx-1,1:nx-2).^2 -
  //     hu0(2:nx-1,3:nx).^2./h0(2:nx-1,3:nx) - 0.5*g*h0(2:nx-1,3:nx).^2 ) ...
  //     + C1*( hu0(1:nx-2,2:nx-1).*hv0(1:nx-2,2:nx-1)./h0(1:nx-2,2:nx-1) -
  //     hu0(3:nx,2:nx-1).*hv0(3:nx,2:nx-1)./h0(3:nx,2:nx-1) );

  // hv(2:nx-1,2:nx-1) = 0.25*(hv0(2:nx-1,1:nx-2)+hv0(2:nx-1,3:nx)+hv0(1:nx-2,2:nx-1)+hv0(3:nx,2:nx-1)) -
  // C2*H(2:nx-1,2:nx-1).*Zdy(2:nx-1,2:nx-1)  ...
  //     + C1*( hu0(2:nx-1,1:nx-2).*hv0(2:nx-1,1:nx-2)./h0(2:nx-1,1:nx-2) -
  //     hu0(2:nx-1,3:nx).*hv0(2:nx-1,3:nx)./h0(2:nx-1,3:nx) ) ...
  //     + C1*( hv0(1:nx-2,2:nx-1).^2./h0(1:nx-2,2:nx-1) + 0.5*g*h0(1:nx-2,2:nx-1).^2 -
  //     hv0(3:nx,2:nx-1).^2./h0(3:nx,2:nx-1) - 0.5*g*h0(3:nx,2:nx-1).^2  );
}

void
SWESolver::solve_step(const double dt,
                      const std::vector<double> &h0,
                      const std::vector<double> &hu0,
                      const std::vector<double> &hv0,
                      std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv) const
{
  for (std::size_t j = 1; j < ny_ - 1; ++j)
  {
    for (std::size_t i = 1; i < nx_ - 1; ++i)
    {
      this->compute_kernel(i, j, dt, h0, hu0, hv0, h, hu, hv);
    }
  }
}

void
SWESolver::update_bcs(const std::vector<double> &h0,
                      const std::vector<double> &hu0,
                      const std::vector<double> &hv0,
                      std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv) const
{
  const double coef = this->reflective_ ? -1.0 : 1.0;

  // Top and bottom boundaries.
  for (std::size_t i = 0; i < nx_; ++i)
  {
    at(h, i, 0) = at(h0, i, 1);
    at(h, i, ny_ - 1) = at(h0, i, ny_ - 2);

    at(hu, i, 0) = at(hu0, i, 1);
    at(hu, i, ny_ - 1) = at(hu0, i, ny_ - 2);

    at(hv, i, 0) = coef * at(hv0, i, 1);
    at(hv, i, ny_ - 1) = coef * at(hv0, i, ny_ - 2);
  }

  // Left and right boundaries.
  for (std::size_t j = 0; j < ny_; ++j)
  {
    at(h, 0, j) = at(h0, 1, j);
    at(h, nx_ - 1, j) = at(h0, nx_ - 2, j);

    at(hu, 0, j) = coef * at(hu0, 1, j);
    at(hu, nx_ - 1, j) = coef * at(hu0, nx_ - 2, j);

    at(hv, 0, j) = at(hv0, 1, j);
    at(hv, nx_ - 1, j) = at(hv0, nx_ - 2, j);
  }
};

#include <cstddef>
#include <vector>
#include <string>
#include <cuda_runtime.h>

// Forward declaration for CUDA struct
struct SWEData;

class SWESolver
{
public:
  /**
   * @brief Deleted default constructor. SWESolver must be constructed with parameters.
   */
  SWESolver() = delete;

  /// Gravity constant in [km/hour^2]: 9.82 * (3.6)^2 * 1000
  static constexpr double g = 127267.20000000;

  /**
   * @brief Construct solver for built-in test cases.
   * @param test_case_id 1: water drop, 2: dummy tsunami.
   * @param nx Number of grid cells in x.
   * @param ny Number of grid cells in y.
   * @param threads_per_block Number of CUDA threads per block.
   */
  SWESolver(const int test_case_id, const std::size_t nx, const std::size_t ny, const int threads_per_block);

  /**
   * @brief Construct solver using initial conditions and topography from HDF5 file.
   * @param h5_file HDF5 file name.
   * @param size_x Domain size in x (km).
   * @param size_y Domain size in y (km).
   * @param threads_per_block Number of CUDA threads per block.
   */
  SWESolver(const std::string &h5_file, const double size_x, const double size_y, const int threads_per_block);

  /**
   * @brief Solve the shallow water equations.
   * @param Tend End time (hours).
   * @param full_log If true, print log at every step.
   * @param output_n Output every n steps (0 disables output).
   * @param fname_prefix Output file prefix.
   */
  void solve(const double Tend,
             const bool full_log = false,
             const std::size_t output_n = 0,
             const std::string &fname_prefix = "test");

  /// Destructor: releases device memory.
  ~SWESolver();

private:
  /**
   * @brief Initialize from HDF5 file.
   * @param h5_file HDF5 file name.
   */
  void init_from_HDF5_file(const std::string &h5_file);

  /**
   * @brief Initialize with two Gaussian peaks (water drop test).
   */
  void init_gaussian();

  /**
   * @brief Initialize with dummy tsunami scenario.
   */
  void init_dummy_tsunami();

  // Grid and domain parameters
  std::size_t nx_;
  std::size_t ny_;
  double size_x_;
  double size_y_;
  bool reflective_;

  // Host-side solution arrays
  std::vector<double> h0_;
  std::vector<double> h1_;
  std::vector<double> hu0_;
  std::vector<double> hu1_;
  std::vector<double> hv0_;
  std::vector<double> hv1_;
  std::vector<double> z_;
  std::vector<double> zdx_;
  std::vector<double> zdy_;

  // Device pointers
  SWEData* data_device_;
  double* dt_;
  bool* reflective_device_;
  double* max_partial_;

  // CUDA launch configuration
  int threads_per_block_;
  dim3 block_dims_;
  dim3 grid_dims_;

  /**
   * @brief 2D accessor for mutable host vectors.
   */
  inline double &at(std::vector<double> &vec, const std::size_t i, const std::size_t j) const
  {
    return vec[j * nx_ + i];
  }

  /**
   * @brief 2D accessor for const host vectors.
   */
  inline const double &at(const std::vector<double> &vec, const std::size_t i, const std::size_t j) const
  {
    return vec[j * nx_ + i];
  }

  /**
   * @brief Update cell (i,j) using SWE kernel (host-side, unused in CUDA).
   * @param i x-index.
   * @param j y-index.
   * @param dt Time step.
   * @param h0 Water height at previous step.
   * @param hu0 x-velocity at previous step.
   * @param hv0 y-velocity at previous step.
   * @param h Water height at current step.
   * @param hu x-velocity at current step.
   * @param hv y-velocity at current step.
   */
  void compute_kernel(const std::size_t i,
                      const std::size_t j,
                      const double dt,
                      const std::vector<double> &h0,
                      const std::vector<double> &hu0,
                      const std::vector<double> &hv0,
                      std::vector<double> &h,
                      std::vector<double> &hu,
                      std::vector<double> &hv) const;

  /**
   * @brief Compute time step size satisfying CFL condition (device).
   * @param T Current time.
   * @param Tend End time.
   */
  void compute_time_step(const double T,
                         const double Tend);

  /**
   * @brief Advance solution by one time step (device).
   */
  void solve_step() const;

  /**
   * @brief Update boundary conditions (device).
   */
  void update_bcs() const;

  /**
   * @brief Swap pointers for old/new solution arrays (device).
   */
  void swap_data() const;
};
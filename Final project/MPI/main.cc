#include "swe.hh"
#include <mpi.h>
#include <string>
#include <cstddef>
#include <iostream>
#include <chrono>

int
main()
{ 

  // Initialize MPI
  MPI_Init(NULL, NULL); // Initialize MPI
   // Get the rank and size of the MPI communicator
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Uncomment the option you want to run.

  // Option 1 - Solving simple problem: water drops in a box
  const int test_case_id = 1;  // Water drops in a box
  const double Tend = 1.0;     // Simulation time in hours
  const std::size_t nx = 1000; // Number of cells per direction.
  const std::size_t ny = 1000; // Number of cells per direction.
  const std::size_t output_n = 10;
  const std::string output_fname = "output_files/water_drops";
  const bool full_log = false;

  SWESolver solver(test_case_id, nx, ny);
  // Time the solver
  auto start_time = std::chrono::high_resolution_clock::now();
  solver.solve(Tend, full_log, output_n, output_fname);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end_time - start_time;



  // // Option 2 - Solving analytical (dummy) tsunami example.
  // const int test_case_id = 2;  // Analytical tsunami test case
  // const double Tend = 1.0;     // Simulation time in hours
  // const std::size_t nx = 1000; // Number of cells per direction.
  // const std::size_t ny = 1000; // Number of cells per direction.
  // const std::size_t output_n = 10;
  // const std::string output_fname = "analytical_tsunami";
  // const bool full_log = false;

  // SWESolver solver(test_case_id, nx, ny);
  // auto start_time = std::chrono::high_resolution_clock::now();
  // solver.solve(Tend, full_log, output_n, output_fname);
  // auto end_time = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_time = end_time - start_time;

  // // Option 3 - Solving tsunami problem with data loaded from file.
  // const double Tend = 0.2;   // Simulation time in hours
  // const double size = 500.0; // Size of the domain in km

  // // const std::string fname = "Data_nx501_500km.h5"; // File containg initial data (501x501 mesh).
  // const std::string fname = "Data_nx1001_500km.h5"; // File containg initial data (1001x1001 mesh).
  // // const std::string fname = "Data_nx2001_500km.h5"; // File containg initial data (2001x2001 mesh).
  // // const std::string fname = "Data_nx4001_500km.h5"; // File containg initial data (4001x4001 mesh).
  // // const std::string fname = "Data_nx8001_500km.h5"; // File containg initial data (8001x8001 mesh).

  // const std::size_t output_n = 0;
  // const std::string output_fname = "tsunami";
  // const bool full_log = false;

  // SWESolver solver(fname, size, size);
  // auto start_time = std::chrono::high_resolution_clock::now();
  // solver.solve(Tend, full_log, output_n, output_fname);
  // auto end_time = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double> elapsed_time = end_time - start_time;

  // Print the elapsed time and number of processors used
  double elapsed_time_seconds = elapsed_time.count();
  if (rank == 0) {
    std::cout << "Elapsed time: " << elapsed_time_seconds << " seconds" << std::endl;
    std::cout << "Number of processors used: " << size << std::endl;
  }

  return 0;
}

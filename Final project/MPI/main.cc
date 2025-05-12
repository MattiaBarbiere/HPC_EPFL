#include "swe.hh"
#include <mpi.h>
#include <string>
#include <cstddef>
#include <iostream>
#include <chrono>


int main(int argc, char* argv[])
{ 
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure the correct number of arguments are passed
    if (argc != 5) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <test_case_id> <nx> <ny> <output_n>\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    // Parse command-line arguments
    int test_case_id = std::atoi(argv[1]);
    std::size_t nx = std::atoi(argv[2]);
    std::size_t ny = std::atoi(argv[3]);
    std::size_t output_n = std::atoi(argv[4]);

    // Simulation time
    double Tend = 1.0;

    if (rank == 0) {
        std::cout << "Running test case " << test_case_id << " with grid " 
                  << nx << "x" << ny << " and " << output_n << " outputs.\n";
    }

    // Variable to store elapsed time
    double elapsed_time_seconds = 0.0;

  // Uncomment the option you want to run.

  if (test_case_id == 1) {
    // Option 1 - Solving simple problem: water drops in a box
    const std::string output_fname = "output_files/water_drops";
    const bool full_log = false;

    SWESolver solver(test_case_id, nx, ny);
    // Time the solver
    auto start_time = std::chrono::high_resolution_clock::now();
    solver.solve(Tend, full_log, output_n, output_fname);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    elapsed_time_seconds = elapsed_time.count();
  }
  else if (test_case_id == 2) {
    // Option 2 - Solving analytical (dummy) tsunami example.

    const std::string output_fname = "output_files/analytical_tsunami";
    const bool full_log = false;

    SWESolver solver(test_case_id, nx, ny);
    auto start_time = std::chrono::high_resolution_clock::now();
    solver.solve(Tend, full_log, output_n, output_fname);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    elapsed_time_seconds = elapsed_time.count();
  }
  else if (test_case_id == 3) {
    // Option 3 - Solving tsunami problem with data loaded from file.
    Tend = 0.2;   // Simulation time in hours

    // Make sure that input nx and ny are the same
    if (nx != ny) {
        if (rank == 0) {
            std::cerr << "nx and ny must be the same for test case 3.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    // Convert nx and ny to double
    const double size = static_cast<double>(nx); // Size of the domain in km

    // const std::string fname = "Data_nx501_500km.h5"; // File containg initial data (501x501 mesh).
    const std::string fname = "Data_nx1001_500km.h5"; // File containg initial data (1001x1001 mesh).
    // const std::string fname = "Data_nx2001_500km.h5"; // File containg initial data (2001x2001 mesh).
    // const std::string fname = "Data_nx4001_500km.h5"; // File containg initial data (4001x4001 mesh).
    // const std::string fname = "Data_nx8001_500km.h5"; // File containg initial data (8001x8001 mesh).

    const std::size_t output_n = 0;
    const std::string output_fname = "output_files/tsunami";
    const bool full_log = false;

    SWESolver solver(fname, size, size);
    auto start_time = std::chrono::high_resolution_clock::now();
    solver.solve(Tend, full_log, output_n, output_fname);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    elapsed_time_seconds = elapsed_time.count();
  }
  else {
    if (rank == 0) {
        std::cerr << "Invalid test case ID. Must be 1, 2, or 3.\n";
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }  

  // Print the elapsed time and number of processors used
  if (rank == 0) {
    std::cout << "Elapsed time: " << elapsed_time_seconds << " seconds" << std::endl;
    std::cout << "Number of processors used: " << size << std::endl;
  }

  return 0;
}

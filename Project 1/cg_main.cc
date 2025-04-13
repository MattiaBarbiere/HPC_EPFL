#include "cg.hh"
#include <chrono>
#include <iostream>
#include <mpi.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

// Flag to print time information
const bool SHOW_TIME = true;

/*
Simple CG solver using a matrix in Matrix Market (MTX) format.
*/

int main(int argc, char **argv)
{
  // Initialize MPI
  MPI_Init(&argc, &argv); // Initialize MPI
   // Get the rank and size of the MPI communicator
   int rank, size;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc < 2)
  {
    // Ensure a matrix file is provided as an argument
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename]" << std::endl;
    return 1;
  }


  CGSolverSparse sparse_solver;

  // Begin timing for loading
  auto start_time_loading = clk::now(); // Start timing

  sparse_solver.read_matrix(argv[1]); // Load the matrix from the file
  int n = sparse_solver.n();         // Number of rows
  //int m = sparse_solver.m();         // Number of columns
  double h = 1. / n;                 // Grid spacing

  sparse_solver.init_source_term(h); // Initialize the source term

  std::vector<double> x_s(n, 0.);    // Initialize solution vector with zeros
   // End timing for loading
  second elapsed_loading = clk::now() - start_time_loading; // Compute elapsed time
  auto t_loading = elapsed_loading.count(); // Get elapsed time in seconds





  if (rank == 0){
  std::cout << argv[1] << std::endl;

  }
  
  auto t1 = clk::now();              // Start timing
  
  sparse_solver.solve(x_s);          // Solve the system
  
  second elapsed = clk::now() - t1;  // Compute elapsed time
  auto t_solving = elapsed.count(); // Get elapsed time in seconds

  // Get the average time across all processes
  double avg_time_solving;
  double avg_time_loading;
  MPI_Reduce(&(t_solving), &avg_time_solving, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&(t_loading), &avg_time_loading, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  avg_time_solving /= size;
  avg_time_loading /= size;

  // Total time for loading and solving
  double total_time = t_loading + t_solving;

  if (rank == 0 && SHOW_TIME)
  {
    std::cout << "p = " << size << " Time = " << total_time << " [s]\n";
    std::cout << std::endl;
  }
  
  
  MPI_Finalize(); // Finalize MPI
  return 0; // Exit successfully
}
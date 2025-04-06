#include "cg.hh"
#include <chrono>
#include <iostream>
#include <mpi.h>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

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
  sparse_solver.read_matrix(argv[1]); // Load the matrix from the file
  int n = sparse_solver.n();         // Number of rows
  int m = sparse_solver.m();         // Number of columns
  double h = 1. / n;                 // Grid spacing

  sparse_solver.init_source_term(h); // Initialize the source term

  std::vector<double> x_s(n, 0.);    // Initialize solution vector with zeros

  if (rank == 0){
  std::cout << "Call CG sparse on matrix size " << m << " x " << n << ")" << std::endl;
  }
  
  auto t1 = clk::now();              // Start timing
  sparse_solver.solve(x_s);          // Solve the system
  
  second elapsed = clk::now() - t1;  // Compute elapsed time
  auto t = elapsed.count(); // Get elapsed time in seconds

  // Get the average time across all processes
  double avg_time;
  MPI_Reduce(&(t), &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  avg_time /= size; // Average time across all processes

  if (rank == 0){
  std::cout << "Time for CG (sparse solver)  = " << avg_time << " [s]\n";
  }
  MPI_Finalize(); // Finalize MPI
  return 0; // Exit successfully
}
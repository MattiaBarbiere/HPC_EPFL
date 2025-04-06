#include "cg.hh"
#include <chrono>
#include <iostream>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Simple CG solver using a matrix in Matrix Market (MTX) format.
*/

int main(int argc, char **argv)
{
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

  std::cout << "Call CG sparse on matrix size " << m << " x " << n << ")" << std::endl;

  auto t1 = clk::now();              // Start timing
  sparse_solver.solve(x_s);          // Solve the system
  second elapsed = clk::now() - t1;  // Compute elapsed time

  std::cout << "Time for CG (sparse solver)  = " << elapsed.count() << " [s]\n";

  return 0; // Exit successfully
}
#include "cg.hh"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code
*/
int main(int argc, char ** argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename]"
              << std::endl;
    return 1;
  }

  CGSolver solver;
  solver.read_matrix(argv[1]);

  // Print the matrix we are working with
  std::cout << argv[1] << std::endl;
  

  int n = solver.n();
  int m = solver.m();
  double h = 1. / n;

  int threads_per_block = std::stoi(argv[2]);
  int blocks_per_grid = (10 + (threads_per_block - 1)) / threads_per_block;

  


  solver.init_source_term(h);

  std::vector<double> x_d(n);
  std::fill(x_d.begin(), x_d.end(), 0.);

  std::cout << "Call CG dense on matrix size " << m << " x " << n 
            << std::endl;

  //Print some information about threads
  std::cout << "Threads per block:" << threads_per_block  << std::endl;
  std::cout << "Blocks per grid:" << blocks_per_grid  << std::endl;
  
  auto t1 = clk::now();
  solver.solve(x_d, threads_per_block, blocks_per_grid);
  second elapsed = clk::now() - t1;
  std::cout << "Time for CG (dense solver)  = " << elapsed.count() << " [s]\n";

  return 0;
}

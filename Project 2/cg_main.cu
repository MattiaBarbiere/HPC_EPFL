#include "cg.hh"
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

__global__ void test_kernel() {
  printf("Hello from the kernel\n");
}

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
  

  int n = solver.n();
  int m = solver.m();
  double h = 1. / n;

  int threads_per_block = std::stoi(argv[2]);
  std::cout << threads_per_block  << std::endl;
  int blocks_per_grid = (10 + (threads_per_block - 1)) / threads_per_block;
  std::cout << blocks_per_grid  << std::endl;

  test_kernel<<<blocks_per_grid, threads_per_block>>>();

  solver.init_source_term(h);

  std::vector<double> x_d(n);
  std::fill(x_d.begin(), x_d.end(), 0.);

  std::cout << "Call CG dense on matrix size " << m << " x " << n 
            << std::endl;
  auto t1 = clk::now();
  solver.solve(x_d);
  second elapsed = clk::now() - t1;
  std::cout << "Time for CG (dense solver)  = " << elapsed.count() << " [s]\n";

  return 0;
}

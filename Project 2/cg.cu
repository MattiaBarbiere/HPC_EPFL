#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

const double NEARZERO = 1.0e-14;
const bool DEBUG = false;
const bool SHOW_RESULT = true;

// A simple function that computes the id of the thread
__device__
int get_id() {
  int id_in_block = threadIdx.y * blockDim.x + threadIdx.x;
  int offset = blockIdx.y * gridDim.x + blockIdx.x;
  return offset * blockDim.x * blockDim.y + id_in_block;
}

// A function that computes the row id of the thread
__device__ 
int get_row_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

// A cuda kernel that computes the dot product of two vectors
__global__
void cuda_ddot(double* result, int n, const double* x, int incx, const double* y, int incy) {
  int id = get_id();
  if (id == 0) {
    *result = 0.0;
    for (int i = 0; i < n; ++i) {
        *result += x[i * incx] * y[i * incy];
    }
  }
}

// A cuda kernel that computes the matrix-vector product
__global__
void cuda_dgemv(int m_m, int m_n,
                double alpha, const double* A, int leadingDim,
                const double* x, int incX,
                double beta, double* y, int incY) {
    int row = get_row_id();
    if (row < m_m) {
        double sum = 0.0;
        for (int j = 0; j < m_n; ++j) {
            sum += A[row * leadingDim + j] * x[j * incX];
        }
        y[row * incY] = alpha * sum + beta * y[row * incY];
    }
}

// A cuda kernel that computes the axpy operation
__global__ // I also add this function to host for debugging
void cuda_daxpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
  int id = get_id();
  if (id < n) {
    y[id * incy] += alpha * x[id * incx];
  }
}

// A cuda kernel that copies a vector on the GPU memory
__global__
void cuda_copy(double* dst, const double* src, int n) {
    int row = get_row_id();
    if (row < n){
      dst[row] = src[row];
    }
}

// A cuda kernel that fills a vector with a given value
__global__
void cuda_fill(double* vec, double val, int n) {
    int row = get_row_id();
    if (row < n) {
      vec[row] = val;
    }
}


void CGSolver::solve(std::vector<double>& x, int threads_per_block, int blocks_per_grid) {
  
  // Variables we will work with
  double* r = new double[m_n];
  double* p = new double[m_n];
  double* Ap = new double[m_n];
  double* tmp = new double[m_n];
  double* rsold;
  double* rsnew;
  double* pAp;
  
  cudaMallocManaged(&r, m_n * sizeof(double));
  cudaMallocManaged(&p, m_n * sizeof(double));
  cudaMallocManaged(&Ap, m_n * sizeof(double));
  cudaMallocManaged(&tmp, m_n * sizeof(double));
  cudaMallocManaged(&rsold, sizeof(double));
  cudaMallocManaged(&rsnew, sizeof(double));
  cudaMallocManaged(&pAp, sizeof(double));

  // Copies the constant data to the device
  double* A_device;
  double* b_device;
  double* x_device;
  cudaMalloc(&A_device, m_m * m_n * sizeof(double));
  cudaMalloc(&b_device, m_n * sizeof(double));
  cudaMalloc(&x_device, m_n * sizeof(double));
  cudaMemcpy(A_device, m_A.data(), m_m * m_n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(b_device, m_b.data(), m_n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(x_device, x.data(), m_n * sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  // r = b - A * x
  cuda_fill<<<blocks_per_grid, threads_per_block>>>(Ap, 0., m_n);
  cuda_dgemv<<<blocks_per_grid, threads_per_block>>>(m_m, m_n, 1., A_device, m_n, x_device, 1, 0., Ap, 1);

  // std::copy(m_b.data(), m_b.data() + m_n, r);
  cuda_copy<<<blocks_per_grid, threads_per_block>>>(r, b_device, m_n);
  cuda_daxpy<<<blocks_per_grid, threads_per_block>>>(m_n, -1., Ap, 1, r, 1);

  // p = r
  // std::copy(r, r + m_n, p);
  cuda_copy<<<blocks_per_grid, threads_per_block>>>(p, r, m_n);
  

  // rsold = r' * r
  cuda_ddot<<<blocks_per_grid, threads_per_block>>>(rsold, m_n, r, 1, p, 1);
  // cudaDeviceSynchronize();

  int k = 0;
  for (; k < m_n; ++k) {
      if (DEBUG) {
          std::cout << "\tSTEP " << k << std::endl;
      }
      // Ap = A * p
      // std::fill_n(Ap, m_n, 0.);
      cuda_fill<<<blocks_per_grid, threads_per_block>>>(Ap, 0., m_n);
      cuda_dgemv<<<blocks_per_grid, threads_per_block>>>(m_m, m_n, 1., A_device, m_n, p, 1, 0., Ap, 1);
      cuda_ddot<<<blocks_per_grid, threads_per_block>>>(pAp, m_n, p, 1, Ap, 1);
      cudaDeviceSynchronize();

      auto denom = std::max(*pAp, *rsold * NEARZERO);
      auto alpha = *rsold / denom;

      // x = x + alpha * p
      cuda_daxpy<<<blocks_per_grid, threads_per_block>>>(m_n, alpha, p, 1, x_device, 1);

      // r = r - alpha * Ap
      cuda_daxpy<<<blocks_per_grid, threads_per_block>>>(m_n, -alpha, Ap, 1, r, 1);

      // rsnew = r' * r
      cuda_ddot<<<blocks_per_grid, threads_per_block>>>(rsnew, m_n, r, 1, r, 1);
      cudaDeviceSynchronize();

      if (std::sqrt(*rsnew) < m_tolerance)
          break;

      auto beta = *rsnew / *rsold;

      // tmp = r + beta * p
      cuda_copy<<<blocks_per_grid, threads_per_block>>>(tmp, r, m_n);
      cuda_daxpy<<<blocks_per_grid, threads_per_block>>>(m_n, beta, p, 1, tmp, 1);
      cuda_copy<<<blocks_per_grid, threads_per_block>>>(p, tmp, m_n);
      cudaDeviceSynchronize();

      *rsold = *rsnew;
  }
  // Copy x back to the cpu
  cudaMemcpy(x.data(), x_device, m_n * sizeof(double), cudaMemcpyDeviceToHost);

  if (SHOW_RESULT) {
      std::fill_n(r, m_n, 0.);
      cblas_dgemv(CblasRowMajor, CblasNoTrans, m_m, m_n, 1., m_A.data(), m_n,
                  x.data(), 1, 0., r, 1);
      cblas_daxpy(m_n, -1., m_b.data(), 1, r, 1);
      std::cout << "\n" << std::endl;
      auto res = std::sqrt(cblas_ddot(m_n, r, 1, r, 1)) /
                 std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
      auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
      std::cout << "[STEP " << k << "] residual = " << std::scientific
                << std::sqrt(*rsold) << ", ||x|| = " << nx
                << ", ||Ax - b||/||b|| = " << res << std::endl;
      std::cout << "\n \n" << std::endl;
  }

  // Clean up
  cudaFree(r);
  cudaFree(p);
  cudaFree(Ap);
  cudaFree(tmp);
  cudaFree(A_device);
  cudaFree(b_device);
  cudaFree(x_device);
  cudaFree(rsold);
  cudaFree(rsnew);
  cudaFree(pAp);
}


void CGSolver::read_matrix(const std::string & filename) {
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}


/*
Initialization of the source term b
*/
void Solver::init_source_term(double h) {
  m_b.resize(m_n);

  for (int i = 0; i < m_n; i++) {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) *
             std::sin(10. * M_PI * i * h);
  }
}
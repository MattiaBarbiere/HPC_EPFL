#include "cg.hh"
#include <algorithm>
#include <chrono>
#include <cblas.h>
#include <cmath>
#include <iostream>

using clk = std::chrono::high_resolution_clock;

const double NEARZERO = 1.0e-14; // Small value to avoid division by zero
const bool DEBUG = false;        // Debugging flag

/*
Sparse version of the Conjugate Gradient (CG) solver
*/
void CGSolverSparse::solve(std::vector<double> &x)
{
  // Start time to keep track of time spent by mat_vec
  double total_time_mat_vec = 0.0;

  std::vector<double> r(m_n), p(m_n), Ap(m_n), tmp(m_n);

  // Compute initial residual: r = b - A * x
  auto start_time = clk::now();
  m_A.mat_vec(x, Ap);
  total_time_mat_vec += std::chrono::duration_cast<std::chrono::duration<double>>(clk::now() - start_time).count();
  
  r = m_b;
  cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);

  // Initialize search direction: p = r
  p = r;

  // Compute initial residual norm: rsold = r' * r
  auto rsold = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

  // Iterate until convergence or max iterations
  int k = 0;
  for (; k < m_n; ++k)
  {
    // Compute Ap = A * p
    auto start_time = clk::now();
    m_A.mat_vec(p, Ap);
    total_time_mat_vec += std::chrono::duration_cast<std::chrono::duration<double>>(clk::now() - start_time).count();

    // Compute step size: alpha = rsold / (p' * Ap)
    auto alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1), rsold * NEARZERO);

    // Update solution: x = x + alpha * p
    cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

    // Update residual: r = r - alpha * Ap
    cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);

    // Compute new residual norm: rsnew = r' * r
    auto rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

    // Check for convergence
    if (std::sqrt(rsnew) < m_tolerance)
      break;

    // Update search direction: p = r + (rsnew / rsold) * p
    auto beta = rsnew / rsold;
    tmp = r;
    cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
    p = tmp;

    rsold = rsnew; // Update residual norm

    // if (DEBUG)
    // {
    //   std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << "\r" << std::flush;
    // }
  }

  if (DEBUG)
  {
    // Compute final residual and print debug info
    m_A.mat_vec(x, r);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    auto res = std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) / std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
    std::cout << "\t[STEP " << k << "] mat_vec time = " << total_time_mat_vec << " [s]" << std::endl;
  }
  
}

void CGSolverSparse::read_matrix(const std::string &filename)
{
  // Read matrix from file and set dimensions
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialize the source term vector b
*/
void Solver::init_source_term(double h)
{
  m_b.resize(m_n);

  // Compute b[i] based on the given formula
  for (int i = 0; i < m_n; i++)
  {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) * std::sin(10. * M_PI * i * h);
  }
}
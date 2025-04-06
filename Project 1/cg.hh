#include "matrix_coo.hh" // MatrixCOO class for sparse matrix representation
#include <cblas.h>       // BLAS library for efficient linear algebra operations
#include <string>        // For std::string
#include <vector>        // For std::vector

#ifndef __CG_HH__
#define __CG_HH__

/*v
 * oid CGSolver::solve(std::vector<double> & x)
void CGSolver::read_matrix(const std::string & filename)
void CGSolverSparse::solve(std::vector<double> & x)
void CGSolverSparse::read_matrix(const std::string & filename)
void MatrixCSR::mvm(const std::vector<double> & x, std::vector<double> & y)
const void MatrixCSR::loadMMMatrix(const std::string & filename) void
Solver::init_source_term(int n, double h)
*/
class Solver
{
public:
  virtual void read_matrix(const std::string &filename) = 0; // Load matrix from file
  void init_source_term(double h);                          // Initialize source term
  virtual void solve(std::vector<double> &x) = 0;           // Solve the system

  inline int m() const { return m_m; } // Get number of rows
  inline int n() const { return m_n; } // Get number of columns

  void tolerance(double tolerance) { m_tolerance = tolerance; } // Set convergence tolerance

protected:
  int m_m{0};                  // Number of rows
  int m_n{0};                  // Number of columns
  std::vector<double> m_b;     // Right-hand side vector
  double m_tolerance{1e-10};   // Convergence tolerance
};

class CGSolverSparse : public Solver
{
public:
  CGSolverSparse() = default;
  virtual void read_matrix(const std::string &filename); // Load sparse matrix
  virtual void solve(std::vector<double> &x);           // Solve using CG method

private:
  MatrixCOO m_A; // Sparse matrix in COO format
};

#endif /* __CG_HH__ */

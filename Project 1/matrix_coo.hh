#include <algorithm>
#include <string>
#include <vector>

#ifndef __MATRIX_COO_H_
#define __MATRIX_COO_H_

class MatrixCOO {
public:
  MatrixCOO() = default; // Default constructor

  inline int m() const { return m_m; } // Get number of rows
  inline int n() const { return m_n; } // Get number of columns

  inline int nz() const { return irn.size(); } // Get number of non-zero entries
  inline int is_sym() const { return m_is_sym; } // Check if the matrix is symmetric

  void read(const std::string &filename); // Read matrix from file

  // Perform matrix-vector multiplication: y = A * x
  void mat_vec(const std::vector<double> &x, std::vector<double> &y) {
    std::fill_n(y.begin(), y.size(), 0.); // Initialize result vector to zero

    for (size_t z = 0; z < irn.size(); ++z) {
      auto i = irn[z]; // Row index
      auto j = jcn[z]; // Column index
      auto a_ = a[z];  // Matrix value

      y[i] += a_ * x[j]; // Multiply and accumulate
      if (m_is_sym and (i != j)) {
        y[j] += a_ * x[i]; // Handle symmetric case
      }
    }
  }

  std::vector<int> irn; // Row indices of non-zero entries
  std::vector<int> jcn; // Column indices of non-zero entries
  std::vector<double> a; // Values of non-zero entries

private:
  int m_m{0}; // Number of rows
  int m_n{0}; // Number of columns
  bool m_is_sym{false}; // Symmetry flag
};

#endif // __MATRIX_COO_H_
#include <algorithm>
#include <string>
#include <vector>
#include <mpi.h>
#include <iostream>
#include <cassert>

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
    double t0 = MPI_Wtime();

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Amount of non zeros
    size_t non_zeros = irn.size();

    // Compute the numbers of rows each process should get
    size_t non_zeros_per_pro = non_zeros / size;
    size_t remainder = non_zeros % size;

    // Each rank gets rows_per_pro, and the first 'remainder' ranks get one more
    size_t start = rank * non_zeros_per_pro + std::min(static_cast<size_t>(rank), remainder);
    size_t end = start + non_zeros_per_pro + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Test that the above logic is correct by printing the start and end rows for each rank
    // print the size
    // std::cout << "Rank " << rank << " of " << size << std::endl;
    // std::cout << "Rank " << rank << ": start = " << start << ", end = " << end << std::endl;

    
    std::fill_n(y.begin(), y.size(), 0.); // Initialize result vector to zero
    std::vector<double> y_local(y.size(), 0.0);

    for (size_t z = start; z < end; ++z) {
      auto i = irn[z]; // Row index
      auto j = jcn[z]; // Column index
      auto a_ = a[z];  // Matrix value

      // print the values
      // std::cout << "Rank " << rank << ": i = " << i << ", j = " << j << ", a = " << a_ << std::endl;
      // assert(i >= 0 && static_cast<size_t>(i) < y.size());
      // assert(j >= 0 && static_cast<size_t>(j) < x.size());
      y_local[i] += a_ * x[j]; // Multiply and accumulate
      if (m_is_sym and (i != j)) {
        y_local[j] += a_ * x[i]; // Handle symmetric case
      }
      // print the values
      // std::cout << "Rank " << rank << ": y[" << i << "] = " << y[i] << ", y[" << j << "] = " << y[j] << std::endl;
    }
    double t1 = MPI_Wtime();
    double dt = t1 - t0;
    std::cout << "Rank " << rank << " time in mat_vec: " << dt << " s" << std::endl;
    MPI_Allreduce(y_local.data(), y.data(), y.size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


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
#include "matrix_coo.hh"
#include <mpi.h>
#include <vector>
#include <iostream>

//void mpi_mat_vec(const MatrixCOO &mat, const std::vector<double> &x, std::vector<double> &y) {
void mpi_mat_vec(){
    // Initialize MPI
    MPI_Init(nullptr, nullptr);

    // Get the rank and size of the MPI communicator
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine the number of rows in the matrix
    // int m = mat.m();
    // int n = mat.n();
    int m = 100; // Placeholder for matrix rows
    int n = 100; // Placeholder for matrix columns

    // The local result for each process
    std::vector<double> y_local(m, 0.0);

    // Compute the numbers of rows each process should get
    int rows_per_pro = m / size;
    int remainder = m % size;

    // Each rank gets rows_per_pro, and the first 'remainder' ranks get one more
    int row_start = rank * rows_per_pro + std::min(rank, remainder);
    int row_end = row_start + rows_per_pro + (rank < remainder ? 1 : 0);

    

    // Print the size of the processor grid
    std::cout << "Rank " << rank << " of " << size << std::endl;


    // Test that the above logic is correct by printing the start and end rows for each rank
    std::cout << "Rank " << rank << ": row_start = " << row_start << ", row_end = " << row_end << std::endl;

    MPI_Finalize();
}

int main(int argc, char *argv[]) {

    // Perform MPI matrix-vector multiplication
    mpi_mat_vec();

    return 0;
}
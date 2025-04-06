#include "matrix_coo.hh"
extern "C" {
#include "mmio.h" // Matrix Market I/O functions
}

void MatrixCOO::read(const std::string &fn) {
  int nz; // Number of non-zero entries
  int ret_code;
  MM_typecode matcode; // Matrix Market type code
  FILE *f;

  // Open the file
  if ((f = fopen(fn.c_str(), "r")) == NULL) {
    printf("Could not open matrix");
    exit(1);
  }

  // Read the Matrix Market banner
  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  // Ensure the matrix is sparse and in coordinate format
  if (not(mm_is_matrix(matcode) and mm_is_coordinate(matcode))) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  // Read matrix dimensions and number of non-zero entries
  if ((ret_code = mm_read_mtx_crd_size(f, &m_m, &m_n, &nz)) != 0) {
    exit(1);
  }

  // Allocate memory for COO representation
  irn.resize(nz); // Row indices
  jcn.resize(nz); // Column indices
  a.resize(nz);   // Values

  /*  NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
  m_is_sym = mm_is_symmetric(matcode);

  // Read non-zero entries
  for (int i = 0; i < nz; i++) {
    int I, J;
    double a_;

    fscanf(f, "%d %d %lg\n", &I, &J, &a_); // Read row, column, and value
    I--; // Convert 1-based to 0-based indexing
    J--;

    irn[i] = I; // Store row index
    jcn[i] = J; // Store column index
    a[i] = a_;  // Store value
  }

  // Close the file
  if (f != stdin) {
    fclose(f);
  }
}

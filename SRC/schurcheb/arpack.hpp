#ifndef CHEBEIG_ARPACK
#define CHEBEIG_ARPACK

/**
 * @file arpack.hpp
 * @brief arpack interface
 */

#include <iostream>

#include "../utils/protos.hpp"
#include "../vectors/vector.hpp"
#include "../utils/mmio.hpp"
#include "../matrices/matrix.hpp"
#include "../matrices/arnoldimatrix.hpp"
#include "../matrices/csr_matrix.hpp"
#include "../matrices/parallel_csr_matrix.hpp"
#include "../matrices/coo_matrix.hpp"
#include "../matrices/dense_matrix.hpp"
#include "../matrices/matrixops.hpp"
#include "dsolver.hpp"

using namespace std;
namespace schurcheb
{

   /**
    * Standard eigenvalue solver with PARPACK.
    * A Matvec is required for A, SetupVectorPtrStr is also required for A.
    */
   template <class VectorType, class MatrixType>
   int ArpackArnoldi( MatrixType &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog);

   /** Generalized eigenvalue solver with PARPACK.
    * MatVec of A, M, and Minv are required.for now we only need one option,
    * if we can compute M = LL^T and OP = inv(L)*A*inv(L^T), we should directly use the previous option.
    */
   template <class VectorType, class MatrixType, class SolverType>
   int ArpackArnoldi( MatrixType &A, MatrixType &M, SolverType &Minv, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, int &nsvs, double &tsvs, parallel_log &parlog);

   /** Generalized eigenvalue solver with PARPACK.
    * MatVec of A, M, and Ainv are required.for now we only need one option,
    * shift and invert mode with Ainv.
    */
   template <class VectorType, class MatrixType, class SolverType>
   int ArpackArnoldi_inv( MatrixType &A, MatrixType &M, SolverType &Ainv, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, int &nsvs, double &tsvs, parallel_log &parlog);
}
#endif

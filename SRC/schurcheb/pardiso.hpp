/* NOTE: ONLY FOR SYMMETRIC MATRIX */

#ifndef CHEBEIG_PARDISO
#define CHEBEIG_PARDISO

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

using namespace std;

namespace schurcheb
{

#ifdef SCHURCHEB_MKL

#include "mkl.h"
#include "mkl_cluster_sparse_solver.h"

   /* combine local matrix into a larger matrix, only keep the upper part */
   int ParallelCsrMatrixGlobalRowU(matrix_csr_par_double &A, vector_int &i_vec, vector_int &j_vec, vector_seq_double &a_vec);

   typedef struct PARDISOClass
   {
    
      vector_long _pt; /* memory pointer */
      int _maxfct;
      int _mnum;
      int _mtype;
      int _fcomm;
      vector_int _iparam;
      int _msglvl;
      int _nrhs;
      int _phase;
      
      int _n;
      
      /* upper tri part */
      vector_int _i_vec;
      vector_int _j_vec;
      vector_seq_double _a_vec;
      vector_seq_double _x_vec;
      
      int _print_level;
      
   }solver_pardiso;

   int PARDISOInit(solver_pardiso &pardiso, matrix_csr_par_double &Apar);

   int PARDISOSolve(solver_pardiso &pardiso, double *x);
   int PARDISOLSolve(solver_pardiso &pardiso, double *x);
   int PARDISOUSolve(solver_pardiso &pardiso, double *x);


   int PARDISOFree(solver_pardiso &pardiso);

#else

   int PARDISOInit(void *dummysolver, matrix_csr_par_double &Apar);

   int PARDISOSolve(void *dummysolver, double *x);

   int PARDISOFree(void *dummysolver);

#endif
}

#endif

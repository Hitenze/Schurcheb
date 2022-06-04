#ifndef CHEBEIG_SUPERLU
#define CHEBEIG_SUPERLU

/**
 * @file superlu.hpp
 * @brief SuperLU_dist interface
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

using namespace std;

namespace schurcheb
{

#ifdef SCHURCHEB_SUPERLU


#include <math.h>
#include "superlu_ddefs.h"


   /* combine local matrix into a larger matrix */
   int ParallelCsrMatrixGlobalRow(matrix_csr_par_double &A, matrix_csr_double &B);

   /* convert to SuperMatrix */
   SuperMatrix* ParallelCsrMatrix2SuperLU( matrix_csr_par_double &Apar );

   /* assign 2D MPI grid */
   unsigned int sqrti( const unsigned int & a );

   typedef struct SuperLUClass
   {
      int _n_global;
      int _n_local;
      SuperMatrix *_A;
      superlu_dist_options_t _options;
      gridinfo_t _grid;
      dScalePermstruct_t _SPstruct;
      dLUstruct_t _LUstruct;
      int _nrhs;
      dSOLVEstruct_t *_SOLVEstruct;
      double   *_berr;
      SuperLUStat_t _stat;
      
      bool _first_solve;
      
      int _print_level;
      
   }solver_superlu;

   int SuperLUInit(solver_superlu &superlu, matrix_csr_par_double &Apar);

   int SuperLUSolve(solver_superlu &superlu, double *x);

   int SuperLUFree(solver_superlu &superlu);

#else

   int SuperLUInit(void *dummysolver, matrix_csr_par_double &Apar);

   int SuperLUSolve(void *dummysolver, double *x);

   int SuperLUFree(void *dummysolver);

#endif

}


#endif

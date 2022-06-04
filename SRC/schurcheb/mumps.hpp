#ifndef CHEBEIG_MUMPS
#define CHEBEIG_MUMPS

/**
 * @file mumps.hpp
 * @brief MUMPS interface
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

#ifdef SCHURCHEB_MUMPS


#include "dmumps_c.h"


   /* combine local matrix into a larger matrix */
   int ParallelCsrMatrixGlobalRow(matrix_csr_par_double &A, vector_int &i_vec, vector_int &j_vec, vector_seq_double &a_vec);

   /* combine local matrix into a larger matrix */
   int CsrMatrixGlobalRow(matrix_csr_double &A, vector_int &i_vec, vector_int &j_vec, vector_seq_double &a_vec);


   /* combine local matrix into a larger matrix */
   int CsrMatrixGlobalRow(matrix_csr_double &A, vector_int &i_vec, vector_int &j_vec, vector_seq_double &a_vec, int shift);


   typedef struct MUMPSClass
   {
      DMUMPS_STRUC_C _id;
      vector_int _i_vec;
      vector_int _j_vec;
      vector_seq_double _a_vec;
      MUMPS_INT _nloc_rhs;
      MUMPS_INT _lrhs_loc;
      MUMPS_INT *_irhs_loc;
      
      vector_seq_double _sol_vec;
      vector_int _isol_vec;
      
      MPI_Comm _comm;
      int _np;
      int _myid;
      
      bool _ready;
      
      vector_int _nsends_vec;
      vector_int _senddisps_vec;
      vector_int _nrecvs_vec;
      vector_int _recvdisps_vec;
      vector_int _send_idx;
      vector_int _recv_idx;
      vector_seq_double _recv_buffer;
      vector_seq_double _send_buffer;
      
      int _loc_size;
      vector_int _loc_send_idx;
      vector_int _loc_recv_idx;
      
      int _print_level;
      
   }solver_mumps;

   int MUMPSInit(solver_mumps &mumps, matrix_csr_par_double &Apar);

   int MUMPSSolve(solver_mumps &mumps, double *x);

   int MUMPSInitSeq(solver_mumps &mumps, matrix_csr_double &Amat);

   int MUMPSSolveSeq(solver_mumps &mumps, double *x);

   int MUMPSFree(solver_mumps &mumps);

   int MUMPSSolComm(solver_mumps &mumps, double *x);

#else

   int MUMPSInit(void *dummysolver, matrix_csr_par_double &Apar);

   int MUMPSSolve(void *dummysolver, double *x);

   int MUMPSFree(void *dummysolver);

#endif
}


#endif

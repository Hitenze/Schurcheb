#include "pardiso.hpp"

namespace schurcheb
{

#ifdef SCHURCHEB_MKL

   int ParallelCsrMatrixGlobalRowU(matrix_csr_par_double &A, vector_int &i_vec, vector_int &j_vec, vector_seq_double &a_vec)
   {
      matrix_csr_double &A_diag = A.GetDiagMat();
      matrix_csr_double &A_offd = A.GetOffdMat();
      vector_SCHURCHEB_long &A_offd_map = A.GetOffdMap();
      
      int nrB = A.GetNumRowsLocal();
      int rsB = A.GetRowStartGlobal();
      int csB = A.GetColStartGlobal();
      
      int row, col;
      int idx = 0;
      
      for(int i = 0 ; i < nrB ; i ++)
      {
         int i1 = A_diag.GetI()[i];
         int i2 = A_diag.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            /* add one extra shift */
            row = rsB + i;
            col = csB + A_diag.GetJ()[j];
            if( row <= col)
            {
               idx++;
            }
         }
         
         i1 = A_offd.GetI()[i];
         i2 = A_offd.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            row = rsB + i;
            col = A_offd_map[A_offd.GetJ()[j]];
            if( row <= col)
            {
               idx++;
            }
         }
      }
      
      i_vec.Setup(nrB+1);
      j_vec.Setup(idx);
      a_vec.Setup(idx);
      
      idx = 0;
      for(int i = 0 ; i < nrB ; i ++)
      {
         i_vec[i] = idx;
         int i1 = A_diag.GetI()[i];
         int i2 = A_diag.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            /* add one extra shift */
            row = rsB + i;
            col = csB + A_diag.GetJ()[j];
            if( row <= col)
            {
               j_vec[idx] = col;
               a_vec[idx++] = A_diag.GetData()[j];
            }
         }
         
         i1 = A_offd.GetI()[i];
         i2 = A_offd.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            row = rsB + i;
            col = A_offd_map[A_offd.GetJ()[j]];
            if( row <= col)
            {
               j_vec[idx] = col;
               a_vec[idx++] = A_offd.GetData()[j];
            }
         }
      }
      i_vec[nrB] = idx;
      
      return SCHURCHEB_SUCCESS;
      
   }

   int PARDISOInit(solver_pardiso &pardiso, matrix_csr_par_double &Apar)
   {
      /* get MPI info */
      int np, myid;
      MPI_Comm comm;
      Apar.GetMpiInfo( np, myid, comm);
      
      pardiso._fcomm = MPI_Comm_c2f(comm);
      
      /* Create pardiso matrix info */
      ParallelCsrMatrixGlobalRowU( Apar, pardiso._i_vec, pardiso._j_vec, pardiso._a_vec);
      
      pardiso._n = Apar.GetNumRowsGlobal();
      pardiso._pt.Setup(64, true);
      pardiso._maxfct = 1;
      pardiso._mnum = 1; /* by default those two options are ignored */
      pardiso._mtype = 2; /* SPD */
      pardiso._nrhs = 1;
      pardiso._iparam.Setup(64, true);
      pardiso._x_vec.Setup(Apar.GetNumRowsLocal(), true);
      
      pardiso._msglvl = 0; /* no output */
      
      MKL_INT err = 0;
      
      /* https://www.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-fortran/top/sparse-solver-routines/parallel-direct-sp-solver-for-clusters-iface/cluster-sparse-solver.html */
      pardiso._iparam[0] = 1; /* do not use default */
      pardiso._iparam[1] = 0; /* 0: MD ordering 2: Metis ordering */
      pardiso._iparam[5] = 1; /* write solution into rhs */
      pardiso._iparam[7] = 0; /* max num of iterative refinement */
      pardiso._iparam[9] = 13; /* default for mat type 11 */
      pardiso._iparam[10] = 0; /* disable scaling for SPD */
      pardiso._iparam[12] = 1; /* Enable matching */
      pardiso._iparam[34] = 1; /* C style zero-based */
      pardiso._iparam[39] = 2; /* distributed sol and rhs */
      pardiso._iparam[40] = Apar.GetRowStartGlobal();
      pardiso._iparam[41] = Apar.GetRowStartGlobal() + Apar.GetNumRowsLocal() - 1;
      
      /* setup phase */
      pardiso._phase = 12;
      pardiso._msglvl = 1;
      cluster_sparse_solver(pardiso._pt.GetData(), 
                              &pardiso._maxfct, 
                              &pardiso._mnum, 
                              &pardiso._mtype, 
                              &pardiso._phase, 
                              &pardiso._n, 
                              pardiso._a_vec.GetData(), 
                              pardiso._i_vec.GetData(), 
                              pardiso._j_vec.GetData(),
                              NULL, 
                              &pardiso._nrhs, 
                              pardiso._iparam.GetData(), 
                              &pardiso._msglvl, 
                              NULL, 
                              NULL, 
                              &pardiso._fcomm, 
                              &err);
      pardiso._msglvl = 0;

      return SCHURCHEB_SUCCESS;
   }

   int PARDISOSolve(solver_pardiso &pardiso, double *x)
   {
      int err;
      
      /* solve phase */
      pardiso._phase = 33;
      cluster_sparse_solver(pardiso._pt.GetData(), 
                              &pardiso._maxfct, 
                              &pardiso._mnum, 
                              &pardiso._mtype, 
                              &pardiso._phase, 
                              &pardiso._n, 
                              pardiso._a_vec.GetData(), 
                              pardiso._i_vec.GetData(), 
                              pardiso._j_vec.GetData(),
                              NULL, 
                              &pardiso._nrhs, 
                              pardiso._iparam.GetData(), 
                              &pardiso._msglvl, 
                              x, 
                              pardiso._x_vec.GetData(), 
                              &pardiso._fcomm, 
                              &err);
      
      return SCHURCHEB_SUCCESS;
   }

   int PARDISOLSolve(solver_pardiso &pardiso, double *x)
   {
      int err;
      
      /* solve phase */
      pardiso._phase = 331;
      cluster_sparse_solver(pardiso._pt.GetData(), 
                              &pardiso._maxfct, 
                              &pardiso._mnum, 
                              &pardiso._mtype, 
                              &pardiso._phase, 
                              &pardiso._n, 
                              pardiso._a_vec.GetData(), 
                              pardiso._i_vec.GetData(), 
                              pardiso._j_vec.GetData(),
                              NULL, 
                              &pardiso._nrhs, 
                              pardiso._iparam.GetData(), 
                              &pardiso._msglvl, 
                              x, 
                              pardiso._x_vec.GetData(), 
                              &pardiso._fcomm, 
                              &err);
      
      return SCHURCHEB_SUCCESS;
   }

   int PARDISOUSolve(solver_pardiso &pardiso, double *x)
   {
      int err;
      
      /* solve phase */
      pardiso._phase = 333;
      cluster_sparse_solver(pardiso._pt.GetData(), 
                              &pardiso._maxfct, 
                              &pardiso._mnum, 
                              &pardiso._mtype, 
                              &pardiso._phase, 
                              &pardiso._n, 
                              pardiso._a_vec.GetData(), 
                              pardiso._i_vec.GetData(), 
                              pardiso._j_vec.GetData(),
                              NULL, 
                              &pardiso._nrhs, 
                              pardiso._iparam.GetData(), 
                              &pardiso._msglvl, 
                              x, 
                              pardiso._x_vec.GetData(), 
                              &pardiso._fcomm, 
                              &err);
      
      return SCHURCHEB_SUCCESS;
   }

   int PARDISOFree(solver_pardiso &pardiso)
   {
      int err;
      
      /* solve phase */
      pardiso._phase = -1;
      cluster_sparse_solver(pardiso._pt.GetData(), 
                              &pardiso._maxfct, 
                              &pardiso._mnum, 
                              &pardiso._mtype, 
                              &pardiso._phase, 
                              &pardiso._n, 
                              pardiso._a_vec.GetData(), 
                              pardiso._i_vec.GetData(), 
                              pardiso._j_vec.GetData(),
                              NULL, 
                              &pardiso._nrhs, 
                              pardiso._iparam.GetData(), 
                              &pardiso._msglvl, 
                              NULL, 
                              NULL, 
                              &pardiso._fcomm, 
                              &err);
      
      pardiso._i_vec.Clear();
      pardiso._j_vec.Clear();
      pardiso._a_vec.Clear();
      pardiso._x_vec.Clear();
      pardiso._pt.Clear();
      pardiso._iparam.Clear();
      
      return SCHURCHEB_SUCCESS;
   }


#else

   int PARDISOInit(void *dummysolver, matrix_csr_par_double &Apar)
   {
      SCHURCHEB_WARNING("No Dsitributed Pardiso Solver.");
      return SCHURCHEB_SUCCESS;
   }

   int PARDISOSolve(void *dummysolver, double *x)
   {
      SCHURCHEB_WARNING("No Dsitributed Pardiso Solver.");
      return SCHURCHEB_SUCCESS;
   }

   int PARDISOFree(void *dummysolver)
   {
      SCHURCHEB_WARNING("No Dsitributed Pardiso Solver.");
      return SCHURCHEB_SUCCESS;
   }

#endif
}

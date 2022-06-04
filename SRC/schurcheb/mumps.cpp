#include "mumps.hpp"

/**
 * @file mumps.cpp
 * @brief MUMPS interface
 */

namespace schurcheb
{

#ifdef SCHURCHEB_MUMPS

   /* combine local matrix into a larger matrix */
   int ParallelCsrMatrixGlobalRow(matrix_csr_par_double &A, vector_int &i_vec, vector_int &j_vec, vector_seq_double &a_vec)
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
            row = rsB + i + 1;
            col = csB + A_diag.GetJ()[j] + 1;
            if( row <= col)
            {
               idx++;
            }
         }
         
         i1 = A_offd.GetI()[i];
         i2 = A_offd.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            row = rsB + i + 1;
            col = A_offd_map[A_offd.GetJ()[j]]+1;
            if( row <= col)
            {
               idx++;
            }
         }
      }
      
      i_vec.Setup(idx);
      j_vec.Setup(idx);
      a_vec.Setup(idx);
      
      idx = 0;
      for(int i = 0 ; i < nrB ; i ++)
      {
         int i1 = A_diag.GetI()[i];
         int i2 = A_diag.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            /* add one extra shift */
            row = rsB + i + 1;
            col = csB + A_diag.GetJ()[j] + 1;
            if( row <= col)
            {
               i_vec[idx] = row;
               j_vec[idx] = col;
               a_vec[idx++] = A_diag.GetData()[j];
            }
         }
         
         i1 = A_offd.GetI()[i];
         i2 = A_offd.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            row = rsB + i + 1;
            col = A_offd_map[A_offd.GetJ()[j]]+1;
            if( row <= col)
            {
               i_vec[idx] = row;
               j_vec[idx] = col;
               a_vec[idx++] = A_offd.GetData()[j];
            }
         }
      }
      
      return SCHURCHEB_SUCCESS;
      
   }

   /* combine local matrix into a larger matrix */
   int CsrMatrixGlobalRow(matrix_csr_double &A, vector_int &i_vec, vector_int &j_vec, vector_seq_double &a_vec)
   {
      return CsrMatrixGlobalRow(A, i_vec, j_vec, a_vec, 1);
   }

   /* combine local matrix into a larger matrix */
   int CsrMatrixGlobalRow(matrix_csr_double &A, vector_int &i_vec, vector_int &j_vec, vector_seq_double &a_vec, int shift)
   {
      int n = A.GetNumRowsLocal();
      
      int row, col;
      int idx = 0;
      
      for(int i = 0 ; i < n ; i ++)
      {
         int i1 = A.GetI()[i];
         int i2 = A.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            /* add one extra shift */
            row = i + shift;
            col = A.GetJ()[j] + shift;
            if( row <= col)
            {
               idx++;
            }
         }
      }
      
      i_vec.Setup(idx);
      j_vec.Setup(idx);
      a_vec.Setup(idx);
      
      idx = 0;
      for(int i = 0 ; i < n ; i ++)
      {
         int i1 = A.GetI()[i];
         int i2 = A.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            /* add one extra shift */
            row = i + shift;
            col = A.GetJ()[j] + shift;
            if( row <= col)
            {
               i_vec[idx] = row;
               j_vec[idx] = col;
               a_vec[idx++] = A.GetData()[j];
            }
         }
      }
      
      return SCHURCHEB_SUCCESS;
      
   }


   int MUMPSInit(solver_mumps &mumps, matrix_csr_par_double &Apar)
   {
      /* get MPI info */
      int np, myid;
      MPI_Comm comm;
      Apar.GetMpiInfo( np, myid, comm);
      
      mumps._comm = comm;
      mumps._np = np;
      mumps._myid = myid;
      
      mumps._irhs_loc = NULL;
      
      /* Create mumps matrix info */
      ParallelCsrMatrixGlobalRow( Apar, mumps._i_vec, mumps._j_vec, mumps._a_vec);
      
      MUMPS_INT n = Apar.GetNumRowsGlobal();
      MUMPS_INT8 nnz_loc = mumps._i_vec.GetLengthLocal();
     
      mumps._nloc_rhs = Apar.GetNumRowsLocal();
      mumps._lrhs_loc = mumps._nloc_rhs;
      SCHURCHEB_MALLOC( mumps._irhs_loc, mumps._nloc_rhs, kMemoryHost, MUMPS_INT);
      
      int rcA = Apar.GetColStartGlobal();
      for(MUMPS_INT i = 0 ; i < mumps._nloc_rhs ; i ++)
      {
         mumps._irhs_loc[i] = rcA + i + 1;
      }
      
      /* INIT */
      mumps._id.comm_fortran = MPI_Comm_c2f(comm);
      mumps._id.par=1; /* rank 0 also working */
      mumps._id.sym = 1; /* symmetric */
      mumps._id.job = -1;
      dmumps_c(&mumps._id);
      
      /* Define the distributed problem */
      mumps._id.n = n;
      mumps._id.nnz_loc = nnz_loc; 
      mumps._id.irn_loc = mumps._i_vec.GetData(); 
      mumps._id.jcn_loc = mumps._j_vec.GetData();
      mumps._id.a_loc = mumps._a_vec.GetData(); 
       
      mumps._id.nloc_rhs = mumps._nloc_rhs;
      mumps._id.lrhs_loc = mumps._lrhs_loc;
      mumps._id.irhs_loc = mumps._irhs_loc;
      
      /* outputs */
      if (mumps._print_level > 2)
      {
         mumps._id.icntl[0]=6; mumps._id.icntl[1]=6; mumps._id.icntl[2]=6; mumps._id.icntl[3]=6; 
      }
      else if(mumps._print_level > 0)
      {
         mumps._id.icntl[0]=-1; mumps._id.icntl[1]=-1; mumps._id.icntl[2]=6; mumps._id.icntl[3]=6; 
      }
      else
      {
         mumps._id.icntl[0]=-1; mumps._id.icntl[1]=-1; mumps._id.icntl[2]=-1; mumps._id.icntl[3]=0; 
      }
      
      mumps._id.icntl[4]=0; /* COO format */
      mumps._id.icntl[9]=0; /* iterative refinement */
      mumps._id.icntl[12]=0; /* enable scalapack */
      mumps._id.icntl[17]=3; /* distribute matrix */
      mumps._id.icntl[19]=10; /* distribute rhs */
      mumps._id.icntl[20]=1; /* distribute sol */
      mumps._id.icntl[21]=0; /* not out of core factorization and solve */
      mumps._id.icntl[22]=0; /* Auto max size of working memory */
      mumps._id.icntl[27]=2; /* Parallel reordering */
      mumps._id.icntl[28]=2; /* Parmetis */

   mumps_ana_start:

      /* Call the MUMPS package analyse and factorization */
      mumps._id.job=1;
      dmumps_c(&mumps._id);
      
      if(mumps._id.infog[0] == -9)
      {
         mumps._id.icntl[13] = mumps._id.icntl[13] + 20;
         goto mumps_ana_start;
      }

   mumps_fct_start:

      /* Call the MUMPS package (analyse, factorization and solve). */
      mumps._id.job=2;
      dmumps_c(&mumps._id);
      if(mumps._id.infog[0] == -9)
      {
         mumps._id.icntl[13] = mumps._id.icntl[13] + 20;
         goto mumps_fct_start;
      }

      if(mumps._id.infog[0] == -40)
      {
         /* scalapack fail */
         mumps._id.icntl[12] = 1;
         goto mumps_ana_start;
      }

      mumps._id.lsol_loc = mumps._id.info[22];
      mumps._sol_vec.Setup(mumps._id.lsol_loc);
      mumps._isol_vec.Setup(mumps._id.lsol_loc);
      mumps._id.sol_loc = mumps._sol_vec.GetData();
      mumps._id.isol_loc = mumps._isol_vec.GetData();
      
      mumps._ready = false;
      
      return SCHURCHEB_SUCCESS;
   }

   int MUMPSSolve(solver_mumps &mumps, double *x)
   {
      //int info;
      
      mumps._id.rhs_loc = x;
      mumps._id.job=3;
      dmumps_c(&mumps._id);
      
      MUMPSSolComm(mumps, x);
      
      return SCHURCHEB_SUCCESS;
   }

   int MUMPSFree(solver_mumps &mumps)
   {
      
      mumps._id.job=-2;
      dmumps_c(&mumps._id);
      
      mumps._i_vec.Clear();
      mumps._j_vec.Clear();
      mumps._a_vec.Clear();
      mumps._sol_vec.Clear();
      mumps._isol_vec.Clear();
      
      SCHURCHEB_FREE( mumps._irhs_loc, kMemoryHost);
      
      return SCHURCHEB_SUCCESS;
   }

   int MUMPSSolComm(solver_mumps &mumps, double *x)
   {
      
      int np = mumps._np;
      int myid = mumps._myid;
      MPI_Comm comm = mumps._comm;
      
      int *isol_loc = mumps._id.isol_loc;
      
      if(!mumps._ready)
      {
         
         int lrhs_loc = mumps._id.lrhs_loc;
         int lsol_loc = mumps._id.lsol_loc;
         
         mumps._nsends_vec.Setup(np);
         
         mumps._nsends_vec.Setup(np, true);
         mumps._senddisps_vec.Setup(np+1, false);
         mumps._nrecvs_vec.Setup(np, false);
         mumps._recvdisps_vec.Setup(np+1, false);
         
         vector_int nsizes, nstarts;
         nsizes.Setup(np);
         nstarts.Setup(np+1);
         SchurchebMpiAllgather( &lrhs_loc, 1, nsizes.GetData(), comm);
         nstarts[0] = 0;
         for(int i = 0 ; i < np ; i ++)
         {
            nstarts[i+1] = nstarts[i] + nsizes[i];
         }
         
         /* compute send size */
         mumps._loc_size = 0;
         for(int i = 0 ; i < lsol_loc ; i ++)
         {
            int idx = isol_loc[i] - 1;
            int myidi;
            if( nstarts.BinarySearch(idx, myidi, true) < 0)
            {
               myidi--;
            }
            
            if(myid == myidi)
            {
               mumps._loc_size++;
            }
            else
            {
               mumps._nsends_vec[myidi]++;
            }
         }
         
         /* compute recv size */
         MPI_Alltoall(mumps._nsends_vec.GetData(), 1, MPI_INT, mumps._nrecvs_vec.GetData(), 1, MPI_INT, comm);
         
         /* get send/recv displacement */
         mumps._senddisps_vec[0] = 0;
         mumps._recvdisps_vec[0] = 0;
         
         for(int i = 0 ; i < np ; i ++)
         {
            mumps._senddisps_vec[i+1] = mumps._senddisps_vec[i] + mumps._nsends_vec[i];
            mumps._recvdisps_vec[i+1] = mumps._recvdisps_vec[i] + mumps._nrecvs_vec[i];
         }
         
         /* allocate memory */
         vector_int send_idx;
         mumps._loc_send_idx.Setup(mumps._loc_size);
         mumps._loc_recv_idx.Setup(mumps._loc_size);
         send_idx.Setup(mumps._senddisps_vec[np]);
         mumps._send_idx.Setup(mumps._senddisps_vec[np]);
         mumps._recv_idx.Setup(mumps._recvdisps_vec[np]);
         mumps._send_buffer.Setup(mumps._senddisps_vec[np]);
         mumps._recv_buffer.Setup(mumps._recvdisps_vec[np]);
         
         mumps._nsends_vec.Fill(0);
         mumps._loc_size = 0;
         for(int i = 0 ; i < lsol_loc ; i ++)
         {
            int idx = isol_loc[i] - 1;
            int myidi;
            if( nstarts.BinarySearch(idx, myidi, true) < 0)
            {
               myidi--;
            }
            
            if(myid == myidi)
            {
               mumps._loc_send_idx[mumps._loc_size] = i;
               mumps._loc_recv_idx[mumps._loc_size++] = idx - nstarts[myid];
            }
            else
            {
               mumps._send_idx[mumps._senddisps_vec[myidi] + mumps._nsends_vec[myidi]] = i;
               send_idx[mumps._senddisps_vec[myidi] + mumps._nsends_vec[myidi]++] = idx - nstarts[myidi];
            }
         }
         
         MPI_Alltoallv(send_idx.GetData(), mumps._nsends_vec.GetData(), mumps._senddisps_vec.GetData(), MPI_INT,
                    mumps._recv_idx.GetData(), mumps._nrecvs_vec.GetData(), mumps._recvdisps_vec.GetData(), MPI_INT, comm);
         
         send_idx.Clear();
         
         mumps._ready = true;
         mumps._id.icntl[0]=-1; mumps._id.icntl[1]=-1; mumps._id.icntl[2]=-1; mumps._id.icntl[3]=-1; 
         
         vector_seq_double solptr;
         solptr.SetupPtr( mumps._id.sol_loc, mumps._id.lsol_loc, kMemoryHost);
         
      }

      /* copy data to send buffer */
      vector_seq_double xptr, solptr;
      xptr.SetupPtr( x, mumps._id.lrhs_loc, kMemoryHost);
      solptr.SetupPtr( mumps._id.sol_loc, mumps._id.lsol_loc, kMemoryHost);
      
      /* v_out := v_in(map) */
      mumps._send_idx.GatherPerm(solptr, mumps._send_buffer);

      MPI_Alltoallv( mumps._send_buffer.GetData(), mumps._nsends_vec.GetData(), mumps._senddisps_vec.GetData(), MPI_DOUBLE,
                 mumps._recv_buffer.GetData(), mumps._nrecvs_vec.GetData(), mumps._recvdisps_vec.GetData(), MPI_DOUBLE, comm);
      
      /* v_out(map) := v_in */
      mumps._recv_idx.ScatterRperm(mumps._recv_buffer, xptr);
      
      for(int i = 0 ; i < mumps._loc_size ; i ++)
      {
         xptr[mumps._loc_recv_idx[i]] = solptr[mumps._loc_send_idx[i]];
      }
      
      return SCHURCHEB_SUCCESS;
      
   }


   int MUMPSInitSeq(solver_mumps &mumps, matrix_csr_double &Amat)
   {
      /* get MPI info */
      MPI_Comm comm = *(parallel_log::_lcomm); // point to the local communicator
      
      mumps._comm = comm;
      mumps._np = 1;
      mumps._myid = 0;
      
      mumps._irhs_loc = NULL;
      
      /* Create mumps matrix info */
      CsrMatrixGlobalRow(Amat, mumps._i_vec, mumps._j_vec, mumps._a_vec);
      
      MUMPS_INT n = Amat.GetNumRowsLocal();
      MUMPS_INT8 nnz = mumps._i_vec.GetLengthLocal();
      
      /* INIT */
      mumps._id.comm_fortran = MPI_Comm_c2f(comm);
      mumps._id.par=1; /* rank 0 also working */
      mumps._id.sym = 2; /* general symmetric */
      mumps._id.job = -1;
      dmumps_c(&mumps._id);
      
      /* Define the problem */
      mumps._id.n = n;
      mumps._id.nnz = nnz;
      mumps._id.irn = mumps._i_vec.GetData(); 
      mumps._id.jcn = mumps._j_vec.GetData();
      mumps._id.a = mumps._a_vec.GetData(); 
      
      //mumps._id.rhs = rhs; /* only if a global rhs and sol is need on rank 0 */
      
      /* outputs */
      if (mumps._print_level > 2)
      {
         mumps._id.icntl[0]=6; mumps._id.icntl[1]=6; mumps._id.icntl[2]=6; mumps._id.icntl[3]=6; 
      }
      else if(mumps._print_level > 1)
      {
         mumps._id.icntl[0]=-1; mumps._id.icntl[1]=-1; mumps._id.icntl[2]=6; mumps._id.icntl[3]=6; 
      }
      else
      {
         mumps._id.icntl[0]=-1; mumps._id.icntl[1]=-1; mumps._id.icntl[2]=-1; mumps._id.icntl[3]=0; 
      }
      
      mumps._id.icntl[4]=0; /* COO format */
      mumps._id.icntl[6]=5; /* METIS reordering */
      mumps._id.icntl[9]=0; /* iterative refinement */
      mumps._id.icntl[12]=1; /* disable scalapack */
      mumps._id.icntl[21]=0; /* not out of core factorization and solve */
      mumps._id.icntl[22]=0; /* Auto max size of working memory */
      mumps._id.icntl[27]=1; /* sequential reordering */

   mumps_ana_start:

      /* Call the MUMPS package analyse and factorization */
      mumps._id.job=1;
      dmumps_c(&mumps._id);
      
      if(mumps._id.infog[0] == -9)
      {
         mumps._id.icntl[13] = mumps._id.icntl[13] + 20;
         goto mumps_ana_start;
      }
      
   mumps_fct_start:

      /* Call the MUMPS package (analyse, factorization and solve). */
      mumps._id.job=2;
      dmumps_c(&mumps._id);
      if(mumps._id.infog[0] == -9)
      {
         mumps._id.icntl[13] = mumps._id.icntl[13] + 20;
         goto mumps_fct_start;
      }

      if(mumps._id.infog[0] == -40)
      {
         /* scalapack fail */
         mumps._id.icntl[12] = 1;
         goto mumps_ana_start;
      }

      mumps._id.lsol_loc = mumps._id.info[22];
      mumps._sol_vec.Setup(mumps._id.lsol_loc);
      mumps._isol_vec.Setup(mumps._id.lsol_loc);
      mumps._id.sol_loc = mumps._sol_vec.GetData();
      mumps._id.isol_loc = mumps._isol_vec.GetData();
      
      mumps._ready = false;
      
      return SCHURCHEB_SUCCESS;
   }

   int MUMPSSolveSeq(solver_mumps &mumps, double *x)
   {
      mumps._id.rhs = x;
      mumps._id.job=3;
      dmumps_c(&mumps._id);
      
      mumps._ready = true;
      mumps._id.icntl[0]=-1; mumps._id.icntl[1]=-1; mumps._id.icntl[2]=-1; mumps._id.icntl[3]=-1; 
      
      return SCHURCHEB_SUCCESS;
   }

#else

   int MUMPSInit(void *dummysolver, matrix_csr_par_double &Apar)
   {
      SCHURCHEB_WARNING("No MUMPS Solver.");
      return SCHURCHEB_SUCCESS;
   }

   int MUMPSSolve(void *dummysolver, double *x)
   {
      SCHURCHEB_WARNING("No MUMPS Solver.");
      return SCHURCHEB_SUCCESS;
   }

   int MUMPSFree(void *dummysolver)
   {
      SCHURCHEB_WARNING("No MUMPS Solver.");
      return SCHURCHEB_SUCCESS;
   }

#endif

}

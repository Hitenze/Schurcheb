#include "superlu.hpp"

/**
 * @file superlu.cpp
 * @brief SuperLU_dist interface
 */

namespace schurcheb
{

#ifdef SCHURCHEB_SUPERLU

   /* combine local matrix into a larger matrix */
   int ParallelCsrMatrixGlobalRow(matrix_csr_par_double &A, matrix_csr_double &B)
   {
      matrix_csr_double &A_diag = A.GetDiagMat();
      matrix_csr_double &A_offd = A.GetOffdMat();
      vector_SCHURCHEB_long &A_offd_map = A.GetOffdMap();
      
      int nnz_diag = A_diag.GetNumNonzeros();
      int nnz_offd = A_offd.GetNumNonzeros();
      
      int ncB = A.GetNumColsGlobal();
      int nrB = A.GetNumRowsLocal();
      int csB = A.GetColStartGlobal();
      int nnzB = nnz_diag + nnz_offd;
      
      B.Setup( nrB, ncB, nnzB, kMemoryHost, true);
      
      int idx = 0;
      for(int i = 0 ; i < nrB ; i ++)
      {
         B.GetI()[i] = idx;
         int i1 = A_diag.GetI()[i];
         int i2 = A_diag.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            B.GetJ()[idx] = csB + A_diag.GetJ()[j];
            B.GetData()[idx++] = A_diag.GetData()[j];
         }
         
         i1 = A_offd.GetI()[i];
         i2 = A_offd.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            B.GetJ()[idx] = A_offd_map[A_offd.GetJ()[j]];
            B.GetData()[idx++] = A_offd.GetData()[j];
         }
      }
      B.GetI()[nrB] = idx;
      B.SetNumNonzeros();
      
      return SCHURCHEB_SUCCESS;
      
   }

   /* convert to SuperMatrix */
   SuperMatrix* ParallelCsrMatrix2SuperLU( matrix_csr_par_double &Apar )
   {
      
      SuperMatrix * A = new SuperMatrix;
      A->Store = NULL;

      matrix_csr_double Aloc;
      ParallelCsrMatrixGlobalRow( Apar, Aloc);
      
      int m         = Apar.GetNumRowsGlobal();
      int n         = Apar.GetNumColsGlobal();
      int fst_row   = Apar.GetRowStartGlobal();
      int nnz_loc   = Aloc.GetNumNonzeros();
      int m_loc     = Aloc.GetNumRowsLocal();
      
      double * nzval  = NULL;
      int    * colind = NULL;
      int    * rowptr = NULL;

      SCHURCHEB_MALLOC( colind, nnz_loc, kMemoryHost, int);
      SCHURCHEB_MALLOC( nzval, nnz_loc, kMemoryHost, double);
      SCHURCHEB_MALLOC( rowptr, m_loc+1, kMemoryHost, int);
      
      SCHURCHEB_MEMCPY( colind, Aloc.GetJ(), nnz_loc, kMemoryHost, kMemoryHost, int);
      SCHURCHEB_MEMCPY( nzval, Aloc.GetData(), nnz_loc, kMemoryHost, kMemoryHost, double);
      SCHURCHEB_MEMCPY( rowptr, Aloc.GetI(), m_loc+1, kMemoryHost, kMemoryHost, int);

      Aloc.Clear();

      // Assign he matrix data to SuperLU's SuperMatrix structure
      dCreate_CompRowLoc_Matrix_dist(A, m, n, nnz_loc, m_loc, fst_row,
                                     nzval, colind, rowptr,
                                     SLU_NR_loc, SLU_D, SLU_GE);

      return A;

   }

   /* assign 2D MPI grid */
   unsigned int sqrti( const unsigned int & a )
   {
      unsigned int a_ = a;
      unsigned int rem = 0;
      unsigned int root = 0;
      unsigned short len   = sizeof(int); len <<= 2;
      unsigned short shift = (unsigned short)((len<<1) - 2);

      for (int i=0; i<len; i++)
      {
         root <<= 1;
         rem = ((rem << 2) + (a_ >> shift));
         a_ <<= 2;
         root ++;
         if (root <= rem)
         {
            rem -= root;
            root++;
         }
         else
         {
            root--;
         }
      }
      return (root >> 1);
   }

   int SuperLUInit(solver_superlu &superlu, matrix_csr_par_double &Apar)
   {
      /* get MPI info */
      int np, myid;
      MPI_Comm comm;
      Apar.GetMpiInfo( np, myid, comm);
      
      /* convert to SuperMatrix */
      superlu._A = ParallelCsrMatrix2SuperLU( Apar );
      superlu._n_global = Apar.GetNumRowsGlobal();
      superlu._n_local = Apar.GetNumRowsLocal();
      
      /* create option */
      set_default_options_dist(&superlu._options);
      if (myid == 0) 
      {
         print_sp_ienv_dist(&superlu._options);
         print_options_dist(&superlu._options);
      }
      
      /* create grid */
      int nprow, npcol;
      nprow = (int)sqrti((unsigned int)np);
      while (np % nprow != 0 && nprow > 0)
      {
         nprow--;
      }
      npcol = (int)(np / nprow);
      
      if(myid == 0)
      {
         cout<<"Grid: "<<nprow<<"*"<<npcol<<endl;
      }
      
      superlu_gridinit(comm, nprow, npcol, &superlu._grid);
      
      /* create permutation & LU structure */
      dScalePermstructInit( superlu._A->nrow, superlu._A->ncol, &superlu._SPstruct);
      dLUstructInit( superlu._A->ncol, &superlu._LUstruct);
      
      /* single right hand side */
      superlu._nrhs = 1;
      superlu._berr = NULL;
      SCHURCHEB_MALLOC( superlu._berr, superlu._nrhs, kMemoryHost, double);
      
      /* create solver */
      superlu._SOLVEstruct = new dSOLVEstruct_t;
      
      /* create stat */
      PStatInit(&superlu._stat);
      
      superlu._first_solve = true;
      
      return SCHURCHEB_SUCCESS;
   }

   int SuperLUSolve(solver_superlu &superlu, double *x)
   {
      int info;
      
      pdgssvx(&superlu._options, superlu._A, &superlu._SPstruct, 
               x, superlu._n_local, superlu._nrhs, &superlu._grid,
               &superlu._LUstruct, superlu._SOLVEstruct, superlu._berr, &superlu._stat, &info);
      
      /* we have done this once */
      if(superlu._first_solve)
      {
         superlu._options.Fact = FACTORED;
         superlu._first_solve = false;
         if(superlu._print_level > 0)
         {
           PStatPrint(&superlu._options, &superlu._stat, &superlu._grid);        /* Print the statistics. */
        }
      }
      
      return SCHURCHEB_SUCCESS;
   }

   int SuperLUFree(solver_superlu &superlu)
   {
      PStatFree(&superlu._stat);
      Destroy_CompRowLoc_Matrix_dist(superlu._A);
      dDestroy_LU(superlu._n_global, &superlu._grid, &superlu._LUstruct);
      dScalePermstructFree(&superlu._SPstruct);
      dLUstructFree(&superlu._LUstruct);
      SUPERLU_FREE(superlu._berr);
      superlu_gridexit(&superlu._grid);
      
      return SCHURCHEB_SUCCESS;
   }

#else

   int SuperLUInit(void *dummysolver, matrix_csr_par_double &Apar)
   {
      SCHURCHEB_WARNING("No SuperLU Solver.");
      return SCHURCHEB_SUCCESS;
   }

   int SuperLUSolve(void *dummysolver, double *x)
   {
      SCHURCHEB_WARNING("No SuperLU Solver.");
      return SCHURCHEB_SUCCESS;
   }

   int SuperLUFree(void *dummysolver)
   {
      SCHURCHEB_WARNING("No SuperLU Solver.");
      return SCHURCHEB_SUCCESS;
   }

#endif
}


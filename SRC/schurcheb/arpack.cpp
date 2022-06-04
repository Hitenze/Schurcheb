#include "arpack.hpp"

/**
 * @file arpack.cpp
 * @brief arpack interface
 */

namespace schurcheb
{

#ifdef SCHURCHEB_PARPACK

   template <class VectorType, class MatrixType>
   int ArpackArnoldi( MatrixType &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog)
   {
      
      /* DNAUPD */
      VectorType v, w;
      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);
      int n = v.GetLengthLocal();
      int n_global = v.GetLengthGlobal();
      
      int ido = 0; //must be 0 for the first call
      char bmat = 'I';
      
      int np, myid;
      MPI_Comm comm;
      parlog.GetMpiInfo(np, myid, comm);
      int fcomm = MPI_Comm_c2f(comm);
      
      nmvs = 0;
      tmvs = 0.0;
      
      vector_seq_double resid; // array of length n
      resid.Setup(n, true);
      
      if(n_global < nev + 2)
      {
         SCHURCHEB_ERROR("nev can't be larger than n-2.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      msteps = SchurchebMax( msteps, nev + 2);
      int ncv = SchurchebMin( msteps, n_global); // number of columns of the matrix V
      // 2 <= ncv - nev && ncv <= n
      
      if(V.GetNumRowsLocal() < n || V.GetNumColsLocal() < ncv)
      {
         V.Setup(n, ncv, true);
      }
      int ldv = V.GetLeadingDimension();
      
      vector_int iparam;
      iparam.Setup(11, true);
      
      iparam[0] = 1; // equv to restart with linear combination of approximateed Schur vectors
      iparam[2] = maxits; // max iter
      iparam[3] = 1;
      iparam[6] = 1; //mode, M is I
      
      vector_int ipntr;
      ipntr.Setup(14, true);
      
      vector_seq_double workd;
      workd.Setup(3*n, true);
      
      int lworkl;
      vector_seq_double workl;
      vector_seq_double workev; // working vector
      
      if(sym)
      {
         if(which[1] == 'R')
         {
            which[1] = 'A'; /* for sym only SA availiable */
         }
         lworkl = ncv*ncv +  8*ncv;
         workl.Setup(lworkl, true);
      }
      else
      {
         lworkl = 3*ncv*ncv +  6*ncv;
         workl.Setup(lworkl, true);
         workev.Setup(3*ncv);
      }
      
      int info = 0;
      
      /* DNEUPD */
      
      int rvec = 1; // Compute Ritz vectors?
      char howmny = 'A'; // Compute nev ritz vectors
      vector_int select; // length nev, since howmny = 'A', this is the workspace
      select.Setup(ncv, true);
      
      dr.Setup(nev+1, true);
      di.Setup(nev+1, true);
      
      double sigmar = 0.0; // represents the real part of the shift
      double sigmai = 0.0; // represents the imag part of the shift
      
      double one = 1.0;
      double zero = 0.0;
      
      /* Outer if, sym? */
      if(sym)
      {
         /* Inner if, timing? */
         if(!timing)
         {
            while(ido != 99)
            {
              SCHURCHEB_PARPACK_PDSAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                break;
              }
              
              switch(ido)
              {
                case -1:
                {
                  /* compute Y = op*X where 
                   * workd[ipntr[0]] is the X
                   * world[ipntr[1]] is the Y
                   */
                  
                  v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                  w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                  
                  w.Fill(zero);
                  A.MatVec( 'N', one, v, zero, w);
                  nmvs++;
                  
                  break;
                }
                case 1:
                {
                  /* compute Y = op*X where 
                   * workd[ipntr[0]] is the X
                   * world[ipntr[1]] is the Y
                   */
                  
                  v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                  w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                  
                  w.Fill(zero);
                  A.MatVec( 'N', one, v, zero, w);
                  nmvs++;
                  
                  break;
                }
              }         
            }
         }
         else
         {
            double ts, te;
            while(ido != 99)
            {
              SCHURCHEB_PARPACK_PDSAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                break;
              }
              
              switch(ido)
              {
                case -1:
                {
                  /* compute Y = op*X where 
                   * workd[ipntr[0]] is the X
                   * world[ipntr[1]] is the Y
                   */
                  
                  v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                  w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                  
                  w.Fill(zero);
                  SchurchebMpiTime( comm, ts);
                  A.MatVec( 'N', one, v, zero, w);
                  SchurchebMpiTime( comm, te);
                  tmvs += (te - ts);
                  nmvs++;
                  
                  break;
                }
                case 1:
                {
                  /* compute Y = op*X where 
                   * workd[ipntr[0]] is the X
                   * world[ipntr[1]] is the Y
                   */
                  
                  v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                  w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                  
                  w.Fill(zero);
                  SchurchebMpiTime( comm, ts);
                  A.MatVec( 'N', one, v, zero, w);
                  SchurchebMpiTime( comm, te);
                  tmvs += (te - ts);
                  nmvs++;
                  
                  break;
                }
              }         
            }
         }/* Inner if, timing? */
         
         if(info >= 0)
         {
           SCHURCHEB_PARPACK_PDSEUPD ( &fcomm,
                    &rvec, 
                    &howmny, 
                    select.GetData(), 
                    dr.GetData(), 
                    V.GetData(), 
                    &ldv, 
                    &sigmar,
                    &bmat, &n, which, &nev, &tol_eig, resid.GetData(), &ncv, V.GetData(), &ldv,
                    iparam.GetData(), ipntr.GetData(), workd.GetData(), workl.GetData(), &lworkl, &info );
           if(info < 0)
           {
             printf("Error: sneupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
             return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
           }
         }
         else
         {
           printf("Error: snaupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
           return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
         }
      }
      else
      {
         /* Inner if, timing? */
         if(!timing)
         {
            while(ido != 99)
            {
              SCHURCHEB_PARPACK_PDNAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                break;
              }
              
              switch(ido)
              {
                case -1:
                {
                  /* compute Y = op*X where 
                   * workd[ipntr[0]] is the X
                   * world[ipntr[1]] is the Y
                   */
                  
                  v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                  w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                  
                  w.Fill(zero);
                  A.MatVec( 'N', one, v, zero, w);
                  nmvs++;
                  
                  break;
                }
                case 1:
                {
                  /* compute Y = op*X where 
                   * workd[ipntr[0]] is the X
                   * world[ipntr[1]] is the Y
                   */
                  
                  v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                  w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                  
                  w.Fill(zero);
                  A.MatVec( 'N', one, v, zero, w);
                  nmvs++;
                  
                  break;
                }
              }         
            }
         }
         else
         {
            double ts, te;
            while(ido != 99)
            {
              SCHURCHEB_PARPACK_PDNAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                break;
              }
              
              switch(ido)
              {
                case -1:
                {
                  /* compute Y = op*X where 
                   * workd[ipntr[0]] is the X
                   * world[ipntr[1]] is the Y
                   */
                  
                  v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                  w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                  
                  w.Fill(zero);
                  SchurchebMpiTime( comm, ts);
                  A.MatVec( 'N', one, v, zero, w);
                  SchurchebMpiTime( comm, te);
                  tmvs += (te - ts);
                  nmvs++;
                  
                  break;
                }
                case 1:
                {
                  /* compute Y = op*X where 
                   * workd[ipntr[0]] is the X
                   * world[ipntr[1]] is the Y
                   */
                  
                  v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                  w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                  
                  w.Fill(zero);
                  SchurchebMpiTime( comm, ts);
                  A.MatVec( 'N', one, v, zero, w);
                  SchurchebMpiTime( comm, te);
                  tmvs += (te - ts);
                  nmvs++;
                  
                  break;
                }
              }         
            }
         }/* Inner if, timing? */
         
         if(info >= 0)
         {
            SCHURCHEB_PARPACK_PDNEUPD ( &fcomm,
                    &rvec, 
                    &howmny, 
                    select.GetData(), 
                    dr.GetData(), 
                    di.GetData(), 
                    V.GetData(), 
                    &ldv, 
                    &sigmar,
                    &sigmai,
                    workev.GetData(),
                    &bmat, &n, which, &nev, &tol_eig, resid.GetData(), &ncv, V.GetData(), &ldv,
                    iparam.GetData(), ipntr.GetData(), workd.GetData(), workl.GetData(), &lworkl, &info );
            if(info < 0)
            {
               printf("Error: dseupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
               return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
            }
         }
         else
         {
            printf("Error: dsaupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
            return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
         }
         
      } /* End of outer if, sym? */
      
      
      return SCHURCHEB_SUCCESS;
   }
   template int ArpackArnoldi<vector_seq_double>( arnoldimatrix_seq_double &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V, 
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog);
   template int ArpackArnoldi<vector_par_double>( arnoldimatrix_par_double &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V, 
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog);
   template int ArpackArnoldi<vector_seq_double>( matrix_csr_double &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V, 
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog);
   template int ArpackArnoldi<vector_par_double>( matrix_csr_par_double &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V, 
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog);


   template <class VectorType, class MatrixType, class SolverType>
   int ArpackArnoldi( MatrixType &A, MatrixType &M, SolverType &Minv, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, int &nsvs, double &tsvs, parallel_log &parlog)
   {
      
      /* DNAUPD */
      VectorType v, w;
      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);
      int n = v.GetLengthLocal();
      int n_global = v.GetLengthGlobal();
      
      int ido = 0; //must be 0 for the first call
      char bmat = 'G'; // generalized eigenvalue problem
      
      int np, myid;
      MPI_Comm comm;
      parlog.GetMpiInfo(np, myid, comm);
      int fcomm = MPI_Comm_c2f(comm);
      
      nmvs = 0;
      nsvs = 0;
      tmvs = 0.0;
      tsvs = 0.0;
      
      vector_seq_double resid; // array of length n
      resid.Setup(n, true);
      
      if(n_global < nev + 2)
      {
         SCHURCHEB_ERROR("nev can't be larger than n-2.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      msteps = SchurchebMax( msteps, nev + 2);
      int ncv = SchurchebMin( msteps, n_global); // number of columns of the matrix V
      // 2 <= ncv - nev && ncv <= n
      
      if(V.GetNumRowsLocal() < n || V.GetNumColsLocal() < ncv)
      {
         V.Setup(n, ncv, true);
      }
      int ldv = V.GetLeadingDimension();
      
      vector_int iparam;
      iparam.Setup(11, true);
      
      iparam[0] = 1; // equv to restart with linear combination of approximateed Schur vectors
      iparam[2] = maxits; // max iter
      iparam[3] = 1;
      iparam[6] = 2; //mode, A*x = lambda*M*x, M symmetric positive definite
                     //===> OP = inv[M]*A  and  B = M.
      
      vector_int ipntr;
      ipntr.Setup(14, true);
      
      vector_seq_double workd;
      workd.Setup(3*n, true);
      
      int lworkl;
      vector_seq_double workl;
      vector_seq_double workev; // working vector
      
      if(sym)
      {
         if(which[1] == 'R')
         {
            which[1] = 'A';
         }
         lworkl = ncv*ncv +  8*ncv;
         workl.Setup(lworkl, true);
      }
      else
      {
         lworkl = 3*ncv*ncv +  6*ncv;
         workl.Setup(lworkl, true);
         workev.Setup(3*ncv);
      }
      
      int info = 0;
      
      /* DNEUPD */
      
      int rvec = 1; // Compute Ritz vectors?
      char howmny = 'A'; // Compute nev ritz vectors
      vector_int select; // length nev, since howmny = 'A', this is the workspace
      select.Setup(ncv, true);
      
      dr.Setup(nev+1, true);
      di.Setup(nev+1, true);
      
      double sigmar = 0.0; // represents the real part of the shift
      double sigmai = 0.0; // represents the imag part of the shift
      
      double one = 1.0;
      double zero = 0.0;
      
      /* Outer if, sym? */
      if(sym)
      {
         /* Inner if, timing? */
         if(!timing)
         {
            while(ido != 99)
            {
               SCHURCHEB_PARPACK_PDSAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                  break;
              }
              
              switch(ido)
              {
                  case -1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     //===> OP = inv[M]*A  and  B = M.
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     A.MatVec( 'N', one, v, zero, w);
                     Minv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     //===> OP = inv[M]*A  and  B = M.
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     A.MatVec( 'N', one, v, zero, w);
                     Minv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 2:
                  {
                      /* compute Y = B*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                      v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                      w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                      w.Fill(zero);
                      M.MatVec( 'N', one, v, zero, w);
                      nmvs++;
                      break;
                  }
               }         
            }
         }
         else
         {
            double ts, te;
            while(ido != 99)
            {
               SCHURCHEB_PARPACK_PDSAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                  break;
              }
              
              switch(ido)
              {
                  case -1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     SchurchebMpiTime( comm, ts);
                     A.MatVec( 'N', one, v, zero, w);
                     SchurchebMpiTime( comm, te);
                     tmvs += (te - ts);
                     Minv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     SchurchebMpiTime( comm, ts);
                     tsvs += (ts - te);
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     SchurchebMpiTime( comm, ts);
                     A.MatVec( 'N', one, v, zero, w);
                     SchurchebMpiTime( comm, te);
                     tmvs += (te - ts);
                     Minv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     SchurchebMpiTime( comm, ts);
                     tsvs += (ts - te);
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 2:
                  {
                      /* compute Y = B*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                      v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                      w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                      w.Fill(zero);
                      SchurchebMpiTime( comm, ts);
                      M.MatVec( 'N', one, v, zero, w);
                      SchurchebMpiTime( comm, te);
                      tmvs += (te - ts);
                      nmvs++;
                      break;
                  }
               }         
            }
         }/* Inner if, timing? */
         
         if(info >= 0)
         {
            SCHURCHEB_PARPACK_PDSEUPD ( &fcomm,
                    &rvec, 
                    &howmny, 
                    select.GetData(), 
                    dr.GetData(), 
                    V.GetData(), 
                    &ldv, 
                    &sigmar,
                    &bmat, &n, which, &nev, &tol_eig, resid.GetData(), &ncv, V.GetData(), &ldv,
                    iparam.GetData(), ipntr.GetData(), workd.GetData(), workl.GetData(), &lworkl, &info );
            if(info < 0)
            {
                printf("Error: dseupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
                return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
            }
         }
         else
         {
            printf("Error: dsaupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
            return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
         }
      }
      else
      {
         /* Inner if, timing? */
         if(!timing)
         {
            while(ido != 99)
            {
               SCHURCHEB_PARPACK_PDNAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                  break;
              }
              
              switch(ido)
              {
                  case -1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     //===> OP = inv[M]*A  and  B = M.
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     A.MatVec( 'N', one, v, zero, w);
                     Minv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     //===> OP = inv[M]*A  and  B = M.
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     A.MatVec( 'N', one, v, zero, w);
                     Minv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 2:
                  {
                      /* compute Y = B*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                      v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                      w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                      w.Fill(zero);
                      M.MatVec( 'N', one, v, zero, w);
                      nmvs++;
                      break;
                  }
               }         
            }
         }
         else
         {
            double ts, te;
            while(ido != 99)
            {
               SCHURCHEB_PARPACK_PDNAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                  break;
              }
              
              switch(ido)
              {
                  case -1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     SchurchebMpiTime( comm, ts);
                     A.MatVec( 'N', one, v, zero, w);
                     SchurchebMpiTime( comm, te);
                     tmvs += (te - ts);
                     Minv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     SchurchebMpiTime( comm, ts);
                     tsvs += (ts - te);
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     SchurchebMpiTime( comm, ts);
                     A.MatVec( 'N', one, v, zero, w);
                     SchurchebMpiTime( comm, te);
                     tmvs += (te - ts);
                     Minv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     SchurchebMpiTime( comm, ts);
                     tsvs += (ts - te);
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 2:
                  {
                      /* compute Y = B*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                      v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                      w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                      w.Fill(zero);
                      SchurchebMpiTime( comm, ts);
                      M.MatVec( 'N', one, v, zero, w);
                      SchurchebMpiTime( comm, te);
                      tmvs += (te - ts);
                      nmvs++;
                      break;
                  }
               }         
            }
         }/* Inner if, timing? */
         
         if(info >= 0)
         {
            SCHURCHEB_PARPACK_PDNEUPD ( &fcomm,
                    &rvec, 
                    &howmny, 
                    select.GetData(), 
                    dr.GetData(), 
                    di.GetData(), 
                    V.GetData(), 
                    &ldv, 
                    &sigmar,
                    &sigmai,
                    workev.GetData(),
                    &bmat, &n, which, &nev, &tol_eig, resid.GetData(), &ncv, V.GetData(), &ldv,
                    iparam.GetData(), ipntr.GetData(), workd.GetData(), workl.GetData(), &lworkl, &info );
            if(info < 0)
            {
                printf("Error: dneupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
                return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
            }
         }
         else
         {
            printf("Error: dnaupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
            return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
         }
      }
      /* end of outer if */
      
      
      return SCHURCHEB_SUCCESS;
   }
   template int ArpackArnoldi<vector_par_double>( matrix_csr_par_double &A, matrix_csr_par_double &M, arnoldimatrix_par_double &Minv, int msteps, 
                        int maxits, int nev, char *which, bool sym, double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, int &nsvs, double &tsvs, parallel_log &parlog);

   template <class VectorType, class MatrixType, class SolverType>
   int ArpackArnoldi_inv( MatrixType &A, MatrixType &M, SolverType &Ainv, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, int &nsvs, double &tsvs, parallel_log &parlog)
   {
      
      /* DNAUPD */
      VectorType v, w;
      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);
      int n = v.GetLengthLocal();
      int n_global = v.GetLengthGlobal();
      
      int ido = 0; //must be 0 for the first call
      char bmat = 'G'; // generalized eigenvalue problem
      
      int np, myid;
      MPI_Comm comm;
      parlog.GetMpiInfo(np, myid, comm);
      int fcomm = MPI_Comm_c2f(comm);
      
      nmvs = 0;
      nsvs = 0;
      tmvs = 0.0;
      tsvs = 0.0;
      
      vector_seq_double resid; // array of length n
      resid.Setup(n, true);
      
      if(n_global < nev + 2)
      {
         SCHURCHEB_ERROR("nev can't be larger than n-2.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      msteps = SchurchebMax( msteps, nev + 2);
      int ncv = SchurchebMin( msteps, n_global); // number of columns of the matrix V
      // 2 <= ncv - nev && ncv <= n
      
      if(V.GetNumRowsLocal() < n || V.GetNumColsLocal() < ncv)
      {
         V.Setup(n, ncv, true);
      }
      int ldv = V.GetLeadingDimension();
      
      vector_int iparam;
      iparam.Setup(11, true);
      
      iparam[0] = 1; // equv to restart with linear combination of approximateed Schur vectors
      iparam[2] = maxits; // max iter
      iparam[3] = 1;
      iparam[6] = 3; //mode, M semi-spd, inverse-mode， //===> OP = inv[A]*M  and  B = M.
      
      vector_int ipntr;
      ipntr.Setup(14, true);
      
      vector_seq_double workd;
      workd.Setup(3*n, true);
      
      int lworkl;
      vector_seq_double workl;
      vector_seq_double workev; // working vector
      
      /* compute the opposite for shift-and-invert */
      if(which[0] == 'S')
      {
         which[0] = 'L';
      }
      else if(which[0] == 'L')
      {
         which[0] = 'S';
      }
      
      if(sym)
      {
         if(which[1] == 'R')
         {
            which[1] = 'A';
         }
         lworkl = ncv*ncv +  8*ncv;
         workl.Setup(lworkl, true);
      }
      else
      {
         lworkl = 3*ncv*ncv +  6*ncv;
         workl.Setup(lworkl, true);
         workev.Setup(3*ncv);
      }
      
      
      int info = 0;
      
      /* DNEUPD */
      
      int rvec = 1; // Compute Ritz vectors?
      char howmny = 'A'; // Compute nev ritz vectors
      vector_int select; // length nev, since howmny = 'A', this is the workspace
      select.Setup(ncv, true);
      
      dr.Setup(nev+1, true);
      di.Setup(nev+1, true);
      
      double sigmar = 0.0; // represents the real part of the shift
      double sigmai = 0.0; // represents the imag part of the shift
      
      double one = 1.0;
      double zero = 0.0;
      
      /* Outer if, sym? */
      if(sym)
      {
         /* Inner if, timing? */
         if(!timing)
         {
            while(ido != 99)
            {
               SCHURCHEB_PARPACK_PDSAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                  break;
              }
              
              switch(ido)
              {
                  case -1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     //===> OP = inv[M]*A  and  B = M.
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     M.MatVec( 'N', one, v, zero, w); // inplace solve requires no extra memory
                     Ainv.MatVec( 'N', one, w, zero, w);
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     //===> OP = inv[A]*B  and  B = M.
                     //v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     SCHURCHEB_MEMCPY(workd.GetData()+ipntr[1]-1, workd.GetData()+ipntr[2]-1, v.GetLengthLocal(),
                     kMemoryHost, kMemoryHost, double);
                     Ainv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     nsvs++;
                     
                     break;
                  }
                  case 2:
                  {
                      /* compute Y = B*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                      v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                      w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                      w.Fill(zero);
                      M.MatVec( 'N', one, v, zero, w);
                      nmvs++;
                      break;
                  }
               }         
            }
         }
         else
         {
            double ts, te;
            while(ido != 99)
            {
               SCHURCHEB_PARPACK_PDSAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                  break;
              }
              
              switch(ido)
              {
                  case -1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     SchurchebMpiTime( comm, ts);
                     M.MatVec( 'N', one, v, zero, w);
                     SchurchebMpiTime( comm, te);
                     tmvs += (te - ts);
                     Ainv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     SchurchebMpiTime( comm, ts);
                     tsvs += (ts - te);
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     //v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     SCHURCHEB_MEMCPY(workd.GetData()+ipntr[1]-1, workd.GetData()+ipntr[2]-1, v.GetLengthLocal(),
                     kMemoryHost, kMemoryHost, double);
                     SchurchebMpiTime( comm, ts);
                     Ainv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     SchurchebMpiTime( comm, te);
                     tsvs += (te - ts);
                     nsvs++;
                     
                     break;
                  }
                  case 2:
                  {
                      /* compute Y = B*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                      v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                      w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                      w.Fill(zero);
                      SchurchebMpiTime( comm, ts);
                      M.MatVec( 'N', one, v, zero, w);
                      SchurchebMpiTime( comm, te);
                      tmvs += (te - ts);
                      nmvs++;
                      break;
                  }
               }         
            }
         }/* Inner if, timing? */
         
         if(info >= 0)
         {
            SCHURCHEB_PARPACK_PDSEUPD ( &fcomm,
                    &rvec, 
                    &howmny, 
                    select.GetData(), 
                    dr.GetData(), 
                    V.GetData(), 
                    &ldv, 
                    &sigmar,
                    &bmat, &n, which, &nev, &tol_eig, resid.GetData(), &ncv, V.GetData(), &ldv,
                    iparam.GetData(), ipntr.GetData(), workd.GetData(), workl.GetData(), &lworkl, &info );
            if(info < 0)
            {
                printf("Error: dseupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
                return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
            }
         }
         else
         {
            printf("Error: dsaupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
            return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
         }
      }
      else
      {
         /* Inner if, timing? */
         if(!timing)
         {
            while(ido != 99)
            {
               SCHURCHEB_PARPACK_PDNAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                  break;
              }
              
              switch(ido)
              {
                  case -1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     //===> OP = inv[A]*M  and  B = M.
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     M.MatVec( 'N', one, v, zero, w);
                     Ainv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * workd[ipntr[1]] is the Y
                      * workd[ipntr[2]] already holds B*X
                      */
                     
                     //===> OP = inv[A]*M  and  B = M.
                     //v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     /* copy B*x to w */
                     SCHURCHEB_MEMCPY(workd.GetData()+ipntr[1]-1, workd.GetData()+ipntr[2]-1, v.GetLengthLocal(),
                     kMemoryHost, kMemoryHost, double);
                     Ainv.MatVec( 'N', one, w, zero, w);
                     nsvs++;
                     
                     break;
                  }
                  case 2:
                  {
                      /* compute Y = B*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                      v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                      w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                      w.Fill(zero);
                      M.MatVec( 'N', one, v, zero, w);
                      nmvs++;
                      break;
                  }
               }         
            }
         }
         else
         {
            double ts, te;
            while(ido != 99)
            {
               SCHURCHEB_PARPACK_PDNAUPD ( &fcomm,
                     &ido, 
                     &bmat,
                     &n, 
                     which, 
                     &nev, 
                     &tol_eig, 
                     resid.GetData(), 
                     &ncv, 
                     V.GetData(), 
                     &ldv, 
                     iparam.GetData(),
                     ipntr.GetData(), 
                     workd.GetData(), 
                     workl.GetData(), 
                     &lworkl, 
                     &info);
              
              if(info != 0)
              {
                  break;
              }
              
              switch(ido)
              {
                  case -1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     SchurchebMpiTime( comm, ts);
                     M.MatVec( 'N', one, v, zero, w);
                     SchurchebMpiTime( comm, te);
                     tmvs += (te - ts);
                     Ainv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     SchurchebMpiTime( comm, ts);
                     tsvs += (ts - te);
                     nmvs++;
                     nsvs++;
                     
                     break;
                  }
                  case 1:
                  {
                     /* compute Y = op*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                     
                     //v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                     w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                     w.Fill(zero);
                     SCHURCHEB_MEMCPY(workd.GetData()+ipntr[1]-1, workd.GetData()+ipntr[2]-1, v.GetLengthLocal(),
                     kMemoryHost, kMemoryHost, double);
                     SchurchebMpiTime( comm, ts);
                     Ainv.MatVec( 'N', one, w, zero, w); // inplace solve requires no extra memory
                     SchurchebMpiTime( comm, te);
                     tsvs += (te - ts);
                     nsvs++;
                     
                     break;
                  }
                  case 2:
                  {
                      /* compute Y = B*X where 
                      * workd[ipntr[0]] is the X
                      * world[ipntr[1]] is the Y
                      */
                      v.UpdatePtr(workd.GetData()+ipntr[0]-1, kMemoryHost);
                      w.UpdatePtr(workd.GetData()+ipntr[1]-1, kMemoryHost);
                     
                      w.Fill(zero);
                      SchurchebMpiTime( comm, ts);
                      M.MatVec( 'N', one, v, zero, w);
                      SchurchebMpiTime( comm, te);
                      tmvs += (te - ts);
                      nmvs++;
                      break;
                  }
               }         
            }
         }/* Inner if, timing? */
         
         if(info >= 0)
         {
            SCHURCHEB_PARPACK_PDNEUPD ( &fcomm,
                    &rvec, 
                    &howmny, 
                    select.GetData(), 
                    dr.GetData(), 
                    di.GetData(), 
                    V.GetData(), 
                    &ldv, 
                    &sigmar,
                    &sigmai,
                    workev.GetData(),
                    &bmat, &n, which, &nev, &tol_eig, resid.GetData(), &ncv, V.GetData(), &ldv,
                    iparam.GetData(), ipntr.GetData(), workd.GetData(), workl.GetData(), &lworkl, &info );
            if(info < 0)
            {
                printf("Error: dneupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
                return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
            }
         }
         else
         {
            printf("Error: dnaupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
            return SCHURCHEB_ERROR_FUNCTION_CALL_ERR;
         }
      }
      
      
      return SCHURCHEB_SUCCESS;
   }
   template int ArpackArnoldi_inv<vector_par_double>( matrix_csr_par_double &A, matrix_csr_par_double &M, arnoldimatrix_par_double &Ainv, int msteps, 
                        int maxits, int nev, char *which, bool sym, double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, int &nsvs, double &tsvs, parallel_log &parlog);
#else
   template <class VectorType, class MatrixType>
   int ArpackArnoldi( MatrixType &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog)
   {
      SCHURCHEB_ERROR("Doesn't compile with ARPACK, ARPACK arnoldi not supported.");
      return SCHURCHEB_ERROR_INVALED_OPTION;
   }
   template int ArpackArnoldi<vector_seq_double>( arnoldimatrix_seq_double &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V, 
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog);
   template int ArpackArnoldi<vector_par_double>( arnoldimatrix_par_double &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V, 
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog);
   template int ArpackArnoldi<vector_seq_double>( matrix_csr_double &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V, 
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog);
   template int ArpackArnoldi<vector_par_double>( matrix_csr_par_double &A, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V, 
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, parallel_log &parlog);


   template <class VectorType, class MatrixType, class SolverType>
   int ArpackArnoldi( MatrixType &A, MatrixType &M, SolverType &Minv, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, int &nsvs, double &tsvs, parallel_log &parlog)
   {
      SCHURCHEB_ERROR("Doesn't compile with ARPACK, ARPACK arnoldi not supported.");
      return SCHURCHEB_ERROR_INVALED_OPTION;
   }
   template int ArpackArnoldi<vector_par_double>( matrix_csr_par_double &A, matrix_csr_par_double &M, arnoldimatrix_par_double &Minv, int msteps, 
                        int maxits, int nev, char *which, bool sym, double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, int &nsvs, double &tsvs, parallel_log &parlog);

   template <class VectorType, class MatrixType, class SolverType>
   int ArpackArnoldi_inv( MatrixType &A, MatrixType &M, SolverType &Ainv, int msteps, int maxits, int nev, char *which, bool sym,
                        double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, int &nsvs, double &tsvs, parallel_log &parlog)
   {
      SCHURCHEB_ERROR("Doesn't compile with ARPACK, ARPACK arnoldi not supported.");
      return SCHURCHEB_ERROR_INVALED_OPTION;
   }
   template int ArpackArnoldi_inv<vector_par_double>( matrix_csr_par_double &A, matrix_csr_par_double &M, arnoldimatrix_par_double &Ainv, int msteps, 
                        int maxits, int nev, char *which, bool sym, double tol_eig, matrix_dense_double &V,
                        vector_seq_double &dr, vector_seq_double &di, bool timing, int &nmvs, double &tmvs, int &nsvs, double &tsvs, parallel_log &parlog);

#endif

}

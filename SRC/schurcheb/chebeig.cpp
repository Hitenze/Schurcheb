#include "chebeig.hpp"

/**
 * @file chebeig.cpp
 * @brief main class for the Schur Chebyshev eigensolver
 */

namespace schurcheb
{

   SchurChebClass::SchurChebClass()
   {
      // do nothing
   }
   
   SchurChebClass::SchurChebClass(const SchurChebClass &str)
   {
      SCHURCHEB_ERROR("Copy constructor not implemented for class SchurChebClass.");
   }
   
   SchurChebClass::SchurChebClass(SchurChebClass &&str)
   {
      SCHURCHEB_ERROR("Move constructor not implemented for class SchurChebClass.");
   }
   
   SchurChebClass& SchurChebClass::operator= (const SchurChebClass &str)
   {
      SCHURCHEB_ERROR("= operator not implemented for class SchurChebClass.");
      return *this;
   }
   
   SchurChebClass& SchurChebClass::operator= (SchurChebClass &&str)
   {
      SCHURCHEB_ERROR("= operator not implemented for class SchurChebClass.");
      return *this;
   }
   
   int SchurChebClass::Clear()
   {
      this->_perm_v.Clear();
      this->_eigs_v.Clear();
      this->_res_v.Clear();
      this->_A_par.Clear();
      this->_M_par.Clear();
      this->_V_mat.Clear();
      this->_V_mat_seq.Clear();
      this->_test_shift.Clear();
      return SCHURCHEB_SUCCESS;
   }
   
   SchurChebClass::~SchurChebClass()
   {
      this->Clear();
   }
   
   int SchurChebClass::Setup( matrix_csr_double &A, int neig, double a, double b, int &ndom, int nnode, int ncol)
   {
      matrix_csr_double M; // not referenced
      return this->Setup( A, M, false, false, neig, a, b, ndom, nnode, ncol,
               false, 2*neig, 100, 1e-08, 1e-12, 1, false, 1, 1, true, 1);
   }
   
   int SchurChebClass::Setup( matrix_csr_double &A, matrix_csr_double &M, int neig, double a, double b, int &ndom, int nnode, int ncol)
   {
      return this->Setup( A, M, true, false, neig, a, b, ndom, nnode, ncol,
               false, 2*neig, 100, 1e-08, 1e-12, 1, false, 1, 1, true, 1);
   }
   
   int SchurChebClass::Setup( matrix_csr_double &A, matrix_csr_double &M, bool gen, bool check_dd,
               int neig, double a, double b, int &ndom, int nnode, int ncol,
               bool lan, int m, int niter, double tol_eig, double tol_eig2, 
               int orth_nB, bool chol_orth, int B_sol_opt, int eigvec_opt, bool compute_res, int print_level)
   {
      double ts, te, tall = .0, t_set = .0, t_ar = .0, t_vmv = .0, t_armv = .0;

      double EPSILON = std::numeric_limits<double>::epsilon();
      int local_solver = B_sol_opt;
      
      int neigs = neig;
      int nnodes = nnode;
      
      int n = A.GetNumRowsLocal();
      int nrow;
      int msteps = m;
      
      bool generalize = gen;
      
      int np, myid;
      
      np = parallel_log::_gsize;
      myid = parallel_log::_grank;
      
      /* -------------------
       * Step 0: Init 
       * ------------------- */
      
      /* Step 0.0: check input params */
#ifndef SCHURCHEB_MKL
      local_solver = 0;
#endif
      
      if(b <= a)
      {
         if(myid == 0)
         {
            cout<<"Error: Invalid intival [a, b]."<<endl;
         }
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      if( np % ncol != 0 )
      {
         if(myid == 0)
         {
            cout<<"Error: Invalid 2D MPI assignment (np %% ncol != 0)."<<endl;
         }
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      if( ncol > nnodes )
      {
         if(myid == 0)
         {
            cout<<"Error: Invalid 2D MPI assignment (nnode < ncol)."<<endl;
         }
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      nrow = np / ncol;
      
      if(ndom < nrow)
      {
         ndom = nrow;
      }
      
      /* Step 0.1: print problem settings */
      if(myid == 0)
      {
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
         cout<<"Problem size:          "<<n<<endl;
         cout<<"2D MPI:                "<<ncol<<" * "<<nrow<<endl;
         cout<<"Number of subdomains:  "<<ndom<<endl;
         cout<<"Number of Eigs         "<<neigs<<endl;
         cout<<"Number of Cheb nodes   "<<nnodes<<endl;
         cout<<"Range:                 ["<<a<<", "<<b<<"]"<<endl;
         cout<<"Eigenvalue res tol:    "<<tol_eig<<endl;
         cout<<"Local res tol:         "<<tol_eig2<<endl;
         cout<<"Krylov m per restart:  "<<msteps<<endl;
         cout<<"Max number restarts:   "<<niter<<endl;
         cout<<"Orth block size:       "<<orth_nB<<endl;
         cout<<"Chol orth?:            "<<chol_orth<<endl;
         cout<<"No full orth?          "<<lan<<endl;
         if(local_solver == 0)
         {
            cout<<"Use LU as local direct solver."<<endl;
         }
         else if(local_solver == 1)
         {
            cout<<"Use pardiso as local direct solver."<<endl;
         }
         else
         {
            cout<<"No local direct solver."<<endl;
            return SCHURCHEB_ERROR_INVALED_OPTION;
         }
         
         cout<<"Print level:           "<<print_level<<endl;
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }
      
      /* -------------------
       * Step 1: Create Matrix
       * ------------------- */
      
      matrix_csr_double       App, Mpp;
      matrix_csr_par_double   B, E, F, C, MB, ME, MF, MC;
      
      /* Step 1.0: create test problem */
      SchurchebMpiTime( MPI_COMM_WORLD, ts); // Start timming
      
      /* compute problem size and create M when necessary */
      if(!generalize)
      {
         M.Setup( n, n, false);
         M.Eye();
      }
      SchurchebMpiTime( MPI_COMM_WORLD, te); // End timming
      
      if(myid == 0 && print_level)
      {
         printf("Create M time: %8.6fs\n",te-ts);
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }  
      
      /* -------------------
       * Step 2: Local MPI
       * ------------------- */
      
      /* 1st direction */
      int color = myid / nrow;
      parallel_log parlog_local;
      
      SCHURCHEB_MALLOC( parlog_local._comm, 1, kMemoryHost, MPI_Comm);
      MPI_Comm_split( MPI_COMM_WORLD, color, myid, parlog_local._comm);
      
      MPI_Comm_rank( *(parlog_local._comm), &(parlog_local._rank) );
      MPI_Comm_size( *(parlog_local._comm), &(parlog_local._size) );
      
      int myidi, npi;
      MPI_Comm commi;
      parlog_local.GetMpiInfo( npi, myidi, commi);
      
      /* 2nd direction */
      int myidj, npj;
      MPI_Comm commj;
      
      int color2 = myid % nrow;
      parallel_log parlog_local2;
      
      SCHURCHEB_MALLOC( parlog_local2._comm, 1, kMemoryHost, MPI_Comm);
      MPI_Comm_split( MPI_COMM_WORLD, color2, myid, parlog_local2._comm);
      
      MPI_Comm_rank( *(parlog_local2._comm), &(parlog_local2._rank) );
      MPI_Comm_size( *(parlog_local2._comm), &(parlog_local2._size) );
      
      parlog_local2.GetMpiInfo( npj, myidj, commj);
      
      /* a new communication in row order 
       * 0 4   0 1
       * 1 5   2 3
       * 2 6 ->4 5
       * 3 7   6 7
       */
      
      int myid2, np2;
      MPI_Comm comm2;
      
      int color3 = myidi*npj+myidj;
      parallel_log parlog2;
      
      SCHURCHEB_MALLOC( parlog2._comm, 1, kMemoryHost, MPI_Comm);
      MPI_Comm_split( MPI_COMM_WORLD, 0, color3, parlog2._comm);
      
      MPI_Comm_rank( *(parlog2._comm), &(parlog2._rank) );
      MPI_Comm_size( *(parlog2._comm), &(parlog2._size) );
      
      parlog2.GetMpiInfo( np2, myid2, comm2);
      
      /* -------------------
       * Step 3: Partition
       * ------------------- */
      
      SchurchebMpiTime( MPI_COMM_WORLD, ts); // Start timming
      
      vector_int &perm = this->_perm_v;
      vector_int dom_ptr, Bdiagsizes;
      
      DD( A, M, nrow, ndom, perm, dom_ptr, Bdiagsizes);
      
      A.SubMatrix( perm, perm, kMemoryHost, App);
      M.SubMatrix( perm, perm, kMemoryHost, Mpp);
      
      if(check_dd)
      {
         // check the quality of the DD, check at most 3 times
         int max_chk = 3, chk = 1;
         double lmin = 0;
         
         {
            int nB, nBi;
            nB = dom_ptr[npi] - dom_ptr[0];
            nBi = dom_ptr[myidi+1] - dom_ptr[myidi];
            B.Setup(nBi, dom_ptr[myidi], nB, nBi, dom_ptr[myidi], nB, parlog_local);
            MB.Setup(nBi, dom_ptr[myidi], nB, nBi, dom_ptr[myidi], nB, parlog_local);
            
            /* B and MB */
            App.SubMatrix( dom_ptr[myidi], dom_ptr[myidi], nBi, nBi, kMemoryHost, B.GetDiagMat());
            B.GetOffdMat().Setup(nBi, 0, 0);
            B.GetOffdMat().GetIVector().Fill(0);
            B.GetOffdMat().GetJVector().Fill(0);
            B.GetOffdMat().GetDataVector().Fill(0.0);
            Mpp.SubMatrix( dom_ptr[myidi], dom_ptr[myidi], nBi, nBi, kMemoryHost, MB.GetDiagMat());
            MB.GetOffdMat().Setup(nBi, 0, 0);
            MB.GetOffdMat().GetIVector().Fill(0);
            MB.GetOffdMat().GetJVector().Fill(0);
            MB.GetOffdMat().GetDataVector().Fill(0.0);
            
            matrix_shift temp_shift;
            matrix_dense_double Z;
            
            temp_shift.Setup(B, E, F, C, MB, ME, MF, MC, 0.0, parlog_local);
            temp_shift.SetBSolveOption(true);
            
            int nB_global = B.GetNumRowsGlobal();
            int m = SchurchebMin( 1, nB_global); // keep some of them
            int m2 = m;
            int lr_m = SchurchebMin( m2, nB_global); // at most 30 its
            int nmvs = 0;
            double tmvs = .0;
            
            char aropt[2];
            aropt[0] = 'L';
            aropt[1] = 'R';
            
            ArnoldiMatrixClass<vector_par_double, double> &temp_mat = temp_shift;
            
#ifdef SCHURCHEB_PARPACK
// use the PARPACK Krylov
            {
               vector_seq_double dr, di;
               ArpackArnoldi<vector_par_double>( temp_mat, lr_m, niter, m2, aropt, lan,
                              1e-12, Z, dr, di, false, nmvs, tmvs, parlog_local);
               if(m2 > 0)
               {
                  lmin = dr[0];
               }
            }
#else
// use the Built-in Krylov
            {
               
               matrix_dense_double H_temp;
               vector_par_double z_temp;
               double normz;
               double one = 1.0;
               int maxsteps  = SchurchebMin( (int)( 2*m2 + lr_m + (lr_m * SCHURCHEB_global::_tr_factor)), nB_global);
               
               Z.Setup( B.GetNumRowsLocal(), maxsteps+1, kMemoryHost, true);
               H_temp.Setup( maxsteps+1, maxsteps, kMemoryHost, true);
               
               /* setup init guess */
               temp_mat.SetupVectorPtrStr(z_temp);
               z_temp.UpdatePtr( &(Z(0,0)), kMemoryHost);
               SCHURCHEB_global::_mersenne_twister_engine.seed(0);
               z_temp.Rand();
               
               /* normalize v */
               z_temp.Norm2(normz);
               z_temp.Scale(one/normz);
               
               m2 = SchurchebArnoldiThickRestartNoLock<vector_par_double>( temp_mat, lr_m, niter, m2, m2, 
                                       SCHURCHEB_global::_tr_factor, 1e-12, aropt,
                                       Z, H_temp, SCHURCHEB_global::_orth_tol,  SCHURCHEB_global::_reorth_tol, nmvs);
               
               if(m2 > 0)
               {
                  lmin = H_temp(0,0);
               }
            }
#endif
         }
         
         while(1.0/lmin < b && chk < max_chk)
         {
            chk++;
            App.Clear();
            Mpp.Clear();
            B.Clear();
            MB.Clear();
            perm.Clear();
            dom_ptr.Clear();
            Bdiagsizes.Clear();
            
            ndom *= 2;
                  
            DD( A, M, nrow, ndom, perm, dom_ptr, Bdiagsizes);
            
            A.SubMatrix( perm, perm, kMemoryHost, App);
            M.SubMatrix( perm, perm, kMemoryHost, Mpp);
            
            
            {
               int nB, nBi;
               nB = dom_ptr[npi] - dom_ptr[0];
               nBi = dom_ptr[myidi+1] - dom_ptr[myidi];
               B.Setup(nBi, dom_ptr[myidi], nB, nBi, dom_ptr[myidi], nB, parlog_local);
               MB.Setup(nBi, dom_ptr[myidi], nB, nBi, dom_ptr[myidi], nB, parlog_local);
               
               /* B and MB */
               App.SubMatrix( dom_ptr[myidi], dom_ptr[myidi], nBi, nBi, kMemoryHost, B.GetDiagMat());
               B.GetOffdMat().Setup(nBi, 0, 0);
               B.GetOffdMat().GetIVector().Fill(0);
               B.GetOffdMat().GetJVector().Fill(0);
               B.GetOffdMat().GetDataVector().Fill(0.0);
               Mpp.SubMatrix( dom_ptr[myidi], dom_ptr[myidi], nBi, nBi, kMemoryHost, MB.GetDiagMat());
               MB.GetOffdMat().Setup(nBi, 0, 0);
               MB.GetOffdMat().GetIVector().Fill(0);
               MB.GetOffdMat().GetJVector().Fill(0);
               MB.GetOffdMat().GetDataVector().Fill(0.0);
               
               matrix_shift temp_shift;
               matrix_dense_double Z;
               
               temp_shift.Setup(B, E, F, C, MB, ME, MF, MC, 0.0, parlog_local);
               temp_shift.SetBSolveOption(true);
               
               int nB_global = B.GetNumRowsGlobal();
               int m = SchurchebMin( 1, nB_global); // keep some of them
               int m2 = m;
               int lr_m = SchurchebMin( m2, nB_global); // at most 30 its
               int nmvs = 0;
               double tmvs = .0;
               
               char aropt[2];
               aropt[0] = 'L';
               aropt[1] = 'R';
               
               ArnoldiMatrixClass<vector_par_double, double> &temp_mat = temp_shift;
               
#ifdef SCHURCHEB_PARPACK
// use the PARPACK Krylov
               {
                  vector_seq_double dr, di;
                  ArpackArnoldi<vector_par_double>( temp_mat, lr_m, niter, m2, aropt, lan,
                                 1e-12, Z, dr, di, false, nmvs, tmvs, parlog_local);
                  if(m2 > 0)
                  {
                     lmin = dr[0];
                  }
               }
#else
// use the Built-in Krylov
               {
                  
                  matrix_dense_double H_temp;
                  vector_par_double z_temp;
                  double normz;
                  double one = 1.0;
                  int maxsteps  = SchurchebMin( (int)( 2*m2 + lr_m + (lr_m * SCHURCHEB_global::_tr_factor)), nB_global);
                  
                  Z.Setup( B.GetNumRowsLocal(), maxsteps+1, kMemoryHost, true);
                  H_temp.Setup( maxsteps+1, maxsteps, kMemoryHost, true);
                  
                  /* setup init guess */
                  temp_mat.SetupVectorPtrStr(z_temp);
                  z_temp.UpdatePtr( &(Z(0,0)), kMemoryHost);
                  SCHURCHEB_global::_mersenne_twister_engine.seed(0);
                  z_temp.Rand();
                  
                  /* normalize v */
                  z_temp.Norm2(normz);
                  z_temp.Scale(one/normz);
                  
                  m2 = SchurchebArnoldiThickRestartNoLock<vector_par_double>( temp_mat, lr_m, niter, m2, m2, 
                                          SCHURCHEB_global::_tr_factor, 1e-12, aropt,
                                          Z, H_temp, SCHURCHEB_global::_orth_tol,  SCHURCHEB_global::_reorth_tol, nmvs);
                  
                  if(m2 > 0)
                  {
                     lmin = H_temp(0,0);
                  }
               }
#endif
               B.Clear();
               MB.Clear();
            }
         }
      }
      
      /* -----------------------
       * Step 4: Extract Matrices
       * ---------------------- */
      
      int nB, nC, nBi, nCi;
      nB = dom_ptr[npi] - dom_ptr[0];
      nC = dom_ptr[2*npi] - dom_ptr[npi];
      nBi = dom_ptr[myidi+1] - dom_ptr[myidi];
      nCi = dom_ptr[npi+myidi+1] - dom_ptr[npi+myidi];
      
      /* setup parallel block diagonal matrices */
      B.Setup(nBi, dom_ptr[myidi], nB, nBi, dom_ptr[myidi], nB, parlog_local);
      E.Setup(nCi, dom_ptr[npi+myidi]-dom_ptr[npi], nC, nBi, dom_ptr[myidi], nB, parlog_local);
      F.Setup(nBi, dom_ptr[myidi], nB, nCi, dom_ptr[npi+myidi]-dom_ptr[npi], nC, parlog_local);
      MB.Setup(nBi, dom_ptr[myidi], nB, nBi, dom_ptr[myidi], nB, parlog_local);
      ME.Setup(nCi, dom_ptr[npi+myidi]-dom_ptr[npi], nC, nBi, dom_ptr[myidi], nB, parlog_local);
      MF.Setup(nBi, dom_ptr[myidi], nB, nCi, dom_ptr[npi+myidi]-dom_ptr[npi], nC, parlog_local);
      
      /* B, E, F */
      App.SubMatrix( dom_ptr[myidi], dom_ptr[myidi], nBi, nBi, kMemoryHost, B.GetDiagMat());
      B.GetOffdMat().Setup(nBi, 0, 0);// empty
      B.GetOffdMat().GetIVector().Fill(0);
      B.GetOffdMat().GetJVector().Fill(0);
      B.GetOffdMat().GetDataVector().Fill(0.0);
      
      App.SubMatrix( dom_ptr[npi+myidi], dom_ptr[myidi], nCi, nBi, kMemoryHost, E.GetDiagMat());
      E.GetOffdMat().Setup(nCi, 0, 0);// empty
      E.GetOffdMat().GetIVector().Fill(0);
      E.GetOffdMat().GetJVector().Fill(0);
      E.GetOffdMat().GetDataVector().Fill(0.0);
      
      App.SubMatrix( dom_ptr[myidi], dom_ptr[npi+myidi], nBi, nCi, kMemoryHost, F.GetDiagMat());
      F.GetOffdMat().Setup(nBi, 0, 0);// empty
      F.GetOffdMat().GetIVector().Fill(0);
      F.GetOffdMat().GetJVector().Fill(0);
      F.GetOffdMat().GetDataVector().Fill(0.0);
      
      /* MB, ME, MF */
      Mpp.SubMatrix( dom_ptr[myidi], dom_ptr[myidi], nBi, nBi, kMemoryHost, MB.GetDiagMat());
      MB.GetOffdMat().Setup(nBi, 0, 0);// empty
      MB.GetOffdMat().GetIVector().Fill(0);
      MB.GetOffdMat().GetJVector().Fill(0);
      MB.GetOffdMat().GetDataVector().Fill(0.0);
      
      Mpp.SubMatrix( dom_ptr[npi+myidi], dom_ptr[myidi], nCi, nBi, kMemoryHost, ME.GetDiagMat());
      ME.GetOffdMat().Setup(nCi, 0, 0);// empty
      ME.GetOffdMat().GetIVector().Fill(0);
      ME.GetOffdMat().GetJVector().Fill(0);
      ME.GetOffdMat().GetDataVector().Fill(0.0);
      
      Mpp.SubMatrix( dom_ptr[myidi], dom_ptr[npi+myidi], nBi, nCi, kMemoryHost, MF.GetDiagMat());
      MF.GetOffdMat().Setup(nBi, 0, 0);// empty
      MF.GetOffdMat().GetIVector().Fill(0);
      MF.GetOffdMat().GetJVector().Fill(0);
      MF.GetOffdMat().GetDataVector().Fill(0.0);
      
      /* C, MC */
      ExtractParallelCsrSubMatrix( App, C, dom_ptr.GetData()+npi, parlog_local);
      ExtractParallelCsrSubMatrix( Mpp, MC, dom_ptr.GetData()+npi, parlog_local);
      
      /* App and Mpp nolonger used */
      App.Clear();
      Mpp.Clear();
      
      /* next prepare the parallel version of each diagonal block (used in V and Z) */
      vector_int dom_ptrdB, dom_ptrdBg, dom_ptrdC, dom_ptrdCg, dom_ptrdg, dom_countdg;
      int dom1, dom2, nBj, nCj;
      
      dom_ptrdB.Setup(npj+1);
      dom_ptrdC.Setup(npj+1);
      
      dom1 = nBi / npj;
      dom2 = nBi % npj;
      dom_ptrdB[0] = 0;
      dom1 ++;
      for(int i = 0 ; i < dom2 ; i ++)
      {
         dom_ptrdB[i+1] = dom_ptrdB[i] + dom1;
      }
      dom1 --;
      for(int i = dom2 ; i < npj ; i ++)
      {
         dom_ptrdB[i+1] = dom_ptrdB[i] + dom1;
      }
      
      dom1 = nCi / npj;
      dom2 = nCi % npj;
      dom_ptrdC[0] = 0;
      dom1 ++;
      for(int i = 0 ; i < dom2 ; i ++)
      {
         dom_ptrdC[i+1] = dom_ptrdC[i] + dom1;
      }
      dom1 --;
      for(int i = dom2 ; i < npj ; i ++)
      {
         dom_ptrdC[i+1] = dom_ptrdC[i] + dom1;
      }
      
      /* C is parallel, do something special */
      dom_ptrdCg.Setup(np2+1);
      dom_ptrdCg[0] = 0;
      for(int j = 0 ; j < npi ; j ++)
      {
         nCj = dom_ptr[npi+j+1] - dom_ptr[npi+j];
         
         dom1 = nCj / npj;
         dom2 = nCj % npj;
         dom1 ++;
         for(int i = 0 ; i < dom2 ; i ++)
         {
            dom_ptrdCg[j*npj+i+1] = dom_ptrdCg[j*npj+i] + dom1;
         }
         dom1 --;
         for(int i = dom2 ; i < npj ; i ++)
         {
            dom_ptrdCg[j*npj+i+1] = dom_ptrdCg[j*npj+i] + dom1;
         }
      }
      
      /* also need the global partition of B for Gram Schmidtz */
      dom_ptrdBg.Setup(np2+1);
      dom_ptrdBg[0] = 0;
      for(int j = 0 ; j < npi ; j ++)
      {
         nBj = dom_ptr[j+1] - dom_ptr[j];
         
         dom1 = nBj / npj;
         dom2 = nBj % npj;
         dom1 ++;
         for(int i = 0 ; i < dom2 ; i ++)
         {
            dom_ptrdBg[j*npj+i+1] = dom_ptrdBg[j*npj+i] + dom1;
         }
         dom1 --;
         for(int i = dom2 ; i < npj ; i ++)
         {
            dom_ptrdBg[j*npj+i+1] = dom_ptrdBg[j*npj+i] + dom1;
         }
      }
      
      dom_ptrdg.Setup(np2+1);
      dom_countdg.Setup(np2);
      dom_ptrdg[0] = 0;
      for(int j = 0 ; j < np2 ; j ++)
      {
         dom_ptrdg[j+1] = dom_ptrdBg[j+1] + dom_ptrdCg[j+1];
         dom_countdg[j] = dom_ptrdg[j+1] - dom_ptrdg[j];
      }
      
      
      SchurchebMpiTime( MPI_COMM_WORLD, te); // End timming
      
      if(myid == 0 && print_level)
      {
         printf("DD time:    %8.6fs\n",te-ts);
         printf("Schur size: %d\n",C.GetNumRowsGlobal());
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }  
      
      /* -----------------------
       * Step 5: Call local solve
       * ---------------------- */
      
      vector_seq_double shifts, local_shifts;
      
      // disable shift and invert
      this->_test_shift._shift_invert = false;
      
      shifts.Setup(nnodes);
      if(nnodes != 1)
      {
         for(int i = 0 ; i < nnodes ; i ++)
         {
            shifts[i] = (a+b)/2.0 + (b-a)/2.0*cos(i*3.1415926535/(nnodes-1));
         }
      }
      else
      {
         shifts[0] = (a+b)/2.0;
      }
      
      vector_int  nodes_ptr;
      int         local_node1, local_node2, local_node3;
      
      nodes_ptr.Setup(ncol+1);
      
      local_node1 = nnodes / ncol;
      local_node2 = nnodes % ncol;
      
      nodes_ptr[0] = 0;
      for(int i = 0 ; i < ncol ; i ++)
      {
         if(i < local_node2)
         {
            nodes_ptr[i+1] = nodes_ptr[i] + local_node1 + 1;
         }
         else
         {
            nodes_ptr[i+1] = nodes_ptr[i] + local_node1;
         }
      }
      
      local_node3 = nodes_ptr[color+1] - nodes_ptr[color];
      
      if(myid == 0)
      {
         cout<<"Cheb nodes: ";
         for(int i = 0 ; i < nnodes ; i ++)
         {
            cout<<shifts[i]<<" ";
         }
         cout<<endl;
      }
      
      int mg;
      matrix_dense_double Vg, Zg;
      vector_par_double v;
      matrix_dense_double Z;
      
      Vg.Setup( nBi, (neigs+1)*local_node3, kMemoryHost, true);
      Zg.Setup( nCi, (neigs+1)*local_node3, kMemoryHost, true);
      mg = 0;

      double one = 1.0;
      double zero = 0.0;
      
      this->_test_shift.SetSolveOption(local_solver);
      this->_test_shift.SetPrintLevel(print_level);
      
      for(int nodei = nodes_ptr[color] ; nodei < nodes_ptr[color+1] ; nodei ++)
      {
         SchurchebMpiTime( commi, ts); // Start timming
         
         if(nodei == nodes_ptr[color])
         {
            this->_test_shift.Setup(B, E, F, C, MB, ME, MF, MC, shifts[nodei], parlog_local);
         }
         else
         {
            this->_test_shift.UpdateShift(shifts[nodei]);
         }
         
         SchurchebMpiTime( commi, te); // End timming
         if(myidi == 0 && print_level > 1)
         {
            printf("Setup Matvec on node %d time: %8.6fs\n",nodei,te-ts);
            SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
         }  
         
         t_set += (te - ts);
         
         /* -----------------------
          * Step 6: Apply Krylov
          * ---------------------- */
         
         int nC_global = C.GetNumRowsGlobal();
         
         int m = SchurchebMin( neigs, nC_global); // keep some of them
         int m2 = m;
         int lr_m = SchurchebMin( msteps, nC_global); // at most 300 its
         int nmvs = 0;
         double tmvs = .0;
         
         /* apply Krylov */
         
         SchurchebMpiTime( commi, ts); // Start timming
         
         char aropt[2];
         aropt[0] = 'S';
         aropt[1] = 'R';
         
         ArnoldiMatrixClass<vector_par_double, double> &temp_mat = this->_test_shift;
         
#ifdef SCHURCHEB_PARPACK
// use the PARPACK Krylov
         {
            vector_seq_double dr, di;
            if(print_level > 1)
            {
               ArpackArnoldi<vector_par_double>( temp_mat, lr_m, niter, m2, aropt, lan,
                           tol_eig, Z, dr, di, true, nmvs, tmvs, parlog_local);
            }
            else
            {
               ArpackArnoldi<vector_par_double>( temp_mat, lr_m, niter, m2, aropt, lan,
                           tol_eig, Z, dr, di, false, nmvs, tmvs, parlog_local);
            }
         }
#else
// use the Built-in Krylov
         {
            
            matrix_dense_double H_temp;
            vector_par_double z_temp;
            double normz;
            int maxsteps  = SchurchebMin( (int)( 2*m2 + lr_m + (lr_m * SCHURCHEB_global::_tr_factor)), nC_global);
            
            Z.Setup( C.GetNumRowsLocal(), maxsteps+1, kMemoryHost, true);
            H_temp.Setup( maxsteps+1, maxsteps, kMemoryHost, true);
            
            /* setup init guess */
            temp_mat.SetupVectorPtrStr(z_temp);
            z_temp.UpdatePtr( &(Z(0,0)), kMemoryHost);
            SCHURCHEB_global::_mersenne_twister_engine.seed(0);
            z_temp.Rand();
            
            /* normalize v */
            z_temp.Norm2(normz);
            z_temp.Scale(one/normz);
            
            m2 = SchurchebArnoldiThickRestartNoLock<vector_par_double>( temp_mat, lr_m, niter, m2, m2, 
                                    SCHURCHEB_global::_tr_factor, tol_eig, aropt,
                                    Z, H_temp, SCHURCHEB_global::_orth_tol,  SCHURCHEB_global::_reorth_tol, nmvs);
            
         }
#endif
         

         if(m2 < m)
         {
            /* faile to comput all */
            if(myid == 0)
            {
               printf("Fail to capture all eigenvalues, please increase niter or m.\n");
            }
               
            SchurchebFinalize();
            return -1;
         
         }      
         
         /* now copy Z to Zg */
         SCHURCHEB_MEMCPY( &(Zg(0, mg)), Z.GetData(), this->_test_shift._nC*m, kMemoryHost, kMemoryHost, double);
         
         SchurchebMpiTime( commi, te); // End timming
         
         if(myidi == 0 && print_level > 1)
         {
            printf("Krylov on node %d time: %8.6fs\n",nodei,te-ts);
            if(print_level > 1)
            {
               printf("Krylov on node %d mvs:  %d, time: %8.6fs\n",nodei,nmvs,tmvs);
            }
            else
            {
               printf("Krylov on node %d mvs:  %d\n",nodei,nmvs);
            }
            SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
         }  
         
         t_ar += (te - ts);
         t_armv += tmvs;
         
         Z.Clear();
         
         /* -----------------------
          * Step 7: Compute V
          * ---------------------- */

         SchurchebMpiTime( commi, ts); // Start timming

         vector_par_double w;
         
         v.SetupPtrStr(C);
         w.SetupPtrStr(B);
         
         for(int i = mg ; i < mg+m ; i ++)
         {
            v.UpdatePtr( &Zg(0,i), kMemoryHost);
            w.UpdatePtr( &Vg(0,i), kMemoryHost);
            this->_test_shift.MatVec2( 'N', one, v, zero, w);
         }
         
         SchurchebMpiTime( commi, te); // End timming
         if(myidi == 0 && print_level > 1)
         {
            printf("V MatVec on node %d time: %8.6fs\n",nodei,te-ts);
            SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
         }  
         
         t_vmv += (te - ts);
         
         mg += m;
         
      }
      
      double t_setg, t_arg, t_armvg, t_vmvg;
      SchurchebMpiAllreduce( &t_set, &t_setg, 1, MPI_MAX, MPI_COMM_WORLD);
      SchurchebMpiAllreduce( &t_ar, &t_arg, 1, MPI_MAX, MPI_COMM_WORLD);
      if(print_level > 1)
      {
         SchurchebMpiAllreduce( &t_armv, &t_armvg, 1, MPI_MAX, MPI_COMM_WORLD);
      }
      SchurchebMpiAllreduce( &t_vmv, &t_vmvg, 1, MPI_MAX, MPI_COMM_WORLD);
      
      tall += t_setg + t_arg + t_vmvg;
      
      /* compute total memory */
      double _gmem;
      int    _gnnz;
      SchurchebMpiAllreduce( &(this->_test_shift._mem), &_gmem, 1, MPI_SUM, MPI_COMM_WORLD);
      SchurchebMpiAllreduce( &(this->_test_shift._nnz), &_gnnz, 1, MPI_SUM, MPI_COMM_WORLD);
      
      if(myid == 0 && print_level)
      {
         printf("Total Setup time: %8.6fs\n",t_setg);
         printf("Total Krylov time: %8.6fs\n",t_arg);
         if(print_level > 1)
         {
            printf("Total B solve memory (0 means not reported) %8.6fMB, NNZ: %d\n", _gmem, _gnnz);
            printf("Total Krylov MV time: %8.6fs\n",t_armvg);
         }
         printf("Total V MatVec time: %8.6fs\n",t_vmvg);
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }
      
      SchurchebMpiTime( MPI_COMM_WORLD, ts); // Start timming
      
      /* -----------------------
       * Step 8: Assemble V and Z
       *     [V0, V1,...]
       * R = [Z0, Z1,...]
       *     |V|
       *   = |Z|
       * 
       * We use the following steps:
       * 
       * Now, we have the following structure
       * 
       * local_rank\group     0     1     2     3
       *         0          V_0_0 V_1_0 V_2_0 V_3_0
       *         0          Z_0_0 Z_1_0 Z_2_0 Z_3_0
       *         1          V_0_1 V_1_1 V_2_1 V_3_1
       *         1          Z_0_1 Z_1_1 Z_2_1 Z_3_1
       *         2          V_0_2 V_1_2 V_2_2 V_3_2
       *         2          Z_0_2 Z_1_2 Z_2_2 Z_3_2
       *         3          V_0_3 V_1_3 V_2_3 V_3_3
       *         3          Z_0_3 Z_1_3 Z_2_3 Z_3_3
       * 
       * we want to have a global permutation
       * (r,c)                local_rank\group     0     1     2     3
       * (0,0)  V1_0               <- 0          V_0_0 V_1_0 V_2_0 V_3_0
       * (0,0)  Z1_0               <- 0          Z_0_0 Z_1_0 Z_2_0 Z_3_0
       * (0,1)  V1_1               <- 1          V_0_1 V_1_1 V_2_1 V_3_1
       * (0,1)  Z1_1               <- 1          Z_0_1 Z_1_1 Z_2_1 Z_3_1
       * (0,2)  V1_2               <- 2          V_0_2 V_1_2 V_2_2 V_3_2
       * (0,2)  Z1_2               <- 2          Z_0_2 Z_1_2 Z_2_2 Z_3_2
       * (0,3)  V1_3               <- 3          V_0_3 V_1_3 V_2_3 V_3_3
       * (0,3)  Z1_3               <- 3          Z_0_3 Z_1_3 Z_2_3 Z_3_3
       * (3,3)  V1_4
       * (3,3)  Z1_4
       * (1,0)  V1_5
       * (1,0)  Z1_5
       * (1,1)  V1_6
       * (1,1)  Z1_6
       * (1,2)  V1_7
       * (1,2)  Z1_7
       * (1,3)  V1_8
       * (1,3)  Z1_8
       * (2,0)  V1_9
       * (2,0)  Z1_9
       * (2,1)  V1_10
       * (2,1)  Z1_10
       * (2,2)  V1_11
       * (2,2)  Z1_11
       * (2,3)  V1_12
       * (2,3)  Z1_12
       * (3,0)  V1_13
       * (3,0)  Z1_13
       * (3,1)  V1_14
       * (3,1)  Z1_14
       * (3,2)  V1_15
       * (3,2)  Z1_15
       * 
       * In local that is
       *               local_rank\group     0     1     2     3
       * V1_0               <- 0          V_0_0 V_1_0 V_2_0 V_3_0
       * Z1_0               <- 0          Z_0_0 Z_1_0 Z_2_0 Z_3_0
       * V1_1 
       * Z1_1 
       * V1_2 
       * Z1_2 
       * V1_3 
       * Z1_3 
       * V1_4
       * Z1_4
       * 
       * re-distribute to 1D, however, the order is a little bit different
       * 
       * then apply parallel orthgonization
       * 
       * ---------------------- */
      
      /* get mg on each subgroup */
      vector_int mgs, mgdisps;
      mgs.Setup(ncol);
      mgdisps.Setup(ncol+1);
      SchurchebMpiAllgather( &mg, 1, mgs.GetData(), commj);
      mgdisps[0] = 0;
      for(int i = 0; i < ncol ; i ++)
      {
         mgdisps[i+1] = mgdisps[i] + mgs[i];
      }
      
      /* first assembel V and Z along 2nd direction to the first groups */
      int                  mr, aa_count;
      matrix_dense_double  V1, Z1, V2, Z2;
      matrix_dense_double  R1, R3;
      vector_seq_double    V_vec, Z_vec; // send buffer
      vector_par_double    temp_v;
      
      /* we send the fisrt several rows, need to transpose */
      V_vec.Setup(mg*this->_test_shift._nB);
      Z_vec.Setup(mg*this->_test_shift._nC);
      
      aa_count = 0;
      for(int i = 0 ; i < npj ; i ++)
      {
         for(int j = 0 ; j < mg ; j ++)
         {
            for(int k = dom_ptrdB[i] ; k < dom_ptrdB[i+1] ; k ++)
            {
               V_vec[aa_count++] = Vg(k, j);
            }
         }
      }
      
      aa_count = 0;
      for(int i = 0 ; i < npj ; i ++)
      {
         for(int j = 0 ; j < mg ; j ++)
         {
            for(int k = dom_ptrdC[i] ; k < dom_ptrdC[i+1] ; k ++)
            {
               Z_vec[aa_count++] = Zg(k, j);
            }
         }
      }
      
      Vg.Clear();
      Zg.Clear();
      
      /* create buffer */
      V1.Setup( dom_ptrdB[myidj+1]-dom_ptrdB[myidj], mgdisps[ncol], true);
      Z1.Setup( dom_ptrdC[myidj+1]-dom_ptrdC[myidj], mgdisps[ncol], true);
      
      /* create buffer for R */
      R1.Setup( dom_ptrdB[myidj+1]-dom_ptrdB[myidj]+dom_ptrdC[myidj+1]-dom_ptrdC[myidj], mgdisps[ncol], true);
      R3.Setup( dom_ptrdB[myidj+1]-dom_ptrdB[myidj]+dom_ptrdC[myidj+1]-dom_ptrdC[myidj], mgdisps[ncol], true);
      
      vector_int mgsB, mgdispsB, mgrB, mgdisprB;
      vector_int mgsC, mgdispsC, mgrC, mgdisprC;
      mgsB.Setup(ncol);
      mgrB.Setup(ncol);
      mgdispsB.Setup(ncol);
      mgdisprB.Setup(ncol);
      mgsC.Setup(ncol);
      mgrC.Setup(ncol);
      mgdispsC.Setup(ncol);
      mgdisprC.Setup(ncol);
      
      /* each processor send several columns */
      for(int i = 0 ; i < npj ; i ++)
      {
         mgsB[i] = mg*(dom_ptrdB[i+1]-dom_ptrdB[i]);
         mgdispsB[i] = mg*dom_ptrdB[i];
         mgsC[i] = mg*(dom_ptrdC[i+1]-dom_ptrdC[i]);
         mgdispsC[i] = mg*dom_ptrdC[i];
      }
      
      /* each processor recv several columns */
      for(int i = 0 ; i < npj ; i ++)
      {
         mgrB[i] = mgs[i]*(dom_ptrdB[myidj+1] - dom_ptrdB[myidj]);
         mgdisprB[i] = mgdisps[i]*(dom_ptrdB[myidj+1] - dom_ptrdB[myidj]);
         mgrC[i] = mgs[i]*(dom_ptrdC[myidj+1] - dom_ptrdC[myidj]);
         mgdisprC[i] = mgdisps[i]*(dom_ptrdC[myidj+1] - dom_ptrdC[myidj]);
      }
      
      MPI_Alltoallv( V_vec.GetData(), mgsB.GetData(), mgdispsB.GetData(), MPI_DOUBLE, 
                     V1.GetData(), mgrB.GetData(), mgdisprB.GetData(), MPI_DOUBLE, 
                     commj);
      
      MPI_Alltoallv( Z_vec.GetData(), mgsC.GetData(), mgdispsC.GetData(), MPI_DOUBLE, 
                     Z1.GetData(), mgrC.GetData(), mgdisprC.GetData(), MPI_DOUBLE, 
                     commj);
      
      /* next copy Z1 and V1 into R1 */
      for(int i = 0 ; i < mgdisps[ncol] ; i ++)
      {
         SCHURCHEB_MEMCPY( &(R1(0, i)), &(V1(0, i)), dom_ptrdB[myidj+1]-dom_ptrdB[myidj], kMemoryHost, kMemoryHost, double);
         SCHURCHEB_MEMCPY( &(R1(dom_ptrdB[myidj+1]-dom_ptrdB[myidj], i)), &(Z1(0, i)), dom_ptrdC[myidj+1]-dom_ptrdC[myidj], kMemoryHost, kMemoryHost, double);
      }
      
      SchurchebMpiTime( MPI_COMM_WORLD, te); // End timming
      tall += te - ts;
      if(myid == 0 && print_level)
      {
         printf("Gather V and Z time: %8.6fs\n",te-ts);
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }  
      SchurchebMpiTime( MPI_COMM_WORLD, ts); // Start timming
      
      /* next apply orthgonization */
      {
         
         //matrix_dense_double Rv, Rz;
         temp_v.Setup(R1.GetNumRowsLocal(), parlog2);
         temp_v.GetDataVector().Clear();
         temp_v.GetDataVector().SetupPtrStr(R1.GetNumRowsLocal()); /* create vector buffer */
         
         if(chol_orth)
         {
            if(orth_nB > 1)
            {
               SchurchebBlockOrthogonal2( R1, temp_v, R3, orth_nB, mr, 1e-14);
            }
            else
            {
               SchurchebCholOrthogonal( R1, temp_v, R3, mr, 1e-14);
            }
         }
         else
         {
            if(orth_nB > 1)
            {
               SchurchebBlockOrthogonal( R1, temp_v, R3, orth_nB, mr, 1e-14);
            }
            else
            {
               SchurchebOrthogonal( R1, temp_v, R3, mr, 1e-14);
            }
         }
         
         temp_v.Clear();
      }
      
      R1.Clear();
      
      SchurchebMpiTime( MPI_COMM_WORLD, te); // End timming
      tall += te - ts;
      if(myid == 0 && print_level)
      {
         printf("Orth time: %8.6fs\n",te-ts);
         printf("R size: %d\n",mr);
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }  
      SchurchebMpiTime( MPI_COMM_WORLD, ts); // Start timming
      
      /* -----------------------
       * Step 9: Transfer R back
       * 
       * We use the following steps:
       * 
       * Now, we want to have the following structure
       * 
       * local_rank\group     0     1     2     3
       *         0          V_0_0 V_1_0 V_2_0 V_3_0
       *         0          Z_0_0 Z_1_0 Z_2_0 Z_3_0
       *         1          V_0_1 V_1_1 V_2_1 V_3_1
       *         1          Z_0_1 Z_1_1 Z_2_1 Z_3_1
       *         2          V_0_2 V_1_2 V_2_2 V_3_2
       *         2          Z_0_2 Z_1_2 Z_2_2 Z_3_2
       *         3          V_0_3 V_1_3 V_2_3 V_3_3
       *         3          Z_0_3 Z_1_3 Z_2_3 Z_3_3
       * 
       * we want to have a global permutation
       * (r,c)                local_rank\group     0     1     2     3
       * (0,0)  V1_0               -> 0          V_0_0 V_1_0 V_2_0 V_3_0
       * (0,0)  Z1_0               -> 0          Z_0_0 Z_1_0 Z_2_0 Z_3_0
       * (0,1)  V1_1               -> 1          V_0_1 V_1_1 V_2_1 V_3_1
       * (0,1)  Z1_1               -> 1          Z_0_1 Z_1_1 Z_2_1 Z_3_1
       * (0,2)  V1_2               -> 2          V_0_2 V_1_2 V_2_2 V_3_2
       * (0,2)  Z1_2               -> 2          Z_0_2 Z_1_2 Z_2_2 Z_3_2
       * (0,3)  V1_3               -> 3          V_0_3 V_1_3 V_2_3 V_3_3
       * (0,3)  Z1_3               -> 3          Z_0_3 Z_1_3 Z_2_3 Z_3_3
       * (3,3)  V1_4
       * (3,3)  Z1_4
       * (1,0)  V1_5
       * (1,0)  Z1_5
       * (1,1)  V1_6
       * (1,1)  Z1_6
       * (1,2)  V1_7
       * (1,2)  Z1_7
       * (1,3)  V1_8
       * (1,3)  Z1_8
       * (2,0)  V1_9
       * (2,0)  Z1_9
       * (2,1)  V1_10
       * (2,1)  Z1_10
       * (2,2)  V1_11
       * (2,2)  Z1_11
       * (2,3)  V1_12
       * (2,3)  Z1_12
       * (3,0)  V1_13
       * (3,0)  Z1_13
       * (3,1)  V1_14
       * (3,1)  Z1_14
       * (3,2)  V1_15
       * (3,2)  Z1_15
       * 
       * In local that is
       *               local_rank\group     0     1     2     3
       * V1_0               -> 0          V_0_0 V_1_0 V_2_0 V_3_0
       * Z1_0               -> 0          Z_0_0 Z_1_0 Z_2_0 Z_3_0
       * V1_1 
       * Z1_1 
       * V1_2 
       * Z1_2 
       * V1_3 
       * Z1_3 
       * V1_4
       * Z1_4
       * 
       * re-distribute to 1D, however, the order is a little bit different
       * 
       * then apply parallel orthgonization
       * 
       * ---------------------- */
      
      /* now we have mr columns, re-distribute to each MPI processes */
      
      matrix_dense_double  V3, Z3;
      
      int mr1, mr2;
      vector_int mrs, mrdisps; // new column in each npj
      
      mr1 = mr / npj;
      mr2 = mr % npj;
      
      mrdisps.Setup(npj + 1);
      mrs.Setup(npj);
      mrdisps[0] = 0;
      
      for(int i = 0 ; i < npj ; i ++)
      {
         if(i < mr2)
         {
            mrs[i] = mr1 + 1;
            mrdisps[i+1] = mrdisps[i] + mrs[i];
         }
         else
         {
            mrs[i] = mr1;
            mrdisps[i+1] = mrdisps[i] + mrs[i];
         }
      }
      
      /* copy R3 back to Z1 and V1 */
      for(int i = 0 ; i < mr ; i ++)
      {
         
         SCHURCHEB_MEMCPY( &(V1(0, i)), &(R3(0, i)), dom_ptrdB[myidj+1]-dom_ptrdB[myidj], kMemoryHost, kMemoryHost, double);
         SCHURCHEB_MEMCPY( &(Z1(0, i)), &(R3(dom_ptrdB[myidj+1]-dom_ptrdB[myidj], i)), dom_ptrdC[myidj+1]-dom_ptrdC[myidj], kMemoryHost, kMemoryHost, double);
         
      }
      
      /* update send/recv buffer */
      for(int i = 0 ; i < npj ; i ++)
      {
         mgsB[i] = mrs[myidj]*(dom_ptrdB[i+1]-dom_ptrdB[i]);
         mgdispsB[i] = mrs[myidj]*dom_ptrdB[i];
         mgsC[i] = mrs[myidj]*(dom_ptrdC[i+1]-dom_ptrdC[i]);
         mgdispsC[i] = mrs[myidj]*dom_ptrdC[i];
      }
      
      /* each processor recv several columns */
      for(int i = 0 ; i < npj ; i ++)
      {
         mgrB[i] = mrs[i]*(dom_ptrdB[myidj+1] - dom_ptrdB[myidj]);
         mgdisprB[i] = mrdisps[i]*(dom_ptrdB[myidj+1] - dom_ptrdB[myidj]);
         mgrC[i] = mrs[i]*(dom_ptrdC[myidj+1] - dom_ptrdC[myidj]);
         mgdisprC[i] = mrdisps[i]*(dom_ptrdC[myidj+1] - dom_ptrdC[myidj]);
      }
      
      
      /* then transfer data back to V/Z vec */
      
      V_vec.Setup(mrs[myidj]*this->_test_shift._nB);
      Z_vec.Setup(mrs[myidj]*this->_test_shift._nC);
      
      MPI_Alltoallv( V1.GetData(), mgrB.GetData(), mgdisprB.GetData(), MPI_DOUBLE, 
                     V_vec.GetData(), mgsB.GetData(), mgdispsB.GetData(), MPI_DOUBLE, 
                     commj);
      
      MPI_Alltoallv( Z1.GetData(), mgrC.GetData(), mgdisprC.GetData(), MPI_DOUBLE, 
                     Z_vec.GetData(), mgsC.GetData(), mgdispsC.GetData(), MPI_DOUBLE, 
                     commj);
      
      V1.Clear();
      Z1.Clear();
      
      /* copy to V3 and Z3 */
      V3.Setup( nBi, mrs[myidj]);
      Z3.Setup( nCi, mrs[myidj]);
      
      aa_count = 0;
      for(int i = 0 ; i < npj ; i ++)
      {
         for(int j = 0 ; j < mrs[myidj] ; j ++)
         {
            for(int k = dom_ptrdB[i] ; k < dom_ptrdB[i+1] ; k ++)
            {
               V3(k, j) = V_vec[aa_count++];
            }
         }
      }
      
      aa_count = 0;
      for(int i = 0 ; i < npj ; i ++)
      {
         for(int j = 0 ; j < mrs[myidj] ; j ++)
         {
            for(int k = dom_ptrdC[i] ; k < dom_ptrdC[i+1] ; k ++)
            {
               Z3(k, j) = Z_vec[aa_count++];
            }
         }
      }
      
      V_vec.Clear();
      Z_vec.Clear();
      
      SchurchebMpiTime( MPI_COMM_WORLD, te); // End timming
      tall += te - ts;
      if(myid == 0 && print_level)
      {
         printf("Re-distribute R time: %8.6fs\n",te-ts);
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }  
      SchurchebMpiTime( MPI_COMM_WORLD, ts); // Start timming
      
      /* -----------------------------
       * Step 10: Generating R'AR, R'MR
       * 
       * We do this in three steps
       * 1. Compute AR and MR. Example:
       * |B0     F0      |   |V00 V01|
       * |    B1     F1  |   |V10 V11|
       * |E0     C0  C01 | * |Z00 Z01|
       * |    E1 C10 C1  |   |Z10 Z11|
       * =
       * |B0V00+F0Z00        B0V01+F0Z01| 
       * |B1V10+F1Z10        B1V11+F1Z11|         => BV + FZ
       * |E0V00+C0Z00+C01Z10 E0V01+C0Z01+C01Z11|
       * |E1V10+C1Z10+C10Z00 E1V11+C1Z11+C10Z01|  => EV + CZ
       * 
       * 2. Compute local R'AR and R'MR. Example
       * 
       *                         |X00 X01|
       *                         |X10 X11|
       * |V00' V10' Z00' Z10'| * |Y00 Y01|
       * |V01' V11' Z01' Z11'|   |Y10 Y11|
       * =
       *                         |X00 X01|
       *                         |Y00 Y01|
       * |V00' Z00' V10' Z10'| * |X10 X11|
       * |V01' Z01' V11' Z11'|   |Y10 Y11|
       * =
       * 
       * |N00' N10'| * |M00 M01|
       * |N01' N11'|   |M10 M11|
       * Parallel MatMat, 2D distribution.
       * 
       * In local we compute 
       * N00' * |M00 M01| on p00
       * N01' * |M00 M01| on p01
       * N10' * |M10 M11| on p10
       * N11' * |M10 M11| on p11
       * 
       * 3. Reduce the result to the first MPI processor
       * N00' * |M00 M01| + N10' * |M10 M11|
       * N01' * |M00 M01| + N11' * |M10 M11|
       * 
       * each column (commi) sum together to form a block row
       * gather along row (commj) to form the final result
       * 
       * ----------------------------- */
      
      matrix_dense_double RAR, RMR;
      matrix_dense_double RART, RMRT, VRAR;
      
      matrix_dense_double ABVFZ, MBVFZ;
      matrix_dense_double AEVCZ, MEVCZ;
      
      matrix_dense_double RARloc, RMRloc;
      matrix_dense_double RARlocj, RMRlocj;
      matrix_dense_double RARlocjT, RMRlocjT;
      matrix_dense_double RAR1, RMR1;
      matrix_dense_double ABVFZ1, MBVFZ1, AEVCZ1, MEVCZ1;
      matrix_dense_double ABVFZ2, MBVFZ2, AEVCZ2, MEVCZ2;
      
      vector_seq_double work_vec;
      
      {
         /* first compute AR and MR */
         ABVFZ.Setup( nBi, mrs[myidj], kMemoryHost, true);
         MBVFZ.Setup( nBi, mrs[myidj], kMemoryHost, true);
         
         AEVCZ.Setup( nCi, mrs[myidj], kMemoryHost, true);
         MEVCZ.Setup( nCi, mrs[myidj], kMemoryHost, true);
         
         CsrMatrixMatMat( one, B.GetDiagMat(), V3, zero, ABVFZ);
         CsrMatrixMatMat( one, MB.GetDiagMat(), V3, zero, MBVFZ);
         
         CsrMatrixMatMat( one, E.GetDiagMat(), V3, zero, AEVCZ);
         CsrMatrixMatMat( one, ME.GetDiagMat(), V3, zero, MEVCZ);
         
         CsrMatrixMatMat( one, F.GetDiagMat(), Z3, one, ABVFZ);
         CsrMatrixMatMat( one, MF.GetDiagMat(), Z3, one, MBVFZ);
         
         ParallelCsrMatrixMatMat( one, C, Z3, one, AEVCZ, work_vec);
         ParallelCsrMatrixMatMat( one, MC, Z3, one, MEVCZ, work_vec);
         
         /* next compute R'AR and R'MR */
         RARloc.Setup( mrs[myidj], mr, kMemoryHost, true);
         RMRloc.Setup( mrs[myidj], mr, kMemoryHost, true);
         
         if(myid == 0)
         {
            RART.Setup( mr, mr, kMemoryHost, true);
            RMRT.Setup( mr, mr, kMemoryHost, true);
         }
         
         if(myidi == 0)
         {
            RARlocj.Setup( mrs[myidj], mr, kMemoryHost, true);
            RMRlocj.Setup( mrs[myidj], mr, kMemoryHost, true);
         }
         
         /* first apply the local matvec */
         RAR1.SetupPtr(RARloc, 0, mrdisps[myidj], mrs[myidj], mrs[myidj]);
         RMR1.SetupPtr(RMRloc, 0, mrdisps[myidj], mrs[myidj], mrs[myidj]);
         
         /* updat to local part */
         DenseMatrixMatMat( one, V3, 'T', ABVFZ, 'N', zero, RAR1);
         DenseMatrixMatMat( one, V3, 'T', MBVFZ, 'N', zero, RMR1);
         
         DenseMatrixMatMat( one, Z3, 'T', AEVCZ, 'N', one, RAR1);
         DenseMatrixMatMat( one, Z3, 'T', MEVCZ, 'N', one, RMR1);
         
         /* prepare temp buffer */
         ABVFZ1.Setup( nBi, mrs[0], kMemoryHost, true);
         MBVFZ1.Setup( nBi, mrs[0], kMemoryHost, true);
         
         AEVCZ1.Setup( nCi, mrs[0], kMemoryHost, true);
         MEVCZ1.Setup( nCi, mrs[0], kMemoryHost, true);
         
         for(int i = 1 ; i < npj ; i ++)
         {
            
            int upidj = (myidj + i) % npj;
            int downidj = (myidj - i + npj) % npj;
            
            SchurchebMpiSendRecv(ABVFZ.GetData(), nBi*mrs[myidj], upidj, myidj*upidj, 
                                 ABVFZ1.GetData(), nBi*mrs[downidj], downidj, downidj*myidj, 
                                 commj, MPI_STATUS_IGNORE);
            
            SchurchebMpiSendRecv(MBVFZ.GetData(), nBi*mrs[myidj], upidj, myidj*upidj, 
                                 MBVFZ1.GetData(), nBi*mrs[downidj], downidj, downidj*myidj, 
                                 commj, MPI_STATUS_IGNORE);
            
            SchurchebMpiSendRecv(AEVCZ.GetData(), nCi*mrs[myidj], upidj, myidj*upidj, 
                                 AEVCZ1.GetData(), nCi*mrs[downidj], downidj, downidj*myidj, 
                                 commj, MPI_STATUS_IGNORE);
            
            SchurchebMpiSendRecv(MEVCZ.GetData(), nCi*mrs[myidj], upidj, myidj*upidj, 
                                 MEVCZ1.GetData(), nCi*mrs[downidj], downidj, downidj*myidj, 
                                 commj, MPI_STATUS_IGNORE);
            
            /* first apply the local matvec */
            RAR1.SetupPtr(RARloc, 0, mrdisps[downidj], mrs[myidj], mrs[downidj]);
            RMR1.SetupPtr(RMRloc, 0, mrdisps[downidj], mrs[myidj], mrs[downidj]);
            
            ABVFZ2.SetupPtr(ABVFZ1, 0, 0, nBi, mrs[downidj]);
            AEVCZ2.SetupPtr(AEVCZ1, 0, 0, nCi, mrs[downidj]);
            MBVFZ2.SetupPtr(MBVFZ1, 0, 0, nBi, mrs[downidj]);
            MEVCZ2.SetupPtr(MEVCZ1, 0, 0, nCi, mrs[downidj]);
            
            /* updat to local part */
            DenseMatrixMatMat( one, V3, 'T', ABVFZ2, 'N', zero, RAR1);
            DenseMatrixMatMat( one, V3, 'T', MBVFZ2, 'N', zero, RMR1);
            
            DenseMatrixMatMat( one, Z3, 'T', AEVCZ2, 'N', one, RAR1);
            DenseMatrixMatMat( one, Z3, 'T', MEVCZ2, 'N', one, RMR1);
         
            
         }
         
         ABVFZ.Clear();
         MBVFZ.Clear();
         AEVCZ.Clear();
         MEVCZ.Clear();
         ABVFZ1.Clear();
         MBVFZ1.Clear();
         AEVCZ1.Clear();
         MEVCZ1.Clear();
         ABVFZ2.Clear();
         MBVFZ2.Clear();
         AEVCZ2.Clear();
         MEVCZ2.Clear();
         
         V3.Clear();
         Z3.Clear();
         
         /* communication */
         SchurchebMpiReduce( RARloc.GetData(), RARlocj.GetData(), mrs[myidj]*mr, MPI_SUM, 0, commi);
         SchurchebMpiReduce( RMRloc.GetData(), RMRlocj.GetData(), mrs[myidj]*mr, MPI_SUM, 0, commi);
         
         RARloc.Clear();
         RMRloc.Clear();
         
         if(myidi == 0)
         {
            for(int i = 0 ; i < npj ; i ++)
            {
               mrs[i] *= mr;
               mrdisps[i] *= mr;
            }
            
            DenseMatrixTransposeHostTemplate(RARlocj, RARlocjT);
            DenseMatrixTransposeHostTemplate(RMRlocj, RMRlocjT);
            
            MPI_Gatherv(RARlocjT.GetData(),
                         mrs[myidj],
                         MPI_DOUBLE,
                         RART.GetData(),
                         mrs.GetData(),
                         mrdisps.GetData(),
                         MPI_DOUBLE,
                         0,
                         commj);
            
            MPI_Gatherv(RMRlocjT.GetData(),
                         mrs[myidj],
                         MPI_DOUBLE,
                         RMRT.GetData(),
                         mrs.GetData(),
                         mrdisps.GetData(),
                         MPI_DOUBLE,
                         0,
                         commj);
            
            RARlocj.Clear();
            RMRlocj.Clear();
            RARlocjT.Clear();
            RMRlocjT.Clear();
            
            DenseMatrixTransposeHostTemplate(RART, RAR);
            DenseMatrixTransposeHostTemplate(RMRT, RMR);
            
            RART.Clear();
            RMRT.Clear();
            
         }
      }
      
      SchurchebMpiTime( MPI_COMM_WORLD, te); // End timming
      tall += te - ts;
      if(myid == 0 && print_level)
      {
         printf("Form local system time: %8.6fs\n",te-ts);
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }  
      
      SchurchebMpiTime( MPI_COMM_WORLD, ts); // Start timming
      
      /* -----------------------
       * Step 11: Eigenvalue Problem
       * ---------------------- */
      
      matrix_dense_double RARRMR;
      matrix_dense_double QS, QE;
      
      VRAR.Setup(mr, neigs, kMemoryHost, false);
      if(myid == 0)
      {
         
         int nvz = mr;
         int ldavz = RAR.GetLeadingDimension();
         int ldbvz = RMR.GetLeadingDimension();
         int ldzvz = VRAR.GetLeadingDimension();
         int info;
         
         int itypes = 1;
         char jobzs = 'V';
         char uplos = 'U';
         char ranges = 'I';
         int lworks = 8*nvz;
         
         int ils = 1;
         int ius = neigs;
         double abstols = tol_eig2;
         int ms = neigs;
         
         vector_seq_double ws, works;
         ws.Setup(nvz);
         works.Setup(lworks);
         
         vector_int iworks, ifails;
         iworks.Setup(5*nvz);
         ifails.Setup(nvz);
         
         SCHURCHEB_BLASLAPACK_DSYGVX(	&itypes,
                     &jobzs,
                     &ranges,
                     &uplos,
                     &nvz,
                     RAR.GetData(),
                     &ldavz,
                     RMR.GetData(),
                     &ldbvz,
                     NULL,
                     NULL,
                     &ils,
                     &ius,
                     &abstols,
                     &ms,
                     ws.GetData(),
                     VRAR.GetData(),
                     &ldzvz,
                     works.GetData(),
                     &lworks,
                     iworks.GetData(),
                     ifails.GetData(),
                     &info);
         
         this->_eigs_v.Setup(neigs);
         
         for(int i = 0 ; i < neigs ; i ++)
         {
            this->_eigs_v[i] = ws[i];
         }
         
      }
      
      SchurchebMpiTime( MPI_COMM_WORLD, te); // End timming
      tall += te - ts;
      if(myid == 0 && print_level)
      {
         printf("local eigen solve time: %8.6fs\n",te-ts);
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }  
      
      /* -----------------------
       * Step 12: Compute Eigenvectors
       * ---------------------- */
      
      SchurchebMpiTime( MPI_COMM_WORLD, ts); // Start timming
      
      /* print eigenvalues */
      /* we keep the projection vectors in R3, now just apply matvec to get it */
      
      /* next we can start to compute the eigenvectors */
      if(eigvec_opt)
      {
         SCHURCHEB_MPI_CALL(SchurchebMpiBcast( VRAR.GetData(), mr*neigs, 0, MPI_COMM_WORLD));
         
         this->_V_mat.Setup( dom_ptrdB[myidj+1]-dom_ptrdB[myidj]+dom_ptrdC[myidj+1]-dom_ptrdC[myidj], neigs, true);
         
         matrix_dense_double R3_ptr;
         R3_ptr.SetupPtr( R3, 0, 0, R3.GetNumRowsLocal(), mr);
         
         /* simply apply local DGEMM */
         DenseMatrixMatMat( one, R3_ptr, 'N', VRAR, 'N', zero, this->_V_mat);
         
         R3_ptr.Clear();
         
         /* get the local permutation vector */
         vector_int &perm2 = this->_perm_v_dist;
         
         perm2.Setup( dom_ptrdB[myidj+1]-dom_ptrdB[myidj]+dom_ptrdC[myidj+1]-dom_ptrdC[myidj], false);
         
         int nBj = dom_ptrdB[myidj+1]-dom_ptrdB[myidj], nCj = dom_ptrdC[myidj+1]-dom_ptrdC[myidj];
         for(int i = 0 ; i < nBj ; i ++)
         {
            perm2[i] = this->_perm_v[dom_ptrdBg[myid2]+i];
         }
         for(int i = 0 ; i < nCj ; i ++)
         {
            perm2[i+nBj] = this->_perm_v[dom_ptrdBg[np2]+dom_ptrdCg[myid2]+i];
         }
      }
      SchurchebMpiTime( MPI_COMM_WORLD, te); // End timming
      tall += te - ts;
      if(myid == 0 && print_level)
      {
         printf("Compute eigenvector and setup new parallel CSR matrix time: %8.6fs\n",te-ts);
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }  
      
      if(myid == 0 && print_level)
      {
         printf("Total time: %8.6fs\n", tall);
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      } 
      
      /* -----------------------
       * Step 13: print and write
       * ---------------------- */
      
      /* compute residual */
      if(compute_res)
      {
         if(myid == 0 && print_level)
         {
            printf("Computing resitual.\n");
         }
         /* communicate to get the global reordering */
         vector_int perm3;
         perm3.Setup(n);
         SCHURCHEB_MPI_CALL(SchurchebMpiAllgatherv( this->_perm_v_dist.GetData(), 
                        this->_perm_v_dist.GetLengthLocal(), perm3.GetData(), dom_countdg.GetData(), dom_ptrdg.GetData(), comm2));
         
         /* also bcast the eigenvalues */
         if(myid != 0)
         {
            this->_eigs_v.Setup(neigs);
         }
         SCHURCHEB_MPI_CALL(SchurchebMpiBcast( this->_eigs_v.GetData(), neigs, 0, MPI_COMM_WORLD));
         
         /* apply the permutation */          
         A.SubMatrix( perm3, perm3, kMemoryHost, App);     
         M.SubMatrix( perm3, perm3, kMemoryHost, Mpp);     
         ExtractParallelCsrSubMatrix( App, this->_A_par, dom_ptrdg.GetData(), parlog2);
         ExtractParallelCsrSubMatrix( Mpp, this->_M_par, dom_ptrdg.GetData(), parlog2);
         App.Clear();
         Mpp.Clear();
         
         /* compute AV and MV */
         double normev2;
         vector_seq_double working;
         vector_par_double vav, vmv;
         matrix_dense_double AV, MV;
         AV.Setup(this->_V_mat.GetNumRowsLocal(),this->_V_mat.GetNumColsLocal(), true);
         MV.Setup(this->_V_mat.GetNumRowsLocal(),this->_V_mat.GetNumColsLocal(), true);
         ParallelCsrMatrixMatMat( one, this->_A_par, this->_V_mat, zero, AV, working);
         ParallelCsrMatrixMatMat( one, this->_M_par, this->_V_mat, zero, MV, working);
         working.Clear();
         
         vav.SetupPtrStr(this->_A_par);
         vmv.SetupPtrStr(this->_M_par);
         this->_res_v.Setup(neigs);
         for(int i = 0 ; i < neigs ; i ++)
         {
            // vav = Av_i - l*Mv_i
            vav.UpdatePtr(&(AV(0,i)), kMemoryHost);
            vmv.UpdatePtr(&(MV(0,i)), kMemoryHost);
            vav.Axpy( -this->_eigs_v[i], vmv);
            vav.Norm2(this->_res_v[i]);
         }
         AV.Clear();
         MV.Clear();
      }
      
      /* print eigenvalues */
      if(myid == 0)
      {
         int neigs2 = 0;
         if(compute_res)
         {
            printf("     approx             |   residual 2-norm\n");
            for(int i = 0; i < neigs ; i ++)
            {
               if(this->_eigs_v[i] < a)
               {
                  continue;
               }
               else if(this->_eigs_v[i] > b)
               {
                  break;
               }
               printf("%24.20f|%24.20e\n", this->_eigs_v[i], this->_res_v[i]);
               neigs2++;
            }
         }
         else
         {
            printf("     approx\n");
            for(int i = 0; i < neigs ; i ++)
            {
               if(this->_eigs_v[i] < a)
               {
                  continue;
               }
               else if(this->_eigs_v[i] > b)
               {
                  break;
               }
               printf("%24.20f\n", this->_eigs_v[i]);
               neigs2++;
            }
         }
         if(neigs2 > 1)
         {
            printf("Found %d eigenvalues in[%8.6g,%8.6g]\n",neigs2,a,b);
         }
         else if(neigs2 == 1)
         {
            printf("Found %d eigenvalue in[%8.6g,%8.6g]\n",neigs2,a,b);
         }
         else
         {
            printf("Found no eigenvalue in[%8.6g,%8.6g]\n",a,b);
         }
      }
      
      if(eigvec_opt > 1)
      {
         if(myid == 0 && print_level)
         {
            printf("Gathering eigenvector to the first MPI process.\n");
         }
         /* need to gather all vec to single MPI process */
         
         matrix_dense_double V_finalT, V_finalc, V_finalgT;
         
         DenseMatrixTransposeHostTemplate( this->_V_mat, V_finalT);
         
         /* first along row direction */
         
         int nsendb = (dom_ptrdB[myidj+1]-dom_ptrdB[myidj])*neigs;
         int nsendc = (dom_ptrdC[myidj+1]-dom_ptrdC[myidj])*neigs;
         
         vector_int nsendbs, nsendcs, ndispb, ndispc;
         
         nsendbs.Setup(npj);
         nsendcs.Setup(npj);
         ndispb.Setup(npj+1);
         ndispc.Setup(npj+1);
         
         SchurchebMpiGather( &nsendb, 1, nsendbs.GetData(), 0, commj);
         SchurchebMpiGather( &nsendc, 1, nsendcs.GetData(), 0, commj);
         
         ndispb[0] = 0;
         ndispc[0] = 0;
         for(int i = 0 ; i < npj ; i ++)
         {
            ndispb[i+1] = ndispb[i] + nsendbs[i];
            ndispc[i+1] = ndispc[i] + nsendcs[i];
         }
         
         if(myidj == 0)
         {
            V_finalc.Setup(neigs,(ndispb[npj]+ndispc[npj])/neigs,kMemoryHost,false);
         }
         
         MPI_Gatherv( V_finalT.GetData(),
                nsendb,
                MPI_DOUBLE,
                V_finalc.GetData(),
                nsendbs.GetData(),
                ndispb.GetData(),
                MPI_DOUBLE,
                0,
                commj);
         
         MPI_Gatherv( V_finalT.GetData()+nsendb,
                nsendc,
                MPI_DOUBLE,
                V_finalc.GetData()+ndispb[npj],
                nsendcs.GetData(),
                ndispc.GetData(),
                MPI_DOUBLE,
                0,
                commj);
         
         /* next along row direction */
         
         if(myidj == 0)
         {
            if(myid == 0)
            {
               V_finalgT.Setup(neigs,n,kMemoryHost,false);
            }
               
            int nsendb2 = ndispb[npj];
            int nsendc2 = ndispc[npj];
               
            vector_int nsendbs2, nsendcs2, ndispb2, ndispc2;
            
            nsendbs2.Setup(npi);
            nsendcs2.Setup(npi);
            ndispb2.Setup(npi+1);
            ndispc2.Setup(npi+1);
            
            SchurchebMpiGather( &nsendb2, 1, nsendbs2.GetData(), 0, commi);
            SchurchebMpiGather( &nsendc2, 1, nsendcs2.GetData(), 0, commi);
            
            ndispb2[0] = 0;
            ndispc2[0] = 0;
            for(int i = 0 ; i < npi ; i ++)
            {
               ndispb2[i+1] = ndispb2[i] + nsendbs2[i];
               ndispc2[i+1] = ndispc2[i] + nsendcs2[i];
            }
            
            MPI_Gatherv( V_finalc.GetData(),
                   nsendb2,
                   MPI_DOUBLE,
                   V_finalgT.GetData(),
                   nsendbs2.GetData(),
                   ndispb2.GetData(),
                   MPI_DOUBLE,
                   0,
                   commi);
            
            MPI_Gatherv( V_finalc.GetData()+nsendb2,
                   nsendc2,
                   MPI_DOUBLE,
                   V_finalgT.GetData()+ndispb2[npi],
                   nsendcs2.GetData(),
                   ndispc2.GetData(),
                   MPI_DOUBLE,
                   0,
                   commi);
            
            if(myid == 0)
            {
               DenseMatrixTransposeHostTemplate(V_finalgT, this->_V_mat_seq);
            }
               
            nsendbs2.Clear();
            nsendcs2.Clear();
            ndispb2.Clear();
            ndispc2.Clear();
         }
         
         nsendbs.Clear();
         nsendcs.Clear();
         ndispb.Clear();
         ndispc.Clear();
         V_finalc.Clear();
         V_finalT.Clear();
         V_finalgT.Clear();
      }
      /* -------------------
       * Step -1: Clean
       * ------------------- */
      
      B.Clear();
      E.Clear();
      F.Clear();
      C.Clear();
      M.Clear();
      MB.Clear();
      ME.Clear();
      MF.Clear();
      MC.Clear();
      
      RAR.Clear();
      RMR.Clear();
      VRAR.Clear();
      R3.Clear();
      
      this->_test_shift.Clear();
      
      parlog2.Clear();
      parlog_local.Clear();
      parlog_local2.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   vector_seq_double& SchurChebClass::GetResiduals()
   {
      return this->_res_v;
   }
   
   matrix_csr_par_double& SchurChebClass::GetParA()
   {
      return this->_A_par;
   }
   
   matrix_csr_par_double& SchurChebClass::GetParM()
   {
      return this->_M_par;
   }
   
   vector_seq_double& SchurChebClass::GetEigenValues()
   {
      return this->_eigs_v;
   }
   
   vector_int& SchurChebClass::GetPermutation()
   {
      return this->_perm_v;
   }
   
   matrix_dense_double& SchurChebClass::GetEigenVectors()
   {
      return this->_V_mat_seq;
   }
   
   vector_int& SchurChebClass::GetDistPermutation()
   {
      return this->_perm_v_dist;
   }
   
   matrix_dense_double& SchurChebClass::GetDistEigenVectors()
   {
      return this->_V_mat;
   }
      
}

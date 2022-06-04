/**
 * @file test_schur.cpp
 * @brief Main test driver for schurcheb.
 */

#include "schurcheb.hpp"
#include <iostream>

using namespace std;
using namespace schurcheb;

/**
 * @brief   Print usage.
 * @details Print usage.
 * @return       Return error message.
 */
int print_usage();

/**
 * @brief   Write matrix data to file. Used to store the eigenvectors Amat(perm:) to file.
 * @details Write matrix data to file. Used to store the eigenvectors Amat(perm:) to file.
 * @return       Return error message.
 */
int WriteToDisk( matrix_dense_double Amat, vector_int &perm, const char *datafilename);

/* main function */
int main(int argc, char *argv[]) 
{
   SchurchebInit( &argc, &argv);

   int err = 0;

   double ts, te, tall = .0, t_set = .0, t_ar = .0, t_vmv = .0, t_armv = .0;

   double EPSILON = std::numeric_limits<double>::epsilon();
   double tol_eig;
   double tol_eig2;
   
   int solve_opt;
   bool lan = false;
   
   int neigs, nnodes;
   
   bool gen = false;
   char amat[1024], mmat[1024], eigfile[1024], vecfile[1024];
   bool writeeig = false;
   bool writevec = false;
   int eigvec_opt = 1;
   
   int local_solver = 0;
   
   int n, nx = 0, ny = 0, nz = 0, n1, n2, ndom;
   int msteps, niter;
   int orth_nB;
   bool chol = false;
   double a, b;
   int ns[3];
   double ab[2];
      
   int print_level;
   bool generalize = false;
   
   int np, myid;
   
   np = parallel_log::_gsize;
   myid = parallel_log::_grank;
   
   /* -------------------
    * Step 0: Init 
    * ------------------- */
   
   /* Step 0.0: read inputs and init */
   if(SchurchebReadInputArg( "help", argc, argv)) // Help
   {
      if(myid == 0)
      {
         print_usage();
      }
      SchurchebFinalize();
      return 0;
   }
   
   if(SchurchebReadInputArg( "chol", argc, argv)) // Help
   {
      chol = true;
   }
   
   if(!SchurchebReadInputArg( "local_solver", 1, &local_solver, argc, argv))
   {
      /* pardiso definitly is the better local solver */
#ifdef SCHURCHEB_MKL
      local_solver = 1;
#else
      local_solver = 0;
#endif
   }
   
   if(SchurchebReadInputArg( "gen", argc, argv))
   {
      gen = true;
   }
   
   if(gen)
   {
      if(!SchurchebReadInputArg("A",amat,argc,argv))
      {
         if(myid == 0)
         {
            printf("Error: Please give A matrix.");
            print_usage();
         }
         SchurchebFinalize();
         return 0;
      }
      if(SchurchebReadInputArg("M",mmat,argc,argv))
      {
         generalize = true;
      }
   }
   else
   {
      if(!SchurchebReadInputArg( "n", 3, ns, argc, argv)) // Size
      {
         /* no input, use default */
         nx = 8;
         ny = 8;
         nz = 1;
      }
      else
      {
         nx = ns[0];
         ny = ns[1];
         nz = ns[2];
      }
   }
   
   if(!SchurchebReadInputArg( "nB", 1, &orth_nB, argc, argv)) // Size
   {
      /* no input, use default */
      orth_nB = 1;
   }
   
   if(!SchurchebReadInputArg( "ab", 2, ab, argc, argv)) // intival
   {
      /* no input, use default */
      a = 0;
      b = 1;
   }
   else
   {
      a = ab[0];
      b = ab[1];
   }
   
   if(!SchurchebReadInputArg( "neig", 1, &neigs, argc, argv)) // intival
   {
      /* no input, use default */
      neigs = 10;
   }
   
   if(b <= a)
   {
      if(myid == 0)
      {
         cout<<"Error: Invalid intival."<<endl;
      }
      SchurchebFinalize();
      return 0;
   }
   
   if(!SchurchebReadInputArg( "ncol", 1, &n1, argc, argv)) // size of the 1nd dimension 
   {
      /* no input, use default */
      n1 = 1;
   }
   
   if( np % n1 != 0 )
   {
      if(myid == 0)
      {
         cout<<"Error: Invalid 2D MPI assignment (np %% ncol != 0)."<<endl;
      }
      SchurchebFinalize();
      return 0;
   }
   
   n2 = np / n1;
   
   if(!SchurchebReadInputArg( "ndom", 1, &ndom, argc, argv)) // size of number of subdomains
   {
      /* no input, use default */
      ndom = SchurchebMax( n2, 2);
   }
   
   if( ndom < n2 || ndom < 2)
   {
      if(myid == 0)
      {
         cout<<"Error: Too few number of subdomains."<<endl;
      }
      SchurchebFinalize();
      return 0;
   }
   
   if(!SchurchebReadInputArg( "nnode", 1, &nnodes, argc, argv)) // intival
   {
      /* no input, use default */
      nnodes = SchurchebMax( n1, 8);
   }
   
   if(!SchurchebReadInputArg( "m", 1, &msteps, argc, argv)) // intival
   {
      /* no input, use default */
      msteps = SchurchebMax( neigs*2, 300);
   }
   
   if(!SchurchebReadInputArg( "niter", 1, &niter, argc, argv)) // intival
   {
      /* no input, use default */
      niter = 100;
   }
   
   if(!SchurchebReadInputArg( "tol_eig", 1, &tol_eig, argc, argv)) // intival
   {
      /* no input, use default, the cheb-eig typically need not to be very accurate */
      tol_eig = 1e-08;
   }
   
   if(!SchurchebReadInputArg( "tol_eig2", 1, &tol_eig2, argc, argv)) // intival
   {
      /* no input, use default */
      tol_eig2 = EPSILON;
   }
   
   if(SchurchebReadInputArg("write_eig", eigfile, argc, argv))
   {
      writeeig = true;
   }
   
   if(SchurchebReadInputArg("write_vec", vecfile, argc, argv))
   {
      writevec = true;
   }
   
   if(writevec)
   {
      eigvec_opt = 2;
   }
   
   if(SchurchebReadInputArg( "lan", argc, argv)) // Help
   {
#ifdef SCHURCHEB_PARPACK
      lan = true;
#endif
   }
   
   if( nnodes < n1 )
   {
      if(myid == 0)
      {
         cout<<"Error: No enough cheb nodes."<<endl;
      }
      SchurchebFinalize();
      return 0;
   }
   
   if(!SchurchebReadInputArg( "solve_opt", 1, &solve_opt, argc, argv)) // intival
   {
      /* no input, use default */
      solve_opt = 0;
   }
   
   if(!SchurchebReadInputArg( "print_level", 1, &print_level, argc, argv)) // intival
   {
      /* no input, use default */
      print_level = 1;
   }
   
   /* Step 0.1: print problem settings */
   if(myid == 0)
   {
      SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      if(gen)
      {
         cout<<"A mat:                 "<<amat<<endl;
         if(generalize)
         {
            cout<<"M mat:                 "<<mmat<<endl;
         }
      }
      else
      {
         cout<<"Laplacian size:        "<<nx<<" * "<<ny<<" * "<<nz<<endl;
      }
      SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
   }
   
   /* -------------------
    * Step 1: Create Matrix
    * ------------------- */
   
   matrix_csr_double       A, M, App, Mpp;
   
   /* Step 1.0: create test problem */
   SchurchebMpiTime( MPI_COMM_WORLD, ts); // Start timming
   /* compute problem size */
   if(gen)
   {
      A.ReadFromMMFile(amat,1);
      n = A.GetNumRowsLocal();
      if(generalize)
      {
         M.ReadFromMMFile(mmat,1);
      }
      else
      {
         M.Setup( n, n, false);
         M.Eye();
      }
   }
   else
   {
      n = nx*ny*nz;
      A.Laplacian(nx,ny,nz,0,0,0,0.0); // Create Laplacian
      M.Setup( n, n, false);
      M.Eye();
   }
   SchurchebMpiTime( MPI_COMM_WORLD, te); // End timming
   
   if(myid == 0 && print_level)
   {
      printf("Create test problem time: %8.6fs\n",te-ts);
      SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
   }  
   
   /* -------------------
    * Step 2: Solve
    * ------------------- */
   
   schurcheb_double eig_solver;
   
   err = eig_solver.Setup( A, M, true, false, neigs, a, b, ndom, nnodes, n1, lan, 
                        msteps, niter, tol_eig, tol_eig2, 
                        orth_nB, chol, local_solver, eigvec_opt, true, print_level);
   
   if(err)
   {
      if(myid == 0)
      {
         printf("Error occurs.\n");
      }
      A.Clear();
      M.Clear();
      eig_solver.Clear();
      
      SchurchebFinalize();
   }
   
   /* -----------------------
    * Step 3: print and write
    * ---------------------- */
   
   /* get the eigenvalue vector */
   vector_seq_double &wr = eig_solver.GetEigenValues();
   vector_int &perm = eig_solver.GetPermutation();
   matrix_dense_double &V_mat = eig_solver.GetEigenVectors();
   
   if(myid == 0)
   {
      if(writeeig)
      {
         char tempsolname[2048];
         snprintf( tempsolname, 2048, "./%s%s", eigfile, ".out" );
         vector_seq_double eigs_write;
         
         eigs_write.Setup(neigs);
         for(int i = 0; i < neigs ; i ++)
         {
            eigs_write[i] = wr[i];
         }
         
         eigs_write.WriteToDisk(tempsolname);
         eigs_write.Clear();
      }
      
      if(writevec)
      {
         char tempsolname[2048];
         snprintf( tempsolname, 2048, "./%s%s", vecfile, ".out" );
         
         WriteToDisk( V_mat ,perm, tempsolname);
      }
   }
   
   /* -------------------
    * Step -1: Clean
    * ------------------- */
   
   A.Clear();
   M.Clear();
   eig_solver.Clear();
   
   SchurchebFinalize();
   
   return 0;
}



/**
 * @brief   Print usage.
 * @details Print usage.
 * @return       Return error message.
 */
int print_usage()
{
   /* Print usage */
   printf("Read mtx from file:    -gen\n");
   printf("Mat A:                 -A          [str]\n");
   printf("Mat M:                 -M          [str]\n");
   printf("                          If M not set, solve eigenvalue problem A*v = lambda*v.\n");
   printf("Laplacian Size:        -n          [int] [int] [int]\n");
   printf("                          If -gen not used, solve standard eigenvalue problem using 5-pt/7-pt Laplacian.\n");
   printf("Num Doms:              -ndom       [int]\n");
   printf("2D MPI cols:           -ncol       [int].\n");
   printf("                          Note: ncol should <= nnode.\n");
   printf("Orth blocl size:       -nB         [int]\n");
   printf("Use chol for orth:        -chol\n");
   printf("Num Eigs wanted        -neig       [int]\n");
   printf("Num Cheb nodes         -nnode      [int]\n");
   printf("                          Note: nnode should >= ncol.\n");
   printf("Eig inteval boundary:  -ab         [double] [double]\n");
   printf("Kryrov Dimension:      -m          [int]\n");
   printf("Max num restarts:      -niter      [int]\n");
   printf("Eig tolerance:         -tol_eig    [double]\n");
   printf("Local eig tolerance:   -tol_eig2   [double]\n");
   printf("Lanczos no full-orth   -lan\n");
   printf("print level:           -print_level  [int]\n");
   printf("                          0. minimal output.\n");
   printf("                          1. show timing results.\n");
   printf("                          2. show more timing results (might influence performance).\n");
   printf("                          3. also print parallel direct solver info and memory usage.\n");
   printf("Solver for B(local):   -local_solver  [int]\n");
   printf("                          0. LU.\n");
   printf("                          1. MKL pardiso LDL^T. Building with MKL required.\n");
   printf("Write to file:         -write_eig     [str]\n");
   printf("                          write eigenvalues to file [str.out].\n");
   printf("Eigenvalues only:      -no_vec\n");
   printf("Write to file:         -write_vec     [str]\n");
   printf("                          write eigenvectors to file [str.out].\n");
   return 0;
}

/* Function for cheb-eig, write to file */
int WriteToDisk( matrix_dense_double Amat, vector_int &perm, const char *datafilename)
{
   int i, j;
   
   FILE *fdata;
   
   if ((fdata = fopen(datafilename, "w")) == NULL)
   {
      printf("Can't open file.\n");
      return SCHURCHEB_ERROR_IO_ERROR;
   }
   
   vector_int rperm;
   int n = Amat.GetNumRowsLocal();
   rperm.Setup(n);
   for(i = 0 ; i < n ; i ++)
   {
      rperm[perm[i]] = i;
   }
   
   for(i = 0 ; i < n ; i ++)
   {
      for(j = 0 ; j < Amat.GetNumColsLocal() ; j ++)
      {
         fprintf(fdata, "%24.20f ", Amat(rperm[i],j));
      }
      fprintf(fdata, "\n");
   }
   fclose(fdata);
   
   
   return SCHURCHEB_SUCCESS;
}


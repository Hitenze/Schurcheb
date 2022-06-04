/**
 * @file test_parpack.cpp
 * @brief Test driver for comparison
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

/* main function */
int main(int argc, char *argv[]) 
{
#ifdef SCHURCHEB_PARPACK
   SchurchebInit( &argc, &argv);

   double ts, te;
   
   double EPSILON = std::numeric_limits<double>::epsilon();
   double tol_eig;
   //double tol_orth;
   
   int neigs;
   
   bool gen = false;
   char amat[1024], mmat[1024];
   
   int nx = 0, ny = 0, nz = 0;
   int msteps, niter;
   int ns[3];
   
   int print_level;
   bool show_exact = true;
   bool generalize = false;
   
   int direct_solver = 0;
   
   bool lan = false;
   int solve_opt;
   
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
   
   if(SchurchebReadInputArg( "lan", argc, argv)) // Help
   {
      lan = true;
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
   
   if(!SchurchebReadInputArg( "neig", 1, &neigs, argc, argv)) // intival
   {
      /* no input, use default */
      neigs = 10;
   }
   
   if(!SchurchebReadInputArg( "m", 1, &msteps, argc, argv)) // intival
   {
      /* no input, use default */
      msteps = 300;
   }
   
   if(!SchurchebReadInputArg( "niter", 1, &niter, argc, argv)) // intival
   {
      /* no input, use default */
      niter = 100;
   }
   
   if(!SchurchebReadInputArg( "tol_eig", 1, &tol_eig, argc, argv)) // intival
   {
      /* no input, use default */
      tol_eig = EPSILON;
   }
   
   if(!SchurchebReadInputArg( "print_level", 1, &print_level, argc, argv)) // intival
   {
      /* no input, use default */
      print_level = 1;
   }
   
   if(!SchurchebReadInputArg( "solve_opt", 1, &solve_opt, argc, argv)) // intival
   {
      /* no input, use default */
      solve_opt = 0;
   }
   
   if(!SchurchebReadInputArg( "direct_solver", 1, &direct_solver, argc, argv)) // intival
   {
      /* no input, use default */
      direct_solver = 2;

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
         cout<<"Problem size:          "<<nx<<" * "<<ny<<" * "<<nz<<endl;
      }
      cout<<"Number of Eigs         "<<neigs<<endl;
      cout<<"Eigenvalue res tol:    "<<tol_eig<<endl;
      cout<<"Krylov dimension:      "<<msteps<<endl;
      cout<<"Max number restarts:   "<<niter<<endl;
      cout<<"No full orth?          "<<lan<<endl;
      cout<<"Solve option:          "<<solve_opt<<endl;
      if(direct_solver == 0)
      {
         cout<<"Use superLU_dist parallel direct solver."<<endl;
      }
      else if(direct_solver == 1)
      {
         cout<<"Use MUMPS parallel direct solver."<<endl;
      }
      else
      {
         cout<<"No parallel direct soler used."<<endl;
      }
      cout<<"Print level:           "<<print_level<<endl;
      SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
   }
   
   /* -------------------
    * Step 1: Create Matrix
    * ------------------- */
   
   vector_seq_double exact_eigs;
   
   /* compute local eig */
   /* outer loop generalized? */
   {
      int nmvs = 0;
      int nsvs = 0;
      double tmvs = .0;
      double tsvs = .0;
      
      bool timing = false;
      if(print_level > 1)
      {
         timing = true;
      }
      
      if(generalize)
      {
         matrix_csr_par_double parA, parM;
         parallel_log parlog;
         parA.ReadFromSingleMMFile( amat, 1, parlog);
         parM.ReadFromSingleMMFile( mmat, 1, parlog);
         
         SchurchebMpiTime( MPI_COMM_WORLD, ts);
         switch(solve_opt)
         {
            case 0:
            {
               /* treat as generalized eigenvalue problem */
               
               dsolver_par_double invM;
               if(print_level > 2)
               {
                  invM.SetPrintLevel(1);
               }
               else
               {
                  invM.SetPrintLevel(0);
               }
            
               invM.SetSolveOption(direct_solver);
               if(invM._solve_opt == kParallelDsolverOptionNo)
               {
                  SCHURCHEB_WARNING("Warning: selected parallel direct solver not linked. Solve exact solution failed");
                  show_exact = false;
                  break;
               }
               invM.Setup(parM);
            
               matrix_dense_double VA;
               
               int nA = parA.GetNumRowsGlobal();
               
               int rank = SchurchebMin( neigs, nA); // keep how many of them
               int rank2 = rank;
               int lr_m = SchurchebMin( msteps, nA); // at most 300 its
               
               /* apply Lanczos */
               
               vector_seq_double dr, di;
               if(lan)
               {
                  char aropt[2];
                  aropt[0] = 'L';
                  aropt[1] = 'R';
                  
                  ArnoldiMatrixClass<vector_par_double, double> &temp_mat = invM;
                  ArpackArnoldi_inv<vector_par_double>( parM, parA, temp_mat, lr_m, niter, rank2, aropt, lan,
                           tol_eig, VA, dr, di, timing, nmvs, tmvs, nsvs, tsvs, parlog);
               }
               else
               {
                  char aropt[2];
                  aropt[0] = 'S';
                  aropt[1] = 'R';
                  
                  ArnoldiMatrixClass<vector_par_double, double> &temp_mat = invM;
                  ArpackArnoldi<vector_par_double>( parA, parM, temp_mat, lr_m, niter, rank2, aropt, lan,
                           tol_eig, VA, dr, di, timing, nmvs, tmvs, nsvs, tsvs, parlog);
               }

               if(rank2 < rank)
               {
                  /* faile to comput all */
                  if(myid == 0)
                  {
                     printf("Fail to capture all eigenvalues, please increase niter or m2.\n");
                  }
                  
                  SchurchebFinalize();
                  return -1;
                  
               }      
               
               VA.Clear();
               
               invM.Clear();
                 
               exact_eigs.Setup(rank);
               if(lan)
               {
                  for(int i = 0 ; i < rank ; i ++)
                  {
                     exact_eigs[i] = 1.0/dr[i];
                  }
               }
               else
               {
                  for(int i = 0 ; i < rank ; i ++)
                  {
                     exact_eigs[i] = dr[i];
                  }
               }
               
               exact_eigs.Sort(true);
            
               break;
            }
            case 1:
            {
               /* treat as generalized eigenvalue problem, shift and invert */
               
               dsolver_par_double invA;
               if(print_level > 2)
               {
                  invA.SetPrintLevel(1);
               }
               else
               {
                  invA.SetPrintLevel(0);
               }
            
               invA.SetSolveOption(direct_solver);
               if(invA._solve_opt == kParallelDsolverOptionNo)
               {
                  SCHURCHEB_WARNING("Warning: selected parallel direct solver not linked. Solve exact solution failed");
                  show_exact = false;
                  break;
               }
               invA.Setup(parA);
            
               matrix_dense_double VA;
               
               int nA = parA.GetNumRowsGlobal();
               
               int rank = SchurchebMin( neigs, nA); // keep how many of them
               int rank2 = rank;
               int lr_m = SchurchebMin( msteps, nA); // at most 300 its
               
               /* apply Krylov */
               
               char aropt[2];
               aropt[0] = 'S';
               aropt[1] = 'R';
               
               vector_seq_double dr, di;
               ArnoldiMatrixClass<vector_par_double, double> &temp_mat = invA;
               ArpackArnoldi_inv<vector_par_double>( parA, parM, temp_mat, lr_m, niter, rank2, aropt, lan,
                        tol_eig, VA, dr, di, timing, nmvs, tmvs, nsvs, tsvs, parlog);

               if(rank2 < rank)
               {
                  /* faile to comput all */
                  if(myid == 0)
                  {
                     printf("Fail to capture all eigenvalues, please increase niter or m2.\n");
                  }
                  
                  SchurchebFinalize();
                  return -1;
                  
               }      
               
               VA.Clear();
               
               invA.Clear();
                     
               exact_eigs.Setup(rank);
               
               for(int i = 0 ; i < rank ; i ++)
               {
                  exact_eigs[i] = dr[i];
               }
               
               exact_eigs.Sort(true);
            
               break;
            }
            case 2:
            {
               /* treat as standard eigenvalue problem */
               
               dsolver2_par_double invMA;
               if(print_level > 2)
               {
                  invMA.SetPrintLevel(1);
               }
               else
               {
                  invMA.SetPrintLevel(0);
               }
            
               invMA.SetSolveOption(direct_solver);
               if(invMA._solve_opt == kParallelDsolverOptionNo)
               {
                  SCHURCHEB_WARNING("Warning: selected parallel direct solver not linked. Solve exact solution failed");
                  show_exact = false;
                  break;
               }
               invMA.Setup(parM,parA);
            
               matrix_dense_double VA;
               
               int nA = parA.GetNumRowsGlobal();
               
               int rank = SchurchebMin( neigs, nA); // keep how many of them
               int rank2 = rank;
               int lr_m = SchurchebMin( msteps, nA); // at most 300 its
               
               /* apply Lanczos */
               
               char aropt[2];
               aropt[0] = 'S';
               aropt[1] = 'R';
               
               vector_seq_double dr, di;
               ArnoldiMatrixClass<vector_par_double, double> &temp_mat = invMA;
               
               /* non symmetric, no lanc, must full orthgonization */
               ArpackArnoldi<vector_par_double>( temp_mat, lr_m, niter, rank2, aropt, false,
                        tol_eig, VA, dr, di, timing, nmvs, tmvs, parlog);
               if(rank2 < rank)
               {
                  /* faile to comput all */
                  if(myid == 0)
                  {
                     printf("Fail to capture all eigenvalues, please increase niter or m2.\n");
                  }
                  
                  SchurchebFinalize();
                  return -1;
                  
               }      
               
               VA.Clear();
               
               invMA.Clear();
                 
               exact_eigs.Setup(rank);
               
               for(int i = 0 ; i < rank ; i ++)
               {
                  exact_eigs[i] = dr[i];
               }
               
               exact_eigs.Sort(true);
            
               break;
            }
            case 3:
            {
               /* treat as standard eigenvalue problem */
               
               dsolver2_par_double invAM;
               if(print_level > 2)
               {
                  invAM.SetPrintLevel(1);
               }
               else
               {
                  invAM.SetPrintLevel(0);
               }
            
               invAM.SetSolveOption(direct_solver);
               if(invAM._solve_opt == kParallelDsolverOptionNo)
               {
                  SCHURCHEB_WARNING("Warning: selected parallel direct solver not linked. Solve exact solution failed");
                  show_exact = false;
                  break;
               }
               invAM.Setup(parA,parM);
            
               matrix_dense_double VA;
               
               int nA = parA.GetNumRowsGlobal();
               
               int rank = SchurchebMin( neigs, nA); // keep how many of them
               int rank2 = rank;
               int lr_m = SchurchebMin( msteps, nA); // at most 300 its
               
               /* apply Lanczos */
               
               char aropt[2];
               aropt[0] = 'L';
               aropt[1] = 'R';
               
               vector_seq_double dr, di;
               ArnoldiMatrixClass<vector_par_double, double> &temp_mat = invAM;
               
               /* non symmetric, no lanc, must full orthgonization */
               ArpackArnoldi<vector_par_double>( temp_mat, lr_m, niter, rank2, aropt, false,
                        tol_eig, VA, dr, di, timing, nmvs, tmvs, parlog);
               if(rank2 < rank)
               {
                  /* faile to comput all */
                  if(myid == 0)
                  {
                     printf("Fail to capture all eigenvalues, please increase niter or m2.\n");
                  }
                  
                  SchurchebFinalize();
                  return -1;
                  
               }      
               
               VA.Clear();
               
               invAM.Clear();
               
               exact_eigs.Setup(rank);
               
               for(int i = 0 ; i < rank ; i ++)
               {
                  exact_eigs[i] = 1.0/dr[i];
               }
               
               exact_eigs.Sort(true);
            
               break;
            }
            default:
            {
               SCHURCHEB_WARNING("Invalid solve option.");
               show_exact = false;
            }
         }
         SchurchebMpiTime( MPI_COMM_WORLD, te);
         
         if(myid == 0)
         {
            if(print_level > 1)
            {
               printf("Exact Lanczos mvs: %d, time: %8.6fs\n",nmvs,tmvs);
               printf("Exact Lanczos svs: %d, time: %8.6fs\n",nsvs,tsvs);
            }
            else
            {
               printf("Exact Lanczos mvs: %d\n",nmvs);
               printf("Exact Lanczos svs: %d\n",nsvs);
            }
         }
         
         parA.Clear();
         parM.Clear();
         
      }/* end of generalized eigs */
      else
      {
         matrix_csr_par_double parA;
         parallel_log parlog;
         
         if(gen)
         {
            parA.ReadFromSingleMMFile( amat, 1, parlog);
         }
         else
         {
            if(np <= nz)
            {
               parA.Laplacian( nx, ny, nz, 1, 1, np, 0, 0, 0, 0, parlog, false);
            }
            else
            {  
               int dy = 2;
               while(np/dy > nz)
               {
                  dy *= 2;
               }
               parA.Laplacian( nx, ny, nz, 1, dy, np/dy, 0, 0, 0, 0, parlog, false);
            }
         }
         
         SchurchebMpiTime( MPI_COMM_WORLD, ts);
         switch(solve_opt)
         {
            case 0:
            {
               
               matrix_dense_double VA;
               
               int nA = parA.GetNumRowsGlobal();
               
               int rank = SchurchebMin( neigs, nA); // keep how many of them
               int rank2 = rank;
               int lr_m = SchurchebMin( msteps, nA); // at most 300 its
               
               /* apply Krylov */
               
               char aropt[2];
               aropt[0] = 'S';
               aropt[1] = 'R';
               
               vector_seq_double dr, di;
               ArpackArnoldi<vector_par_double>( parA, lr_m, niter, rank2, aropt, lan,
                        tol_eig, VA, dr, di, timing, nmvs, tmvs, parlog);

               if(rank2 < rank)
               {
                  /* faile to comput all */
                  if(myid == 0)
                  {
                     printf("Fail to capture all eigenvalues, please increase niter or m2.\n");
                  }
                  
                  SchurchebFinalize();
                  return -1;
                  
               }  
               
               VA.Clear();
                     
               exact_eigs.Setup(rank);
               
               for(int i = 0 ; i < rank ; i ++)
               {
                  exact_eigs[i] = dr[i];
               }
               
               exact_eigs.Sort(true);
               
               break;
            }
            case 1:
            {
               
               dsolver_par_double invA;
               if(print_level > 2)
               {
                  invA.SetPrintLevel(1);
               }
               else
               {
                  invA.SetPrintLevel(0);
               }
               
               invA.SetSolveOption(direct_solver);
               if(invA._solve_opt == kParallelDsolverOptionNo)
               {
                  SCHURCHEB_WARNING("Warning: selected parallel direct solver not linked. Solve exact solution failed");
                  show_exact = false;
                  break;
               }
               invA.Setup(parA);
            
               matrix_dense_double VA;
               
               int nA = parA.GetNumRowsGlobal();
               
               int rank = SchurchebMin( neigs, nA); // keep how many of them
               int rank2 = rank;
               int lr_m = SchurchebMin( msteps, nA); // at most 300 its
               
               /* apply Krylov */
               
               char aropt[2];
               aropt[0] = 'L';
               aropt[1] = 'R';
               
               vector_seq_double dr, di;
               ArnoldiMatrixClass<vector_par_double, double> &temp_mat = invA;
               ArpackArnoldi<vector_par_double>( temp_mat, lr_m, niter, rank2, aropt, lan,
                        tol_eig, VA, dr, di, timing, nmvs, tmvs, parlog);
                     

               if(rank2 < rank)
               {
                  /* faile to comput all */
                  if(myid == 0)
                  {
                     printf("Fail to capture all eigenvalues, please increase niter or m2.\n");
                  }
                  
                  SchurchebFinalize();
                  return -1;
                  
               }
               
               invA.Clear();
               
               VA.Clear();
               
               exact_eigs.Setup(rank);
               
               for(int i = 0 ; i < rank ; i ++)
               {
                  exact_eigs[i] = 1.0/dr[i];
               }
               
               exact_eigs.Sort(true);
               
               break;
            }
            default:
            {
               SCHURCHEB_WARNING("Invalid solve option.");
               show_exact = false;
            }
         }
         SchurchebMpiTime( MPI_COMM_WORLD, te);
         
         if(myid == 0)
         {
            if(print_level > 1)
            {
               printf("Exact Lanczos mvs: %d, time: %8.6fs\n",nmvs, tmvs);
            }
            else
            {
               printf("Exact Lanczos mvs: %d\n",nmvs);
            }
         }
               
         parA.Clear();
      }/* end of standard eigs */
      
   }
   
   if(myid == 0 && print_level)
   {
      printf("Krylov Time (If compute exact eigs): %8.6fs\n",te-ts);
      SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
   }
   
   if(show_exact && myid == 0)
   {
      printf("     approx     \n");
      for(int i = 0; i < neigs ; i ++)
      {
         printf("%24.20f\n",exact_eigs[i]);
      }
   }
   
   /* -------------------
    * Step -1: Clean
    * ------------------- */
   
   SchurchebFinalize();
   
   return 0;
#else
   cout<<"Doesn't compile using PARPACK, comparison not availiable."<<endl;
#endif
}



/**
 * @brief   Print usage.
 * @details Print usage.
 * @return       Return error message.
 */
int print_usage()
{
   /* Print usage */
   printf("Mtx?:        		      -gen\n");
   printf("Mat A:       	   	   -A          [str]\n");
   printf("Mat M:       		      -M          [str]\n");
   printf("                			   If not set, solve eigenvalue problem Av = lv.\n");
   printf("Lap Size:    		      -n          [int] [int] [int]\n");
   printf("                            If gen not used, solve standard eigenvalue problem using 5-pt/7-pt Laplacian.\n");
   printf("Num Eigs     		      -neig       [int]\n");
   printf("Kryrov Dimension::   	   -m          [int]\n");
   printf("Restarts:    		      -niter      [int]\n");
   printf("Eig tol:     		      -tol_eig    [double]\n");
   printf("Exact solve option       -solve_opt\n");
   printf("                        	   For standard eigenvalue problem:\n");
   printf("                          	   0. A = matA, OP = A, compute 'SR' or 'SA'\n");
   printf("                          	   1. A = inv[matA], OP = A, compute 'LR' or 'LA'\n");
   printf("                  		   For generalized eigenvalue problem:\n");
   printf("                          	0. A = matA, M = matM, OP = inv[M]*A, compute 'SR' or 'SA'.\n");
   printf("                          	1. A = matA, M = matM, OP = inv[A]*M (shift-and-invert), compute 'LR' or 'LA'.\n");
   printf("                          	2. A = inv[matM]*matA, treat as nonsymmetric standard eigenvalue problem, compute 'SR' or 'SA'.\n");
   printf("                          	3. A = inv[matA]*matM, treat as nonsymmetric standard eigenvalue problem, compute 'LR' or 'LA'.\n");
   printf("                          	4. A = inv[L]*A*inv[L'], M = LL', treat as symmetric standard eigenvalue problem, compute 'SR' or 'SA'.\n");
   printf("Lanczos no full-reorth -lan\n");
   printf("                          Note: only works for symmetric options. Might be unstable without shift-and-invert. Otherwise using Lanczos with full-reorthgonization or Arnoldi.\n");
   printf("print level: 		      -print_level  [int]\n");
   printf("                            0. minimal output.\n");
   printf("                            1. show timing results.\n");
   printf("                            2. show more timing results (might influence performance).\n");
   printf("                            3. also print parallel direct solver info and memory usage.\n");
   printf("M^{-1}:      		      -direct_solver  [int]\n");
   printf("                            0. superlu.\n");
   printf("                            1. mumps.\n");
   printf("                            2. pardiso.\n");
   return 0;
}

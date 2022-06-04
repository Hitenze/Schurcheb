#include "dsolver.hpp"

/**
 * @file dsolver.cpp
 * @brief Direct solver wrapper
 */

namespace schurcheb
{

   ParallelDirectSolverClass::ParallelDirectSolverClass()
   {
      _A_par = NULL;
      
      /* default no solver */
      _solve_opt = kParallelDsolverOptionNo;
      
   }
   
   int ParallelDirectSolverClass::Clear()
   {
      if (_A_par)
      {
         switch(_solve_opt)
         {
            case kParallelDsolverOptionMUMPS:
            {
               MUMPSFree(_solver_mumps);
               break;
            }
            case kParallelDsolverOptionSuperLU:
            {
               SuperLUFree(_solver_superlu);
               break;
            }
            case kParallelDsolverOptionPardiso:
            {
               PARDISOFree(_solver_pardiso);
               break;
            }
            default:
            {
               SCHURCHEB_WARNING("No direct solver selected.");
            }
         }
         
         _A_par = NULL;
      }
      
      _temp_vec.Clear();
      
      _solve_opt = kParallelDsolverOptionNo;
      
      return SCHURCHEB_SUCCESS;
      
   }

   ParallelDirectSolverClass::~ParallelDirectSolverClass()
   {
      this->Clear();
   }

   int ParallelDirectSolverClass::GetDataLocation() const
   {
      return kMemoryHost;
   }

   int ParallelDirectSolverClass::GetNumRowsLocal() const
   {
      return _A_par->GetNumRowsLocal();
   }

   int ParallelDirectSolverClass::GetNumColsLocal() const
   {
      return _A_par->GetNumColsLocal();
   }

   int ParallelDirectSolverClass::SetupVectorPtrStr(vector_par_double &v)
   {
      return _A_par->SetupVectorPtrStr(v);
   }

   int ParallelDirectSolverClass::MatVec( char trans, const double &alpha, vector_par_double &x, const double &beta, vector_par_double &y)
   {
      /* transpost not supported */
      SCHURCHEB_CHKERR(trans != 'N');
      
      double *x_data = x.GetData();
      double *y_data = y.GetData();
      
      if(x_data == y_data)
      {
         /* in place, no need to copy data 
          * x = alpha*A*x + beta*x
          */
         if(beta == 0)
         {
            /* no need to keep a copy of x, directly scale and solve */
            if(alpha != 1.0)
            {
               x.Scale(alpha);
            }
            switch(_solve_opt)
            {
               case kParallelDsolverOptionMUMPS:
               {
                  MUMPSSolve(_solver_mumps, x_data);
                  break;
               }
               case kParallelDsolverOptionSuperLU:
               {
                  SuperLUSolve(_solver_superlu, x_data);
                  break;
               }
               case kParallelDsolverOptionPardiso:
               {
                  PARDISOSolve(_solver_pardiso, x_data);
                  break;
               }
               default:
               {
                  SCHURCHEB_WARNING("No direct solver selected.");
               }
            }
         }
         else
         {
            /* need to keep a copy of x before the solve, we don't want to modify x 
             * first set temp_vec = beta * x
             */
            
            _temp_vec.Scale(0.0);
            _temp_vec.Axpy(beta, x);
            
            /* then solve on x, x = alpha * A * x */
            if(alpha != 1.0)
            {
               x.Scale(alpha);
            }
            switch(_solve_opt)
            {
               case kParallelDsolverOptionMUMPS:
               {
                  MUMPSSolve(_solver_mumps, x_data);
                  break;
               }
               case kParallelDsolverOptionSuperLU:
               {
                  SuperLUSolve(_solver_superlu, x_data);
                  break;
               }
               case kParallelDsolverOptionPardiso:
               {
                  PARDISOSolve(_solver_pardiso, x_data);
                  break;
               }
               default:
               {
                  SCHURCHEB_WARNING("No direct solver selected.");
               }
            }
            
            /* finally x = beta*x + alpha*A*x */
            x.Axpy(1.0, _temp_vec);
         }
      }
      else
      {
         /* out place 
          * y = alpha*A*x + beta*y
          */
         if(beta == 0)
         {
            /* in this case copy x into y and solve 
             * y = alpha*A*x 
             */
            y.Scale(0.0);
            y.Axpy(alpha, x);
            
            switch(_solve_opt)
            {
               case kParallelDsolverOptionMUMPS:
               {
                  MUMPSSolve(_solver_mumps, y_data);
                  break;
               }
               case kParallelDsolverOptionSuperLU:
               {
                  SuperLUSolve(_solver_superlu, y_data);
                  break;
               }
               case kParallelDsolverOptionPardiso:
               {
                  PARDISOSolve(_solver_pardiso, y_data);
                  break;
               }
               default:
               {
                  SCHURCHEB_WARNING("No direct solver selected.");
               }
            }
         }
         else
         {
            /* in this case need to solve into a new place 
             * y = alpha*A*x + beta*y
             */
            
            /* first temp_vec = alpha*A*x */
            _temp_vec.Scale(0.0);
            _temp_vec.Axpy(alpha, x);
            
            switch(_solve_opt)
            {
               case kParallelDsolverOptionMUMPS:
               {
                  MUMPSSolve(_solver_mumps, _temp_vec.GetData());
                  break;
               }
               case kParallelDsolverOptionSuperLU:
               {
                  SuperLUSolve(_solver_superlu, _temp_vec.GetData());
                  break;
               }
               case kParallelDsolverOptionPardiso:
               {
                  PARDISOSolve(_solver_pardiso, _temp_vec.GetData());
                  break;
               }
               default:
               {
                  SCHURCHEB_WARNING("No direct solver selected.");
               }
            }
            
            /* next y = alpha*A*x + beta*y */
            y.Axpy(beta, _temp_vec);
         }
      }
      
      return SCHURCHEB_SUCCESS;
      
   }

   int ParallelDirectSolverClass::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const
   {
      return _A_par->GetMpiInfo( np, myid, comm);
   }

   MPI_Comm ParallelDirectSolverClass::GetComm() const
   {
      return _A_par->GetComm();
   }

   int ParallelDirectSolverClass::Setup( matrix_csr_par_double &Apar)
   {
      _A_par = &Apar;
      _temp_vec.Setup(Apar.GetNumRowsLocal());
      
      switch(_solve_opt)
      {
         case kParallelDsolverOptionMUMPS:
         {
            MUMPSInit(_solver_mumps, Apar);
            break;
         }
         case kParallelDsolverOptionSuperLU:
         {
            SuperLUInit(_solver_superlu, Apar);
            break;
         }
         case kParallelDsolverOptionPardiso:
         {
            PARDISOInit(_solver_pardiso, Apar);
            break;
         }
         default:
         {
            SCHURCHEB_WARNING("No direct solver selected.");
         }
      }
      
      return SCHURCHEB_SUCCESS;
   }

   int ParallelDirectSolverClass::SetSolveOption(int solve_opt)
   {
      switch(solve_opt)
      {
         case kParallelDsolverOptionMUMPS:
         {
#ifdef SCHURCHEB_MUMPS
            _solve_opt = kParallelDsolverOptionMUMPS;
#else
            solve_opt = kParallelDsolverOptionNo;
#endif
            break;
         }
         case kParallelDsolverOptionSuperLU:
         {
#ifdef SCHURCHEB_SUPERLU
            _solve_opt = kParallelDsolverOptionMUMPS;
#else
            solve_opt = kParallelDsolverOptionNo;
#endif
            break;
         }
         case kParallelDsolverOptionPardiso:
         {
#ifdef SCHURCHEB_MKL
            _solve_opt = kParallelDsolverOptionPardiso;
#else
            solve_opt = kParallelDsolverOptionNo;
#endif
            break;
         }
         default:
         {
            solve_opt = kParallelDsolverOptionNo;
         }
      }
      return SCHURCHEB_SUCCESS;
   }


   int ParallelDirectSolverClass::SetPrintLevel(int print_level)
   {
#ifdef SCHURCHEB_MUMPS
      _solver_mumps._print_level = print_level;
#endif

#ifdef SCHURCHEB_SUPERLU
      _solver_superlu._print_level = print_level;
#endif

#ifdef SCHURCHEB_MKL
      _solver_pardiso._print_level = print_level;
#endif
      return SCHURCHEB_SUCCESS;
   }
   
   ParallelDirectSolver2Class::ParallelDirectSolver2Class()
   {
      _A_par = NULL;
      
      /* default no solver */
      _solve_opt = kParallelDsolverOptionNo;
      
   }
   
   int ParallelDirectSolver2Class::Clear()
   {
      if (_A_par)
      {
         switch(_solve_opt)
         {
            case kParallelDsolverOptionMUMPS:
            {
               MUMPSFree(_solver_mumps);
               break;
            }
            case kParallelDsolverOptionSuperLU:
            {
               SuperLUFree(_solver_superlu);
               break;
            }
            case kParallelDsolverOptionPardiso:
            {
               PARDISOFree(_solver_pardiso);
               break;
            }
            default:
            {
               SCHURCHEB_WARNING("No direct solver selected.");
            }
         }
         
         _A_par = NULL;
      }
      
      _temp_vec.Clear();
      
      _B_par = NULL;
      
      _solve_opt = kParallelDsolverOptionNo;
      
      return SCHURCHEB_SUCCESS;
      
   }

   ParallelDirectSolver2Class::~ParallelDirectSolver2Class()
   {
      this->Clear();
   }

   int ParallelDirectSolver2Class::GetDataLocation() const
   {
      return kMemoryHost;
   }

   int ParallelDirectSolver2Class::GetNumRowsLocal() const
   {
      return _A_par->GetNumRowsLocal();
   }

   int ParallelDirectSolver2Class::GetNumColsLocal() const
   {
      return _A_par->GetNumColsLocal();
   }

   int ParallelDirectSolver2Class::SetupVectorPtrStr(vector_par_double &v)
   {
      return _A_par->SetupVectorPtrStr(v);
   }

   int ParallelDirectSolver2Class::MatVec( char trans, const double &alpha, vector_par_double &x, const double &beta, vector_par_double &y)
   {
      /* transpost not supported */
      SCHURCHEB_CHKERR(trans != 'N');
      
      //double *x_data = x.GetData();
      double *y_data = y.GetData();
      
      /* assume x_data != y_data */
      if(beta == 0)
      {
         /* y = alpha * B * x */
         _B_par->MatVec( trans, alpha, x, beta, y);
         
         /* y = alpha * inv[A] * B * x */
         switch(_solve_opt)
         {
            case kParallelDsolverOptionMUMPS:
            {
               MUMPSSolve(_solver_mumps, y_data);
               break;
            }
            case kParallelDsolverOptionSuperLU:
            {
               SuperLUSolve(_solver_superlu, y_data);
               break;
            }
            case kParallelDsolverOptionPardiso:
            {
               PARDISOSolve(_solver_pardiso, y_data);
               break;
            }
            default:
            {
               SCHURCHEB_WARNING("No direct solver selected.");
            }
         }
      }
      else
      {
         /* keep a copy in temp_vec, first set temp_vec = beta * x */
         _temp_vec.Scale(0.0);
         _temp_vec.Axpy(beta, x);
         
         /* y = alpha * B * x */
         _B_par->MatVec( trans, alpha, x, 0.0, y);
         
         /* next solve y = alpha * inv[A] * B * x  */
         switch(_solve_opt)
         {
            case kParallelDsolverOptionMUMPS:
            {
               MUMPSSolve(_solver_mumps, y_data);
               break;
            }
            case kParallelDsolverOptionSuperLU:
            {
               SuperLUSolve(_solver_superlu, y_data);
               break;
            }
            case kParallelDsolverOptionPardiso:
            {
               PARDISOSolve(_solver_pardiso, y_data);
               break;
            }
            default:
            {
               SCHURCHEB_WARNING("No direct solver selected.");
            }
         }
         
         /* finally add temp_vec in */
         x.Axpy(1.0, _temp_vec);
         
      }
      
      return SCHURCHEB_SUCCESS;
      
   }

   int ParallelDirectSolver2Class::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const
   {
      return _A_par->GetMpiInfo( np, myid, comm);
   }

   MPI_Comm ParallelDirectSolver2Class::GetComm() const
   {
      return _A_par->GetComm();
   }

   int ParallelDirectSolver2Class::Setup( matrix_csr_par_double &Apar, matrix_csr_par_double &Bpar)
   {
      _A_par = &Apar;
      _B_par = &Bpar;
      
      _temp_vec.Setup(Apar.GetNumRowsLocal());
      
      switch(_solve_opt)
      {
         case kParallelDsolverOptionMUMPS:
         {
            MUMPSInit(_solver_mumps, Apar);
            break;
         }
         case kParallelDsolverOptionSuperLU:
         {
            SuperLUInit(_solver_superlu, Apar);
            break;
         }
         case kParallelDsolverOptionPardiso:
         {
            PARDISOInit(_solver_pardiso, Apar);
            break;
         }
         default:
         {
            SCHURCHEB_WARNING("No direct solver selected.");
         }
      }
      
      return SCHURCHEB_SUCCESS;
   }

   int ParallelDirectSolver2Class::SetSolveOption(int solve_opt)
   {
      switch(solve_opt)
      {
         case kParallelDsolverOptionMUMPS:
         {
#ifdef SCHURCHEB_MUMPS
            _solve_opt = kParallelDsolverOptionMUMPS;
#else
            solve_opt = kParallelDsolverOptionNo;
#endif
            break;
         }
         case kParallelDsolverOptionSuperLU:
         {
#ifdef SCHURCHEB_SUPERLU
            _solve_opt = kParallelDsolverOptionMUMPS;
#else
            solve_opt = kParallelDsolverOptionNo;
#endif
            break;
         }
         case kParallelDsolverOptionPardiso:
         {
#ifdef SCHURCHEB_MKL
            _solve_opt = kParallelDsolverOptionPardiso;
#else
            solve_opt = kParallelDsolverOptionNo;
#endif
            break;
         }
         default:
         {
            solve_opt = kParallelDsolverOptionNo;
         }
      }
      return SCHURCHEB_SUCCESS;
   }


   int ParallelDirectSolver2Class::SetPrintLevel(int print_level)
   {
#ifdef SCHURCHEB_MUMPS
      _solver_mumps._print_level = print_level;
#endif

#ifdef SCHURCHEB_SUPERLU
      _solver_superlu._print_level = print_level;
#endif

#ifdef SCHURCHEB_MKL
      _solver_pardiso._print_level = print_level;
#endif
      return SCHURCHEB_SUCCESS;
   }
}
   

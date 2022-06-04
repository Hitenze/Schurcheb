#ifndef CHEBEIG_DSOLVER
#define CHEBEIG_DSOLVER

/**
 * @file dsolver.hpp
 * @brief Direct solver wrapper
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
#include "superlu.hpp"
#include "mumps.hpp"
#include "pardiso.hpp"

using namespace std;

namespace schurcheb
{

   enum ParallelDsolverOptionEnum
   {
      kParallelDsolverOptionSuperLU = 0,
      kParallelDsolverOptionMUMPS,
      kParallelDsolverOptionPardiso,
      kParallelDsolverOptionNo
   };

   enum DsolverOptionEnum
   {
      kDsolverOptionLU,
      kDsolverOptionPardiso
   };

   class ParallelDirectSolverClass : public ArnoldiMatrixClass<vector_par_double, double>, public parallel_log
   {
   public:

      /* the solver option */
      int _solve_opt;
      vector_seq_double _temp_vec;

      /* Do not free this pointer */
      matrix_csr_par_double *_A_par;

#ifdef SCHURCHEB_SUPERLU

      solver_superlu _solver_superlu;

#else

      void *_solver_superlu;

#endif

#ifdef SCHURCHEB_MUMPS

      solver_mumps _solver_mumps;

#else

      void *_solver_mumps;

#endif

#ifdef SCHURCHEB_MKL

      solver_pardiso _solver_pardiso;

#else

      void *_solver_pardiso;

#endif

      ParallelDirectSolverClass();
      
      /* no other constructors */
      
      int Clear();
      
      virtual ~ParallelDirectSolverClass();
      
      int GetDataLocation() const;
      
      int GetNumRowsLocal() const;
      
      int GetNumColsLocal() const;
      
      int SetupVectorPtrStr(vector_par_double &v);
      
      int MatVec( char trans, const double &alpha, vector_par_double &x, const double &beta, vector_par_double &y);
      
      int GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
      
      MPI_Comm GetComm() const;
      
      /* Local functions */
      
      /* Setup the solver */
      int Setup( matrix_csr_par_double &Apar);
     
      int SetSolveOption(int solve_opt);
      
      int SetPrintLevel(int print_level);
      
   };

   typedef ParallelDirectSolverClass dsolver_par_double;

   /* the following class combine a matvec together as inv[A]*B*x */
   class ParallelDirectSolver2Class : public ArnoldiMatrixClass<vector_par_double, double>, public parallel_log
   {
   public:

      /* the solver option */
      int _solve_opt;
      vector_seq_double _temp_vec;

      /* Do not free this pointer */
      matrix_csr_par_double *_A_par;
      
      /* used in matvec */
      matrix_csr_par_double *_B_par;

#ifdef SCHURCHEB_SUPERLU

      solver_superlu _solver_superlu;

#else

      void *_solver_superlu;

#endif

#ifdef SCHURCHEB_MUMPS

      solver_mumps _solver_mumps;

#else

      void *_solver_mumps;

#endif

#ifdef SCHURCHEB_MKL

      solver_pardiso _solver_pardiso;

#else

      void *_solver_pardiso;

#endif

      ParallelDirectSolver2Class();
      
      
      /* no other constructors */
      //ParallelDirectSolverClass(const ParallelDirectSolverClass<VectorType, DataType> &mat);
      
      //ParallelDirectSolverClass(ParallelDirectSolverClass<VectorType, DataType> &&mat);
      
      int Clear();
      
      virtual ~ParallelDirectSolver2Class();
      
      int GetDataLocation() const;
      
      int GetNumRowsLocal() const;
      
      int GetNumColsLocal() const;
      
      int SetupVectorPtrStr(vector_par_double &v);
      
      int MatVec( char trans, const double &alpha, vector_par_double &x, const double &beta, vector_par_double &y);
      
      int GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
      
      MPI_Comm GetComm() const;
      
      /* Local functions */
      
      /* Setup the solver inv[A]*B */
      int Setup( matrix_csr_par_double &Apar, matrix_csr_par_double &Bpar);
     
      int SetSolveOption(int solve_opt);
      
      int SetPrintLevel(int print_level);
      
   };

   typedef ParallelDirectSolver2Class dsolver2_par_double;
}

#endif

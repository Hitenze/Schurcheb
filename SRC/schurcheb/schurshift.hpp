#ifndef CHEBEIG_SCHURSHIFT
#define CHEBEIG_SCHURSHIFT

/**
 * @file schurshift.hpp
 * @brief Helper class for the eigenvalues corresponding to each shift
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
#include "../preconditioners/ilu.hpp"
#include "superlu.hpp"
#include "mumps.hpp"
#include "pardiso.hpp"
#include "dsolver.hpp"
#include "arpack.hpp"

using namespace std;

namespace schurcheb
{

   /**
    * @brief   Class for computing matvec with S(shift).
    * @details Class for computing matvec with S(shift). Used in Krylov. (B-shift*MB)^{-1} is solved using LU factorization.
    */
   class SchurEigvalueClass : public ArnoldiMatrixClass<vector_par_double, double>, public parallel_log
   {
   public:
      
      int                           _solve_opt; /* 0: LU, 1: Pardiso, 2: MUMPS */
      bool                          _shift_invert; /* do we use shift and invert? must have direct solver */
      int                           _direct_solver_opt; /* 0: superlu, 1: MUMPS, 2: Pardiso */
      
      int                           _n;
      int                           _nB;
      int                           _nC;
      double                        _shift; /* the shift duiring the eigenvalue computation */
      double                        _shift_invert_shift;
      int                           _shift_invert_reserve;
      
      bool                          _bonly; // solve B only, for checking the smallest eig of B
      
      double                        _mem;
      int                           _nnz;
      
      matrix_csr_par_double         _B;
      matrix_csr_par_double         _E;
      matrix_csr_par_double         _F;
      matrix_csr_par_double         _C;
      
      matrix_csr_par_double         _MB;
      matrix_csr_par_double         _ME;
      matrix_csr_par_double         _MF;
      matrix_csr_par_double         _MC;
      
      vector_par_double             _temp_x_B;
      vector_par_double             _temp_y_B;
      vector_par_double             _temp_x_C;
      
      precond_ilu_csr_seq_double    _B_solve;
      
#ifdef SCHURCHEB_MKL
      
      matrix_csr_double _Bu;
      matrix_csr_double _MBu;
      matrix_csr_double _BsMB;

      vector_long _pt; /* memory pointer */
      int _maxfct;
      int _mnum;
      int _mtype;
      vector_int _iparam;
      int _msglvl;
      int _nrhs;
      int _phase;
      
#endif
      
#ifdef SCHURCHEB_MUMPS
      matrix_csr_par_double _CsMC;
      solver_mumps _mumps, _mumpss;
      int _mphase, _mphases;
#endif
      
      /**
       * @brief   The constructor.
       * @details The constructor.
       */
      SchurEigvalueClass();
      
      /**
       * @brief   Clear the data structure. This function is also called by the destructor.
       * @details Clear the data structure. This function is also called by the destructor.
       * @return           Return error message.
       */
      int      Clear();
      
      /**
       * @brief   The destructor.
       * @details The destructor.
       */
      virtual ~SchurEigvalueClass();
      
      /**
       * @brief   The copy constructor of SchurEigvalueClass.
       * @details The copy constructor of SchurEigvalueClass.
       * @param [in] str The target datastr.
       */
      SchurEigvalueClass(const SchurEigvalueClass &str);
      
      /**
       * @brief   The move constructor of SchurEigvalueClass.
       * @details The move constructor of SchurEigvalueClass.
       * @param [in] str The target datastr.
       */
      SchurEigvalueClass(SchurEigvalueClass &&str);
      
      /**
       * @brief   The = operator of SchurEigvalueClass.
       * @details The = operator of SchurEigvalueClass.
       * @param [in] str The target datastr.
       * @return     Return the datastr.
       */
      SchurEigvalueClass& operator= (const SchurEigvalueClass &str);
      
      /**
       * @brief   The = operator of SchurEigvalueClass.
       * @details The = operator of SchurEigvalueClass.
       * @param [in] str The target datastr.
       * @return     Return the datastr.
       */
      SchurEigvalueClass& operator= (SchurEigvalueClass &&str);
      
      /**
       * @brief   Set solve option for B.
       * @details Set solve option for B. 0: LU factorization. 1: Pardiso LDL^T (need to link to Pardiso).
       * @param [in]       solve_opt New solve option.
       */
      void     SetSolveOption(int solve_opt);
      
      /**
       * @brief   Set the level of terminal output.
       * @details Set the level of terminal output.
       * @param [in]       print_level New print level.
       */
      void     SetPrintLevel(int print_level);
      
      /**
       * @brief   Setup the class with with B, E, F, C, and shift.
       * @details Setup the class with with B, E, F, C, and shift.
       * @param [in]       B
       * @param [in]       E
       * @param [in]       F
       * @param [in]       C
       * @param [in]       MB
       * @param [in]       ME
       * @param [in]       MF
       * @param [in]       MC The E, B, F, and C for A and M.
       * @param [in]       shift The shift value.
       * @param [in]       parlog The parallel data structure.
       * @return           Return error message.
       */
      int      Setup(matrix_csr_par_double &B, matrix_csr_par_double &E, matrix_csr_par_double &F, matrix_csr_par_double &C, matrix_csr_par_double &MB, matrix_csr_par_double &ME, matrix_csr_par_double &MF, matrix_csr_par_double &MC, double shift, parallel_log &parlog);
      
      /**
       * @brief   After the initial setup, update the shift value without changing E, B, F, and C.
       * @details After the initial setup, update the shift value without changing E, B, F, and C.
       * @param [in]       shift New shift value.
       * @return           Return error message.
       */
      int      UpdateShift(double shift);
            
      /**
       * @brief   In place Matrix-Vector product with S(l) ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @details In place Matrix-Vector product with S(l) ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The first vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The second vector.
       * @return           Return error message.
       */
      int      MatVec( char trans, const double &alpha, vector_par_double &x, const double &beta, vector_par_double &y);

      /**
       * @brief   In place Matrix-Vector product with B(l)\E(l) ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @details In place Matrix-Vector product with B(l)\E(l) ==>  y := alpha*A*x + beta*y, or y := alpha*A'*x + beta*y.
       * @param [in]       trans Whether or not transpose matrix A.
       * @param [in]       alpha The alpha value.
       * @param [in]       x The first vector.
       * @param [in]       beta The beta value.
       * @param [in,out]   y The second vector.
       * @return           Return error message.
       */
      int      MatVec2( char trans, const double &alpha, vector_par_double &x, const double &beta, vector_par_double &y);

      /**
       * @brief   Update the structure of a vector to have same row permutation.
       * @details Update the structure of a vector to have same row permutation.
       * @param [out] vec The target vector.
       * @return      Return error message.
       */
      int      SetupVectorPtrStr(vector_par_double &v);
      
      /**
       * @brief   Get the local number of rows of the matrix.
       * @details Get the local number of rows of the matrix.
       * @return     Return the local number of rows of the matrix.
       */
      int      GetNumRowsLocal() const;

      /**
       * @brief   Get the local number of columns of the matrix.
       * @details Get the local number of columns of the matrix.
       * @return     Return the local number of columns of the matrix.
       */
      int      GetNumColsLocal() const;
      
      /**
       * @brief   Only apply the B solve M_B^{-1}B?
       * @details Only apply the B solve M_B^{-1}B?
       * @param [out] vec The target vector.
       * @return      Return error message.
       */
      int      SetBSolveOption(bool bonly);
      
      /**
       * @brief   Get comm.
       * @details Get comm.
       * @return     Return the MPI_comm.
       */
      MPI_Comm GetComm() const;
      
      /**
       * @brief   Get comm, np, and myid. Get the global one.
       * @details Get comm, np, and myid. Get the global one.
       * @param   [in]        np The number of processors.
       * @param   [in]        myid The local MPI rank number.
       * @param   [in]        comm The MPI_Comm.
       * @return     Return error message.
       */
      int      GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
      
      /**
       * @brief   Get the data location of the class.
       * @details Get the data location of the class. For now this code portion is host only.
       * @return     Return the data location.
       */
      int      GetDataLocation() const;
      
   };

   typedef SchurEigvalueClass matrix_shift;
}

#endif

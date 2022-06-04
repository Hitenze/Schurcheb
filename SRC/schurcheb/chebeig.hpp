#ifndef CHEBEIG_CHEBEIG
#define CHEBEIG_CHEBEIG

/**
 * @file chebeig.hpp
 * @brief main class for the Schur Chebyshev eigensolver
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
#include "schurshift.hpp" 

namespace schurcheb
{
 
   class SchurChebClass
   {
   private:
   
      /**
       * @brief   Permutation vector.
       * @details Permutation vector.
       */
      vector_int                 _perm_v;
   
      /**
       * @brief   Permutation vector.
       * @details Permutation vector.
       */
      vector_int                 _perm_v_dist;
   
      /**
       * @brief   Eigenvalues.
       * @details Eigenvalues.
       */
      vector_seq_double          _eigs_v;
      
      /**
       * @brief   Distributed eigenvectors.
       * @details Distributed eigenvectors.
       */
      matrix_dense_double        _V_mat;
      
      /**
       * @brief   Eigenvectors.
       * @details Eigenvectors.
       */
      matrix_dense_double        _V_mat_seq;
   
      /**
       * @brief   Residuals.
       * @details Residuals.
       */
      vector_seq_double          _res_v;
      
      /**
       * @brief   Dist A for residual.
       * @details Dist A for residual.
       */
      matrix_csr_par_double      _A_par;
      
      /**
       * @brief   Dist M for residual.
       * @details Dist M for residual.
       */
      matrix_csr_par_double      _M_par;
      
      /**
       * @brief   Shifts.
       * @details Shifts.
       */
      matrix_shift               _test_shift;
   
   public:
   
      /**
       * @brief   The constructor.
       * @details The constructor.
       */
      SchurChebClass();
      
      /**
       * @brief   The copy constructor of SchurChebClass.
       * @details The copy constructor of SchurChebClass.
       * @param [in] str The target datastr.
       */
      SchurChebClass(const SchurChebClass &str);
      
      /**
       * @brief   The move constructor of SchurChebClass.
       * @details The move constructor of SchurChebClass.
       * @param [in] str The target datastr.
       */
      SchurChebClass(SchurChebClass &&str);
      
      /**
       * @brief   The = operator of SchurChebClass.
       * @details The = operator of SchurChebClass.
       * @param [in] str The target datastr.
       * @return     Return the datastr.
       */
      SchurChebClass& operator= (const SchurChebClass &str);
      
      /**
       * @brief   The = operator of SchurChebClass.
       * @details The = operator of SchurChebClass.
       * @param [in] str The target datastr.
       * @return     Return the datastr.
       */
      SchurChebClass& operator= (SchurChebClass &&str);
      
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
      virtual ~SchurChebClass();
      
      /**
       * @brief   Compute eigenpairs of A within [a,b] with default accuracy.
       * @details Compute eigenpairs of A within [a,b] with default accuracy.
       * @param [in]       A eigenvalues problem Av = lv.
       * @param [in]       neigs Target number of eigenvalues.
       * @param [in]       a 
       * @param [in]       b [a, b] is the target region.
       * @param [in, out]  ndom target number of subdomains. 
       * @param [in]       nnode number of Chebyshev nodes.
       * @param [in]       ncol number of columns in the 2D MPI structure. ncol <= nnode.
       * @return           Return error message.
       */
      int Setup( matrix_csr_double &A, int neig, double a, double b, int &ndom, int nnode, int ncol);
      
      /**
       * @brief   Compute eigenpairs of (A,M) within [a,b] with default accuracy.
       * @details Compute eigenpairs of (A,M) within [a,b] with default accuracy.
       * @param [in]       A 
       * @param [in]       M Generalized eigenvalues problem Av = lMv.
       * @param [in]       neigs Target number of eigenvalues.
       * @param [in]       a 
       * @param [in]       b [a, b] is the target region.
       * @param [in, out]  ndom target number of subdomains. 
       * @param [in]       nnode number of Chebyshev nodes.
       * @param [in]       ncol number of columns in the 2D MPI structure. ncol <= nnode.
       * @return           Return error message.
       */
      int Setup( matrix_csr_double &A, matrix_csr_double &M, int neig, double a, double b, int &ndom, int nnode, int ncol);
      
      /**
       * @brief   Compute eigenpairs of (A,M) within [a,b] with default accuracy.
       * @details Compute eigenpairs of (A,M) within [a,b] with default accuracy.
       * @param [in]       A 
       * @param [in]       M Generalized eigenvalues problem Av = lMv.
       * @param [in]       gen If set to false M is treated as I.
       * @param [in]       check_dd If set to true, recompute DD if b > l(B, M_B), repeat for at most 3 times. \n
       *                   Note that the results are typically still satisfied even if b > l(B, M_B).
       * @param [in]       neigs Target number of eigenvalues.
       * @param [in]       a 
       * @param [in]       b [a, b] is the target region.
       * @param [in,out]   ndom target number of subdomains. 
       *                   If check_dd == true the value of ndom might increase on return.
       * @param [in]       nnode number of Chebyshev nodes.
       * @param [in]       ncol number of columns in the 2D MPI structure. ncol <= nnode.
       * @param [in]       lan set to true to use Lanczos without full orthgonization (PARPACK required).
       * @param [in]       m restart dimension.
       * @param [in]       niter max number of restarts.
       * @param [in]       tol_eig tolerance for Schur complement eigenvalue problems.
       * @param [in]       tol_eig2 tolerance for final local eigenvalue problems.
       * @param [in]       nB block size for reorthgonization.
       * @param [in]       chol_orth Chol based orth? (might be unstable).
       * @param [in]       B_sol_opt B solution option. 0: LU; 1: MKL pardiso (MKL pardiso required).
       * @param [in]       eigvec_opt eigenvector option. 0: do not compute; 1: compute; 2: compute and gather to rank 0.
       * @param [in]       compute_res do we compute residual vector?
       * @param [in]       print_level Level of output message.
       * @return           Return error message.
       */
      int Setup( matrix_csr_double &A, matrix_csr_double &M, bool gen, bool check_dd,
               int neig, double a, double b, int &ndom, int nnode, int ncol,
               bool lan, int m, int niter, double tol_eig, double tol_eig2, 
               int orth_nB, bool chol_orth, int B_sol_opt, int eigvec_opt, bool compute_res, int print_level);
      
      /**
       * @brief   Get the residual vector. Only form if residual is wanted.
       * @details Get the residual vector. Only form if residual is wanted.
       * @return           Return the residual vector.
       */
      vector_seq_double& GetResiduals();
      
      /**
       * @brief   Get the distributed A matrix. Only form if residual is wanted.
       * @details Get the distributed A matrix. Only form if residual is wanted.
       * @return           Return the residual vector.
       */
      matrix_csr_par_double& GetParA();
      
      /**
       * @brief   Get the distributed M matrix. Only form if residual is wanted.
       * @details Get the distributed M matrix. Only form if residual is wanted.
       * @return           Return the residual vector.
       */
      matrix_csr_par_double& GetParM();
      
      /**
       * @brief   Get the eigenvalue vector.
       * @details Get the eigenvalue vector.
       * @return           Return the eigenvalue vector.
       */
      vector_seq_double& GetEigenValues();
      
      /**
       * @brief   Get the permutation vector (global permutation of length n).
       * @details Get the permutation vector (global permutation of length n).
       * @note    The global permutation is different from the local permutation.
       * @return           Return the permutation vector.
       */
      vector_int& GetPermutation();
      
      /**
       * @brief   Get the global eigenvector matrix (after permutation). Use GetPermutation() to get the permutation.
       * @details Get the global eigenvector matrix (after permutation). Use GetPermutation() to get the permutation.
       * @return           Return the global eigenvector matrix.
       */
      matrix_dense_double& GetEigenVectors();
      
      /**
       * @brief   Get the distributed permutation vector.
       * @details Get the distributed permutation vector.
       * @note    The distributed permutation is different from the global permutation.
       * @return           Return the distributed permutation vector.
       */
      vector_int& GetDistPermutation();
      
      /**
       * @brief   Get the distributed eigenvector matrix. Use GetDistPermutation() to get the permutation.
       * @details Get the distributed eigenvector matrix. Use GetDistPermutation() to get the permutation.
       * @return           Return the distributed eigenvector matrix.
       */
      matrix_dense_double& GetDistEigenVectors();
      
   };
   
   typedef SchurChebClass schurcheb_double;

}

#endif

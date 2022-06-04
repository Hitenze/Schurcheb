/**
 * @file test_min.cpp
 * @brief An example of calling Schurcheb from C++
 */

#include "schurcheb.hpp"
#include <iostream>

using namespace std;
using namespace schurcheb;

int main(int argc, char *argv[]) 
{
   SchurchebInit( &argc, &argv);
   
   // load matrix into 0-based CSR format
   matrix_csr_double A, M;
   A.ReadFromMMFile("./Matrices/Ale2475.mtx", 1); // (filename, shift)
   M.ReadFromMMFile("./Matrices/Mle2475.mtx", 1);
   
   // compute neig smallest eigenvalues between [a, b]
   // domain decomposition with ndom subdomains
   // use nnode Chebyshev nodes and assign Chebyshev nodes to ncol MPI communicators.
   //
   // notes:
   // ndom should be passed by reference since the value might be changed on return
   
   double a = 0.0, b = 2.0;
   int ndom = 8, nnode = 8, ncol = 1;
   
   schurcheb_double eig_solver;
   eig_solver.Setup( A, M, 20, a, b, ndom, nnode, ncol);
   
   // access eigenvalue vector, 0-based
   // access entries in eigs using eigs[i]
   vector_seq_double &eigs = eig_solver.GetEigenValues();
   if(parallel_log::_grank == 0)
   {
      cout<<"the 3rd smallest eigenvalue is "<<eigs[2]<<endl;
   }
   
   // access distributed eigenvectors in matrix format, 0-based column major
   // access entries in V using V_mat(i, j)
   // V(perm,:) is stored on each MPI process.
   matrix_dense_double &V_mat = eig_solver.GetDistEigenVectors();
   vector_int &perm = eig_solver.GetDistPermutation();
   
   // free
   A.Clear();
   M.Clear();
   eig_solver.Clear();
   
   SchurchebFinalize();
}


#include <unordered_map>
#include <unordered_set>
#include "../utils/parallel.hpp"
#include "../utils/utils.hpp"
#include "../utils/memory.hpp"
#include "../utils/protos.hpp"
#include "../vectors/vector.hpp"
#include "../vectors/parallel_vector.hpp"
#include "matrix.hpp"
#include "matrixops.hpp"
#include "dense_matrix.hpp"
#include "csr_matrix.hpp"
#include "parallel_csr_matrix.hpp"
#include "../preconditioners/ilu.hpp"

#include <iostream>
#include <complex>
#include <limits>
#include <limits.h>

#ifdef SCHURCHEB_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "cusparse.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#endif

using namespace std;

/* Arnoldi using PARPACK-NG */


namespace schurcheb
{
   
   PrecisionEnum GetMatrixPrecision(const MatrixClass<int> &mat)
   {
      return kInt;
   }
   
   PrecisionEnum GetMatrixPrecision(const MatrixClass<long int> &mat)
   {
      return kLongInt;
   }
   
   PrecisionEnum GetMatrixPrecision(const MatrixClass<float> &mat)
   {
      return kSingleReal;
   }
   
   PrecisionEnum GetMatrixPrecision(const MatrixClass<double> &mat)
   {
      return kDoubleReal;
   }
   
   PrecisionEnum GetMatrixPrecision(const MatrixClass<complexs> &mat)
   {
      return kSingleComplex;
   }
   
   PrecisionEnum GetMatrixPrecision(const MatrixClass<complexd> &mat)
   {
      return kDoubleComplex;
   }
   
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<int> *mat)
   {
      return kInt;
   }
   
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<long int> *mat)
   {
      return kLongInt;
   }
   
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<float> *mat)
   {
      return kSingleReal;
   }
   
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<double> *mat)
   {
      return kDoubleReal;
   }
   
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<complexs> *mat)
   {
      return kSingleComplex;
   }
   
   PrecisionEnum GetMatrixPPrecision(const MatrixClass<complexd> *mat)
   {
      return kDoubleComplex;
   }
   
   template <typename T>
   int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const T &alpha, const T *aa, int ldim, const T *x, const T &beta, T *y)
   {
      
      int      i, j;
      T        *x_temp = NULL;
      T        one = 1.0;
      T        zero = 0.0;
      const T  *a_temp;
      
      /* 1. Compute y = beta*y
       * note that when x==y and alpha != 0.0, we need to copy x
       */
      
      /* copy x when x==y, otherwise scale y would modify x 
       * TODO: memcpy or omp parallel?
       */
      if( (x == y) && (alpha != zero) )
      {
         if (trans == 'N') 
         {
            if(x == y)
            {
               SCHURCHEB_MALLOC(x_temp, nrows, kMemoryHost, T);
               SCHURCHEB_MEMCPY(x_temp, y, nrows, kMemoryHost, kMemoryHost, T);
               x = x_temp;
            }
         }
         else if( (trans == 'T') || (trans == 'C') )
         {
            if(x == y)
            {
               SCHURCHEB_MALLOC(x_temp, ncols, kMemoryHost, T);
               SCHURCHEB_MEMCPY(x_temp, y, ncols, kMemoryHost, kMemoryHost, T);
               x = x_temp;
            }
         }
         else
         {
            return SCHURCHEB_ERROR_INVALED_PARAM;
         }
      }
      
      /* now scale y */
      if(beta != one)
      {
         /* when beta == 1.0, y = y, do nothing */
         if(beta != zero)
         {
            /* y = beta*y */
            if (trans == 'N') 
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < nrows; i++) 
               {
                  y[i] *= beta;
               }
            }
            else if( (trans == 'T') || (trans == 'C') )
            {
               /* if x == y need to create new x */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < ncols; i++) 
               {
                  y[i] *= beta;
               }
            }
            else
            {
               return SCHURCHEB_ERROR_INVALED_PARAM;
            }
         }
         else
         {
            /* beta == 0.0, y = 0 */
            if (trans == 'N') 
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < nrows; i++) 
               {
                  y[i] = zero;
               }
            }
            else if( (trans == 'T') || (trans == 'C') )
            {
               /* if x == y need to create new x */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < ncols; i++) 
               {
                  y[i] = zero;
               }
            }
            else
            {
               return SCHURCHEB_ERROR_INVALED_PARAM;
            }
         }
      }
      
      
      /* 2. the matvec y = alpha*A*x + y
       * when alpha == 0 we have y = y, do nothing
       */
      
      if(alpha != zero)
      {
         if(alpha != one)
         {
            if (trans == 'N') 
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel private(i, j, a_temp)
#endif
               {
                  for(i = 0 ; i < ncols ; i ++)
                  {
                     a_temp = aa + i * ldim;
#ifdef SCHURCHEB_OPENMP
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
#endif
                     for(j = 0 ; j < nrows ; j ++)
                     {
                        y[j] += alpha * a_temp[j] * x[i];
                     }
                  }
               }/* end of OpenMP parallel */
            }
            else if(trans == 'T')
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, a_temp) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
#endif
               for(i = 0 ; i < ncols ; i ++)
               {
                  a_temp = aa + i * ldim;
                  for(j = 0 ; j < nrows ; j ++)
                  {
                     y[i] += alpha * a_temp[j] * x[j];
                  }
               }
            }
            else if(trans == 'C')
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, a_temp) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
#endif
               for(i = 0 ; i < ncols ; i ++)
               {
                  a_temp = aa + i * ldim;
                  for(j = 0 ; j < nrows ; j ++)
                  {
                     y[i] += alpha * SchurchebConj(a_temp[j]) * x[j];
                  }
               }
            }
            else
            {
               return SCHURCHEB_ERROR_INVALED_PARAM;
            }
         }
         else
         {
            if (trans == 'N') 
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel private(i, j, a_temp)
#endif
               {
                  for(i = 0 ; i < ncols ; i ++)
                  {
                     a_temp = aa + i * ldim;
#ifdef SCHURCHEB_OPENMP
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
#endif
                     for(j = 0 ; j < nrows ; j ++)
                     {
                        y[j] += a_temp[j] * x[i];
                     }
                  }
               }/* end of OpenMP parallel */
            }
            else if(trans == 'T')
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, a_temp) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
#endif
               for(i = 0 ; i < ncols ; i ++)
               {
                  a_temp = aa + i * ldim;
                  for(j = 0 ; j < nrows ; j ++)
                  {
                     y[i] += a_temp[j] * x[j];
                  }
               }
            }
            else if(trans == 'C')
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, a_temp) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
#endif
               for(i = 0 ; i < ncols ; i ++)
               {
                  a_temp = aa + i * ldim;
                  for(j = 0 ; j < nrows ; j ++)
                  {
                     y[i] += SchurchebConj(a_temp[j]) * x[j];
                  }
               }
            }
            else
            {
               return SCHURCHEB_ERROR_INVALED_PARAM;
            }
         }
      }
      
      if(x_temp)
      {
         SCHURCHEB_FREE( x_temp, kMemoryHost);
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const float &alpha, const float *aa, int ldim, const float *x, const float &beta, float *y);
   template int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const double &alpha, const double *aa, int ldim, const double *x, const double &beta, double *y);
   template int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const complexs &alpha, const complexs *aa, int ldim, const complexs *x, const complexs &beta, complexs *y);
   template int DenseMatrixPMatVecTemplate( char trans, int nrows, int ncols, const complexd &alpha, const complexd *aa, int ldim, const complexd *x, const complexd &beta, complexd *y);
   
   int DenseMatrixMatVec( const DenseMatrixClass<float> &A, char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y)
   {
      int   m, n, ldim_a;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();
      ldim_a= A.GetLeadingDimension();

#ifdef SCHURCHEB_DEBUG
      if (trans == 'N') 
      {
         SCHURCHEB_CHKERR( m != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( n != x.GetLengthLocal() );
      } 
      else if(trans == 'C' || trans == 'T')
      {
         SCHURCHEB_CHKERR( n != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( m != x.GetLengthLocal() );
         trans = 'T';
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef SCHURCHEB_CUDA 
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               SCHURCHEB_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
               
               return DenseMatrixSMatVecDevice( A, trans, alpha, x, beta, y);
               
               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
                     
                     return DenseMatrixSMatVecDevice( A, trans, alpha, x, beta, y);
                     
                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return DenseMatrixSMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     SCHURCHEB_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               SCHURCHEB_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
#ifdef SCHURCHEB_BLAS
         int one = 1;
         SCHURCHEB_BLASLAPACK_SGEMV( &trans, &m, &n, &alpha, A.GetData(), &ldim_a, x.GetData(), &one, &beta, y.GetData(), &one);
#else
         DenseMatrixPMatVecTemplate( trans, m, n, alpha, A.GetData(), ldim_a, x.GetData(), beta, y.GetData());
#endif

      }
      else if( ((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0) )
      {
         y.Scale(beta);
      }
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixMatVec( const DenseMatrixClass<double> &A, char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y)
   {
      int   m, n, ldim_a;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();
      ldim_a= A.GetLeadingDimension();

#ifdef SCHURCHEB_DEBUG
      if (trans == 'N') 
      {
         SCHURCHEB_CHKERR( m != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( n != x.GetLengthLocal() );
      } 
      else if(trans == 'C' || trans == 'T')
      {
         SCHURCHEB_CHKERR( n != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( m != x.GetLengthLocal() );
         trans = 'T';
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef SCHURCHEB_CUDA 
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               SCHURCHEB_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
               
               return DenseMatrixDMatVecDevice( A, trans, alpha, x, beta, y);
               
               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
                     
                     return DenseMatrixDMatVecDevice( A, trans, alpha, x, beta, y);
                     
                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return DenseMatrixDMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     SCHURCHEB_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               SCHURCHEB_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
#ifdef SCHURCHEB_BLAS
         int one = 1;
         SCHURCHEB_BLASLAPACK_DGEMV( &trans, &m, &n, &alpha, A.GetData(), &ldim_a, x.GetData(), &one, &beta, y.GetData(), &one);
#else
         DenseMatrixPMatVecTemplate( trans, m, n, alpha, A.GetData(), ldim_a, x.GetData(), beta, y.GetData());
#endif

      }
      else if( ((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0) )
      {
         y.Scale(beta);
      }
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixMatVec( const DenseMatrixClass<complexs> &A, char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y)
   {
      int   m, n, ldim_a;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();
      ldim_a= A.GetLeadingDimension();

#ifdef SCHURCHEB_DEBUG
      if (trans == 'N') 
      {
         SCHURCHEB_CHKERR( m != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( n != x.GetLengthLocal() );
      } 
      else if(trans == 'C' || trans == 'T')
      {
         SCHURCHEB_CHKERR( n != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( m != x.GetLengthLocal() );
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef SCHURCHEB_CUDA 
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               SCHURCHEB_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
               
               return DenseMatrixCMatVecDevice( A, trans, alpha, x, beta, y);
               
               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
                     
                     return DenseMatrixCMatVecDevice( A, trans, alpha, x, beta, y);
                     
                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return DenseMatrixCMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     SCHURCHEB_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               SCHURCHEB_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
#ifdef SCHURCHEB_BLAS
         int one = 1;
         SCHURCHEB_BLASLAPACK_CGEMV( &trans, &m, &n, SCHURCHEB_CAST( const ccomplexs*, &alpha), 
                                    SCHURCHEB_CAST( const ccomplexs*, A.GetData()), &ldim_a, 
                                    SCHURCHEB_CAST( const ccomplexs*, x.GetData()), &one, 
                                    SCHURCHEB_CAST( const ccomplexs*, &beta), SCHURCHEB_CAST( ccomplexs*, y.GetData()), &one);
#else
         DenseMatrixPMatVecTemplate( trans, m, n, alpha, A.GetData(), ldim_a, x.GetData(), beta, y.GetData());
#endif

      }
      else if(((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixMatVec( const DenseMatrixClass<complexd> &A, char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y)
   {
      int   m, n, ldim_a;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();
      ldim_a= A.GetLeadingDimension();

#ifdef SCHURCHEB_DEBUG
      if (trans == 'N') 
      {
         SCHURCHEB_CHKERR( m != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( n != x.GetLengthLocal() );
      } 
      else if(trans == 'C' || trans == 'T')
      {
         SCHURCHEB_CHKERR( n != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( m != x.GetLengthLocal() );
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef SCHURCHEB_CUDA 
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               SCHURCHEB_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
               
               return DenseMatrixZMatVecDevice( A, trans, alpha, x, beta, y);
               
               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
                     
                     return DenseMatrixZMatVecDevice( A, trans, alpha, x, beta, y);
                     
                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return DenseMatrixZMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     SCHURCHEB_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               SCHURCHEB_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
#ifdef SCHURCHEB_BLAS
         int one = 1;
         SCHURCHEB_BLASLAPACK_ZGEMV( &trans, &m, &n, SCHURCHEB_CAST( const ccomplexd*, &alpha), 
                                    SCHURCHEB_CAST( const ccomplexd*, A.GetData()), &ldim_a, 
                                    SCHURCHEB_CAST( const ccomplexd*, x.GetData()), &one, 
                                    SCHURCHEB_CAST( const ccomplexd*, &beta), SCHURCHEB_CAST( ccomplexd*, y.GetData()), &one);
#else
         DenseMatrixPMatVecTemplate( trans, m, n, alpha, A.GetData(), ldim_a, x.GetData(), beta, y.GetData());
#endif

      }
      else if(((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixInvertHost( DenseMatrixClass<float> &A)
   {
      int m, ldim;
      
      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR(m != A.GetNumColsLocal());
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return SCHURCHEB_SUCCESS;
      }
      
      int                  info, lwork;
      vector_int           ipiv;
      vector_seq_float     work;
      
      ipiv.Setup(m);
      
      /* LU factorization */
      SCHURCHEB_BLASLAPACK_SGETRF( &m, &m, A.GetData(), &ldim, ipiv.GetData(), &info); SCHURCHEB_CHKERR(info);
      
      lwork = m;
      work.Setup(m);
      
      /* invert */
      SCHURCHEB_BLASLAPACK_SGETRI( &m, A.GetData(), &ldim, ipiv.GetData(), work.GetData(), &lwork, &info); SCHURCHEB_CHKERR(info);
      
      work.Clear();
      ipiv.Clear();
      
      return SCHURCHEB_SUCCESS;
      
   }
   
   int DenseMatrixInvertHost( DenseMatrixClass<double> &A)
   {
      int m, ldim;
      
      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR(m != A.GetNumColsLocal());
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return SCHURCHEB_SUCCESS;
      }
      
      int                  info, lwork;
      vector_int           ipiv;
      vector_seq_double    work;
      
      ipiv.Setup(m);
      
      /* LU factorization */
      SCHURCHEB_BLASLAPACK_DGETRF( &m, &m, A.GetData(), &ldim, ipiv.GetData(), &info); SCHURCHEB_CHKERR(info);
      
      lwork = m;
      work.Setup(m);
      
      /* invert */
      SCHURCHEB_BLASLAPACK_DGETRI( &m, A.GetData(), &ldim, ipiv.GetData(), work.GetData(), &lwork, &info); SCHURCHEB_CHKERR(info);
      
      work.Clear();
      ipiv.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixInvertHost( DenseMatrixClass<complexs> &A)
   {
      int m, ldim;
      
      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR(m != A.GetNumColsLocal());
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return SCHURCHEB_SUCCESS;
      }
      
      int                  info, lwork;
      vector_int           ipiv;
      vector_seq_complexs  work;
      
      ipiv.Setup(m);
      
      /* LU factorization */
      SCHURCHEB_BLASLAPACK_CGETRF( &m, &m, SCHURCHEB_CAST(ccomplexs* ,A.GetData()),
                                 &ldim, ipiv.GetData(), &info); SCHURCHEB_CHKERR(info);
      
      lwork = m;
      work.Setup(m);
      
      /* invert */
      SCHURCHEB_BLASLAPACK_CGETRI( &m, SCHURCHEB_CAST(ccomplexs* ,A.GetData()), 
                                 &ldim, ipiv.GetData(), SCHURCHEB_CAST(ccomplexs* ,work.GetData()), &lwork, &info); SCHURCHEB_CHKERR(info);
      
      work.Clear();
      ipiv.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixInvertHost( DenseMatrixClass<complexd> &A)
   {
      int m, ldim;
      
      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR(m != A.GetNumColsLocal());
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return SCHURCHEB_SUCCESS;
      }
      
      int                  info, lwork;
      vector_int           ipiv;
      vector_seq_complexd  work;
      
      ipiv.Setup(m);
      
      /* LU factorization */
      SCHURCHEB_BLASLAPACK_ZGETRF( &m, &m, SCHURCHEB_CAST(ccomplexd* ,A.GetData()),
                                 &ldim, ipiv.GetData(), &info); SCHURCHEB_CHKERR(info);
      
      lwork = m;
      work.Setup(m);
      
      /* invert */
      SCHURCHEB_BLASLAPACK_ZGETRI( &m, SCHURCHEB_CAST(ccomplexd* ,A.GetData()), 
                                 &ldim, ipiv.GetData(), SCHURCHEB_CAST(ccomplexd* ,work.GetData()), &lwork, &info); SCHURCHEB_CHKERR(info);
      
      work.Clear();
      ipiv.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<float> &A)
   {
      int m, ldim;
      
      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR(m != A.GetNumColsLocal());
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return SCHURCHEB_SUCCESS;
      }
      
      int         info;
      char        uplo = 'U';
      char        diag = 'N';
      
      SCHURCHEB_BLASLAPACK_STRTRI( &uplo, &diag, &m, A.GetData(), &ldim, &info); SCHURCHEB_CHKERR(info);
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<double> &A)
   {
      int m, ldim;
      
      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR(m != A.GetNumColsLocal());
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return SCHURCHEB_SUCCESS;
      }
      
      int         info;
      char        uplo = 'U';
      char        diag = 'N';
      
      SCHURCHEB_BLASLAPACK_DTRTRI( &uplo, &diag, &m, A.GetData(), &ldim, &info); SCHURCHEB_CHKERR(info);
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<complexs> &A)
   {
      int m, ldim;
      
      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR(m != A.GetNumColsLocal());
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return SCHURCHEB_SUCCESS;
      }
      
      int         info;
      char        uplo = 'U';
      char        diag = 'N';
      
      SCHURCHEB_BLASLAPACK_CTRTRI( &uplo, &diag, &m, SCHURCHEB_CAST(ccomplexs* ,A.GetData()), &ldim, &info); SCHURCHEB_CHKERR(info);
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixInvertUpperTriangularHost( DenseMatrixClass<complexd> &A)
   {
      int m, ldim;
      
      m  = A.GetNumRowsLocal();
      //n  = A.GetNumColsLocal();
      ldim = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR(m != A.GetNumColsLocal());
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix invert can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(m == 0)
      {
         /* don't need to compute inverse when empty */
         return SCHURCHEB_SUCCESS;
      }
      
      int         info;
      char        uplo = 'U';
      char        diag = 'N';
      
      SCHURCHEB_BLASLAPACK_ZTRTRI( &uplo, &diag, &m, SCHURCHEB_CAST(ccomplexd* ,A.GetData()), &ldim, &info); SCHURCHEB_CHKERR(info);
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixQRDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumRowsLocal();
      int                  n           = A.GetNumColsLocal();
      int                  minmn       = SchurchebMin(m, n);
      int                  lwork       = n*n;
      int                  ldim_A      = A.GetLeadingDimension();
      DenseMatrixClass<float>          R;
      vector_seq_float     work;             //working array
      vector_seq_float     tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m < n)
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
         {
            Q.Setup( m, m, kMemoryHost, false);
         }
      }
      else
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != n )
         {
            Q.Setup( m, n, kMemoryHost, false);
         }
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m == 0 || n == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(minmn);
      work.Setup(lwork);
      
      /* QR factorization */
      SCHURCHEB_BLASLAPACK_SGEQRF( &m, &n, A.GetData(), &ldim_A, tau.GetData(), work.GetData(), &lwork, &info);
      
      /* Q might be allocated outside */
      Q.Fill(0.0);
      R.Setup( minmn, n, kMemoryHost, true);
      
      /* copy Q data, below the diagonal */
      for (i = 0; i < n; i++) 
      {
         for (j = i+1; j < m; j++) 
         {
            Q(j,i) = A(j,i);
         }
      }
      
      /* copy R data */
      for (i = 0; i < minmn; i++) 
      {
         for (j = 0; j <= i; j++) 
         {
            R(j,i) = A(j,i);
         }
      }
      /* m might be smaller than n */
      for (i = minmn; i < n; i++) 
      {
         for (j = 0; j < minmn; j++) 
         {
            R(j,i) = A(j,i);
         }
      }
      
      /* set A for return */
      A = R;
      
      /* generate matrix Q */
      if(m > n)
      {
         SCHURCHEB_BLASLAPACK_SORGQR(&m, &n, &minmn, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &lwork, &info); SCHURCHEB_CHKERR(info);
      }
      else
      {
         SCHURCHEB_BLASLAPACK_SORGQR(&m, &m, &minmn, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &lwork, &info); SCHURCHEB_CHKERR(info);
      }
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      R.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixQRDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumRowsLocal();
      int                  n           = A.GetNumColsLocal();
      int                  minmn       = SchurchebMin(m, n);
      int                  lwork       = n*n;
      int                  ldim_A      = A.GetLeadingDimension();
      DenseMatrixClass<double>         R;
      vector_seq_double    work;             //working array
      vector_seq_double    tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m < n)
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
         {
            Q.Setup( m, m, kMemoryHost, false);
         }
      }
      else
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != n )
         {
            Q.Setup( m, n, kMemoryHost, false);
         }
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m == 0 || n == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(minmn);
      work.Setup(lwork);
      
      /* QR factorization */
      
      SCHURCHEB_BLASLAPACK_DGEQRF( &m, &n, A.GetData(), &ldim_A, tau.GetData(), work.GetData(), &lwork, &info);
      
      /* Q might be allocated outside */
      Q.Fill(0.0);
      R.Setup( minmn, n, kMemoryHost, true);
      
      /* copy Q data, below the diagonal */
      for (i = 0; i < n; i++) 
      {
         for (j = i+1; j < m; j++) 
         {
            Q(j,i) = A(j,i);
         }
      }
      
      /* copy R data */
      for (i = 0; i < minmn; i++) 
      {
         for (j = 0; j <= i; j++) 
         {
            R(j,i) = A(j,i);
         }
      }
      /* m might be smaller than n */
      for (i = minmn; i < n; i++) 
      {
         for (j = 0; j < minmn; j++) 
         {
            R(j,i) = A(j,i);
         }
      }
      
      /* set A for return */
      A = R;
      
      /* generate matrix Q */
      if(m > n)
      {
         SCHURCHEB_BLASLAPACK_DORGQR(&m, &n, &minmn, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &lwork, &info); SCHURCHEB_CHKERR(info);
      }
      else
      {
         SCHURCHEB_BLASLAPACK_DORGQR(&m, &m, &minmn, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &lwork, &info); SCHURCHEB_CHKERR(info);
      }
      
      /* deallocate */
      tau.Clear();
      work.Clear();
      R.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixQRDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumRowsLocal();
      int                  n           = A.GetNumColsLocal();
      int                  minmn       = SchurchebMin(m, n);
      int                  lwork       = n*n;
      int                  ldim_A      = A.GetLeadingDimension();
      DenseMatrixClass<complexs>       R;
      vector_seq_complexs  work;             //working array
      vector_seq_complexs  tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m < n)
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
         {
            Q.Setup( m, m, kMemoryHost, false);
         }
      }
      else
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != n )
         {
            Q.Setup( m, n, kMemoryHost, false);
         }
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m == 0 || n == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(minmn);
      work.Setup(lwork);
      
      /* QR factorization */
      SCHURCHEB_BLASLAPACK_CGEQRF( &m, &n, SCHURCHEB_CAST( ccomplexs*, A.GetData()), &ldim_A, SCHURCHEB_CAST( ccomplexs*, tau.GetData()), 
                                    SCHURCHEB_CAST( ccomplexs*, work.GetData()), &lwork, &info);
      
      /* Q might be allocated outside */
      Q.Fill(0.0);
      R.Setup( minmn, n, kMemoryHost, true);
      
      /* copy Q data, below the diagonal */
      for (i = 0; i < n; i++) 
      {
         for (j = i+1; j < m; j++) 
         {
            Q(j,i) = A(j,i);
         }
      }
      
      /* copy R data */
      for (i = 0; i < minmn; i++) 
      {
         for (j = 0; j <= i; j++) 
         {
            R(j,i) = A(j,i);
         }
      }
      /* m might be smaller than n */
      for (i = minmn; i < n; i++) 
      {
         for (j = 0; j < minmn; j++) 
         {
            R(j,i) = A(j,i);
         }
      }
      
      /* set A for return */
      A = R;
      
      /* generate matrix Q */
      if(m > n)
      {
         SCHURCHEB_BLASLAPACK_CUNGQR(&m, &n, &minmn, SCHURCHEB_CAST( ccomplexs*, Q.GetData()), &ldim_Q, SCHURCHEB_CAST( ccomplexs*, tau.GetData()), 
                                    SCHURCHEB_CAST( ccomplexs*, work.GetData()), &lwork, &info); SCHURCHEB_CHKERR(info);
      }
      else
      {
         SCHURCHEB_BLASLAPACK_CUNGQR(&m, &m, &minmn, SCHURCHEB_CAST( ccomplexs*, Q.GetData()), &ldim_Q, SCHURCHEB_CAST( ccomplexs*, tau.GetData()), 
                                    SCHURCHEB_CAST( ccomplexs*, work.GetData()), &lwork, &info); SCHURCHEB_CHKERR(info);
      }
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      R.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixQRDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumRowsLocal();
      int                  n           = A.GetNumColsLocal();
      int                  minmn       = SchurchebMin(m, n);
      int                  lwork       = n*n;
      int                  ldim_A      = A.GetLeadingDimension();
      DenseMatrixClass<complexd>       R;
      vector_seq_complexd  work;             //working array
      vector_seq_complexd  tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m < n)
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
         {
            Q.Setup( m, m, kMemoryHost, false);
         }
      }
      else
      {
         if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != n )
         {
            Q.Setup( m, n, kMemoryHost, false);
         }
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m == 0 || n == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(minmn);
      work.Setup(lwork);
      
      /* QR factorization */
      SCHURCHEB_BLASLAPACK_ZGEQRF( &m, &n, SCHURCHEB_CAST( ccomplexd*, A.GetData()), &ldim_A, SCHURCHEB_CAST( ccomplexd*, tau.GetData()), 
                                    SCHURCHEB_CAST( ccomplexd*, work.GetData()), &lwork, &info);
      
      /* Q might be allocated outside */
      Q.Fill(0.0);
      R.Setup( minmn, n, kMemoryHost, true);
      
      /* copy Q data, below the diagonal */
      for (i = 0; i < n; i++) 
      {
         for (j = i+1; j < m; j++) 
         {
            Q(j,i) = A(j,i);
         }
      }
      
      /* copy R data */
      for (i = 0; i < minmn; i++) 
      {
         for (j = 0; j <= i; j++) 
         {
            R(j,i) = A(j,i);
         }
      }
      /* m might be smaller than n */
      for (i = minmn; i < n; i++) 
      {
         for (j = 0; j < minmn; j++) 
         {
            R(j,i) = A(j,i);
         }
      }
      
      /* set A for return */
      A = R;
      
      /* generate matrix Q */
      if(m > n)
      {
         SCHURCHEB_BLASLAPACK_ZUNGQR(&m, &n, &minmn, SCHURCHEB_CAST( ccomplexd*, Q.GetData()), &ldim_Q, SCHURCHEB_CAST( ccomplexd*, tau.GetData()), 
                                 SCHURCHEB_CAST( ccomplexd*, work.GetData()), &lwork, &info); SCHURCHEB_CHKERR(info);
      }
      else
      {
         SCHURCHEB_BLASLAPACK_ZUNGQR(&m, &m, &minmn, SCHURCHEB_CAST( ccomplexd*, Q.GetData()), &ldim_Q, SCHURCHEB_CAST( ccomplexd*, tau.GetData()), 
                                 SCHURCHEB_CAST( ccomplexd*, work.GetData()), &lwork, &info); SCHURCHEB_CHKERR(info);
      }
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      R.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q)
   {
      int               i, j;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A      = A.GetLeadingDimension();
      int               one         = 1;  //H is not already upper triangular in any part
      vector_seq_float  work;             //working array
      vector_seq_float  tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      if(m == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(m);
      work.Setup(m);
      
      /* reduce to upper Hessenberg form */
      SCHURCHEB_BLASLAPACK_SGEHRD( &m, &one, &m, A.GetData(), &ldim_A, tau.GetData()+1, work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
      
      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++) 
      {
         for (j = i+2; j < m; j++) 
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0; 
         }
      }
      
      /* generate matrix Q */
      tau[0] = 0.0;
      SCHURCHEB_BLASLAPACK_SORGQR(&m, &m, &m, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<float> &A, int start, int end, DenseMatrixClass<float> &Q)
   {
      int               i, j;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A      = A.GetLeadingDimension();
      int               ilo, ihi;
      vector_seq_float  work;             //working array
      vector_seq_float  tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      if(m == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(m);
      work.Setup(m);
      
      /* reduce to upper Hessenberg form 
       * A is already upper triangular in rows and columns 1:ILO-1 and IHI+1:N
       */
      ilo = start + 1;
      ihi = end;
      SCHURCHEB_BLASLAPACK_SGEHRD( &m, &ilo, &ihi, A.GetData(), &ldim_A, tau.GetData()+1, work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
      
      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++) 
      {
         for (j = i+2; j < m; j++) 
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0; 
         }
      }
      
      /* generate matrix Q */
      tau[0] = 0.0;
      SCHURCHEB_BLASLAPACK_SORGQR(&m, &m, &m, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q)
   {
      int               i, j;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A      = A.GetLeadingDimension();
      int               one         = 1;  //H is not already upper triangular in any part
      vector_seq_double work;             //working array
      vector_seq_double tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(m);
      work.Setup(m);
      
      /* reduce to upper Hessenberg form */
      SCHURCHEB_BLASLAPACK_DGEHRD( &m, &one, &m, A.GetData(), &ldim_A, tau.GetData()+1, work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
      
      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++) 
      {
         for (j = i+2; j < m; j++) 
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0; 
         }
      }
      
      /* generate matrix Q */
      tau[0] = 0.0;
      SCHURCHEB_BLASLAPACK_DORGQR(&m, &m, &m, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<double> &A, int start, int end, DenseMatrixClass<double> &Q)
   {
      int               i, j;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A      = A.GetLeadingDimension();
      int               ilo, ihi;
      vector_seq_double work;             //working array
      vector_seq_double tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(m);
      work.Setup(m);
      
      /* reduce to upper Hessenberg form 
       * A is already upper triangular in rows and columns 1:ILO-1 and IHI+1:N
       */
      ilo = start + 1;
      ihi = end;
      SCHURCHEB_BLASLAPACK_DGEHRD( &m, &ilo, &ihi, A.GetData(), &ldim_A, tau.GetData()+1, work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
      
      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++) 
      {
         for (j = i+2; j < m; j++) 
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0; 
         }
      }
      
      /* generate matrix Q */
      tau[0] = 0.0;
      SCHURCHEB_BLASLAPACK_DORGQR(&m, &m, &m, Q.GetData(), &ldim_Q, tau.GetData(), work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumColsLocal();
      int                  ldim_A      = A.GetLeadingDimension();
      int                  one         = 1;  //H is not already upper triangular in any part
      vector_seq_complexs  work;             //working array
      vector_seq_complexs  tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(m);
      work.Setup(m);
      
      /* reduce to upper Hessenberg form */
      SCHURCHEB_BLASLAPACK_CGEHRD( &m, &one, &m, SCHURCHEB_CAST( ccomplexs*, A.GetData()), &ldim_A, SCHURCHEB_CAST( ccomplexs*, tau.GetData()+1), 
                                 SCHURCHEB_CAST( ccomplexs*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
      
      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++) 
      {
         for (j = i+2; j < m; j++) 
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0; 
         }
      }
      
      /* generate matrix Q */
      tau[0] = complexs(0.0,0.0);
      SCHURCHEB_BLASLAPACK_CUNGQR(&m, &m, &m, SCHURCHEB_CAST( ccomplexs*, Q.GetData()), &ldim_Q, SCHURCHEB_CAST( ccomplexs*, tau.GetData()), 
                                 SCHURCHEB_CAST( ccomplexs*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexs> &A, int start, int end, DenseMatrixClass<complexs> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumColsLocal();
      int                  ldim_A      = A.GetLeadingDimension();
      int                  ilo, ihi;
      vector_seq_complexs  work;             //working array
      vector_seq_complexs  tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(m);
      work.Setup(m);
      
      /* reduce to upper Hessenberg form 
       * A is already upper triangular in rows and columns 1:ILO-1 and IHI+1:N
       */
      ilo = start + 1;
      ihi = end;
      SCHURCHEB_BLASLAPACK_CGEHRD( &m, &ilo, &ihi, SCHURCHEB_CAST( ccomplexs*, A.GetData()), &ldim_A, SCHURCHEB_CAST( ccomplexs*, tau.GetData()+1), 
                                 SCHURCHEB_CAST( ccomplexs*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
      
      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++) 
      {
         for (j = i+2; j < m; j++) 
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0; 
         }
      }
      
      /* generate matrix Q */
      tau[0] = complexs(0.0,0.0);
      SCHURCHEB_BLASLAPACK_CUNGQR(&m, &m, &m, SCHURCHEB_CAST( ccomplexs*, Q.GetData()), &ldim_Q, SCHURCHEB_CAST( ccomplexs*, tau.GetData()), 
                                 SCHURCHEB_CAST( ccomplexs*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumColsLocal();
      int                  ldim_A      = A.GetLeadingDimension();
      int                  one         = 1;  //H is not already upper triangular in any part
      vector_seq_complexd  work;             //working array
      vector_seq_complexd  tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(m);
      work.Setup(m);
      
      /* reduce to upper Hessenberg form */
      SCHURCHEB_BLASLAPACK_ZGEHRD( &m, &one, &m, SCHURCHEB_CAST( ccomplexd*, A.GetData()), &ldim_A, SCHURCHEB_CAST( ccomplexd*, tau.GetData()+1), 
                                 SCHURCHEB_CAST( ccomplexd*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
      
      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++) 
      {
         for (j = i+2; j < m; j++) 
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0; 
         }
      }
      
      /* generate matrix Q */
      tau[0] = complexd(0.0,0.0);
      SCHURCHEB_BLASLAPACK_ZUNGQR(&m, &m, &m, SCHURCHEB_CAST( ccomplexd*, Q.GetData()), &ldim_Q, SCHURCHEB_CAST( ccomplexd*, tau.GetData()), 
                                 SCHURCHEB_CAST( ccomplexd*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixHessDecompositionHost( DenseMatrixClass<complexd> &A, int start, int end, DenseMatrixClass<complexd> &Q)
   {
      int                  i, j;
      int                  info        = 0;
      int                  m           = A.GetNumColsLocal();
      int                  ldim_A      = A.GetLeadingDimension();
      int                  ilo, ihi;
      vector_seq_complexd  work;             //working array
      vector_seq_complexd  tau;
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q      = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(m == 0)
      {
         /* do nothing when A is empty */
         return SCHURCHEB_SUCCESS;
      }
      
      tau.Setup(m);
      work.Setup(m);
      
      /* reduce to upper Hessenberg form 
       * A is already upper triangular in rows and columns 1:ILO-1 and IHI+1:N
       */
      ilo = start + 1;
      ihi = end;
      SCHURCHEB_BLASLAPACK_ZGEHRD( &m, &ilo, &ihi, SCHURCHEB_CAST( ccomplexd*, A.GetData()), &ldim_A, SCHURCHEB_CAST( ccomplexd*, tau.GetData()+1), 
                                 SCHURCHEB_CAST( ccomplexd*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
      
      Q.Fill(0.0);
      /* copy data */
      for (i = 0; i < m-1; i++) 
      {
         for (j = i+2; j < m; j++) 
         {
            Q(j,i+1) = A(j,i);
            A(j,i) = 0.0; 
         }
      }
      
      /* generate matrix Q */
      tau[0] = complexd(0.0,0.0);
      SCHURCHEB_BLASLAPACK_ZUNGQR(&m, &m, &m, SCHURCHEB_CAST( ccomplexd*, Q.GetData()), &ldim_Q, SCHURCHEB_CAST( ccomplexd*, tau.GetData()), 
                                 SCHURCHEB_CAST( ccomplexd*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      tau.Clear();
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int              ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     one         = 1;  //H is not already upper triangular in any part
      vector_seq_float        work;           //work array
      
      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      work.Setup(m);
      
      SCHURCHEB_BLASLAPACK_SHSEQR(&job_schur, &compz, &m, &one, &m, A.GetData(), &ldim_A, 
            wr.GetData(), wi.GetData(), Q.GetData(), &ldim_Q, work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
      
      /* deallocate */
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<float> &A, int start, int end, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int              ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     ilo, ihi;
      vector_seq_float        work;           //work array
      
      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      work.Setup(m);
      
      /* note that if there are 2 by 2 blocks on 1:ilo-1 and ihi+1:m, 
       * wr and wi would be inaccuarte on those entries 
       */
      ilo = start + 1;
      ihi = end;
      SCHURCHEB_BLASLAPACK_SHSEQR(&job_schur, &compz, &m, &ilo, &ihi, A.GetData(), &ldim_A, 
            wr.GetData(), wi.GetData(), Q.GetData(), &ldim_Q, work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
      
      /* deallocate */
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int              ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     one         = 1;  //H is not already upper triangular in any part
      vector_seq_double       work;           //work array
      
      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      work.Setup(m);
      
      SCHURCHEB_BLASLAPACK_DHSEQR(&job_schur, &compz, &m, &one, &m, A.GetData(), &ldim_A, 
            wr.GetData(), wi.GetData(), Q.GetData(), &ldim_Q, work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
      
      /* deallocate */
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixRealHessSchurDecompositionHost( DenseMatrixClass<double> &A, int start, int end, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int              ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     ilo, ihi;
      vector_seq_double       work;           //work array
      
      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      work.Setup(m);
      
      /* note that if there are 2 by 2 blocks on 1:ilo-1 and ihi+1:m, 
       * wr and wi would be inaccuarte on those entries 
       */
      ilo = start + 1;
      ihi = end;
      SCHURCHEB_BLASLAPACK_DHSEQR(&job_schur, &compz, &m, &ilo, &ihi, A.GetData(), &ldim_A, 
            wr.GetData(), wi.GetData(), Q.GetData(), &ldim_Q, work.GetData(), &m, &info); SCHURCHEB_CHKERR(info);
      
      /* deallocate */
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int              ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     one         = 1;  //H is not already upper triangular in any part
      vector_seq_complexs     work;           //work array
      
      /* allocate memory */
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      // work array, just use max(1,n)
      work.Setup(m);
      
      SCHURCHEB_BLASLAPACK_CHSEQR(&job_schur, &compz, &m, &one, &m, SCHURCHEB_CAST(ccomplexs*, A.GetData()),
            &ldim_A, SCHURCHEB_CAST(ccomplexs*, w.GetData()), SCHURCHEB_CAST(ccomplexs*, Q.GetData()), 
            &ldim_Q, SCHURCHEB_CAST(ccomplexs*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
      
      /* deallocate */
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexs> &A, int start, int end, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int              ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     ilo, ihi;
      vector_seq_complexs     work;           //work array
      
      /* allocate memory */
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      // work array, just use max(1,n)
      work.Setup(m);
      
      ilo = start + 1;
      ihi = end;
      SCHURCHEB_BLASLAPACK_CHSEQR(&job_schur, &compz, &m, &ilo, &ihi, SCHURCHEB_CAST(ccomplexs*, A.GetData()),
            &ldim_A, SCHURCHEB_CAST(ccomplexs*, w.GetData()), SCHURCHEB_CAST(ccomplexs*, Q.GetData()), 
            &ldim_Q, SCHURCHEB_CAST(ccomplexs*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
      
      /* deallocate */
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int              ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     one         = 1;  //H is not already upper triangular in any part
      vector_seq_complexd     work;           //work array
      
      /* allocate memory */
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      // work array, just use max(1,n)
      work.Setup(m);
      
      SCHURCHEB_BLASLAPACK_ZHSEQR(&job_schur, &compz, &m, &one, &m, SCHURCHEB_CAST(ccomplexd*, A.GetData()),
            &ldim_A, SCHURCHEB_CAST(ccomplexd*, w.GetData()), SCHURCHEB_CAST(ccomplexd*, Q.GetData()), 
            &ldim_Q, SCHURCHEB_CAST(ccomplexd*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
      
      /* deallocate */
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixComplexHessSchurDecompositionHost( DenseMatrixClass<complexd> &A, int start, int end, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w)
   {
      int              info   = 0;
      int              m      = A.GetNumColsLocal();
      int              ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int              ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      /* Lapack params */
      char                    job_schur   = 'S';//eigenvalue and Schur form
      char                    compz       = 'I';//Initial U1 to be unit matrix
      int                     ilo, ihi;
      vector_seq_complexd     work;           //work array
      
      /* allocate memory */
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Matrix decomposition in host only.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      // work array, just use max(1,n)
      work.Setup(m);
      
      ilo = start + 1;
      ihi = end;
      SCHURCHEB_BLASLAPACK_ZHSEQR(&job_schur, &compz, &m, &ilo, &ihi, SCHURCHEB_CAST(ccomplexd*, A.GetData()),
            &ldim_A, SCHURCHEB_CAST(ccomplexd*, w.GetData()), SCHURCHEB_CAST(ccomplexd*, Q.GetData()), 
            &ldim_Q, SCHURCHEB_CAST(ccomplexd*, work.GetData()), &m, &info); SCHURCHEB_CHKERR(info);
      
      /* deallocate */
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixRealHessEigenDecompositionHost( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi)
   {
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( wr.GetLengthLocal() != m);
      SCHURCHEB_CHKERR( wi.GetLengthLocal() != m);
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }
      
      /* Lapack paras */
      int               mmm         = m;
      char              side        = 'R';//only compute right eigenvalues
      char              eigsrc      = 'Q';//Eigenvalues were computed from DHSEQR
      char              initv       = 'N';//No initial vectors
      vector_int        select;              //Fortran LOGICAL array, select which eigenvalue to be computed.
      vector_int        ifailr;              //Check convergence of DHSEIN
      
      int               mm;                  //# of cols in eigenvector matrix
      vector_seq_float  work;                //work array
      vector_int        iwork;
      int               one         = 1;  //H is not already upper triangular in any part
      
      /* select array, select all of them so set to 1 */
      select.Setup(m);
      select.Fill(1);
      
      /* work array */
      work.Setup((m+2)*m);
      /* fail array */
      ifailr.Setup(m);
      
      mm = m;
      
      SCHURCHEB_BLASLAPACK_SHSEIN(&side, &eigsrc, &initv, select.GetData(), &m, 
                                 A.GetData(), &ldim_A, wr.GetData(), wi.GetData(), NULL, &one, 
                                 Q.GetData(), &ldim_Q, &mm, &mmm, work.GetData(), NULL, ifailr.GetData(), &info); SCHURCHEB_CHKERR(info);
      
      //deallocate
      work.Clear();
      ifailr.Clear();
      select.Clear();
            
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixRealHessEigenDecompositionHost( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi)
   {
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( wr.GetLengthLocal() != m);
      SCHURCHEB_CHKERR( wi.GetLengthLocal() != m);
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }
      
      /* Lapack paras */
      int               mmm         = m;
      char              side        = 'R';//only compute right eigenvalues
      char              eigsrc      = 'Q';//Eigenvalues were computed from DHSEQR
      char              initv       = 'N';//No initial vectors
      vector_int        select;              //Fortran LOGICAL array, select which eigenvalue to be computed.
      vector_int        ifailr;              //Check convergence of DHSEIN
      
      int               mm;                  //# of cols in eigenvector matrix
      vector_seq_double work;                //work array
      vector_int        iwork;
      int               one         = 1;  //H is not already upper triangular in any part
      
      /* select array, select all of them so set to 1 */
      select.Setup(m);
      select.Fill(1);
      
      /* work array */
      work.Setup((m+2)*m);
      /* fail array */
      ifailr.Setup(m);
      
      mm = m;
      
      SCHURCHEB_BLASLAPACK_DHSEIN(&side, &eigsrc, &initv, select.GetData(), &m, 
                                 A.GetData(), &ldim_A, wr.GetData(), wi.GetData(), NULL, &one, 
                                 Q.GetData(), &ldim_Q, &mm, &mmm, work.GetData(), NULL, ifailr.GetData(), &info); SCHURCHEB_CHKERR(info);
      
      //deallocate
      work.Clear();
      ifailr.Clear();
      select.Clear();
            
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixComplexHessEigenDecompositionHost( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w)
   {
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( w.GetLengthLocal() != m);
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }
      
      /* Lapack paras */
      int                  mmm         = m;
      char                 side        = 'R';//only compute right eigenvalues
      char                 eigsrc      = 'Q';//Eigenvalues were computed from DHSEQR
      char                 initv       = 'N';//No initial vectors
      vector_int           select;              //Fortran LOGICAL array, select which eigenvalue to be computed.
      vector_int           ifailr;              //Check convergence of DHSEIN
      
      int                  mm;                  //# of cols in eigenvector matrix
      vector_seq_float     rwork;                //work array
      vector_int           iwork;
      vector_seq_complexs  work;           //work array
      int                  one         = 1;  //H is not already upper triangular in any part
      
      // select array, select all of them so set to 1
      select.Setup(m);
      select.Fill(1);
      
      // work array
      work.Setup(m*m);
      rwork.Setup(m);
      // fail array
      ifailr.Setup(m);
      
      mm = m;
      
      SCHURCHEB_BLASLAPACK_CHSEIN(&side, &eigsrc, &initv, select.GetData(), &m,
                              SCHURCHEB_CAST(ccomplexs*, A.GetData()), &ldim_A, SCHURCHEB_CAST(ccomplexs*, w.GetData()), 
                              NULL, &one, SCHURCHEB_CAST(ccomplexs*, Q.GetData()), &ldim_Q, 
                              &mm, &mmm, SCHURCHEB_CAST(ccomplexs*, work.GetData()), rwork.GetData(),
                              NULL, ifailr.GetData(), &info); SCHURCHEB_CHKERR(info);
      
      //deallocate
      work.Clear();
      rwork.Clear();
      ifailr.Clear();
      select.Clear();
            
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixComplexHessEigenDecompositionHost( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w)
   {
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               ldim_A = A.GetLeadingDimension();
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( w.GetLengthLocal() != m);
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if( Q.GetNumRowsLocal() != m || Q.GetNumColsLocal() != m )
      {
         Q.Setup( m, m, kMemoryHost, false);
      }
      
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }
      
      /* Lapack paras */
      int                  mmm         = m;
      char                 side        = 'R';//only compute right eigenvalues
      char                 eigsrc      = 'Q';//Eigenvalues were computed from DHSEQR
      char                 initv       = 'N';//No initial vectors
      vector_int           select;              //Fortran LOGICAL array, select which eigenvalue to be computed.
      vector_int           ifailr;              //Check convergence of DHSEIN
      
      int                  mm;                  //# of cols in eigenvector matrix
      vector_seq_double    rwork;                //work array
      vector_int           iwork;
      vector_seq_complexd  work;           //work array
      int                  one         = 1;  //H is not already upper triangular in any part
      
      // select array, select all of them so set to 1
      select.Setup(m);
      select.Fill(1);
      
      // work array
      work.Setup(m*m);
      rwork.Setup(m);
      // fail array
      ifailr.Setup(m);
      
      mm = m;
      
      SCHURCHEB_BLASLAPACK_ZHSEIN(&side, &eigsrc, &initv, select.GetData(), &m,
                              SCHURCHEB_CAST(ccomplexd*, A.GetData()), &ldim_A, SCHURCHEB_CAST(ccomplexd*, w.GetData()), 
                              NULL, &one, SCHURCHEB_CAST(ccomplexd*, Q.GetData()), &ldim_Q, 
                              &mm, &mmm, SCHURCHEB_CAST(ccomplexd*, work.GetData()), rwork.GetData(),
                              NULL, ifailr.GetData(), &info); SCHURCHEB_CHKERR(info);
      
      //deallocate
      work.Clear();
      rwork.Clear();
      ifailr.Clear();
      select.Clear();
            
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixRealOrderSchur( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi, vector_int &select)
   {
      /* pre processing the select array */
      int                  info         = 0;
      int                  m           = A.GetNumColsLocal();
      int                  mm;             // out put, dim of the invariant subspace
      vector_seq_float     work;           //work array
      vector_int           iwork;
      int                  lwork, liwork;          // length of working space
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumColsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( select.GetLengthLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }
      
      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }
      
      if(select.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         select.MoveData(kMemoryHost);
      }
      
      char                    job         = 'N';  // condition numbers not required
      char                    compq       = 'V';  // update the matrix 
      
      lwork = m;
      work.Setup(lwork);
      liwork = 1;
      iwork.Setup(liwork);
      
      SCHURCHEB_BLASLAPACK_STRSEN( &job, &compq, select.GetData(), &m, A.GetData(), &ldim_A, 
               Q.GetData(), &ldim_Q, wr.GetData(), wi.GetData(), &mm, NULL, NULL, work.GetData(), 
               &lwork, iwork.GetData(), &liwork, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      work.Clear();
      iwork.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixRealOrderSchur( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi, vector_int &select)
   {
      /* pre processing the select array */
      int                  info         = 0;
      int                  m           = A.GetNumColsLocal();
      int                  mm;             // out put, dim of the invariant subspace
      vector_seq_double    work;           //work array
      vector_int           iwork;
      int                  lwork, liwork;          // length of working space
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumColsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( select.GetLengthLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }
      
      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }
      
      if(select.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         select.MoveData(kMemoryHost);
      }
      
      char                    job         = 'N';  // condition numbers not required
      char                    compq       = 'V';  // update the matrix 
      
      lwork = m;
      work.Setup(lwork);
      liwork = 1;
      iwork.Setup(liwork);
      
      SCHURCHEB_BLASLAPACK_DTRSEN( &job, &compq, select.GetData(), &m, A.GetData(), &ldim_A, 
               Q.GetData(), &ldim_Q, wr.GetData(), wi.GetData(), &mm, NULL, NULL, work.GetData(), 
               &lwork, iwork.GetData(), &liwork, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      work.Clear();
      iwork.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixComplexOrderSchur( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w, vector_int &select)
   {
      /* pre processing the select array */
      int                  info         = 0;
      int                  m           = A.GetNumColsLocal();
      int                  mm;             // out put, dim of the invariant subspace
      vector_seq_complexs  work;           //work array
      int                  lwork;          // length of working space
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumColsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( select.GetLengthLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }
      
      if(select.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         select.MoveData(kMemoryHost);
      }
      
      char                    job         = 'N';  // condition numbers not required
      char                    compq       = 'V';  // update the matrix 
      
      lwork = m;
      work.Setup(lwork);
      
      SCHURCHEB_BLASLAPACK_CTRSEN(&job, &compq, select.GetData(), &m, SCHURCHEB_CAST( ccomplexs*, A.GetData()), &ldim_A,
                                 SCHURCHEB_CAST( ccomplexs*, Q.GetData()), &ldim_Q, SCHURCHEB_CAST( ccomplexs*, w.GetData()),
                                 &mm, NULL, NULL, SCHURCHEB_CAST( ccomplexs*, work.GetData()), &lwork, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixComplexOrderSchur( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w, vector_int &select)
   {
      /* pre processing the select array */
      int                  info         = 0;
      int                  m           = A.GetNumColsLocal();
      int                  mm;             // out put, dim of the invariant subspace
      vector_seq_complexd  work;           //work array
      int                  lwork;          // length of working space
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumColsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( select.GetLengthLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }
      
      if(select.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         select.MoveData(kMemoryHost);
      }
      
      char                    job         = 'N';  // condition numbers not required
      char                    compq       = 'V';  // update the matrix 
      
      lwork = m;
      work.Setup(lwork);
      
      SCHURCHEB_BLASLAPACK_ZTRSEN(&job, &compq, select.GetData(), &m, SCHURCHEB_CAST( ccomplexd*, A.GetData()), &ldim_A,
                                 SCHURCHEB_CAST( ccomplexd*, Q.GetData()), &ldim_Q, SCHURCHEB_CAST( ccomplexd*, w.GetData()),
                                 &mm, NULL, NULL, SCHURCHEB_CAST( ccomplexd*, work.GetData()), &lwork, &info); SCHURCHEB_CHKERR(info);
            
      /* deallocate */
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixRealOrderSchurClusters( DenseMatrixClass<float> &A, DenseMatrixClass<float> &Q, vector_seq_float &wr, vector_seq_float &wi, vector_int &clusters)
   {
      /* pre processing the select array */
      int               i, j;
      int               ifst, ilst;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               idx;
      int               temp_idx;
      int               case_number = 0;
      vector_int        order;
      vector_int        iorder; // working array
      vector_seq_float  work;
      float             eps = std::numeric_limits<float>::epsilon();
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumColsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( clusters.GetLengthLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }
      
      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }
      
      if(clusters.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         clusters.MoveData(kMemoryHost);
      }
      
      // condition numbers not required
      // update the matrix 
      char                    compq       = 'V';
      
      /* not working yet, the stable sort */
      clusters.Sort( order, false, true);
      
      iorder.Setup(m);
      
      for(i = 0 ; i < m ; i ++)
      {
         iorder[order[i]] = i;
      }
      
      work.Setup(m);
      
      i = 0;
      
      while(true)
      {
         idx = order[i];
         if(clusters[idx] <= 0)
         {
            break;
         }
         
         case_number = 0;
         /* we use the stable sort, so the order of the 2x2 block kept */
         if(idx != i)
         {
            
            /* need to insert */
            if(idx < m-1)
            {
               if(i < m-1)
               {
                  /* both not the last one, check if they are 2 by 2 blocks */
                  if( SchurchebAbs( A(idx+1, idx) ) > eps )
                  {
                     /* in this case, a 2 by 2 block is found at idx */
                     if( SchurchebAbs( A(i+1, i) ) > eps )
                     {
                        /* in this case, a 2 by 2 block is found at i 
                         * replace a 2 by 2 block with another 2 by 2 block
                         * swap idx with i, and idx+1 with i+1
                         */
                        
                        case_number = 22; 
                        
                     }
                     else
                     {
                        /* in this case, a 1 by 1 block is found at i 
                         * replace a 1 by 1 block with a 2 by 2 block
                         */
                        
                        case_number = 12; 
                        
                     }
                  }
                  else
                  {
                     /* in this case, a 1 by 1 block is found at idx */
                     if( SchurchebAbs( A(i+1, i) ) > eps )
                     {
                        /* in this case, a 2 by 2 block is found at i 
                         * replace a 2 by 2 block with a 1 by 1 block
                         */
                        
                        case_number = 21;
                        
                     }
                     else
                     {
                        /* in this case, a 1 by 1 block is found at i 
                         * both 1 by 1
                         */
                        
                        case_number = 11;
                        
                     }
                  }
               }
               else
               {
                  /* i is the last one, 1 by 1 block */
                  if( SchurchebAbs( A(idx+1, idx) ) > eps )
                  {
                     /* in this case, a 2 by 2 block is found at idx 
                      * replace 1 by 1 with 2 by 2
                      */
                     
                     case_number = 12;
                     
                  }
                  else
                  {
                     /* in this case, a 1 by 1 block is found at idx 
                      * both 1 by 1
                      */
                     
                     case_number = 11;
                     
                  }
               }
            }
            else
            {
               /* idx is the last one, 1 by 1 block */
               if( SchurchebAbs( A(i+1, i) ) > eps )
               {
                  /* in this case, a 2 by 2 block is found at i 
                   * replace 2 by 2 with 1 by 1
                   */
                  
                  case_number = 21;
                  
               }
               else
               {
                  /* in this case, a 1 by 1 block is found at i 
                   * both 1 by 1
                   */
                  case_number = 11;
               }
            }
            
            /* apply the insert */
            ifst = idx+1;
            ilst = i+1;
            SCHURCHEB_BLASLAPACK_STREXC(&compq, &m, A.GetData(), &ldim_A, Q.GetData(), &ldim_Q, &ifst, &ilst, work.GetData(), &info); SCHURCHEB_CHKERR(info);
            
            switch (case_number)
            {
               case 11:
               {
                  /* if we have i, i+1, ... , i+m, idx, and order[k0] = i, order[k1] = i+1, ..., order[km] = i+m,
                   * after swap we'll have order[k0] = i+1, order[k1] = i+2, ..., order[km] = idx
                   */
                  
                  for(j = idx-1 ; j >= i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+1;
                     iorder[j+1] = temp_idx;
                  }
                  i++;
                  
                  break;
               }
               case 12:
               {
                  
                  /* if we have i, i+1, ... , i+m, idx, idx+1, and order[k0] = i, order[k1] = i+1, ..., order[km] = i+m,
                   * after swap we'll have order[k0] = i+2, order[k1] = i+3, ..., order[km] = idx+1
                   */
                  
                  for(j = idx-1 ; j > i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+2;
                     iorder[j+2] = temp_idx;
                  }
                  i+=2;
                  
                  break;
               }
               case 21:
               {
                  
                  /* in this case same as 11, since we can guarentee that i is not the second of a 2 by 2 block */
                  
                  for(j = idx-1 ; j >= i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+1;
                     iorder[j+1] = temp_idx;
                  }
                  i++;
                  
                  break;
               }
               case 22:
               {
                  
                  /* in this case same as 12, since we can guarentee that i is not the second of a 2 by 2 block */
                  
                  for(j = idx-1 ; j > i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+2;
                     iorder[j+2] = temp_idx;
                  }
                  i+=2;
                  
                  break;
               }
               default:
               {
                  return SCHURCHEB_ERROR_INVALED_PARAM;
               }
               
            }
            
         }
         else
         {
            /* no need to insert */
            if(idx < m-1)
            {
               /* both not the last one, check if they are 2 by 2 blocks */
               if( SchurchebAbs( A(idx+1, idx) ) > eps )
               {
                  i+=2;
               }
               else
               {
                  i++;
               }
            }
            else
            {
               i++;
            }
         }
         
         if(i >= m)
         {
            break;
         }
         
      }
      
      for(i = 0 ; i < m ; i ++)
      {
         if(i < m-1)
         {
            /* both not the last one, check if they are 2 by 2 blocks */
            if( SchurchebAbs( A(i+1, i) ) > eps )
            {
               wr[i] = A(i, i);
               wr[i+1] = A(i, i);
               wi[i] = sqrt(-A(i+1,i)*A(i,i+1));
               wi[i+1] = -wi[i];
               i++;
            }
            else
            {
               wr[i] = A(i, i);
               wi[i] = 0.0f;
            }
         }
         else
         {
            wr[i] = A(i, i);
            wi[i] = 0.0f;
         }
      }
      
      //deallocate
      order.Clear();
      iorder.Clear();
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixRealOrderSchurClusters( DenseMatrixClass<double> &A, DenseMatrixClass<double> &Q, vector_seq_double &wr, vector_seq_double &wi, vector_int &clusters)
   {
      /* pre processing the select array */
      int               i, j;
      int               ifst, ilst;
      int               info        = 0;
      int               m           = A.GetNumColsLocal();
      int               idx;
      int               temp_idx;
      int               case_number = 0;
      vector_int        order;
      vector_int        iorder; // working array
      vector_seq_double work;
      float             eps = std::numeric_limits<double>::epsilon();
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumColsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( clusters.GetLengthLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(wr.GetLengthLocal() != m)
      {
         wr.Setup(m, kMemoryHost, false);
      }
      if(wr.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wr.MoveData(kMemoryHost);
      }
      
      if(wi.GetLengthLocal() != m)
      {
         wi.Setup(m, kMemoryHost, false);
      }
      if(wi.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         wi.MoveData(kMemoryHost);
      }
      
      if(clusters.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         clusters.MoveData(kMemoryHost);
      }
      
      // condition numbers not required
      // update the matrix 
      char                    compq       = 'V';
      
      /* not working yet, the stable sort */
      clusters.Sort( order, false, true);
      
      iorder.Setup(m);
      
      for(i = 0 ; i < m ; i ++)
      {
         iorder[order[i]] = i;
      }
      
      work.Setup(m);
      
      i = 0;
      
      while(true)
      {
         idx = order[i];
         if(clusters[idx] <= 0)
         {
            break;
         }
         
         case_number = 0;
         /* we use the stable sort, so the order of the 2x2 block kept */
         if(idx != i)
         {
            
            /* need to insert */
            if(idx < m-1)
            {
               if(i < m-1)
               {
                  /* both not the last one, check if they are 2 by 2 blocks */
                  if( SchurchebAbs( A(idx+1, idx) ) > eps )
                  {
                     /* in this case, a 2 by 2 block is found at idx */
                     if( SchurchebAbs( A(i+1, i) ) > eps )
                     {
                        /* in this case, a 2 by 2 block is found at i 
                         * replace a 2 by 2 block with another 2 by 2 block
                         * swap idx with i, and idx+1 with i+1
                         */
                        
                        case_number = 22; 
                        
                     }
                     else
                     {
                        /* in this case, a 1 by 1 block is found at i 
                         * replace a 1 by 1 block with a 2 by 2 block
                         */
                        
                        case_number = 12; 
                        
                     }
                  }
                  else
                  {
                     /* in this case, a 1 by 1 block is found at idx */
                     if( SchurchebAbs( A(i+1, i) ) > eps )
                     {
                        /* in this case, a 2 by 2 block is found at i 
                         * replace a 2 by 2 block with a 1 by 1 block
                         */
                        
                        case_number = 21;
                        
                     }
                     else
                     {
                        /* in this case, a 1 by 1 block is found at i 
                         * both 1 by 1
                         */
                        
                        case_number = 11;
                        
                     }
                  }
               }
               else
               {
                  /* i is the last one, 1 by 1 block */
                  if( SchurchebAbs( A(idx+1, idx) ) > eps )
                  {
                     /* in this case, a 2 by 2 block is found at idx 
                      * replace 1 by 1 with 2 by 2
                      */
                     
                     case_number = 12;
                     
                  }
                  else
                  {
                     /* in this case, a 1 by 1 block is found at idx 
                      * both 1 by 1
                      */
                     
                     case_number = 11;
                     
                  }
               }
            }
            else
            {
               /* idx is the last one, 1 by 1 block */
               if( SchurchebAbs( A(i+1, i) ) > eps )
               {
                  /* in this case, a 2 by 2 block is found at i 
                   * replace 2 by 2 with 1 by 1
                   */
                  
                  case_number = 21;
                  
               }
               else
               {
                  /* in this case, a 1 by 1 block is found at i 
                   * both 1 by 1
                   */
                  case_number = 11;
               }
            }
            
            /* apply the insert */
            ifst = idx+1;
            ilst = i+1;
            SCHURCHEB_BLASLAPACK_DTREXC(&compq, &m, A.GetData(), &ldim_A, Q.GetData(), &ldim_Q, &ifst, &ilst, work.GetData(), &info); SCHURCHEB_CHKERR(info);
            
            switch (case_number)
            {
               case 11:
               {
                  /* if we have i, i+1, ... , i+m, idx, and order[k0] = i, order[k1] = i+1, ..., order[km] = i+m,
                   * after swap we'll have order[k0] = i+1, order[k1] = i+2, ..., order[km] = idx
                   */
                  
                  for(j = idx-1 ; j >= i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+1;
                     iorder[j+1] = temp_idx;
                  }
                  i++;
                  
                  break;
               }
               case 12:
               {
                  
                  /* if we have i, i+1, ... , i+m, idx, idx+1, and order[k0] = i, order[k1] = i+1, ..., order[km] = i+m,
                   * after swap we'll have order[k0] = i+2, order[k1] = i+3, ..., order[km] = idx+1
                   */
                  
                  for(j = idx-1 ; j > i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+2;
                     iorder[j+2] = temp_idx;
                  }
                  i+=2;
                  
                  break;
               }
               case 21:
               {
                  
                  /* in this case same as 11, since we can guarentee that i is not the second of a 2 by 2 block */
                  
                  for(j = idx-1 ; j >= i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+1;
                     iorder[j+1] = temp_idx;
                  }
                  i++;
                  
                  break;
               }
               case 22:
               {
                  
                  /* in this case same as 12, since we can guarentee that i is not the second of a 2 by 2 block */
                  
                  for(j = idx-1 ; j > i ; j --)
                  {
                     temp_idx = iorder[j];
                     order[temp_idx] = j+2;
                     iorder[j+2] = temp_idx;
                  }
                  i+=2;
                  
                  break;
               }
               default:
               {
                  return SCHURCHEB_ERROR_INVALED_PARAM;
               }
               
            }
            
         }
         else
         {
            /* no need to insert */
            if(idx < m-1)
            {
               /* both not the last one, check if they are 2 by 2 blocks */
               if( SchurchebAbs( A(idx+1, idx) ) > eps )
               {
                  i+=2;
               }
               else
               {
                  i++;
               }
            }
            else
            {
               i++;
            }
         }
         
         if(i >= m)
         {
            break;
         }
         
      }
      
      for(i = 0 ; i < m ; i ++)
      {
         if(i < m-1)
         {
            /* both not the last one, check if they are 2 by 2 blocks */
            if( SchurchebAbs( A(i+1, i) ) > eps )
            {
               wr[i] = A(i, i);
               wr[i+1] = A(i, i);
               wi[i] = sqrt(-A(i+1,i)*A(i,i+1));
               wi[i+1] = -wi[i];
               i++;
            }
            else
            {
               wr[i] = A(i, i);
               wi[i] = 0.0f;
            }
         }
         else
         {
            wr[i] = A(i, i);
            wi[i] = 0.0f;
         }
      }
      
      //deallocate
      order.Clear();
      iorder.Clear();
      work.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixComplexOrderSchurClusters( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &Q, vector_seq_complexs &w, vector_int &clusters)
   {
      /* pre processing the select array */
      int              i, j;
      int              ifst, ilst;
      int              info        = 0;
      int              m           = A.GetNumColsLocal();
      int              idx;
      int              temp_idx;
      vector_int       order;
      vector_int       iorder; // working array
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumColsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( clusters.GetLengthLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }
      
      if(clusters.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         clusters.MoveData(kMemoryHost);
      }
      
      // condition numbers not required
      // update the matrix 
      char                    compq       = 'V';
      
      /* not working yet, the stable sort */
      clusters.Sort( order, false, true);
      
      iorder.Setup(m);
      
      for(i = 0 ; i < m ; i ++)
      {
         iorder[order[i]] = i;
      }
      
      i = 0;
      
      while(true)
      {
         idx = order[i];
         
         if(clusters[idx] <= 0)
         {
            break;
         }
         
         if(idx != i)
         {
            ifst = idx+1;
            ilst = i+1;
            SCHURCHEB_BLASLAPACK_CTREXC(&compq, &m, SCHURCHEB_CAST( ccomplexs*, A.GetData()), &ldim_A, 
                                       SCHURCHEB_CAST( ccomplexs*, Q.GetData()), &ldim_Q, &ifst, &ilst, &info); SCHURCHEB_CHKERR(info);
            
            /* after the swap, if we have order[k] = i, we need to update order[k] = idx */
            
            for(j = idx-1 ; j >= i ; j --)
            {
               temp_idx = iorder[j];
               order[temp_idx] = j+1;
               iorder[j+1] = temp_idx;
            }
            
            /* we'll not touch index i anymore, no need to update order[i] and iorder[i] */
            
         }
         i++;
         
         if( i >= m)
         {
            break;
         }
         
      }
      
      for(i = 0 ; i < m ; i ++)
      {
         w[i] = A(i, i);
      }
      
      //deallocate
      order.Clear();
      iorder.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixComplexOrderSchurClusters( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &Q, vector_seq_complexd &w, vector_int &clusters)
   {
      /* pre processing the select array */
      int              i, j;
      int              ifst, ilst;
      int              info        = 0;
      int              m           = A.GetNumColsLocal();
      int              idx;
      int              temp_idx;
      vector_int       order;
      vector_int       iorder; // working array
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      SCHURCHEB_CHKERR( A.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumColsLocal() != m );
      SCHURCHEB_CHKERR( Q.GetNumRowsLocal() != m );
      SCHURCHEB_CHKERR( clusters.GetLengthLocal() != m );
      
      if(A.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         A.MoveData(kMemoryHost);
      }
      
      if(Q.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving matrix to the host."<<std::endl;
         Q.MoveData(kMemoryHost);
      }
      
      int               ldim_A = A.GetLeadingDimension();
      int               ldim_Q = Q.GetLeadingDimension();
      
      if(w.GetLengthLocal() != m)
      {
         w.Setup(m, kMemoryHost, false);
      }
      if(w.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         w.MoveData(kMemoryHost);
      }
      
      if(clusters.GetDataLocation() == kMemoryDevice)
      {
         std::cout<<"Matrix decomposition can only be done on the host currently. Moving vector to the host."<<std::endl;
         clusters.MoveData(kMemoryHost);
      }
      
      // condition numbers not required
      // update the matrix 
      char                    compq       = 'V';
      
      /* not working yet, the stable sort */
      clusters.Sort( order, false, true);
      
      iorder.Setup(m);
      
      for(i = 0 ; i < m ; i ++)
      {
         iorder[order[i]] = i;
      }
      
      i = 0;
      
      while(true)
      {
         idx = order[i];
         
         if(clusters[idx] <= 0)
         {
            break;
         }
         
         if(idx != i)
         {
            ifst = idx+1;
            ilst = i+1;
            SCHURCHEB_BLASLAPACK_ZTREXC(&compq, &m, SCHURCHEB_CAST( ccomplexd*, A.GetData()), &ldim_A, 
                                       SCHURCHEB_CAST( ccomplexd*, Q.GetData()), &ldim_Q, &ifst, &ilst, &info); SCHURCHEB_CHKERR(info);
            
            /* after the swap, if we have order[k] = i, we need to update order[k] = idx */
            
            for(j = idx-1 ; j >= i ; j --)
            {
               temp_idx = iorder[j];
               order[temp_idx] = j+1;
               iorder[j+1] = temp_idx;
            }
            
            /* we'll not touch index i anymore, no need to update order[i] and iorder[i] */
            
         }
         i++;
         
         if( i >= m)
         {
            break;
         }
         
      }
      
      for(i = 0 ; i < m ; i ++)
      {
         w[i] = A(i, i);
      }
      
      //deallocate
      order.Clear();
      iorder.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T>
   int DenseMatrixMatMatTemplate( const T &alpha, const DenseMatrixClass<T> &A, char transa, const DenseMatrixClass<T> &B, char transb, const T &beta, DenseMatrixClass<T> &C)
   {
      
      /* C = alpha*op(A)*op(B) + beta*C */
      int      i, j, k;
      T        *A_data, *B_data, *C_data, *A_data_i, *B_data_i, *C_data_i;
      T        *A_data_temp = NULL, *B_data_temp = NULL;
      T        temp_val;
      T        zero = T();
      T        one = T(1.0);
      int      A_ldim, B_ldim, C_ldim;
      int      A_nrow, B_nrow, C_nrow;
      int      A_ncol, B_ncol, C_ncol;
      
      A_data = A.GetData();
      B_data = B.GetData();
      C_data = C.GetData();
      
      A_ldim = A.GetLeadingDimension();
      B_ldim = B.GetLeadingDimension();
      C_ldim = C.GetLeadingDimension();
      
      A_nrow = A.GetNumRowsLocal();
      B_nrow = B.GetNumRowsLocal();
      C_nrow = C.GetNumRowsLocal();
      
      A_ncol = A.GetNumColsLocal();
      B_ncol = B.GetNumColsLocal();
      C_ncol = C.GetNumColsLocal();
      
      /* 1. Compute C = beta*C
       * note that if alpha != 0, when A == C we need to copy A, 
       * when B == C we need to copy B
       * TODO: memcpy or omp parallel?
       */
      
      if(alpha != zero)
      {
         if(A_data == C_data)
         {
            SCHURCHEB_MALLOC(A_data_temp, A_nrow * A_ncol, kMemoryHost, T);
            for(i = 0 ; i < A_ncol ; i ++)
            {
                j = i * A_ldim;
                k = i * A_nrow;
                SCHURCHEB_MEMCPY(A_data_temp+k, A_data+j, A_nrow, kMemoryHost, kMemoryHost, T);
            }
            A_data = A_data_temp;
            A_ldim = A_nrow;
         }
         if(B_data == C_data)
         {
            SCHURCHEB_MALLOC(B_data_temp, B_nrow * B_ncol, kMemoryHost, T);
            for(i = 0 ; i < B_ncol ; i ++)
            {
                j = i * B_ldim;
                k = i * B_nrow;
                SCHURCHEB_MEMCPY(B_data_temp+k, B_data+j, B_nrow, kMemoryHost, kMemoryHost, T);
            }
            B_data = B_data_temp;
            B_ldim = B_nrow;
         }
      }
      
      /* now scale C */
      if(beta != one)
      {
         /* when beta == 1.0, C = C, do nothing */
         if(beta != zero)
         {
            /* C = beta*C */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
            for(i = 0 ; i < C_ncol ; i ++)
            {
               for(j = 0, k = i*C_ldim ; j < C_nrow ; j++, k++)
               {
                  C_data[k] *= beta;
               }
            }
         }
         else
         {
            /* beta == 0.0, C = 0 */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
            for(i = 0 ; i < C_ncol ; i ++)
            {
               for(j = 0, k = i*C_ldim ; j < C_nrow ; j++, k++)
               {
                  C_data[k] = zero;
               }
            }
         }
      }
      
      /* 2. the matmat C = alpha*op(A)*op(B) + C
       * when alpha == 0 we have C = C, do nothing
       */
      
      if(alpha != zero)
      {
         if(alpha != one)
         {
            if(transa == 'N')
            {
               if(transb == 'N')
               {
                  /* alpha*A*B 
                   * Each time we pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_nrow ; k ++)
                     {
                        temp_val = alpha * B_data[j*B_ldim+k];
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A*B^T
                   * Each time we again pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_ncol ; k ++)
                     {
                        temp_val = alpha * B_data[k*B_ldim+j];
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A*B^H 
                   * Each time we again pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_ncol ; k ++)
                     {
                        temp_val = alpha * SchurchebConj(B_data[k*B_ldim+j]);
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else
               {
                  SCHURCHEB_ERROR("Unknown matrix operator.");
                  return SCHURCHEB_ERROR_INVALED_PARAM;
               }
            }
            else if(transa == 'T')
            {
               if(transb == 'N')
               {
                  /* alpha*A^T*B 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, B_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     B_data_i = B_data + j*B_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * B_data_i[k];
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A^T*B^T 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * B_data[k*B_ldim+j];
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A^T*B^T 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * SchurchebConj(B_data[k*B_ldim+j]);
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else
               {
                  SCHURCHEB_ERROR("Unknown matrix operator.");
                  return SCHURCHEB_ERROR_INVALED_PARAM;
               }
            }
            else if(transa == 'C')
            {
               if(transb == 'N')
               {
                  /* alpha*A^C*B 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, B_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     B_data_i = B_data + j*B_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += SchurchebConj(A_data_i[k]) * B_data_i[k];
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A^C*B^T 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += SchurchebConj(A_data_i[k]) * B_data[k*B_ldim+j];
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A^C*B^T 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += SchurchebConj(A_data_i[k]) * SchurchebConj(B_data[k*B_ldim+j]);
                        }
                        C_data_i[i] +=  alpha * temp_val;
                     }
                  }
               }
               else
               {
                  SCHURCHEB_ERROR("Unknown matrix operator.");
                  return SCHURCHEB_ERROR_INVALED_PARAM;
               }
            }
            else
            {
               SCHURCHEB_ERROR("Unknown matrix operator.");
               return SCHURCHEB_ERROR_INVALED_PARAM;
            }
         }
         else
         {
            if(transa == 'N')
            {
               if(transb == 'N')
               {
                  /* alpha*A*B 
                   * Each time we pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_nrow ; k ++)
                     {
                        temp_val = B_data[j*B_ldim+k];
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A*B^T
                   * Each time we again pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_ncol ; k ++)
                     {
                        temp_val = B_data[k*B_ldim+j];
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A*B^H 
                   * Each time we again pick an element from B
                   * multiply a column with A, and add to column of C
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     for(k = 0 ; k < B_ncol ; k ++)
                     {
                        temp_val = SchurchebConj(B_data[k*B_ldim+j]);
                        A_data_i = A_data + k*A_ldim;
                        C_data_i = C_data + j*C_ldim;
                        for(i = 0 ; i < C_nrow ; i ++)
                        {
                           C_data_i[i] += temp_val * A_data_i[i];
                        }
                     }
                  }
               }
               else
               {
                  SCHURCHEB_ERROR("Unknown matrix operator.");
                  return SCHURCHEB_ERROR_INVALED_PARAM;
               }
            }
            else if(transa == 'T')
            {
               if(transb == 'N')
               {
                  /* alpha*A^T*B 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, B_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     B_data_i = B_data + j*B_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * B_data_i[k];
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A^T*B^T 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * B_data[k*B_ldim+j];
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A^T*B^T 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += A_data_i[k] * SchurchebConj(B_data[k*B_ldim+j]);
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else
               {
                  SCHURCHEB_ERROR("Unknown matrix operator.");
                  return SCHURCHEB_ERROR_INVALED_PARAM;
               }
            }
            else if(transa == 'C')
            {
               if(transb == 'N')
               {
                  /* alpha*A^C*B 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, B_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     B_data_i = B_data + j*B_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += SchurchebConj(A_data_i[k]) * B_data_i[k];
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else if(transb == 'T')
               {
                  /* alpha*A^C*B^T 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += SchurchebConj(A_data_i[k]) * B_data[k*B_ldim+j];
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else if(transb == 'C')
               {
                  /* alpha*A^C*B^T 
                   * In this case, we can pick a row of A,
                   * multiple with a column of B
                   */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, k, temp_val, A_data_i, C_data_i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
                  for(j = 0 ; j < C_ncol ; j ++)
                  {
                     C_data_i = C_data + j*C_ldim;
                     
                     for(i = 0 ; i < C_nrow ; i ++)
                     {
                        A_data_i = A_data + i*A_ldim;
                        temp_val = zero;
                        for(k = 0 ; k < B_nrow ; k ++)
                        {
                           temp_val += SchurchebConj(A_data_i[k]) * SchurchebConj(B_data[k*B_ldim+j]);
                        }
                        C_data_i[i] +=  temp_val;
                     }
                  }
               }
               else
               {
                  SCHURCHEB_ERROR("Unknown matrix operator.");
                  return SCHURCHEB_ERROR_INVALED_PARAM;
               }
            }
            else
            {
               SCHURCHEB_ERROR("Unknown matrix operator.");
               return SCHURCHEB_ERROR_INVALED_PARAM;
            }
         }
      }
      
      if(A_data_temp)
      {
         SCHURCHEB_FREE( A_data_temp, kMemoryHost);
      }
      
      if(B_data_temp)
      {
         SCHURCHEB_FREE( B_data_temp, kMemoryHost);
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int DenseMatrixMatMatTemplate( const float &alpha, const DenseMatrixClass<float> &A, char transa, const DenseMatrixClass<float> &B, char transb, const float &beta, DenseMatrixClass<float> &C);
   template int DenseMatrixMatMatTemplate( const double &alpha, const DenseMatrixClass<double> &A, char transa, const DenseMatrixClass<double> &B, char transb, const double &beta, DenseMatrixClass<double> &C);
   template int DenseMatrixMatMatTemplate( const complexs &alpha, const DenseMatrixClass<complexs> &A, char transa, const DenseMatrixClass<complexs> &B, char transb, const complexs &beta, DenseMatrixClass<complexs> &C);
   template int DenseMatrixMatMatTemplate( const complexd &alpha, const DenseMatrixClass<complexd> &A, char transa, const DenseMatrixClass<complexd> &B, char transb, const complexd &beta, DenseMatrixClass<complexd> &C);
   
   int DenseMatrixMatMat( const float &alpha, const DenseMatrixClass<float> &A, char transa, const DenseMatrixClass<float> &B, char transb, const float &beta, DenseMatrixClass<float> &C)
   {
      int m, n, k;
      
      if(transa == 'N')
      {
         if(transb == 'N')
         {
            SCHURCHEB_CHKERR( A.GetNumColsLocal() != B.GetNumRowsLocal() );
         }
         else
         {
            SCHURCHEB_CHKERR( A.GetNumColsLocal() != B.GetNumColsLocal() );
         }
      }
      else
      {
         if(transb == 'N')
         {
            SCHURCHEB_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal());
         }
         else
         {
            SCHURCHEB_CHKERR(A.GetNumRowsLocal() != B.GetNumColsLocal());
         }
      }


      m = transa == 'N' ? A.GetNumRowsLocal() : A.GetNumColsLocal();
      n = transb == 'N' ? B.GetNumColsLocal() : B.GetNumRowsLocal();
      k = transa == 'N' ? A.GetNumColsLocal() : A.GetNumRowsLocal();
      
      if(C.GetNumRowsLocal() != m || C.GetNumColsLocal() != n)
      {
         C.Setup(m, n, A.GetDataLocation(), true);
      }
      
      if( k==0 )
      {
         /* In this case, A and B empty, C is not empty. Scale C. */
         C.Scale(beta);
         return SCHURCHEB_SUCCESS;
      }
      
      if( m==0 || n==0 )
      {
         /* In this case, C is also empty. */
         return SCHURCHEB_SUCCESS;
      }
      
#ifdef SCHURCHEB_CUDA
      
      int loc_A = A.GetDataLocation();
      int loc_B = B.GetDataLocation();
      int loc_C = C.GetDataLocation();
      
      switch(loc_A)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matmat on device */
            SCHURCHEB_CHKERR( loc_B == kMemoryHost || loc_B == kMemoryPinned );
            SCHURCHEB_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );
            
            return DenseMatrixSMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
            
            break;
         }
         case kMemoryUnified:
         {
            /* typically matrices should not be on the unified memory */
            switch(loc_B)
            {
               case kMemoryDevice:
               {
                  /* B is on device, need to apply matmat on device */
                  SCHURCHEB_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );
                  
                  return DenseMatrixSMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  
                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_C == kMemoryDevice || loc_C == kMemoryUnified )
                  {
                     /* apply matmat on device */
                     return DenseMatrixSMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  }
                  /* otherwise on host */
                  break;
               }
               default:
               {
                  /* matmat on host */
                  SCHURCHEB_CHKERR( loc_C == kMemoryDevice );
                  break;
               }
            }
            break;
         }
         default:
         {
            /* matvec on host */
            SCHURCHEB_CHKERR( loc_B == kMemoryDevice || loc_C == kMemoryDevice );
            break;
         }
      }
      
#endif

#ifdef SCHURCHEB_BLAS
      
      int ldim_A, ldim_B, ldim_C;
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();
      
      SCHURCHEB_BLASLAPACK_SGEMM( &transa, &transb, &m, &n, &k, &alpha, A.GetData(), &ldim_A, B.GetData(), &ldim_B, &beta, C.GetData(), &ldim_C);
#else
      DenseMatrixMatMatTemplate(alpha, A, transa, B, transb, beta, C);
#endif
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixMatMat( const double &alpha, const DenseMatrixClass<double> &A, char transa, const DenseMatrixClass<double> &B, char transb, const double &beta, DenseMatrixClass<double> &C)
   {
      int m, n, k;
      
      if(transa == 'N')
      {
         if(transb == 'N')
         {
            SCHURCHEB_CHKERR( A.GetNumColsLocal() != B.GetNumRowsLocal() );
         }
         else
         {
            SCHURCHEB_CHKERR( A.GetNumColsLocal() != B.GetNumColsLocal() );
         }
      }
      else
      {
         if(transb == 'N')
         {
            SCHURCHEB_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal());
         }
         else
         {
            SCHURCHEB_CHKERR(A.GetNumRowsLocal() != B.GetNumColsLocal());
         }
      }


      m = transa == 'N' ? A.GetNumRowsLocal() : A.GetNumColsLocal();
      n = transb == 'N' ? B.GetNumColsLocal() : B.GetNumRowsLocal();
      k = transa == 'N' ? A.GetNumColsLocal() : A.GetNumRowsLocal();
      
      if(C.GetNumRowsLocal() != m || C.GetNumColsLocal() != n)
      {
         C.Setup(m, n, A.GetDataLocation(), true);
      }
      
      if( k==0 )
      {
         /* In this case, A and B empty, C is not empty. Scale C. */
         C.Scale(beta);
         return SCHURCHEB_SUCCESS;
      }
      
      if( m==0 || n==0 )
      {
         /* In this case, C is also empty. */
         return SCHURCHEB_SUCCESS;
      }
      
#ifdef SCHURCHEB_CUDA
      
      int loc_A = A.GetDataLocation();
      int loc_B = B.GetDataLocation();
      int loc_C = C.GetDataLocation();
      
      switch(loc_A)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matmat on device */
            SCHURCHEB_CHKERR( loc_B == kMemoryHost || loc_B == kMemoryPinned );
            SCHURCHEB_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );
            
            return DenseMatrixDMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
            
            break;
         }
         case kMemoryUnified:
         {
            /* typically matrices should not be on the unified memory */
            switch(loc_B)
            {
               case kMemoryDevice:
               {
                  /* B is on device, need to apply matmat on device */
                  SCHURCHEB_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );
                  
                  return DenseMatrixDMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  
                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_C == kMemoryDevice || loc_C == kMemoryUnified )
                  {
                     /* apply matmat on device */
                     return DenseMatrixDMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  }
                  /* otherwise on host */
                  break;
               }
               default:
               {
                  /* matmat on host */
                  SCHURCHEB_CHKERR( loc_C == kMemoryDevice );
                  break;
               }
            }
            break;
         }
         default:
         {
            /* matvec on host */
            SCHURCHEB_CHKERR( loc_B == kMemoryDevice || loc_C == kMemoryDevice );
            break;
         }
      }
      
#endif

#ifdef SCHURCHEB_BLAS
      
      int ldim_A, ldim_B, ldim_C;
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();
      
      SCHURCHEB_BLASLAPACK_DGEMM( &transa, &transb, &m, &n, &k, &alpha, A.GetData(), &ldim_A, B.GetData(), &ldim_B, &beta, C.GetData(), &ldim_C);
#else
      DenseMatrixMatMatTemplate(alpha, A, transa, B, transb, beta, C);
#endif
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixMatMat( const complexs &alpha, const DenseMatrixClass<complexs> &A, char transa, const DenseMatrixClass<complexs> &B, char transb, const complexs &beta, DenseMatrixClass<complexs> &C)
   {
      int m, n, k;
      
      if(transa == 'N')
      {
         if(transb == 'N')
         {
            SCHURCHEB_CHKERR( A.GetNumColsLocal() != B.GetNumRowsLocal() );
         }
         else
         {
            SCHURCHEB_CHKERR( A.GetNumColsLocal() != B.GetNumColsLocal() );
         }
      }
      else
      {
         if(transb == 'N')
         {
            SCHURCHEB_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal());
         }
         else
         {
            SCHURCHEB_CHKERR(A.GetNumRowsLocal() != B.GetNumColsLocal());
         }
      }
      
      m = transa == 'N' ? A.GetNumRowsLocal() : A.GetNumColsLocal();
      n = transb == 'N' ? B.GetNumColsLocal() : B.GetNumRowsLocal();
      k = transa == 'N' ? A.GetNumColsLocal() : A.GetNumRowsLocal();
      
      if(C.GetNumRowsLocal() != m || C.GetNumColsLocal() != n)
      {
         C.Setup(m, n, A.GetDataLocation(), true);
      }
      
      if( k==0 )
      {
         /* In this case, A and B empty, C is not empty. Scale C. */
         C.Scale(beta);
         return SCHURCHEB_SUCCESS;
      }
      
      if( m==0 || n==0 )
      {
         /* In this case, C is also empty. */
         return SCHURCHEB_SUCCESS;
      }
      
#ifdef SCHURCHEB_CUDA
      
      int loc_A = A.GetDataLocation();
      int loc_B = B.GetDataLocation();
      int loc_C = C.GetDataLocation();
      
      switch(loc_A)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matmat on device */
            SCHURCHEB_CHKERR( loc_B == kMemoryHost || loc_B == kMemoryPinned );
            SCHURCHEB_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );
            
            return DenseMatrixCMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
            
            break;
         }
         case kMemoryUnified:
         {
            /* typically matrices should not be on the unified memory */
            switch(loc_B)
            {
               case kMemoryDevice:
               {
                  /* B is on device, need to apply matmat on device */
                  SCHURCHEB_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );
                  
                  return DenseMatrixCMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  
                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_C == kMemoryDevice || loc_C == kMemoryUnified )
                  {
                     /* apply matmat on device */
                     return DenseMatrixCMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  }
                  /* otherwise on host */
                  break;
               }
               default:
               {
                  /* matmat on host */
                  SCHURCHEB_CHKERR( loc_C == kMemoryDevice );
                  break;
               }
            }
            break;
         }
         default:
         {
            /* matvec on host */
            SCHURCHEB_CHKERR( loc_B == kMemoryDevice || loc_C == kMemoryDevice );
            break;
         }
      }
      
#endif

#ifdef SCHURCHEB_BLAS
      
      int ldim_A, ldim_B, ldim_C;
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();
      
      SCHURCHEB_BLASLAPACK_CGEMM( &transa, &transb, &m, &n, &k, 
                                 SCHURCHEB_CAST( const ccomplexs*, &alpha), SCHURCHEB_CAST( const ccomplexs*, A.GetData()), &ldim_A, 
                                 SCHURCHEB_CAST( const ccomplexs*, B.GetData()), &ldim_B, 
                                 SCHURCHEB_CAST( const ccomplexs*, &beta), SCHURCHEB_CAST( ccomplexs*, C.GetData()), &ldim_C);
#else
      DenseMatrixMatMatTemplate(alpha, A, transa, B, transb, beta, C);
#endif
      
      return SCHURCHEB_SUCCESS;
   }
   
   int DenseMatrixMatMat( const complexd &alpha, const DenseMatrixClass<complexd> &A, char transa, const DenseMatrixClass<complexd> &B, char transb, const complexd &beta, DenseMatrixClass<complexd> &C)
   {
      int m, n, k;
      
      if(transa == 'N')
      {
         if(transb == 'N')
         {
            SCHURCHEB_CHKERR( A.GetNumColsLocal() != B.GetNumRowsLocal() );
         }
         else
         {
            SCHURCHEB_CHKERR( A.GetNumColsLocal() != B.GetNumColsLocal() );
         }
      }
      else
      {
         if(transb == 'N')
         {
            SCHURCHEB_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal());
         }
         else
         {
            SCHURCHEB_CHKERR(A.GetNumRowsLocal() != B.GetNumColsLocal());
         }
      }


      m = transa == 'N' ? A.GetNumRowsLocal() : A.GetNumColsLocal();
      n = transb == 'N' ? B.GetNumColsLocal() : B.GetNumRowsLocal();
      k = transa == 'N' ? A.GetNumColsLocal() : A.GetNumRowsLocal();
      
      if(C.GetNumRowsLocal() != m || C.GetNumColsLocal() != n)
      {
         C.Setup(m, n, A.GetDataLocation(), true);
      }
      
      if( k==0 )
      {
         /* In this case, A and B empty, C is not empty. Scale C. */
         C.Scale(beta);
         return SCHURCHEB_SUCCESS;
      }
      
      if( m==0 || n==0 )
      {
         /* In this case, C is also empty. */
         return SCHURCHEB_SUCCESS;
      }
      
#ifdef SCHURCHEB_CUDA
      
      int loc_A = A.GetDataLocation();
      int loc_B = B.GetDataLocation();
      int loc_C = C.GetDataLocation();
      
      switch(loc_A)
      {
         case kMemoryDevice:
         {
            /* A is on the device memory, need to do matmat on device */
            SCHURCHEB_CHKERR( loc_B == kMemoryHost || loc_B == kMemoryPinned );
            SCHURCHEB_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );
            
            return DenseMatrixZMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
            
            break;
         }
         case kMemoryUnified:
         {
            /* typically matrices should not be on the unified memory */
            switch(loc_B)
            {
               case kMemoryDevice:
               {
                  /* B is on device, need to apply matmat on device */
                  SCHURCHEB_CHKERR( loc_C == kMemoryHost || loc_C == kMemoryPinned );
                  
                  return DenseMatrixZMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  
                  break;
               }
               case kMemoryUnified:
               {
                  if( loc_C == kMemoryDevice || loc_C == kMemoryUnified )
                  {
                     /* apply matmat on device */
                     return DenseMatrixZMatMatDevice( m, n, k, alpha, A, transa, B, transb, beta, C);
                  }
                  /* otherwise on host */
                  break;
               }
               default:
               {
                  /* matmat on host */
                  SCHURCHEB_CHKERR( loc_C == kMemoryDevice );
                  break;
               }
            }
            break;
         }
         default:
         {
            /* matvec on host */
            SCHURCHEB_CHKERR( loc_B == kMemoryDevice || loc_C == kMemoryDevice );
            break;
         }
      }
      
#endif

#ifdef SCHURCHEB_BLAS
      
      int ldim_A, ldim_B, ldim_C;
      ldim_A = A.GetLeadingDimension();
      ldim_B = B.GetLeadingDimension();
      ldim_C = C.GetLeadingDimension();
      
      SCHURCHEB_BLASLAPACK_ZGEMM( &transa, &transb, &m, &n, &k, 
                                 SCHURCHEB_CAST( const ccomplexd*, &alpha), SCHURCHEB_CAST( const ccomplexd*, A.GetData()), &ldim_A, 
                                 SCHURCHEB_CAST( const ccomplexd*, B.GetData()), &ldim_B, 
                                 SCHURCHEB_CAST( const ccomplexd*, &beta), SCHURCHEB_CAST( ccomplexd*, C.GetData()), &ldim_C);
#else
      DenseMatrixMatMatTemplate(alpha, A, transa, B, transb, beta, C);
#endif
      
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T>
   int DenseMatrixTransposeHostTemplate( DenseMatrixClass<T> &A, DenseMatrixClass<T> &AT)
   {
      int i, j, nrowsA, ncolsA;
      
      nrowsA = A.GetNumRowsLocal();
      ncolsA = A.GetNumColsLocal();
      
      if(AT.GetNumRowsLocal() != ncolsA || AT.GetNumColsLocal() != nrowsA)
      {
         /* in this case we need to create AT */
         AT.Setup(ncolsA, nrowsA, kMemoryHost);
      }
      else
      {
         /* only design for host memory! */
         SCHURCHEB_CHKERR(AT.GetDataLocation() == kMemoryDevice);
      }
      
      for(i = 0 ; i < ncolsA ; i ++)
      {
         for(j = 0 ; j < nrowsA ; j ++)
         {
            /* copy */
            AT(i, j) = A(j, i);
         }
      }
      
      return SCHURCHEB_SUCCESS;
      
   }
   template int DenseMatrixTransposeHostTemplate( DenseMatrixClass<float> &A, DenseMatrixClass<float> &AT);
   template int DenseMatrixTransposeHostTemplate( DenseMatrixClass<double> &A, DenseMatrixClass<double> &AT);
   template int DenseMatrixTransposeHostTemplate( DenseMatrixClass<complexs> &A, DenseMatrixClass<complexs> &AT);
   template int DenseMatrixTransposeHostTemplate( DenseMatrixClass<complexd> &A, DenseMatrixClass<complexd> &AT);
   
   template<typename T>
   int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const T *aa, int nrows, int ncols, char trans, const T &alpha, const T *x, const T &beta, T *y)
   {
      
      int      i, j, j1, j2;
      T        r, xi, *x_temp = NULL;
      T        one = 1.0;
      T        zero = 0.0;
#ifdef SCHURCHEB_OPENMP
      int      num_threads, my_thread_id;
      T        *y_temp;
#endif
      
      /* 1. Compute y = beta*y
       * note that when x==y and alpha != 0.0, we need to copy x
       */
      
      /* copy x when x==y, otherwise scale y would modify x 
       * TODO: memcpy or omp parallel?
       */
      if( (x == y) && (alpha != zero) )
      {
         if (trans == 'N') 
         {
            SCHURCHEB_MALLOC(x_temp, nrows, kMemoryHost, T);
            SCHURCHEB_MEMCPY(x_temp, y, nrows, kMemoryHost, kMemoryHost, T);
            x = x_temp;
         }
         else if( (trans == 'T') || (trans == 'C') )
         {
            SCHURCHEB_MALLOC(x_temp, ncols, kMemoryHost, T);
            SCHURCHEB_MEMCPY(x_temp, y, ncols, kMemoryHost, kMemoryHost, T);
            x = x_temp;
         }
         else
         {
            return SCHURCHEB_ERROR_INVALED_PARAM;
         }
      }
      
      /* now scale y */
      if(beta != one)
      {
         /* when beta == 1.0, y = y, do nothing */
         if(beta != zero)
         {
            /* y = beta*y */
            if (trans == 'N') 
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < nrows; i++) 
               {
                  y[i] *= beta;
               }
            }
            else if( (trans == 'T') || (trans == 'C') )
            {
               /* if x == y need to create new x */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < ncols; i++) 
               {
                  y[i] *= beta;
               }
            }
            else
            {
               return SCHURCHEB_ERROR_INVALED_PARAM;
            }
         }
         else
         {
            /* beta == 0.0, y = 0 */
            if (trans == 'N') 
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < nrows; i++) 
               {
                  y[i] = zero;
               }
            }
            else if( (trans == 'T') || (trans == 'C') )
            {
               /* if x == y need to create new x */
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < ncols; i++) 
               {
                  y[i] = zero;
               }
            }
            else
            {
               return SCHURCHEB_ERROR_INVALED_PARAM;
            }
         }
      }
      
      
      /* 2. the matvec y = alpha*A*x + y
       * when alpha == 0 we have y = y, do nothing
       */
      
      if(alpha != zero)
      {
         if(alpha != one)
         {
            if (trans == 'N') 
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, r, j1, j2) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
#endif
               for (i = 0; i < nrows; i++) 
               {
                  r = 0.0;
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++) 
                  {
                     r += aa[j] * x[ja[j]];
                  }
                  y[i] += alpha*r;
               }
            }
            else if(trans == 'T')
            {
#ifdef SCHURCHEB_OPENMP
               /* create buffer for OpenMP when needed */
               num_threads = SchurchebGetOpenmpMaxNumThreads();
               if(num_threads>1)
               {
                  SCHURCHEB_CALLOC(y_temp, num_threads * ncols, kMemoryHost, T);
#pragma omp parallel private(i, j, j1, j2, my_thread_id, xi)
                  {
                     my_thread_id = SchurchebGetOpenmpThreadNum();
                     T* y_local = y_temp + my_thread_id * ncols;
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
                     /* sum to local buffer */
                     for (i = 0; i < nrows; i++) 
                     {
                        xi = alpha * x[i];
                        j1 = ia[i];
                        j2 = ia[i+1];
                        for (j = j1; j < j2; j++) 
                        {
                           y_local[ja[j]] += aa[j] * xi;
                        }
                     }
                     /* sumup the local buffer to y */
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
                     for(i = 0 ; i < ncols ; i ++)
                     {
                        for(j = 0 ; j < num_threads ; j ++)
                        {
                           y[i] = y[i] + y_temp[i+j*ncols];
                        }
                     }
                  }
                  /* free temp buffer after openmp finished */
                  SCHURCHEB_FREE(y_temp, kMemoryHost);
               }
               else
               {
#endif
                  for (i = 0; i < nrows; i++) 
                  {
                     xi = alpha * x[i];
                     j1 = ia[i];
                     j2 = ia[i+1];
                     for (j = j1; j < j2; j++) 
                     {
                        y[ja[j]] += aa[j] * xi;
                     }
                  }
#ifdef SCHURCHEB_OPENMP
               }
#endif
            }
            else if(trans == 'C')
            {
#ifdef SCHURCHEB_OPENMP
               /* create buffer for OpenMP when needed */
               num_threads = SchurchebGetOpenmpMaxNumThreads();
               if(num_threads>1)
               {
                  SCHURCHEB_CALLOC(y_temp, num_threads * ncols, kMemoryHost, T);
#pragma omp parallel private(i, j, j1, j2, my_thread_id, xi)
                  {
                     my_thread_id = SchurchebGetOpenmpThreadNum();
                     T* y_local = y_temp + my_thread_id * ncols;
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
                     /* sum to local buffer */
                     for (i = 0; i < nrows; i++) 
                     {
                        xi = alpha * x[i];
                        j1 = ia[i];
                        j2 = ia[i+1];
                        for (j = j1; j < j2; j++) 
                        {
                           y_local[ja[j]] += SchurchebConj(aa[j]) * xi;
                        }
                     }
                     /* sumup the local buffer to y */
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
                     for(i = 0 ; i < ncols ; i ++)
                     {
                        for(j = 0 ; j < num_threads ; j ++)
                        {
                           y[i] = y[i] + y_temp[i+j*ncols];
                        }
                     }
                  }
                  /* free temp buffer after openmp finished */
                  SCHURCHEB_FREE(y_temp, kMemoryHost);
               }
               else
               {
#endif
                  for (i = 0; i < nrows; i++) 
                  {
                     xi = alpha * x[i];
                     j1 = ia[i];
                     j2 = ia[i+1];
                     for (j = j1; j < j2; j++) 
                     {
                        y[ja[j]] += SchurchebConj(aa[j]) * xi;
                     }
                  }
#ifdef SCHURCHEB_OPENMP
               }
#endif
            }
            else
            {
               return SCHURCHEB_ERROR_INVALED_PARAM;
            }
         }
         else
         {
            if (trans == 'N') 
            {
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i, j, j1, j2) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
               for (i = 0; i < nrows; i++) 
               {
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++) 
                  {
                     y[i] += aa[j] * x[ja[j]];
                  }
               }
            }
            else if(trans == 'T')
            {
#ifdef SCHURCHEB_OPENMP
               /* create buffer for OpenMP when needed */
               num_threads = SchurchebGetOpenmpMaxNumThreads();
               if(num_threads>1)
               {
                  SCHURCHEB_CALLOC(y_temp, num_threads * ncols, kMemoryHost, T);
#pragma omp parallel private(i, j, j1, j2, my_thread_id)
                  {
                     my_thread_id = SchurchebGetOpenmpThreadNum();
                     T* y_local = y_temp + my_thread_id * ncols;
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
                     /* sum to local buffer */
                     for (i = 0; i < nrows; i++) 
                     {
                        j1 = ia[i];
                        j2 = ia[i+1];
                        for (j = j1; j < j2; j++) 
                        {
                           y_local[ja[j]] += aa[j] * x[i];
                        }
                     }
                     /* sumup the local buffer to y */
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
                     for(i = 0 ; i < ncols ; i ++)
                     {
                        for(j = 0 ; j < num_threads ; j ++)
                        {
                           y[i] = y[i] + y_temp[i+j*ncols];
                        }
                     }
                  }
                  /* free temp buffer after openmp finished */
                  SCHURCHEB_FREE(y_temp, kMemoryHost);
               }
               else
               {
#endif
                  for (i = 0; i < nrows; i++) 
                  {
                     j1 = ia[i];
                     j2 = ia[i+1];
                     for (j = j1; j < j2; j++) 
                     {
                        y[ja[j]] += aa[j] * x[i];
                     }
                  }
#ifdef SCHURCHEB_OPENMP
               }
#endif
            }
            else if(trans == 'C')
            {
#ifdef SCHURCHEB_OPENMP
               /* create buffer for OpenMP when needed */
               num_threads = SchurchebGetOpenmpMaxNumThreads();
               if(num_threads>1)
               {
                  SCHURCHEB_CALLOC(y_temp, num_threads * ncols, kMemoryHost, T);
#pragma omp parallel private(i, j, j1, j2, my_thread_id)
                  {
                     my_thread_id = SchurchebGetOpenmpThreadNum();
                     T* y_local = y_temp + my_thread_id * ncols;
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
                     /* sum to local buffer */
                     for (i = 0; i < nrows; i++) 
                     {
                        j1 = ia[i];
                        j2 = ia[i+1];
                        for (j = j1; j < j2; j++) 
                        {
                           y_local[ja[j]] += SchurchebConj(aa[j]) * x[i];
                        }
                     }
                     /* sumup the local buffer to y */
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
                     for(i = 0 ; i < ncols ; i ++)
                     {
                        for(j = 0 ; j < num_threads ; j ++)
                        {
                           y[i] = y[i] + y_temp[i+j*ncols];
                        }
                     }
                  }
                  /* free temp buffer after openmp finished */
                  SCHURCHEB_FREE(y_temp, kMemoryHost);
               }
               else
               {
#endif
                  for (i = 0; i < nrows; i++) 
                  {
                     j1 = ia[i];
                     j2 = ia[i+1];
                     for (j = j1; j < j2; j++) 
                     {
                        y[ja[j]] += SchurchebConj(aa[j]) * x[i];
                     }
                  }
#ifdef SCHURCHEB_OPENMP
               }
#endif
            }
            else
            {
               return SCHURCHEB_ERROR_INVALED_PARAM;
            }
         }
      }
      
      if(x_temp)
      {
         SCHURCHEB_FREE( x_temp, kMemoryHost);
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const float *aa, int nrows, int ncols, char trans, const float &alpha, const float *x, const float &beta, float *y);
   template int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const double *aa, int nrows, int ncols, char trans, const double &alpha, const double *x, const double &beta, double *y);
   template int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const complexs *aa, int nrows, int ncols, char trans, const complexs &alpha, const complexs *x, const complexs &beta, complexs *y);
   template int CsrMatrixPMatVecHostTemplate( const int *ia, const int *ja, const complexd *aa, int nrows, int ncols, char trans, const complexd &alpha, const complexd *x, const complexd &beta, complexd *y);
   
   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const float *aa, int nrows, int ncols, char trans, const float &alpha, const float *x, const float &beta, float *y)
   {
#ifdef SCHURCHEB_MKL
      /* mkl matdescra, general 0-based matrix 
       * for MKL, I is two vectors of size nrow, I_start and I_end, thus, can be discontinues on J
       */
      mkl_scsrmv( &trans, &nrows, &ncols, &alpha, "GXXCXX", aa, ja, ia, ia+1, x, &beta , y);
      return SCHURCHEB_SUCCESS;
#endif
      return CsrMatrixPMatVecHostTemplate(ia, ja, aa, nrows, ncols, trans, alpha, x, beta, y);
   }
   
   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const double *aa, int nrows, int ncols, char trans, const double &alpha, const double *x, const double &beta, double *y)
   {
#ifdef SCHURCHEB_MKL
      /* mkl matdescra, general 0-based matrix 
       * for MKL, I is two vectors of size nrow, I_start and I_end, thus, can be discontinues on J
       */
      mkl_dcsrmv( &trans, &nrows, &ncols, &alpha, "GXXCXX", aa, ja, ia, ia+1, x, &beta, y);
      return SCHURCHEB_SUCCESS;
#endif
      return CsrMatrixPMatVecHostTemplate(ia, ja, aa, nrows, ncols, trans, alpha, x, beta, y);
   }
   
   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const complexs *aa, int nrows, int ncols, char trans, const complexs &alpha, const complexs *x, const complexs &beta, complexs *y)
   {
#ifdef SCHURCHEB_MKL
      /* mkl matdescra, general 0-based matrix 
       * for MKL, I is two vectors of size nrow, I_start and I_end, thus, can be discontinues on J
       */
      mkl_ccsrmv( &trans, &nrows, &ncols, (MKL_Complex8*)&alpha, "GXXCXX", (MKL_Complex8*)aa, ja, ia, ia+1, (MKL_Complex8*)x, (MKL_Complex8*)&beta , (MKL_Complex8*)y);
      return SCHURCHEB_SUCCESS;
#endif
      return CsrMatrixPMatVecHostTemplate(ia, ja, aa, nrows, ncols, trans, alpha, x, beta, y);
   }
   
   int CsrMatrixPMatVecHost( const int *ia, const int *ja, const complexd *aa, int nrows, int ncols, char trans, const complexd &alpha, const complexd *x, const complexd &beta, complexd *y)
   {
#ifdef SCHURCHEB_MKL
      /* mkl matdescra, general 0-based matrix 
       * for MKL, I is two vectors of size nrow, I_start and I_end, thus, can be discontinues on J
       */
      mkl_zcsrmv( &trans, &nrows, &ncols, (MKL_Complex16*)&alpha, "GXXCXX", (MKL_Complex16*)aa, ja, ia, ia+1, (MKL_Complex16*)x, (MKL_Complex16*)&beta , (MKL_Complex16*)y);
      return SCHURCHEB_SUCCESS;
#endif
      return CsrMatrixPMatVecHostTemplate(ia, ja, aa, nrows, ncols, trans, alpha, x, beta, y);
   }
   
   int CsrMatrixMatVec( const CsrMatrixClass<float> &A, char trans, const float &alpha, const VectorClass<float> &x, const float &beta, VectorClass<float> &y)
   {
      int   m, n;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();

#ifdef SCHURCHEB_DEBUG
      if (trans == 'N') 
      {
         SCHURCHEB_CHKERR( m != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( n != x.GetLengthLocal() );
      } 
      else if(trans == 'C' || trans == 'T')
      {
         SCHURCHEB_CHKERR( n != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( m != x.GetLengthLocal() );
         trans = 'T';
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef SCHURCHEB_CUDA 
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               SCHURCHEB_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
               
               return CsrMatrixSMatVecDevice( A, trans, alpha, x, beta, y);
               
               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
                     
                     return CsrMatrixSMatVecDevice( A, trans, alpha, x, beta, y);
                     
                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return CsrMatrixSMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     SCHURCHEB_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               SCHURCHEB_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
         CsrMatrixPMatVecHost( A.GetI(), A.GetJ(), A.GetData(), A.GetNumRowsLocal(), A.GetNumColsLocal(), trans, alpha, x.GetData(), beta, y.GetData());
      }
      else if(( trans == 'T' && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return SCHURCHEB_SUCCESS;
   }
   
   int CsrMatrixMatVec( const CsrMatrixClass<double> &A, char trans, const double &alpha, const VectorClass<double> &x, const double &beta, VectorClass<double> &y)
   {
      int   m, n;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();

#ifdef SCHURCHEB_DEBUG
      if (trans == 'N') 
      {
         SCHURCHEB_CHKERR( m != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( n != x.GetLengthLocal() );
      } 
      else if(trans == 'C' || trans == 'T')
      {
         SCHURCHEB_CHKERR( n != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( m != x.GetLengthLocal() );
         trans = 'T';
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef SCHURCHEB_CUDA 
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               SCHURCHEB_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
               
               return CsrMatrixDMatVecDevice( A, trans, alpha, x, beta, y);
               
               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
                     
                     return CsrMatrixDMatVecDevice( A, trans, alpha, x, beta, y);
                     
                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return CsrMatrixDMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     SCHURCHEB_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               SCHURCHEB_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
         CsrMatrixPMatVecHost( A.GetI(), A.GetJ(), A.GetData(), A.GetNumRowsLocal(), A.GetNumColsLocal(), trans, alpha, x.GetData(), beta, y.GetData());

      }
      else if(( trans == 'T' && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return SCHURCHEB_SUCCESS;
   }
   
   int CsrMatrixMatVec( const CsrMatrixClass<complexs> &A, char trans, const complexs &alpha, const VectorClass<complexs> &x, const complexs &beta, VectorClass<complexs> &y)
   {
      int   m, n;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();

#ifdef SCHURCHEB_DEBUG
      if (trans == 'N') 
      {
         SCHURCHEB_CHKERR( m != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( n != x.GetLengthLocal() );
      } 
      else if(trans == 'C' || trans == 'T')
      {
         SCHURCHEB_CHKERR( n != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( m != x.GetLengthLocal() );
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef SCHURCHEB_CUDA 
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               SCHURCHEB_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
               
               return CsrMatrixCMatVecDevice( A, trans, alpha, x, beta, y);
               
               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
                     
                     return CsrMatrixCMatVecDevice( A, trans, alpha, x, beta, y);
                     
                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return CsrMatrixCMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     SCHURCHEB_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               SCHURCHEB_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
         CsrMatrixPMatVecHost( A.GetI(), A.GetJ(), A.GetData(), A.GetNumRowsLocal(), A.GetNumColsLocal(), trans, alpha, x.GetData(), beta, y.GetData());

      }
      else if(((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return SCHURCHEB_SUCCESS;
   }
   
   int CsrMatrixMatVec( const CsrMatrixClass<complexd> &A, char trans, const complexd &alpha, const VectorClass<complexd> &x, const complexd &beta, VectorClass<complexd> &y)
   {
      int   m, n;
      m     = A.GetNumRowsLocal();
      n     = A.GetNumColsLocal();

#ifdef SCHURCHEB_DEBUG
      if (trans == 'N') 
      {
         SCHURCHEB_CHKERR( m != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( n != x.GetLengthLocal() );
      } 
      else if(trans == 'C' || trans == 'T')
      {
         SCHURCHEB_CHKERR( n != y.GetLengthLocal() );
         SCHURCHEB_CHKERR( m != x.GetLengthLocal() );
      }
#endif

      //cpu version dense matvec
      if( m > 0 && n > 0)
      {
#ifdef SCHURCHEB_CUDA 
         int loc_a = A.GetDataLocation();
         int loc_x = x.GetDataLocation();
         int loc_y = y.GetDataLocation();
         switch(loc_a)
         {
            case kMemoryDevice:
            {
               /* A is on the device memory, need to do matvec on device */
               SCHURCHEB_CHKERR( loc_x == kMemoryHost || loc_x == kMemoryPinned );
               SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
               
               return CsrMatrixZMatVecDevice( A, trans, alpha, x, beta, y);
               
               break;
            }
            case kMemoryUnified:
            {
               /* typically matrices should not be on the unified memory */
               switch( loc_x)
               {
                  case kMemoryDevice:
                  {
                     /* x is on device, need to apply matvec on device */
                     SCHURCHEB_CHKERR( loc_y == kMemoryHost || loc_y == kMemoryPinned );
                     
                     return CsrMatrixZMatVecDevice( A, trans, alpha, x, beta, y);
                     
                     break;
                  }
                  case kMemoryUnified:
                  {
                     if( loc_y == kMemoryDevice || loc_y == kMemoryUnified )
                     {
                        /* apply matvec on device */
                        return CsrMatrixZMatVecDevice( A, trans, alpha, x, beta, y);
                     }
                     /* otherwise on host */
                     break;
                  }
                  default:
                  {
                     /* matvec on host */
                     SCHURCHEB_CHKERR( loc_y == kMemoryDevice );
                     break;
                  }
               }
               break;
            }
            default:
            {
               /* matvec on host */
               SCHURCHEB_CHKERR( loc_x == kMemoryDevice || loc_y == kMemoryDevice );
               break;
            }
         }
#endif
         CsrMatrixPMatVecHost( A.GetI(), A.GetJ(), A.GetData(), A.GetNumRowsLocal(), A.GetNumColsLocal(), trans, alpha, x.GetData(), beta, y.GetData());

      }
      else if(((trans == 'T' || trans == 'C') && m == 0) || (trans == 'N' && n == 0))
      {
         y.Scale(beta);
      }
      return SCHURCHEB_SUCCESS;
   }
   
   template <int INIDX, int OUTIDX, typename T>
   int CsrMatrixP2CscMatrixPHost( int nrows, int ncols, bool copy_data, T* ai, int *ji, int *ii, T *ao, int *jo, int *io)
   {
      int i, j, k;
#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
      for ( i = 0; i < ncols + 1; i++) 
      {
         io[i] = 0;
      }
      
#ifdef SCHURCHEB_OPENMP
      int num_threads = SchurchebGetOpenmpMaxNumThreads();
      if(num_threads > 1)
      {
         /* create temp buffer */
         vector_int io_temp_vec, size_temp_vec;
         int *io_temp, *size_temp;
         int ncolsp1 = ncols+1;
         io_temp_vec.Setup(ncolsp1*num_threads,true);
         size_temp_vec.Setup(ncolsp1*num_threads,true);
         io_temp = io_temp_vec.GetData();
         size_temp = size_temp_vec.GetData();
#pragma omp parallel private(i, j, k)
         {
            int idx1, idx2;
            int my_thread_id = SchurchebGetOpenmpThreadNum();
            int *io_local = io_temp + ncolsp1 * my_thread_id;
            int *size_local = size_temp + ncolsp1 * my_thread_id;
            
            /* get nnz of each columns of A, store in io 
             * in this step, io is parallel, we have 
             *     thread1        thread2      ...     threadn
             * | size_local_1 | size_local_2 | ... | size_local_n  |
             * size_local
             */
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
            for ( i = 0; i < nrows; i++) 
            {
               for ( k = ii[i]; k < ii[i+1]; k++)
               {
                  size_local[ji[k]-INIDX] ++;
               }
            }
            
            /* copy to size_temp to io_temp and get
             * the accumulate size
             * memcpy is not thread safe
             */
            for(i = 0 ; i < ncols ; i ++)
            {
               io_local[i+1] = size_local[i] + io_local[i];
            }
            
            /* accumulate to result to the io vector 
             * omp barrier is required, since we 
             * need to touch io_temp from other threads
             * note that we don't need io[ncol]
             */
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
            for ( i = 1; i <= ncols; i++) 
            {
               for(j = 0 ; j < num_threads ; j ++)
               {
                  io[i] += io_temp[i+j*ncolsp1];
               }
            }
            
            /* now start to copy data in 
             * also will touch data from other refion, add #pragma omp barrier
             * we want to have io_local now store the shift value
             */
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
            for(i = 0 ; i < ncols ; i ++)
            {
               io_temp[i] = io[i];
               for(j = 1 ; j < num_threads ; j ++)
               {
                  idx2 = i+j*ncolsp1;
                  idx1 = idx2 - ncolsp1;
                  io_temp[idx2] = io_temp[idx1] + size_temp[idx1];
               }
            }
            
            /* now copy data
             */
            if(copy_data)
            {
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
               for ( i = 0; i < nrows; i++) 
               {
                  for ( k = ii[i]; k < ii[i+1]; k++) 
                  {
                     j = ji[k]-INIDX;
                     ao[io_local[j]] = ai[k];
                     jo[io_local[j]++] = i + OUTIDX;
                  }
               }
            }
            else
            {
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
               for ( i = 0; i < nrows; i++) 
               {
                  for ( k = ii[i]; k < ii[i+1]; k++) 
                  {
                     j = ji[k]-INIDX;
                     jo[io_local[j]++] = i + OUTIDX;
                  }
               }
            }
         }/* openmp ends */
         
         io_temp_vec.Clear();
         size_temp_vec.Clear();
         
         /*---- reshift iao and leave 
          * no need of barrier here, io is not used
          */
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
         for(i = 1 ; i < ncols ; i ++)
         {
            io[i] += OUTIDX;
         }
         io[0] = OUTIDX;
         
         /* finally we need to sort local rows */
         /* csr matrix */
         if(copy_data)
         {
#pragma omp parallel private(i)
            {
               int length;
               vector_int cols, order;
               SequentialVectorClass<T> vals;
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
               for(i = 0 ; i < ncols ; i ++)
               {
                  length = io[i+1]-io[i];
                  cols.SetupPtr( jo + io[i], length, kMemoryHost);
                  
                  /* sort in ascending order */
                  cols.Sort( order, true, false);
                  
                  vals.SetupPtr( ao + io[i], length, kMemoryHost);
                  
                  /* apply the permutation */
                  cols.Perm(order);
                  vals.Perm(order);
                  
                  vals.Clear();
                  order.Clear();
                  
                  cols.Clear();
               }
            }/* openmp ends */
         }
         else
         {
#pragma omp parallel private(i)
            {
               int length;
               vector_int cols;
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
               for(i = 0 ; i < ncols ; i ++)
               {
                  length = io[i+1]-io[i];
                  
                  cols.SetupPtr( jo + io[i], length, kMemoryHost);
                  
                  /* in this case just sort cols in ascending order */
                  cols.Sort(true);
                  
                  cols.Clear();
               }
            }/* openmp ends */
         }
      }
      else
      {
#endif
         /* get nnz of each columns of A, store in io 
          * note that the input might be 1-based, shift by INIDX
          * Note that for 1-based indexing, io should still start from
          * io[0], just the value should be 1
          */
         for ( i = 0; i < nrows; i++) 
         {
            for ( k = ii[i]; k < ii[i+1]; k++)
            {
               /* ji[k] has shift INIDX */
               io[ji[k]+1-INIDX] ++;
            }
         }
         /* accumulate to get the I vector */
         for ( i = 0; i < ncols; i++) 
         {
            io[i+1] += io[i];
         }
         /* copy J and A 
          * j is the column in in-based, we store it in out-based
          */
         for ( i = 0; i < nrows; i++) 
         {
            for ( k = ii[i]; k < ii[i+1]; k++) 
            {
               /* get location in io */
               j = ji[k]-INIDX;
               if (copy_data)
               {
                  ao[io[j]] = ai[k];
               }
               jo[io[j]++] = i + OUTIDX;
            }
         }
         /* shift io to OUTIDX-based */
         for (i = ncols; i > 0; i--) 
         {
            io[i] = io[i-1] + OUTIDX;
         }
         io[0] = OUTIDX;
#ifdef SCHURCHEB_OPENMP
      }
#endif
      
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixP2CscMatrixPHost<0,0>( int nrows, int ncols, bool copy_data, float* ai, int *ji, int *ii, float *ao, int *jo, int *io);
   template int CsrMatrixP2CscMatrixPHost<0,0>( int nrows, int ncols, bool copy_data, double* ai, int *ji, int *ii, double *ao, int *jo, int *io);
   template int CsrMatrixP2CscMatrixPHost<0,0>( int nrows, int ncols, bool copy_data, complexs* ai, int *ji, int *ii, complexs *ao, int *jo, int *io);
   template int CsrMatrixP2CscMatrixPHost<0,0>( int nrows, int ncols, bool copy_data, complexd* ai, int *ji, int *ii, complexd *ao, int *jo, int *io);
   
   template <int INIDX, int OUTIDX, typename T>
   int CooMatrixP2CsrMatrixPHost( int nrows, int ncols, int nnz, T* ai, int *ji, int *ii, T *ao, int *jo, int *io)
   {
      int   i, j;
      T     val;
      int   idx_shift = OUTIDX - INIDX;

#ifdef SCHURCHEB_OPENMP
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
#endif
      for ( i = 0; i <= nrows; i++) 
      {
         io[i] = 0;
      }

#ifdef SCHURCHEB_OPENMP
      int num_threads = SchurchebGetOpenmpMaxNumThreads();
      if(num_threads > 1)
      {
         vector_int io_temp_vec, size_temp_vec;
         int *io_temp, *size_temp;
         int nrowsp1 = nrows+1;
         io_temp_vec.Setup(nrowsp1*num_threads,true);
         size_temp_vec.Setup(nrowsp1*num_threads,true);
         io_temp = io_temp_vec.GetData();
         size_temp = size_temp_vec.GetData();
#pragma omp parallel private(i, j)
         {
            int my_thread_id = SchurchebGetOpenmpThreadNum();
            int *io_local = io_temp + nrowsp1 * my_thread_id;
            int *size_local = size_temp + nrowsp1 * my_thread_id;
            int idx1, idx2;
            
            /* this step is to prepare the io size */
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
            for ( i = 0; i < nnz; i++) 
            {
               size_local[ii[i] - INIDX]++;
            }
            
            /* copy to size_temp to io_temp and get
             * the accumulate size
             * memcpy is not thread safe
             */
            for(i = 0 ; i < nrows ; i ++)
            {
               io_local[i+1] = size_local[i] + io_local[i];
            }
            
            /* accumulate to result to the io vector 
             * omp barrier is required, since we 
             * need to touch io_temp from other threads
             * note that we don't need io[ncol]
             */
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
            for ( i = 1; i <= nrows; i++) 
            {
               for(j = 0 ; j < num_threads ; j ++)
               {
                  io[i] += io_temp[i+j*nrowsp1];
               }
            }
            
            /* now start to copy data in 
             * also will touch data from other refion, add #pragma omp barrier
             * we want to have io_local now store the shift value
             */
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
            for(i = 0 ; i < nrows ; i ++)
            {
               io_temp[i] = io[i];
               for(j = 1 ; j < num_threads ; j ++)
               {
                  idx2 = i+j*nrowsp1;
                  idx1 = idx2 - nrowsp1;
                  io_temp[idx2] = io_temp[idx1] + size_temp[idx1];
               }
            }
            
            /* now copy data */
#pragma omp barrier
#pragma omp for SCHURCHEB_OPENMP_SCHEDULE_STATIC
            for ( i = 0; i < nnz; i++) 
            {
               j = ii[i] - INIDX;
               val = ai[i];
               /* io[j] is now 0-based */
               ao[io_local[j]] = val;
               jo[io_local[j]++] = ji[i] + idx_shift;
            }
            
         }/* openmp ends */
         
         io_temp_vec.Clear();
         size_temp_vec.Clear();
         
#pragma omp parallel for private(i) SCHURCHEB_OPENMP_SCHEDULE_STATIC
         for ( i = nrows; i > 0; i--) 
         {
            io[i] += OUTIDX;
         }
         io[0] = OUTIDX;
      }
      else
      {
#endif
         /* this step is to prepare the io size */
         for ( i = 0; i < nnz; i++) 
         {
            io[ii[i]-INIDX+1]++;
         }
         for ( i = 0; i < nrows; i++) 
         {
            io[i+1] += io[i];
         }
         /* note that io is now 0-based */
         for ( i = 0; i < nnz; i++) 
         {
            j = ii[i] - INIDX;
            val = ai[i];
            /* io[j] is now 0-based */
            ao[io[j]] = val;
            jo[io[j]++] = ji[i] + idx_shift;
         }
         for ( i = nrows; i > 0; i--) 
         {
            io[i] = io[i-1] + OUTIDX;
         }
         io[0] = OUTIDX;
#ifdef SCHURCHEB_OPENMP
      }
#endif
      return SCHURCHEB_SUCCESS;
   }
   template int CooMatrixP2CsrMatrixPHost<0,0>( int nrows, int ncols, int nnz, float* ai, int *ji, int *ii, float *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<0,0>( int nrows, int ncols, int nnz, double* ai, int *ji, int *ii, double *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<0,0>( int nrows, int ncols, int nnz, complexs* ai, int *ji, int *ii, complexs *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<0,0>( int nrows, int ncols, int nnz, complexd* ai, int *ji, int *ii, complexd *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<1,0>( int nrows, int ncols, int nnz, float* ai, int *ji, int *ii, float *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<1,0>( int nrows, int ncols, int nnz, double* ai, int *ji, int *ii, double *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<1,0>( int nrows, int ncols, int nnz, complexs* ai, int *ji, int *ii, complexs *ao, int *jo, int *io);
   template int CooMatrixP2CsrMatrixPHost<1,0>( int nrows, int ncols, int nnz, complexd* ai, int *ji, int *ii, complexd *ao, int *jo, int *io);


   template <typename T>
   int CsrMatrixMatMat( const T &alpha, CsrMatrixClass<T> &A, DenseMatrixClass<T> &B, const T &beta, DenseMatrixClass<T> &C)
   {

   /*
   #ifdef SCHURCHEB_MKL
      
      char transa = 'N';
      int i;
      int m = A.GetNumRowsLocal();
      int n = B.GetNumColsLocal();
      int k = A.GetNumColsLocal();
      int ldb = B.GetLeadingDimension();
      int ldc = C.GetLeadingDimension();
      
      for(i = 0 ; i < n ; i ++)
      {
         mkl_dcsrmv( &transa, &m, &k, &alpha, "GXXCXX", A.GetData(), A.GetJ(), A.GetI(), A.GetI()+1, B.GetData()+ldb*i, &beta, C.GetData()+ldc*i);
      }
      
   #else
   */

      int      i, j, k, j1, j2;
      T        r;
      T        one = 1.0;
      T        zero = 0.0;
      int      *ia = A.GetI();
      int      *ja = A.GetJ();
      T        *aa = A.GetData();
      
      /* now scale C */
      if(beta != one)
      {
         /* when beta == 1.0, y = y, do nothing */
         if(beta != zero)
         {
            C.Scale(beta);
         }
         else
         {
            C.Fill(zero);
         }
      }
      
      /* 2. the matmat C = alpha*A*B + C
       * when alpha == 0 we have C = C, do nothing
       */
      
      if(alpha != zero)
      {
         if(alpha != one)
         {
            for (k = 0 ; k < B.GetNumColsLocal() ; k ++)
            {
               for (i = 0; i < A.GetNumRowsLocal(); i++) 
               {
                  r = 0.0;
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++) 
                  {
                     r += aa[j] * B(ja[j],k);
                  }
                  C(i,k) += alpha*r;
               }
            }
         }
         else
         {
            for (k = 0 ; k < B.GetNumColsLocal() ; k ++)
            {
               for (i = 0; i < A.GetNumRowsLocal(); i++) 
               {
                  j1 = ia[i];
                  j2 = ia[i+1];
                  for (j = j1; j < j2; j++) 
                  {
                     C(i,k) += aa[j] * B(ja[j],k);
                  }
               }
            }
         }
      }

   /*
   #endif
   */

      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixMatMat( const double &alpha, CsrMatrixClass<double> &A, DenseMatrixClass<double> &B, const double &beta, DenseMatrixClass<double> &C);

   template <typename T>
   int ParallelCsrMatrixMatMat( const T &alpha, ParallelCsrMatrixClass<T> &A, DenseMatrixClass<T> &B, const T &beta, DenseMatrixClass<T> &C, SequentialVectorClass<T> &working)
   {
      T one = 1.0;

      if( !(A._comm_helper._is_ready) )
      {
         A.SetupMatvec();
      }
      
      SCHURCHEB_CHKERR(B.GetNumColsLocal() != C.GetNumColsLocal());
      
      int ncols = B.GetNumColsLocal();
      int noffd = A.GetOffdMap().GetLengthLocal();
      
      /* setup working matrix */
      if(working.GetLengthLocal() < ncols*noffd)
      {
         working.Resize( ncols*noffd, false, false);
      }
      
      DenseMatrixClass<T> working_mat;
      working_mat.SetupPtr( working.GetData(), noffd, ncols, noffd, kMemoryHost);
      
      /* start communication of offd part */
      A._comm_helper.DataTransferStart(B, kMemoryHost);
      
      /* apply diagonal matvec while communicating */
      CsrMatrixMatMat( alpha, A.GetDiagMat(), B, beta, C);
      //this->_diag_mat.MatVec( 'N', alpha, x, beta, y);
      
      /* finishing up the communication */
      A._comm_helper.DataTransferOver( working_mat, kMemoryHost);
      
      /* now apply offd matvec */
      if( noffd > 0)
      {
         CsrMatrixMatMat( alpha, A.GetOffdMat(), working_mat, one, C);
         //this->_offd_mat.MatVec( 'N', alpha, this->_matvec_working_vec, one, y);
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int ParallelCsrMatrixMatMat( const double &alpha, ParallelCsrMatrixClass<double> &A, 
                                 DenseMatrixClass<double> &B, const double &beta, DenseMatrixClass<double> &C, 
                                 SequentialVectorClass<double> &working);

   /* DD with edge separator */

   int DD(matrix_csr_double &A, matrix_csr_double &M, int n2, int ndom, vector_int &perm, vector_int &dom_ptr, vector_int &Bdiagsizes)
   {
      
      matrix_csr_double AM;
      vector_int map;
      vector_int sep;
      vector_int temp_perm;
      vector_int temp_dom_ptr;
      
      int nC;
      
      /* sum them */
      CsrMatrixAddHost(A, M, AM);
      
      //AM.Plot(NULL,parallel_log::_grank,0,6);
      
      /* apply METIS */
      CsrMatrixMetisKwayHost( AM, ndom, map, false, sep, nC, temp_perm, temp_dom_ptr);
      
      /* next we map domain to new domain number */
      vector_int dom_map;
      dom_map.Setup(ndom);
      
      int i1, i2, i3;
      i1 = ndom % n2;
      i2 = ndom / n2;
      i3 = 0;
      for(int i = 0 ; i < n2 ; i ++)
      {
         if(i < i1)
         {
            for(int j = 0 ; j < i2 + 1 ; j ++)
            {
               dom_map[i3++] = i;
            }
         }
         else
         {
            for(int j = 0 ; j < i2 ; j ++)
            {
               dom_map[i3++] = i;
            }
         }
      }
      
      /* get order */
      int n = A.GetNumRowsLocal();
      //int nB = n - nC;
      
      vector_int num_i;
      vector_int num_o;
      
      num_i.Setup(n2, true);
      num_o.Setup(n2, true);
      
      Bdiagsizes.Setup( n2, true);
      
      vector_int numg_i;
      vector_int numg_o;
      
      numg_i.Setup( ndom, true);
      numg_o.Setup( ndom, true);
      
      for(int i = 0 ; i < n ; i ++)
      {
         if(sep[i] != 0)
         {
            num_o[dom_map[map[i]]]++;
            numg_o[map[i]]++;
         }
         else
         {
            num_i[dom_map[map[i]]]++;
            numg_i[map[i]]++;
         }
      }
      
      /* compute B size */
      for(int i = 0 ; i < ndom ; i ++)
      {
         /* add diagonal block size of a subdomain */
         Bdiagsizes[dom_map[i]] += numg_i[i] * numg_i[i];
      }
      
      dom_ptr.Setup(n2*2+1);
      perm.Setup(n);
      
      dom_ptr[0] = 0;
      for(int i = 0 ; i < n2 ; i ++)
      {
         dom_ptr[i+1] = dom_ptr[i] + num_i[i];
         num_i[i] = dom_ptr[i];
      }
      for(int i = 0 ; i < n2 ; i ++)
      {
         dom_ptr[n2+i+1] = dom_ptr[n2+i] + num_o[i];
         num_o[i] = dom_ptr[n2+i];
      }
      
      for(int i = 0 ; i < n ; i ++)
      {
         if(sep[i] != 0)
         {
            /* C */
            perm[num_o[dom_map[map[i]]]++] = i;
         }
         else
         {
            /* B */
            perm[num_i[dom_map[map[i]]]++] = i;
         }
      }
      
      /*
      if(parallel_log::_grank == 0)
      {
         dom_ptr.Plot(0,0,6);
         perm.Plot(0,0,6);
      }
      */
      
      /* free */
      AM.Clear();
      map.Clear();
      sep.Clear();
      temp_perm.Clear();
      temp_dom_ptr.Clear();
      num_i.Clear();
      num_o.Clear();
      
      return 0;
   }

   /**
    * @brief   This function gets C matrix from original matrix, dom_ptr-> the start of each row of C in A, length n+1. 
    * @details This function gets C matrix from original matrix, dom_ptr-> the start of each row of Cin A, length n+1. \n
    *          C should be initialized outisde
    * @param [in]    A The input parallel CSR matrix.
    * @param [out]   C The out parallel CSR matrix. C should be initialized outisde.
    * @param [in]    dom_ptr The start of each row of C in A, length n+1.
    * @param [in]    parlog The MPI info.
    * @return       Return error message.
    */
   int ExtractParallelCsrSubMatrix(matrix_csr_double &A, matrix_csr_par_double &C, int *dom_ptr, parallel_log &parlog)
   {
      
      int myid, np;
      MPI_Comm comm;
      parlog.GetMpiInfo(np, myid, comm);
      
      int n1 = dom_ptr[0]; // the begin of this submatrix
      int n_local, n_start, n_global;
      
      n_start = dom_ptr[myid] - n1;
      n_local = dom_ptr[myid+1] - dom_ptr[myid];
      n_global = dom_ptr[np] - dom_ptr[0];
      
      C.Setup(n_local, n_start, n_global, n_local, n_start, n_global, parlog);
      
      /* first let's extract diagonal mat */
      A.SubMatrix( n1+n_start, n1+n_start, n_local, n_local, kMemoryHost, C.GetDiagMat());
      
      /* next let's extract offdiagonal mat */
      if(np == 1)
      {
         // Offd is empty
         C.GetOffdMat().Setup(n_global, 0, 0);// empty
         C.GetOffdMat().GetIVector().Fill(0);
         C.GetOffdMat().GetJVector().Fill(0);
         C.GetOffdMat().GetDataVector().Fill(0.0);
      }
      else
      {
         vector_int row_starts, row_ends, col_starts, col_ends;
         
         row_starts.Setup(2, false);
         row_starts[0] = 0;
         row_starts[1] = n_start+n1;
         row_ends.Setup(2, false);
         row_ends[0] = 0;
         row_ends[1] = n_start+n1 + n_local;
         
         col_starts.Setup(2, false);
         col_starts[0] = n1;
         col_starts[1] = n1+n_start+n_local;
         
         col_ends.Setup(2, false);
         col_ends[0] = n1+n_start;
         col_ends[1] = n1+n_global;
         
         A.SubMatrix( row_starts, col_starts, row_ends, col_ends, kMemoryHost, C.GetOffdMat());
         
         /* update the offdiagonal column map */
         vector_int marker;
         int col;
         int *C_offd_i = C.GetOffdMat().GetI();
         int *C_offd_j = C.GetOffdMat().GetJ();
         
         marker.Setup(n_global-n_local, true);
         
         /* mark all offd cols */
         for(int i = 0 ; i < n_local ; i ++)
         {
            int j1 = C_offd_i[i];
            int j2 = C_offd_i[i+1];
            for(int j = j1 ; j < j2 ; j ++)
            {
               col = C_offd_j[j];
               marker[col] = 1;
            }
         }
         
         /* construct mapping, remove empty columns */
         col = 0;
         for(int i = 0 ; i < n_global - n_local ; i ++)
         {
            if(marker[i] > 0)
            {
               marker[i] = col++;
            }
            else
            {
               marker[i] = -1;
            }
         }
         
         /* update column value */
         for(int i = 0 ; i < n_local ; i ++)
         {
            int j1 = C_offd_i[i];
            int j2 = C_offd_i[i+1];
            for(int j = j1 ; j < j2 ; j ++)
            {
               C_offd_j[j] = marker[C_offd_j[j]];
            }
         }
         
         C.GetOffdMat().SetNumCols(col);
         
         /* create offdiagonal map array */
         C.GetOffdMap().Setup(col);
         col = 0;
         for(int i = 0 ; i < n_start ; i ++)
         {
            if(marker[i] >= 0)
            {
               C.GetOffdMap()[col++] = i;
            }
         }
         
         for(int i = n_start ; i < n_global - n_local ; i ++)
         {
            if(marker[i] >= 0)
            {
               C.GetOffdMap()[col++] = n_local + i;
            }
         }
         
         row_starts.Clear();
         row_ends.Clear();
         col_starts.Clear();
         col_ends.Clear();
         marker.Clear();
      }
      
      return 0;
   }

   /**
    * @brief   This function gets C matrix from original matrix, dom_ptr-> the start of each row of C in A, length n+1. 
    * @details This function gets C matrix from original matrix, dom_ptr-> the start of each row of Cin A, length n+1. \n
    *          C should be initialized outisde
    * @param [in]    A The input parallel CSR matrix.
    * @param [out]   C The out parallel CSR matrix. C should be initialized outisde.
    * @param [in]    dom_ptrr The start of each row of C in A, length n+1.
    * @param [in]    dom_ptrc The start of each column of C in A, length n+1.
    * @param [in]    parlog The MPI info.
    * @return       Return error message.
    */
   int ExtractParallelCsrSubMatrix(matrix_csr_double &A, matrix_csr_par_double &C, int *dom_ptrr, int *dom_ptrc, parallel_log &parlog)
   {
      
      int myid, np;
      MPI_Comm comm;
      parlog.GetMpiInfo(np, myid, comm);
      
      int nr = dom_ptrr[0]; // the begin of this submatrix
      int nc = dom_ptrc[0]; // the begin of this submatrix
      int nr_local, nr_start, nr_global;
      int nc_local, nc_start, nc_global;
      
      nr_start = dom_ptrr[myid] - nr;
      nr_local = dom_ptrr[myid+1] - dom_ptrr[myid];
      nr_global = dom_ptrr[np] - dom_ptrr[0];
      
      nc_start = dom_ptrc[myid] - nc;
      nc_local = dom_ptrc[myid+1] - dom_ptrc[myid];
      nc_global = dom_ptrc[np] - dom_ptrc[0];
      
      C.Setup(nr_local, nr_start, nr_global, nc_local, nc_start, nc_global, parlog);
      
      /* first let's extract diagonal mat */
      A.SubMatrix( nr+nr_start, nc+nc_start, nr_local, nc_local, kMemoryHost, C.GetDiagMat());
      
      /* next let's extract offdiagonal mat */
      vector_int row_starts, row_ends, col_starts, col_ends;
      
      row_starts.Setup(2, false);
      row_starts[0] = 0;
      row_starts[1] = nr_start+nr;
      row_ends.Setup(2, false);
      row_ends[0] = 0;
      row_ends[1] = nr_start+nr + nr_local;
      
      col_starts.Setup(2, false);
      col_starts[0] = nc;
      col_starts[1] = nc+nc_start+nc_local;
      
      col_ends.Setup(2, false);
      col_ends[0] = nc+nc_start;
      col_ends[1] = nc+nc_global;
      
      A.SubMatrix( row_starts, col_starts, row_ends, col_ends, kMemoryHost, C.GetOffdMat());
      
      /* update the offdiagonal column map */
      vector_int marker;
      int col;
      int *C_offd_i = C.GetOffdMat().GetI();
      int *C_offd_j = C.GetOffdMat().GetJ();
      
      marker.Setup(nc_global-nc_local, true);
      
      /* mark all offd cols */
      for(int i = 0 ; i < nr_local ; i ++)
      {
         int j1 = C_offd_i[i];
         int j2 = C_offd_i[i+1];
         for(int j = j1 ; j < j2 ; j ++)
         {
            col = C_offd_j[j];
            marker[col] = 1;
         }
      }
      
      /* construct mapping, remove empty columns */
      col = 0;
      for(int i = 0 ; i < nc_global - nc_local ; i ++)
      {
         if(marker[i] > 0)
         {
            marker[i] = col++;
         }
         else
         {
            marker[i] = -1;
         }
      }
      
      /* update column value */
      for(int i = 0 ; i < nr_local ; i ++)
      {
         int j1 = C_offd_i[i];
         int j2 = C_offd_i[i+1];
         for(int j = j1 ; j < j2 ; j ++)
         {
            C_offd_j[j] = marker[C_offd_j[j]];
         }
      }
      
      C.GetOffdMat().SetNumCols(col);
      
      /* create offdiagonal map array */
      C.GetOffdMap().Setup(col);
      col = 0;
      for(int i = 0 ; i < nc_start ; i ++)
      {
         if(marker[i] >= 0)
         {
            C.GetOffdMap()[col++] = i;
         }
      }
      
      for(int i = nc_start ; i < nc_global - nc_local ; i ++)
      {
         if(marker[i] >= 0)
         {
            C.GetOffdMap()[col++] = nc_local + i;
         }
      }
      
      row_starts.Clear();
      row_ends.Clear();
      col_starts.Clear();
      col_ends.Clear();
      marker.Clear();
      
      return 0;
   }

   int TriU(matrix_csr_double &A, matrix_csr_double &B)
   {
      //int ncB = A.GetNumColsGlobal();
      int n = A.GetNumRowsLocal();
      //int nnz = A.GetNumNonzeros();
      
      int row, col;
      int idx = 0;
      
      for(int i = 0 ; i < n ; i ++)
      {
         int i1 = A.GetI()[i];
         int i2 = A.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            /* add one extra shift */
            row = i;
            col = A.GetJ()[j];
            if( row <= col)
            {
               idx++;
            }
         }
      }
      
      B.Setup( n, n, idx);
      
      idx = 0;
      for(int i = 0 ; i < n ; i ++)
      {
         B.GetI()[i] = idx;
         int i1 = A.GetI()[i];
         int i2 = A.GetI()[i+1];
         for(int j = i1 ; j < i2 ; j ++)
         {
            /* add one extra shift */
            row = i;
            col = A.GetJ()[j];
            if( row <= col)
            {
               B.GetJ()[idx] = col;
               B.GetData()[idx++] = A.GetData()[j];
            }
         }
      }
      B.GetI()[n] = idx;
      
      return SCHURCHEB_SUCCESS;
      
   }

   template <typename T>
   int CsrMatrixTransposeHost( CsrMatrixClass<T> &A, CsrMatrixClass<T> &AT)
   {
      int nrows, ncols, nnz;
      
      if( A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Csr matrix transpose only works for the host memory.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();
      nnz = A.GetNumNonzeros();
      
      /* transpose switch rows and cols */
      AT.Setup( ncols, nrows, nnz, A.IsHoldingData(), false);
      
      CsrMatrixP2CscMatrixPHost<0,0>( nrows, ncols, A.IsHoldingData(), A.GetData(), A.GetJ(), A.GetI(), AT.GetData(), AT.GetJ(), AT.GetI());
      
      AT.IsCsr() = A.IsCsr();
      
      /* by the algorithm the col/row of the new csr/csc should be sorted */
      AT.IsRowSorted() = true;
      
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixTransposeHost( CsrMatrixClass<float> &A, CsrMatrixClass<float> &AT);
   template int CsrMatrixTransposeHost( CsrMatrixClass<double> &A, CsrMatrixClass<double> &AT);
   template int CsrMatrixTransposeHost( CsrMatrixClass<complexs> &A, CsrMatrixClass<complexs> &AT);
   template int CsrMatrixTransposeHost( CsrMatrixClass<complexd> &A, CsrMatrixClass<complexd> &AT);
   
   template <typename T>
   int CsrMatrixAddHost( CsrMatrixClass<T> &A, CsrMatrixClass<T> &B, CsrMatrixClass<T> &C)
   {
      /* TODO: add OpenMP support */
      SCHURCHEB_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal() || A.GetNumColsLocal() != B.GetNumColsLocal());
      SCHURCHEB_CHKERR(A.IsCsr() != B.IsCsr());
      
      if( A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("csr matrix add only works for the host memory.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      if( B.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("csr matrix add only works for the host memory.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      //int *iw;
      int   nnzC, i, j, col, pos, nrows, ncols, nnzA, nnzB;
      bool  csr;
      int   *A_i, *A_j, *B_i, *B_j, *C_i, *C_j;
      T     *A_data, *B_data, *C_data;
      
      csr = A.IsCsr();
      
      if( !csr )
      {
         SCHURCHEB_WARNING("in the current version, csradd only works for csr matrix, turn A and B into csr.");
         A.Transpose();
         B.Transpose();
      }
      
      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();
      nnzA = A.GetNumNonzeros();
      nnzB = B.GetNumNonzeros();
      
      /* reserve the max size */
      C.Setup( nrows, ncols, SchurchebMin( SchurchebMax(nrows * ncols, INT_MAX), nnzA + nnzB));
      
      A_i = A.GetI();
      A_j = A.GetJ();
      A_data = A.GetData();
      B_i = B.GetI();
      B_j = B.GetJ();
      B_data = B.GetData();
      C_i = C.GetI();
      C_j = C.GetJ();
      C_data = C.GetData();
      
      //Malloc(iw, A._ncol, int);
      IntVectorClass<int> iw;
      
      iw.Setup( ncols);
      iw.Fill(-1);

      nnzC = 0;
      
      C_i[0] = nnzC;
      for (i = 0; i < nrows; i++) 
      {
         // A
         for (j = A_i[i]; j < A_i[i+1]; j++) 
         {
            col = A_j[j];
            C_j[nnzC] = col;
            C_data[nnzC] = A_data[j];
            iw[col] = nnzC++;
         }
         // B
         for (j = B_i[i]; j < B_i[i+1]; j++) 
         {
            col = B_j[j];
            pos = iw[col];
            if (-1 == pos) 
            {
               C_j[nnzC] = col;
               C_data[nnzC] = B_data[j];
               iw[col] = nnzC++;
            } 
            else 
            {

               SCHURCHEB_CHKERR(C_j[pos] != col);

               C_data[pos] += B_data[j];
            }
         }
         C_i[i+1] = nnzC;
         // reset iw
         for (j = C_i[i]; j < C_i[i+1]; j++) 
         {
            iw[C_j[j]] = -1;
         }
      }
      
      /* update the nnz */
      C.SetNumNonzeros();

      if( A.IsRowSorted() || B.IsRowSorted() )
      {
         C.SortRow();
      }
      
      iw.Clear();
      
      if( !csr )
      {
         A.Transpose();
         B.Transpose();
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixAddHost( CsrMatrixClass<float> &A, CsrMatrixClass<float> &B, CsrMatrixClass<float> &C);
   template int CsrMatrixAddHost( CsrMatrixClass<double> &A, CsrMatrixClass<double> &B, CsrMatrixClass<double> &C);
   template int CsrMatrixAddHost( CsrMatrixClass<complexs> &A, CsrMatrixClass<complexs> &B, CsrMatrixClass<complexs> &C);
   template int CsrMatrixAddHost( CsrMatrixClass<complexd> &A, CsrMatrixClass<complexd> &B, CsrMatrixClass<complexd> &C);
   
   template <typename T>
   int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<T> &A, ParallelCsrMatrixClass<T> &AT)
   {
      
      if( A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Parallel Csr matrix transpose only works for the host memory.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      /* MPI info */
      int np, myid;
      MPI_Comm comm;
      
      A.GetMpiInfo(np, myid, comm);
      
      /* setup matvec to get communication helper */
      A.SetupMatvec();
      
      CsrMatrixClass<T> &A_diag = A.GetDiagMat();
      CsrMatrixClass<T> &A_offd = A.GetOffdMat();
      
      CsrMatrixClass<T> AT_diag;
      CsrMatrixClass<T> AT_offd;
      CooMatrixClass<T> AT_offd_coo;
      
      CsrMatrixTransposeHost(A_diag, AT_diag);
      CsrMatrixTransposeHost(A_offd, AT_offd);
      
      int   *AT_offd_i = AT_offd.GetI();
      int   *AT_offd_j = AT_offd.GetJ();
      T     *AT_offd_data = AT_offd.GetData();
      
      /* setup the structure of AT */
      AT.Setup(A.GetNumColsLocal(), A.GetColStartGlobal(), A.GetNumColsGlobal(), A.GetNumRowsLocal(), A.GetRowStartGlobal(), A.GetNumRowsGlobal(), A);
      
      /* setup the diagonal first */
      AT.GetDiagMat() = std::move(AT_diag);
      
      vector_SCHURCHEB_long &offd_map_v = AT.GetOffdMap();
      
      /* now, we need to send data to targer processor 
       * A._comm_helper._recv_from_v: list of nodes recv data from
       * A._comm_helper._recv_idx_v2: list of solc recv from each proc
       * To send the transpose, send the size of each row first
       */
      int nsends, nsendi, nsendi2, nrecvs, nrecvi, nrecvi2, i, j, j1, j2, k, idx, idx2, n_local, n_offd;
      vector_int marker;
      std::vector<vector_int> send_size_v2, recv_size_v2;
      vector_SCHURCHEB_long send_se_v;
      std::vector<vector_SCHURCHEB_long> recv_se_v2;
      std::vector<vector_int> send_j_v2, recv_j_v2;
      std::vector<SequentialVectorClass<T> > send_data_v2, recv_data_v2;
      vector<MPI_Request> requests_v;
      
      nsends = A._comm_helper._recv_from_v.GetLengthLocal();
      nrecvs = A._comm_helper._send_to_v.GetLengthLocal();
      send_size_v2.resize(nsends);
      send_j_v2.resize(nsends);
      send_data_v2.resize(nsends);
      
      recv_size_v2.resize(nrecvs);
      recv_j_v2.resize(nrecvs);
      recv_data_v2.resize(nrecvs);
      
      requests_v.resize(nsends + nrecvs);
      
      send_se_v.Setup(2);
      send_se_v[0] = A.GetRowStartGlobal();
      send_se_v[1] = send_se_v[0] + A.GetNumRowsLocal();
      recv_se_v2.resize(nrecvs);
      
      /* get the size of each row */
      for(i = 0 ; i < nsends ; i ++)
      {
         nsendi = A._comm_helper._recv_idx_v2[i].GetLengthLocal();
         /* we use the last index to store the total length */
         send_size_v2[i].Setup(nsendi+1);
         
         send_size_v2[i][nsendi] = 0;
         for(j = 0 ; j < nsendi ; j ++)
         {
            k = A._comm_helper._recv_idx_v2[i][j];
            send_size_v2[i][j] = AT_offd_i[k+1] - AT_offd_i[k];
            send_size_v2[i][nsendi] += send_size_v2[i][j];
         }
         
         nsendi2 = send_size_v2[i][nsendi];
         send_j_v2[i].Setup(nsendi2);
         send_data_v2[i].Setup(nsendi2);
         
         idx = 0;
         for(j = 0 ; j < nsendi ; j ++)
         {
            k = A._comm_helper._recv_idx_v2[i][j];
            j1 = AT_offd_i[k];
            j2 = AT_offd_i[k+1];
            for(k = j1 ; k < j2 ; k ++)
            {
               send_j_v2[i][idx] = AT_offd_j[k];
               send_data_v2[i][idx] = AT_offd_data[k];
               idx ++;
            }
         }
      }
      
      /* send target row size */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         nsendi = A._comm_helper._recv_idx_v2[i].GetLengthLocal();
         SCHURCHEB_MPI_CALL( SchurchebMpiIsend( send_size_v2[i].GetData(), nsendi+1, 
               A._comm_helper._recv_from_v[i], 0, comm, &(requests_v[j++])) );
      }
      
      for(i = 0 ; i < nrecvs ; i ++)
      {
         nrecvi = A._comm_helper._send_idx_v2[i].GetLengthLocal();
         recv_size_v2[i].Setup(nrecvi+1);
         
         SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( recv_size_v2[i].GetData(), nrecvi+1,
               A._comm_helper._send_to_v[i], 0, comm, &(requests_v[j++])) );
      }
      
      SCHURCHEB_MPI_CALL(MPI_Waitall( nsends+nrecvs, requests_v.data(), MPI_STATUSES_IGNORE));
      
      //recv_size_v2[0].Plot(0,0,6);
      
      /* The data str should be ready now
       * start to create buffer to recv income data 
       */
      
      /* send idx first */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         nsendi = A._comm_helper._recv_idx_v2[i].GetLengthLocal();
         nsendi2 = send_size_v2[i][nsendi];
         SCHURCHEB_MPI_CALL( SchurchebMpiIsend( send_j_v2[i].GetData(), nsendi2, 
               A._comm_helper._recv_from_v[i], 0, comm, &(requests_v[j++])) );
      }
      
      n_offd = 0;
      for(i = 0 ; i < nrecvs ; i ++)
      {
         nrecvi = A._comm_helper._send_idx_v2[i].GetLengthLocal();
         nrecvi2 = recv_size_v2[i][nrecvi];
         recv_j_v2[i].Setup(nrecvi2);
         n_offd += nrecvi2;
         
         SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( recv_j_v2[i].GetData(), nrecvi2,
               A._comm_helper._send_to_v[i], 0, comm, &(requests_v[j++])) );
      }
      
      SCHURCHEB_MPI_CALL(MPI_Waitall( nsends+nrecvs, requests_v.data(), MPI_STATUSES_IGNORE));
      
      //recv_j_v2[0].Plot(0,0,6);
      
      /* now send data */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         nsendi = A._comm_helper._recv_idx_v2[i].GetLengthLocal();
         nsendi2 = send_size_v2[i][nsendi];
         SCHURCHEB_MPI_CALL( SchurchebMpiIsend( send_data_v2[i].GetData(), nsendi2, 
               A._comm_helper._recv_from_v[i], 0, comm, &(requests_v[j++])) );
      }
      
      for(i = 0 ; i < nrecvs ; i ++)
      {
         nrecvi = A._comm_helper._send_idx_v2[i].GetLengthLocal();
         nrecvi2 = recv_size_v2[i][nrecvi];
         recv_data_v2[i].Setup(nrecvi2);
         
         SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( recv_data_v2[i].GetData(), nrecvi2,
               A._comm_helper._send_to_v[i], 0, comm, &(requests_v[j++])) );
      }
      
      SCHURCHEB_MPI_CALL(MPI_Waitall( nsends+nrecvs, requests_v.data(), MPI_STATUSES_IGNORE));
      
      //recv_data_v2[0].Plot(0,0,6);
      
      /* now send start/end index of new cols */
      j = 0;
      for(i = 0 ; i < nsends ; i ++)
      {
         SCHURCHEB_MPI_CALL( SchurchebMpiIsend( send_se_v.GetData(), 2, 
               A._comm_helper._recv_from_v[i], 0, comm, &(requests_v[j++])) );
      }
      
      for(i = 0 ; i < nrecvs ; i ++)
      {
         recv_se_v2[i].Setup(2);
         
         SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( recv_se_v2[i].GetData(), 2,
               A._comm_helper._send_to_v[i], 0, comm, &(requests_v[j++])) );
      }
      
      SCHURCHEB_MPI_CALL(MPI_Waitall( nsends+nrecvs, requests_v.data(), MPI_STATUSES_IGNORE));
      
      //recv_data_v2[0].Plot(0,0,6);
      
      /* Now, each proc has the external data
       * We can start putting data in
       * 
       * |---------|---------|          |---------|
       * | x x o o | x o x o |          | x o o o |
       * | x x x o | o x o x |          | o x o o |
       * | o x x x | o o x o |          | x o x o |
       * | o o x x | o o o x |          | o x o x |
       * |---------|---------| => offd  |---------| => send to p1, need to build offd_map first.
       * | x o x o | x x o o |
       * | o x o x | x x x o |
       * | o o x o | o x x x |
       * | o o o x | o o x x |
       * |---------|---------|
       * 
       * we do it section by section.
       * 
       */
      
      /* the coo matrix of offd */
      AT_offd_coo.Setup( A.GetNumColsLocal(), INT_MAX, n_offd);
      
      /* loop through all recv parts */
      idx = 0;
      for(i = 0 ; i < nrecvs ; i ++)
      {
         /* nrecvi is the number of rows in this section */
         nrecvi = A._comm_helper._send_idx_v2[i].GetLengthLocal();
         
         /* n_local is the number of local columns in this proc */
         n_local = recv_se_v2[i][1] - recv_se_v2[i][0];
         
         /* create the helper array of size n_local */
         marker.Setup(n_local);
         marker.Fill(-1);
         
         /* now loop through all rows 
          * nrecvi2 is the size of local offd matrix
          */
         nrecvi2 = recv_size_v2[i][nrecvi];
         
         for(j = 0 ; j < nrecvi2 ; j ++)
         {
            k = recv_j_v2[i][j];
            if(marker[k] < 0)
            {
               /* this is a new col, mark it */
               marker[k] = 1;
            }
         }
         
         /* now update offd_map_v */
         for(j = 0 ; j < n_local ; j ++)
         {
            if(marker[j] > 0)
            {
               /* got an offd map */
               marker[j] = idx;
               offd_map_v.PushBack(recv_se_v2[i][0]+j);
               idx ++;
            }
         }
         
         /* now insert value */
         idx2 = 0;
         for(j = 0 ; j < nrecvi ; j ++)
         {
            /* j1 is the new row idx */
            j1 = A._comm_helper._send_idx_v2[i][j];
            
            /* j2 is the number of data in this row */
            j2 = recv_size_v2[i][j];
            for(k = 0 ; k < j2 ; k ++)
            {
               AT_offd_coo.PushBack( j1, marker[recv_j_v2[i][idx2]], recv_data_v2[i][idx2]);
               idx2 ++;
            }
         }
      }
      
      /* done */
      AT_offd_coo.ToCsr(kMemoryHost, AT.GetOffdMat());
      
      /* deallocate */
      
      AT_diag.Clear();
      AT_offd.Clear();
      AT_offd_coo.Clear();
      
      marker.Clear();
      
      for(i = 0 ; i < nrecvs ; i ++)
      {
         recv_se_v2[i].Clear();
         recv_size_v2[i].Clear();
         recv_j_v2[i].Clear();
         recv_data_v2[i].Clear();
      }
      std::vector<vector_int>().swap(recv_size_v2);
      std::vector<vector_SCHURCHEB_long>().swap(recv_se_v2);
      std::vector<vector_int>().swap(recv_j_v2);
      std::vector<SequentialVectorClass<T> >().swap(recv_data_v2);
      
      send_se_v.Clear();
      for(i = 0 ; i < nsends ; i ++)
      {
         send_size_v2[i].Clear();
         send_j_v2[i].Clear();
         send_data_v2[i].Clear();
      }
      std::vector<vector_int>().swap(send_size_v2);
      std::vector<vector_int>().swap(send_j_v2);
      std::vector<SequentialVectorClass<T> >().swap(send_data_v2);
      
      vector<MPI_Request>().swap(requests_v);
      
      return SCHURCHEB_SUCCESS;
   }
   template int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<float> &A, ParallelCsrMatrixClass<float> &AT);
   template int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<double> &A, ParallelCsrMatrixClass<double> &AT);
   template int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<complexs> &A, ParallelCsrMatrixClass<complexs> &AT);
   template int ParallelCsrMatrixTransposeHost( ParallelCsrMatrixClass<complexd> &A, ParallelCsrMatrixClass<complexd> &AT);
   
   template <typename T>
   int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<T> &A, ParallelCsrMatrixClass<T> &B, ParallelCsrMatrixClass<T> &C)
   {
      /* TODO: add OpenMP support */
      
      if( A.GetDataLocation() == kMemoryDevice || B.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Parallel Csr matrix add only works for the host memory.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      /* size should match */
      SCHURCHEB_CHKERR(A.GetNumRowsGlobal() != B.GetNumRowsGlobal());
      SCHURCHEB_CHKERR(A.GetNumColsGlobal() != B.GetNumColsGlobal());
      SCHURCHEB_CHKERR(A.GetNumRowsLocal() != B.GetNumRowsLocal());
      SCHURCHEB_CHKERR(A.GetNumColsLocal() != B.GetNumColsLocal());
      SCHURCHEB_CHKERR(A.GetRowStartGlobal() != B.GetRowStartGlobal());
      SCHURCHEB_CHKERR(A.GetColStartGlobal() != B.GetColStartGlobal());
      
      /* MPI info */
      int np, myid;
      MPI_Comm comm;
      
      A.GetMpiInfo(np, myid, comm);
      
      /* setup C */
      C.Setup(A.GetNumRowsLocal(), A.GetRowStartGlobal(), A.GetNumRowsGlobal(), A.GetNumColsLocal(), A.GetColStartGlobal(), A.GetNumColsGlobal(), A);
      
      /* add diagonal */
      CsrMatrixAddHost(A.GetDiagMat(), B.GetDiagMat(), C.GetDiagMat());
      
      int A_n_offd, B_n_offd, C_n_offd;
      
      /* now, add offdiagonal 
       * need to build the offdiagonal map first
       */
      int i, j, j1, j2, idx, nnzA, nnzB, nnzC, n_local, col, pos;
      vector_int order, B_map, iw;
      vector_SCHURCHEB_long A_offd_map_v_sorted;
      
      n_local = A.GetNumRowsLocal();
      
      CsrMatrixClass<T> &A_offd = A.GetOffdMat();
      CsrMatrixClass<T> &B_offd = B.GetOffdMat();
      CsrMatrixClass<T> &C_offd = C.GetOffdMat();
      
      vector_SCHURCHEB_long &A_offd_map_v = A.GetOffdMap();
      vector_SCHURCHEB_long &B_offd_map_v = B.GetOffdMap();
      vector_SCHURCHEB_long &C_offd_map_v = C.GetOffdMap();
      
      A_n_offd = A_offd_map_v.GetLengthLocal();
      B_n_offd = B_offd_map_v.GetLengthLocal();
      
      /* merge them together */
      
      /* 1st step, sort A */
      A_offd_map_v.Sort(order, true, false);
      A_offd_map_v_sorted.Setup(A_n_offd);
      order.GatherPerm(A_offd_map_v, A_offd_map_v_sorted);
      
      /* 2nd step, get map from B to new C_offd_map */
      C_offd_map_v = A_offd_map_v;
      B_map.Setup(B_n_offd);
      for(i = 0 ; i < B_n_offd ; i ++)
      {
        if(A_offd_map_v_sorted.BinarySearch(B_offd_map_v[i], idx, true) < 0)
        {
           /* this is a new column */
           B_map[i]  = C_offd_map_v.GetLengthLocal();
           C_offd_map_v.PushBack(B_offd_map_v[i]);
        }
        else
        {
           /* this is a old column */
           B_map[i] = idx;
        }
      }
      
      nnzA = A_offd.GetNumNonzeros();
      nnzB = B_offd.GetNumNonzeros();
      
      C_n_offd = C_offd_map_v.GetLengthLocal();
      
      /* now, start adding those two offdiagonal matrices */
      C_offd.Setup( n_local, C_n_offd, SchurchebMin( SchurchebMax(n_local * C_n_offd, INT_MAX), nnzA + nnzB));
      iw.Setup(C_n_offd);
      iw.Fill(-1);
      
      int *A_i = A_offd.GetI();
      int *B_i = B_offd.GetI();
      int *C_i = C_offd.GetI();
      int *A_j = A_offd.GetJ();
      int *B_j = B_offd.GetJ();
      int *C_j = C_offd.GetJ();
      T *A_data = A_offd.GetData();
      T *B_data = B_offd.GetData();
      T *C_data = C_offd.GetData();
      
      nnzC = 0;
      C_i[0] = nnzC;
      for (i = 0; i < n_local; i++) 
      {
         // A
         j1 = A_i[i];
         j2 = A_i[i+1];
         for (j = j1; j < j2; j++) 
         {
            col = A_j[j];
            C_j[nnzC] = col;
            C_data[nnzC] = A_data[j];
            iw[col] = nnzC++;
         }
         // B
         j1 = B_i[i];
         j2 = B_i[i+1];
         for (j = j1; j < j2; j++) 
         {
            col = B_map[B_j[j]];
            pos = iw[col];
            if (-1 == pos) 
            {
               C_j[nnzC] = col;
               C_data[nnzC] = B_data[j];
               iw[col] = nnzC++;
            } 
            else 
            {
               SCHURCHEB_CHKERR(C_j[pos] != col);
               C_data[pos] += B_data[j];
            }
         }
         C_i[i+1] = nnzC;
         // reset iw
         for (j = C_i[i]; j < C_i[i+1]; j++) 
         {
            iw[C_j[j]] = -1;
         }
      }
      
      /* update the nnz */
      C_offd.SetNumNonzeros();
      
      /* deallocate */
      order.Clear(); 
      B_map.Clear();
      iw.Clear();
      A_offd_map_v_sorted.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   template int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<float> &A, ParallelCsrMatrixClass<float> &B, ParallelCsrMatrixClass<float> &C);
   template int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<double> &A, ParallelCsrMatrixClass<double> &B, ParallelCsrMatrixClass<double> &C);
   template int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<complexs> &A, ParallelCsrMatrixClass<complexs> &B, ParallelCsrMatrixClass<complexs> &C);
   template int ParallelCsrMatrixAddHost( ParallelCsrMatrixClass<complexd> &A, ParallelCsrMatrixClass<complexd> &B, ParallelCsrMatrixClass<complexd> &C);
   
   template <typename T>
   int CsrMatrixSortRow( CsrMatrixClass<T> &A)
   {
      if(A.IsRowSorted() || A.GetNumNonzeros() == 0)
      {
         /* if already sorted, do nothing */
         return SCHURCHEB_SUCCESS;
      }
      
      int                        i, length;
      int                        ncols, nrows;
      int                        *a_i, *a_j;
      T                          *a_data = NULL;
      int                        location;
      bool                       hold_data;
      
      vector_int                 cols, ord;
      SequentialVectorClass<T>   vals;

#ifdef SCHURCHEB_CUDA
      if(A.GetDataLocation() == kMemoryDevice || A.GetDataLocation() == kMemoryUnified)
      {
         /* prefer to use the GPU for unified memory */
         CsrMatrixSortRowDevice(A);
         
         return SCHURCHEB_SUCCESS;
      }
#endif
      
      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();
      
      hold_data = A.IsHoldingData();
      location = A.GetDataLocation();
      
      a_i = A.GetI();
      a_j = A.GetJ();
      
      if(hold_data)
      {
         a_data = A.GetData();
      }
      
      /* sort row index in ascending order */
      if(A.IsCsr())
      {
         /* csr matrix */
#ifdef SCHURCHEB_OPENMP
/* use dynamic here since we don't know the size of each row in advance */
#pragma omp parallel for private(i, length, cols, vals, ord) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
#endif
         for(i = 0 ; i < nrows ; i ++)
         {
            length = a_i[i+1]-a_i[i];
            
            cols.SetupPtr( a_j + a_i[i], length, location);
            
            if(hold_data)
            {
               /* sort in ascending order */
               cols.Sort( ord, true, false);
               
               vals.SetupPtr( a_data + a_i[i], length, location);
               
               /* apply the permutation */
               cols.Perm(ord);
               vals.Perm(ord);
               
               vals.Clear();
               ord.Clear();
               
            }
            else
            {
               /* in this case just sort cols in ascending order */
               cols.Sort(true);
            }
            
            cols.Clear();
         }
      }
      else
      {
         /* csc matrix */
#ifdef SCHURCHEB_OPENMP
/* use dynamic here since we don't know the size of each col in advance */
#pragma omp parallel for private(i, length, cols, vals, ord) SCHURCHEB_OPENMP_SCHEDULE_DEFAULT
#endif
         for(i = 0 ; i < ncols ; i ++)
         {
            length = a_i[i+1]-a_i[i];
            
            cols.SetupPtr( a_j + a_i[i], length, location);
            
            if(hold_data)
            {
               /* sort in ascending order */
               cols.Sort( ord, true, false);
               
               vals.SetupPtr( a_data + a_i[i], length, location);
               
               /* apply the permutation */
               cols.Perm(ord);
               vals.Perm(ord);
               
               vals.Clear();
               ord.Clear();
               
            }
            else
            {
               /* in this case just sort cols in ascending order */
               cols.Sort(true);
            }
            
            cols.Clear();
         }
      }
      
      A.IsRowSorted() = true;
      
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixSortRow( CsrMatrixClass<float> &A);
   template int CsrMatrixSortRow( CsrMatrixClass<double> &A);
   template int CsrMatrixSortRow( CsrMatrixClass<complexs> &A);
   template int CsrMatrixSortRow( CsrMatrixClass<complexd> &A);
   
   template <typename T>
   int CsrMatrixMetisKwayHost( CsrMatrixClass<T> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr)
   {
      /* TODO: OpenMP implementation */
      SCHURCHEB_CHKERR(num_dom <= 0);
      
      int      nrows, ncols, nnz, col, p, i, i1, i2, j, jj, err = 0;
      int      *A_i, *A_j;
      //T        *A_data;
      
      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();
      
      if (nrows != ncols) 
      {
         SCHURCHEB_ERROR("METIS partition only works for square matrix.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      if( A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("METIS partition only works on the host memory.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      if(num_dom == 1)
      {
         /* in this case no need to do the partition */
      }
      
      /* # of nnz */
      nnz = A.GetNumNonzeros();

      /* sort rows of A */
      A.SortRow();
      
      A_i = A.GetI();
      A_j = A.GetJ();
      //A_data = A.GetData();
      
      // Prepare data structures used by METIS
      //SCHURCHEB_long lone = 1;
      IntVectorClass<SCHURCHEB_long> xadj;
      IntVectorClass<SCHURCHEB_long> adjncy;
      IntVectorClass<SCHURCHEB_long> vwgt;
      IntVectorClass<SCHURCHEB_long> adjwgt;
      IntVectorClass<SCHURCHEB_long> lmap;

      lmap.Setup(nrows);
      xadj.Setup(nrows+1);
      adjncy.Setup(nnz);
      vwgt.Setup(nrows, true);
      adjwgt.Setup(nnz);

      /* Fill the vectors with the appropriate values */
      
      map.Setup(nrows);
      perm.Setup(nrows);
      dom_ptr.Setup(num_dom+1, true);
      
      // Costruct a CSR-like representation of A as required by METIS. Extract and keep the diagonal entries separately. 
      xadj[0] = 0;
      jj = 0;
      for ( i = 0 ; i < nrows; i++) 
      {
         i1 = A_i[i];
         i2 = A_i[i+1];
         for ( j = i1; j < i2; j++) 
         {
            col = A_j[j];
            if (col != i) 
            {
               adjncy[jj] = col;
               /* at least should be 1 */
               //adjwgt[jj] = SchurchebMax(adjwgt[jj], lone);
               //adjwgt[jj] = (SCHURCHEB_long) SchurchebAbs(A_data[j]);
               adjwgt[jj] = 1;
               jj++;
            } 
            else 
            {
               //vwgt[i] = (SCHURCHEB_long) SchurchebAbs(A_data[j]);
               //vwgt[i] = SchurchebMax(vwgt[i], lone);
               vwgt[i] = 6;
            }
         }
         if(vwgt[i] == 0)
         {
            /* in this case, we don't have a diagonal entry */
            vwgt[i] = 6;
         }
         xadj[i+1] = jj;
      }
      
      /* METIS parameters, note that  is probably used */
      
      SCHURCHEB_long lnrows = (SCHURCHEB_long)nrows;
      SCHURCHEB_long lnum_dom = (SCHURCHEB_long)num_dom;
      SCHURCHEB_long ledgecut;
      SCHURCHEB_long ncon = 1;        //no weight in use
      
      /* call METIS */
      
      if(lnum_dom > lnrows)
      {
         lnum_dom = lnrows;
      }
      
      if(lnum_dom >= 8)
      {
         METIS_PartGraphKway(&lnrows, &ncon, xadj.GetData(), adjncy.GetData(), vwgt.GetData(), NULL, adjwgt.GetData(), &lnum_dom, NULL, NULL, NULL, &ledgecut, lmap.GetData());
      }
      else
      {
         METIS_PartGraphRecursive(&lnrows, &ncon, xadj.GetData(), adjncy.GetData(), vwgt.GetData(), NULL, adjwgt.GetData(), &lnum_dom, NULL, NULL, NULL, &ledgecut, lmap.GetData());
      }
      
      /* transfer from  to int */
      num_dom = (int) lnum_dom;
      for ( i = 0; i < nrows; i++) 
      {
         map[i] = (int)lmap[i];
      }
      
      /* Determine the number of nodes associated with each partition (subdomain)
       * e.g., two subdomains: dom_ptr = [0,x,y]
       */
      for ( i = 0; i < nrows; i++) 
      {
         dom_ptr[map[i]+1] ++;
      }
      /* Accumulate nodes, e.g., two subdomains: dom_ptr = [0,x,x+y] */
      int num_dom2 = 0;
      for ( i = 0; i < num_dom; i++) 
      {
         if(dom_ptr[i+1] > 0)
         {
            num_dom2 ++;
         }
         dom_ptr[i+1] += dom_ptr[i];
      }
      
      /* in this case, we have some empty domains 
       * remove them
       */
      if(num_dom2 < num_dom)
      {
         vector_int dom_ptr2, map2;
         
         dom_ptr2.Setup(num_dom+1);
         map2.Setup(num_dom);
         map2.Fill(-1);
         
         SCHURCHEB_MEMCPY( dom_ptr2.GetData(), dom_ptr.GetData(), num_dom+1, kMemoryHost, kMemoryHost, int);
         
         /* locate empty domains */
         dom_ptr.Setup(num_dom2+1);
         dom_ptr.Setup(num_dom2+1);
         num_dom2 = 0;
         dom_ptr[0] = 0;
         for(i = 0 ; i < num_dom ; i ++)
         {
            if(dom_ptr2[i+1] > dom_ptr2[i])
            {
               map2[i] = num_dom2;
               dom_ptr[++num_dom2] = dom_ptr2[i+1];
            }
         }
         
         /* update the map */
         for ( i = 0; i < nrows; i++) 
         {
            map[i] = map2[map[i]];
         }
         
         dom_ptr2.Clear();
         map2.Clear();
         num_dom = num_dom2;
      }
      
      /* dom_ptr[num_dom] should be equal to nrow */
      SCHURCHEB_CHKERR(dom_ptr[num_dom] != nrows);

      /* build perm array */
      for ( i = 0; i < nrows; i++) 
      {
         /* determine the domain of the ith node */
         p = map[i]; // maps i to its subdomain
         perm[dom_ptr[p]++] = i;
      }

      // Re-adjust: dom_ptr = [0,x,x+y]
      for ( i = num_dom; i > 0; i--) 
      {
         dom_ptr[i] = dom_ptr[i-1];
      }
      dom_ptr[0] = 0;
      
      /* now start building the seperator */
      if(vertexsep && num_dom == 2)
      {
         /* vertex seperator, only supports num_dom == 2 */
         sep.Setup(nrows, true);
         edgecut = 0;
         
         vector_int sep_size;
         sep_size.Setup(num_dom, true);
         
         int nmatch, nbound_a, nbound_b;
         vector_int bound_a, bound_b;
         vector_int match_a, match_b;
         CsrMatrixClass<T> B;
         
         bound_a.Setup(0, (int)ledgecut, kMemoryHost, false);
         bound_b.Setup(0, (int)ledgecut, kMemoryHost, false);
         
         for(i = 0 ; i < nrows ; i ++)
         {
            i1 = A_i[i];
            i2 = A_i[i+1];
            p = map[i];
            if(p == 0)
            {
               for(j = i1 ; j < i2 ; j ++)
               {
                  col = A_j[j];
                  if(map[col] == 1)
                  {
                     /* have nbhs in different domain */
                     bound_a.PushBack(i);
                     break;
                  }
               }
            }
            else
            {
               for(j = i1 ; j < i2 ; j ++)
               {
                  col = A_j[j];
                  if(map[col] == 0)
                  {
                     /* have nbhs in different domain */
                     bound_b.PushBack(i);
                     break;
                  }
               }
            }
         }
         
         A.SubMatrix(bound_a, bound_b, kMemoryHost, B);
         
         int *B_i = B.GetI();
         int *B_j = B.GetJ();
         
         CsrMatrixMaxMatchingHost( B, nmatch, match_a, match_b);
         
         nbound_a = match_a.GetLengthLocal();
         nbound_b = match_b.GetLengthLocal();
         
         /* for CSR matrix, just pick with match_a, otherwise we would need to compute the transpose
          * now we have A, B, and D (unmatched)
          * we pick those from A \intersection D from A, and thse not its neiborhood in B, as not neiborhood
          */
         match_b.Fill(-1);
         for(i = 0 ; i < nbound_a ; i ++)
         {
            if(match_a[i] < 0)
            {
               /* this is an unmatched node */
               i1 = B_i[i];
               i2 = B_i[i+1];
               for(j = i1 ; j < i2 ; j ++)
               {
                  col = B_j[j];
                  /* this is a neiborhood, pick it */
                  match_b[col] = 1;
               }
            }
            else
            {
               /* this is an matched node, put into the seperator */
               sep[bound_a[i]] = 1;
               sep_size[0]++;
               edgecut++;
            }
         }
         
         for(i = 0 ; i < nbound_b ; i ++)
         {
            if(match_b[i] > 0)
            {
               /* this is an matched node, put into the seperator */
               sep[bound_b[i]] = 1;
               sep_size[1]++;
               edgecut++;
            }
         }
         
         /* now we need to check if any subdomain
          * has no interior nodes
          */
         
         for(i = 0 ; i < num_dom ; i ++)
         {
            if(sep_size[i] == dom_ptr[i+1] - dom_ptr[i])
            {
               err = SCHURCHEB_RETURN_METIS_NO_INTERIOR;
               break;
            }
         }
         sep_size.Clear();
      }
      else
      {
         /* edge seperator */
         sep.Setup(nrows, true);
         edgecut = 0;
         
         vector_int sep_size;
         sep_size.Setup(num_dom, true);
         
         for(i = 0 ; i < nrows ; i ++)
         {
            i1 = A_i[i];
            i2 = A_i[i+1];
            p = map[i];
            for(j = i1 ; j < i2 ; j ++)
            {
               col = A_j[j];
               if(p != map[col])
               {
                  /* have nbhs in different domain */
                  sep[i] = 1;
                  sep_size[p]++;
                  edgecut++;
                  break;
               }
            }
         }
         
         /* now we need to check if any subdomain
          * has no interior nodes
          */
         for(i = 0 ; i < num_dom ; i ++)
         {
            if(sep_size[i] == dom_ptr[i+1] - dom_ptr[i])
            {
               err = SCHURCHEB_RETURN_METIS_NO_INTERIOR;
               break;
            }
         }
         sep_size.Clear();
      }
      
      lmap.Clear();
      xadj.Clear();
      adjncy.Clear();
      vwgt.Clear();
      adjwgt.Clear();
      
      return err;
      
   }
   template int CsrMatrixMetisKwayHost( CsrMatrixClass<float> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr);
   template int CsrMatrixMetisKwayHost( CsrMatrixClass<double> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr);
   template int CsrMatrixMetisKwayHost( CsrMatrixClass<complexs> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr);
   template int CsrMatrixMetisKwayHost( CsrMatrixClass<complexd> &A, int &num_dom, IntVectorClass<int> &map, bool vertexsep, IntVectorClass<int> &sep, int &edgecut, IntVectorClass<int> &perm, IntVectorClass<int> &dom_ptr);
   
   template <typename T>
   int CsrMatrixMaxMatchingHost( CsrMatrixClass<T> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col)
   {
      int      nrows, ncols;
      
      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();
      
      if(match_row.GetLengthLocal() != nrows)
      {
         match_row.Setup(nrows);
      }
      
      if(match_col.GetLengthLocal() != ncols)
      {
         match_col.Setup(ncols);
      }
      
      if(A.GetDataLocation() == kMemoryDevice || match_row.GetDataLocation() == kMemoryDevice || match_col.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("MaxMatching only works for the host memory.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      if(nrows == 0 || ncols == 0)
      {
         nmatch = 0;
         return SCHURCHEB_SUCCESS;
      }
      
      int         i;
      vector_int  dist;
      int         *A_i = A.GetI();
      int         *A_j = A.GetJ();
      
      /* set all to unmached */
      match_row.Fill(ncols);
      match_col.Fill(nrows);
      nmatch = 0;
      dist.Setup(nrows+1);
      
      while( CsrMatrixMaxMatchingBfsHost(nrows, ncols, A_i, A_j, dist, match_row, match_col) )
      {
         for(i = 0 ; i < nrows ; i ++)
         {
            if(match_row[i] == ncols)
            {
               if( CsrMatrixMaxMatchingDfsHost(nrows, ncols, A_i, A_j, dist, match_row, match_col, i) )
               {
                  /* this is an augmenting path */
                  nmatch++;
               }
            }
         }
      }
      
      for(i = 0 ; i < nrows ; i ++)
      {
         if(match_row[i] == ncols)
         {
            match_row[i] = -1;
         }
      }
      
      for(i = 0 ; i < ncols ; i ++)
      {
         if(match_col[i] == nrows)
         {
            match_col[i] = -1;
         }
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixMaxMatchingHost( CsrMatrixClass<float> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col);
   template int CsrMatrixMaxMatchingHost( CsrMatrixClass<double> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col);
   template int CsrMatrixMaxMatchingHost( CsrMatrixClass<complexs> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col);
   template int CsrMatrixMaxMatchingHost( CsrMatrixClass<complexd> &A, int &nmatch, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col);
   
   bool CsrMatrixMaxMatchingBfsHost(int nrows, int ncols, int *A_i, int *A_j, IntVectorClass<int> &dist, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col)
   {
      int i, j, j1, j2, col, row, qs, qe, maxlength;
      vector_int queue;
      
      /* we can't visit more than nrows times */
      maxlength = nrows+1;
      
      queue.Setup(maxlength);
      qe = 0;
      qs = 0;
      
      for(i = 0 ; i < nrows ; i ++)
      {
         if(match_row[i] == ncols)
         {
            /* free vertex, enqueue */
            dist[i] = 0;
            queue[qe++] = i;
         }
         else
         {
            /* matched vertex */
            dist[i] = maxlength;
         }
      }
      dist[nrows] = maxlength;
      
      while(qe > qs)
      {
         /* dequeue */
         i = queue[qs++];
         if( dist[i] < dist[nrows])
         {
            /* we only add the shortest ones 
             * If i == nrows, this is the end node.
             * Otherwise:
             * 
             * if dist[i] == 0, this is a free vertex
             * the connection from i to any of its nbhd is through unmatched edge.
             * 
             * if dist[i] > 0, this is a matched vertex connected with another matched vertex
             * through matched edge
             */
            j1 = A_i[i];
            j2 = A_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               /* for each row find all the col nbhd 
                * note that we are going through an unmatched edge
                * also we won't go back to the same u node
                */
               col = A_j[j];
               
               /* find the match row of this col 
                * if row == nrows this is unmatched
                */
               row = match_col[col];

               /* go through the matched edge 
                * note that we won't have row == i
                */
               if(dist[row] == maxlength)
               {
                  /* if the matched edge has not yet been inside the queue */
                  dist[row] = dist[i] + 1;
                  queue[qe++] = row;
               }
            }
         }
      }
      
      queue.Clear();
      
      return dist[nrows] < maxlength;
   }

   bool CsrMatrixMaxMatchingDfsHost(int nrows, int ncols, int *A_i, int *A_j, IntVectorClass<int> &dist, IntVectorClass<int> &match_row, IntVectorClass<int> &match_col, int i)
   {
      int j, j1, j2, col, row, maxlength, distip1;
      
      maxlength = nrows + 1;
      distip1 = dist[i]+1;
      
      if(i < nrows)
      {
         j1 = A_i[i];
         j2 = A_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = A_j[j];
            row = match_col[col];
            if(dist[row] == distip1)
            {
               /* possible next, search */
               if( CsrMatrixMaxMatchingDfsHost(nrows, ncols, A_i, A_j, dist, match_row, match_col, row) )
               {
                  /* reset match */
                  match_row[i] = col;
                  match_col[col] = i;
                  return true;
               }
            }
         }
         /* deadend, avoid visit here again */
         dist[i] = maxlength;
         return false;
      }
      return true;
   }
   
   int ParmetisKwayHost(vector_SCHURCHEB_long &vtxdist, vector_SCHURCHEB_long &xadj, vector_SCHURCHEB_long &adjncy, SCHURCHEB_long &num_dom, vector_SCHURCHEB_long &map, parallel_log &parlog)
   {
      /* Declare variables */
      SCHURCHEB_long    i, j, j1, j2, k, nrow, refs;
      SCHURCHEB_long    wtflag, numflag, edgecut, num_dom2, ncon, idx, idx2, idx3;
      
      int               nI, nC;
      SCHURCHEB_long    nI_l, nC_l, nI_global, nC_global;
      vector_int        isolate;
      
      /* first we mark those isolated nodes, that is, those nodes that has no connection with other nodes */
      MPI_Comm          comm;
      int               np, myid;
      parlog.GetMpiInfo(np, myid, comm);
      
      /* first check the size of the problem */
      if(vtxdist[np] < np || vtxdist[np] < num_dom)
      {
         /* In this case, we have empty processors, or ndom is larger than the problem size
          * there is no points for keep doing the partition
          */
         //SCHURCHEB_WARNING("Problem too small, try to reduce number of suddomains.");
         num_dom = 0;
         return SCHURCHEB_RETURN_METIS_PROBLEM_TOO_SMALL;
      }
      
      nrow = vtxdist[myid + 1] - vtxdist[myid];
      
      isolate.Setup(nrow, true);
      nI = 0;
      
      j1 = xadj[0];
      for(i = 0 ; i < nrow ; i ++)
      {
         j2 = xadj[i+1];
         if(j1 == j2)
         {
            isolate[i] = 1;
            nI ++;
         }
         j1 = j2;
      }
      
      nC = nrow - nI;
      
      /* now we have local isolated nI and connected nC  */
      nI_l = nI;
      SCHURCHEB_MPI_CALL(SchurchebMpiAllreduce( &nI_l, &nI_global, 1, MPI_SUM,comm));
      nC_global = vtxdist[np] - nI_global;
      
      if(nC_global < np || nC_global < num_dom)
      {
         /* in this case, we don't have enough "connected" nodes,
          * we apply an naive partition without calling ParMETIS.
          * We assign node to i%num_dom
          */
         map.Setup(nrow);
         for(i = 0, j = vtxdist[myid] ; i < nrow ; i ++, j++)
         {
            /* assign global node number ig = ig % num_dom */
            map[i] = j % num_dom;
         }
         return SCHURCHEB_SUCCESS;
      }
      
      /* if we reach here, we have enough connected nodes (more than np), now we need to check
       * if we need redistribute
       * Exaplme:
       * np  node
       *  0  1 3 4
       *  1  
       *  2  9 10 12
       *  => redistribute:
       * np  node
       *  0  1 3
       *  1  4 9
       *  2  10 12
       */
      double      lb_factor, lb_factor_global;
      
      lb_factor = nC / (double)(nC_global/np);
      SCHURCHEB_MPI_CALL(SchurchebMpiAllreduce( &lb_factor, &lb_factor_global, 1, MPI_MIN,comm));
      
      if(lb_factor_global > SCHURCHEB_global::_metis_loading_balance_tol)
      {
         /* fits the loading balance, no need to re-assign */
         
         /* partition */
         vector_SCHURCHEB_long   marker, marker2;
         vector_seq_float        tpwgts, ubvec;
         
         /* setup helper arrays and parameters */
         map.Setup(nrow);
         marker.Setup(num_dom);
         marker.Fill(-1);
         tpwgts.Setup(num_dom);
         tpwgts.Fill(1.0/num_dom);
         ubvec.Setup(1);
         ubvec.Fill(1.05);
         
         SCHURCHEB_long option[40]  = {0};
         wtflag                     = 0;      //  2: Weights on the vertices only (adjwgt is NULL).
         numflag                    = 0;      //  C-style
         ncon                       = 1;      //  no weight in use
         
         ParMETIS_V3_PartKway(vtxdist.GetData(), xadj.GetData(), adjncy.GetData(), NULL, NULL, &wtflag, &numflag,
                                    &ncon, &num_dom, tpwgts.GetData(), ubvec.GetData(), &option[0], &edgecut, map.GetData(), &comm);
         
         for(refs = 0 ; refs < SCHURCHEB_global::_metis_refine ; refs++)
         {
            ParMETIS_V3_RefineKway(vtxdist.GetData(), xadj.GetData(), adjncy.GetData(), NULL, NULL, &wtflag, &numflag,
                                    &ncon, &num_dom, tpwgts.GetData(), ubvec.GetData(), &option[0], &edgecut, map.GetData(), &comm);
         }
         
         /* mark local domains */
         for (i = 0; i < nrow; i++)
         {
            marker[map[i]] = 1;
         }
         
         /* check for empty domain */
         SCHURCHEB_MPI_CALL( SchurchebMpiAllreduceInplace( marker.GetData(), num_dom, MPI_MAX, comm) );

         num_dom2 = 0;
         for (i = 0; i < num_dom; i++)
         {
            if(marker[i] > 0)
            {
               marker[i] = num_dom2++;
            }
         }
         
         /* now swap the marker number for balence size
          * 0 1 2 3 4 5 ... k*np-1 into
          * 0 np 2np ... 1 np+1 2np+1
          */
         idx = 0;
         idx2 = 0;
         idx3 = 0;
         marker2.Setup(num_dom2);
         while(idx < num_dom2)
         {
            marker2[idx2] = idx;
            idx ++;
            idx2 += np;
            if(idx2 >= num_dom2)
            {
               idx3++;
               idx2 = idx3;
            }
         }
         
         /* remove empty domains */
         for (i = 0; i < nrow; i++)
         {
            map[i] = marker2[marker[map[i]]];
         }
         
         num_dom = num_dom2;
         
         tpwgts.Clear();
         ubvec.Clear();
         //vwgt.Clear();
         marker.Clear();
         marker2.Clear();
         
         return SCHURCHEB_SUCCESS;

      }
      else
      {
         /* number of local nodes on some processors is too small, redistribute */
         //SCHURCHEB_WARNING("Redistribute.");
         
         int                     pid;
         vector_SCHURCHEB_long   vtxdistc, vtxdisteven;
         vector_SCHURCHEB_long   vtxdist2, xadj2, adjncy2;
         
         /* we redistribute those connected nodes to each processor */
         SCHURCHEB_long    nC1, nC2;
         
         nC1      = nC_global/np;
         nC2      = nC_global%np;
         
         vtxdistc.Setup(np+1);
         vtxdist2.Setup(np+1);
         vtxdisteven.Setup(np+1);
         
         vtxdistc[0] = 0;
         vtxdisteven[0] = 0;
         vtxdisteven[np] = nC_global;
         
         nC_l = nC;
         SCHURCHEB_MPI_CALL( SchurchebMpiAllgather( &nC_l, 1, vtxdistc.GetData()+1, comm) );
         
         for(i = 1 ; i < np ; i ++)
         {
            vtxdistc[i] += vtxdistc[i-1];
            if(i <= nC2)
            {
               vtxdisteven[i] = vtxdisteven[i-1] + nC1 + 1;
            }
            else
            {
               vtxdisteven[i] = vtxdisteven[i-1] + nC1;
            }
         }
         
         vtxdistc[np] = nC_global;
         
         /* now setup vtxdist2 */
         vtxdist2[0] = 0;
         vtxdist2[np] = vtxdist[np];
         
         j = 0;
         j1 = vtxdistc[myid];
         for(i = 0 ; i < np-1 ; i ++)
         {
            /* Example:
             * [0 3 6 9]
             * [0 6 6 9]
             * search for 3, should be on p0
             * search for 6, both p0, p1, and p2 works
             */
            
            if( vtxdistc.BinarySearch(vtxdisteven[i+1], pid, true) < 0)
            {
               /* In this case, we haven't found it, belongs to the previous MPI rank */
               pid--;
            }
            
            /* note that vtxdisteven[i+1] is nonzero */
            
            /* the last element of vtxdistc[i+1] belongs to processor ps */
            if(myid == pid)
            {
               /* search for it */
               while(j1 < vtxdisteven[i+1])
               {
                  j1++;
                  while(isolate[j])
                  {
                     j++;
                  }
                  j++;
               }
               vtxdist2[i+1] = vtxdist[myid] + j;
            }
            SCHURCHEB_MPI_CALL( SchurchebMpiBcast( vtxdist2.GetData()+i+1, 1, pid, comm) );
         }
         
         /* now get the amount we need to send to other processor 
          * search in the array, if not found, 
          */
         int                        ps, pe, nsend, nrecv, nsendrecv, nadj2, toid;
         vector_int                 sends, recvs, send_to_v, recv_from_v, send_size_v, recv_size_v, send_size2_v, recv_size2_v;
         std::vector<vector_int >   send_count_v2,recv_count_v2;   
         std::vector<MPI_Request >  request_v;
         
         if( vtxdist2.BinarySearch(vtxdist[myid], ps, true) < 0)
         {
            /* In this case, we haven't found it, belongs to the previous MPI rank */
            ps--;
         }
         
         if(ps == np) 
         {
            ps--;
         }
         
         if( vtxdist2.BinarySearch(vtxdist[myid+1], pe, true) < 0)
         {
            /* In this case, we haven't found it, belongs to the previous MPI rank */
            pe--;
         }
         
         if(pe == np) 
         {
            pe--;
         }
         
         sends.Setup(np, true);
         recvs.Setup(np, true);
         
         if(nrow > 0)
         {
            for(i = ps ; i <= pe ; i ++)
            {
               sends[i] = 1;
            }
         }
         
         SCHURCHEB_MPI_CALL( MPI_Alltoall( sends.GetData(), 1, MPI_INT, recvs.GetData(), 1, MPI_INT, comm) );
         
         nsend = 0;
         nrecv = 0;
         for(i = 0 ; i < np ; i ++)
         {
            if(sends[i] > 0)
            {
               send_to_v.PushBack(i);
               nsend++;
            }
            if(recvs[i] > 0)
            {
               recv_from_v.PushBack(i);
               nrecv++;
            }
         }
         
         nsendrecv = nsend + nrecv;
         send_size_v.Setup(nsend);
         recv_size_v.Setup(nrecv);
         request_v.resize(nsendrecv);
         
         send_count_v2.resize(nsend);
         
         j = 0;
         j1 = vtxdist[myid];
         for(i = 0 ; i < nsend-1 ; i ++)
         {
            /* get the amound of data we need to send */
            toid = send_to_v[i];
            j2 = vtxdist2[toid+1];
            send_size_v[i] = j2 - j1;
            send_count_v2[i].Setup(send_size_v[i]);
            for(k = 0 ; k < send_size_v[i] ; k ++)
            {
               send_count_v2[i][k] = xadj[j+1] - xadj[j];
               j++;
            }
            j1 = j2;
         }
         if(nsend > 0)
         {
            i = nsend-1;
            j2 = vtxdist[myid+1];
            send_size_v[i] = j2 - j1;
            send_count_v2[i].Setup(send_size_v[i]);
            for(k = 0 ; k < send_size_v[i] ; k ++)
            {
               send_count_v2[i][k] = xadj[j+1] - xadj[j];
               j++;
            }
         }
         
         j = 0;
         for(i = 0 ; i < nsend ; i ++)
         {
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( send_size_v.GetData()+i, 1, send_to_v[i], send_to_v[i], comm, &(request_v[j++]) ) );
         }
         
         for(i = 0 ; i < nrecv ; i ++)
         {
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( recv_size_v.GetData()+i, 1, recv_from_v[i], myid, comm, &(request_v[j++]) ) );
         }
         
         SCHURCHEB_MPI_CALL( MPI_Waitall( nsendrecv, request_v.data(), MPI_STATUSES_IGNORE) );
         
         /* prepare the send count */
         
         recv_count_v2.resize(nrecv);
         
         j = 0;
         for(i = 0 ; i < nsend ; i ++)
         {
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( send_count_v2[i].GetData(), send_size_v[i], send_to_v[i], send_to_v[i], comm, &(request_v[j++]) ) );
         }
         
         for(i = 0 ; i < nrecv ; i ++)
         {
            recv_count_v2[i].Setup(recv_size_v[i]);
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( recv_count_v2[i].GetData(), recv_size_v[i], recv_from_v[i], myid, comm, &(request_v[j++]) ) );
         }
         
         SCHURCHEB_MPI_CALL( MPI_Waitall( nsendrecv, request_v.data(), MPI_STATUSES_IGNORE) );
         
         /* prepare to send the adjncy */
         send_size2_v.Setup(nsend, true);
         recv_size2_v.Setup(nrecv, true);
         
         xadj2.Setup(vtxdist2[myid+1]-vtxdist2[myid]+1);
         xadj2[0] = 0;
         
         nadj2 = 0;
         for(i = 0 ; i < nsend ; i ++)
         {
            for( j = 0; j < send_size_v[i] ; j ++)
            {
               send_size2_v[i] += send_count_v2[i][j];
            }
            //nadj2 += send_size2_v[i];
         }
         
         k = 0;
         for(i = 0 ; i < nrecv ; i ++)
         {
            for( j = 0; j < recv_size_v[i] ; j ++)
            {
               recv_size2_v[i] += recv_count_v2[i][j];
               xadj2[k+1] = xadj2[k] + recv_count_v2[i][j];
               k++;
            }
            nadj2 += recv_size2_v[i];
         }
         
         adjncy2.Setup(nadj2);
         
         j = 0;
         j1 = 0;
         for(i = 0 ; i < nsend ; i ++)
         {
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( adjncy.GetData()+j1, send_size2_v[i], send_to_v[i], send_to_v[i], comm, &(request_v[j++]) ) );
            j1 += send_size2_v[i];
         }
         
         j1 = 0;
         for(i = 0 ; i < nrecv ; i ++)
         {
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( adjncy2.GetData()+j1, recv_size2_v[i], recv_from_v[i], myid, comm, &(request_v[j++]) ) );
            j1 += recv_size2_v[i];
         }
         
         //SCHURCHEB_MPI_CALL( MPI_Waitany( sendrecv, this->_comm_helper._requests_v.data(), &reqidx, MPI_STATUSES_IGNORE) );
         SCHURCHEB_MPI_CALL( MPI_Waitall( nsendrecv, request_v.data(), MPI_STATUSES_IGNORE) );
         
         /* partition of the redistributed graph */
         vector_SCHURCHEB_long   marker, marker2, map2;
         vector_seq_float        tpwgts, ubvec;
         
         /* setup helper arrays and parameters */
         map.Setup(nrow);
         map2.Setup(vtxdist2[myid+1]-vtxdist2[myid]);
         marker.Setup(num_dom);
         marker.Fill(-1);
         tpwgts.Setup(num_dom);
         tpwgts.Fill(1.0/num_dom);
         ubvec.Setup(1);
         ubvec.Fill(1.05);
         
         SCHURCHEB_long option[40]  = {0};
         wtflag                     = 0;      //  2: Weights on the vertices only (adjwgt is NULL).
         numflag                    = 0;      //  C-style
         ncon                       = 1;      //  no weight in use
         
         ParMETIS_V3_PartKway(vtxdist2.GetData(), xadj2.GetData(), adjncy2.GetData(), NULL, NULL, &wtflag, &numflag,
                                    &ncon, &num_dom, tpwgts.GetData(), ubvec.GetData(), &option[0], &edgecut, map2.GetData(), &comm);
         
         for(refs = 0 ; refs < SCHURCHEB_global::_metis_refine ; refs++)
         {
            ParMETIS_V3_RefineKway(vtxdist2.GetData(), xadj2.GetData(), adjncy2.GetData(), NULL, NULL, &wtflag, &numflag,
                                    &ncon, &num_dom, tpwgts.GetData(), ubvec.GetData(), &option[0], &edgecut, map2.GetData(), &comm);
         }
         
         /* send the map data back */
         
         j = 0;
         j1 = 0;
         for(i = 0 ; i < nrecv ; i ++)
         {
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( map2.GetData()+j1, recv_size_v[i], recv_from_v[i], myid, comm, &(request_v[j++]) ) );
            j1 += recv_size_v[i];
         }
         
         j1 = 0;
         for(i = 0 ; i < nsend ; i ++)
         {
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( map.GetData()+j1, send_size_v[i], send_to_v[i], send_to_v[i], comm, &(request_v[j++]) ) );
            j1 += send_size_v[i];
         }
         
         SCHURCHEB_MPI_CALL( MPI_Waitall( nsendrecv, request_v.data(), MPI_STATUSES_IGNORE) );
         
         /* mark local domains */
         for (i = 0; i < nrow; i++)
         {
            marker[map[i]] = 1;
         }
         
         /* check for empty domain */
         SchurchebMpiAllreduceInplace( marker.GetData(), num_dom, MPI_MAX, comm);

         num_dom2 = 0;
         for (i = 0; i < num_dom; i++)
         {
            if(marker[i] > 0)
            {
               marker[i] = num_dom2++;
            }
         }
         
         /* now swap the marker number for balence size
          * 0 1 2 3 4 5 ... k*np-1 into
          * 0 np 2np ... 1 np+1 2np+1
          */
         idx = 0;
         idx2 = 0;
         idx3 = 0;
         marker2.Setup(num_dom2);
         while(idx < num_dom2)
         {
            marker2[idx2] = idx;
            idx ++;
            idx2 += np;
            if(idx2 >= num_dom2)
            {
               idx3++;
               idx2 = idx3;
            }
         }
         
         /* remove empty domains */
         for (i = 0; i < nrow; i++)
         {
            map[i] = marker2[marker[map[i]]];
         }
         
         num_dom = num_dom2;
         
         sends.Clear(); 
         recvs.Clear();
         send_to_v.Clear();
         recv_from_v.Clear();
         send_size_v.Clear();
         recv_size_v.Clear();
         send_size2_v.Clear();
         recv_size2_v.Clear();
         
         for(i = 0 ; i < nsend ; i ++)
         {
            send_count_v2[i].Clear();
         }
         
         for(i = 0 ; i < nrecv ; i ++)
         {
            recv_count_v2[i].Clear();
         }
         
         std::vector<vector_int >().swap(send_count_v2);
         std::vector<vector_int >().swap(recv_count_v2);
         
         std::vector<MPI_Request >().swap(request_v);
         
         tpwgts.Clear();
         ubvec.Clear();
         //vwgt.Clear();
         marker.Clear();
         marker2.Clear();
         
         return SCHURCHEB_SUCCESS;
         
      }
   }
   
   int ParmetisNodeND(vector_SCHURCHEB_long &vtxdist, vector_SCHURCHEB_long &xadj, vector_SCHURCHEB_long &adjncy, SCHURCHEB_long &num_dom, vector_SCHURCHEB_long &map, parallel_log &parlog)
   {
      /* Declare variables */
      vector_SCHURCHEB_long   order, sizes;
      
      SCHURCHEB_long          i, nrow;
      SCHURCHEB_long          numflag;
      int                     idx;
      
      /* MPI */
      MPI_Comm    comm;
      int         np, myid;
      
      parlog.GetMpiInfo(np, myid, comm);
      
      nrow = vtxdist[myid + 1] - vtxdist[myid];
      
      /* setup helper arrays and parameters */
      map.Setup(nrow);
      order.Setup(nrow);
      sizes.Setup(2*np+1, true);
      
      SCHURCHEB_long option[40]  = {0};
      numflag                    = 0;      //  C-style
      
      /* parMetis ND partition into log(p) comp
       * order[i]: now global number of i-th local vertex.
       * sizez[i]: each of the sizes arrays are identical.
       *           
       */
      ParMETIS_V3_NodeND(vtxdist.GetData(), xadj.GetData(), adjncy.GetData(), &numflag, &option[0], order.GetData(), sizes.GetData()+1, &comm);
      
      //SCHURCHEB_GLOBAL_SEQUENTIAL_RUN({order.Plot(0,0,6);});
      
      /* now setup the map array */
      sizes[0] = 0;
      
      num_dom = 2*np;
      
      for(i = 0; i < num_dom; i++)
      {
         sizes[i+1] += sizes[i];
      }
      
      /* ignore empty ones at the end */
      while(num_dom > 0 && (sizes[num_dom] == sizes[num_dom-1]))
      {
         num_dom--;
      }
      
      /* resize the size array */
      sizes.Resize( num_dom+1, true, false);
      
      /* set the map array */
      for(i = 0 ; i < nrow ; i ++)
      {
         /* if we didn't find it, the real domain number is the smaller one
          * Example: [0, 2, 4], if we search for 1, the result is 1, the domain number is 0
          */
         if(sizes.BinarySearch(order[i], idx, true) < 0 )
         {
            idx--;
         }
         map[i] = idx;
      }
      
      sizes.Clear();
      order.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T>
   int CsrSubMatrixAmdHost(CsrMatrixClass<T> &A, vector_int &rowscols, vector_int &perm)
   {
      int err;
      
      CsrMatrixClass<T> B;
      
      /* get that sub matrix */
      err = A.SubMatrix(rowscols, rowscols, kMemoryHost, B); SCHURCHEB_CHKERR(err);
      
      /* apply RCM */
      err = CsrMatrixAmdHost(B, perm);
      
      return err;
      
   }
   template int CsrSubMatrixAmdHost(CsrMatrixClass<float> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixAmdHost(CsrMatrixClass<double> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixAmdHost(CsrMatrixClass<complexs> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixAmdHost(CsrMatrixClass<complexd> &A, vector_int &rowscols, vector_int &perm);
   
   template <typename T>
   int CsrMatrixAmdHost(CsrMatrixClass<T> &A, vector_int &perm)
   {
      /*---------------------------------------------------------------------------
      * AMD ordering of a sparse symmetric matrix A.
      * 
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * A         = CSR Matrix object.
      *
      * on return:
      * ==========
      * err       = return value.
      *             err  == 0   --> successful return.
      *             err  != 0   --> Error occurs.
      * perm      = Integer vector. Permutation vector.
      *----------------------------------------------------------------------------
      * Note:
      * C-style 0-based index.
      * 
      *--------------------------------------------------------------------------*/
      SCHURCHEB_ERROR("Csr matrix AMD ordering currenlty unsupported.");
      return SCHURCHEB_ERROR_INVALED_OPTION;
      /*
      if( A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Csr matrix RCM ordering only works on the host memory.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      int                  err = 0;
      int                  nA;
      CsrMatrixClass<T>    A2;
      CsrMatrixClass<T>    AT;
      
      nA = A.GetNumRowsLocal();
      
      perm.Clear();
      perm.Setup(nA);
      
      SCHURCHEB_CHKERR(nA != A.GetNumColsLocal() || nA < 0);
      
      // directly return if A is empty
      if(nA==0)
      {
         return 0;
      }
      
      //-----------------------------------
      //-----Build Graph Data Structure----
      //-----------------------------------
      
      CsrMatrixTransposeHost(A, AT);
      CsrMatrixAddHost(A, AT, A2);
      
      AT.Clear();
      if(A2.GetNumNonzeros() < nA)
      {
         SCHURCHEB_ERROR("Zero diagonal in RCM ordering.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      A2.SortRow();
      
      err = amd_order( nA, A2.GetI(), A2.GetJ(), perm.GetData(), NULL, NULL); SCHURCHEB_CHKERR(err);
      
      A2.Clear();
       
      return err;
      */
   }
   template int CsrMatrixAmdHost(CsrMatrixClass<float> &A, vector_int &perm);
   template int CsrMatrixAmdHost(CsrMatrixClass<double> &A, vector_int &perm);
   template int CsrMatrixAmdHost(CsrMatrixClass<complexs> &A, vector_int &perm);
   template int CsrMatrixAmdHost(CsrMatrixClass<complexd> &A, vector_int &perm);
   
   template <typename T>
   int CsrSubMatrixNdHost(CsrMatrixClass<T> &A, vector_int &rowscols, vector_int &perm)
   {
      int err;
      
      CsrMatrixClass<T> B;
      
      /* get that sub matrix */
      err = A.SubMatrix(rowscols, rowscols, kMemoryHost, B); SCHURCHEB_CHKERR(err);
      
      /* apply RCM */
      err = CsrMatrixNdHost(B, perm);
      
      return err;
      
   }
   template int CsrSubMatrixNdHost(CsrMatrixClass<float> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixNdHost(CsrMatrixClass<double> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixNdHost(CsrMatrixClass<complexs> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixNdHost(CsrMatrixClass<complexd> &A, vector_int &rowscols, vector_int &perm);
   
   template <typename T>
   int CsrMatrixNdHost(CsrMatrixClass<T> &A, vector_int &perm)
   {
      /*---------------------------------------------------------------------------
      * Nd ordering of a sparse symmetric matrix A.
      * 
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * A         = CSR Matrix object.
      *
      * on return:
      * ==========
      * err       = return value.
      *             err  == 0   --> successful return.
      *             err  != 0   --> Error occurs.
      * perm      = Integer vector. Permutation vector.
      *----------------------------------------------------------------------------
      * Note:
      * C-style 0-based index.
      * 
      *--------------------------------------------------------------------------*/
      
      /*
      int      nrows, ncols, nnz, col, i, i1, i2, j, jj, err = 0;
      int      *A_i, *A_j;
      T        *A_data;
      
      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();
      
      if (nrows != ncols) 
      {
         SCHURCHEB_ERROR("Csr matrix ND ordering only works for square matrix.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      if( A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Csr matrix ND ordering only works on the host memory.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      // # of nnz
      nnz = A.GetNumNonzeros();

      // sort rows of A
      A.SortRow();
      
      A_i = A.GetI();
      A_j = A.GetJ();
      A_data = A.GetData();
      
      // Prepare data structures used by METIS
      IntVectorClass<SCHURCHEB_long> xadj;
      IntVectorClass<SCHURCHEB_long> adjncy;
      IntVectorClass<SCHURCHEB_long> vwgt;
      IntVectorClass<SCHURCHEB_long> perml;
      IntVectorClass<SCHURCHEB_long> iperml;
      
      adjncy.Setup(nnz);
      xadj.Setup(nrows+1);
      vwgt.Setup(nrows, true);

      // Fill the vectors with the appropriate values
      
      perm.Setup(nrows);
      perml.Setup(nrows);
      iperml.Setup(nrows);
      
      // Costruct a CSR-like representation of A as required by METIS. Extract and keep the diagonal entries separately. 
      xadj[0] = 0;
      jj = 0;
      for ( i = 0 ; i < nrows; i++) 
      {
         i1 = A_i[i];
         i2 = A_i[i+1];
         for ( j = i1; j < i2; j++) 
         {
            col = A_j[j];
            if (col != i) 
            {
               adjncy[jj] = col;
            } 
            else 
            {
               vwgt[i] = (SCHURCHEB_long) SchurchebAbs(A_data[j]);
            }
         }
         if(vwgt[i] == 0)
         {
            // in this case, we don't have a diagonal entry, still give it a default weight
            vwgt[i] = 1.0;
         }
         xadj[i+1] = jj;
      }
      
      // METIS parameters, note that  is used
      
      SCHURCHEB_long lnrows = (SCHURCHEB_long)nrows;
      
      // call METIS
      
      METIS_NodeND( &lnrows, xadj.GetData(), adjncy.GetData(), vwgt.GetData(), NULL, perml.GetData(), iperml.GetData());
      
      for ( i = 0; i < nrows; i++) 
      {
         perm[i] = (int)perml[i];
      }
      
      xadj.Clear();
      adjncy.Clear();
      vwgt.Clear();
      perml.Clear();
      iperml.Clear();
      
      return err;
      */
      
      
      SCHURCHEB_ERROR("Csr matrix ND ordering currenlty unsupported.");
      
      return SCHURCHEB_ERROR_INVALED_OPTION;
      
   }
   template int CsrMatrixNdHost(CsrMatrixClass<float> &A, vector_int &perm);
   template int CsrMatrixNdHost(CsrMatrixClass<double> &A, vector_int &perm);
   template int CsrMatrixNdHost(CsrMatrixClass<complexs> &A, vector_int &perm);
   template int CsrMatrixNdHost(CsrMatrixClass<complexd> &A, vector_int &perm);
   
   template <typename T>
   int CsrSubMatrixRcmHost(CsrMatrixClass<T> &A, vector_int &rowscols, vector_int &perm)
   {
      int err;
      
      CsrMatrixClass<T> B;
      
      /* get that sub matrix */
      err = A.SubMatrix(rowscols, rowscols, kMemoryHost, B); SCHURCHEB_CHKERR(err);
      
      /* apply RCM */
      err = CsrMatrixRcmHost(B, perm);
      
      return err;
      
   }
   template int CsrSubMatrixRcmHost(CsrMatrixClass<float> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixRcmHost(CsrMatrixClass<double> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixRcmHost(CsrMatrixClass<complexs> &A, vector_int &rowscols, vector_int &perm);
   template int CsrSubMatrixRcmHost(CsrMatrixClass<complexd> &A, vector_int &rowscols, vector_int &perm);
   
   template <typename T>
   int CsrMatrixRcmHost(CsrMatrixClass<T> &A, vector_int &perm)
   {
      /*---------------------------------------------------------------------------
      * RCM ordering of a sparse symmetric matrix A.
      * 
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * A         = CSR Matrix object.
      *
      * on return:
      * ==========
      * err       = return value.
      *             err  == 0   --> successful return.
      *             err  != 0   --> Error occurs.
      * perm      = Integer vector. Permutation vector.
      *----------------------------------------------------------------------------
      * Note:
      * C-style 0-based index.
      * 
      *--------------------------------------------------------------------------*/
      
      if( A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Csr matrix RCM ordering only works on the host memory.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      int                  i, j, k1, k2;
      int                  nA, nodei, current_num;
      vector_int           marker;
      CsrMatrixClass<T>    G;
      CsrMatrixClass<T>    A2;
      CsrMatrixClass<T>    AT;
      
      nA = A.GetNumRowsLocal();
      
      perm.Clear();
      perm.Setup(nA);
      
      SCHURCHEB_CHKERR(nA != A.GetNumColsLocal() || nA < 0);
      
      /* skip reorder if A is empty */
      if(nA==0)
      {
         return 0;
      }
      
      //-----------------------------------
      //-----Build Graph Data Structure----
      //-----------------------------------
      
      CsrMatrixTransposeHost(A, AT);
      CsrMatrixAddHost(A, AT, A2);
      
      AT.Clear();
      if(A2.GetNumNonzeros() < nA)
      {
         SCHURCHEB_ERROR("Zero diagonal in RCM ordering.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      G.Setup(nA, nA, A2.GetNumNonzeros()-nA, false, false);
      
      int   *G_i = G.GetI();
      int   *G_j = G.GetJ();
      int   *A2_i = A2.GetI();
      int   *A2_j = A2.GetJ();
      
      G_i[0] = 0;
      for(i = 0 ; i < nA ; i ++)
      {
         G_i[i+1] = G_i[i];
         k1 = A2_i[i], k2 = A2_i[i+1];
         for(j = k1 ; j < k2 ; j ++)
         {
            if(A2_j[j] != i)
            {
               G_j[G_i[i+1]++] = A2_j[j];
            }
         }
      }
      
      A2.Clear();
   
      //------------------------
      //-----Find RCM ORDER-----
      //------------------------
      
      //create working array
      marker.Setup(nA);
      marker.Fill(-1);
      current_num = 0;
      while( current_num < nA )
      {
         //find unvised node with minimum degree
         CsrMatrixRcmRootHost(G, marker, nodei);
         //find pseudo-peripheral node
         CsrMatrixRcmPerphnHost(G, nodei, marker);
         //number this connect component
         CsrMatrixRcmNumberingHost(G, nodei, marker, perm, current_num);
      }
      
      //De-allocate
      marker.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixRcmHost(CsrMatrixClass<float> &A, vector_int &perm);
   template int CsrMatrixRcmHost(CsrMatrixClass<double> &A, vector_int &perm);
   template int CsrMatrixRcmHost(CsrMatrixClass<complexs> &A, vector_int &perm);
   template int CsrMatrixRcmHost(CsrMatrixClass<complexd> &A, vector_int &perm);

   template <typename T>
   int CsrMatrixRcmRootHost(CsrMatrixClass<T> &G, vector_int &marker, int &root)
   {
      int                              i, j, k, n;
      int                              nlev, degree, min_degree, lev_degree;
      std::vector<std::vector<int> >   level;
      
      n           = G.GetNumRowsLocal();
      min_degree  = n + 1;
      root        = 0;
      
      for(i = 0 ; i < n ; i ++)
      {
         if(marker[i] < 0)
         {
            /* find the connect component starting from here */
            CsrMatrixRcmBfsHost(G, i, marker, level);
            break;
         }
      }
      
      int *G_i = G.GetI();
      
      nlev = level.size();
      for(i = 0 ; i < nlev ; i ++)
      {
         lev_degree = level[i].size();
         for(j = 0 ; j < lev_degree ; j ++)
         {
            k = level[i][j];
            degree = G_i[k+1]-G_i[k];
            if( degree < min_degree)
            {
               root = k;
               min_degree = degree;
            }
         }
      }
      
      CsrMatrixRcmClearLevelHost(level);
      
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixRcmRootHost(CsrMatrixClass<float> &G, vector_int &marker, int &root);
   template int CsrMatrixRcmRootHost(CsrMatrixClass<double> &G, vector_int &marker, int &root);
   template int CsrMatrixRcmRootHost(CsrMatrixClass<complexs> &G, vector_int &marker, int &root);
   template int CsrMatrixRcmRootHost(CsrMatrixClass<complexd> &G, vector_int &marker, int &root);

   template <typename T>
   int CsrMatrixRcmNumberingHost(CsrMatrixClass<T> &G, int root, vector_int &marker, vector_int &perm, int &current_num)
   {
      /*---------------------------------------------------------------------------
      * RCM numbering for a connect component start form root with BFS.
      * 
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * G            = CSR Pattern object. Adjencency graph.
      * root         = Integer. The node we start seartch from.
      * marker       = Integer vector. Helper array, length equal to # of nodes. 
      *                Should be allocated already, value will be used.
      *                Untouched nodes should be marked with negative value.
      * perm         = Integer vector. Permutation array, length equal to # of nodes.
      *                Should be allocated already. Value not used and will be overwritten.
      * current_num  = Integer. # of root in the permutation array. Since we might have built
      *                the RCM for some connect components already.
      *
      * on return:
      * ==========
      * err          = return value.
      *                err  == 0   --> successful return.
      *                err  != 0   --> Error occurs.
      * marker       = Integer vector. If node i belongs to the current connect-component, 
      *                marker[i] will be set into some positive value.
      * perm         = Integer vector. Permutation vector.
      * current_num  = Integer. # of next root in the permutation array.
      * 
      *--------------------------------------------------------------------------*/
      
      int      i;
      int      j, j1, j2;
      int      nodei;
      int      nodej;
      int      node_start;
      int      node_end;
      
      int      comp_start        = current_num;
      int      lev_start         = current_num;
      marker[root]               = 0;
      perm[current_num++]        = root;
      int      lev_end           = current_num;
      int      *G_i              = G.GetI();
      int      *G_j              = G.GetJ();
      
      //explore nbhds of all nodes in current level
      while(lev_end > lev_start)
      {
         //loop through all nodes in current level
         for(i = lev_start ;  i < lev_end ; i ++)
         {
            //node to be explored
            nodei = perm[i];
            //explore nbhds of this node
            node_start = current_num;
            j1 = G_i[nodei];
            j2 = G_i[nodei+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               nodej = G_j[j];
               if(marker[nodej]<0)
               {
                  //this is an unmarked node, save its degree in marker
                  marker[nodej] = G_i[nodej+1] - G_i[nodej];
                  perm[current_num++] = nodej;
               }
            }
            node_end = current_num;
            if(node_end-1 > node_start)
            {
               /* sort based on degree when we have at least 2 nodes */
               std::sort(perm.GetData()+node_end, perm.GetData()+node_end);      
            }
         }
         lev_start = lev_end;
         lev_end = current_num;
      }
      
      //reverse
      CsrMatrixRcmReverseHost(perm, comp_start, lev_end-1);
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixRcmNumberingHost(CsrMatrixClass<float> &G, int root, vector_int &marker, vector_int &perm, int &current_num);
   template int CsrMatrixRcmNumberingHost(CsrMatrixClass<double> &G, int root, vector_int &marker, vector_int &perm, int &current_num);
   template int CsrMatrixRcmNumberingHost(CsrMatrixClass<complexs> &G, int root, vector_int &marker, vector_int &perm, int &current_num);
   template int CsrMatrixRcmNumberingHost(CsrMatrixClass<complexd> &G, int root, vector_int &marker, vector_int &perm, int &current_num);

   template <typename T>
   int CsrMatrixRcmPerphnHost(CsrMatrixClass<T> &G, int &root, vector_int &marker)
   {
      /*---------------------------------------------------------------------------
      * RCM numbering for a connect component start form root with BFS.
      * 
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * G            = CSR Pattern object. Adjencency graph.
      * root         = Integer. The node we start seartch from.
      * marker       = Integer vector. Helper array, length equal to # of nodes. 
      *                Should be allocated already, value will be used.
      *                Untouched nodes should be marked with negative value.
      *
      * on return:
      * ==========
      * err          = return value.
      *                err  == 0   --> successful return.
      *                err  != 0   --> Error occurs.
      * root         = Integer vector. An end of the pseudo-peripheral.
      * 
      *--------------------------------------------------------------------------*/
      
      int                              i;
      int                              last_level_size;
      int                              min_degree;
      int                              lev_degree;
      std::vector<std::vector<int> >   level;
      //build level structure for root
      int                              nG       = G.GetNumRowsLocal();
      CsrMatrixRcmBfsHost(G, root, marker, level);
      int                              nlev     = level.size();
      int                              newnlev  = nlev + 1;
      int                              *G_i = G.GetI();
      
      while(nlev < newnlev)
      {
         nlev = level.size();
         std::vector<int> &last_level = level[nlev-1];
         last_level_size = last_level.size();
         min_degree = nG;
         for(i = 0 ; i < last_level_size ; i ++)
         {
            //we select the last level, pick min-degree node
            lev_degree = G_i[last_level[i]+1] - G_i[last_level[i]];
            if(min_degree > lev_degree)
            {
               min_degree = lev_degree;
               root = last_level[i];
            }
         }
         CsrMatrixRcmClearLevelHost(level);
         CsrMatrixRcmBfsHost(G, root, marker, level);
         newnlev = level.size();
      }
      CsrMatrixRcmClearLevelHost(level);
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixRcmPerphnHost(CsrMatrixClass<float> &G, int &root, vector_int &marker);
   template int CsrMatrixRcmPerphnHost(CsrMatrixClass<double> &G, int &root, vector_int &marker);
   template int CsrMatrixRcmPerphnHost(CsrMatrixClass<complexs> &G, int &root, vector_int &marker);
   template int CsrMatrixRcmPerphnHost(CsrMatrixClass<complexd> &G, int &root, vector_int &marker);

   template <typename T>
   int CsrMatrixRcmBfsHost(CsrMatrixClass<T> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level)
   {
      /*---------------------------------------------------------------------------
      * Apply the BFS start from root.
      * 
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * G            = CSR Pattern object. Adjencency graph.
      * root         = Integer. The node we start seartch from.
      * marker       = Integer vector. Helper array, length equal to # of nodes. 
      *                Untouched nodes should have negative value. Only search them.
      * level        = Vector of integer vector. Level structure, nodes at level[i]
      *                are on the ith level in the BFS.
      *
      * on return:
      * ==========
      * err          = return value.
      *                err  == 0   --> successful return.
      *                err  != 0   --> Error occurs.
      * root         = Integer vector. An end of the pseudo-peripheral.
      * 
      *--------------------------------------------------------------------------*/
      
      int     i;
      int     j, j1, j2;
      int     nodei;
      int     nodej;
      int     lev_degree;
      
      //current version use BFS to build this level structure
      //we assume to be given a empty vector level
      level.push_back(std::vector<int>());
      level[0].push_back(root);
      marker[root] = 0;
      int nlev = 0;
      int *G_i = G.GetI();
      int *G_j = G.GetJ();
      
      //explore nbhds of all nodes in current level
      while(level[nlev].size() > 0)
      {
         //create next level
         level.push_back(std::vector<int>());
         std::vector<int> &last_lev = level[nlev];
         nlev++;
         std::vector<int> &next_lev = level[nlev];
         int last_lev_size = last_lev.size();
         for(i = 0 ;  i < last_lev_size ; i ++)
         {
            //node to be explored
            nodei = last_lev[i];
            //explore nbhds of nodei
            j1 = G_i[nodei];
            j2 = G_i[nodei+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               nodej = G_j[j];
               if(marker[nodej]<0)
               {
                  //an unmarked node
                  marker[nodej] = 0;
                  next_lev.push_back(nodej);
               }
            }
         }
      }
      //the last level is empty, just pop it
      level.pop_back();
      
      //new we have set the level structure, reset marker array
      for(i = 0 ; i < nlev ; i ++)
      {
         lev_degree = level[i].size();
         for(j = 0 ; j < lev_degree ; j ++)
         {
            marker[level[i][j]] = -1;
         }
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int CsrMatrixRcmBfsHost(CsrMatrixClass<float> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level);
   template int CsrMatrixRcmBfsHost(CsrMatrixClass<double> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level);
   template int CsrMatrixRcmBfsHost(CsrMatrixClass<complexs> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level);
   template int CsrMatrixRcmBfsHost(CsrMatrixClass<complexd> &G, int root, vector_int &marker, std::vector<std::vector<int> > &level);

   int CsrMatrixRcmClearLevelHost(std::vector<std::vector<int> > &level)
   {
      /*---------------------------------------------------------------------------
      * Clear level struct (vector of integer vector)
      * 
      *----------------------------------------------------------------------------
      * Parameters
      *----------------------------------------------------------------------------
      * on entry:
      * =========
      * level        = Vector of integer vector. Level structure.
      *
      * on return:
      * ==========
      * err          = return value.
      *                err  == 0   --> successful return.
      *                err  != 0   --> Error occurs.
      * 
      *--------------------------------------------------------------------------*/
      
      int     i;
      int     nlev = level.size();
      for(i = 0 ; i < nlev ; i ++)
      {
         std::vector<int>().swap(level[i]);
      }
      std::vector<std::vector<int> >().swap(level);
      
      return SCHURCHEB_SUCCESS;
   }

   int CsrMatrixRcmSwapHost(vector_int &perm, int a, int b)
   {
      //helper function in sort and reverse. Swap two elements in an array
      
      int      temp;
      temp     = perm[a];
      perm[a]  = perm[b];
      perm[b]  = temp;
      return SCHURCHEB_SUCCESS;
   }

   int CsrMatrixRcmReverseHost(vector_int &perm, int start, int end)
   {
      //helper function. Reverse permutation to change from CM to RCM.
      
      int     i;
      int     j;
      int     mid = (start + end + 1) / 2;
      
      for(i = start, j = end ; i < mid ; i ++, j--)
      {
         CsrMatrixRcmSwapHost(perm, i, j);
      }
      return SCHURCHEB_SUCCESS;
   }
   
   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int SchurchebSubSpaceIteration( MatrixType &A, int k, int its, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, RealDataType tol, int &nmvs)
   {
      
      int                  i, j, ii, n;
      DataType             one, zero;
      
      int                  location = A.GetDataLocation();
      
      RealDataType         orth_tol = SCHURCHEB_global::_orth_tol;
      RealDataType         reorth_tol = SCHURCHEB_global::_reorth_tol;
      RealDataType         t, normv;
      
      if(orth_tol < std::numeric_limits<RealDataType>::epsilon())
      {
         /* the tolerance should not be too small */
         orth_tol = std::numeric_limits<RealDataType>::epsilon();
      }
      
      VectorType           v, w;
      
      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);
      
      n = A.GetNumRowsLocal();
      one = 1.0;
      zero = 0.0;
      
      int np, myid;
      MPI_Comm comm;
      A.GetMpiInfo(np, myid, comm);
      
      DenseMatrixClass<DataType>                         B, Q1, Q, R, R1;
      void                                               *q_data;
      DenseMatrixClass<RealDataType>                     Qr;
      DenseMatrixClass<ComplexValueClass<RealDataType> > Qc;
      B.Setup(n, k, location, false);
      Q.Setup(n, k, location, false);
      
      /* R is on host, store the R of QR, not used */
      R.Setup(k, k, kMemoryHost, false);
      R1.SetupPtr(R, 0, 1, k, k-1);
      
      /* reset the seed */
      SCHURCHEB_global::_mersenne_twister_engine.seed(0);
      
      B.Rand();
      
      nmvs = 0;
      
      for(i = 0 ; i < its ; i ++)
      {
         //DenseMatrixQRDecompositionHost(B, Q);
         /* QR decomposition of B */
         
         /* first normoralize first column in B */
         if( n > 0)
         {
            v.UpdatePtr( &B(0, 0), location );
         }
         
         v.Norm2(normv);
         v.Scale(one/normv);
         
         R.Fill(zero);
         R(0,0) = normv;
         for(ii = 0 ; ii < k-1 ; ii ++)
         {
            if( n > 0)
            {
               v.UpdatePtr( &B(0, ii+1), location );
            }
            if(SCHURCHEB_global::_gram_schmidt == 1)
            {
               SchurchebMgs( v, B, R1, t, ii, orth_tol, reorth_tol);
            }
            else
            {
               SchurchebCgs2( v, B, R1, t, ii, orth_tol);
            }
         }
         
         SCHURCHEB_MEMCPY( Q.GetData(), B.GetData(), n*k, location, location, DataType);
         
         /* B = A * Q */
         for(j = 0 ; j < k ; j ++)
         {
            if( n > 0)
            {
               v.UpdatePtr( &Q(0, j), location );
               w.UpdatePtr( &B(0, j), location );
            }
            A.MatVec('N', one, v, zero, w);
            
            /* one matvec */
            nmvs++;
            
         }
      }
      
      /* H = Q^H * B */
      H.Setup(k, k, location, true);
      H.MatMat( one, Q, 'C', B, 'N', zero);
      
      if(SchurchebIsComplex<DataType>::value)
      {
         DenseMatrixClass<ComplexValueClass<RealDataType> > H1;
         SequentialVectorClass<ComplexValueClass<RealDataType> > w;
         
         if(location != kMemoryDevice)
         {
            if(np > 1)
            {
               SCHURCHEB_MPI_CALL( SchurchebMpiAllreduceInplace( H.GetData(), k*k, MPI_SUM, comm) );
            }
            H1.SetupPtr( (ComplexValueClass<RealDataType>*)(H.GetData()), H.GetNumRowsLocal(), H.GetNumColsLocal(), H.GetLeadingDimension(), location);
            
            /* Schur of H => H1(H) and Q1 */
            
            H1.Schur(Qc, w);
            w.Clear();
            
            q_data = (void*)Qc.GetData();
            
         }
         else
         {
            DenseMatrixClass<DataType> H_host;
            /* copy to host */
            H_host.Setup(k, k);
            SCHURCHEB_MEMCPY( H_host.GetData(), H.GetData(), k*k, kMemoryHost, location, DataType);
            if(np > 1)
            {
               SCHURCHEB_MPI_CALL( SchurchebMpiAllreduceInplace( H_host.GetData(), k*k, MPI_SUM, comm) );
            }
            H1.SetupPtr( (ComplexValueClass<RealDataType>*)(H_host.GetData()), H_host.GetNumRowsLocal(), H_host.GetNumColsLocal(), H_host.GetLeadingDimension(), kMemoryHost);
            
            /* Schur of H => H1(H) and Q1 */
            H1.Schur(Qc, w);
            w.Clear();
            
            /* move back to device */
            SCHURCHEB_MEMCPY( H.GetData(), H_host.GetData(), k*k, location, kMemoryHost, DataType);
            H_host.Clear();
            
            Qc.MoveData(location);
            
            q_data = (void*)Qc.GetData();
            
         }
         
      }
      else
      {
         DenseMatrixClass<RealDataType> H1;
         SequentialVectorClass<RealDataType> wr, wi;
         
         if(location != kMemoryDevice)
         {
            if(np > 1)
            {
               SCHURCHEB_MPI_CALL( SchurchebMpiAllreduceInplace( H.GetData(), k*k, MPI_SUM, comm) );
            }
            H1.SetupPtr( (RealDataType*)(H.GetData()), H.GetNumRowsLocal(), H.GetNumColsLocal(), H.GetLeadingDimension(), location);
            
            /* Schur of H => H1(H) and Q1 */
            
            H1.Schur(Qr, wr, wi);
            wr.Clear();
            wi.Clear();
            
            q_data = (void*)Qr.GetData();
            
         }
         else
         {
            DenseMatrixClass<DataType> H_host;
            /* copy to host */
            H_host.Setup(k, k);
            SCHURCHEB_MEMCPY( H_host.GetData(), H.GetData(), k*k, kMemoryHost, location, DataType);
            if(np > 1)
            {
               SCHURCHEB_MPI_CALL( SchurchebMpiAllreduceInplace( H_host.GetData(), k*k, MPI_SUM, comm) );
            }
            H1.SetupPtr( (RealDataType*)(H_host.GetData()), H_host.GetNumRowsLocal(), H_host.GetNumColsLocal(), H_host.GetLeadingDimension(), kMemoryHost);
            
            /* Schur of H => H1(H) and Q1 */
            H1.Schur(Qr, wr, wi);
            wr.Clear();
            wi.Clear();
            
            /* move back to device */
            SCHURCHEB_MEMCPY( H.GetData(), H_host.GetData(), k*k, location, kMemoryHost, DataType);
            H_host.Clear();
            
            Qr.MoveData(location);
            
            q_data = (void*)Qr.GetData();
            
         }
         
      }
      
      Q1.SetupPtr( (DataType*)(q_data), k, k, k, location);
      
      /* V = Q*Q1; */
      V.MatMat( one, Q, 'N', Q1, 'N', zero);
      
      Q1.Clear();
      Qr.Clear();
      Qc.Clear();
      
      v.Clear();
      w.Clear();
      Q.Clear();
      B.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebSubSpaceIteration<SequentialVectorClass<float> >( arnoldimatrix_seq_float &A, int k, int its, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol, int &nmvs);
   template int SchurchebSubSpaceIteration<SequentialVectorClass<double> >( arnoldimatrix_seq_double &A, int k, int its, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol, int &nmvs);
   template int SchurchebSubSpaceIteration<SequentialVectorClass<complexs> >( arnoldimatrix_seq_complexs &A, int k, int its, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol, int &nmvs);
   template int SchurchebSubSpaceIteration<SequentialVectorClass<complexd> >( arnoldimatrix_seq_complexd &A, int k, int its, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol, int &nmvs);
   template int SchurchebSubSpaceIteration<ParallelVectorClass<float> >( arnoldimatrix_par_float &A, int k, int its, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol, int &nmvs);
   template int SchurchebSubSpaceIteration<ParallelVectorClass<double> >( arnoldimatrix_par_double &A, int k, int its, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol, int &nmvs);
   template int SchurchebSubSpaceIteration<ParallelVectorClass<complexs> >( arnoldimatrix_par_complexs &A, int k, int its, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol, int &nmvs);
   template int SchurchebSubSpaceIteration<ParallelVectorClass<complexd> >( arnoldimatrix_par_complexd &A, int k, int its, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol, int &nmvs);
   template int SchurchebSubSpaceIteration<SequentialVectorClass<float> >( matrix_csr_float &A, int k, int its, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol, int &nmvs);
   template int SchurchebSubSpaceIteration<SequentialVectorClass<double> >( matrix_csr_double &A, int k, int its, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol, int &nmvs);
   template int SchurchebSubSpaceIteration<SequentialVectorClass<complexs> >( matrix_csr_complexs &A, int k, int its, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol, int &nmvs);
   template int SchurchebSubSpaceIteration<SequentialVectorClass<complexd> >( matrix_csr_complexd &A, int k, int its, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol, int &nmvs);
   template int SchurchebSubSpaceIteration<ParallelVectorClass<float> >( matrix_csr_par_float &A, int k, int its, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol, int &nmvs);
   template int SchurchebSubSpaceIteration<ParallelVectorClass<double> >( matrix_csr_par_double &A, int k, int its, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol, int &nmvs);
   template int SchurchebSubSpaceIteration<ParallelVectorClass<complexs> >( matrix_csr_par_complexs &A, int k, int its, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol, int &nmvs);
   template int SchurchebSubSpaceIteration<ParallelVectorClass<complexd> >( matrix_csr_par_complexd &A, int k, int its, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol, int &nmvs);
   
   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int SchurchebArnoldiNoRestart( MatrixType &A, int mstart, int msteps, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, 
                                 RealDataType tol_orth, RealDataType tol_reorth, int &nmvs)
   {
      
      int                  k, n;
      RealDataType         t;
      DataType             one, zero;
      
      VectorType           v, w;
      
      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);
      
      n = A.GetNumRowsLocal();
      one = 1.0;
      zero = 0.0;
      
      if(tol_orth < std::numeric_limits<RealDataType>::epsilon())
      {
         /* the tolerance should not be too small */
         tol_orth = std::numeric_limits<RealDataType>::epsilon();
      }
      
      /*------------------------
       * Start arnoldi loop
       * Compute matvec u = A*v
       * Apply Modified Gram - Schmidt
       *------------------------*/
      
      nmvs = 0;
      
      for (k = mstart; k < msteps; k++) 
      {
         /* set to each column. v is the current row, w is the next row 
          * only set is local size is not 0
          */
         if( n > 0)
         {
            v.UpdatePtr( &V(0, k), V.GetDataLocation() );
            w.UpdatePtr( &V(0, k+1), V.GetDataLocation() );
         }
         
         A.MatVec('N', one, v, zero, w);
         if(SCHURCHEB_global::_gram_schmidt == 1)
         {
            SchurchebMgs( w, V, H, t, k, tol_orth, tol_reorth);
         }
         else
         {
            SchurchebCgs2( w, V, H, t, k, tol_orth);
         }
         
         nmvs++;
         
         /* check "0.0" norm -- breakdown */
         if (SchurchebAbs(t) < tol_orth)
         {
            k++;
            break;
         }
         
      }
      
      v.Clear();
      w.Clear();
      
      return k;
   }
   template int SchurchebArnoldiNoRestart<SequentialVectorClass<float> >( arnoldimatrix_seq_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<SequentialVectorClass<double> >( arnoldimatrix_seq_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<SequentialVectorClass<complexs> >( arnoldimatrix_seq_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<SequentialVectorClass<complexd> >( arnoldimatrix_seq_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<ParallelVectorClass<float> >( arnoldimatrix_par_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<ParallelVectorClass<double> >( arnoldimatrix_par_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<ParallelVectorClass<complexs> >( arnoldimatrix_par_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<ParallelVectorClass<complexd> >( arnoldimatrix_par_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<SequentialVectorClass<float> >( matrix_csr_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<SequentialVectorClass<double> >( matrix_csr_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<SequentialVectorClass<complexs> >( matrix_csr_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<SequentialVectorClass<complexd> >( matrix_csr_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<ParallelVectorClass<float> >( matrix_csr_par_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<ParallelVectorClass<double> >( matrix_csr_par_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<ParallelVectorClass<complexs> >( matrix_csr_par_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiNoRestart<ParallelVectorClass<complexd> >( matrix_csr_par_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth, int &nmvs);
   
   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int SchurchebArnoldiNoRestart2( MatrixType &A, int mstart, int msteps, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, 
                                 RealDataType tol_orth, RealDataType tol_reorth, DataType shift, int &nmvs)
   {
      
      int                  k, n;
      RealDataType         t;
      DataType             one, zero;
      
      VectorType           v, w;
      
      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);
      
      n = A.GetNumRowsLocal();
      one = 1.0;
      zero = 0.0;
      
      if(tol_orth < std::numeric_limits<RealDataType>::epsilon())
      {
         /* the tolerance should not be too small */
         tol_orth = std::numeric_limits<RealDataType>::epsilon();
      }
      
      /*------------------------
       * Start arnoldi loop
       * Compute matvec u = A*v
       * Apply Modified Gram - Schmidt
       *------------------------*/
      
      nmvs = 0;
      
      for (k = mstart; k < msteps; k++) 
      {
         /* set to each column. v is the current row, w is the next row 
          * only set is local size is not 0
          */
         if( n > 0)
         {
            v.UpdatePtr( &V(0, k), V.GetDataLocation() );
            w.UpdatePtr( &V(0, k+1), V.GetDataLocation() );
         }
         
         w.Fill( zero);
         w.Axpy( shift, v);
         A.MatVec('N', one, v, one, w); /* compute (A+shift*I)*x */
         if(SCHURCHEB_global::_gram_schmidt == 1)
         {
            SchurchebMgs( w, V, H, t, k, tol_orth, tol_reorth);
         }
         else
         {
            SchurchebCgs2( w, V, H, t, k, tol_orth);
         }
         
         nmvs++;
         
         /* check "0.0" norm -- breakdown */
         if (SchurchebAbs(t) < tol_orth)
         {
            k++;
            break;
         }
         
      }
      
      v.Clear();
      w.Clear();
      
      return k;
   }
   template int SchurchebArnoldiNoRestart2<SequentialVectorClass<float> >( arnoldimatrix_seq_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth, float shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<SequentialVectorClass<double> >( arnoldimatrix_seq_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth, double shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<SequentialVectorClass<complexs> >( arnoldimatrix_seq_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth, complexs shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<SequentialVectorClass<complexd> >( arnoldimatrix_seq_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth, complexd shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<ParallelVectorClass<float> >( arnoldimatrix_par_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth, float shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<ParallelVectorClass<double> >( arnoldimatrix_par_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth, double shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<ParallelVectorClass<complexs> >( arnoldimatrix_par_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth, complexs shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<ParallelVectorClass<complexd> >( arnoldimatrix_par_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth, complexd shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<SequentialVectorClass<float> >( matrix_csr_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth, float shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<SequentialVectorClass<double> >( matrix_csr_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth, double shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<SequentialVectorClass<complexs> >( matrix_csr_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth, complexs shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<SequentialVectorClass<complexd> >( matrix_csr_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth, complexd shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<ParallelVectorClass<float> >( matrix_csr_par_float &A, int mstart, int msteps, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, float tol_orth, float tol_reorth, float shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<ParallelVectorClass<double> >( matrix_csr_par_double &A, int mstart, int msteps, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, double tol_orth, double tol_reorth, double shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<ParallelVectorClass<complexs> >( matrix_csr_par_complexs &A, int mstart, int msteps, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, float tol_orth, float tol_reorth, complexs shift, int &nmvs);
   template int SchurchebArnoldiNoRestart2<ParallelVectorClass<complexd> >( matrix_csr_par_complexd &A, int mstart, int msteps, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, double tol_orth, double tol_reorth, complexd shift, int &nmvs);
   
#ifdef SCHURCHEB_PARPACK
   
   template <class VectorType, class MatrixType>
   int ArnoldiImplicitRestartHost( MatrixType &A, int msteps, int maxits, int nev, char *which,
                     float tol_eig, matrix_dense_float &V,
                     matrix_dense_float &H, int &nmvs)
   {
      
      /* first call NAUPD to apply Arnoldi
       * then use NEUPD to generate the final result
       */
      int i, j;
      float* H_temp;
      matrix_dense_float V_temp;
      
      /* NAUPD only */
      
      VectorType v, w; /* helping vectors for MatVec */
      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);
      
      /* Now get the problem size */
      int n = v.GetLengthLocal();
      int n_global = v.GetLengthGlobal();
      
      int ido = 0; /* must be 0 for the first call */
      char bmat = 'I'; /* B = I */
      
      int np, myid;
      MPI_Comm comm;
      A.GetMpiInfo(np, myid, comm);
      int fcomm = MPI_Comm_c2f(comm); /* convert to Fortran MPI_Comm */
      
      nmvs = 0; /* count mvs */
      
      vector_seq_float resid; /* array of length n, final residual vector */
      resid.Setup(n, true);
      
      if(n_global < nev + 2)
      {
         SCHURCHEB_ERROR("nev can't be larger than n-2.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      msteps = SchurchebMax( msteps, nev + 2);
      int ncv = SchurchebMin( msteps, n_global); // number of columns of the matrix V
      // 2 <= ncv - nev && ncv <= n
      
      /* create buffer */
      V_temp.Setup(n, ncv, true);
      int ldv = V_temp.GetLeadingDimension();
      
      vector_int iparam;
      iparam.Setup(11, true);
      
      iparam[0] = 1; // equv to restart with linear combination of approximateed Schur vectors
      iparam[2] = maxits; // max iter
      iparam[3] = 1;
      //iparam[6] = 3; //mode, M semi-spd, inverse-mode
      iparam[6] = 1; //mode, M is I
      /* on return iparam[4] holds the # of converged Ritz values */
      
      vector_int ipntr;
      ipntr.Setup(14, true);
      /* ipntr[0] and [1] are used for matvec (v and w in workd)
       * ipntr[4] is the ncv by ncv upper Hessenberg matrix H in workl 
       * ipntr[11] is the ncv by ncv upper quasi-triangular Schur matrix in workl
       */
      
      vector_seq_float workd; /* working array of length 3*n */
      workd.Setup(3*n, true);
      
      int lworkl = 3*ncv*ncv +  6*ncv; /* second work array */
      vector_seq_float workl;
      workl.Setup(lworkl, true);
      
      int info = 0; /* random init guess is used */
      
      /* NEUPD */
      
      int rvec = 1; /* Compute Ritz/Schur vectors */
      char howmny = 'P'; /* Compute nev Schur vectors, set to 'A' to compute Ritz vectors */
      vector_int select; /* length nev, since howmny = 'A' or 'P', this is the workspace */
      select.Setup(ncv, true);
      
      vector_seq_float dr, di; /* Ritz value or Ritz vector */
      dr.Setup(nev+1, true);
      di.Setup(nev+1, true);
      
      /* Since we use the 'P' option Z is not referenced */
      //int ldz = 1;
      //matrix_dense_double Z; // n by nev+1 matrix, Ritz vectors
      //Z.Setup(n, nev+1, true);
      //int ldz = n; // the leading dimension of Z
      
      /* not referenced */
      float sigmar = 0.0; /* represents the real part of the shift */
      float sigmai = 0.0; /* represents the imag part of the shift */
      
      vector_seq_float workev; /* working vector */
      workev.Setup(3*ncv);
      
      float one = 1.0;
      float zero = 0.0;
      
      while(ido != 99)
      {
         SCHURCHEB_PARPACK_PSNAUPD( &fcomm,
                  &ido, 
                  &bmat,
                  &n, 
                  which, 
                  &nev, 
                  &tol_eig, 
                  resid.GetData(), 
                  &ncv, 
                  V_temp.GetData(), 
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
      
      if(info >= 0)
      {
         SCHURCHEB_PARPACK_PSNEUPD ( &fcomm,
                     &rvec, 
                     &howmny, 
                     select.GetData(), 
                     dr.GetData(), 
                     di.GetData(), 
                     V_temp.GetData(), 
                     &ldv, 
                     &sigmar,
                     &sigmai,
                     workev.GetData(),
                     &bmat, &n, which, &nev, &tol_eig, resid.GetData(), &ncv, V_temp.GetData(), &ldv,
                     iparam.GetData(), ipntr.GetData(), workd.GetData(), workl.GetData(), &lworkl, &info );
         if(info < 0)
         {
            printf("Error: xneupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
            return -1;
         }
      }
      else
      {
         printf("Error: xnaupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
         return -1;
      }
      
      if(V.GetNumRowsLocal() != n || V.GetNumColsLocal() != nev)
      {
         /* create buffer */
         V.Setup(n, nev, true);
      }
      if(n > 0)
      {
         for(i = 0 ; i < nev ; i ++)
         {
            SCHURCHEB_MEMCPY( &(V(0,i)), &(V_temp(0,i)), n, V.GetDataLocation(), V_temp.GetDataLocation(), float);
         }
      }
      
      /* now setup H */
      nev = iparam[4];
      
      H.Setup(nev, nev, false);
      for(i = 0 ; i < nev ; i ++)
      {
         H_temp = workl.GetData()+ipntr[11]-1+i*ncv;
         for(j = 0 ; j < nev ; j ++)
         {
            H(j, i) = *(H_temp++);
         }
      }
      
      V_temp.Clear();
      dr.Clear();
      di.Clear();
      workl.Clear();
      workd.Clear();
      workev.Clear();
      resid.Clear();
      iparam.Clear();
      ipntr.Clear();
      select.Clear();
      
      return nev;
   }
   template int ArnoldiImplicitRestartHost<SequentialVectorClass<float> >( arnoldimatrix_seq_float &A, int msteps, int maxits, int nev, char *which, float tol_eig, matrix_dense_float &V, matrix_dense_float &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<ParallelVectorClass<float> >( arnoldimatrix_par_float &A, int msteps, int maxits, int nev, char *which, float tol_eig, matrix_dense_float &V, matrix_dense_float &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<SequentialVectorClass<float> >( matrix_csr_float &A, int msteps, int maxits, int nev, char *which, float tol_eig, matrix_dense_float &V, matrix_dense_float &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<ParallelVectorClass<float> >( matrix_csr_par_float &A, int msteps, int maxits, int nev, char *which, float tol_eig, matrix_dense_float &V, matrix_dense_float &H, int &nmvs);

   template <class VectorType, class MatrixType>
   int ArnoldiImplicitRestartHost( MatrixType &A, int msteps, int maxits, int nev, char *which,
                     double tol_eig, matrix_dense_double &V,
                     matrix_dense_double &H, int &nmvs)
   {
      
      /* first call NAUPD to apply Arnoldi
       * then use NEUPD to generate the final result
       */
      int i, j;
      double* H_temp;
      matrix_dense_double V_temp;
      
      /* NAUPD only */
      
      VectorType v, w; /* helping vectors for MatVec */
      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);
      
      /* Now get the problem size */
      int n = v.GetLengthLocal();
      int n_global = v.GetLengthGlobal();
      
      int ido = 0; /* must be 0 for the first call */
      char bmat = 'I'; /* B = I */
      
      int np, myid;
      MPI_Comm comm;
      A.GetMpiInfo(np, myid, comm);
      int fcomm = MPI_Comm_c2f(comm); /* convert to Fortran MPI_Comm */
      
      nmvs = 0; /* count mvs */
      
      vector_seq_double resid; /* array of length n, final residual vector */
      resid.Setup(n, true);
      
      if(n_global < nev + 2)
      {
         SCHURCHEB_ERROR("nev can't be larger than n-2.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      msteps = SchurchebMax( msteps, nev + 2);
      int ncv = SchurchebMin( msteps, n_global); // number of columns of the matrix V
      // 2 <= ncv - nev && ncv <= n
      
      /* create buffer */
      V_temp.Setup(n, ncv, true);
      int ldv = V_temp.GetLeadingDimension();
      
      vector_int iparam;
      iparam.Setup(11, true);
      
      iparam[0] = 1; // equv to restart with linear combination of approximateed Schur vectors
      iparam[2] = maxits; // max iter
      iparam[3] = 1;
      //iparam[6] = 3; //mode, M semi-spd, inverse-mode
      iparam[6] = 1; //mode, M is I
      /* on return iparam[4] holds the # of converged Ritz values */
      
      vector_int ipntr;
      ipntr.Setup(14, true);
      /* ipntr[0] and [1] are used for matvec (v and w in workd)
       * ipntr[4] is the ncv by ncv upper Hessenberg matrix H in workl 
       * ipntr[11] is the ncv by ncv upper quasi-triangular Schur matrix in workl
       */
      
      vector_seq_double workd; /* working array of length 3*n */
      workd.Setup(3*n, true);
      
      int lworkl = 3*ncv*ncv +  6*ncv; /* second work array */
      vector_seq_double workl;
      workl.Setup(lworkl, true);
      
      int info = 0; /* random init guess is used */
      
      /* NEUPD */
      
      int rvec = 1; /* Compute Ritz/Schur vectors */
      char howmny = 'P'; /* Compute nev Schur vectors, set to 'A' to compute Ritz vectors */
      vector_int select; /* length nev, since howmny = 'A' or 'P', this is the workspace */
      select.Setup(ncv, true);
      
      vector_seq_double dr, di; /* Ritz value or Ritz vector */
      dr.Setup(nev+1, true);
      di.Setup(nev+1, true);
      
      /* Since we use the 'P' option Z is not referenced */
      //int ldz = 1;
      //matrix_dense_double Z; // n by nev+1 matrix, Ritz vectors
      //Z.Setup(n, nev+1, true);
      //int ldz = n; // the leading dimension of Z
      
      /* not referenced */
      double sigmar = 0.0; /* represents the real part of the shift */
      double sigmai = 0.0; /* represents the imag part of the shift */
      
      vector_seq_double workev; /* working vector */
      workev.Setup(3*ncv);
      
      double one = 1.0;
      double zero = 0.0;
      
      while(ido != 99)
      {
         SCHURCHEB_PARPACK_PDNAUPD( &fcomm,
                  &ido, 
                  &bmat,
                  &n, 
                  which, 
                  &nev, 
                  &tol_eig, 
                  resid.GetData(), 
                  &ncv, 
                  V_temp.GetData(), 
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
      
      if(info >= 0)
      {
         SCHURCHEB_PARPACK_PDNEUPD ( &fcomm,
                     &rvec, 
                     &howmny, 
                     select.GetData(), 
                     dr.GetData(), 
                     di.GetData(), 
                     V_temp.GetData(), 
                     &ldv, 
                     &sigmar,
                     &sigmai,
                     workev.GetData(),
                     &bmat, &n, which, &nev, &tol_eig, resid.GetData(), &ncv, V_temp.GetData(), &ldv,
                     iparam.GetData(), ipntr.GetData(), workd.GetData(), workl.GetData(), &lworkl, &info );
         if(info < 0)
         {
            printf("Error: xneupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
            return -1;
         }
      }
      else
      {
         printf("Error: xnaupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
         return -1;
      }
      
      if(V.GetNumRowsLocal() != n || V.GetNumColsLocal() != nev)
      {
         /* create buffer */
         V.Setup(n, nev, true);
      }
      if(n > 0)
      {
         for(i = 0 ; i < nev ; i ++)
         {
            SCHURCHEB_MEMCPY( &(V(0,i)), &(V_temp(0,i)), n, V.GetDataLocation(), V_temp.GetDataLocation(), double);
         }
      }
      
      /* now setup H */
      nev = iparam[4];
      
      H.Setup(nev, nev, false);
      for(i = 0 ; i < nev ; i ++)
      {
         H_temp = workl.GetData()+ipntr[11]-1+i*ncv;
         for(j = 0 ; j < nev ; j ++)
         {
            H(j, i) = *(H_temp++);
         }
      }
      
      V_temp.Clear();
      dr.Clear();
      di.Clear();
      workl.Clear();
      workd.Clear();
      workev.Clear();
      resid.Clear();
      iparam.Clear();
      ipntr.Clear();
      select.Clear();
      
      return nev;
   }
   template int ArnoldiImplicitRestartHost<SequentialVectorClass<double> >( arnoldimatrix_seq_double &A, int msteps, int maxits, int nev, char *which, double tol_eig, matrix_dense_double &V, matrix_dense_double &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<ParallelVectorClass<double> >( arnoldimatrix_par_double &A, int msteps, int maxits, int nev, char *which, double tol_eig, matrix_dense_double &V, matrix_dense_double &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<SequentialVectorClass<double> >( matrix_csr_double &A, int msteps, int maxits, int nev, char *which, double tol_eig, matrix_dense_double &V, matrix_dense_double &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<ParallelVectorClass<double> >( matrix_csr_par_double &A, int msteps, int maxits, int nev, char *which, double tol_eig, matrix_dense_double &V, matrix_dense_double &H, int &nmvs);

   template <class VectorType, class MatrixType>
   int ArnoldiImplicitRestartHost( MatrixType &A, int msteps, int maxits, int nev, char *which,
                     float tol_eig, matrix_dense_complexs &V,
                     matrix_dense_complexs &H, int &nmvs)
   {
      
      /* first call NAUPD to apply Arnoldi
       * then use NEUPD to generate the final result
       */
      int i, j;
      complexs* H_temp;
      matrix_dense_complexs V_temp;
      
      /* NAUPD only */
      
      VectorType v, w; /* helping vectors for MatVec */
      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);
      
      /* Now get the problem size */
      int n = v.GetLengthLocal();
      int n_global = v.GetLengthGlobal();
      
      int ido = 0; /* must be 0 for the first call */
      char bmat = 'I'; /* B = I */
      
      int np, myid;
      MPI_Comm comm;
      A.GetMpiInfo(np, myid, comm);
      int fcomm = MPI_Comm_c2f(comm); /* convert to Fortran MPI_Comm */
      
      nmvs = 0; /* count mvs */
      
      vector_seq_complexs resid; /* array of length n, final residual vector */
      resid.Setup(n, true);
      
      if(n_global < nev + 2)
      {
         SCHURCHEB_ERROR("nev can't be larger than n-2.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      msteps = SchurchebMax( msteps, nev + 2);
      int ncv = SchurchebMin( msteps, n_global); // number of columns of the matrix V
      // 2 <= ncv - nev && ncv <= n
      
      /* create buffer */
      V_temp.Setup(n, ncv, true);
      int ldv = V_temp.GetLeadingDimension();
      
      vector_int iparam;
      iparam.Setup(11, true);
      
      iparam[0] = 1; // equv to restart with linear combination of approximateed Schur vectors
      iparam[2] = maxits; // max iter
      iparam[3] = 1;
      //iparam[6] = 3; //mode, M semi-spd, inverse-mode
      iparam[6] = 1; //mode, M is I
      /* on return iparam[4] holds the # of converged Ritz values */
      
      vector_int ipntr;
      ipntr.Setup(14, true);
      /* ipntr[0] and [1] are used for matvec (v and w in workd)
       * ipntr[4] is the ncv by ncv upper Hessenberg matrix H in workl 
       * ipntr[11] is the ncv by ncv upper quasi-triangular Schur matrix in workl
       */
      
      vector_seq_complexs workd; /* working array of length 3*n */
      workd.Setup(3*n, true);
      
      int lworkl = 3*ncv*ncv +  6*ncv; /* second work array */
      vector_seq_complexs workl;
      workl.Setup(lworkl, true);
      
      vector_seq_float rwork;
      rwork.Setup(ncv, true);
      
      int info = 0; /* random init guess is used */
      
      /* NEUPD */
      
      int rvec = 1; /* Compute Ritz/Schur vectors */
      char howmny = 'P'; /* Compute nev Schur vectors, set to 'A' to compute Ritz vectors */
      vector_int select; /* length nev, since howmny = 'A' or 'P', this is the workspace */
      select.Setup(ncv, true);
      
      vector_seq_complexs d; /* Ritz value or Ritz vector */
      d.Setup(nev+1, true);
      
      /* Since we use the 'P' option Z is not referenced */
      //int ldz = 1;
      //matrix_dense_double Z; // n by nev+1 matrix, Ritz vectors
      //Z.Setup(n, nev+1, true);
      //int ldz = n; // the leading dimension of Z
      
      /* not referenced */
      complexs sigma = 0.0; /* represents the shift */
      
      vector_seq_complexs workev; /* working vector */
      workev.Setup(3*ncv);
      
      complexs one = 1.0;
      complexs zero = 0.0;
      
      while(ido != 99)
      {
         SCHURCHEB_PARPACK_PCNAUPD( &fcomm,
                  &ido, 
                  &bmat,
                  &n, 
                  which, 
                  &nev, 
                  &tol_eig, 
                  SCHURCHEB_CAST( ccomplexs*, resid.GetData()), 
                  &ncv, 
                  SCHURCHEB_CAST( ccomplexs*, V_temp.GetData()), 
                  &ldv, 
                  iparam.GetData(),
                  ipntr.GetData(), 
                  SCHURCHEB_CAST( ccomplexs*, workd.GetData()), 
                  SCHURCHEB_CAST( ccomplexs*, workl.GetData()), 
                  &lworkl, 
                  rwork.GetData(),
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
      
      if(info >= 0)
      {
         SCHURCHEB_PARPACK_PCNEUPD ( &fcomm,
                     &rvec, 
                     &howmny, 
                     select.GetData(), 
                     SCHURCHEB_CAST( ccomplexs*, d.GetData()), 
                     SCHURCHEB_CAST( ccomplexs*, V_temp.GetData()), 
                     &ldv, 
                     SCHURCHEB_CAST( ccomplexs*, &sigma),
                     SCHURCHEB_CAST( ccomplexs*, workev.GetData()),
                     &bmat, &n, which, &nev, &tol_eig, SCHURCHEB_CAST( ccomplexs*, resid.GetData()), &ncv, SCHURCHEB_CAST( ccomplexs*, V_temp.GetData()), &ldv,
                     iparam.GetData(), ipntr.GetData(), SCHURCHEB_CAST( ccomplexs*, workd.GetData()), SCHURCHEB_CAST( ccomplexs*, workl.GetData()), &lworkl, rwork.GetData(), &info );
         if(info < 0)
         {
            printf("Error: xneupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
            return -1;
         }
      }
      else
      {
         printf("Error: xnaupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
         return -1;
      }
      
      if(V.GetNumRowsLocal() != n || V.GetNumColsLocal() != nev)
      {
         /* create buffer */
         V.Setup(n, nev, true);
      }
      if(n > 0)
      {
         for(i = 0 ; i < nev ; i ++)
         {
            SCHURCHEB_MEMCPY( &(V(0,i)), &(V_temp(0,i)), n, V.GetDataLocation(), V_temp.GetDataLocation(), complexs);
         }
      }
      
      /* now setup H */
      nev = iparam[4];
      
      H.Setup(nev, nev, false);
      for(i = 0 ; i < nev ; i ++)
      {
         H_temp = workl.GetData()+ipntr[11]-1+i*ncv;
         for(j = 0 ; j < nev ; j ++)
         {
            H(j, i) = *(H_temp++);
         }
      }
      
      V_temp.Clear();
      d.Clear();
      workl.Clear();
      workd.Clear();
      rwork.Clear();
      workev.Clear();
      resid.Clear();
      iparam.Clear();
      ipntr.Clear();
      select.Clear();
      
      return nev;
   }
   template int ArnoldiImplicitRestartHost<SequentialVectorClass<complexs> >( arnoldimatrix_seq_complexs &A, int msteps, int maxits, int nev, char *which, float tol_eig, matrix_dense_complexs &V, matrix_dense_complexs &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<ParallelVectorClass<complexs> >( arnoldimatrix_par_complexs &A, int msteps, int maxits, int nev, char *which, float tol_eig, matrix_dense_complexs &V, matrix_dense_complexs &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<SequentialVectorClass<complexs> >( matrix_csr_complexs &A, int msteps, int maxits, int nev, char *which, float tol_eig, matrix_dense_complexs &V, matrix_dense_complexs &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<ParallelVectorClass<complexs> >( matrix_csr_par_complexs &A, int msteps, int maxits, int nev, char *which, float tol_eig, matrix_dense_complexs &V, matrix_dense_complexs &H, int &nmvs);

   template <class VectorType, class MatrixType>
   int ArnoldiImplicitRestartHost( MatrixType &A, int msteps, int maxits, int nev, char *which,
                     double tol_eig, matrix_dense_complexd &V,
                     matrix_dense_complexd &H, int &nmvs)
   {
      
      /* first call NAUPD to apply Arnoldi
       * then use NEUPD to generate the final result
       */
      int i, j;
      complexd* H_temp;
      matrix_dense_complexd V_temp;
      
      /* NAUPD only */
      
      VectorType v, w; /* helping vectors for MatVec */
      A.SetupVectorPtrStr(v);
      A.SetupVectorPtrStr(w);
      
      /* Now get the problem size */
      int n = v.GetLengthLocal();
      int n_global = v.GetLengthGlobal();
      
      int ido = 0; /* must be 0 for the first call */
      char bmat = 'I'; /* B = I */
      
      int np, myid;
      MPI_Comm comm;
      A.GetMpiInfo(np, myid, comm);
      int fcomm = MPI_Comm_c2f(comm); /* convert to Fortran MPI_Comm */
      
      nmvs = 0; /* count mvs */
      
      vector_seq_complexd resid; /* array of length n, final residual vector */
      resid.Setup(n, true);
      
      if(n_global < nev + 2)
      {
         SCHURCHEB_ERROR("nev can't be larger than n-2.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      msteps = SchurchebMax( msteps, nev + 2);
      int ncv = SchurchebMin( msteps, n_global); // number of columns of the matrix V
      // 2 <= ncv - nev && ncv <= n
      
      /* create buffer */
      V_temp.Setup(n, ncv, true);
      int ldv = V_temp.GetLeadingDimension();
      
      vector_int iparam;
      iparam.Setup(11, true);
      
      iparam[0] = 1; // equv to restart with linear combination of approximateed Schur vectors
      iparam[2] = maxits; // max iter
      iparam[3] = 1;
      //iparam[6] = 3; //mode, M semi-spd, inverse-mode
      iparam[6] = 1; //mode, M is I
      /* on return iparam[4] holds the # of converged Ritz values */
      
      vector_int ipntr;
      ipntr.Setup(14, true);
      /* ipntr[0] and [1] are used for matvec (v and w in workd)
       * ipntr[4] is the ncv by ncv upper Hessenberg matrix H in workl 
       * ipntr[11] is the ncv by ncv upper quasi-triangular Schur matrix in workl
       */
      
      vector_seq_complexd workd; /* working array of length 3*n */
      workd.Setup(3*n, true);
      
      int lworkl = 3*ncv*ncv +  6*ncv; /* second work array */
      vector_seq_complexd workl;
      workl.Setup(lworkl, true);
      
      vector_seq_double rwork;
      rwork.Setup(ncv, true);
      
      int info = 0; /* random init guess is used */
      
      /* NEUPD */
      
      int rvec = 1; /* Compute Ritz/Schur vectors */
      char howmny = 'P'; /* Compute nev Schur vectors, set to 'A' to compute Ritz vectors */
      vector_int select; /* length nev, since howmny = 'A' or 'P', this is the workspace */
      select.Setup(ncv, true);
      
      vector_seq_complexd d; /* Ritz value or Ritz vector */
      d.Setup(nev+1, true);
      
      /* Since we use the 'P' option Z is not referenced */
      //int ldz = 1;
      //matrix_dense_double Z; // n by nev+1 matrix, Ritz vectors
      //Z.Setup(n, nev+1, true);
      //int ldz = n; // the leading dimension of Z
      
      /* not referenced */
      complexd sigma = 0.0; /* represents the shift */
      
      vector_seq_complexd workev; /* working vector */
      workev.Setup(3*ncv);
      
      complexd one = 1.0;
      complexd zero = 0.0;
      
      while(ido != 99)
      {
         SCHURCHEB_PARPACK_PZNAUPD( &fcomm,
                  &ido, 
                  &bmat,
                  &n, 
                  which, 
                  &nev, 
                  &tol_eig, 
                  SCHURCHEB_CAST( ccomplexd*, resid.GetData()), 
                  &ncv, 
                  SCHURCHEB_CAST( ccomplexd*, V_temp.GetData()), 
                  &ldv, 
                  iparam.GetData(),
                  ipntr.GetData(), 
                  SCHURCHEB_CAST( ccomplexd*, workd.GetData()), 
                  SCHURCHEB_CAST( ccomplexd*, workl.GetData()), 
                  &lworkl, 
                  rwork.GetData(),
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
      
      if(info >= 0)
      {
         SCHURCHEB_PARPACK_PZNEUPD ( &fcomm,
                     &rvec, 
                     &howmny, 
                     select.GetData(), 
                     SCHURCHEB_CAST( ccomplexd*, d.GetData()), 
                     SCHURCHEB_CAST( ccomplexd*, V_temp.GetData()), 
                     &ldv, 
                     SCHURCHEB_CAST( ccomplexd*, &sigma),
                     SCHURCHEB_CAST( ccomplexd*, workev.GetData()),
                     &bmat, &n, which, &nev, &tol_eig, SCHURCHEB_CAST( ccomplexd*, resid.GetData()), &ncv, SCHURCHEB_CAST( ccomplexd*, V_temp.GetData()), &ldv,
                     iparam.GetData(), ipntr.GetData(), SCHURCHEB_CAST( ccomplexd*, workd.GetData()), SCHURCHEB_CAST( ccomplexd*, workl.GetData()), &lworkl, rwork.GetData(), &info );
         if(info < 0)
         {
            printf("Error: xneupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
            return -1;
         }
      }
      else
      {
         printf("Error: xnaupd error code %d on MPI rank %d\n", info, parallel_log::_grank);
         return -1;
      }
      
      if(V.GetNumRowsLocal() != n || V.GetNumColsLocal() != nev)
      {
         /* create buffer */
         V.Setup(n, nev, true);
      }
      if(n > 0)
      {
         for(i = 0 ; i < nev ; i ++)
         {
            SCHURCHEB_MEMCPY( &(V(0,i)), &(V_temp(0,i)), n, V.GetDataLocation(), V_temp.GetDataLocation(), complexd);
         }
      }
      
      /* now setup H */
      nev = iparam[4];
      
      H.Setup(nev, nev, false);
      for(i = 0 ; i < nev ; i ++)
      {
         H_temp = workl.GetData()+ipntr[11]-1+i*ncv;
         for(j = 0 ; j < nev ; j ++)
         {
            H(j, i) = *(H_temp++);
         }
      }
      
      V_temp.Clear();
      d.Clear();
      workl.Clear();
      workd.Clear();
      rwork.Clear();
      workev.Clear();
      resid.Clear();
      iparam.Clear();
      ipntr.Clear();
      select.Clear();
      
      return nev;
   }
   template int ArnoldiImplicitRestartHost<SequentialVectorClass<complexd> >( arnoldimatrix_seq_complexd &A, int msteps, int maxits, int nev, char *which, double tol_eig, matrix_dense_complexd &V, matrix_dense_complexd &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<ParallelVectorClass<complexd> >( arnoldimatrix_par_complexd &A, int msteps, int maxits, int nev, char *which, double tol_eig, matrix_dense_complexd &V, matrix_dense_complexd &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<SequentialVectorClass<complexd> >( matrix_csr_complexd &A, int msteps, int maxits, int nev, char *which, double tol_eig, matrix_dense_complexd &V, matrix_dense_complexd &H, int &nmvs);
   template int ArnoldiImplicitRestartHost<ParallelVectorClass<complexd> >( matrix_csr_par_complexd &A, int msteps, int maxits, int nev, char *which, double tol_eig, matrix_dense_complexd &V, matrix_dense_complexd &H, int &nmvs);

#endif
   
   template <class VectorType, typename DataType, typename RealDataType>
   int SchurchebArnoldiThickRestartBuildThickRestartNewVector( DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, int m, RealDataType tol_orth, RealDataType tol_reorth, VectorType &v)
   {
      int                                 n_local, location;
      DataType                            one;
      RealDataType                        normv, t = 0.0;
      
      location = V.GetDataLocation();
      n_local = V.GetNumRowsLocal();
      one = 1.0;
      
      /* Pick a new vector */
      if(n_local > 0)
      {
         v.UpdatePtr( &(V(0, m)), location );
      }
      v.Rand();
      v.Norm2(normv);
      v.Scale(one/normv);
      
      /* now orth againist all previous cols, note that k is number of column "used" in V, exclude w */
      if(m > 0)
      {
         SchurchebOrthogonal( v, V, t, m-1, tol_orth, tol_reorth);
      }
      H(m, m-1) = 0;
      
      /* now we have a new v that is orthogal to all previous vectors, check if it's norm is too small */
      if(t < tol_orth)
      {
         /* in this case the new guess is not good enough, give up */
         //SCHURCHEB_PRINT("Thick restart can't add more eigenvectors.\n");
         return -1;
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_float &V, matrix_dense_float &H, int m, float tol_orth, float tol_reorth, vector_seq_float &v);
   template int SchurchebArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_double &V, matrix_dense_double &H, int m, double tol_orth, double tol_reorth, vector_seq_double &v);
   template int SchurchebArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_complexs &V, matrix_dense_complexs &H, int m, float tol_orth, float tol_reorth, vector_seq_complexs &v);
   template int SchurchebArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_complexd &V, matrix_dense_complexd &H, int m, double tol_orth, double tol_reorth, vector_seq_complexd &v);
   template int SchurchebArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_float &V, matrix_dense_float &H, int m, float tol_orth, float tol_reorth, vector_par_float &v);
   template int SchurchebArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_double &V, matrix_dense_double &H, int m, double tol_orth, double tol_reorth, vector_par_double &v);
   template int SchurchebArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_complexs &V, matrix_dense_complexs &H, int m, float tol_orth, float tol_reorth, vector_par_complexs &v);
   template int SchurchebArnoldiThickRestartBuildThickRestartNewVector( matrix_dense_complexd &V, matrix_dense_complexd &H, int m, double tol_orth, double tol_reorth, vector_par_complexd &v);
   
   template <typename T>
   int TestPlotGnuPlotEigReal( const char *datafilename, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, IntVectorClass<int> &select)
   {
      int n, i;
      
      FILE *fdata, *pgnuplot;
         
      char tempfilename[1024];
      snprintf( tempfilename, 1024, "./TempData/%s", datafilename );
      
      if ((fdata = fopen(tempfilename, "w")) == NULL)
      {
         printf("Can't open file.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if ((pgnuplot = popen("gnuplot -persistent", "w")) == NULL)
      {
         printf("Can't open gnuplot file.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      n = wr.GetLengthLocal();
      
      SCHURCHEB_CHKERR(n != wi.GetLengthLocal());
      
      for(i = 0 ; i < n ; i ++)
      {
         if(select[i] > 0)
         {
            fprintf(fdata, "%f %f \n", wr[i], wi[i]);
         }
      }
      
      fclose(fdata);
      /*
      fprintf(pgnuplot, "set title \"Final Eigenvalues\"\n");
      fprintf(pgnuplot, "set logscale x\n");
      //fprintf(pgnuplot, "set xrange [1:%d]\n", _ncols);
      //fprintf(pgnuplot, "set yrange [1:%d]\n", _nrows);
      fprintf(pgnuplot, "plot '%s' pt 1\n", tempfilename);
      */
      pclose(pgnuplot);
      
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T>
   int SchurchebArnoldiThickRestartBuildResultReal(DenseMatrixClass<T> &Vm, DenseMatrixClass<T> &Hm, DenseMatrixClass<T> &Q, 
                                                int &ncov, int neig_keep, vector_int &icov, SequentialVectorClass<T> &dcov,
                                                vector_int &work_int, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi,
                                                DenseMatrixClass<T> &V, DenseMatrixClass<T> &H)
   {
      if(ncov == 0)
      {
         /* in this case return immediatly */
         return SCHURCHEB_SUCCESS;
      }
      
      int                        i, j, n_local, m, location;
      vector_int                 select, order;
      T                          one, zero;
      DenseMatrixClass<T>        Vm_temp, Q_temp, V_temp;
      
      one = T(1.0);
      zero = T();
      
      /* get size first */
      n_local = Vm.GetNumRowsLocal();
      m = Vm.GetNumColsLocal();
      
      location = V.GetDataLocation();
      /* avoid allocating memory, use the working vector */
      select.SetupPtr(work_int, m, 0);
      select.Fill(0);
      order.SetupPtr(work_int, m, m);
      
      if(ncov > neig_keep)
      {
         /* select best ones
          * sort based on weight on descending order */
         dcov.Sort(order, false, true);
         for(i = 0 ; i < neig_keep ; i ++)
         {
            select[icov[order[i]]] = 1;
         }
         /* update ncov */
         ncov = neig_keep;
         
         /* check if the last two are in pairs */
         i = icov[order[neig_keep-1]];
         /* note that i might be m-1, check it to avoid index out of bound */
         if( i < m-1 && wi[i] > 0 && wi[i] == -wi[i+1])
         {
            /* In this case, we keep an extra one 
             * note that we are using the stable sort, should be here
             */
            select[icov[order[neig_keep]]] = 1;
            ncov++;
         }
         
         order.Clear();
      }
      else
      {
         /* select all */
         for(i = 0 ; i < ncov ; i ++)
         {
            select[icov[i]] = 1;
         }
      }
      /*
      if(parallel_log::_grank == 0 && n_local > 400)
      {
         TestPlotGnuPlotEigReal( "TR-Final", wr, wi, select);
      }
      */
      /* ordschur to put those into leading part */
      Hm.OrdSchur(Q, wr, wi, select);

      /* update V and H for return 
       * V first 
       */
      Vm_temp.SetupPtr(Vm, 0, 0, n_local, m);
      V_temp.SetupPtr(V, 0, 0, n_local, ncov);
      if(location == kMemoryDevice)
      {
         /* extract matrix on device to apply matmat */
         Q.SubMatrix( 0, 0, m, ncov, kMemoryDevice, Q_temp);
      }
      else
      {
         Q_temp.SetupPtr(Q, 0, 0, m, ncov);
      }
      
      V_temp.MatMat( one, Vm_temp, 'N', Q_temp, 'N', zero);
      
      /* now H */
      H.Fill(zero);
      for (i = 0; i < ncov; i++) 
      {
         for (j = 0; j < ncov; j++)
         {
            H(j,i) = Hm(j,i);
         }
      }
      
      Vm_temp.Clear();
      Q_temp.Clear();
      V_temp.Clear();
      select.Clear();
      order.Clear();
      
      return SCHURCHEB_SUCCESS;
      
   }
   
   template <typename DataType, typename RealDataType>
   int SchurchebArnoldiThickRestartBuildResultComplex(DenseMatrixClass<DataType> &Vm, DenseMatrixClass<DataType> &Hm, DenseMatrixClass<DataType> &Q, 
                                                int &ncov, int neig_keep, vector_int &icov, SequentialVectorClass<RealDataType> &dcov,
                                                vector_int &work_int, SequentialVectorClass<DataType> &w,
                                                DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H)
   {
      
      if(ncov == 0)
      {
         /* in this case return immediatly */
         return SCHURCHEB_SUCCESS;
      }
      
      int                        i, j, n_local, m, location;
      vector_int                 select, order;
      DataType                   one, zero;
      DenseMatrixClass<DataType> Vm_temp, Q_temp, V_temp;
      
      one = DataType(1.0);
      zero = DataType();
      
      /* get size first */
      n_local = Vm.GetNumRowsLocal();
      m = Vm.GetNumColsLocal();
      
      location = V.GetDataLocation();
      
      /* avoid allocating memory, use the working vector */
      select.SetupPtr(work_int, m, 0);
      select.Fill(0);
      order.SetupPtr(work_int, m, m);
      
      if(ncov > neig_keep)
      {
         /* select best ones
          * sort based on weight on descending order */
         dcov.Sort(order, false, true);
         for(i = 0 ; i < neig_keep ; i ++)
         {
            select[icov[order[i]]] = 1;
         }
         /* update ncov */
         ncov = neig_keep;
         
         order.Clear();
      }
      else
      {
         /* select all */
         for(i = 0 ; i < ncov ; i ++)
         {
            select[icov[i]] = 1;
         }
      }
      
      /* ordschur to put those into leading part */
      Hm.OrdSchur(Q, w, select);
      
      /* update V and H for return 
       * V first 
       */
      Vm_temp.SetupPtr(Vm, 0, 0, n_local, m);
      V_temp.SetupPtr(V, 0, 0, n_local, ncov);
      if(location == kMemoryDevice)
      {
         /* extract matrix on device to apply matmat */
         Q.SubMatrix( 0, 0, m, ncov, kMemoryDevice, Q_temp);
      }
      else
      {
         Q_temp.SetupPtr(Q, 0, 0, m, ncov);
      }
      
      V_temp.MatMat( one, Vm_temp, 'N', Q_temp, 'N', zero);
      
      /* now H */
      H.Fill(zero);
      for (i = 0; i < ncov; i++) 
      {
         for (j = 0; j < ncov; j++)
         {
            H(j,i) = Hm(j,i);
         }
      }
      
      Vm_temp.Clear();
      Q_temp.Clear();
      V_temp.Clear();
      select.Clear();
      order.Clear();
      
      return SCHURCHEB_SUCCESS;
      
   }
   
   template <typename T>
   int SchurchebArnoldiThickRestartBuildThickRestartNoLockReal(DenseMatrixClass<T> &Vm, DenseMatrixClass<T> &Hm, DenseMatrixClass<T> &Q, 
                                                T h_last, int ncov, int nicov, int &npick, 
                                                vector_int &icov, vector_int &iicov, SequentialVectorClass<T> &dicov, 
                                                vector_int &work_int, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, 
                                                DenseMatrixClass<T> &V, DenseMatrixClass<T> &H)
   {
      SCHURCHEB_CHKERR(npick == 0);
      if(npick == nicov)
      {
         /* restart with all vectors 
          * just apply more Arnoldi. Typically should not reach here.
          */
         return SCHURCHEB_SUCCESS;
      }
      
      int                  i, j, n_local, m, location;
      int                  trlen = npick + ncov;
      T                    one, zero;
      vector_int           select, order;
      DenseMatrixClass<T>  Vm_temp, Q_temp, V_temp;
      
      one = T(1.0);
      zero =T();
      
      /* get size first */
      n_local = Vm.GetNumRowsLocal();
      m = Vm.GetNumColsLocal();
      
      location = V.GetDataLocation();
      
      /* avoid allocating memory, use the working vector */
      select.SetupPtr(work_int, m, 0);
      select.Fill(0);
      order.SetupPtr(work_int, m, m);
      
      /* select all the convergenced eigenvalues first */
      for(i = 0 ; i < ncov ; i ++)
      {
         select[icov[i]] = 1;
      }
      
      /* now pick those "best" unconvergenced eigenvalues */
      dicov.Sort(order, false, true);
      
      for(i = 0 ; i < npick ; i ++)
      {
         select[iicov[order[i]]] = 1;
      }
      
      /* check if the last two are in pair */
      i = iicov[order[npick-1]];
      /* note that i might be m-1, check it to avoid index out of bound */
      if( i < m-1 && wi[i] > 0 && wi[i] == -wi[i+1] )
      {
         /* In this case, we keep an extra one */
         select[iicov[order[npick]]] = 1;
         npick++;
         trlen++;
      }
      
      /* now build the thick-restart */
      
      /* ordschur to put those into leading part */
      Hm.OrdSchur(Q, wr, wi, select);
      
      /* copy the first trlen * trlen */
      H.Fill(zero);
      for(i = 0 ; i < trlen ; i ++)
      {
         for(j = 0 ; j < trlen ; j ++)
         {
            H(j, i) = Hm(j, i);
         }
      }
      
      /* update the row trlen + 1 
       *   * * * *
       *     * * *
       *     * * * <= might have 2x2 blocks
       *         *
       *   * * * * <= now this row
       */
      for(i = 0 ; i < trlen ; i ++)
      {
         H(trlen, i) = h_last * Q(m-1, i);
      }
      
      /* now prepare V, V := (Vm * Qhs)(:, 1:trlen) */
      Vm_temp.SetupPtr(Vm, 0, 0, n_local, m);
      V_temp.SetupPtr(V, 0, 0, n_local, trlen);
      if(location == kMemoryDevice)
      {
         Q.SubMatrix( 0, 0, m, trlen, kMemoryDevice, Q_temp);
      }
      else
      {
         Q_temp.SetupPtr(Q, 0, 0, m, trlen);
      }
      
      V_temp.MatMat( one, Vm_temp, 'N', Q_temp, 'N', zero);
      
      /* now copy the restart vector, the original last vector */
      SCHURCHEB_MEMCPY(V.GetData()+trlen*n_local, V.GetData()+m*n_local, n_local, location, location, T);
      
      /* all set, ready to restart */
      
      Vm_temp.Clear();
      Q_temp.Clear();
      V_temp.Clear();
      select.Clear();
      order.Clear();
      
      return SCHURCHEB_SUCCESS;
      
   }
   
   template <typename DataType, typename RealDataType>
   int SchurchebArnoldiThickRestartBuildThickRestartNoLockComplex(DenseMatrixClass<DataType> &Vm, DenseMatrixClass<DataType> &Hm, DenseMatrixClass<DataType> &Q, 
                                                DataType h_last, int ncov, int nicov, int &npick, 
                                                vector_int &icov, vector_int &iicov, SequentialVectorClass<RealDataType> &dicov, 
                                                vector_int &work_int, SequentialVectorClass<DataType> &w, 
                                                DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H)
   {
      SCHURCHEB_CHKERR(npick == 0);
      if(npick == nicov)
      {
         /* restart with all vectors 
          * just apply more Arnoldi. Typically should not reach here.
          */
         return SCHURCHEB_SUCCESS;
      }
      
      int                           i, j, n_local, m, location;
      int                           trlen = npick + ncov;
      DataType                      one, zero;
      vector_int                    select, order;
      DenseMatrixClass<DataType>    Vm_temp, Q_temp, V_temp;
      
      one = DataType(1.0);
      zero = DataType();
      
      /* get size first */
      n_local = Vm.GetNumRowsLocal();
      m = Vm.GetNumColsLocal();
      
      location = V.GetDataLocation();
      
      /* avoid allocating memory, use the working vector */
      select.SetupPtr(work_int, m, 0);
      select.Fill(0);
      order.SetupPtr(work_int, m, m);
      
      /* select all the convergenced eigenvalues first */
      for(i = 0 ; i < ncov ; i ++)
      {
         select[icov[i]] = 1;
      }
      
      /* now pick those "best" unconvergenced eigenvalues */
      dicov.Sort(order, false, true);
      
      for(i = 0 ; i < npick ; i ++)
      {
         select[iicov[order[i]]] = 1;
      }
      
      /* now build the thick-restart */
      
      /* ordschur to put those into leading part */
      Hm.OrdSchur(Q, w, select);
      
      /* copy the first trlen * trlen */
      H.Fill(zero);
      for(i = 0 ; i < trlen ; i ++)
      {
         for(j = 0 ; j < trlen ; j ++)
         {
            H(j, i) = Hm(j, i);
         }
      }
      
      /* update the row trlen + 1 
       *   * * * *
       *     * * *
       *     * * * <= might have 2x2 blocks
       *         *
       *   * * * * <= now this row
       */
      for(i = 0 ; i < trlen ; i ++)
      {
         H(trlen, i) = h_last * Q(m-1, i);
      }
      
      /* now prepare V, V := (Vm * Qhs)(:, 1:trlen) */
      Vm_temp.SetupPtr(Vm, 0, 0, n_local, m);
      V_temp.SetupPtr(V, 0, 0, n_local, trlen);
      if(location == kMemoryDevice)
      {
         Q.SubMatrix( 0, 0, m, trlen, kMemoryDevice, Q_temp);
      }
      else
      {
         Q_temp.SetupPtr(Q, 0, 0, m, trlen);
      }
      
      V_temp.MatMat( one, Vm_temp, 'N', Q_temp, 'N', zero);
      
      /* now copy the restart vector, the original last vector */
      SCHURCHEB_MEMCPY(V.GetData()+trlen*n_local, V.GetData()+m*n_local, n_local, location, location, DataType);
      
      /* all set, ready to restart */
      
      Vm_temp.Clear();
      Q_temp.Clear();
      V_temp.Clear();
      select.Clear();
      order.Clear();
      
      return SCHURCHEB_SUCCESS;
      
   }
   
   template <typename T>
   int TestPlotGnuPlotEigReal( const char *datafilename, SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, IntVectorClass<int> &icov, IntVectorClass<int> &iicov)
   {
      int n, i;
      
      FILE *fdata, *pgnuplot;
         
      char tempfilename[1024];
      snprintf( tempfilename, 1024, "./TempData/%s", datafilename );
      
      if ((fdata = fopen(tempfilename, "w")) == NULL)
      {
         printf("Can't open file.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if ((pgnuplot = popen("gnuplot -persistent", "w")) == NULL)
      {
         printf("Can't open gnuplot file.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      n = icov.GetLengthLocal();
      
      for(i = 0 ; i < n ; i ++)
      {
         fprintf(fdata, "%f %f \n", wr[icov[i]], wi[icov[i]]);
      }
      
      fclose(fdata);
      
      /*
      fprintf(pgnuplot, "set title \"Eigenvalues\"\n");
      fprintf(pgnuplot, "set logscale x\n");
      //fprintf(pgnuplot, "set xrange [1:%d]\n", _ncols);
      //fprintf(pgnuplot, "set yrange [1:%d]\n", _nrows);
      fprintf(pgnuplot, "plot '%s' pt 1\n", tempfilename);
      */
      
      pclose(pgnuplot);
      
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T>
   int SchurchebArnoldiThickRestartChooseEigenValuesReal( DenseMatrixClass<T> &H, DenseMatrixClass<T> &Q, T h_last, 
                                                         T (*weight)(ComplexValueClass<T>), T truncate, 
                                                         int &ncov, int &nicov, int &nsatis, T tol_eig, bool &cut, 
                                                         SequentialVectorClass<T> &wr, SequentialVectorClass<T> &wi, 
                                                         vector_int &icov, vector_int &iicov, vector_int &work_int, 
                                                         SequentialVectorClass<T> &dcov, SequentialVectorClass<T> &dicov, SequentialVectorClass<T> &dall)
   {
      /* typically we should not have those on the device memory */
      SCHURCHEB_CHKERR(icov.GetDataLocation() == kMemoryDevice);
      SCHURCHEB_CHKERR(iicov.GetDataLocation() == kMemoryDevice);
      SCHURCHEB_CHKERR(dcov.GetDataLocation() == kMemoryDevice);
      SCHURCHEB_CHKERR(dicov.GetDataLocation() == kMemoryDevice);
      
      /* This function does the following:
       * 1. Check the convergence of Ritz values
       * 2. Get the number of satisfied eigenvalues
       * 
       * We select "continues" largest weight as satisfied.
       * Example: 4 Ritz value with weights 4,3,2,1.
       *          If 4, 3, 1 converged, 2 doesn't, only
       *          4 and 3 are satisfied, 1 is not.
       */
      
      typedef ComplexValueClass<T> TC;
      
      int                              i, m;
      T                                res, zero;
      TC                               temp_val;
      T                                temp_val_real, maxweight;
      DenseMatrixClass<T>              Qe, Q_last, Qe_last;
      vector_int                       marker, order;
      
      m = wr.GetLengthLocal();
      SCHURCHEB_CHKERR(wi.GetLengthLocal() != m);
      
      /* the index */
      icov.Resize(0, false, false);
      iicov.Resize(0, false, false);
      /* the distance */
      dcov.Resize(0, false, false);
      dicov.Resize(0, false, false);
      dall.Resize(m, false, false);
      
      cut = false;
      zero = 0.0;
      maxweight = 0.0;
      
      /* Eigen-decomposition */
      H.HessEig(Qe, wr, wi);
      
      /* check convergence based on the lase row of the eigen decomposition Qh*Qs*Qe */
      Q_last.SetupPtr( Q, m-1, 0, 1, m);
      Qe_last.MatMat( h_last, Q_last, 'N', Qe, 'N', zero);
      
      ncov = 0;
      nicov = 0;
      
      marker.SetupPtr(work_int, m, 0);
      order.SetupPtr(work_int, m, m);
      
      /* loop through w */
      for(i = 0 ; i < m ; i ++)
      {
         if( i < m-1 && wi[i] > 0 && wi[i] == -wi[i+1])
         {
            /* In this case we have a pair of eigenvalues */
            res = SchurchebAbs(Qe_last(0, i)+Qe_last(0, i+1));
            if(res < tol_eig)
            {
               /* those are two convergenced eigenvalues */
               icov.PushBack(i);
               icov.PushBack(i+1);
               temp_val = TC(wr[i], wi[i]);
               temp_val_real = (*weight)(temp_val);
               maxweight = SchurchebMax(temp_val_real, maxweight);
               dcov.PushBack(temp_val_real);
               dcov.PushBack(temp_val_real);
               
               /* update the weight and marker */
               dall[i] = temp_val_real;
               dall[i+1] = temp_val_real;
               marker[i] = 1;
               marker[i+1] = 1;
               
               i++;
               ncov+=2;
            }
            else
            {
               /* those are two inconvergenced eigenvalue */
               iicov.PushBack(i);
               iicov.PushBack(i+1);
               temp_val = TC(wr[i], wi[i]);
               temp_val_real = (*weight)(temp_val);
               maxweight = SchurchebMax(temp_val_real, maxweight);
               dicov.PushBack(temp_val_real);
               dicov.PushBack(temp_val_real);
               
               /* update the weight and marker */
               dall[i] = temp_val_real;
               dall[i+1] = temp_val_real;
               marker[i] = 0;
               marker[i+1] = 0;
               
               i++;
               nicov+=2;
            }
         }
         else
         {
            res = SchurchebAbs(Qe_last(0, i));
            if(res < tol_eig)
            {
               /* this is a convergenced eigenvalue */
               icov.PushBack(i);
               temp_val = TC(wr[i], T());
               temp_val_real = (*weight)(temp_val);
               maxweight = SchurchebMax(temp_val_real, maxweight);
               dcov.PushBack(temp_val_real);
               
               /* update the weight and marker */
               dall[i] = temp_val_real;
               marker[i] = 1;
               
               ncov++;
            }
            else
            {
               /* this is not a convergenced eigenvalue */
               iicov.PushBack(i);
               temp_val = TC(wr[i], T());
               temp_val_real = (*weight)(temp_val);
               maxweight = SchurchebMax(temp_val_real, maxweight);
               dicov.PushBack(temp_val_real);
               
               /* update the weight and marker */
               dall[i] = temp_val_real;
               marker[i] = 0;
               
               nicov++;
            }
         }
      }
      
      /* now select satisfied values */
      
      nsatis = 0;
      maxweight *= truncate;
      dall.Sort( order, false, true);
      
      for(i = 0 ; i < ncov ; i ++)
      {
         if(marker[order[i]] == 1)
         {
            /* converge, check for truncation */
            if(dall[order[i]] >= maxweight || maxweight <= 0.0)
            {
               nsatis++;
            }
            else
            {
               /* stops here */
               SCHURCHEB_PRINT_DEBUG(parallel_log::_grank, 0, "Cut val (%f,%f)\n", wr[icov[i]], wi[icov[i]]);
               cut = true;
               break;
            }
         }
         else
         {
            break;
         }
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebArnoldiThickRestartChooseEigenValuesReal( matrix_dense_float &H, matrix_dense_float &Q, float h_last, float (*weight)(complexs), float truncate, int &ncov, int &nicov, int &nsatis, float tol_eig, bool &cut, vector_seq_float &wr, vector_seq_float &wi, vector_int &icov, vector_int &iicov, vector_int &work_int, vector_seq_float &dcov, vector_seq_float &dicov, vector_seq_float &dall);
   template int SchurchebArnoldiThickRestartChooseEigenValuesReal( matrix_dense_double &H, matrix_dense_double &Q, double h_last, double (*weight)(complexd), double truncate, int &ncov, int &nicov, int &nsatis, double tol_eig, bool &cut, vector_seq_double &wr, vector_seq_double &wi, vector_int &icov, vector_int &iicov, vector_int &work_int, vector_seq_double &dcov, vector_seq_double &dicov, vector_seq_double &dall);
   
   template <typename DataType, typename RealDataType>
   int SchurchebArnoldiThickRestartChooseEigenValuesComplex( DenseMatrixClass<DataType> &H, DenseMatrixClass<DataType> &Q, DataType h_last, 
                                                         RealDataType (*weight)(DataType), RealDataType truncate, 
                                                         int &ncov, int &nicov, int &nsatis, RealDataType tol_eig, bool &cut,
                                                         SequentialVectorClass<DataType> &w, 
                                                         vector_int &icov, vector_int &iicov, vector_int &work_int,
                                                         SequentialVectorClass<RealDataType> &dcov, SequentialVectorClass<RealDataType> &dicov, SequentialVectorClass<RealDataType> &dall)
   {
      /* typically we should not have those on the device memory */
      SCHURCHEB_CHKERR(icov.GetDataLocation() == kMemoryDevice);
      SCHURCHEB_CHKERR(iicov.GetDataLocation() == kMemoryDevice);
      SCHURCHEB_CHKERR(dcov.GetDataLocation() == kMemoryDevice);
      SCHURCHEB_CHKERR(dicov.GetDataLocation() == kMemoryDevice);
      
      int                              i, m;
      RealDataType                     res, temp_val, maxweight;
      DataType                         zero;
      DenseMatrixClass<DataType>       Qe, Q_last, Qe_last;
      vector_int                       marker, order;
      
      m = w.GetLengthLocal();
      
      /* the index */
      icov.Resize(0, false, false);
      iicov.Resize(0, false, false);
      /* the distance */
      dcov.Resize(0, false, false);
      dicov.Resize(0, false, false);
      dall.Resize(m, false, false);
      
      cut = false;
      zero = 0.0;
      maxweight = 0.0;
      
      /* Eigen-decomposition */
      H.HessEig(Qe, w);
      
      /* check convergence based on the lase row of the eigen decomposition Qh*Qs*Qe */
      Q_last.SetupPtr( Q, m-1, 0, 1, m);
      Qe_last.MatMat( h_last, Q_last, 'N', Qe, 'N', zero);
      
      ncov = 0;
      nicov = 0;
      
      marker.SetupPtr(work_int, m, 0);
      order.SetupPtr(work_int, m, m);
      
      /* loop through w */
      for(i = 0 ; i < m ; i ++)
      {
         res = SchurchebAbs(Qe_last(0, i));
         if(res < tol_eig)
         {
            /* this is a convergenced eigenvalue */
            icov.PushBack(i);
            temp_val = (*weight)(w[i]);
            dcov.PushBack(temp_val);
            maxweight = SchurchebMax(temp_val, maxweight);
            
            dall[i] = temp_val;
            marker[i] = 1;
            
            ncov++;
         }
         else
         {
            /* this is not a convergenced eigenvalue */
            iicov.PushBack(i);
            temp_val = (*weight)(w[i]);
            dicov.PushBack(temp_val);
            maxweight = SchurchebMax(temp_val, maxweight);
            
            dall[i] = temp_val;
            marker[i] = 0;
            
            nicov++;
         }
      }
      
      /* now select satisfied values */
      
      nsatis = 0;
      
      maxweight *= truncate;
      
      dall.Sort( order, false, true);
      
      for(i = 0 ; i < ncov ; i ++)
      {
         if( marker[order[i]] == 1)
         {
            if(dall[order[i]] > maxweight || maxweight <= 0.0)
            {
               nsatis++;
            }
            else
            {
               SCHURCHEB_PRINT_DEBUG(parallel_log::_grank, 0, "Cut val (%f,%f)\n", w[icov[i]].Real(), w[icov[i]].Imag());
               cut = true;
               break;
            }
         }
         else
         {
            break;
         }
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebArnoldiThickRestartChooseEigenValuesComplex( matrix_dense_complexs &H, matrix_dense_complexs &Q, complexs h_last, float (*weight)(complexs), float truncate, int &ncov, int &nicov, int &nsatis, float tol_eig, bool &cut, vector_seq_complexs &w, vector_int &icov, vector_int &iicov, vector_int &work_int, vector_seq_float &dcov, vector_seq_float &dicov, vector_seq_float &dall);
   template int SchurchebArnoldiThickRestartChooseEigenValuesComplex( matrix_dense_complexd &H, matrix_dense_complexd &Q, complexd h_last, double (*weight)(complexd), double truncate, int &ncov, int &nicov, int &nsatis, double tol_eig, bool &cut, vector_seq_complexd &w, vector_int &icov, vector_int &iicov, vector_int &work_int, vector_seq_double &dcov, vector_seq_double &dicov, vector_seq_double &dall);
   
   /* weight functions */
   template <typename T1, typename T2>
   T1 ComputeLargeMag(T2 val)
   {
      return (T1)SchurchebAbs(val);
   }
   
   float ComputeSmallReal(float val)
   {
      return -val;
   }
   
   double ComputeSmallReal(double val)
   {
      return -val;
   }
   
   float ComputeSmallReal(complexs val)
   {
      return -val.Real();
   }
   
   double ComputeSmallReal(complexd val)
   {
      return -val.Real();
   }
   
   float ComputeLargeReal(float val)
   {
      return val;
   }
   
   double ComputeLargeReal(double val)
   {
      return val;
   }
   
   float ComputeLargeReal(complexs val)
   {
      return val.Real();
   }
   
   double ComputeLargeReal(complexd val)
   {
      return val.Real();
   }
   
   float ComputeSmallComplex(float val)
   {
      return 0.0;
   }
   
   double ComputeSmallComplex(double val)
   {
      return 0.0;
   }
   
   float ComputeSmallComplex(complexs val)
   {
      return -val.Imag();
   }
   
   double ComputeSmallComplex(complexd val)
   {
      return -val.Imag();
   }
   
   float ComputeLargeComplex(float val)
   {
      return 0.0;
   }
   
   double ComputeLargeComplex(double val)
   {
      return 0.0;
   }
   
   float ComputeLargeComplex(complexs val)
   {
      return val.Imag();
   }
   
   double ComputeLargeComplex(complexd val)
   {
      return val.Imag();
   }
   
   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int SchurchebArnoldiThickRestartNoLock( MatrixType &A, int msteps, int maxits, int rank, int rank2, 
                                 RealDataType tr_fact, RealDataType tol_eig, char* opt,
                                 DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, 
                                 RealDataType tol_orth, RealDataType tol_reorth, int &nmvs)
   {
      
      switch(opt[0])
      {
         case 'S':
         {
            switch(opt[1])
            {
               case 'R':
               {
                  /* SR: smallest real */
                  return SchurchebArnoldiThickRestartNoLock<VectorType>( A, msteps, maxits, rank, rank2, RealDataType(0.0), tr_fact, tol_eig, 
                                                               &ComputeSmallReal, V, H, tol_orth, tol_reorth, nmvs);
                  break;
               }
               case 'C':
               {
                  /* SR: smallest complex */
                  return SchurchebArnoldiThickRestartNoLock<VectorType>( A, msteps, maxits, rank, rank2, RealDataType(0.0), tr_fact, tol_eig, 
                                                               &ComputeSmallComplex, V, H, tol_orth, tol_reorth, nmvs);
                  break;
               }
               case 'M':
               {
                  SCHURCHEB_ERROR("Smallest Magnitude not supported in this version.");
                  break;
               }
               default:
               {
                  SCHURCHEB_ERROR("Invalid Arnoldi option 2.");
               }
            }
            
            break;
         }
         case 'L':
         {
            switch(opt[1])
            {
               case 'R':
               {
                  /* SR: smallest real */
                  return SchurchebArnoldiThickRestartNoLock<VectorType>( A, msteps, maxits, rank, rank2, RealDataType(0.0), tr_fact, tol_eig, 
                                                               &ComputeLargeReal, V, H, tol_orth, tol_reorth, nmvs);
                  break;
               }
               case 'C':
               {
                  /* SR: smallest complex */
                  return SchurchebArnoldiThickRestartNoLock<VectorType>( A, msteps, maxits, rank, rank2, RealDataType(0.0), tr_fact, tol_eig, 
                                                               &ComputeLargeComplex, V, H, tol_orth, tol_reorth, nmvs);
                  break;
               }
               case 'M':
               {
                  return SchurchebArnoldiThickRestartNoLock<VectorType>( A, msteps, maxits, rank, rank2, RealDataType(0.0), tr_fact, tol_eig, 
                                                               &ComputeLargeMag, V, H, tol_orth, tol_reorth, nmvs);
                  break;
               }
               default:
               {
                  SCHURCHEB_ERROR("Invalid Arnoldi option 2.");
               }
            }
            
            break;
         }
         default:
         {
            SCHURCHEB_ERROR("Invalid Arnoldi option 1.");
         }
      }
      
      return SCHURCHEB_ERROR_INVALED_OPTION;
   }
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_float>( arnoldimatrix_seq_float &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, char* opt, matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_double>( arnoldimatrix_seq_double &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, char* opt, matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_float>( arnoldimatrix_par_float &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, char* opt, matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_double>( arnoldimatrix_par_double &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, char* opt, matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexs>( arnoldimatrix_seq_complexs &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, char* opt, matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexd>( arnoldimatrix_seq_complexd &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, char* opt, matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexs>( arnoldimatrix_par_complexs &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, char* opt, matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexd>( arnoldimatrix_par_complexd &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, char* opt, matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_float>( matrix_csr_float &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, char* opt, matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_double>( matrix_csr_double &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, char* opt, matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_float>( matrix_csr_par_float &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, char* opt, matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_double>( matrix_csr_par_double &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, char* opt, matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexs>( matrix_csr_complexs &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, char* opt, matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexd>( matrix_csr_complexd &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, char* opt, matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexs>( matrix_csr_par_complexs &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, char* opt, matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexd>( matrix_csr_par_complexd &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, char* opt, matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   
   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int SchurchebArnoldiThickRestartNoLock( MatrixType &A, int msteps, int maxits, int rank, int rank2, 
                                 RealDataType tr_fact, RealDataType tol_eig, 
                                 RealDataType (*weight)(ComplexValueClass<RealDataType>),
                                 DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, 
                                 RealDataType tol_orth, RealDataType tol_reorth, int &nmvs)
   {
      /* call arnoldi with truncate = 0.0 */
      return SchurchebArnoldiThickRestartNoLock<VectorType>( A, msteps, maxits, rank, rank2, RealDataType(0.0), tr_fact, tol_eig, 
                                 weight, V, H, tol_orth, tol_reorth, nmvs);
   }
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_float>( arnoldimatrix_seq_float &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_double>( arnoldimatrix_seq_double &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_float>( arnoldimatrix_par_float &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_double>( arnoldimatrix_par_double &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexs>( arnoldimatrix_seq_complexs &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexd>( arnoldimatrix_seq_complexd &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexs>( arnoldimatrix_par_complexs &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexd>( arnoldimatrix_par_complexd &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_float>( matrix_csr_float &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_double>( matrix_csr_double &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_float>( matrix_csr_par_float &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_double>( matrix_csr_par_double &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexs>( matrix_csr_complexs &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexd>( matrix_csr_complexd &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexs>( matrix_csr_par_complexs &A, int msteps, int maxits, int rank, int rank2, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexd>( matrix_csr_par_complexd &A, int msteps, int maxits, int rank, int rank2, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   
   template <class VectorType, class MatrixType, typename DataType, typename RealDataType>
   int SchurchebArnoldiThickRestartNoLock( MatrixType &A, int msteps, int maxits, int rank, int rank2, RealDataType truncate, 
                                 RealDataType tr_fact, RealDataType tol_eig, 
                                 RealDataType (*weight)(ComplexValueClass<RealDataType>),
                                 DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, 
                                 RealDataType tol_orth, RealDataType tol_reorth, int &nmvs)
   {
      SCHURCHEB_CHKERR(rank < 0);
      SCHURCHEB_CHKERR(maxits < 0);
      SCHURCHEB_CHKERR(msteps < 0);
      
      if(rank == 0 || msteps == 0 || maxits == 0)
      {
         return 0;
      }
      
      if(tol_orth < std::numeric_limits<RealDataType>::epsilon())
      {
         /* the tolerance should not be too small */
         tol_orth = std::numeric_limits<RealDataType>::epsilon();
      }
      
      if(tol_eig < std::numeric_limits<RealDataType>::epsilon())
      {
         /* the tolerance should not be too small */
         tol_eig = std::numeric_limits<RealDataType>::epsilon();
      }
      
      /* - - - - - - - - - - - - - - - - - -
       * Thick-restart version Arnoldi.
       * 
       * Version 1: When the dropping tolorance is large.
       * 
       * In this setup, no locked eigenvalues.
       * 
       * Step 1: Generate the intial guess, set trlen := 0.
       * 
       * Step 2: Apply Arnoldi iteration starting from trlen, to 
       *         msteps = m + trlen.
       * 
       * Step 3: H is now having the structure:
       *         * * * * * * *
       *           * * * * * *
       *             * * * * *
       *         * * * * * * *
       *               * * * *
       *                 * * *
       *                   * *
       *                     *
       *         Extract Hm := H(1:m, :), and check convergence.
       * 
       * Step 4: Assume we have ncov convergenced eigenvalues, nicov inconvergenced
       *         eigenvalues, restart with trlen = ncov + nicov * trfact.
       *         When there are no incovergenced eigenvalues, it's likely that we've
       *         found an invarient subspace. In this case, keep the current result,
       *         start from a new initial guess.
       *         Go back to step 2, or stop.
       */
      /* define the data type */
      typedef ComplexValueClass<RealDataType> ComplexDataType;
      
      int                                                      n_local, i, j, m, mstepsi, expand, maxsteps, maxsteps_new, trlen, npick, its, nmvs_loc;
      SCHURCHEB_long                                           n_global;
      int                                                      location;
      int                                                      ncov, nicov, nsatis;
      bool                                                     cut;
      vector_int                                               icov, iicov;
      SequentialVectorClass<RealDataType>                      dcov, dicov, dall;
      vector_int                                               work_int;
      SequentialVectorClass<RealDataType>                      wr, wi;
      SequentialVectorClass<ComplexDataType>                   w;
      
      DataType                                                 h_last;
      RealDataType                                             zero_r, h_last_r;
      ComplexDataType                                          zero_c, h_last_c;
      
      DenseMatrixClass<DataType>                               V_data, H_data, Qhs;
      DenseMatrixClass<DataType>                               V_new, H_new, V_data_new, H_data_new;
      DenseMatrixClass<DataType>                               Vm, Hm;
      DenseMatrixClass<RealDataType>                           *V_r, *H_r, *Vm_r, *Hm_r, *Qhs_r;
      DenseMatrixClass<ComplexDataType>                        *V_c, *H_c, *Vm_c, *Hm_c, *Qhs_c;
      DenseMatrixClass<DataType>                               V_temp, Vm_temp, Q_temp;
      
      /* update pointers 
       * TODO: find a better way to combine those templates
       */
      V_r = (DenseMatrixClass<RealDataType>*)&V;
      V_c = (DenseMatrixClass<ComplexDataType>*)&V;
      H_r = (DenseMatrixClass<RealDataType>*)&H;
      H_c = (DenseMatrixClass<ComplexDataType>*)&H;
      Vm_r = (DenseMatrixClass<RealDataType>*)&Vm;
      Vm_c = (DenseMatrixClass<ComplexDataType>*)&Vm;
      Hm_r = (DenseMatrixClass<RealDataType>*)&Hm;
      Hm_c = (DenseMatrixClass<ComplexDataType>*)&Hm;
      Qhs_r = (DenseMatrixClass<RealDataType>*)&Qhs;
      Qhs_c = (DenseMatrixClass<ComplexDataType>*)&Qhs;
      
      VectorType                                               v;
      
      /*------------------------
       * 1: Init Phase
       * Declare all variables and setup parameters
       *------------------------*/
      
      /* this is the maximum number of columns we can have, use this to generate the working buffer */
      maxsteps = H.GetNumColsLocal();
      
      zero_r                     = RealDataType();  
      zero_c                     = ComplexDataType();  
      
      A.SetupVectorPtrStr(v);
      
      n_local = V.GetNumRowsLocal();
      n_global = v.GetLengthGlobal();
      location = V.GetDataLocation();
      
      /* create working buffer.
       * Note that V can be on the device.
       * H can be on the host.
       */
      V_data.Setup( n_local, maxsteps+1, location, true);
      H_data.Setup( maxsteps+1, maxsteps, kMemoryHost, true);
      
      work_int.Setup(2*maxsteps);
      
      icov.Setup(maxsteps);
      iicov.Setup(maxsteps);
      /* the distance */
      dcov.Setup(maxsteps);
      dicov.Setup(maxsteps);
      dall.Setup(maxsteps);
      
      /*------------------------ 
      * 2: Arnoldi and get result
      *------------------------*/
      
      its   = 0;
      trlen = 0;
      ncov = 0;
      expand = SchurchebMax( 1, (int)(msteps*tr_fact)); // each time we expand the size by this
      
      /* main loop */
      
      /* The tolorance for the residual of the dropping tolorance is too large.
       * Lock of eigenvalues disabled, more restarts doesn't guarantee more eigenvalues.
       */
      
      nmvs = 0; 
      
      while(its < maxits)
      {
         its ++;
         
         /*------------------------ 
         * 2.0: Check data structure
         *------------------------*/
         
         /* we restart with length m + trlen */
         mstepsi   = msteps + trlen;
         
         if(mstepsi > maxsteps)
         {
            maxsteps_new = maxsteps;
            while(mstepsi > maxsteps_new)
            {
               maxsteps_new += expand;
            }
            
            maxsteps_new = (int)SchurchebMin((SCHURCHEB_long)maxsteps_new, n_global);
         
            /* update V_data, H_data, and work_int when needed */
            if(maxsteps_new > maxsteps)
            {
               /* expand needed */
               
               V_data_new = std::move(V_data);
               V_new = std::move(V);
               H_data_new = std::move(H_data);
               H_new = std::move(H);
               
               V_data.Setup( n_local, maxsteps_new+1, location, true);
               V.Setup( n_local, maxsteps_new+1, location, true);
               H_data.Setup( maxsteps_new+1, maxsteps_new, kMemoryHost, true);
               H.Setup( maxsteps_new+1, maxsteps_new, kMemoryHost, true);
               
               SCHURCHEB_MEMCPY( V_data.GetData(), V_data_new.GetData(), n_local*maxsteps, location, location, DataType);
               SCHURCHEB_MEMCPY( V.GetData(), V_new.GetData(), n_local*maxsteps, location, location, DataType);
               /* copy H */
               for(i = 0 ; i < maxsteps ; i ++)
               {
                  for(j = 0 ; j < maxsteps+1 ; j ++)
                  {
                     H_data(j, i) = H_data_new(j, i);
                     H(j, i) = H_new(j, i);
                  }
               }
               
               work_int.Resize( 2*maxsteps_new, false, false);
               icov.Resize( maxsteps_new, false, false);
               iicov.Resize( maxsteps_new, false, false);
               /* the distance */
               dcov.Resize( maxsteps_new, false, false);
               dicov.Resize( maxsteps_new, false, false);
               dall.Resize( maxsteps_new, false, false);
               
               
               maxsteps = maxsteps_new;
               V_data_new.Clear();
               V_new.Clear();
               H_data_new.Clear();
               H_new.Clear();
            }
            
         }
         
         /*------------------------ 
         * 2.1: Apply Arnoldi
         *------------------------*/
         
         /* compute arnoldi starting from the trlen
          * |<----- trlen ----->|<----- new ----->|
          * |<-cov->|<-- tr  -->|<----- new ----->|
          * since eig_tol can be large, we do not
          * use "locking" in this algorithm
          */
         m = SchurchebArnoldiNoRestart<VectorType>( A, trlen, mstepsi, V, H, tol_orth, tol_reorth, nmvs_loc);

         /* record total matvec applied */
         nmvs += nmvs_loc;
         /* record the last entry */
         h_last = H(m, m-1);
         h_last_c = h_last;
         h_last_r = SchurchebAbs(h_last);
         
         /* check if H(m, m-1) is "zero" 
          * if too small, restart with new vector
          * otherwise keep going
          */
         if(SchurchebAbs(h_last) < tol_orth)
         {
            
            /* In this case, we restart with a new vector
             * If failed, stop.
             */
            
            /* reset to exact 0 */
            h_last_c = ComplexDataType(0.0);
            h_last_r = RealDataType(0.0);
            /* if H(m, m-1) is zero, and this is not the last loop, we find another vector to restart with */
            if(m < rank && its < maxits && SchurchebArnoldiThickRestartBuildThickRestartNewVector(V, H, m, tol_orth, tol_reorth, v) >= 0)
            {
               /* in this case, we can restart
                * and we've built the new vector
                * restart with length m
                */
               trlen = m;
               continue;
            }
            
            /* if we reach here, we can't find the new vector, stop here
             * we still need to apply a Schur Decomposition on H, copy to Vm and Hm
             */
            
            Vm.SetupPtr(V_data, 0, 0, n_local, m);
            Hm.SetupPtr(H_data, 0, 0, m, m);
            
            /* copy data from V to Vm */
            SCHURCHEB_MEMCPY(Vm.GetData(), V.GetData(), n_local*m, location, location, DataType);
            
            /* copy data from H to Hm */
            for (i = 0; i < m; i++) 
            {
               for (j = 0; j < m; j++)
               {
                  Hm(j,i) = H(j,i);
               }
            }
            
            /* Compute the schur decomposition Hm = QhsUQhs^H 
             * note that in the real case, there might be diagonal 2*2 blocks
             */
            if(SchurchebIsComplex<DataType>::value)
            {
               Hm_c->Schur(*Qhs_c, w);
            }
            else
            {
               Hm_r->Schur(*Qhs_r, wr, wi);
            }
            
            if(truncate <= 0.0 && m <= rank && rank <= rank2)
            {
               /* In this case, we return all the eigenvalues
                * All converged, no truncation, keep all
                */
               nsatis = m;
               ncov = m;
               nicov = 0;
               icov.Resize(m, false, false);
               icov.UnitPerm();
            }
            else
            {
               /* In this case, we need to select eigenvalues */
               if(SchurchebIsComplex<DataType>::value)
               {
                  SchurchebArnoldiThickRestartChooseEigenValuesComplex(*Hm_c, *Qhs_c, zero_c, weight, truncate, ncov, nicov, nsatis, tol_eig, cut, w, icov, iicov, work_int, dcov, dicov, dall);
               }
               else
               {
                  SchurchebArnoldiThickRestartChooseEigenValuesReal(*Hm_r, *Qhs_r, zero_r, weight, truncate, ncov, nicov, nsatis, tol_eig, cut, wr, wi, icov, iicov, work_int, dcov, dicov, dall);
               }
            }
            
            /* go to build V and H */
            break;
            
         }
         
         /*--------------------------
         * 2.2: Compute decomposition
         *--------------------------*/
         
         /* if we reach here, H(m, m-1) is not zero */
         
         /* get Vm and Hm */
         Vm.SetupPtr(V_data, 0, 0, n_local, m);
         Hm.SetupPtr(H_data, 0, 0, m, m);
         
         /* copy data from V to Vm */
         SCHURCHEB_MEMCPY(Vm.GetData(), V.GetData(), n_local*m, location, location, DataType);
         
         for (i = 0; i < m; i++) 
         {
            for (j = 0; j < m; j++)
            {
               Hm(j,i) = H(j,i);
            }
         }
         
         /* Compute the schur decomposition Hm = QhsUQhs^H 
          * note that in the real case, there might be diagonal 2*2 blocks
          */
         if(SchurchebIsComplex<DataType>::value)
         {
            Hm_c->Schur(*Qhs_c, w);
         }
         else
         {
            Hm_r->Schur(*Qhs_r, wr, wi);
         }
         
         /*------------------------ 
          * 2.3: Check convergence
          *------------------------*/
         
         /* Now check convergence based on the eigen decomposition */
         if(SchurchebIsComplex<DataType>::value)
         {
            SchurchebArnoldiThickRestartChooseEigenValuesComplex(*Hm_c, *Qhs_c, h_last_c, weight, truncate, ncov, nicov, nsatis, tol_eig, cut, w, icov, iicov, work_int, dcov, dicov, dall);
         }
         else
         {
            SchurchebArnoldiThickRestartChooseEigenValuesReal(*Hm_r, *Qhs_r, h_last_r, weight, truncate, ncov, nicov, nsatis, tol_eig, cut, wr, wi, icov, iicov, work_int, dcov, dicov, dall);
         }
         
         if(nsatis >= rank || its == maxits || cut)
         {
            /* we've got enough, or this is the last loop, stop */
            //SCHURCHEB_PRINT_DEBUG(parallel_log::_grank, 0, "Thick restart arnoldi breat with cut: %s\n", (cut) ? "true" : "false");
            break;
         }
         
         /* if we reach here, we still have the next loop, prepare restart */
         
         /*------------------------ 
          * 2.4: Prepare restart
          *------------------------*/
         
         if(nicov > 0)
         {
            /* Case 1: Not enough, have unconvergenced eigenvalues.
             * Prepare for the thick restart with cov + icov*tr_factor */
            
            /* first pick number of eigevalues, we want to restart with at least one vector 
             * we first need to check if we have extra size
             */
            npick = SchurchebMax((int)(nicov * tr_fact), 1);
            npick = SchurchebMin(npick, nicov);
            
            /* buld thick restart */
            if(SchurchebIsComplex<DataType>::value)
            {
               SchurchebArnoldiThickRestartBuildThickRestartNoLockComplex(*Vm_c, *Hm_c, *Qhs_c, h_last_c, ncov, nicov, npick, icov, iicov, dicov, work_int, w, *V_c, *H_c);
            }
            else
            {
               SchurchebArnoldiThickRestartBuildThickRestartNoLockReal(*Vm_r, *Hm_r, *Qhs_r, h_last_r, ncov, nicov, npick, icov, iicov, dicov, work_int, wr, wi, *V_r, *H_r);
            }
            
            /* npick might be updated in the loop */
            trlen = ncov + npick;
            
         }
         else
         {
            /* In this case, no inconvergenced eigs, but we haven't got enough, just keep doing Arnoldi */
            trlen = m;
            /* if h_last is too small, we need to restart with new vector */
            if( SchurchebAbs(h_last) < tol_orth && SchurchebArnoldiThickRestartBuildThickRestartNewVector(V, H, m, tol_orth, tol_reorth, v) < 0)
            {
               //SCHURCHEB_PRINT("Thick restart can't add more eigenvectors.\n");
               /* ncov already computed */
               break;
            }
         }
         
      }
      
      /* done here, return all the convergenced result */
      //SCHURCHEB_PRINT_DEBUG(parallel_log::_grank, 0, "Thick restart arnoldi got %d eig convergenced, %d kept, with %d numits\n", ncov, nsatis, its);
      if(SchurchebIsComplex<DataType>::value)
      {
         SchurchebArnoldiThickRestartBuildResultComplex(*Vm_c, *Hm_c, *Qhs_c, ncov, rank2, icov, dcov, work_int, w, *V_c, *H_c);
      }
      else
      {
         SchurchebArnoldiThickRestartBuildResultReal(*Vm_r, *Hm_r, *Qhs_r, ncov, rank2, icov, dcov, work_int, wr, wi, *V_r, *H_r);
      }
      
      /* Deallocate */
      V_data.Clear();
      H_data.Clear();
      V_temp.Clear();
      Vm_temp.Clear();
      Q_temp.Clear();
      work_int.Clear();
      icov.Clear();
      iicov.Clear();
      dcov.Clear();
      dicov.Clear();
      dall.Clear();
      wr.Clear();
      wi.Clear();
      
      return ncov;
   }
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_float>( arnoldimatrix_seq_float &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_double>( arnoldimatrix_seq_double &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_float>( arnoldimatrix_par_float &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_double>( arnoldimatrix_par_double &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexs>( arnoldimatrix_seq_complexs &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexd>( arnoldimatrix_seq_complexd &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexs>( arnoldimatrix_par_complexs &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexd>( arnoldimatrix_par_complexd &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_float>( matrix_csr_float &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_double>( matrix_csr_double &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_float>( matrix_csr_par_float &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_float &V, matrix_dense_float &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_double>( matrix_csr_par_double &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_double &V, matrix_dense_double &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexs>( matrix_csr_complexs &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_seq_complexd>( matrix_csr_complexd &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexs>( matrix_csr_par_complexs &A, int msteps, int maxits, int rank, int rank2, float truncate, float tr_fact, float tol_eig, float (*weight)(complexs), matrix_dense_complexs &V, matrix_dense_complexs &H, float tol_orth, float tol_reorth, int &nmvs);
   template int SchurchebArnoldiThickRestartNoLock<vector_par_complexd>( matrix_csr_par_complexd &A, int msteps, int maxits, int rank, int rank2, double truncate, double tr_fact, double tol_eig, double (*weight)(complexd), matrix_dense_complexd &V, matrix_dense_complexd &H, double tol_orth, double tol_reorth, int &nmvs);
   
   template <class VectorType, typename DataType, typename RealDataType>
   int SchurchebCgs2( VectorType &w, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, 
                  RealDataType &t, int k, RealDataType tol_orth)
   {
      
      if(k < 0)
      {
         /* in this case, we don't have any previous vectors, return immediatly */
         return SCHURCHEB_SUCCESS;
      }
      
      int np, myid;
      MPI_Comm comm;
      w.GetMpiInfo(np, myid, comm);
      
      /*------------------------
       * 1: Modified Gram-Schmidt
       *------------------------*/
      
      int               i, n_local, err = 0;
      DataType          t1, one, zero;
      
      one = 1.0;
      zero = 0.0;
      
      VectorType        v;
      v.SetupPtrStr(w);
      
      n_local = w.GetLengthLocal();
      
      SequentialVectorClass<DataType>  v_local, w_local;
      SequentialVectorClass<DataType>  s;
      
      s.Setup(k+1);
      
      if(n_local > 0)
      {
         w_local.SetupPtr( w.GetData(), n_local, w.GetDataLocation() );
      }
      
      /* H(1:i,i) = V(:,1:i)'*V(:,i1)
       * V(:,i1) = V(:,i1) - V(:,1:i)*H(1:i,i)
       */
      
      for( i = 0 ; i <= k ; i ++)
      {
         /* inner produce and update H, w */
         if(n_local > 0)
         {
            v_local.SetupPtr( &V(0, i), n_local, V.GetDataLocation() );
            err = v_local.Dot(w_local, t1); SCHURCHEB_CHKERR(err);
            H(i, k) = t1;
         }
         else
         {
            H(i, k) = zero;
         }
      }
      
      if(w.IsParallel())
      {
         SCHURCHEB_MPI_CALL( SchurchebMpiAllreduceInplace( &H(0, k), k+1, MPI_SUM, comm) );
      }
      
      for( i = 0 ; i <= k ; i ++)
      {
         if(n_local > 0)
         {
            v.UpdatePtr( &V(0, i), V.GetDataLocation() );
         }
         err = w.Axpy(-H(i, k), v); SCHURCHEB_CHKERR(err);
      }
      
      /* re-orth
       * s = V(:,1:i)'*V(:,i1)
       * V(:,i1) = V(:,i1) - V(:,1:i)*s
       * H(1:i,i) = H(1:i,i) + s;
       */
      
      for( i = 0 ; i <= k ; i ++)
      {
         /* inner produce and update H, w */
         if(n_local > 0)
         {
            v_local.SetupPtr( &V(0, i), n_local, V.GetDataLocation() );
            err = v_local.Dot(w_local, t1); SCHURCHEB_CHKERR(err);
            s[i] = t1;
         }
         else
         {
            s[i] = zero;
         }
      }
      
      if(w.IsParallel())
      {
         SCHURCHEB_MPI_CALL( SchurchebMpiAllreduceInplace( s.GetData(), k+1, MPI_SUM, comm) );
      }
      
      for( i = 0 ; i <= k ; i ++)
      {
         if(n_local > 0)
         {
            v.UpdatePtr( &V(0, i), V.GetDataLocation() );
         }
         err = w.Axpy(-s[i], v); SCHURCHEB_CHKERR(err);
         H(i, k) = H(i, k) + s[i];
      }
      
      /* Compute ||w|| */
      err = w.Norm2(t); SCHURCHEB_CHKERR(err);
      
      H(k+1,k) = t;
      
      /* scale w in this function */
      err = w.Scale(one/t); SCHURCHEB_CHKERR(err);
      
      v.Clear();
      s.Clear();
      
      return err;

   }
   template int SchurchebCgs2( SequentialVectorClass<float> &w, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, 
                  float &t, int k, float tol_orth);
   template int SchurchebCgs2( SequentialVectorClass<double> &w, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, 
                  double &t, int k, double tol_orth);
   template int SchurchebCgs2( SequentialVectorClass<complexs> &w, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, 
                  float &t, int k, float tol_orth);
   template int SchurchebCgs2( SequentialVectorClass<complexd> &w, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, 
                  double &t, int k, double tol_orth);
   template int SchurchebCgs2( ParallelVectorClass<float> &w, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, 
                  float &t, int k, float tol_orth);
   template int SchurchebCgs2( ParallelVectorClass<double> &w, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, 
                  double &t, int k, double tol_orth);
   template int SchurchebCgs2( ParallelVectorClass<complexs> &w, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, 
                  float &t, int k, float tol_orth);
   template int SchurchebCgs2( ParallelVectorClass<complexd> &w, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, 
                  double &t, int k, double tol_orth);
   
   template <class VectorType, typename DataType, typename RealDataType>
   int SchurchebMgs( VectorType &w, DenseMatrixClass<DataType> &V, DenseMatrixClass<DataType> &H, 
                  RealDataType &t, int k, RealDataType tol_orth, RealDataType tol_reorth)
   {
      
      if(k < 0)
      {
         /* in this case, we don't have any previous vectors, return immediatly */
         return SCHURCHEB_SUCCESS;
      }
      
      /*------------------------
       * 1: Modified Gram-Schmidt
       *------------------------*/
      
      int               i, n_local, err = 0;
      DataType          t1;
      RealDataType      normw;
      
      VectorType        v;
      v.SetupPtrStr(w);
      
      n_local = w.GetLengthLocal();
      
      /* compute initial ||w|| if we need to reorth */
      if(tol_reorth > 0.0)
      {
         err = w.Norm2(normw); SCHURCHEB_CHKERR(err);
      }
      else
      {
         normw = 0.0;
      }
      
      for( i = 0 ; i <= k ; i ++)
      {
         /* inner produce and update H, w */
         if(n_local > 0)
         {
            v.UpdatePtr( &V(0, i), V.GetDataLocation() );
         }
         err = v.Dot(w, t1); SCHURCHEB_CHKERR(err);
         H(i, k) = t1;
         err = w.Axpy(-t1, v); SCHURCHEB_CHKERR(err);
      }
      /* Compute ||w|| */
      err = w.Norm2(t); SCHURCHEB_CHKERR(err);
      
      /*------------------------
       * 2: Re-orth step
       *------------------------*/
      
      /* t < tol_orth is considered be lucky breakdown */
      while(t < normw * tol_reorth && t >= tol_orth)
      {
         normw = t;
         /* Re-orth */
         for (i = 0; i <= k; i++) 
         {
            if(n_local > 0)
            {
               v.UpdatePtr( &V(0, i), V.GetDataLocation() );
            }
            err = v.Dot(w, t1); SCHURCHEB_CHKERR(err);
            H(i, k) += t1;
            err = w.Axpy(-t1,v); SCHURCHEB_CHKERR(err);
         }
         /* Compute ||w|| */
         err = w.Norm2(t); SCHURCHEB_CHKERR(err);
         
      }
      H(k+1,k) = t;
      
      /* scale w in this function */
      err = w.Scale(DataType(1.0)/t); SCHURCHEB_CHKERR(err);
      
      v.Clear();
      
      return err;

   }
   template int SchurchebMgs( SequentialVectorClass<float> &w, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, 
                  float &t, int k, float tol_orth, float tol_reorth);
   template int SchurchebMgs( SequentialVectorClass<double> &w, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, 
                  double &t, int k, double tol_orth, double tol_reorth);
   template int SchurchebMgs( SequentialVectorClass<complexs> &w, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, 
                  float &t, int k, float tol_orth, float tol_reorth);
   template int SchurchebMgs( SequentialVectorClass<complexd> &w, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, 
                  double &t, int k, double tol_orth, double tol_reorth);
   template int SchurchebMgs( ParallelVectorClass<float> &w, DenseMatrixClass<float> &V, DenseMatrixClass<float> &H, 
                  float &t, int k, float tol_orth, float tol_reorth);
   template int SchurchebMgs( ParallelVectorClass<double> &w, DenseMatrixClass<double> &V, DenseMatrixClass<double> &H, 
                  double &t, int k, double tol_orth, double tol_reorth);
   template int SchurchebMgs( ParallelVectorClass<complexs> &w, DenseMatrixClass<complexs> &V, DenseMatrixClass<complexs> &H, 
                  float &t, int k, float tol_orth, float tol_reorth);
   template int SchurchebMgs( ParallelVectorClass<complexd> &w, DenseMatrixClass<complexd> &V, DenseMatrixClass<complexd> &H, 
                  double &t, int k, double tol_orth, double tol_reorth);
   
   template <class VectorType, typename DataType, typename RealDataType>
   int SchurchebOrthogonal( VectorType &w, DenseMatrixClass<DataType> &V, 
                  RealDataType &t, int k, RealDataType tol_orth, RealDataType tol_reorth)
   {
      
      if(k < 0)
      {
         /* in this case, we don't have any previous vectors, return immediatly */
      }
      
      /*------------------------
       * 1: Modified Gram-Schmidt
       *------------------------*/
      
      int               i, n_local, err;
      DataType          t1;
      RealDataType      normw;
      
      VectorType        v;
      v.SetupPtrStr(w);
      
      n_local = w.GetLengthLocal();
      
      /* compute initial ||w|| if we need to reorth */
      if(tol_reorth > 0.0)
      {
         err = w.Norm2(normw); SCHURCHEB_CHKERR(err);
      }
      else
      {
         normw = 0.0;
      }
      
      for( i = 0 ; i <= k ; i ++)
      {
         /* inner produce and update w */
         if(n_local > 0)
         {
            v.UpdatePtr( &V(0, i), V.GetDataLocation() );
         }
         err = v.Dot(w, t1); SCHURCHEB_CHKERR(err);
         err = w.Axpy(-t1, v); SCHURCHEB_CHKERR(err);
         
      }
      /* Compute ||w|| */
      err = w.Norm2(t); SCHURCHEB_CHKERR(err);
      
      /*------------------------
       * 2: Re-orth step
       *------------------------*/
      
      /* t < tol_orth is considered be lucky breakdown */
      while(t < normw * tol_reorth && t >= tol_orth)
      {
         normw = t;
         /* Re-orth */
         for (i = 0; i <= k; i++) 
         {
            if(n_local > 0)
            {
               v.UpdatePtr( &V(0, i), V.GetDataLocation() );
            }
            err = v.Dot(w, t1); SCHURCHEB_CHKERR(err);
            err = w.Axpy(-t1,v); SCHURCHEB_CHKERR(err);
         }
         /* Compute ||w|| */
         err = w.Norm2(t); SCHURCHEB_CHKERR(err);
         
      }
      
      /* scale w in this function */
      err = w.Scale(DataType(1.0)/t); SCHURCHEB_CHKERR(err);
      
      v.Clear();
      
      return err;

   }
   template int SchurchebOrthogonal( SequentialVectorClass<float> &w, DenseMatrixClass<float> &V, 
                  float &t, int k, float tol_orth, float tol_reorth);
   template int SchurchebOrthogonal( SequentialVectorClass<double> &w, DenseMatrixClass<double> &V, 
                  double &t, int k, double tol_orth, double tol_reorth);
   template int SchurchebOrthogonal( SequentialVectorClass<complexs> &w, DenseMatrixClass<complexs> &V, 
                  float &t, int k, float tol_orth, float tol_reorth);
   template int SchurchebOrthogonal( SequentialVectorClass<complexd> &w, DenseMatrixClass<complexd> &V, 
                  double &t, int k, double tol_orth, double tol_reorth);
   template int SchurchebOrthogonal( ParallelVectorClass<float> &w, DenseMatrixClass<float> &V, 
                  float &t, int k, float tol_orth, float tol_reorth);
   template int SchurchebOrthogonal( ParallelVectorClass<double> &w, DenseMatrixClass<double> &V, 
                  double &t, int k, double tol_orth, double tol_reorth);
   template int SchurchebOrthogonal( ParallelVectorClass<complexs> &w, DenseMatrixClass<complexs> &V, 
                  float &t, int k, float tol_orth, float tol_reorth);
   template int SchurchebOrthogonal( ParallelVectorClass<complexd> &w, DenseMatrixClass<complexd> &V, 
                  double &t, int k, double tol_orth, double tol_reorth);
   
   template <class VectorType, typename DataType, typename RealDataType>
   int SchurchebOrthogonal( DenseMatrixClass<DataType> &V, VectorType &v, DenseMatrixClass<DataType> &W, int &n, RealDataType tol_orth)
   {
      
      int                        m, i, n_local, err = 0;
      double                     t, one;
      DenseMatrixClass<DataType> H;
      VectorType                 w;
      
      if(parallel_log::_grank == 0)
      {
         cout<<"Using standard orth"<<endl;
      }
      
      /*------------------------
       * 0: Prepare
       *------------------------*/
      
      m = V.GetNumColsLocal();
      w.SetupPtrStr(v);
      n_local = w.GetLengthLocal();
      H.Setup( m+1, m, false);
      one = 1.0;
      
      /*------------------------
       * 1: Classical Gram-Schmidt
       *------------------------*/
      
      n = 0;
      for( i = 0 ; i < m ; i ++)
      {
         /* update pointer */
         if(n_local > 0)
         {
            v.UpdatePtr( &V(0, i), V.GetDataLocation() );
            w.UpdatePtr( &W(0, n), W.GetDataLocation() );
         }
         /* Copy current V into W */
         SCHURCHEB_MEMCPY( w.GetData(), v.GetData(), n_local, V.GetDataLocation(), W.GetDataLocation(), double );
         
         /* re-orth aganist all previous in V */
         if(n > 0)
         {
            err = SchurchebCgs2( w, W, H, t, n-1, tol_orth); SCHURCHEB_CHKERR(err);
            
            if( t >= tol_orth)
            {
               /* in this case, the result is large enough, keep it */
               n++;
            }
         }
         else
         {
            /* this is the first */
            w.Norm2(t);
            
            if( t >= tol_orth)
            {
               /* in this case, the result is large enough, keep it */
               w.Scale(one/t);
               n++;
            }
            
         }
         
         if(n >= m)
         {
            /* in this case we have enough, no need to proceed */
            break;
         }
         
      }
      
      return err;

   }
   template int SchurchebOrthogonal( DenseMatrixClass<double> &V, vector_par_double &v, 
                              DenseMatrixClass<double> &W, int &n, double tol_orth);

   template <class VectorType, typename DataType, typename RealDataType>
   int SchurchebQR( DenseMatrixClass<DataType> &A, VectorType &v, DenseMatrixClass<DataType> &Q, DenseMatrixClass<DataType> &R, 
                  int &m, bool keepR, RealDataType tol_orth)
   {
      
      if (m <= 0)
      {
         /* empty */
         return SCHURCHEB_SUCCESS;
      }
      
      int                        m1, i, j, n_local, err = 0;
      long int                   n;
      double                     t, one, normw, zero;
      DenseMatrixClass<DataType> H;
      VectorType                 w;
      
      /*------------------------
       * 0: Prepare
       *------------------------*/
      
      n = v.GetLengthGlobal();
      
      if(n < m)
      {
         /* too long! */
         cout<<n<<" "<<m<<endl;
         SCHURCHEB_ERROR("Only works for tall-thin matrices.");
         return SCHURCHEB_ERROR_INVALED_PARAM;
      }
      
      w.SetupPtrStr(v);
      n_local = w.GetLengthLocal();
      if(keepR)
      {
         H.Setup( m, m-1);
      }
      else
      {
         if(R.GetNumRowsLocal()*R.GetNumColsLocal() > m*(m-1))
         {
            H.SetupPtr( R, 0, 0, m, m-1);
         }
         else
         {
            R.Setup(m, m-1);
            H.SetupPtr( R, 0, 0, m, m-1);
         }
      }
      one = 1.0;
      zero = 0.0;
      
      /*------------------------
       * 1: Classical Gram-Schmidt
       *------------------------*/
      
      if(Q.GetNumRowsLocal() != n_local || Q.GetNumColsLocal() < m)
      {
         Q.Setup(n_local, m);
      }
      
      /* frist normalize the first column */
      m1 = 0;
      
      for( i = 0 ; i < m ; i ++)
      {
         /* update pointer */
         if(n_local > 0)
         {
            v.UpdatePtr( &A(0, i), A.GetDataLocation() );
            w.UpdatePtr( &Q(0, m1), Q.GetDataLocation() );
         }
         /* Copy current V into W */
         SCHURCHEB_MEMCPY( w.GetData(), v.GetData(), n_local, Q.GetDataLocation(), A.GetDataLocation(), double );
         
         /* re-orth aganist all previous in V */
         if(m1 > 0)
         {
            err = SchurchebCgs2( w, Q, H, t, m1-1, tol_orth); SCHURCHEB_CHKERR(err);
            
            if( t >= tol_orth)
            {
               /* in this case, the result is large enough, keep it */
               m1++;
            }
         }
         else
         {
            /* this is the first */
            w.Norm2(normw);
            
            if( normw >= tol_orth)
            {
               /* in this case, the result is large enough, keep it */
               w.Scale(one/normw);
               m1++;
            }
            
         }
      }
      
      m = m1;
      
      if(keepR && m1 > 0)
      {
         if(R.GetNumRowsLocal() != m1 || R.GetNumColsLocal() != m1 )
         {
            R.Setup(m1, m1, true);
         }
         else
         {
            R.Fill(zero);
         }
         
         R(0,0) = normw;
         
         for( i = 1 ; i < m1 ; i ++)
         {
            for(j = 0 ; j <= i ; j ++)
            {
               R(j, i) = H(j, i-1);
            }
         }
         
      }
      
      H.Clear();
      
      return err;

   }
   template int SchurchebQR( DenseMatrixClass<double> &A, vector_par_double &v, 
                        DenseMatrixClass<double> &Q, DenseMatrixClass<double> &R, 
                        int &m, bool keepR, double tol_orth);

   template <class VectorType, typename DataType, typename RealDataType>
   int SchurchebBlockOrthogonal( DenseMatrixClass<DataType> &V, VectorType &v, DenseMatrixClass<DataType> &W, int nB, int &n, RealDataType tol_orth)
   {
      int m, m1, k1, k2, n_local;
      matrix_dense_double R, R_temp, Q, Q1, Qi, Vi, Wi, Wi2, Ri, Rg;
      double one, zero, mone;
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      int np, myid;
      MPI_Comm comm;
      v.GetMpiInfo(np, myid, comm);
      
      if(myid == 0)
      {
         cout<<"Using block orth with block size "<<nB<<endl;
      }
      
      m = V.GetNumColsLocal();
      n_local = V.GetNumRowsLocal();
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      R.Setup(m, m);
      
      k1 = m / nB;
      k2 = m % nB;
      
      R_temp.Setup(nB, nB);
      Q.Setup(n_local, nB);
      Q1.Setup(n_local, nB);
      
      /* standard blocks */
      if( k1 > 0)
      {
         m1 = nB;
         SchurchebQR( V, v, W, R_temp, m1, false, tol_orth);
      }
      else
      {
         m1 = k2;
         SchurchebQR( V, v, W, R_temp, m1, false, tol_orth);
      
         return SCHURCHEB_SUCCESS;
      }
      
      n = m1;
      
      for(int i = 1 ; i < k1 ; i ++)
      {
         Wi2.SetupPtr(W,0,0,n_local,n);
         Vi.SetupPtr(V,0,i*nB,n_local,nB);
         Wi.SetupPtr(W,0,n,n_local,nB);
         
         DenseMatrixMatMat( one, Wi2, 'T', Vi, 'N', zero, Ri);
         Rg.Setup(Ri.GetNumRowsLocal(), Ri.GetNumColsLocal());
         SchurchebMpiAllreduce( &Ri(0,0), &Rg(0,0), Ri.GetNumNonzeros(), MPI_SUM, comm);
         
         //R(b1:k-1,bk)=Q(:,b1:k-1)'*A(:,bk);

         /* next, a re-orth like approach */
         SCHURCHEB_MEMCPY( &Q(0,0), &Vi(0,0), n_local*nB, Q.GetDataLocation(), Vi.GetDataLocation(), double );
         DenseMatrixMatMat( mone, Wi2, 'N', Rg, 'N', one, Q);
         
         //W = A(:,bk)-Q(:,b1:k-1)*R(b1:k-1,bk);
         
         DenseMatrixMatMat( one, Wi2, 'T', Q, 'N', zero, Ri);
         Rg.Setup(Ri.GetNumRowsLocal(), Ri.GetNumColsLocal());
         SchurchebMpiAllreduce( &Ri(0,0), &Rg(0,0), Ri.GetNumNonzeros(), MPI_SUM, comm);
           
         //RT=Q(:,b1:k-1)'*W;
         SCHURCHEB_MEMCPY( &Q1(0,0), &Q(0,0), n_local*nB, Q1.GetDataLocation(), Q.GetDataLocation(), double );
         DenseMatrixMatMat( mone, Wi2, 'N', Rg, 'N', one, Q1);
         //W = W - Q(:,b1:k-1)*RT;
         
         m1 = nB;
         SchurchebQR( Q1, v, Wi, R_temp, m1, false, tol_orth);
         
         //[Q(:,bk),R(bk,bk)] = qr(W,0);
         
         n += m1;
         
      }
      
      if(k2 > 0)
      {
         /* last block of different size */
         
         Q.Setup(n_local, k2);
         Q1.Setup(n_local, k2);
         
         Wi2.SetupPtr(W,0,0,n_local,n);
         Vi.SetupPtr(V,0,k1*nB,n_local,k2);
         Wi.SetupPtr(W,0,n,n_local,k2);
         
         DenseMatrixMatMat( one, Wi2, 'T', Vi, 'N', zero, Ri);
         Rg.Setup(Ri.GetNumRowsLocal(), Ri.GetNumColsLocal());
         SchurchebMpiAllreduce( &Ri(0,0), &Rg(0,0), Ri.GetNumNonzeros(), MPI_SUM, comm);
         
         //R(b1:k-1,bk)=Q(:,b1:k-1)'*A(:,bk);

         /* next, a re-orth like approach */
         SCHURCHEB_MEMCPY( &Q(0,0), &Vi(0,0), n_local*k2, Q.GetDataLocation(), Vi.GetDataLocation(), double );
         DenseMatrixMatMat( mone, Wi2, 'N', Rg, 'N', one, Q);
         
         //W = A(:,bk)-Q(:,b1:k-1)*R(b1:k-1,bk);
         
         DenseMatrixMatMat( one, Wi2, 'T', Q, 'N', zero, Ri);
         Rg.Setup(Ri.GetNumRowsLocal(), Ri.GetNumColsLocal());
         SchurchebMpiAllreduce( &Ri(0,0), &Rg(0,0), Ri.GetNumNonzeros(), MPI_SUM, comm);
           
         //RT=Q(:,b1:k-1)'*W;
         SCHURCHEB_MEMCPY( &Q1(0,0), &Q(0,0), n_local*k2, Q1.GetDataLocation(), Q.GetDataLocation(), double );
         DenseMatrixMatMat( mone, Wi2, 'N', Rg, 'N', one, Q1);
         //W = W - Q(:,b1:k-1)*RT;
         
         m1 = k2;
         SchurchebQR( Q1, v, Wi, R_temp, m1, false, tol_orth);
         
         //[Q(:,bk),R(bk,bk)] = qr(W,0);
         
         n += m1;
         
      }
      
      return SCHURCHEB_SUCCESS;
      
   }
   template int SchurchebBlockOrthogonal( DenseMatrixClass<double> &V, vector_par_double &v, 
                                 DenseMatrixClass<double> &W, int nB, int &n, double tol_orth);

   template <class VectorType, typename DataType, typename RealDataType>
   int SchurchebCholOrthogonal( DenseMatrixClass<DataType> &V, VectorType &v, DenseMatrixClass<DataType> &W, int &n, RealDataType tol_orth)
   {
      
      // VV = V'*V;
      // Vc = chol(VV);
      // WW = V*inv(Vc);
      // W = WW(i) s.t. Vc(i,i) > n*eps
      
      int m, n_local;
      matrix_dense_double VV_local, VV, Vc, iVc, WW;
      double one, zero;
      one = 1.0;
      zero = 0.0;
      
      m = V.GetNumColsLocal();
      n_local = V.GetNumRowsLocal();
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      int np, myid;
      MPI_Comm comm;
      v.GetMpiInfo(np, myid, comm);
      
      if(myid == 0)
      {
         cout<<"Using chol orth"<<endl;
      }
      
      VV_local.MatMat( one, V, 'T', V, 'N', zero);
      VV.Setup(m, m, kMemoryHost, true);
      Vc.Setup(m, m, kMemoryHost, true);
      
      SchurchebMpiAllreduce( VV_local.GetData(), VV.GetData(), m*m, MPI_SUM, comm);
      
      char uplo = 'U';
      int info;
      
      /* compute the chol factorization */
      SCHURCHEB_BLASLAPACK_DPOTRF(	&uplo, &m, VV.GetData(), &m, &info);
      for(int i = 0 ; i < m ; i ++)
      {
         for(int j = 0 ; j <= i ; j ++)
         {
            Vc(j,i) = VV(j,i);
         }
      }
      
      iVc = Vc;
      iVc.InvertUpperTriangular();
      
      WW.MatMat( one, V, 'N', iVc, 'N', zero);
      
      //double tol = v.GetLengthGlobal()*std::numeric_limits<double>::epsilon();
      
      n = 0;
      for(int i = 0 ; i < m ; i ++)
      {
         if(SchurchebAbs(Vc(i,i)) > tol_orth)
         {
            SCHURCHEB_MEMCPY( &W(0,n), &WW(0,i), n_local, W.GetDataLocation(), V.GetDataLocation(), double );
            n ++;
         }
      }
      
      return SCHURCHEB_SUCCESS;
      
   }
   template int SchurchebCholOrthogonal( DenseMatrixClass<double> &V, vector_par_double &v, 
                                 DenseMatrixClass<double> &W, int &n, double tol_orth);

   template <class VectorType, typename DataType, typename RealDataType>
   int SchurchebCholOrthogonal2( DenseMatrixClass<DataType> &V, VectorType &v, DenseMatrixClass<DataType> &W, int &n, RealDataType tol_orth)
   {
      
      // VV = V'*V;
      // Vc = chol(VV);
      // WW = V*inv(Vc);
      // W = WW(i) s.t. Vc(i,i) > n*eps
      
      int m, n_local;
      matrix_dense_double VV_local, VV, Vc, iVc, WW;
      matrix_dense_double V_temp;
      double one, zero;
      one = 1.0;
      zero = 0.0;
      
      m = n;
      n_local = V.GetNumRowsLocal();
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      int np, myid;
      MPI_Comm comm;
      v.GetMpiInfo(np, myid, comm);
      
      V_temp.SetupPtr( V, 0, 0, n_local, m);
      
      VV_local.MatMat( one, V_temp, 'T', V_temp, 'N', zero);
      VV.Setup(m, m, kMemoryHost, true);
      Vc.Setup(m, m, kMemoryHost, true);
      
      SchurchebMpiAllreduce( VV_local.GetData(), VV.GetData(), m*m, MPI_SUM, comm);
      
      char uplo = 'U';
      int info;
      
      /* compute the chol factorization */
      SCHURCHEB_BLASLAPACK_DPOTRF(	&uplo, &m, VV.GetData(), &m, &info);
      
      for(int i = 0 ; i < m ; i ++)
      {
         for(int j = 0 ; j <= i ; j ++)
         {
            Vc(j,i) = VV(j,i);
         }
      }
      
      iVc = Vc;
      iVc.InvertUpperTriangular();
      
      WW.MatMat( one, V_temp, 'N', iVc, 'N', zero);
      
      //double tol = v.GetLengthGlobal()*std::numeric_limits<double>::epsilon();
      
      n = 0;
      for(int i = 0 ; i < m ; i ++)
      {
         if(SchurchebAbs(Vc(i,i)) > tol_orth)
         {
            SCHURCHEB_MEMCPY( &W(0,n), &WW(0,i), n_local, W.GetDataLocation(), WW.GetDataLocation(), double );
            n ++;
         }
      }
      
      return SCHURCHEB_SUCCESS;
      
   }
   template int SchurchebCholOrthogonal2( DenseMatrixClass<double> &V, vector_par_double &v, 
                                 DenseMatrixClass<double> &W, int &n, double tol_orth);

   template <class VectorType, typename DataType, typename RealDataType>
   int SchurchebBlockOrthogonal2( DenseMatrixClass<DataType> &V, VectorType &v, DenseMatrixClass<DataType> &W, int nB, int &n, RealDataType tol_orth)
   {
      int m, m1, k1, k2, n_local;
      matrix_dense_double R, R_temp, Q, Q1, Qi, Vi, Wi, Wi2, Ri, Rg;
      double one, zero, mone;
      one = 1.0;
      zero = 0.0;
      mone = -1.0;
      
      int np, myid;
      MPI_Comm comm;
      v.GetMpiInfo(np, myid, comm);
      
      if(myid == 0)
      {
         cout<<"Using chol block orth with block size "<<nB<<endl;
      }
      
      m = V.GetNumColsLocal();
      n_local = V.GetNumRowsLocal();
      
      if(m == 0)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      R.Setup(m, m);
      
      k1 = m / nB;
      k2 = m % nB;
      
      R_temp.Setup(nB, nB);
      Q.Setup(n_local, nB);
      Q1.Setup(n_local, nB);
      
      /* standard blocks */
      if( k1 > 0)
      {
         m1 = nB;
         SchurchebCholOrthogonal2( V, v, W, m1, tol_orth);
      }
      else
      {
         m1 = k2;
         SchurchebCholOrthogonal2( V, v, W, m1, tol_orth);
      
         return SCHURCHEB_SUCCESS;
      }
      
      n = m1;
      
      for(int i = 1 ; i < k1 ; i ++)
      {
         Wi2.SetupPtr(W,0,0,n_local,n);
         Vi.SetupPtr(V,0,i*nB,n_local,nB);
         Wi.SetupPtr(W,0,n,n_local,nB);
         
         DenseMatrixMatMat( one, Wi2, 'T', Vi, 'N', zero, Ri);
         Rg.Setup(Ri.GetNumRowsLocal(), Ri.GetNumColsLocal());
         SchurchebMpiAllreduce( &Ri(0,0), &Rg(0,0), Ri.GetNumNonzeros(), MPI_SUM, comm);
         
         //R(b1:k-1,bk)=Q(:,b1:k-1)'*A(:,bk);

         /* next, a re-orth like approach */
         SCHURCHEB_MEMCPY( &Q(0,0), &Vi(0,0), n_local*nB, Q.GetDataLocation(), Vi.GetDataLocation(), double );
         DenseMatrixMatMat( mone, Wi2, 'N', Rg, 'N', one, Q);
         
         //W = A(:,bk)-Q(:,b1:k-1)*R(b1:k-1,bk);
         
         DenseMatrixMatMat( one, Wi2, 'T', Q, 'N', zero, Ri);
         Rg.Setup(Ri.GetNumRowsLocal(), Ri.GetNumColsLocal());
         SchurchebMpiAllreduce( &Ri(0,0), &Rg(0,0), Ri.GetNumNonzeros(), MPI_SUM, comm);
           
         //RT=Q(:,b1:k-1)'*W;
         SCHURCHEB_MEMCPY( &Q1(0,0), &Q(0,0), n_local*nB, Q1.GetDataLocation(), Q.GetDataLocation(), double );
         DenseMatrixMatMat( mone, Wi2, 'N', Rg, 'N', one, Q1);
         //W = W - Q(:,b1:k-1)*RT;
         
         m1 = nB;
         //SchurchebQR( Q1, v, Wi, R_temp, m1, false, tol_orth);
         SchurchebCholOrthogonal2( Q1, v, Wi, m1, tol_orth);
         
         //[Q(:,bk),R(bk,bk)] = qr(W,0);
         
         n += m1;
         
      }
      
      if(k2 > 0)
      {
         /* last block of different size */
         
         Q.Setup(n_local, k2);
         Q1.Setup(n_local, k2);
         
         Wi2.SetupPtr(W,0,0,n_local,n);
         Vi.SetupPtr(V,0,k1*nB,n_local,k2);
         Wi.SetupPtr(W,0,n,n_local,k2);
         
         DenseMatrixMatMat( one, Wi2, 'T', Vi, 'N', zero, Ri);
         Rg.Setup(Ri.GetNumRowsLocal(), Ri.GetNumColsLocal());
         SchurchebMpiAllreduce( &Ri(0,0), &Rg(0,0), Ri.GetNumNonzeros(), MPI_SUM, comm);
         
         //R(b1:k-1,bk)=Q(:,b1:k-1)'*A(:,bk);

         /* next, a re-orth like approach */
         SCHURCHEB_MEMCPY( &Q(0,0), &Vi(0,0), n_local*k2, Q.GetDataLocation(), Vi.GetDataLocation(), double );
         DenseMatrixMatMat( mone, Wi2, 'N', Rg, 'N', one, Q);
         
         //W = A(:,bk)-Q(:,b1:k-1)*R(b1:k-1,bk);
         
         DenseMatrixMatMat( one, Wi2, 'T', Q, 'N', zero, Ri);
         Rg.Setup(Ri.GetNumRowsLocal(), Ri.GetNumColsLocal());
         SchurchebMpiAllreduce( &Ri(0,0), &Rg(0,0), Ri.GetNumNonzeros(), MPI_SUM, comm);
           
         //RT=Q(:,b1:k-1)'*W;
         SCHURCHEB_MEMCPY( &Q1(0,0), &Q(0,0), n_local*k2, Q1.GetDataLocation(), Q.GetDataLocation(), double );
         DenseMatrixMatMat( mone, Wi2, 'N', Rg, 'N', one, Q1);
         //W = W - Q(:,b1:k-1)*RT;
         
         m1 = k2;
         //SchurchebQR( Q1, v, Wi, R_temp, m1, false, tol_orth);
         SchurchebCholOrthogonal2( Q1, v, Wi, m1, tol_orth);
         
         //[Q(:,bk),R(bk,bk)] = qr(W,0);
         
         n += m1;
         
      }
      
      return SCHURCHEB_SUCCESS;
      
   }
   template int SchurchebBlockOrthogonal2( DenseMatrixClass<double> &V, vector_par_double &v, 
                                 DenseMatrixClass<double> &W, int nB, int &n, double tol_orth);

   template <typename T>
   int DenseMatrixPlotHost( DenseMatrixClass<T> &A, int conditiona, int conditionb, int width)
   {
      
      if(conditiona != conditionb)
      {
         return SCHURCHEB_SUCCESS;
      }
      
      std::cout<<"Ploting "<<A.GetNumRowsLocal()<<" by "<<A.GetNumColsLocal()<<" matrix."<<std::endl;
      
      int i, j, nrows, ncols;
      
      nrows = A.GetNumRowsLocal();
      ncols = A.GetNumColsLocal();
      
      for(i = 0 ; i < nrows ; i ++)
      {
         for(j = 0 ; j < ncols ; j ++)
         {
            SchurchebPrintValueHost(A(i,j), width);
            std::cout<<", ";
         }
         std::cout<<std::endl;
      }
      return SCHURCHEB_SUCCESS;
   }
   template int DenseMatrixPlotHost( DenseMatrixClass<float> &A, int conditiona, int conditionb, int width);
   template int DenseMatrixPlotHost( DenseMatrixClass<double> &A, int conditiona, int conditionb, int width);
   template int DenseMatrixPlotHost( DenseMatrixClass<complexs> &A, int conditiona, int conditionb, int width);
   template int DenseMatrixPlotHost( DenseMatrixClass<complexd> &A, int conditiona, int conditionb, int width);
   
   template <typename T>
   int CsrMatrixPlotHost( CsrMatrixClass<T> &A, int *perm, int conditiona, int conditionb, int width)
   {
      if(!A.IsHoldingData())
      {
         std::cout<<"Plot a Matrix not holding value."<<std::endl;
         return SCHURCHEB_ERROR_INVALED_OPTION;
      }
      
      if(!(A.IsCsr()))
      {
         std::cout<<"Plot only for CSR matrix, convert to csr."<<std::endl;
         A.Convert(true);
      }
      
      int   i, ii, j, j1, j2;
      int   *A_i = A.GetI();
      int   *A_j = A.GetJ();
      T     *A_data = A.GetData();
      
      vector_int marker;
      SequentialVectorClass<T> value;
      if(conditiona == conditionb)
      {
         std::cout<<"Ploting "<<A.GetNumRowsLocal()<<" by "<<A.GetNumColsLocal()<<" matrix with "<<A.GetNumNonzeros()<<" nnzs."<<std::endl;
         if(A.GetNumRowsLocal() == 0 || A.GetNumColsLocal() == 0)
         {
            return SCHURCHEB_SUCCESS;
         }
         if(perm == NULL)
         {
            marker.Setup(A.GetNumColsLocal());
            value.Setup(A.GetNumColsLocal());
            for(i = 0 ; i < A.GetNumRowsLocal() ; i ++)
            {
               marker.Fill(-1);
               j1 = A_i[i];
               j2 = A_i[i+1];
               for(j = j1 ; j < j2 ; j ++)
               {
                  marker[A_j[j]] = 1;
                  value[A_j[j]] = A_data[j];
               }
               for( j = 0 ; j < A.GetNumColsLocal() ; j ++)
               {
                  if( marker[j] < 0)
                  {
                     SchurchebPrintValueHost(T(),width);
                  }
                  else
                  {
                     SchurchebPrintValueHost(value[j],width);
                  }
                  std::cout<<", ";
               }
               std::cout<<std::endl;
            }
            value.Clear();
            marker.Clear();
         }
         else
         {
            marker.Setup(A.GetNumColsLocal());
            value.Setup(A.GetNumColsLocal());
            for(ii = 0 ; ii < A.GetNumRowsLocal() ; ii ++)
            {
               marker.Fill(-1);
               i = perm[ii];
               j1 = A_i[i];
               j2 = A_i[i+1];
               for(j = j1 ; j < j2 ; j ++)
               {
                  marker[A_j[j]] = 1;
                  value[A_j[j]] = A_data[j];
               }
               for( j = 0 ; j < A.GetNumColsLocal() ; j ++)
               {
                  if( marker[perm[j]] < 0)
                  {
                     SchurchebPrintValueHost(T(),width);
                  }
                  else
                  {
                     SchurchebPrintValueHost(value[perm[j]],width);
                  }
                  std::cout<<", ";
               }
               std::cout<<std::endl;
            }
            value.Clear();
            marker.Clear();
         }
      }
      
      return SCHURCHEB_SUCCESS;
      
   }
   template int CsrMatrixPlotHost( CsrMatrixClass<float> &A, int *perm, int conditiona, int conditionb, int width);
   template int CsrMatrixPlotHost( CsrMatrixClass<double> &A, int *perm, int conditiona, int conditionb, int width);
   template int CsrMatrixPlotHost( CsrMatrixClass<complexs> &A, int *perm, int conditiona, int conditionb, int width);
   template int CsrMatrixPlotHost( CsrMatrixClass<complexd> &A, int *perm, int conditiona, int conditionb, int width);
   
   int CooMatrixReadFromFile(CooMatrixClass<float> &coo, const char *matfile, int idxin, int idxout)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;   
      int i, I, J;
      int shift = idxin - idxout;
      float val;
      
      if ((f = fopen( matfile, "r")) == NULL)
      {
         return SCHURCHEB_ERROR_IO_ERROR;
      }

      if (mm_read_banner(f, &matcode) != 0)
      {
         printf("Could not process Matrix Market banner.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }


      /*  This is how one can screen matrix types if their application */
      /*  only supports a subset of the Matrix Market data types.      */

      if (  !(mm_is_real(matcode) || mm_is_integer(matcode)) || 
            !mm_is_matrix(matcode) || 
            !mm_is_coordinate(matcode))
      {
         printf("Sorry, this application does not support ");
         printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
         SCHURCHEB_ERROR("Error reading MM file.");
         return SCHURCHEB_ERROR_IO_ERROR;
      }

      /* find out size of sparse matrix .... */

      if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      {
         printf("Invalid Size.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( mm_is_general(matcode) )
      {
         coo.Setup( M, N, nz);
      }
      else
      {
         coo.Setup( M, N, 2*nz);
      }
      
      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
      
      for (i=0; i<nz; i++)
      {
         if( fscanf(f, "%d %d %f\n", &I, &J, &val) != 3 ) 
         {
            SCHURCHEB_ERROR("Error reading MM file.");
            return SCHURCHEB_ERROR_IO_ERROR;
         }
         I -= shift;  /* adjust from 1-based to 0-based */
         J -= shift;
         coo.PushBack( I, J, val);
         if(I != J)
         {
            if( mm_is_symmetric(matcode) )
            {
               coo.PushBack( J, I, val);
            }
         }
      }

      if (f !=stdin) fclose(f);
       
      return SCHURCHEB_SUCCESS;
   }
   
   int CooMatrixReadFromFile(CooMatrixClass<double> &coo, const char *matfile, int idxin, int idxout)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;   
      int i, I, J;
      int shift = idxin - idxout;
      double val;
      
      if ((f = fopen( matfile, "r")) == NULL)
      {
         return SCHURCHEB_ERROR_IO_ERROR;
      }

      if (mm_read_banner(f, &matcode) != 0)
      {
         printf("Could not process Matrix Market banner.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }


      /*  This is how one can screen matrix types if their application */
      /*  only supports a subset of the Matrix Market data types.      */

      if (  !(mm_is_real(matcode) || mm_is_integer(matcode)) || 
            !mm_is_matrix(matcode) || 
            !mm_is_coordinate(matcode) )
      {
         printf("Sorry, this application does not support ");
         printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
         return SCHURCHEB_ERROR_IO_ERROR;
      }

      /* find out size of sparse matrix .... */

      if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      {
         printf("Invalid Size.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( mm_is_general(matcode) )
      {
         coo.Setup( M, N, nz);
      }
      else
      {
         coo.Setup( M, N, 2*nz);
      }
      
      /* reseve memory for matrices */
      

      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
      
      for (i=0; i<nz; i++)
      {
         if( fscanf(f, "%d %d %lg\n", &I, &J, &val) != 3 ) 
         {
            SCHURCHEB_ERROR("Error reading MM file.");
            return SCHURCHEB_ERROR_IO_ERROR;
         }
         I -= shift;  /* adjust from 1-based to 0-based */
         J -= shift;
         coo.PushBack( I, J, val);
         if(I != J)
         {
            if( mm_is_symmetric(matcode) )
            {
               coo.PushBack( J, I, val);
            }
         }
      }

      if (f !=stdin) fclose(f);
       
      return SCHURCHEB_SUCCESS;
   }
   
   int CooMatrixReadFromFile(CooMatrixClass<complexs> &coo, const char *matfile, int idxin, int idxout)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;   
      int i, I, J;
      int shift = idxin - idxout;
      float valr, vali;
      
      if ((f = fopen( matfile, "r")) == NULL)
      {
         return SCHURCHEB_ERROR_IO_ERROR;
      }

      if (mm_read_banner(f, &matcode) != 0)
      {
         printf("Could not process Matrix Market banner.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }


      /*  This is how one can screen matrix types if their application */
      /*  only supports a subset of the Matrix Market data types.      */

      if ( !(mm_is_complex(matcode) || mm_is_real(matcode) || mm_is_integer(matcode)) || 
            !mm_is_matrix(matcode) || !mm_is_coordinate(matcode) )
      {
         printf("Sorry, this application does not support ");
         printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
         return SCHURCHEB_ERROR_IO_ERROR;
      }

      /* find out size of sparse matrix .... */

      if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      {
         printf("Invalid Size.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( mm_is_general(matcode) )
      {
         coo.Setup( M, N, nz);
      }
      else
      {
         coo.Setup( M, N, 2*nz);
      }
      
      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
      if(mm_is_complex(matcode))
      {
         /* read complex matrix */
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %f %f\n", &I, &J, &valr, &vali) != 4 ) 
            {
               SCHURCHEB_ERROR("Error reading MM file.");
               return SCHURCHEB_ERROR_IO_ERROR;
            }
            I -= shift;  /* adjust from 1-based to 0-based */
            J -= shift;
            coo.PushBack( I, J, complexs(valr, vali));
            if(I != J)
            {
               if( mm_is_symmetric(matcode) )
               {
                  coo.PushBack( J, I, complexs(valr, vali));
               }
               else if( mm_is_hermitian(matcode) )
               {
                  coo.PushBack( J, I, complexs(valr, -vali));
               }
               else if( mm_is_skew(matcode) )
               {
                  coo.PushBack( J, I, complexs(-valr, vali));
               }
            }
         }
      }
      else
      {
         /* read real matrix */
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %f\n", &I, &J, &valr) != 3 ) 
            {
               SCHURCHEB_ERROR("Error reading MM file.");
               return SCHURCHEB_ERROR_IO_ERROR;
            }
            I -= shift;  /* adjust from 1-based to 0-based */
            J -= shift;
            coo.PushBack( I, J, complexs(valr, 0.0));
            if(I != J)
            {
               if( mm_is_symmetric(matcode) )
               {
                  coo.PushBack( J, I, complexs(valr, 0.0));
               }
               else if( mm_is_hermitian(matcode) )
               {
                  coo.PushBack( J, I, complexs(valr, 0.0));
               }
               else if( mm_is_skew(matcode) )
               {
                  coo.PushBack( J, I, complexs(-valr, 0.0));
               }
            }
         }
      }

      if (f !=stdin) fclose(f);
       
      return SCHURCHEB_SUCCESS;
   }
   
   int CooMatrixReadFromFile(CooMatrixClass<complexd> &coo, const char *matfile, int idxin, int idxout)
   {
      int ret_code;
      MM_typecode matcode;
      FILE *f;
      int M, N, nz;   
      int i, I, J;
      int shift = idxin - idxout;
      double valr, vali;
      
      if ((f = fopen( matfile, "r")) == NULL)
      {
         return SCHURCHEB_ERROR_IO_ERROR;
      }

      if (mm_read_banner(f, &matcode) != 0)
      {
         printf("Could not process Matrix Market banner.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }


      /*  This is how one can screen matrix types if their application */
      /*  only supports a subset of the Matrix Market data types.      */

      if ( !(mm_is_complex(matcode) || mm_is_real(matcode) || mm_is_integer(matcode)) || 
            !mm_is_matrix(matcode) || !mm_is_coordinate(matcode) )
      {
         printf("Sorry, this application does not support ");
         printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
         return SCHURCHEB_ERROR_IO_ERROR;
      }

      /* find out size of sparse matrix .... */

      if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0)
      {
         printf("Invalid Size.\n");
         return SCHURCHEB_ERROR_IO_ERROR;
      }
      
      if( mm_is_general(matcode) )
      {
         coo.Setup( M, N, nz);
      }
      else
      {
         coo.Setup( M, N, 2*nz);
      }
      
      /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
      /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
      /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
      
      if(mm_is_complex(matcode))
      {
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %lg %lg\n", &I, &J, &valr, &vali) != 4 ) 
            {
               SCHURCHEB_ERROR("Error reading MM file.");
               return SCHURCHEB_ERROR_IO_ERROR;
            }
            I -= shift;  /* adjust from 1-based to 0-based */
            J -= shift;
            coo.PushBack( I, J, complexd(valr, vali));
            if(I != J)
            {
               if( mm_is_symmetric(matcode) )
               {
                  coo.PushBack( J, I, complexd(valr, vali));
               }
               else if( mm_is_hermitian(matcode) )
               {
                  coo.PushBack( J, I, complexd(valr, -vali));
               }
               else if( mm_is_skew(matcode) )
               {
                  coo.PushBack( J, I, complexd(-valr, vali));
               }
            }
         }
      }
      else
      {
         for (i=0; i<nz; i++)
         {
            if( fscanf(f, "%d %d %lg\n", &I, &J, &valr) != 3 ) 
            {
               SCHURCHEB_ERROR("Error reading MM file.");
               return SCHURCHEB_ERROR_IO_ERROR;
            }
            I -= shift;  /* adjust from 1-based to 0-based */
            J -= shift;
            coo.PushBack( I, J, complexd(valr, 0.0));
            if(I != J)
            {
               if( mm_is_symmetric(matcode) )
               {
                  coo.PushBack( J, I, complexd(valr, 0.0));
               }
               else if( mm_is_hermitian(matcode) )
               {
                  coo.PushBack( J, I, complexd(valr, 0.0));
               }
               else if( mm_is_skew(matcode) )
               {
                  coo.PushBack( J, I, complexd(-valr, 0.0));
               }
            }
         }
      }

      if (f !=stdin) fclose(f);
       
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T>
   int SetupPermutationRKwayRecursive( CsrMatrixClass<T> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v)
   {
      
      /* immediatly return if we don't have enoguh levels */
      if(tlvl < 2 || A.GetNumRowsLocal() < 2)
      {
         /* only one level */
         mapptr_v[1]=1;
         map_v.Fill(0);
         tlvl = 1;
         return SCHURCHEB_SUCCESS;
      }
      
      
      /* should not call this function with num_dom < 2 */
      SCHURCHEB_CHKERR(num_dom < 2);
      
      /* TODO: disconnected components */
      /* now start calling the recursive KWay main function */
      int               i, j, k, nA, domi, edgecut, num_dom2;
      int               num_dom_temp, nd_clvl, nd_tlvl, nd_minsep;
      std::vector<std::vector<vector_int> > nd_level_str;
      vector_int        map, vtxsep, perm, dom_ptr, perm_c, mapc;
      CsrMatrixClass<T> C;
      
      if( A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("RKway partition only works for host.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      if( A.GetNumRowsLocal() != A.GetNumColsLocal())
      {
         SCHURCHEB_ERROR("RKway partition only works for square matrix.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      if(clvl == tlvl-1)
      {
         /* treat the last level as a single block */
         domi = mapptr_v[clvl];
         mapptr_v[clvl+1]=domi+1;
         map_v.Fill(domi);
         tlvl = clvl+1;
         return SCHURCHEB_SUCCESS;
      }
      
      nA = A.GetNumRowsLocal();
      
      /* stop if we don't have enough number of nodes on the next level */
      if(minsep >= nA || num_dom >= nA )
      {
         /* in this case, we can't apply the partition any more, 
          * put the reaminning part into one single level */
         domi = mapptr_v[clvl];
         mapptr_v[clvl+1]=domi+1;
         map_v.Fill(domi);
         tlvl = clvl+1;
         
         return SCHURCHEB_SUCCESS;
      }
      
      /* otherwise we have enough levels */
      if(vertexsep)
      {
         /* Use vertex seperator, we use multiple 2-way partition
          * we can take advantage of the ND code
          */
          
         /* first make sure num_dom is power of 2 */
         SCHURCHEB_CHKERR( ( ( num_dom ) & ( num_dom - 1 ) ) != 0 );
         
         /* now find 2^k = num_dom */
         num_dom_temp = num_dom;
         nd_tlvl = 0;
         while (num_dom_temp > 0)
         {
            num_dom_temp = num_dom_temp >> 1;
            nd_tlvl++;
         }
         /* now 2^nd_tlvl = num_dom, we apply a k level ND 
          * with vertex seperator, keep all other level as edge seperator
          */
         nd_clvl = 0;
         /* in this partition we partition till very small blocks */
         nd_minsep = 2;
         
         /* call ND ordering */
         SetupPermutationNDRecursive( A, true, nd_clvl, nd_tlvl, nd_minsep, nd_level_str);
         
         /* combine different domains together */
         SetupPermutationNDCombineLevels(nd_level_str[0], num_dom);
         
         /* now, nd_tlvl is the total number of levels 
          * we only going to keep the level 0, and put all other levels into a big vertex seperator.
          * Note that perm is not used in this function, we are not going to build it.
          */
         map.Setup(nA);
         vtxsep.Setup(nA);
         
         /* the size on level 0 is the final k */
         num_dom2 = nd_level_str[0].size();
         dom_ptr.Setup(num_dom2+1);
         
         /* we assign all other levels to the last level */
         map.Fill(num_dom2-1);
         vtxsep.Fill(1);
         
         /* first setup the interior nodes */
         domi = 0;
         dom_ptr[0] = 0;
         
         for(i = 0 ; i < num_dom2 ; i ++)
         {
            k = nd_level_str[0][i].GetLengthLocal();
            for(j = 0 ; j < k ; j ++)
            {
               map[nd_level_str[0][i][j]] = domi;
               vtxsep[nd_level_str[0][i][j]] = 0;
            }
            domi++;
            dom_ptr[domi] = dom_ptr[domi-1] + k;
         }
         
         edgecut = nA - dom_ptr[num_dom2];
         dom_ptr[num_dom2] = nA;
         
         /* now put the remaining le */
         
         for(i = 0 ; i < nd_tlvl ; i ++)
         {
            k = nd_level_str[i].size();
            for(j = 0 ; j < k ; j ++)
            {
               nd_level_str[i][j].Clear();
            }
            std::vector<vector_int>().swap(nd_level_str[i]);
         }
         std::vector<std::vector<vector_int> >().swap(nd_level_str);
         
      }
      else
      {
         /* use the standart edge seperator */
         num_dom2 = num_dom;
         std::vector<vector_int> test;
         int testcomp;
         A.GetConnectedComponents( test, testcomp);
         
         if(CsrMatrixMetisKwayHost( A, num_dom2, map, false, vtxsep, edgecut, perm, dom_ptr) == SCHURCHEB_RETURN_METIS_NO_INTERIOR )
         {
            /* in this case, at least one subdomain has no interior nodes, we should stop on this level 
             * Set edgecut to nA so that we'll go to the terminate phase
             */
            edgecut = nA;
         }
      }
      
      if(edgecut == 0)
      {
         /* we have no next level, and the partition of this level is perfect */
         domi = mapptr_v[clvl];
         
         mapptr_v[clvl+1]=domi + num_dom2;
         
         /* add a shift to the map vector */
         for(i = 0 ; i < nA ; i ++)
         {
            map_v[i] = map[i] + domi;
         }
         
         tlvl = clvl+1;
         return SCHURCHEB_SUCCESS;
      }
      
      if( num_dom2 < num_dom || edgecut == nA )
      {
         /* treat the last level as a single block */
         domi = mapptr_v[clvl];
         mapptr_v[clvl+1]=domi+1;
         map_v.Fill(domi);
         tlvl = clvl+1;
         return SCHURCHEB_SUCCESS;
      }
      
      /* start forming the C matrix */
      perm_c.Setup(edgecut);
      
      j = 0;
      for(i = 0 ; i < nA ; i ++)
      {
         if(vtxsep[i] > 0)
         {
            perm_c[j++] = i;
         }
      }
      
      /* extract the C matrix */
      A.SubMatrix( perm_c, perm_c, kMemoryHost, C);
      perm_c.Clear();
      
      /* update the mapptr */
      mapptr_v[clvl+1] = mapptr_v[clvl] + num_dom2;
      
      /* recursive partition */
      if(num_dom > kmin)
      {
         num_dom = num_dom / kfactor;
         if(num_dom < kmin)
         {
            num_dom = kmin;
         }
      }
      
      mapc.Setup(edgecut);
      /* keep using nd */
      //SetupPermutationRKwayRecursive( C, vertexsep, clvl+1, tlvl, num_dom, minsep, kmin, kfactor, mapc, mapptr_v);
      /* only use vertex sep on the top level */
      SetupPermutationRKwayRecursive( C, false, clvl+1, tlvl, num_dom, minsep, kmin, kfactor, mapc, mapptr_v);
      
      /* go back to this level, update map_v */
      j = 0;
      domi = mapptr_v[clvl];
      for(i = 0 ; i < nA ; i ++)
      {
         if(vtxsep[i] > 0)
         {
            map_v[i] = mapc[j++];
         }
         else
         {
            map_v[i] = map[i] + domi;
         }
      }
      
      mapc.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   template int SetupPermutationRKwayRecursive( CsrMatrixClass<float> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v);
   template int SetupPermutationRKwayRecursive( CsrMatrixClass<double> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v);
   template int SetupPermutationRKwayRecursive( CsrMatrixClass<complexs> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v);
   template int SetupPermutationRKwayRecursive( CsrMatrixClass<complexd> &A, bool vertexsep, int clvl, int &tlvl, int num_dom, int minsep, int kmin, int kfactor, vector_int &map_v, vector_int &mapptr_v);
   
   /**
    * @brief   Compress a certain level of level_str from SetupPermutationNDRecursive into a give number of domains.
    * @details Compress a certain level of level_str from SetupPermutationNDRecursive into a give number of domains.
    * @param   [in]     level_stri The level of a level_str.
    * @param   [in,out] ndom The target number of domains, on return the ndom we get.
    * @return     Return error message.
    */
   int SetupPermutationNDCombineLevels(std::vector<vector_int> &level_stri, int &ndom)
   {
      int i, n, ndom_in;
      vector_int size, marker;
      
      ndom_in = level_stri.size();
      
      if(ndom_in <= ndom)
      {
         /* in this case, do nothing */
         return SCHURCHEB_SUCCESS;
      }
      
      size.Setup(ndom_in);
      marker.Setup(ndom_in, true);
      
      n = 0;
      for(i = 0 ; i < ndom_in ; i ++)
      {
         size[i] = level_stri[i].GetLengthLocal();
         n += size[i];
      }
      
      /* TODO: some dynamic programming algorithms */
      
      size.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T>
   int SetupPermutationNDRecursive( CsrMatrixClass<T> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str)
   {
      /* ND ordering
       * start with each connected components of A. For example, A has 2 connected compononts, 0 and 1.
       * The size of 0 might be very small, and can only be partitioned into two parts, 2 and 3, we put it into a higher level.
       * After that, the partition of 1 again leads to a small component 4.
       * 
       * Partition:            A
       *                       |
       *              0------------------1
       *                                 |
       *                           2-----3-----4
       *                                 |     |
       *                                5-6   7-8
       * where 0 and 2 are very small connected components that can't be further partitioned. We form the following level structure:
       * 
       * level 2:   1
       * level 1:   3, 4
       * level 0:   0, 2, 5, 6, 7, 8
       * 
       * Algorith: we use a recursive algorithm, starting from the first level.
       * 
       * When raeching the leaf component, we push it to level_str[0] immediatly
       */
      
      /* now start calling the recursive ND main function */
      int                     i, j, k, idx, nS, ndom, edgecut, ncomps, size, k1, k2, err = 0;
      vector_int              map, vtxsep, perm, dom_ptr, row_perm, col_perm, perm_c, mapc, tlvls;
      std::vector<vector_int> comp_indices;
      
      CsrMatrixClass<T> B, C;
      
      if( A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("ND partition only works for host.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      if( A.GetNumRowsLocal() != A.GetNumColsLocal())
      {
         SCHURCHEB_ERROR("ND partition only works for square matrix.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      /* setup level_str] */
      for(i = 0 ; i < (int)level_str.size() ; i ++)
      {
         for(j = 0 ; j < (int)level_str[i].size() ; j ++ )
         {
            level_str[i][j].Clear();
         }
         std::vector<vector_int>().swap(level_str[i]);
      }
      std::vector<std::vector<vector_int> >().swap(level_str);
      
      if(tlvl < 2 || A.GetNumRowsLocal() < 2)
      {
         /* only one level */
         tlvl = 1;
         level_str.resize(1);
         level_str[0].resize(1);
         level_str[0][0].Setup(A.GetNumRowsLocal());
         level_str[0][0].UnitPerm();
         return SCHURCHEB_SUCCESS;
      }
      
      if(clvl >= tlvl - 1)
      {
         /* first step, find the connected components */
         ncomps = 0;
         err = A.GetConnectedComponents( comp_indices, ncomps);
         tlvl = 1;
         level_str.resize(1);
         level_str[0].resize(ncomps);
         for(i = 0 ; i < ncomps ; i ++)
         {
            level_str[0][i] = std::move(comp_indices[i]);
         }
         return SCHURCHEB_SUCCESS;
      }
      
      level_str.resize(tlvl-clvl);
      
      /* first step, find the connected components */
      ncomps = 0;
      err = A.GetConnectedComponents( comp_indices, ncomps);
      
      /* apply ND for each component */
      tlvls.Setup(ncomps);
      for(i = 0 ; i < ncomps ; i ++)
      {
         /* First, check the size of the current component 
          * if the size is too small, stop here
          */
         nS = comp_indices[i].GetLengthLocal();
         
         if( nS <= minsep || nS <= 2)
         {
            /* we've reach the end of this components, this is a node of level 0 
             * mark the level of this components as clvl + 1
             */
            tlvls[i] = 1;
            level_str[0].push_back(comp_indices[i]);
            continue;
         }
         else
         {
            /* get this component */
            A.SubMatrix( comp_indices[i], comp_indices[i], kMemoryHost, C);
            
            /* apply 2-way partition */
            ndom = 2;
            if( CsrMatrixMetisKwayHost( C, ndom, map, vertexsep, vtxsep, edgecut, perm, dom_ptr) == SCHURCHEB_RETURN_METIS_NO_INTERIOR )
            {
               /* in this case, we don't have 2 subdomains, or at least one subdomain has no interior nodes 
                * we should stop here. Go the the terminate phase by setting edgecut to nS
                */
               edgecut = nS;
            }
            
            if( ndom < 2 || edgecut == nS)
            {
               tlvls[i] = 1;
               level_str[0].push_back(comp_indices[i]);
               continue;
            }
            
            /* remove the seperator */
            C.SubMatrixNoPerm(vtxsep, vtxsep, row_perm, col_perm, true, kMemoryHost, B);
            
            /* go to next level 
             * on exit, tlvls[i] is the max level of this component
             */
            tlvls[i] = tlvl;
            std::vector<std::vector<vector_int> > sub_level_str;
            err = SetupPermutationNDRecursive( B, vertexsep, clvl+1, tlvls[i], minsep, sub_level_str); SCHURCHEB_CHKERR(err);
            
            /* now back, set indices */
            for( j = 0 ; j < tlvls[i] ; j ++)
            {
               idx = j;
               size = sub_level_str[idx].size();
               for(k = 0 ; k < size ; k ++)
               {
                  /* push this to the same level of level_str */
                  level_str[idx].push_back(sub_level_str[idx][k]);
                  
                  /* update index */
                  vector_int &nodes = level_str[idx].back();
                  k2 = nodes.GetLengthLocal();
                  for(k1 = 0 ; k1 < k2 ; k1 ++)
                  {
                     nodes[k1] = comp_indices[i][row_perm[nodes[k1]]];
                  }
               }
            }
            /* also add the edge seperator */
            idx = tlvls[i];
            level_str[idx].push_back(vector_int());
            vector_int &nodes = level_str[idx].back();
            nodes.Setup(0, edgecut, kMemoryHost, false);
            for(j = 0 ; j < nS ; j ++)
            {
               if(vtxsep[j] != 0)
               {
                  nodes.PushBack(comp_indices[i][j]);
               }
            }
            tlvls[i] += 1;
         }
      }
      
      tlvl = tlvls.Max();
      
      return err;
   }
   template int SetupPermutationNDRecursive( CsrMatrixClass<float> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str);
   template int SetupPermutationNDRecursive( CsrMatrixClass<double> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str);
   template int SetupPermutationNDRecursive( CsrMatrixClass<complexs> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str);
   template int SetupPermutationNDRecursive( CsrMatrixClass<complexd> &A, bool vertexsep, int clvl, int &tlvl, int minsep, std::vector<std::vector<vector_int> > &level_str);
   
   template <typename T>
   int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<T> &A, bool vertexsep, int &nlev, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last)
   {
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Recursive kway partition only works for host matrices.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      int                              err = 0;
      int                              clvl;
      vector_SCHURCHEB_long            vtxdist, xadj, adjncy;
      
      A.GetGraphArrays(vtxdist, xadj, adjncy);
      
      if( A.GetSeparatorNumSubdomains() == ncomp)
      {
         /* the partition on the top level is provided, use it */
         if(vertexsep)
         {
            int                           i, j;
            int                           n_local; 
            SCHURCHEB_long                col;
            vector_int                    map, vtxsep, map2_v;
            int                           err = 0;
            vector_SCHURCHEB_long         vtxdist_s, xadj_s, adjncy_s;
            
            n_local = A.GetNumRowsLocal();
            map = A.GetSeparatorDomi();
            
            /* get the separator. In array vtxsep, mark as 1 to be in the seprator */
            err = ParallelRKwayGetSeparator2( vtxdist, xadj, adjncy, vertexsep, vtxdist_s, xadj_s, adjncy_s, map, ncomp, vtxsep, A);
            
            if(err == -1)
            {
               /* Get separator failes, no next level availiable */
               if(bj_last)
               {
                  /* in this case just use the input partition */
                  mapptr_v.Setup(2, true);
                  mapptr_v[1] = ncomp;
                  map_v.Setup(n_local);
                  
                  for(i = 0 ; i < n_local ; i ++)
                  {
                     map_v[i] = map[i];
                  }
               }
               else
               {
                  /* only one subdomain, 0 */
                  mapptr_v.Setup(2, true);
                  mapptr_v[1] = 1;
                  map_v.Setup(n_local);
                  col = 0;
                  map_v.Fill(col);
               }
               
               nlev = 1;
               return SCHURCHEB_SUCCESS;
            }
            SCHURCHEB_CHKERR(err);
            
            clvl = 1;
            mapptr_v.Setup(1, true);
            mapptr_v.PushBack(ncomp);
            
            if(ncomp > kmin)
            {
               ncomp = ncomp / kfactor;
               if(ncomp < kmin)
               {
                  ncomp = kmin;
               }
            }
            
            // (Increase clvl by 1)
            SetupPermutationParallelRKwayRecursive( vtxdist_s, xadj_s, adjncy_s, vertexsep, clvl+1, nlev, ncomp, minsep, kmin, kfactor, map2_v, mapptr_v, bj_last, A);
            
            /* udpate map information */
            map_v.Setup(n_local);
            j = 0;
            for(i = 0 ; i < n_local ; i ++)
            {
               if(map[i] >= 0)
               {
                  /* interior nodes */
                  map_v[i] = map[i];
               }
               else
               {
                  /* exterior nodes */
                  map_v[i] = map2_v[j++];
               }
            }
            
            vtxdist_s.Clear(); 
            xadj_s.Clear();
            adjncy_s.Clear();
            vtxsep.Clear();
            map2_v.Clear();
         }
         else
         {
            int                           i, j;
            int                           n_local; 
            SCHURCHEB_long                col;
            vector_int                    map2, vtxsep, map2_v;
            vector_SCHURCHEB_long         map;
            int                           err = 0;
            vector_SCHURCHEB_long         vtxdist_s, xadj_s, adjncy_s;
            
            n_local = A.GetNumRowsLocal();
            map2 = A.GetSeparatorDomi();
            
            map.Setup(n_local);
            
            for(i = 0 ; i < n_local ; i ++)
            {
               if(map2[i] >= 0)
               {
                  map[i] = map2[i];
               }
               else
               {
                  map[i] = -map2[i]-1;
               }
            }
            
            /* get the separator. In array vtxsep, mark as 1 to be in the seprator */
            err = ParallelRKwayGetSeparator( vtxdist, xadj, adjncy, vertexsep, vtxdist_s, xadj_s, adjncy_s, map, ncomp, vtxsep, A);
            
            if(err == -1)
            {
               /* Get separator failes, no next level availiable */
               if(bj_last)
               {
                  /* in this case just use the input partition */
                  mapptr_v.Setup(2, true);
                  mapptr_v[1] = ncomp;
                  map_v.Setup(n_local);
                  
                  for(i = 0 ; i < n_local ; i ++)
                  {
                     map_v[i] = map[i];
                  }
               }
               else
               {
                  /* only one subdomain, 0 */
                  mapptr_v.Setup(2, true);
                  mapptr_v[1] = 1;
                  map_v.Setup(n_local);
                  col = 0;
                  map_v.Fill(col);
               }
               
               nlev = 1;
               return SCHURCHEB_SUCCESS;
            }
            SCHURCHEB_CHKERR(err);
            
            clvl = 1;
            mapptr_v.Setup(1, true);
            mapptr_v.PushBack(ncomp);
            
            if(ncomp > kmin)
            {
               ncomp = ncomp / kfactor;
               if(ncomp < kmin)
               {
                  ncomp = kmin;
               }
            }
            
            // (Increase clvl by 1)
            SetupPermutationParallelRKwayRecursive( vtxdist_s, xadj_s, adjncy_s, vertexsep, clvl+1, nlev, ncomp, minsep, kmin, kfactor, map2_v, mapptr_v, bj_last, A);
            
            /* udpate map information */
            map_v.Setup(n_local);
            j = 0;
            for(i = 0 ; i < n_local ; i ++)
            {
               if(vtxsep[i] <= 0)
               {
                  /* interior nodes */
                  map_v[i] = map2[i];
               }
               else
               {
                  /* exterior nodes */
                  map_v[i] = map2_v[j++];
               }
            }
            
            vtxdist_s.Clear(); 
            xadj_s.Clear();
            adjncy_s.Clear();
            vtxsep.Clear();
            map2_v.Clear();
            map2.Clear();
         }
         
      }
      else
      {
         /* call rKway function */
         clvl = 1;
         mapptr_v.Setup(1, true);
         
         err = SetupPermutationParallelRKwayRecursive( vtxdist, xadj, adjncy, vertexsep, clvl, nlev, ncomp, minsep, kmin, kfactor, map_v, mapptr_v, bj_last, A); SCHURCHEB_CHKERR(err);
      }
      
      return err;
   }
   template int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<float> &A, bool vertexsep, int &nlev, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last);
   template int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<double> &A, bool vertexsep, int &nlev, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last);
   template int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<complexs> &A, bool vertexsep, int &nlev, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last);
   template int ParallelCsrMatrixSetupPermutationParallelRKway( ParallelCsrMatrixClass<complexd> &A, bool vertexsep, int &nlev, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last);
   
   int SetupPermutationParallelRKwayRecursive(vector_SCHURCHEB_long &vtxdist, vector_SCHURCHEB_long &xadj, vector_SCHURCHEB_long &adjncy, bool vertexsep, int clvl, int &tlvl, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog)
   {
      SCHURCHEB_long          i, j, n_local, nA, col, ncomp2;
      int                     err = 0;
      vector_SCHURCHEB_long   map;
      vector_int              map2_v;
      vector_int              vtxsep;
      vector_SCHURCHEB_long   vtxdist_s, xadj_s, adjncy_s;
      
      MPI_Comm          comm;
      int               myid, np;
      parlog.GetMpiInfo(np, myid, comm);
      
      nA = vtxdist[np];
      n_local = vtxdist[myid+1] - vtxdist[myid];
      
      /* main loop */
      if (minsep < nA && ncomp <= nA && clvl < tlvl) 
      {

         /* call parMetis for partition */
         ncomp2 = ncomp;
         
         err = ParmetisKwayHost( vtxdist, xadj, adjncy, ncomp2, map, parlog); //SCHURCHEB_CHKERR(err);
         
         if( err || ncomp2 < ncomp)
         {
            /* in this case, we don't have enough domains, stop here */
            if(bj_last && nA >= np)
            {
               /* treat last level with block-Jacobi like partition 
                * since in this case the METIS fails
                * we assign ncomp subdomains to the last level evenly
                */
               
               SCHURCHEB_long          bjdom1, bjdom2, k;
               vector_SCHURCHEB_long   bjdisps, bjns;
               
               /* if we have enough nodes, assign ncomp, otherwise np */
               SCHURCHEB_long ncomp3 = nA >= ncomp ? ncomp : np;
               
               bjdom1 = nA / ncomp3;
               bjdom2 = nA % ncomp3;
               
               bjdisps.Setup(np+1);
               bjns.Setup(np);
               
               for(i = 0 ; i < np ; i ++)
               {
                  if(i < bjdom2)
                  {
                     bjns[i] = bjdom1 + 1;
                  }
                  else
                  {
                     bjns[i] = bjdom1;
                  }
               }
               
               bjdisps[0] = 0;
               for(i = 0 ; i < np ; i ++)
               {
                  bjdisps[i+1] = bjdisps[i] + bjns[i];
               }
               
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+np);
               map_v.Setup(n_local);
               
               j = 0;
               for(i = 0, k = vtxdist[myid]; i < n_local ; i ++, k++)
               {
                  while(bjdisps[j+1] <= k)
                  {
                     /* in this case, current node blongs to the next one 
                      * we want k in [bjdisps[j], bjdisps[j+1])
                      * example: 0, 2, 4, 6, 8, when k = 2, j should be 2.
                      */
                     j++;
                  }
                  map_v[i] = j+col;
               }
            }
            else
            {
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+1);
               map_v.Setup(n_local);
               map_v.Fill(col);
            }
            
            tlvl = clvl;
         }
         else
         {
            /* get the separator. In array vtxsep, mark as 1 to be in the seprator */
            err = ParallelRKwayGetSeparator( vtxdist, xadj, adjncy, vertexsep, vtxdist_s, xadj_s, adjncy_s, map, ncomp, vtxsep, parlog);
            
            if(err == -1)
            {
               /* Get separator failes, no next level availiable */
               if(bj_last)
               {
                  /* in this case just use the partition provided by parMETIS */
                  mapptr_v.PushBack(mapptr_v.Back()+ncomp2);
                  map_v.Setup(n_local);
                  for(i = 0 ; i < n_local ; i ++)
                  {
                     map_v[i] = map[i] + mapptr_v[clvl-1];
                  }
               }
               else
               {
                  col = mapptr_v.Back();
                  mapptr_v.PushBack(col+1);
                  map_v.Setup(n_local);
                  map_v.Fill(col);
               }
               
               tlvl = clvl;
               return SCHURCHEB_SUCCESS;
            }
            SCHURCHEB_CHKERR(err);
            
            mapptr_v.PushBack(mapptr_v.Back()+ncomp2);
            
            if(ncomp > kmin)
            {
               ncomp = ncomp / kfactor;
               if(ncomp < kmin)
               {
                  ncomp = kmin;
               }
            }
            
            // (Increase clvl by 1)
            SetupPermutationParallelRKwayRecursive( vtxdist_s, xadj_s, adjncy_s, vertexsep, clvl+1, tlvl, ncomp, minsep, kmin, kfactor, map2_v, mapptr_v, bj_last, parlog);
                  
            /* udpate map information */
            map_v.Setup(n_local);
            j = 0;
            for(i = 0 ; i < n_local ; i ++)
            {
               if(vtxsep[i] <= 0)
               {
                  /* interior nodes */
                  map_v[i] = map[i] + mapptr_v[clvl-1];
               }
               else
               {
                  /* exterior nodes */
                  map_v[i] = map2_v[j++];
               }
            }
         }
      }
      else
      {
         /* in this case, we don't have enough domains, stop here 
          * a special case is when nA == 0
          */
         if(nA > 0)
         {
            
            if(bj_last && nA >= np)
            {
               if(nA >= ncomp)
               {
                  /* good for one extra partition */
                  
                  ncomp2 = ncomp;
                  
                  err = ParmetisKwayHost( vtxdist, xadj, adjncy, ncomp2, map, parlog); //SCHURCHEB_CHKERR(err);
                  
                  if( err || ncomp2 < ncomp)
                  {
                     /* fails, use bj
                      * treat last level with block-Jacobi like partition 
                      * since in this case the METIS fails
                      * we assign ncomp subdomains to the last level evenly
                      */
                     
                     SCHURCHEB_long          bjdom1, bjdom2, k;
                     vector_SCHURCHEB_long   bjdisps, bjns;
                     
                     bjdom1 = nA / ncomp;
                     bjdom2 = nA % ncomp;
                     
                     bjdisps.Setup(np+1);
                     bjns.Setup(np);
                     
                     for(i = 0 ; i < np ; i ++)
                     {
                        if(i < bjdom2)
                        {
                           bjns[i] = bjdom1 + 1;
                        }
                        else
                        {
                           bjns[i] = bjdom1;
                        }
                     }
                     
                     bjdisps[0] = 0;
                     for(i = 0 ; i < np ; i ++)
                     {
                        bjdisps[i+1] = bjdisps[i] + bjns[i];
                     }
                     
                     col = mapptr_v.Back();
                     mapptr_v.PushBack(col+np);
                     map_v.Setup(n_local);
                     
                     j = 0;
                     for(i = 0, k = vtxdist[myid]; i < n_local ; i ++, k++)
                     {
                        while(bjdisps[j+1] <= k)
                        {
                           /* in this case, current node blongs to the next one 
                            * we want k in [bjdisps[j], bjdisps[j+1])
                            * example: 0, 2, 4, 6, 8, when k = 2, j should be 2.
                            */
                           j++;
                        }
                        map_v[i] = j+col;
                     }
                  }
                  else
                  {
                     mapptr_v.PushBack(mapptr_v.Back()+ncomp2);
                     map_v.Setup(n_local);
                     /* use the partition */
                     for(i = 0 ; i < n_local ; i ++)
                     {
                        map_v[i] = map[i] + mapptr_v[clvl-1];
                     }
                  }
               }
               else
               {
                  /* treat last level with block-Jacobi like partition 
                   * since in this case the METIS fails
                   * we assign ncomp subdomains to the last level evenly
                   */
                  
                  SCHURCHEB_long          bjdom1, bjdom2, k;
                  vector_SCHURCHEB_long   bjdisps, bjns;
                  
                  bjdom1 = nA / np;
                  bjdom2 = nA % np;
                  
                  bjdisps.Setup(np+1);
                  bjns.Setup(np);
                  
                  for(i = 0 ; i < np ; i ++)
                  {
                     if(i < bjdom2)
                     {
                        bjns[i] = bjdom1 + 1;
                     }
                     else
                     {
                        bjns[i] = bjdom1;
                     }
                  }
                  
                  bjdisps[0] = 0;
                  for(i = 0 ; i < np ; i ++)
                  {
                     bjdisps[i+1] = bjdisps[i] + bjns[i];
                  }
                  
                  col = mapptr_v.Back();
                  mapptr_v.PushBack(col+np);
                  map_v.Setup(n_local);
                  
                  j = 0;
                  for(i = 0, k = vtxdist[myid]; i < n_local ; i ++, k++)
                  {
                     while(bjdisps[j+1] <= k)
                     {
                        /* in this case, current node blongs to the next one 
                         * we want k in [bjdisps[j], bjdisps[j+1])
                         * example: 0, 2, 4, 6, 8, when k = 2, j should be 2.
                         */
                        j++;
                     }
                     map_v[i] = j+col;
                  }
               }
               
            }
            else
            {
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+1);
               map_v.Setup(n_local);
               map_v.Fill(col);
            }
            
            tlvl = clvl;
         }
         else
         {
            tlvl = clvl-1;
         }
      }
      
      return SCHURCHEB_SUCCESS;
   }
   
   int ParallelRKwayGetSeparator( vector_SCHURCHEB_long &vtxdist, vector_SCHURCHEB_long &xadj, vector_SCHURCHEB_long &adjncy, bool vertexsep, vector_SCHURCHEB_long &vtxdist_s,  vector_SCHURCHEB_long &xadj_s,  vector_SCHURCHEB_long &adjncy_s, vector_SCHURCHEB_long &map, int num_dom, vector_int &vtxsep, parallel_log &parlog)
   {
      SCHURCHEB_long                            i, ii, j, j1, j2;
      SCHURCHEB_long                            n_local, n_start, n_end, dom, col, nnz, n_local_s, local_diff, global_diff;
      int                                       id, idx;
      vector_int                                ids, marker2;
      vector_SCHURCHEB_long                     n_local_ss;
      
      std::unordered_map<SCHURCHEB_long, int>   col_map_hash;
      int                                       ncols;
      vector_SCHURCHEB_long                     cols;
      vector_int                                col_ids;
      
      std::unordered_map<SCHURCHEB_long, int>   col_map_uncertain_hash;
      vector_int                                sendsize, recvsize;
      std::vector<vector_SCHURCHEB_long>        send_v2, recv_v2;
      std::vector<vector_int>                   send2_v2, recv2_v2;
      
      MPI_Comm                comm;
      int                     myid, np, numwaits;
      vector<MPI_Request>     requests;
      
      parlog.GetMpiInfo(np, myid, comm);
      
      n_start = vtxdist[myid];
      n_end = vtxdist[myid+1];
      n_local = n_end - n_start;
      
      if(n_local == 0)
      {
         /* empty, no need to setup */
         nnz = 0;
         vtxsep.Clear();
      }
      else
      {
         nnz = xadj[n_local];
         vtxsep.Setup(n_local, true);
         ids.Setup(nnz, true);
      }
      
      /* -------------------------
       * Step 1: put all local columns into the hash table
       * -------------------------
       */
      ncols = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = xadj[i];
         j2 = xadj[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            /* get the processor holds this index 
             * note that there might be duplicate entries 
             */
            auto find_col =  col_map_hash.find(col);
            if(find_col == col_map_hash.end())
            {
               /* a new one */
               if(vtxdist.BinarySearch( col, id, true) < 0)
               {
                  /* in this case, col fall in between */
                  id--;
               }
               
               cols.PushBack(col);
               col_ids.PushBack(id);
               col_map_hash[col] = ncols;
               ncols++;
            }
         }
      }
      
      /* ids[j] is the MPI process adjncy[j] belongs to */
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = xadj[i];
         j2 = xadj[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            /* get the processor holds this index 
             * note that there might be duplicate entries 
             */
            auto find_col =  col_map_hash.find(col);
            ids[j] = col_ids[find_col->second];
         }
      }
      
      /* -------------------------
       * Step 2: check local first
       * some rows/cols requires 
       * accessing offdiagonal entries
       * -------------------------
       */
      
      /* This loop find local vtxsep
       * 1. local cols already has multiple maps:
       *    => set vtxsep to 1, separator.
       * 2. all cols are local, and all same map values:
       *    => set vtxsep to 0, interior.
       * 3. all local cols same map, however, have exterior cols
       *    => set vtxsep to -1, TBD
       */
      for(i = 0 ; i < n_local ; i ++)
      {
         /* dom is the domain of the row */
         dom = map[i];
         j1 = xadj[i];
         j2 = xadj[i+1];
         
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            id = ids[j]; /* this is the id this col belongs to */
            
            /* we can only check the map value of local parts */
            if(id == myid)
            {
               col -= n_start;
               if(map[col] != dom)
               {
                  /* this is diagonal, exterior node */
                  vtxsep[i] = 1;
                  break;
               }
            }
            else
            {
               /* if has offd, mark to -1 instead */
               vtxsep[i] = -1;
            }
         }
      }
      
      /* -------------------------
       * Step 3: check remaining
       * -------------------------
       */
      
      sendsize.Setup(np, true);
      recvsize.Setup(np, true);
      send_v2.resize(np);
      recv_v2.resize(np);
      send2_v2.resize(np);
      recv2_v2.resize(np);
      
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] == -1 || (vtxsep[i] == 1 && vertexsep))
         {
            /* this is a target column */
            j1 = xadj[i];
            j2 = xadj[i+1];
            
            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */
               
               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(find_col == col_map_uncertain_hash.end())
                  {
                     /* breaking news! a NEW column! */
                     send_v2[id].PushBack(col);
                     col_map_uncertain_hash[col] = sendsize[id];
                     sendsize[id]++;
                  }
               }
            }
         }
      }
      
      /* communicate send and recv size */
      SCHURCHEB_MPI_CALL( MPI_Alltoall( sendsize.GetData(), 1, MPI_INT, recvsize.GetData(), 1, MPI_INT, comm) );
      
      /* then apply communication */
      
      requests.resize(2*np);
      
      /* first send cols */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            /* myid have data for processor i */
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( send_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv_v2[i].Setup(recvsize[i]);
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( recv_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      SCHURCHEB_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );
      
      /* get the map value */
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv2_v2[i].Setup(recvsize[i]);
            for(j = 0 ; j < recvsize[i]; j ++)
            {
               recv2_v2[i][j] = map[recv_v2[i][j] - n_start];
            }
         }
      }
      
      /* then apply communication again */
      
      /* or MPI_Alltoallv? */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            /* myid have data for processor i */
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( recv2_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            send2_v2[i].Setup(sendsize[i]);
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( send2_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      SCHURCHEB_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );
      
      /* finally check the received columns */
      
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] == -1)
         {
            /* this is a target column */
            dom = map[i];
            j1 = xadj[i];
            j2 = xadj[i+1];
            
            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */
               
               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(send2_v2[id][find_col->second] != dom)
                  {
                     vtxsep[i] = 1;
                     break;
                  }
               }
            }
            
            /* if still -1, this is interior */
            if(vtxsep[i] == -1)
            {
               vtxsep[i] = 0;
            }
         }
      }
      
      /* -------------------------
       * Step 3.5: update the edge
       * separator into a rough
       * vertex separator
       * -------------------------
       */
       
      /* We use a simply recursive algorithm
       * Each time split node set V into V1 and V2
       * Find all edges across V1 and V2, mark one of
       * the end as vertex separator.
       * Recursively apply this strategy to V1 and V2
       */
      
      if(vertexsep)
      {
         /* search vector on each level:
          * 0          m          ndom  2+1
          * 0    n1    m    n2    ndom  4+1
          * 0 o1 n1 o2 m o3 n2 o4 ndom  8+1
          * 2^{n + 1) + n - 2
          */
         vector_SCHURCHEB_long tree;
         
         int k, ts, te, tlen, idxi, leveli, tree_size, tree_level;
         SCHURCHEB_long ns[4], nsg[4], domi;
         
         bool marker, even;
         
         tree_size = 2;
         tree_level = 1;
         
         while( tree_size < num_dom )
         {
            tree_size = tree_size << 1;
            tree_level++;
         }
         
         tree_size = 2*pow(2,tree_level)+tree_level-2;
         
         tree.Setup(tree_size, true);
         
         tree[0] = 0;
         tree[1] = num_dom/2;
         tree[2] = num_dom;
         
         ts = 3;
         
         for(i = 1 ; i < tree_level ; i ++)
         {
            /* [0-2] [3-7] [8-17] [7-14] */
            te = ts + pow(2,i+1) + 1;
            
            for(j = ts, k = ts-pow(2,i)-1 ; j < te ; j +=2, k+=1)
            {
               tree[j] = tree[k];
            }
            for(j = ts + 1, k = ts-pow(2,i)-1 ; j < te - 1 ; j +=2, k+=1)
            {
               tree[j] = (tree[k+1]+tree[k])/2;
            }
            ts = te;
         }
         
         ts = 0;
         for(leveli = 0 ; leveli < tree_level ; leveli ++)
         {
            /* search within each level */
            tlen = pow(2,leveli+1) + 1;
            te = ts + tlen;
            
            ns[0] = 0;
            ns[1] = 0;
            ns[2] = 0;
            ns[3] = 0;
            
            vector_SCHURCHEB_long treei;
            std::unordered_map<SCHURCHEB_long, int> tree_idx_hash;
            
            treei.SetupPtr( tree, tlen, ts);
            
            /* before we start, we first update the index of each off-diagonal columns to avoid duplicate search */
            for(i = 0 ; i < n_local ; i ++)
            {
               if(vtxsep[i] == 1)
               {
                  dom = map[i];
                  
                  /* check if we know the index of the local domain */
                  auto find_dom =  tree_idx_hash.find(dom);
                  if(find_dom == tree_idx_hash.end())
                  {
                     /* a new one */
                     if( treei.BinarySearch( dom, idx, true) < 0)
                     {
                        /* In this case, we haven't found it, belongs to the previous inteval */
                        idx--;
                     }
                     
                     tree_idx_hash[dom] = idx;
                  }
                  
                  /* new check nbhds */
                  j1 = xadj[i];
                  j2 = xadj[i+1];
                  for(j = j1 ; j < j2 ; j ++)
                  {
                     
                     col = adjncy[j];
                     id = ids[j]; /* this is the id this col belongs to */
                     
                     if(id != myid)
                     {
                        auto find_col = col_map_uncertain_hash.find(col);
                        domi = send2_v2[id][find_col->second];
                     }
                     else
                     {
                        domi = map[col-n_start];
                     }
                     
                     auto find_dom =  tree_idx_hash.find(domi);
                     if(find_dom == tree_idx_hash.end())
                     {
                        /* a new one */
                        if( treei.BinarySearch( domi, idx, true) < 0)
                        {
                           /* In this case, we haven't found it, belongs to the previous inteval */
                           idx--;
                        }
                        
                        tree_idx_hash[domi] = idx;
                     }
                  }
               }
               else
               {
                  /* interior node, just check the tree location */
                  dom = map[i];
                  
                  /* check if we know the index of the local domain */
                  auto find_dom =  tree_idx_hash.find(dom);
                  if(find_dom == tree_idx_hash.end())
                  {
                     /* a new one */
                     if( treei.BinarySearch( dom, idx, true) < 0)
                     {
                        /* In this case, we haven't found it, belongs to the previous inteval */
                        idx--;
                     }
                     
                     tree_idx_hash[dom] = idx;
                  }
               }
            }
            
            
            /* search all nodes */
            for(i = 0 ; i < n_local ; i ++)
            {
               if(vtxsep[i] == 1)
               {
                  /* this is in the edge separator, and haven't been marked */
                  dom = map[i];
                  
                  auto find_idx = tree_idx_hash.find(dom);
                  idx = find_idx->second;
                  
                  marker = false;
                  
                  if( idx % 2 == 0)
                  {
                     even = true;
                  }
                  else
                  {
                     even = false;
                  }
                  
                  /* even value, this is the LOWER half of a pair */
                  j1 = xadj[i];
                  j2 = xadj[i+1];
                  
                  for(j = j1 ; j < j2 ; j ++)
                  {
                     col = adjncy[j];
                     id = ids[j]; /* this is the id this col belongs to */
                     
                     if(id != myid)
                     {
                        /* only check off-diagonal entries */
                        auto find_col = col_map_uncertain_hash.find(col);
                        domi = send2_v2[id][find_col->second];
                     }
                     else
                     {
                        domi = map[col-n_start];
                     }
                     
                     auto find_idx = tree_idx_hash.find(domi);
                     idxi = find_idx->second;
                     
                     if( (even && idx +1 == idxi) || (!even && idx -1 == idxi) )
                     {
                        /* target col, this node is in the separator */
                        marker = true;
                        break;
                     }
                  }
                  
                  if(marker)
                  {
                     if(even)
                     {
                        vtxsep[i] = 3;
                        ns[0]++;
                     }
                     else
                     {
                        vtxsep[i] = 4;
                        ns[1]++;
                     }
                  }
                  else
                  {
                     if(even)
                     {
                        ns[2]++;
                     }
                     else
                     {
                        ns[3]++;
                     }
                  }
               }
               else if(vtxsep[i] == 0)
               {
                  /* this is the interior nodes */
                  dom = map[i];
                  
                  auto find_idx = tree_idx_hash.find(dom);
                  idx = find_idx->second;
                  
                  if( idx % 2 == 0)
                  {
                     even = true;
                  }
                  else
                  {
                     even = false;
                  }
                  
                  if(even)
                  {
                     ns[2]++;
                  }
                  else
                  {
                     ns[3]++;
                  }
                  
               }
            }
            
            /* now check which to add to separator */
            SCHURCHEB_MPI_CALL(SchurchebMpiAllreduce( ns, nsg, 4, MPI_SUM,comm));
            
            /* main loop done */
            //if(nsg[0] <= nsg[1])
            //if(nsg[3] <= nsg[2])
            if(SchurchebAbs(nsg[2]-nsg[1]-nsg[3]) <= SchurchebAbs(nsg[0]+nsg[2]-nsg[3]))
            {
               /* in this case putting those marked 3 into 2 */
               for(i = 0 ; i < n_local ; i ++)
               {
                  if(vtxsep[i] == 3)
                  {
                     vtxsep[i] = 2;
                  }
                  else if(vtxsep[i] == 4)
                  {
                     vtxsep[i] = 1;
                  }
               }
            }
            else
            {
               /* in this case putting those marked 3 into 2 */
               for(i = 0 ; i < n_local ; i ++)
               {
                  if(vtxsep[i] == 4)
                  {
                     vtxsep[i] = 2;
                  }
                  else if(vtxsep[i] == 3)
                  {
                     vtxsep[i] = 1;
                  }
               }
            } 
            
            tree_idx_hash.clear();
            treei.Clear();
            
            ts = te;
         }
         
         /* adjust marker value */
         for(i = 0 ; i < n_local ; i ++)
         {
            if(vtxsep[i] == 1)
            {
               vtxsep[i] = 0;
            }
            else if(vtxsep[i] == 2)
            {
               vtxsep[i] = 1;
            }
         }
      }
      
      /* -------------------------
       * Step 4: check the separator
       * If some subdomain has no
       * interior nodes, we would
       * have to reject this
       * -------------------------
       */
      
      /* check if some color has no interior nodes */
      
      marker2.Setup(num_dom);
      marker2.Fill(-1);
      
      /* mark local domains */
      for (i = 0; i < n_local; i++)
      {
         /* if found interior, mark to 1 */
         if(vtxsep[i] == 0)
         {
            marker2[map[i]] = 1;
         }
      }
      
      /* check for empty domain */
      SCHURCHEB_MPI_CALL( SchurchebMpiAllreduceInplace( marker2.GetData(), num_dom, MPI_MAX, comm) );
      
      for (i = 0; i < num_dom; i++)
      {
         if(marker2[i] == -1)
         {
            return -1;
         }
      }
      
      /* -------------------------
       * Step 5: now form the 
       * reduced system
       * first get the local vertices
       * -------------------------
       */
      
      n_local_s = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] > 0)
         {
            vtxsep[i] = n_local_s;
            n_local_s++;
         }
         else
         {
            vtxsep[i] = -1;
         }
      }
      
      local_diff = n_local - n_local_s;
      SCHURCHEB_MPI_CALL( SchurchebMpiAllreduce( &local_diff, &global_diff, 1, MPI_MIN, comm) );
      
      if(global_diff == 0)
      {
         /* we have no exterior nodes, stop here */
         return -1;
      }
      
      /* global displacement */
      n_local_ss.Setup(np);
      
      SchurchebMpiAllgather(&n_local_s, 1, n_local_ss.GetData(), comm);
      
      vtxdist_s.Setup(np+1);
      vtxdist_s[0] = 0;
      for(i = 0 ; i < np ; i ++)
      {
         vtxdist_s[i+1] = vtxdist_s[i] + n_local_ss[i];
      }
      
      /* --------------------------------
       * Step 6: get vtxsep info of offds
       * --------------------------------
       */
      
      col_map_uncertain_hash.clear();
      
      sendsize.Fill(0);
      recvsize.Fill(0);
      for(i = 0 ; i < np ; i ++)
      {
         send_v2[i].Clear();
         send_v2[i].Clear();
         send2_v2[i].Clear();
         recv2_v2[i].Clear();
      }
      
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] >= 0)
         {
            /* this is a vtxsep */
            j1 = xadj[i];
            j2 = xadj[i+1];
            
            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */
               
               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(find_col == col_map_uncertain_hash.end())
                  {
                     /* breaking news! a NEW column! */
                     send_v2[id].PushBack(col);
                     col_map_uncertain_hash[col] = sendsize[id];
                     sendsize[id]++;
                  }
               }
            }
         }
      }
      
      /* communicate send and recv size */
      SCHURCHEB_MPI_CALL( MPI_Alltoall( sendsize.GetData(), 1, MPI_INT, recvsize.GetData(), 1, MPI_INT, comm) );
      
      /* then apply communication */
      
      /* first send cols */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            /* myid have data for processor i */
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( send_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv_v2[i].Setup(recvsize[i]);
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( recv_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      SCHURCHEB_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );
      
      /* get the map value */
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv2_v2[i].Setup(recvsize[i]);
            for(j = 0 ; j < recvsize[i]; j ++)
            {
               recv2_v2[i][j] = vtxsep[recv_v2[i][j] - n_start];
            }
         }
      }
      
      /* then apply communication again */
      
      /* or MPI_Alltoallv? */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            /* myid have data for processor i */
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( recv2_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            send2_v2[i].Setup(sendsize[i]);
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( send2_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      SCHURCHEB_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );
      
      /* now adding elements */
      
      xadj_s.Setup(n_local_s+1);
      adjncy_s.Resize(0, false, false);
      
      xadj_s[0] = 0;
      
      ii = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] >= 0)
         {
            xadj_s[ii+1] = xadj_s[ii];
            j1 = xadj[i];
            j2 = xadj[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j];
               
               if(id != myid)
               {
                  /* off-diagonal blocks */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(find_col != col_map_uncertain_hash.end())
                  {
                     idx = send2_v2[id][find_col->second];
                     if(idx >= 0)
                     {
                        adjncy_s.PushBack(idx+vtxdist_s[id]);
                        xadj_s[ii+1]++;
                     }
                  }
               }
               else
               {
                  /* diagonal block */
                  idx = vtxsep[col-n_start];
                  if(idx >= 0)
                  {
                     adjncy_s.PushBack(idx+vtxdist_s[myid]);
                     xadj_s[ii+1]++;
                  }
               }
            }
            ii++;
         }
      }
      
      /* reset vtxsep */
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] >= 0)
         {
            vtxsep[i] = 1;
         }
         else
         {
            vtxsep[i] = -1;
         }
      }
      
      /* deallocate */
      ids.Clear();
      marker2.Clear();
      n_local_ss.Clear();
      sendsize.Clear();
      recvsize.Clear();
      
      for(i = 0 ; i < np ; i ++)
      {
         send_v2[i].Clear();
         recv_v2[i].Clear();
         send2_v2[i].Clear();
         recv2_v2[i].Clear();
      }
      
      std::vector<vector_SCHURCHEB_long>().swap(send_v2);
      std::vector<vector_SCHURCHEB_long>().swap(recv_v2);
      std::vector<vector_int>().swap(send2_v2);
      std::vector<vector_int>().swap(recv2_v2);
      
      col_map_uncertain_hash.clear();
      vector<MPI_Request>().swap(requests);
      
      return SCHURCHEB_SUCCESS;
   }
   
   int ParallelRKwayGetSeparator2( vector_SCHURCHEB_long &vtxdist, vector_SCHURCHEB_long &xadj, vector_SCHURCHEB_long &adjncy, bool vertexsep, vector_SCHURCHEB_long &vtxdist_s,  vector_SCHURCHEB_long &xadj_s,  vector_SCHURCHEB_long &adjncy_s, vector_int &map, int num_dom, vector_int &vtxsep, parallel_log &parlog)
   {
      SCHURCHEB_long                            i, ii, j, j1, j2;
      SCHURCHEB_long                            n_local, n_start, n_end, col, nnz, n_local_s, local_diff, global_diff;
      int                                       id, idx;
      vector_int                                ids, marker2;
      vector_SCHURCHEB_long                     n_local_ss;
      
      std::unordered_map<SCHURCHEB_long, int>   col_map_hash;
      int                                       ncols;
      vector_SCHURCHEB_long                     cols;
      vector_int                                col_ids;
      
      std::unordered_map<SCHURCHEB_long, int>   col_map_uncertain_hash;
      vector_int                                sendsize, recvsize;
      std::vector<vector_SCHURCHEB_long>        send_v2, recv_v2;
      std::vector<vector_int>                   send2_v2, recv2_v2;
      
      MPI_Comm                comm;
      int                     myid, np, numwaits;
      vector<MPI_Request>     requests;
      
      parlog.GetMpiInfo(np, myid, comm);
      
      n_start = vtxdist[myid];
      n_end = vtxdist[myid+1];
      n_local = n_end - n_start;
      
      if(n_local == 0)
      {
         /* empty, no need to setup */
         nnz = 0;
         vtxsep.Clear();
      }
      else
      {
         nnz = xadj[n_local];
         vtxsep.Setup(n_local, true);
         ids.Setup(nnz, true);
      }
      
      /* -------------------------
       * Step 1: put all local columns into the hash table
       * -------------------------
       */
      ncols = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = xadj[i];
         j2 = xadj[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            /* get the processor holds this index 
             * note that there might be duplicate entries 
             */
            auto find_col =  col_map_hash.find(col);
            if(find_col == col_map_hash.end())
            {
               /* a new one */
               if(vtxdist.BinarySearch( col, id, true) < 0)
               {
                  /* in this case, col fall in between */
                  id--;
               }
               
               cols.PushBack(col);
               col_ids.PushBack(id);
               col_map_hash[col] = ncols;
               ncols++;
            }
         }
      }
      
      /* ids[j] is the MPI process adjncy[j] belongs to */
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = xadj[i];
         j2 = xadj[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            /* get the processor holds this index 
             * note that there might be duplicate entries 
             */
            auto find_col =  col_map_hash.find(col);
            ids[j] = col_ids[find_col->second];
         }
      }
      
      requests.resize(2*np);
      sendsize.Setup(np, true);
      recvsize.Setup(np, true);
      send_v2.resize(np);
      recv_v2.resize(np);
      send2_v2.resize(np);
      recv2_v2.resize(np);
      
      /* -------------------------
       * Step 2: check the separator
       * If some subdomain has no
       * interior nodes, we would
       * have to reject this
       * -------------------------
       */
      
      /* check if some color has no interior nodes */
      
      marker2.Setup(num_dom);
      marker2.Fill(-1);
      
      /* mark local domains */
      for (i = 0; i < n_local; i++)
      {
         /* if found interior, mark to 1 */
         if(map[i] >= 0)
         {
            marker2[map[i]] = 1;
         }
      }
      
      /* check for empty domain */
      SCHURCHEB_MPI_CALL( SchurchebMpiAllreduceInplace( marker2.GetData(), num_dom, MPI_MAX, comm) );
      
      for (i = 0; i < num_dom; i++)
      {
         if(marker2[i] == -1)
         {
            return -1;
         }
      }
      
      /* -------------------------
       * Step 2: now form the 
       * reduced system
       * first get the local vertices
       * -------------------------
       */
      
      n_local_s = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         if(map[i] < 0)
         {
            vtxsep[i] = n_local_s;
            n_local_s++;
         }
         else
         {
            vtxsep[i] = -1;
         }
      }
      
      local_diff = n_local - n_local_s;
      SCHURCHEB_MPI_CALL( SchurchebMpiAllreduce( &local_diff, &global_diff, 1, MPI_MIN, comm) );
      
      if(global_diff == 0)
      {
         /* we have no exterior nodes, stop here */
         return -1;
      }
      
      /* global displacement */
      n_local_ss.Setup(np);
      
      SchurchebMpiAllgather(&n_local_s, 1, n_local_ss.GetData(), comm);
      
      vtxdist_s.Setup(np+1);
      vtxdist_s[0] = 0;
      for(i = 0 ; i < np ; i ++)
      {
         vtxdist_s[i+1] = vtxdist_s[i] + n_local_ss[i];
      }
      
      /* --------------------------------
       * Step 3: get vtxsep info of offds
       * --------------------------------
       */
      
      col_map_uncertain_hash.clear();
      
      sendsize.Fill(0);
      recvsize.Fill(0);
      for(i = 0 ; i < np ; i ++)
      {
         send_v2[i].Clear();
         send_v2[i].Clear();
         send2_v2[i].Clear();
         recv2_v2[i].Clear();
      }
      
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] >= 0)
         {
            /* this is a vtxsep */
            j1 = xadj[i];
            j2 = xadj[i+1];
            
            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */
               
               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(find_col == col_map_uncertain_hash.end())
                  {
                     /* breaking news! a NEW column! */
                     send_v2[id].PushBack(col);
                     col_map_uncertain_hash[col] = sendsize[id];
                     sendsize[id]++;
                  }
               }
            }
         }
      }
      
      /* communicate send and recv size */
      SCHURCHEB_MPI_CALL( MPI_Alltoall( sendsize.GetData(), 1, MPI_INT, recvsize.GetData(), 1, MPI_INT, comm) );
      
      /* then apply communication */
      
      /* first send cols */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            /* myid have data for processor i */
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( send_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv_v2[i].Setup(recvsize[i]);
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( recv_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      SCHURCHEB_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );
      
      /* get the map value */
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv2_v2[i].Setup(recvsize[i]);
            for(j = 0 ; j < recvsize[i]; j ++)
            {
               recv2_v2[i][j] = vtxsep[recv_v2[i][j] - n_start];
            }
         }
      }
      
      /* then apply communication again */
      
      /* or MPI_Alltoallv? */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            /* myid have data for processor i */
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( recv2_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            send2_v2[i].Setup(sendsize[i]);
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( send2_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      SCHURCHEB_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );
      
      /* now adding elements */
      
      xadj_s.Setup(n_local_s+1);
      adjncy_s.Resize(0, false, false);
      
      xadj_s[0] = 0;
      
      ii = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] >= 0)
         {
            xadj_s[ii+1] = xadj_s[ii];
            j1 = xadj[i];
            j2 = xadj[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j];
               
               if(id != myid)
               {
                  /* off-diagonal blocks */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(find_col != col_map_uncertain_hash.end())
                  {
                     idx = send2_v2[id][find_col->second];
                     if(idx >= 0)
                     {
                        adjncy_s.PushBack(idx+vtxdist_s[id]);
                        xadj_s[ii+1]++;
                     }
                  }
               }
               else
               {
                  /* diagonal block */
                  idx = vtxsep[col-n_start];
                  if(idx >= 0)
                  {
                     adjncy_s.PushBack(idx+vtxdist_s[myid]);
                     xadj_s[ii+1]++;
                  }
               }
            }
            ii++;
         }
      }
      
      /* reset vtxsep */
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] >= 0)
         {
            vtxsep[i] = 1;
         }
         else
         {
            vtxsep[i] = -1;
         }
      }
      
      /* deallocate */
      ids.Clear();
      marker2.Clear();
      n_local_ss.Clear();
      sendsize.Clear();
      recvsize.Clear();
      
      for(i = 0 ; i < np ; i ++)
      {
         send_v2[i].Clear();
         recv_v2[i].Clear();
         send2_v2[i].Clear();
         recv2_v2[i].Clear();
      }
      
      std::vector<vector_SCHURCHEB_long>().swap(send_v2);
      std::vector<vector_SCHURCHEB_long>().swap(recv_v2);
      std::vector<vector_int>().swap(send2_v2);
      std::vector<vector_int>().swap(recv2_v2);
      
      col_map_uncertain_hash.clear();
      vector<MPI_Request>().swap(requests);
      
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T>
   int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<T> &A, int clvl, int &tlvl, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog)
   {
      SCHURCHEB_long             i, j, n_local, n_start, nA, col, ncomp2;
      int                        err = 0;
      vector_int                 map;
      vector_SCHURCHEB_long      perm_sep;
      vector_int                 map2_v;
      vector_int                 vtxsep;
      ParallelCsrMatrixClass<T>  S;
      
      MPI_Comm          comm;
      int               myid, np;
      parlog.GetMpiInfo(np, myid, comm);
      
      nA = A.GetNumRowsGlobal();
      n_local = A.GetNumRowsLocal();
      n_start = A.GetRowStartGlobal();
      
      /* main loop */
      if (minsep < nA && ncomp <= nA && clvl < tlvl) 
      {

         /* build a k-way partition with vertex separator 
          * perm_sep is the global partition perm_sep = p such that A(p, p) is the global separator
          * map2_v[i] = j if i-th node is in the j-th domain. Domain start from 0, domain k is the edge separator
          */
         ncomp2 = ncomp; 
         
         if( A.GetSeparatorNumSubdomains() == ncomp2)
         {
            /* the partition is provided */
            map = A.GetSeparatorDomi();
            for(i = 0 ; i < n_local ; i ++)
            {
               if(map[i] < 0)
               {
                  map[i] = ncomp;
                  perm_sep.PushBack((SCHURCHEB_long)(n_start + i));
               }
            }
         }
         else
         {
            err = SetupPermutationParallelKwayVertexSep( A, ncomp2, map, perm_sep, parlog); SCHURCHEB_CHKERR(err);
         }
         
         if(ncomp2 < ncomp)
         {
            /* in this case, partition failes, stop here */
            if(bj_last && nA >= np)
            {
               /* treat last level with block-Jacobi like partition 
                * we assign np subdomains to the last level
                */
               
               SCHURCHEB_long          bjdom1, bjdom2, k;
               vector_SCHURCHEB_long   bjdisps, bjns;
               
               bjdom1 = nA / np;
               bjdom2 = nA % np;
               
               bjdisps.Setup(np+1);
               bjns.Setup(np);
               
               for(i = 0 ; i < np ; i ++)
               {
                  if(i < bjdom2)
                  {
                     bjns[i] = bjdom1 + 1;
                  }
                  else
                  {
                     bjns[i] = bjdom1;
                  }
               }
               
               bjdisps[0] = 0;
               for(i = 0 ; i < np ; i ++)
               {
                  bjdisps[i+1] = bjdisps[i] + bjns[i];
               }
               
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+np);
               map_v.Setup(n_local);
               
               j = 0;
               for(i = 0, k = n_start; i < n_local ; i ++, k++)
               {
                  while(bjdisps[j+1] <= k)
                  {
                     /* in this case, current node blongs to the next one 
                      * we want k in [bjdisps[j], bjdisps[j+1])
                      * example: 0, 2, 4, 6, 8, when k = 2, j should be 2.
                      */
                     j++;
                  }
                  map_v[i] = j+col;
               }
            }
            else
            {
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+1);
               map_v.Setup(n_local);
               map_v.Fill(col);
            }
            
            tlvl = clvl;
         }
         else
         {
            /* prepare the recursive call */
            A.SubMatrix( perm_sep, perm_sep, kMemoryHost, S);
            
            mapptr_v.PushBack(mapptr_v.Back()+ncomp2);
            
            if(ncomp > kmin)
            {
               ncomp = ncomp / kfactor;
               if(ncomp < kmin)
               {
                  ncomp = kmin;
               }
            }
            
            // (Increase clvl by 1)
            SetupPermutationParallelRKwayRecursive2( S, clvl+1, tlvl, ncomp, minsep, kmin, kfactor, map2_v, mapptr_v, bj_last, parlog);
            
            S.Clear();      
            
            /* udpate map information */
            map_v.Setup(n_local);
            j = 0;
            for(i = 0 ; i < n_local ; i ++)
            {
               if(map[i] < ncomp2)
               {
                  /* interior nodes */
                  map_v[i] = map[i] + mapptr_v[clvl-1];
               }
               else
               {
                  /* exterior nodes */
                  map_v[i] = map2_v[j++];
               }
            }
            
         }
      }
      else
      {
         /* in this case, we don't have enough domains, stop here 
          * a special case is when nA == 0
          */
         if(nA > 0)
         {
            if(bj_last && nA >= np)
            {
               /* treat last level with block-Jacobi like partition 
                * we assign np subdomains to the last level
                */
               
               SCHURCHEB_long          bjdom1, bjdom2, k;
               vector_SCHURCHEB_long   bjdisps, bjns;
               
               bjdom1 = nA / np;
               bjdom2 = nA % np;
               
               bjdisps.Setup(np+1);
               bjns.Setup(np);
               
               for(i = 0 ; i < np ; i ++)
               {
                  if(i < bjdom2)
                  {
                     bjns[i] = bjdom1 + 1;
                  }
                  else
                  {
                     bjns[i] = bjdom1;
                  }
               }
               
               bjdisps[0] = 0;
               for(i = 0 ; i < np ; i ++)
               {
                  bjdisps[i+1] = bjdisps[i] + bjns[i];
               }
               
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+np);
               map_v.Setup(n_local);
               
               j = 0;
               for(i = 0, k = n_start; i < n_local ; i ++, k++)
               {
                  while(bjdisps[j+1] <= k)
                  {
                     /* in this case, current node blongs to the next one 
                      * we want k in [bjdisps[j], bjdisps[j+1])
                      * example: 0, 2, 4, 6, 8, when k = 2, j should be 2.
                      */
                     j++;
                  }
                  map_v[i] = j+col;
               }
            }
            else
            {
               col = mapptr_v.Back();
               mapptr_v.PushBack(col+1);
               map_v.Setup(n_local);
               map_v.Fill(col);
            }
            
            tlvl = clvl;
         }
         else
         {
            tlvl = clvl-1;
         }
      }
      
      return err;
   }
   template int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<float> &A, int clvl, int &tlvl, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog);
   template int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<double> &A, int clvl, int &tlvl, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog);
   template int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<complexs> &A, int clvl, int &tlvl, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog);
   template int SetupPermutationParallelRKwayRecursive2(ParallelCsrMatrixClass<complexd> &A, int clvl, int &tlvl, SCHURCHEB_long ncomp, SCHURCHEB_long minsep, SCHURCHEB_long kmin, SCHURCHEB_long kfactor, vector_int &map_v, vector_int &mapptr_v, bool bj_last, parallel_log &parlog);
   
   template <typename T>
   int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<T> &A, SCHURCHEB_long &ncomp, vector_int &map_v, vector_SCHURCHEB_long &perm_sep, parallel_log &parlog)
   {
      int            n_local, i, nd_tlvl, nd_clvl, ncomp_temp;
      SCHURCHEB_long n_start;
      bool           succeed;
      
      /* now find 2^k = num_dom */
      ncomp_temp = ncomp;
      
      SCHURCHEB_CHKERR( ( ( ncomp_temp ) & ( ncomp_temp - 1 ) ) != 0 );
      
      nd_tlvl = 0;
      while (ncomp_temp > 0)
      {
         ncomp_temp = ncomp_temp >> 1;
         nd_tlvl++;
      }
      
      /* start the ND ordering */
      nd_clvl = 0;
      SetupPermutationParallelKwayVertexSepRecursive(A, nd_clvl, nd_tlvl, succeed, map_v, parlog);
      
      if(!succeed)
      {
         /* partition fails */
         ncomp = 0;
      }
      else
      {
         perm_sep.Resize(0, false, false);
         n_local = A.GetNumRowsLocal();
         n_start = A.GetRowStartGlobal();
         for(i = 0 ; i < n_local ; i ++)
         {
            SCHURCHEB_CHKERR( map_v[i] > ncomp);
            if(map_v[i] == ncomp)
            {
               /* this is an edge cut */
               perm_sep.PushBack((SCHURCHEB_long)(n_start + i));
            }
         }
      }
      
      return SCHURCHEB_SUCCESS;
   }
   template int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<float> &A, SCHURCHEB_long &ncomp, vector_int &map_v, vector_SCHURCHEB_long &perm_sep, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<double> &A, SCHURCHEB_long &ncomp, vector_int &map_v, vector_SCHURCHEB_long &perm_sep, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<complexs> &A, SCHURCHEB_long &ncomp, vector_int &map_v, vector_SCHURCHEB_long &perm_sep, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSep( ParallelCsrMatrixClass<complexd> &A, SCHURCHEB_long &ncomp, vector_int &map_v, vector_SCHURCHEB_long &perm_sep, parallel_log &parlog);
   
   template <typename T>
   int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<T> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog)
   {
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("Recursive kway partition only works for host matrices.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      /* at least should have two levels */
      SCHURCHEB_CHKERR(tlvl < 2);
      
      if(clvl == 0)
      {
         /* on the root level, set succeed to true */
         succeed = true;
      }
      
      int                              i, j, j1, j2, err = 0, idx;
      int                              n_local, nnz_local, ncomp, lev_shift;
      SCHURCHEB_long                   n_start, two, ndom1, ndom2, edge_cut;
      vector_SCHURCHEB_long            vtxdist, xadj, adjncy, dom1, dom2, map;
      vector_int                       vtxsep, map1_v, map2_v, idx_v;
      ParallelCsrMatrixClass<T>        A1, A2;
      
      MPI_Comm                         comm;
      int                              myid, np;
      
      A.GetMpiInfo(np, myid, comm);
      
      CsrMatrixClass<T> &A_diag           = A.GetDiagMat();
      CsrMatrixClass<T> &A_offd           = A.GetOffdMat();
      vector_SCHURCHEB_long &offd_map_v   = A.GetOffdMap();
      
      n_start                             = A.GetRowStartGlobal();
      n_local                             = A.GetNumRowsLocal();           
      nnz_local                           = A_diag.GetNumNonzeros() + A_offd.GetNumNonzeros();
      
      /* set vtxdist */
      vtxdist.Setup(np+1);
      vtxdist[np] = A.GetNumRowsGlobal();
      
      SchurchebMpiAllgather(&n_start, 1, vtxdist.GetData(), comm);
      
      /* xadj and adjncy */
      xadj.Setup(n_local+1);
      adjncy.Setup(nnz_local-n_local);
      
      /* Costruct a CSR-like representation of A as required by METIS. Extract and keep the diagonal entries separately */ 
      
      int *A_diag_i = A_diag.GetI();
      int *A_diag_j = A_diag.GetJ();
      int *A_offd_i = A_offd.GetI();
      int *A_offd_j = A_offd.GetJ();
      SCHURCHEB_long *offd_map = offd_map_v.GetData();
      
      xadj[0] = 0;
      for (i = 0; i < n_local; i++)
      {
         xadj[i+1] = xadj[i];
         j1 = A_diag_i[i];
         j2 = A_diag_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            if(A_diag_j[j] != i)
            {
               adjncy[xadj[i+1]] = A_diag_j[j] + n_start;
               xadj[i+1]++;
            }
         }
         
         j1 = A_offd_i[i];
         j2 = A_offd_i[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            adjncy[xadj[i+1]] = offd_map[A_offd_j[j]];
            xadj[i+1]++;
         }
      }
      
      /* now, call a two-way partition of the current matrix */
      two = 2;
      
      err = ParmetisKwayHost( vtxdist, xadj, adjncy, two, map, parlog);
      
      if(two < 2 || err)
      {
         /* in this case, we can't have two subdomains, stop here
          * reset error message, this is not a critical error
          */
         succeed = false;
         err = SCHURCHEB_SUCCESS;
      }
      else
      {
         /* in this case, we build the separator */
         ParallelNDGetSeparator(vtxdist, xadj, adjncy, map, ndom1, ndom2, edge_cut, parlog);
         
         if(ndom1 == 0 || ndom2 == 0)
         {
            succeed = false;
         }
         else
         {
            if(clvl == tlvl-2)
            {
               /* no more levels, mark vtxsep and done */
               ncomp = (int)pow(2.0, (double)(tlvl-1));
               
               map_v.Setup(n_local);
               for(i = 0 ; i < n_local ; i ++)
               {
                  switch(map[i])
                  {
                     case 0:
                     {
                        /* the first subdomain */
                        map_v[i] = 0;
                        break;
                     }
                     case 1:
                     {
                        /* the second subdomain */
                        map_v[i] = 1;
                        break;
                     }
                     default:
                     {
                        /* the separator */
                        map_v[i] = ncomp;
                        break;
                     }
                  }
               }
            }
            else
            {
            
               /* build the dom1, dom2, and idx_v */
            
               idx_v.Setup(n_local);
               dom1.Setup(0, n_local/2, kMemoryHost, false);
               dom2.Setup(0, n_local/2, kMemoryHost, false);
               j1 = 0;
               j2 = 0;
               for(i = 0 ; i < n_local ; i ++)
               {
                  switch(map[i])
                  {
                     case 0:
                     {
                        /* this node belongs to the first subdomain */
                        dom1.PushBack(i+n_start);
                        idx_v[i] = j1++;
                        break;
                     }
                     case 1:
                     {
                        /* this node belongs to the second subdomain */
                        dom2.PushBack(i+n_start);
                        idx_v[i] = j2++;
                        break;
                     }
                     default:
                     {
                        /* the separator, do nothing */
                        break;
                     }
                  }
               }
               
               /* get the first and the second  */
               A.SubMatrix(dom1, dom1, kMemoryHost, A1);
               A.SubMatrix(dom2, dom2, kMemoryHost, A2);
               
               SetupPermutationParallelKwayVertexSepRecursive(A1, clvl+1, tlvl, succeed, map1_v, parlog);
               
               if(succeed)
               {
                  SetupPermutationParallelKwayVertexSepRecursive(A2, clvl+1, tlvl, succeed, map2_v, parlog);
               }
               
               if(succeed)
               {
                  /* the above two recursive partition succeed, proceed to update the local map 
                   * 
                   *       - - - 
                   *      /     \
                   *     -       -   
                   *    / \     / \
                   *   -   -   -   -
                   *  / \ / \ / \ / \
                   *  1 2 3 4 5 6 7 8
                   * 
                   * the returned mapping infomation
                   * 
                   */
                  
                  /* nlev has 2^{nlev-1} comps */
                  ncomp = (int)pow(2.0, (double)(tlvl-1));
                  
                  /* apply the shift 
                   * for example, on the third lst level when tlvl - clvl = 2
                   * 0 1 0 1 -> 0 1 2 3, the shift is 2.
                   */
                  lev_shift = (int)pow(2.0, (double)(tlvl-clvl-2));
                  
                  map_v.Setup(n_local);
                  for(i = 0 ; i < n_local ; i ++)
                  {
                     switch(map[i])
                     {
                        case 0:
                        {
                           /* the first subdomain */
                           idx = map1_v[idx_v[i]];
                           if(idx == ncomp)
                           {
                              /* in the saperator */
                              map_v[i] =ncomp;
                           }
                           else
                           {
                              /* keep the map */
                              map_v[i] = idx;
                           }
                           break;
                        }
                        case 1:
                        {
                           /* the second subdomain */
                           idx = map2_v[idx_v[i]];
                           if(idx == ncomp)
                           {
                              /* in the saperator */
                              map_v[i] = ncomp;
                           }
                           else
                           {
                              /* keep the map plus a shift */
                              map_v[i] = idx + lev_shift;
                           }
                           break;
                        }
                        default:
                        {
                           /* the separator */
                           map_v[i] = ncomp;
                           break;
                        }
                     }
                  }
                  
               }
            }
         }
      }/* end of else for ncomp < 2 */
      
      /* deallocate */
      vtxdist.Clear();
      xadj.Clear();
      adjncy.Clear();
      dom1.Clear();
      dom2.Clear();
      map.Clear();
      map1_v.Clear();
      map2_v.Clear();
      idx_v.Clear();
      A1.Clear();
      A2.Clear();
      
      return err;
   }
   template int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<float> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<double> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<complexs> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog);
   template int SetupPermutationParallelKwayVertexSepRecursive(ParallelCsrMatrixClass<complexd> &A, int clvl, int tlvl, bool &succeed, vector_int &map_v, parallel_log &parlog);
   
   int ParallelNDGetSeparator( vector_SCHURCHEB_long &vtxdist, vector_SCHURCHEB_long &xadj, vector_SCHURCHEB_long &adjncy, vector_SCHURCHEB_long &map, SCHURCHEB_long &ndom1, SCHURCHEB_long &ndom2, SCHURCHEB_long &edge_cut, parallel_log &parlog)
   {
      SCHURCHEB_long                            i, j, j1, j2;
      SCHURCHEB_long                            n_local, n_start, n_end, dom, col, nnz;
      int                                       id;
      vector_int                                ids, vtxsep;
      
      std::unordered_map<SCHURCHEB_long, int>   col_map_hash;
      int                                       ncols;
      vector_SCHURCHEB_long                     cols;
      vector_int                                col_ids;
      
      std::unordered_map<SCHURCHEB_long, int>   col_map_uncertain_hash;
      vector_int                                sendsize, recvsize;
      std::vector<vector_SCHURCHEB_long>        send_v2, recv_v2;
      std::vector<vector_int>                   send2_v2, recv2_v2;
      
      MPI_Comm                   comm;
      int                        myid, np, numwaits;
      vector<MPI_Request>        requests;
      
      parlog.GetMpiInfo(np, myid, comm);
      
      n_start = vtxdist[myid];
      n_end = vtxdist[myid+1];
      n_local = n_end - n_start;
      
      if(n_local == 0)
      {
         /* empty, no need to setup */
         nnz = 0;
         vtxsep.Clear();
      }
      else
      {
         nnz = xadj[n_local];
         vtxsep.Setup(n_local, true);
         ids.Setup(nnz, true);
      }
      
      /* -------------------------
       * Step 1: put all local columns into the hash table
       * -------------------------
       */
      ncols = 0;
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = xadj[i];
         j2 = xadj[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            /* get the processor holds this index 
             * note that there might be duplicate entries 
             */
            auto find_col =  col_map_hash.find(col);
            if(find_col == col_map_hash.end())
            {
               /* a new one */
               if(vtxdist.BinarySearch( col, id, true) < 0)
               {
                  /* in this case, col fall in between */
                  id--;
               }
               
               cols.PushBack(col);
               col_ids.PushBack(id);
               col_map_hash[col] = ncols;
               ncols++;
            }
         }
      }
      
      /* ids[j] is the MPI process adjncy[j] belongs to */
      for(i = 0 ; i < n_local ; i ++)
      {
         j1 = xadj[i];
         j2 = xadj[i+1];
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            /* get the processor holds this index 
             * note that there might be duplicate entries 
             */
            auto find_col =  col_map_hash.find(col);
            ids[j] = col_ids[find_col->second];
         }
      }
      
      /* -------------------------
       * Step 2: check local first
       * some rows/cols requires 
       * accessing offdiagonal entries
       * -------------------------
       */
      
      /* This loop find local vtxsep
       * 1. local cols already has multiple maps:
       *    => set vtxsep to 1, separator.
       * 2. all cols are local, and all same map values:
       *    => set vtxsep to 0, interior.
       * 3. all local cols same map, however, have exterior cols
       *    => set vtxsep to -1, TBD
       */
      for(i = 0 ; i < n_local ; i ++)
      {
         /* dom is the domain of the row */
         dom = map[i];
         j1 = xadj[i];
         j2 = xadj[i+1];
         
         for(j = j1 ; j < j2 ; j ++)
         {
            col = adjncy[j];
            id = ids[j]; /* this is the id this col belongs to */
            
            /* we can only check the map value of local parts */
            if(id == myid)
            {
               col -= n_start;
               if(map[col] != dom)
               {
                  /* this is diagonal, exterior node */
                  vtxsep[i] = 1;
                  break;
               }
            }
            else
            {
               /* if has offd, mark to -1 instead */
               vtxsep[i] = -1;
            }
         }
      }
      
      /* -------------------------
       * Step 3: check remaining
       * -------------------------
       */
      
      sendsize.Setup(np, true);
      recvsize.Setup(np, true);
      send_v2.resize(np);
      recv_v2.resize(np);
      send2_v2.resize(np);
      recv2_v2.resize(np);
      
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] == -1)
         {
            /* this is a target column */
            j1 = xadj[i];
            j2 = xadj[i+1];
            
            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */
               
               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(find_col == col_map_uncertain_hash.end())
                  {
                     /* breaking news! a NEW column! */
                     send_v2[id].PushBack(col);
                     col_map_uncertain_hash[col] = sendsize[id];
                     sendsize[id]++;
                  }
               }
            }
         }
      }
      
      /* communicate send and recv size */
      SCHURCHEB_MPI_CALL( MPI_Alltoall( sendsize.GetData(), 1, MPI_INT, recvsize.GetData(), 1, MPI_INT, comm) );
      
      /* then apply communication */
      
      requests.resize(2*np);
      
      /* first send cols */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            /* myid have data for processor i */
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( send_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv_v2[i].Setup(recvsize[i]);
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( recv_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      SCHURCHEB_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );
      
      /* get the map value */
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            recv2_v2[i].Setup(recvsize[i]);
            for(j = 0 ; j < recvsize[i]; j ++)
            {
               recv2_v2[i][j] = map[recv_v2[i][j] - n_start];
            }
         }
      }
      
      /* then apply communication again */
      
      /* or MPI_Alltoallv? */
      numwaits = 0;
      for(i = 0 ; i < np ; i ++)
      {
         if(recvsize[i] > 0)
         {
            /* myid have data for processor i */
            SCHURCHEB_MPI_CALL( SchurchebMpiIsend( recv2_v2[i].GetData(), recvsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      for(i = 0 ; i < np ; i ++)
      {
         if(sendsize[i] > 0)
         {
            send2_v2[i].Setup(sendsize[i]);
            SCHURCHEB_MPI_CALL( SchurchebMpiIrecv( send2_v2[i].GetData(), sendsize[i], i, 0, comm, &(requests[numwaits++])) );
         }
      }
      
      SCHURCHEB_MPI_CALL( MPI_Waitall( numwaits, requests.data(), MPI_STATUSES_IGNORE) );
      
      /* finally check the received columns */
      
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] == -1)
         {
            /* this is a target column */
            dom = map[i];
            j1 = xadj[i];
            j2 = xadj[i+1];
            
            for(j = j1 ; j < j2 ; j ++)
            {
               col = adjncy[j];
               id = ids[j]; /* this is the id this col belongs to */
               
               if(id != myid)
               {
                  /* only check off-diagonal entries */
                  auto find_col = col_map_uncertain_hash.find(col);
                  if(send2_v2[id][find_col->second] != dom)
                  {
                     vtxsep[i] = 1;
                     break;
                  }
               }
            }
            
            /* if still -1, this is interior */
            if(vtxsep[i] == -1)
            {
               vtxsep[i] = 0;
            }
         }
      }
      
      /* now form the reduced system */
      SCHURCHEB_long ndom1_local = 0, ndom2_local = 0, ec1_local = 0, ec2_local = 0, ec1, ec2;
      for(i = 0 ; i < n_local ; i ++)
      {
         if(vtxsep[i] > 0)
         {
            /* this is in the seperator */
            if(map[i] == 0)
            {
               ec1_local++;
            }
            else
            {
               ec2_local++;
               map[i] = 1;
            }
         }
         else
         {
            if(map[i] == 0)
            {
               ndom1_local++;
            }
            else
            {
               ndom2_local++;
               map[i] = 1;
            }
         }
      }
      
      SCHURCHEB_MPI_CALL(SchurchebMpiAllreduce( &ndom1_local, &ndom1, 1, MPI_SUM,comm));
      SCHURCHEB_MPI_CALL(SchurchebMpiAllreduce( &ndom2_local, &ndom2, 1, MPI_SUM,comm));
      SCHURCHEB_MPI_CALL(SchurchebMpiAllreduce( &ec1_local, &ec1, 1, MPI_SUM,comm));
      SCHURCHEB_MPI_CALL(SchurchebMpiAllreduce( &ec2_local, &ec2, 1, MPI_SUM,comm));
      
      if(ec2 > ec1)
      {
         edge_cut = ec2;
         for(i = 0 ; i < n_local ; i ++)
         {
            if(vtxsep[i] > 0)
            {
               /* this is in the seperator */
               if(map[i] == 1)
               {
                  map[i] = 2;
               }
            }
         }
      }
      else
      {
         edge_cut = ec1;
         for(i = 0 ; i < n_local ; i ++)
         {
            if(vtxsep[i] > 0)
            {
               /* this is in the seperator */
               if(map[i] == 0)
               {
                  map[i] = 2;
               }
            }
         }
      }
      
      /* deallocate */
      ids.Clear();
      sendsize.Clear();
      recvsize.Clear();
      
      for(i = 0 ; i < np ; i ++)
      {
         send_v2[i].Clear();
         recv_v2[i].Clear();
         send2_v2[i].Clear();
         recv2_v2[i].Clear();
      }
      
      std::vector<vector_SCHURCHEB_long>().swap(send_v2);
      std::vector<vector_SCHURCHEB_long>().swap(recv_v2);
      std::vector<vector_int>().swap(send2_v2);
      std::vector<vector_int>().swap(recv2_v2);
      
      col_map_uncertain_hash.clear();
      vector<MPI_Request>().swap(requests);
      
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T>
   int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<T> &A, bool vertexsep, int &nlev, SCHURCHEB_long minsep, vector_int &map_v, vector_int &mapptr_v)
   {
      if(A.GetDataLocation() == kMemoryDevice)
      {
         SCHURCHEB_ERROR("ND partition only works for host matrices.");
         return SCHURCHEB_ERROR_MEMORY_LOCATION;
      }
      
      int                              i, j, k, j1, j2, err = 0;
      int                              n_local, nnz_local, clvl, tlvl, domi, size1, size2, ndom_temp;
      SCHURCHEB_long                   n_start, ndom;
      vector_SCHURCHEB_long            vtxdist, xadj, adjncy, map_long;
      
      MPI_Comm                         comm;
      int                              myid, np;
      
      A.GetMpiInfo(np, myid, comm);
      
      CsrMatrixClass<T> &A_diag           = A.GetDiagMat();
      CsrMatrixClass<T> &A_offd           = A.GetOffdMat();
      vector_SCHURCHEB_long &offd_map_v   = A.GetOffdMap();
      
      n_start                             = A.GetRowStartGlobal();
      n_local                             = A.GetNumRowsLocal();           
      nnz_local                           = A_diag.GetNumNonzeros() + A_offd.GetNumNonzeros();
      
      if(np > 1)
      {
      
         /* set vtxdist */
         vtxdist.Setup(np+1);
         vtxdist[np] = A.GetNumRowsGlobal();

         SchurchebMpiAllgather(&n_start, 1, vtxdist.GetData(), comm);
         
         /* xadj and adjncy */
         xadj.Setup(n_local+1);
         adjncy.Setup(nnz_local-n_local);
         
         /* Costruct a CSR-like representation of A as required by METIS. Extract and keep the diagonal entries separately */ 
         
         int *A_diag_i = A_diag.GetI();
         int *A_diag_j = A_diag.GetJ();
         int *A_offd_i = A_offd.GetI();
         int *A_offd_j = A_offd.GetJ();
         SCHURCHEB_long *offd_map = offd_map_v.GetData();
         
         xadj[0] = 0;
         for (i = 0; i < n_local; i++)
         {
            xadj[i+1] = xadj[i];
            j1 = A_diag_i[i];
            j2 = A_diag_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               if(A_diag_j[j] != i)
               {
                  adjncy[xadj[i+1]] = A_diag_j[j] + n_start;
                  xadj[i+1]++;
               }
            }
            
            j1 = A_offd_i[i];
            j2 = A_offd_i[i+1];
            for(j = j1 ; j < j2 ; j ++)
            {
               adjncy[xadj[i+1]] = offd_map[A_offd_j[j]];
               xadj[i+1]++;
            }
         }
         
         /* call metis ND */
         ParmetisNodeND(vtxdist, xadj, adjncy, ndom, map_long, A);
         
         nlev = 0;
         ndom_temp = ndom + 1;
         while (ndom_temp > 1)
         {
            ndom_temp = ndom_temp >> 1;
            nlev++;
         }
         
         map_v.Setup(n_local);
         mapptr_v.Setup(nlev+1, true);
         
         for(i = 0 ; i < n_local ; i ++)
         {
            map_v[i] = map_long[i];
         }
         
         mapptr_v[0] = 0;
         j = (ndom + 1)/2;
         for(i = 1 ; i < nlev ; i ++)
         {
            mapptr_v[i] = mapptr_v[i-1] + j;
            j/=2;
         }
         mapptr_v[nlev] = ndom;
         
      }
      else
      {
         std::vector<std::vector<vector_int> > level_str;
         
         /* in this case, only a single processor, apply the sequential one */
         clvl = 0;
         tlvl = nlev;
         err = SetupPermutationNDRecursive( A_diag, vertexsep, clvl, tlvl, minsep, level_str); SCHURCHEB_CHKERR(err);
         
         /* 4. prepare return value */
         map_v.Setup(n_local);
         mapptr_v.Setup(tlvl+1);
         
         /* update the final level */
         nlev = tlvl;
         
         domi = 0;
         mapptr_v[0] = 0;
         for(i = 0 ; i < nlev ; i ++)
         {
            size1 = level_str[i].size();
            for(j = 0 ; j < size1 ; j ++)
            {
               size2 = level_str[i][j].GetLengthLocal();
               for(k = 0 ; k < size2 ; k ++)
               {
                  map_v[level_str[i][j][k]] = domi;
               }
               domi++;
            }
            mapptr_v[i+1] = domi;
         }
         
         /* free */
         for(i = 0 ; i < nlev ; i ++)
         {
            size1 = level_str[i].size();
            for(j = 0 ; j < size1 ; j ++)
            {
               level_str[i][j].Clear();
            }
            std::vector<vector_int>().swap(level_str[i]);
         }
         std::vector<std::vector<vector_int> >().swap(level_str);
         
      }
      
      return err;
   }
   template int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<float> &A, bool vertexsep, int &nlev, SCHURCHEB_long minsep, vector_int &map_v, vector_int &mapptr_v);
   template int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<double> &A, bool vertexsep, int &nlev, SCHURCHEB_long minsep, vector_int &map_v, vector_int &mapptr_v);
   template int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<complexs> &A, bool vertexsep, int &nlev, SCHURCHEB_long minsep, vector_int &map_v, vector_int &mapptr_v);
   template int ParallelCsrMatrixSetupPermutationParallelND( ParallelCsrMatrixClass<complexd> &A, bool vertexsep, int &nlev, SCHURCHEB_long minsep, vector_int &map_v, vector_int &mapptr_v);
   
   template <typename T>
   int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<T> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<T> &B_mat, CsrMatrixClass<T> &E_mat, CsrMatrixClass<T> &F_mat, ParallelCsrMatrixClass<T> &C_mat, int perm_option, bool perm_c)
   {
      
      int                     i, j, n_local, nE, nsends, nsendsi, s, e;
      SCHURCHEB_long          n_start;
      vector_int              marker;
      vector_SCHURCHEB_long   exterior_row;
      
      MPI_Comm comm;
      int np, myid;
      
      parcsr_in.GetMpiInfo(np, myid, comm);
      
      parcsr_in.SetupMatvecStart();
      n_local = parcsr_in.GetNumRowsLocal();
      n_start = parcsr_in.GetRowStartGlobal();
      
      ParallelCsrMatrixClass<T> &A = parcsr_in;
      CsrMatrixClass<T> &A_diag = A.GetDiagMat();
      //CsrMatrixClass<DataType> &A_offd = A.GetOffdMat();
      
      int *A_offd_i = parcsr_in.GetOffdMat().GetI();
      
      /* start to update marker while waiting for the result */
      marker.Setup(n_local);
      marker.Fill(-1);
      
      /* mark local  */
      for(i = 0 ; i < n_local ; i ++)
      {
         /* check if this is an external node */
         if(A_offd_i[i] < A_offd_i[i+1])
         {
            marker[i] = 0;
         }
      }
      
      parcsr_in.SetupMatvecOver();
      
      /* now check col */
      nsends = (int) parcsr_in._comm_helper._send_idx_v2.size();
      for(i = 0 ; i < nsends ; i++)
      {
         if(parcsr_in._comm_helper._send_to_v[i] != myid)
         {
            nsendsi = parcsr_in._comm_helper._send_idx_v2[i].GetLengthLocal();
            for(j = 0 ; j < nsendsi ; j ++)
            {
               marker[parcsr_in._comm_helper._send_idx_v2[i][j]] = 0;
            }
         }
      }
      
      /* on host */
      local_perm.Setup(n_local);
      s = 0;
      e = n_local - 1;
      for(i = 0 ; i < n_local ; i ++)
      {
         if(marker[i] < 0)
         {
            /* interior node */
            local_perm[s++] = i;
         }
         else
         {
            /* exterior node */
            local_perm[e--] = i;
         }
      }
      
      nI = s;
      nE = n_local - s;
      
      /* apply RCM/AMD */
      switch(perm_option)
      {
         case kIluReorderingNo:
         {
            vector_int temp_perm1, temp_perm2;
            
            temp_perm1.SetupPtr(local_perm, nI, 0);
            temp_perm2.SetupPtr(local_perm, nE, nI);
            
            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, B_mat);
            A_diag.SubMatrix(temp_perm2, temp_perm1, kMemoryHost, E_mat);
            A_diag.SubMatrix(temp_perm1, temp_perm2, kMemoryHost, F_mat);
            
            break;
         }
         case kIluReorderingRcm:
         {
            vector_int temp_perm1, temp_perm2, rcm_order;
            CsrMatrixClass<T> Temp_diag;
            
            temp_perm1.SetupPtr(local_perm, nI, 0);
            temp_perm2.SetupPtr(local_perm, nE, nI);
            
            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, Temp_diag);
            
            CsrMatrixRcmHost( Temp_diag, rcm_order);
            
            temp_perm1.Perm(rcm_order);
            Temp_diag.Clear();
            rcm_order.Clear();
            
            if(perm_c)
            {
               A_diag.SubMatrix(temp_perm2, temp_perm2, kMemoryHost, Temp_diag);
            
               CsrMatrixRcmHost( Temp_diag, rcm_order);
               
               temp_perm2.Perm(rcm_order);
               Temp_diag.Clear();
               rcm_order.Clear();
            }
            
            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, B_mat);
            A_diag.SubMatrix(temp_perm2, temp_perm1, kMemoryHost, E_mat);
            A_diag.SubMatrix(temp_perm1, temp_perm2, kMemoryHost, F_mat);
            
            break;
         }
         case kIluReorderingAmd:
         {
            vector_int temp_perm1, temp_perm2, rcm_order;
            CsrMatrixClass<T> Temp_diag;
            
            temp_perm1.SetupPtr(local_perm, nI, 0);
            temp_perm2.SetupPtr(local_perm, nE, nI);
            
            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, Temp_diag);
            
            CsrMatrixAmdHost( Temp_diag, rcm_order);
            
            temp_perm1.Perm(rcm_order);
            Temp_diag.Clear();
            rcm_order.Clear();
            
            if(perm_c)
            {
               A_diag.SubMatrix(temp_perm2, temp_perm2, kMemoryHost, Temp_diag);
            
               CsrMatrixAmdHost( Temp_diag, rcm_order);
               
               temp_perm2.Perm(rcm_order);
               Temp_diag.Clear();
               rcm_order.Clear();
            }
            
            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, B_mat);
            A_diag.SubMatrix(temp_perm2, temp_perm1, kMemoryHost, E_mat);
            A_diag.SubMatrix(temp_perm1, temp_perm2, kMemoryHost, F_mat);
            
            break;
         }
         case kIluReorderingNd:
         {
            vector_int temp_perm1, temp_perm2, rcm_order;
            CsrMatrixClass<T> Temp_diag;
            
            temp_perm1.SetupPtr(local_perm, nI, 0);
            temp_perm2.SetupPtr(local_perm, nE, nI);
            
            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, Temp_diag);
            
            CsrMatrixNdHost( Temp_diag, rcm_order);
            
            temp_perm1.Perm(rcm_order);
            Temp_diag.Clear();
            rcm_order.Clear();
            
            if(perm_c)
            {
               A_diag.SubMatrix(temp_perm2, temp_perm2, kMemoryHost, Temp_diag);
            
               CsrMatrixNdHost( Temp_diag, rcm_order);
               
               temp_perm2.Perm(rcm_order);
               Temp_diag.Clear();
               rcm_order.Clear();
            }
            
            A_diag.SubMatrix(temp_perm1, temp_perm1, kMemoryHost, B_mat);
            A_diag.SubMatrix(temp_perm2, temp_perm1, kMemoryHost, E_mat);
            A_diag.SubMatrix(temp_perm1, temp_perm2, kMemoryHost, F_mat);
            
            break;
         }
         default:
         {
            SCHURCHEB_ERROR("Unknown local reordering option.");
            return SCHURCHEB_ERROR_INVALED_OPTION;
         }
      }
      
      exterior_row.Setup(nE);
      
      for(i = 0 ; i < nE ; i ++)
      {
         exterior_row[i] = (SCHURCHEB_long)local_perm[++e] + n_start;
      }
      
      parcsr_in.SubMatrix( exterior_row, exterior_row, kMemoryHost, C_mat);
      
      C_mat.SortOffdMap();
      
      exterior_row.Clear();
      
      return SCHURCHEB_SUCCESS;
   }
   template int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<float> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<float> &B_mat, CsrMatrixClass<float> &E_mat, CsrMatrixClass<float> &F_mat, ParallelCsrMatrixClass<float> &C_mat, int perm_option, bool perm_c);
   template int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<double> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<double> &B_mat, CsrMatrixClass<double> &E_mat, CsrMatrixClass<double> &F_mat, ParallelCsrMatrixClass<double> &C_mat, int perm_option, bool perm_c);
   template int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<complexs> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<complexs> &B_mat, CsrMatrixClass<complexs> &E_mat, CsrMatrixClass<complexs> &F_mat, ParallelCsrMatrixClass<complexs> &C_mat, int perm_option, bool perm_c);
   template int ParallelCsrMatrixSetupIOOrder(ParallelCsrMatrixClass<complexd> &parcsr_in, vector_int &local_perm, int &nI, CsrMatrixClass<complexd> &B_mat, CsrMatrixClass<complexd> &E_mat, CsrMatrixClass<complexd> &F_mat, ParallelCsrMatrixClass<complexd> &C_mat, int perm_option, bool perm_c);
   
}

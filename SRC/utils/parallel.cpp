

#include <assert.h>
#include "parallel.hpp"
#include "utils.hpp"
#include "memory.hpp"
#include "protos.hpp"

namespace schurcheb
{
   /* variables */
#ifdef SCHURCHEB_CUDA
   
   curandGenerator_t parallel_log::_curand_gen = NULL;
   cublasHandle_t parallel_log::_cublas_handle = NULL;
   cusparseHandle_t parallel_log::_cusparse_handle = NULL;
   cudaStream_t parallel_log::_stream = 0;
   cusparseIndexBase_t parallel_log::_cusparse_idx_base = CUSPARSE_INDEX_BASE_ZERO;
   cusparseMatDescr_t parallel_log::_mat_des = NULL;
   cusparseMatDescr_t parallel_log::_matL_des = NULL;
   cusparseMatDescr_t parallel_log::_matU_des = NULL;
   //cusparseSolvePolicy_t parallel_log::_ilu_solve_policy = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
   cusparseSolvePolicy_t parallel_log::_ilu_solve_policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
   void* parallel_log::_cusparse_buffer = NULL;
   size_t parallel_log::_cusparse_buffer_length = 0;

#if (SCHURCHEB_CUDA_VERSION == 11)

   cusparseIndexType_t parallel_log::_cusparse_idx_type = CUSPARSE_INDEX_32I;
   cusparseSpMVAlg_t parallel_log::_cusparse_spmv_algorithm = CUSPARSE_CSRMV_ALG1;

#endif
#endif

   int parallel_log::_working_location = kMemoryHost;
   int parallel_log::_gsize = 0;
   int parallel_log::_grank = 0;
   MPI_Comm* parallel_log::_gcomm = NULL;
   MPI_Comm* parallel_log::_lcomm = NULL;
   
   int parallel_log::Clear()
   {
      if(this->_comm != NULL)
      {
#ifdef SCHURCHEB_DEBUG
         if(parallel_log::_gcomm == NULL)
         {
            SCHURCHEB_ERROR("Free MPI_Comm after MPI_Finalize().");
         }
#endif
         SCHURCHEB_MPI_CALL( MPI_Comm_free(this->_comm) );
         SCHURCHEB_FREE(this->_comm, kMemoryHost);
      }
      this->_commref = MPI_COMM_WORLD;
      return SCHURCHEB_SUCCESS;
   }
   
   parallel_log::ParallelLogClass()
   {
      this->_commref = MPI_COMM_WORLD;
      this->_comm = NULL;
   }
   
   parallel_log::ParallelLogClass(const ParallelLogClass &parlog)
   {
      this->_commref = parlog._commref;
      this->_size = parlog._size;
      this->_rank = parlog._rank;
      if(parlog._comm)
      {
         SCHURCHEB_MALLOC(this->_comm, 1, kMemoryHost, MPI_Comm);
         SCHURCHEB_MPI_CALL( (MPI_Comm_dup(*(parlog._comm), this->_comm)) );
      }
      else
      {
         this->_comm = NULL;
      }
   }
   
   parallel_log::ParallelLogClass( ParallelLogClass &&parlog)
   {
      this->_commref = parlog._commref;
      parlog._commref = MPI_COMM_WORLD;
      this->_comm = parlog._comm;
      parlog._comm = NULL;
      this->_rank = parlog._rank;
      parlog._rank = 0;
      this->_size = parlog._size;
      parlog._size = 0;
   }
   
   ParallelLogClass& parallel_log::operator= (const ParallelLogClass &parlog)
   {
      this->Clear();
      this->_size = parlog._size;
      this->_rank = parlog._rank;
      this->_commref = parlog._commref;
      if(parlog._comm)
      {
         SCHURCHEB_MALLOC(this->_comm, 1, kMemoryHost, MPI_Comm);
         SCHURCHEB_MPI_CALL( (MPI_Comm_dup(*(parlog._comm), this->_comm)) );
      }
      else
      {
         this->_comm = NULL;
      }
      return *this;
   }
   
   ParallelLogClass& parallel_log::operator= ( ParallelLogClass &&parlog)
   {
      this->Clear();
      this->_commref = parlog._commref;
      parlog._commref = MPI_COMM_WORLD;
      this->_comm = parlog._comm;
      parlog._comm = NULL;
      this->_rank = parlog._rank;
      parlog._rank = 0;
      this->_size = parlog._size;
      parlog._size = 0;
      return *this;
   }
   
   parallel_log::ParallelLogClass(MPI_Comm comm_in)
   {
      /* can only be called after we call the MPI_Init */
      SCHURCHEB_CHKERR(parallel_log::_gcomm == NULL);
      SCHURCHEB_MALLOC( this->_comm, 1, kMemoryHost, MPI_Comm);
      SCHURCHEB_MPI_CALL( MPI_Comm_dup(comm_in, this->_comm) );
      SCHURCHEB_MPI_CALL( MPI_Comm_size(comm_in, &(this->_size)) );
      SCHURCHEB_MPI_CALL( MPI_Comm_rank(comm_in, &(this->_rank)) );
      this->_commref = MPI_COMM_WORLD;
   }
   
   parallel_log::~ParallelLogClass()
   {
      Clear();
   }
   
   int parallel_log::GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const
   {
      
      if(this->_comm == NULL)
      {
         /* use the global one */
         if(this->_commref != MPI_COMM_WORLD)
         {
            comm = this->_commref;
            np    = this->_size;
            myid  = this->_rank;
         }
         else
         {
            comm  = *(parallel_log::_gcomm);
            np    = parallel_log::_gsize;
            myid  = parallel_log::_grank;
         }
      }
      else
      {
         /* use this one */
         comm  = *(this->_comm);
         np    = this->_size;
         myid  = this->_rank;
      }
      
      return SCHURCHEB_SUCCESS;
      
   }
   
   MPI_Comm parallel_log::GetComm() const
   {
      
      if(this->_comm == NULL)
      {
         /* use the global one */
         if(this->_commref != MPI_COMM_WORLD)
         {
            return this->_commref;
         }
         else
         {
            return  *(parallel_log::_gcomm);
         }
      }
      else
      {
         /* use this one */
         return  *(this->_comm);
      }
      
      return SCHURCHEB_SUCCESS;
      
   }
   
   int SchurchebSetOpenmpNumThreads(int nthreads)
   {
#ifdef SCHURCHEB_OPENMP
      omp_set_num_threads(nthreads);
#endif
#ifdef SCHURCHEB_OPENBLAS
      openblas_set_num_threads(nthreads);
#endif
#ifdef SCHURCHEB_MKL
      mkl_set_num_threads(nthreads);
#endif
      return SCHURCHEB_SUCCESS;
   }
   
#ifdef SCHURCHEB_OPENMP

   int SchurchebGetOpenmpThreadNum()
   {
      return omp_get_thread_num();
   }
   
   int SchurchebGetOpenmpNumThreads()
   {
      return omp_get_num_threads();
   }
   
   int SchurchebGetOpenmpMaxNumThreads()
   {
      /* if already inside a parallel region, 
       * omp_get_max_threads() == omp_get_num_threads()
       * and thus this function returns 1, avoid nested OpenMP call.
       */
      return omp_get_max_threads()/omp_get_num_threads();
   }
   
   int SchurchebGetOpenmpGlobalMaxNumThreads()
   {
      return omp_get_max_threads();
   }
   
#endif

   int SchurchebNLocalToNGlobal( int n_local, SCHURCHEB_long &n_start, SCHURCHEB_long &n_global, MPI_Comm &comm)
   {
      
      int      np;
      SCHURCHEB_MPI_CALL(MPI_Comm_size(comm, &np));
      
      SCHURCHEB_long n_local_long = (SCHURCHEB_long) n_local;
      
      /* after scan, n_start is the exact n_start plus n_local */
      SchurchebMpiScan( &n_local_long, &n_start, 1, MPI_SUM, comm);
      
      /* now, the n_global on the last MPI rank is the exact one, bcast it */
      n_global = n_start;
      SchurchebMpiBcast( &n_global, 1, np-1, comm);
      
      /* shift back to get n_start */
      n_start -= n_local_long;
      
      return SCHURCHEB_SUCCESS;
   }
   
   int SchurchebNLocalToNGlobal( int nrow_local, int ncol_local, SCHURCHEB_long &nrow_start, SCHURCHEB_long &ncol_start, SCHURCHEB_long &nrow_global, SCHURCHEB_long &ncol_global, MPI_Comm &comm)
   {
      
      int      np;
      SCHURCHEB_MPI_CALL(MPI_Comm_size(comm, &np));
      
      SCHURCHEB_long *pbuffer;
      SCHURCHEB_MALLOC( pbuffer, 6, kMemoryHost, SCHURCHEB_long);
      
      pbuffer[0] = (SCHURCHEB_long) nrow_local;
      pbuffer[1] = (SCHURCHEB_long) ncol_local;
      
      /* after scan, n_start is the exact n_start plus n_local */
      SchurchebMpiScan( pbuffer, pbuffer+2, 2, MPI_SUM, comm);
      
      /* now, the n_global on the last MPI rank is the exact one, bcast it */
      pbuffer[4] = pbuffer[2];
      pbuffer[5] = pbuffer[3];
      SchurchebMpiBcast( pbuffer+4, 2, np-1, comm);
      
      /* shift back to get n_start */
      pbuffer[2] -= pbuffer[0];
      pbuffer[3] -= pbuffer[1];
      
      nrow_start = pbuffer[2];
      ncol_start = pbuffer[3];
      nrow_global = pbuffer[4];
      ncol_global = pbuffer[5];
      
      SCHURCHEB_FREE( pbuffer, kMemoryHost);
      
      return SCHURCHEB_SUCCESS;
   }
   
   int SchurchebMpiTime(MPI_Comm comm, double &t)
   {
      SCHURCHEB_CUDA_SYNCHRONIZE;
      SCHURCHEB_MPI_CALL( MPI_Barrier(comm) );
      t = MPI_Wtime();
      return SCHURCHEB_SUCCESS;
   }
   
   int SchurchebInit(int *argc, char ***argv)
   {
      int size, rank, nthreads = 1;
      /* if this is not null, we have called the init */
      if(parallel_log::_gcomm != NULL)
      {
         return SCHURCHEB_ERROR_DOUBLE_INIT_FREE;
      }
      
      /* init MPI */
      SCHURCHEB_MPI_CALL( MPI_Init(argc, argv) );
      
      /* We do not directly use MPI_COMM_WORLD */
      SCHURCHEB_MALLOC( parallel_log::_gcomm, 1, kMemoryHost, MPI_Comm);
      SCHURCHEB_MPI_CALL( MPI_Comm_dup(MPI_COMM_WORLD, parallel_log::_gcomm) );
      
      SCHURCHEB_MPI_CALL( MPI_Comm_size( *(parallel_log::_gcomm), &size) );
      SCHURCHEB_MPI_CALL( MPI_Comm_rank( *(parallel_log::_gcomm), &rank) );
      
      parallel_log::_gsize = size;
      parallel_log::_grank = rank;
      
      SCHURCHEB_MALLOC( parallel_log::_lcomm, 1, kMemoryHost, MPI_Comm);
      SCHURCHEB_MPI_CALL( MPI_Comm_split(MPI_COMM_WORLD, rank, rank, parallel_log::_lcomm) );
      
      /* prepare openmp and MKL */
      SchurchebReadInputArg( "nthreads", 1, &nthreads, *argc, *argv);

      SchurchebInitOpenMP(nthreads);

      SchurchebInitCUDA();
      
      return SCHURCHEB_SUCCESS;
   }
   
   int SchurchebInitMpi(MPI_Comm comm)
   {
      int size, rank;
      /* We do not directly use MPI_COMM_WORLD */
      SCHURCHEB_MALLOC( parallel_log::_gcomm, 1, kMemoryHost, MPI_Comm);
      SCHURCHEB_MPI_CALL( MPI_Comm_dup(comm, parallel_log::_gcomm) );
      
      SCHURCHEB_MPI_CALL( MPI_Comm_size( *(parallel_log::_gcomm), &size) );
      SCHURCHEB_MPI_CALL( MPI_Comm_rank( *(parallel_log::_gcomm), &rank) );
      
      parallel_log::_gsize = size;
      parallel_log::_grank = rank;
      
      SCHURCHEB_MALLOC( parallel_log::_lcomm, 1, kMemoryHost, MPI_Comm);
      SCHURCHEB_MPI_CALL( MPI_Comm_split(comm, rank, rank, parallel_log::_lcomm) );
      
      return SCHURCHEB_SUCCESS;
      
   }
   
   int SchurchebInitOpenMP(int nthreads)
   {
#ifdef SCHURCHEB_OPENMP     
      omp_set_num_threads(nthreads);
#endif
#ifdef SCHURCHEB_OPENBLAS
      openblas_set_num_threads(nthreads);
#endif
#ifdef SCHURCHEB_MKL
      mkl_set_num_threads(nthreads);
      /* set to 0 to fix the thread on each node to be nthreads, otherwise MKL might change this number during runtime */
      mkl_set_dynamic(1);
#endif
      return SCHURCHEB_SUCCESS;
   }

   int SchurchebInitCUDA()
   {

#ifdef SCHURCHEB_CUDA

      /* with CUDA enabled, the default working location is the device */
      parallel_log::_working_location = kMemoryDevice;
      
      /* split comm by shared memory region to know the number of GPU availiable in each range */
      
      int dc;
      SCHURCHEB_CUDA_CALL( (cudaGetDeviceCount(&dc)) );
      
      if(dc == 0 && parallel_log::_grank == 0)
      {
         printf("Error: no availiable device for MPI rank %d.\n", parallel_log::_grank == 0);
         exit(0);
      }
      
#ifdef SCHURCHEB_DEBUG
      printf("MPI rank %d have access to %d GPUs, set to %d/%d of them.\n", parallel_log::_grank, dc, parallel_log::_grank%dc, dc);
#endif   

      /* now set the device for each processor */
      SCHURCHEB_CUDA_CALL( (cudaSetDevice(parallel_log::_grank % dc)) );
      
      /* create new non-blocking stream 
       * stream does not synchronize with stream 0
       */
      //SCHURCHEB_CUDA_CALL( (cudaStreamCreateWithFlags(&(parallel_log::_stream), cudaStreamNonBlocking)) );
      parallel_log::_stream = 0;
      
      /* create new cublas handle */
      SCHURCHEB_CUBLAS_CALL( (cublasCreate(&(parallel_log::_cublas_handle))) );
      
      /* create new cusparse handle */
      SCHURCHEB_CUSPARSE_CALL( (cusparseCreate(&(parallel_log::_cusparse_handle))) );
      
      /* bind stream */
      SCHURCHEB_CUBLAS_CALL( (cublasSetStream(parallel_log::_cublas_handle, parallel_log::_stream)) );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetStream(parallel_log::_cusparse_handle, parallel_log::_stream)) );
      
      /* cublas setup */
      
      /* CUBLAS_POINTER_MODE_HOST, pass alpha/beta from the host, return value to the host would block the cuda */
      SCHURCHEB_CUBLAS_CALL( (cublasSetPointerMode(parallel_log::_cublas_handle, CUBLAS_POINTER_MODE_HOST)) );
      
      /* cusparse setup */
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetPointerMode(parallel_log::_cusparse_handle, CUSPARSE_POINTER_MODE_HOST)) );
      
      /* create cusparse matrix descriptor */
      SCHURCHEB_CUSPARSE_CALL( (cusparseCreateMatDescr(&(parallel_log::_mat_des))) );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetMatIndexBase(parallel_log::_mat_des, CUSPARSE_INDEX_BASE_ZERO)) );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetMatType(parallel_log::_mat_des, CUSPARSE_MATRIX_TYPE_GENERAL)) );
      
      SCHURCHEB_CUSPARSE_CALL( (cusparseCreateMatDescr(&(parallel_log::_matL_des)))  );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetMatIndexBase(parallel_log::_matL_des, CUSPARSE_INDEX_BASE_ZERO)) );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetMatType(parallel_log::_matL_des, CUSPARSE_MATRIX_TYPE_GENERAL)) );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetMatFillMode(parallel_log::_matL_des, CUSPARSE_FILL_MODE_LOWER)) );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetMatDiagType(parallel_log::_matL_des, CUSPARSE_DIAG_TYPE_UNIT)) );
      
      SCHURCHEB_CUSPARSE_CALL( (cusparseCreateMatDescr(&(parallel_log::_matU_des))) );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetMatIndexBase(parallel_log::_matU_des, CUSPARSE_INDEX_BASE_ZERO)) );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetMatType(parallel_log::_matU_des, CUSPARSE_MATRIX_TYPE_GENERAL)) );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetMatFillMode(parallel_log::_matU_des, CUSPARSE_FILL_MODE_UPPER)) );
      SCHURCHEB_CUSPARSE_CALL( (cusparseSetMatDiagType(parallel_log::_matU_des, CUSPARSE_DIAG_TYPE_NON_UNIT)) );
  
      /* ilu solving policy */
      parallel_log::_ilu_solve_policy        = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
      parallel_log::_cusparse_buffer_length  = 0;
      parallel_log::_cusparse_buffer         = NULL;
      
#if (SCHURCHEB_CUDA_VERSION == 11)

      /* cusparse general API */
      int size_of_int = sizeof(int);
      switch(size_of_int)
      {
         case 4:
         {
            /* int 32 */
            parallel_log::_cusparse_idx_type = CUSPARSE_INDEX_32I;
            break;
         }
         case 8:
         {
            /* int 64 */
            parallel_log::_cusparse_idx_type = CUSPARSE_INDEX_64I;
            break;
         }
         default:
         {
            return SCHURCHEB_ERROR_COMPILER;
            break;
         }
      }
      parallel_log::_cusparse_spmv_algorithm = CUSPARSE_CSRMV_ALG1;
      
#endif
    
      
      /* curand */
      SCHURCHEB_CURAND_CALL( curandCreateGenerator(&(parallel_log::_curand_gen), CURAND_RNG_PSEUDO_DEFAULT) );
      SCHURCHEB_CURAND_CALL( curandSetStream(parallel_log::_curand_gen, parallel_log::_stream) );
      
#endif

      return SCHURCHEB_SUCCESS;
   }

   int SchurchebPrintParallelInfo()
   {
      if(parallel_log::_grank == 0)
      {
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
         printf("Printing parallel information\n");
         
         printf("MPI Info:\n");
         printf("\tNumber of MPI Ranks: %d\n", parallel_log::_gsize);
#ifdef SCHURCHEB_OPENMP
         printf("OPENMP Info:\n");
         printf("\tMaxinum Number of OpenMP Threads Per Node: %d\n",omp_get_max_threads());
#endif
#ifdef SCHURCHEB_CUDA
         printf("GPU Info:\n");
         printf("\tGPU Enabled\n");
#endif
#ifdef SCHURCHEB_MKL
         printf("MKL Info:\n");
         printf("\tMaxinum Number of OpenMP Threads Per Node: %d\n",mkl_get_max_threads());
         printf("\tMKL Dynamic Enabled\n");
#endif
         
         SchurchebPrintDashLine(SCHURCHEB_global::_dash_line_width);
      }
      return SCHURCHEB_SUCCESS;
   }
   
   int SchurchebFinalize()
   {
      /* if this is null, we have free the value */
      if(parallel_log::_gcomm == NULL)
      {
         return SCHURCHEB_ERROR_DOUBLE_INIT_FREE;
      }
      SCHURCHEB_MPI_CALL( MPI_Comm_free(parallel_log::_gcomm) );
      SCHURCHEB_FREE(parallel_log::_gcomm, kMemoryHost);
      SCHURCHEB_MPI_CALL( MPI_Comm_free(parallel_log::_lcomm) );
      SCHURCHEB_FREE(parallel_log::_lcomm, kMemoryHost);
      
      SCHURCHEB_MPI_CALL( MPI_Finalize() );
      
      SchurchebFinalizeCUDA();
      
      /* close output */
      if( SCHURCHEB_global::_out_file != stdout )
      {
         /* free the current */
         fclose(SCHURCHEB_global::_out_file);
         SCHURCHEB_global::_out_file = stdout;
      }
      
      
      return SCHURCHEB_SUCCESS;
   }
   
   int SchurchebFinalizeMpi()
   {
      if(parallel_log::_gcomm == NULL)
      {
         return SCHURCHEB_ERROR_DOUBLE_INIT_FREE;
      }
      SCHURCHEB_MPI_CALL( MPI_Comm_free(parallel_log::_gcomm) );
      SCHURCHEB_FREE(parallel_log::_gcomm, kMemoryHost);
      SCHURCHEB_MPI_CALL( MPI_Comm_free(parallel_log::_lcomm) );
      SCHURCHEB_FREE(parallel_log::_lcomm, kMemoryHost);
      
      return SCHURCHEB_SUCCESS;
   }
   
   int SchurchebFinalizeOpenMP()
   {
      return SCHURCHEB_SUCCESS;
   }
   
   int SchurchebFinalizeCUDA()
   {
      
#ifdef SCHURCHEB_CUDA
      /* free handles and data structures */
      
      SCHURCHEB_CURAND_CALL( curandDestroyGenerator( parallel_log::_curand_gen) );
      parallel_log::_curand_gen = NULL;
      
      SCHURCHEB_CUBLAS_CALL( (cublasDestroy(parallel_log::_cublas_handle)) );
      parallel_log::_cublas_handle = NULL;
         
      SCHURCHEB_CUSPARSE_CALL( (cusparseDestroyMatDescr(parallel_log::_mat_des)) );
      parallel_log::_mat_des = NULL;
      
      SCHURCHEB_CUSPARSE_CALL( (cusparseDestroyMatDescr(parallel_log::_matL_des)) );
      parallel_log::_matL_des = NULL;
      
      SCHURCHEB_CUSPARSE_CALL( (cusparseDestroyMatDescr(parallel_log::_matU_des)) );
      parallel_log::_matU_des = NULL;
      
      SCHURCHEB_CUSPARSE_CALL( (cusparseDestroy(parallel_log::_cusparse_handle)) );
      parallel_log::_cusparse_handle = NULL;
      
      parallel_log::_cusparse_buffer_length = 0;
      SCHURCHEB_FREE(parallel_log::_cusparse_buffer, kMemoryDevice);
      
#endif
      
      return SCHURCHEB_SUCCESS;
   }

#ifdef MPI_C_FLOAT_COMPLEX
   
   template <typename T>
   int SchurchebMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request)
   {
      SCHURCHEB_MPI_CALL( MPI_Isend( buf, count, SchurchebMpiDataType<T>(), dest, tag, comm, request) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiIsend(int *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIsend(long int *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIsend(float *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIsend(double *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIsend(complexs *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIsend(complexd *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   int SchurchebMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request)
   {
      SCHURCHEB_MPI_CALL( MPI_Irecv( buf, count, SchurchebMpiDataType<T>(), source, tag, comm, request) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiIrecv(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIrecv(long int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIrecv(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIrecv(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIrecv(complexs *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIrecv(complexd *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   int SchurchebMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Send( buf, count, SchurchebMpiDataType<T>(), dest, tag, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiSend(int *buf, int count, int dest, int tag, MPI_Comm comm);
   template int SchurchebMpiSend(long int *buf, int count, int dest, int tag, MPI_Comm comm);
   template int SchurchebMpiSend(float *buf, int count, int dest, int tag, MPI_Comm comm);
   template int SchurchebMpiSend(double *buf, int count, int dest, int tag, MPI_Comm comm);
   template int SchurchebMpiSend(complexs *buf, int count, int dest, int tag, MPI_Comm comm);
   template int SchurchebMpiSend(complexd *buf, int count, int dest, int tag, MPI_Comm comm);
   
   template <typename T>
   int SchurchebMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status)
   {
      SCHURCHEB_MPI_CALL( MPI_Recv( buf, count, SchurchebMpiDataType<T>(), source, tag, comm, status) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiRecv(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiRecv(long int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiRecv(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiRecv(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiRecv(complexs *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiRecv(complexd *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status *status);
   
   template <typename T1, typename T2>
   int SchurchebMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
   {
      SCHURCHEB_MPI_CALL( MPI_Sendrecv( sendbuf, sendcount, SchurchebMpiDataType<T1>(), dest, sendtag, recvbuf, recvcount, SchurchebMpiDataType<T2>(), source, recvtag, comm, status) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiSendRecv(int *sendbuf, int sendcount, int dest, int sendtag, int *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiSendRecv(long int *sendbuf, int sendcount, int dest, int sendtag, long int *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiSendRecv(float *sendbuf, int sendcount, int dest, int sendtag, float *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiSendRecv(double *sendbuf, int sendcount, int dest, int sendtag, double *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiSendRecv(complexs *sendbuf, int sendcount, int dest, int sendtag, complexs *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiSendRecv(complexd *sendbuf, int sendcount, int dest, int sendtag, complexd *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
#else
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request)
   {
      SCHURCHEB_MPI_CALL( MPI_Isend( buf, count, SchurchebMpiDataType<T>(), dest, tag, comm, request) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiIsend(int *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIsend(long int *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIsend(float *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIsend(double *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request)
   {
      SCHURCHEB_MPI_CALL( MPI_Isend( buf, 2*count, SchurchebMpiDataType<T>(), dest, tag, comm, request) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiIsend(complexs *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIsend(complexd *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request)
   {
      SCHURCHEB_MPI_CALL( MPI_Irecv( buf, count, SchurchebMpiDataType<T>(), source, tag, comm, request) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiIrecv(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIrecv(long int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIrecv(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIrecv(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request)
   {
      SCHURCHEB_MPI_CALL( MPI_Irecv( buf, 2*count, SchurchebMpiDataType<T>(), source, tag, comm, request) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiIrecv(complexs *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   template int SchurchebMpiIrecv(complexd *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Send( buf, count, SchurchebMpiDataType<T>(), dest, tag, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiSend(int *buf, int count, int dest, int tag, MPI_Comm comm);
   template int SchurchebMpiSend(long int *buf, int count, int dest, int tag, MPI_Comm comm);
   template int SchurchebMpiSend(float *buf, int count, int dest, int tag, MPI_Comm comm);
   template int SchurchebMpiSend(double *buf, int count, int dest, int tag, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Send( buf, 2*count, SchurchebMpiDataType<T>(), dest, tag, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiSend(complexs *buf, int count, int dest, int tag, MPI_Comm comm);
   template int SchurchebMpiSend(complexd *buf, int count, int dest, int tag, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status)
   {
      SCHURCHEB_MPI_CALL( MPI_Recv( buf, count, SchurchebMpiDataType<T>(), source, tag, comm, status) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiRecv(int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   template int SchurchebMpiRecv(long int *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   template int SchurchebMpiRecv(float *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   template int SchurchebMpiRecv(double *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status)
   {
      SCHURCHEB_MPI_CALL( MPI_Recv( buf, 2*count, SchurchebMpiDataType<T>(), source, tag, comm, status) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiRecv(complexs *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   template int SchurchebMpiRecv(complexd *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   
   template <typename T1, typename T2>
   typename std::enable_if<!SchurchebIsComplex<T1>::value&&!SchurchebIsComplex<T2>::value, int>::type
   int SchurchebMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
   {
      SCHURCHEB_MPI_CALL( MPI_Sendrecv( sendbuf, sendcount, SchurchebMpiDataType<T1>(), dest, sendtag, recvbuf, recvcount, SchurchebMpiDataType<T2>(), source, recvtag, comm, status) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiSendRecv(int *sendbuf, int sendcount, int dest, int sendtag, int *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiSendRecv(long int *sendbuf, int sendcount, int dest, int sendtag, long int *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiSendRecv(float *sendbuf, int sendcount, int dest, int sendtag, float *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiSendRecv(double *sendbuf, int sendcount, int dest, int sendtag, double *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
   template <typename T1, typename T2>
   typename std::enable_if<!SchurchebIsComplex<T1>::value&&SchurchebIsComplex<T2>::value, int>::type
   int SchurchebMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
   {
      SCHURCHEB_MPI_CALL( MPI_Sendrecv( sendbuf, sendcount, SchurchebMpiDataType<T1>(), dest, sendtag, recvbuf, 2*recvcount, SchurchebMpiDataType<T2>(), source, recvtag, comm, status) );
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T1, typename T2>
   typename std::enable_if<SchurchebIsComplex<T1>::value&&!SchurchebIsComplex<T2>::value, int>::type
   int SchurchebMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
   {
      SCHURCHEB_MPI_CALL( MPI_Sendrecv( sendbuf, 2*sendcount, SchurchebMpiDataType<T1>(), dest, sendtag, recvbuf, recvcount, SchurchebMpiDataType<T2>(), source, recvtag, comm, status) );
      return SCHURCHEB_SUCCESS;
   }
   
   template <typename T1, typename T2>
   typename std::enable_if<SchurchebIsComplex<T1>::value&&SchurchebIsComplex<T2>::value, int>::type
   int SchurchebMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status)
   {
      SCHURCHEB_MPI_CALL( MPI_Sendrecv( sendbuf, 2*sendcount, SchurchebMpiDataType<T1>(), dest, sendtag, recvbuf, 2*recvcount, SchurchebMpiDataType<T2>(), source, recvtag, comm, status) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiSendRecv(complexs *sendbuf, int sendcount, int dest, int sendtag, complexs *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   template int SchurchebMpiSendRecv(complexd *sendbuf, int sendcount, int dest, int sendtag, complexd *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX
   
   template <typename T>
   int SchurchebMpiBcast(T *buf, int count, int root, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Bcast( buf, count, SchurchebMpiDataType<T>(), root, comm ) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiBcast(int *buf, int count, int root, MPI_Comm comm);
   template int SchurchebMpiBcast(long int *buf, int count, int root, MPI_Comm comm);
   template int SchurchebMpiBcast(float *buf, int count, int root, MPI_Comm comm);
   template int SchurchebMpiBcast(double *buf, int count, int root, MPI_Comm comm);
   template int SchurchebMpiBcast(complexs *buf, int count, int root, MPI_Comm comm);
   template int SchurchebMpiBcast(complexd *buf, int count, int root, MPI_Comm comm);
   
#else
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiBcast(T *buf, int count, int root, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Bcast( buf, count, SchurchebMpiDataType<T>(), root, comm ) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiBcast(int *buf, int count, int root, MPI_Comm comm);
   template int SchurchebMpiBcast(long int *buf, int count, int root, MPI_Comm comm);
   template int SchurchebMpiBcast(float *buf, int count, int root, MPI_Comm comm);
   template int SchurchebMpiBcast(double *buf, int count, int root, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiBcast(T *buf, int count, int root, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Bcast( buf, 2*count, SchurchebMpiDataType<T>(), root, comm ) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiBcast(complexs *buf, int count, int root, MPI_Comm comm);
   template int SchurchebMpiBcast(complexd *buf, int count, int root, MPI_Comm comm);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int SchurchebMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Scan( sendbuf, recvbuf, count, SchurchebMpiDataType<T>(), op, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiScan(int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiScan(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiScan(float *sendbuf, float *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiScan(double *sendbuf, double *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiScan(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiScan(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, MPI_Comm comm);

#else

   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Scan( sendbuf, recvbuf, count, SchurchebMpiDataType<T>(), op, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiScan(int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiScan(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiScan(float *sendbuf, float *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiScan(double *sendbuf, double *recvbuf, int count, MPI_Op op, MPI_Comm comm);


   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Scan( sendbuf, recvbuf, 2*count, SchurchebMpiDataType<T>(), op, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiScan(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiScan(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, MPI_Comm comm);


#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int SchurchebMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Reduce( sendbuf, recvbuf, count, SchurchebMpiDataType<T>(), op, root, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiReduce(int *sendbuf, int *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int SchurchebMpiReduce(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int SchurchebMpiReduce(float *sendbuf, float *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int SchurchebMpiReduce(double *sendbuf, double *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int SchurchebMpiReduce(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int SchurchebMpiReduce(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   
#else

   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Reduce( sendbuf, recvbuf, count, SchurchebMpiDataType<T>(), op, root, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiReduce(int *sendbuf, int *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int SchurchebMpiReduce(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int SchurchebMpiReduce(float *sendbuf, float *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int SchurchebMpiReduce(double *sendbuf, double *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Reduce( sendbuf, recvbuf, 2*count, SchurchebMpiDataType<T>(), op, root, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiReduce(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);
   template int SchurchebMpiReduce(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int SchurchebMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allreduce( sendbuf, recvbuf, count, SchurchebMpiDataType<T>(), op, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllreduce(int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduce(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduce(float *sendbuf, float *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduce(double *sendbuf, double *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduce(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduce(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
#else

   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allreduce( sendbuf, recvbuf, count, SchurchebMpiDataType<T>(), op, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllreduce(int *sendbuf, int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduce(long int *sendbuf, long int *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduce(float *sendbuf, float *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduce(double *sendbuf, double *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allreduce( sendbuf, recvbuf, 2*count, SchurchebMpiDataType<T>(), op, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllreduce(complexs *sendbuf, complexs *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduce(complexd *sendbuf, complexd *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int SchurchebMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allreduce( MPI_IN_PLACE, buf, count, SchurchebMpiDataType<T>(), op, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllreduceInplace(int *rbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduceInplace(long int *buf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduceInplace(float *buf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduceInplace(double *buf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduceInplace(complexs *buf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduceInplace(complexd *buf, int count, MPI_Op op, MPI_Comm comm);
   
#else

   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allreduce( MPI_IN_PLACE, buf, count, SchurchebMpiDataType<T>(), op, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllreduceInplace(int *rbuf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduceInplace(long int *buf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduceInplace(float *buf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduceInplace(double *buf, int count, MPI_Op op, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allreduce( MPI_IN_PLACE, buf, 2*count, SchurchebMpiDataType<T>(), op, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllreduceInplace(complexs *buf, int count, MPI_Op op, MPI_Comm comm);
   template int SchurchebMpiAllreduceInplace(complexd *buf, int count, MPI_Op op, MPI_Comm comm);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int SchurchebMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Gather( sendbuf, count, SchurchebMpiDataType<T>(), recvbuf, count, SchurchebMpiDataType<T>(), root, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiGather(int *sendbuf, int count, int *recvbuf, int root, MPI_Comm comm);
   template int SchurchebMpiGather(long int *sendbuf, int count, long int *recvbuf, int root, MPI_Comm comm);
   template int SchurchebMpiGather(float *sendbuf, int count, float *recvbuf, int root, MPI_Comm comm);
   template int SchurchebMpiGather(double *sendbuf, int count, double *recvbuf, int root, MPI_Comm comm);
   template int SchurchebMpiGather(complexs *sendbuf, int count, complexs *recvbuf, int root, MPI_Comm comm);
   template int SchurchebMpiGather(complexd *sendbuf, int count, complexd *recvbuf, int root, MPI_Comm comm);
   
#else
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Gather( sendbuf, count, SchurchebMpiDataType<T>(), recvbuf, count, SchurchebMpiDataType<T>(), root, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiGather(int *sendbuf, int count, int *recvbuf, int root, MPI_Comm comm);
   template int SchurchebMpiGather(long int *sendbuf, int count, long int *recvbuf, int root, MPI_Comm comm);
   template int SchurchebMpiGather(float *sendbuf, int count, float *recvbuf, int root, MPI_Comm comm);
   template int SchurchebMpiGather(double *sendbuf, int count, double *recvbuf, int root, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Gather( sendbuf, 2*count, SchurchebMpiDataType<T>(), recvbuf, 2*count, SchurchebMpiDataType<T>(), root, comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiGather(complexs *sendbuf, int count, complexs *recvbuf, int root, MPI_Comm comm);
   template int SchurchebMpiGather(complexd *sendbuf, int count, complexd *recvbuf, int root, MPI_Comm comm);
     
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int SchurchebMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allgather( sendbuf, count, SchurchebMpiDataType<T>(),recvbuf, count, SchurchebMpiDataType<T>(), comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllgather(int *sendbuf, int count, int *recvbuf, MPI_Comm comm);
   template int SchurchebMpiAllgather(long int *sendbuf, int count, long int *recvbuf, MPI_Comm comm);
   template int SchurchebMpiAllgather(float *sendbuf, int count, float *recvbuf, MPI_Comm comm);
   template int SchurchebMpiAllgather(double *sendbuf, int count, double *recvbuf, MPI_Comm comm);
   template int SchurchebMpiAllgather(complexs *sendbuf, int count, complexs *recvbuf, MPI_Comm comm);
   template int SchurchebMpiAllgather(complexd *sendbuf, int count, complexd *recvbuf, MPI_Comm comm);
   
#else
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allgather( sendbuf, count, SchurchebMpiDataType<T>(),recvbuf, count, SchurchebMpiDataType<T>(), comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllgather(int *sendbuf, int count, int *recvbuf, MPI_Comm comm);
   template int SchurchebMpiAllgather(long int *sendbuf, int count, long int *recvbuf, MPI_Comm comm);
   template int SchurchebMpiAllgather(float *sendbuf, int count, float *recvbuf, MPI_Comm comm);
   template int SchurchebMpiAllgather(double *sendbuf, int count, double *recvbuf, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allgather( sendbuf, 2*count, SchurchebMpiDataType<T>(),recvbuf, 2*count, SchurchebMpiDataType<T>(), comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllgather(complexs *sendbuf, int count, complexs *recvbuf, MPI_Comm comm);
   template int SchurchebMpiAllgather(complexd *sendbuf, int count, complexd *recvbuf, MPI_Comm comm);
     
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   template <typename T>
   int SchurchebMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allgatherv( sendbuf, count, SchurchebMpiDataType<T>(), recvbuf, recvcounts, recvdisps, SchurchebMpiDataType<T>(), comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllgatherv(int *sendbuf, int count, int *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int SchurchebMpiAllgatherv(long int *sendbuf, int count, long int *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int SchurchebMpiAllgatherv(float *sendbuf, int count, float *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int SchurchebMpiAllgatherv(double *sendbuf, int count, double *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int SchurchebMpiAllgatherv(complexs *sendbuf, int count, complexs *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int SchurchebMpiAllgatherv(complexd *sendbuf, int count, complexd *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   
#else
   
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm)
   {
      SCHURCHEB_MPI_CALL( MPI_Allgatherv( sendbuf, count, SchurchebMpiDataType<T>(), recvbuf, recvcounts, recvdisps, SchurchebMpiDataType<T>(), comm) );
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllgatherv(int *sendbuf, int count, int *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int SchurchebMpiAllgatherv(long int *sendbuf, int count, long int *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int SchurchebMpiAllgatherv(float *sendbuf, int count, float *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int SchurchebMpiAllgatherv(double *sendbuf, int count, double *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm)
   {
      int *recvdisps2, *recvcounts2;
      int np, i;
      MPI_Comm_size(comm, &np);
      SCHURCHEB_MALLOC( recvdisps2, np, kMemoryHost, int);
      SCHURCHEB_MALLOC( recvcounts2, np, kMemoryHost, int);
      
      for(i = 0 ; i < np ; i ++)
      {
         recvdisps2[i] = recvdisps[i] * 2;
         recvcounts2[i] = recvcounts[i] * 2;
      }
      
      SCHURCHEB_MPI_CALL( MPI_Allgatherv( sendbuf, 2*count, SchurchebMpiDataType<T>(), recvbuf, recvcounts2, recvdisps2, SchurchebMpiDataType<T>(), comm) );
      SCHURCHEB_FREE( recvdisps2, kMemoryHost);
      SCHURCHEB_FREE( recvcounts2, kMemoryHost);
      
      return SCHURCHEB_SUCCESS;
   }
   template int SchurchebMpiAllgatherv(complexs *sendbuf, int count, complexs *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   template int SchurchebMpiAllgatherv(complexd *sendbuf, int count, complexd *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
     
#endif

#ifdef SCHURCHEB_CUDA
   int SchurchebCudaSynchronize()
   {
      cudaDeviceSynchronize();
      return SCHURCHEB_SUCCESS;
   }
#endif
   
   template <typename T> 
   MPI_Datatype SchurchebMpiDataType()
   {
      SCHURCHEB_ERROR("Unimplemented MPI_Datatype.");
      return MPI_DATATYPE_NULL;
   }
   
   template<>
   MPI_Datatype SchurchebMpiDataType<int>()
   {
      return MPI_INT;
   }
   
   template<>
   MPI_Datatype SchurchebMpiDataType<long int>()
   {
      return MPI_LONG;
   }
   
   template<>
   MPI_Datatype SchurchebMpiDataType<float>()
   {
      return MPI_FLOAT;
   }
   
   template<>
   MPI_Datatype SchurchebMpiDataType<double>()
   {
      return MPI_DOUBLE;
   }
   
   template<>
   MPI_Datatype SchurchebMpiDataType<complexs>()
   {
#ifdef MPI_C_FLOAT_COMPLEX
      return MPI_C_FLOAT_COMPLEX;
#else
      return MPI_FLOAT;
#endif
   }
   
   template<>
   MPI_Datatype SchurchebMpiDataType<complexd>()
   {
#ifdef MPI_C_DOUBLE_COMPLEX
      return MPI_C_DOUBLE_COMPLEX;
#else
      return MPI_DOUBLE;
#endif
   }
}

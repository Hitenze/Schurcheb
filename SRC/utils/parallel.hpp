#ifndef SCHURCHEB_PARALLEL_H
#define SCHURCHEB_PARALLEL_H

/**
 * @file parallel.hpp
 * @brief Parallel related data structures and functions.
 */

#include <assert.h>
#include <mpi.h>
#include <vector>
#ifdef SCHURCHEB_OPENMP
#include "omp.h"
#endif
#ifdef SCHURCHEB_CUDA
#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"
#include "cusparse.h"
#endif

#include "utils.hpp"

using namespace std;

/*- - - - - - - - - Timing information */

namespace schurcheb
{
	/** 
    * @brief   The data structure for parallel computing, including data structures for MPI and CUDA.
    * @details The data structure for parallel computing, including data structures for MPI and CUDA. \n
    *          All CUDA information are shared, local MPI information can be different.
    */
   typedef class ParallelLogClass 
   {
      public:
      /* variables */
#ifdef SCHURCHEB_CUDA
      
      /**
       * @brief   The cuda random number generator.
       * @details The cuda random number generator.
       */
      static curandGenerator_t            _curand_gen;
      
      /**
       * @brief   The cuBLAS handle.
       * @details The cuBLAS handle.
       */
      static cublasHandle_t               _cublas_handle;
      
      /**
       * @brief   The cuSPARSE handle.
       * @details The cuSPARSE handle.
       */
      static cusparseHandle_t             _cusparse_handle;
      
      /**
       * @brief   The CUDA stream.
       * @details The CUDA stream.
       */
      static cudaStream_t                 _stream;
      
      /**
       * @brief   The cuSPARSE integer base, by default CUSPARSE_INDEX_BASE_ZERO.
       * @details The cuSPARSE integer base, by default CUSPARSE_INDEX_BASE_ZERO.
       */
      static cusparseIndexBase_t          _cusparse_idx_base;
      
      /**
       * @brief   The cuSPARSE general matrix descriptor.
       * @details The cuSPARSE general matrix descriptor.
       */
      static cusparseMatDescr_t           _mat_des;
      
      /**
       * @brief   The cuSPARSE unit diagonal lower triangular matrix descriptor.
       * @details The cuSPARSE unit diagonal lower triangular matrix descriptor.
       */
      static cusparseMatDescr_t           _matL_des;
      
      /**
       * @brief   The cuSPARSE arbitrary diagonal upper triangular matrix descriptor.
       * @details The cuSPARSE arbitrary diagonal upper triangular matrix descriptor.
       */
      static cusparseMatDescr_t           _matU_des;
      
      /**
       * @brief   The ilu solving policy for cuSPARSE.
       * @details The ilu solving policy for cuSPARSE. (Enable/disable level structure)
       */
      static cusparseSolvePolicy_t        _ilu_solve_policy;
      
      /**
       * @brief   Buffers for the cuSPARSE routines.
       * @details Buffers for the cuSPARSE routines.
       */
      static void                         *_cusparse_buffer;
      
      /**
       * @brief   Length of the buffers for the cuSPARSE routines.
       * @details Length of the buffers for the cuSPARSE routines.
       */
      static size_t                       _cusparse_buffer_length;

#if (SCHURCHEB_CUDA_VERSION == 11)
      /**
       * @brief   The cuSPARSE integer type, CUSPARSE_INDEX_32I or CUSPARSE_INDEX_64I.
       * @details The cuSPARSE integer type, CUSPARSE_INDEX_32I or CUSPARSE_INDEX_64I.
       */
      static cusparseIndexType_t          _cusparse_idx_type;
      
      /**
       * @brief   The algorithm of spMV for cuSPARSE.
       * @details The algorithm of spMV for cuSPARSE.
       */
      static cusparseSpMVAlg_t            _cusparse_spmv_algorithm;
#endif
      
#endif

      /**
       * @brief   The working location of the code (device/host).
       * @details The working location of the code (device/host).
       */
      static int                          _working_location;
      
      /**
       * @brief   The total number of global MPI ranks.
       * @details The total number of global MPI ranks.
       */
      static int                          _gsize;
      
      /**
       * @brief   The number of global MPI rank.
       * @details The number of global MPI rank.
       */
      static int                          _grank;
      
      /**
       * @brief   The global MPI comm.
       * @details The global MPI comm.
       */
      static MPI_Comm                     *_gcomm;
      
      /**
       * @brief   The local MPI comm (one np only, for consistancy).
       * @details The local MPI comm (one np only, for consistancy).
       */
      static MPI_Comm                     *_lcomm;
      
      /**
       * @brief   The total number of local MPI ranks.
       * @details The total number of local MPI ranks.
       */
      int                                 _size;
      
      /**
       * @brief   The number of local MPI rank.
       * @details The number of local MPI rank.
       */
      int                                 _rank;
      
      /**
       * @brief   The MPI comm that doesn't need to be freed.
       * @details The MPI comm that doesn't need to be freed.
       */
      MPI_Comm                            _commref;
      
      /**
       * @brief   The local MPI comm.
       * @details The local MPI comm.
       */
      MPI_Comm                            *_comm;
      
      /**
       * @brief   Free the parallel_log.
       * @details Free the parallel_log.
       */
      int                                 Clear();
      
      /**
       * @brief   The default constructor of parallel_log.
       * @details The default constructor of parallel_log.
       */
      ParallelLogClass();
      
      /**
       * @brief   The copy constructor of parallel_log.
       * @details The copy constructor of parallel_log.
       */
      ParallelLogClass(const ParallelLogClass &parlog);
      
      /**
       * @brief   The = operator of parallel_log.
       * @details The = operator of parallel_log.
       * @param   [in]        parlog The ParallelLogClass.
       */
      ParallelLogClass( ParallelLogClass &&parlog);
      
      /**
       * @brief   The = operator of parallel_log.
       * @details The = operator of parallel_log.
       * @param   [in]        parlog The ParallelLogClass.
       * @return              Return the ParallelLogClass.
       */
      ParallelLogClass& operator= (const ParallelLogClass &parlog);
      
      /**
       * @brief   The = operator of parallel_log.
       * @details The = operator of parallel_log.
       * @param   [in]        parlog The ParallelLogClass.
       * @return              Return the ParallelLogClass.
       */
      ParallelLogClass& operator= ( ParallelLogClass &&parlog);
      
      /**
       * @brief   The constructor of parallel_log, setup a new local comm.
       * @details The constructor of parallel_log, setup a new local comm.
       * @param [in] comm_in The new comm.
       */
      ParallelLogClass(MPI_Comm comm_in);
      
      /**
       * @brief   The destructor of parallel_log.
       * @details The destructor of parallel_log.
       */
      ~ParallelLogClass();
      
      /**
       * @brief   Get comm, np, and myid. When _comm is NULL, get the global one, otherwise get the local one.
       * @details Get comm, np, and myid. When _comm is NULL, get the global one, otherwise get the local one.
       * @param   [in]        np The number of processors.
       * @param   [in]        myid The local MPI rank number.
       * @param   [in]        comm The MPI_Comm.
       * @return              Return error message.
       */
      int                                 GetMpiInfo(int &np, int &myid, MPI_Comm &comm) const;
      
      /**
       * @brief      Get the MPI_comm. When _comm is NULL, get the global one, otherwise get the local one.
       * @details    Get the MPI_comm. When _comm is NULL, get the global one, otherwise get the local one.
       * @return     Return the MPI_comm.
       */
      MPI_Comm                            GetComm() const;
      
   }parallel_log, *parallel_logp;
   
   /**
    * @brief   Set the OpenMP thread number for each MPI process.
    * @details Set the OpenMP thread number for each MPI process.
    * @param   [in]     nthreads The number of threads per MPI process.
    * @return           Return error message.                                               
    */
   int SchurchebSetOpenmpNumThreads(int nthreads);
   
#ifdef SCHURCHEB_OPENMP

   /**
    * @brief   Get the local OpenMP thread number for each MPI process.
    * @details Get the local OpenMP thread number for each MPI process.
    * @return  Return the local thread number for each MPI process.
    */
   int SchurchebGetOpenmpThreadNum();
   

   /**
    * @brief   Get the current number of OpenMP threads.
    * @details Get the current number of OpenMP threads.
    * @return  Return the current thread number for each MPI process.
    */
   int SchurchebGetOpenmpNumThreads();
   
   /**
    * @brief   Get the max number of availiable OpenMP threads. Note that inside omp parallel region this function would return 1.
    * @details Get the max number of availiable OpenMP threads. Note that inside omp parallel region this function would return 1.
    * @return  Return the max availiable thread number for each MPI process.
    */
   int SchurchebGetOpenmpMaxNumThreads();
   
   /**
    * @brief   Get the max number of OpenMP threads.
    * @details Get the max number of OpenMP threads.
    * @return  Return the max thread number for each MPI process.
    */
   int SchurchebGetOpenmpGlobalMaxNumThreads();
   
#endif

   /**
    * @brief   Each MPI rank holds n_local, get the n_start and n_global.
    * @details Each MPI rank holds n_local, get the n_start and n_global.
    * @param [in]  	   n_local The local size.
    * @param [out]      n_start The start index.
    * @param [out]      n_global The global size.
    * @param [out]      comm The MPI_comm.
    * @return           Return error message.                                                         
    */
   int SchurchebNLocalToNGlobal( int n_local, SCHURCHEB_long &n_start, SCHURCHEB_long &n_global, MPI_Comm &comm);
   
   /**
    * @brief   Each MPI rank holds two n_locals, get the n_starts and n_globals.
    * @details Each MPI rank holds two n_locals, get the n_starts and n_globals.
    * @param [in]  	   nrow_local The first local size.
    * @param [in]  	   ncol_local The second local size.
    * @param [out]      nrow_start The first start index.
    * @param [out]      ncol_start The second start index.
    * @param [out]      nrow_global The first global size.
    * @param [out]      ncol_global The second global size.
    * @param [out]      comm The MPI_comm.
    * @return           Return error message.                                                         
    */
   int SchurchebNLocalToNGlobal( int nrow_local, int ncol_local, SCHURCHEB_long &nrow_start, SCHURCHEB_long &ncol_start, SCHURCHEB_long &nrow_global, SCHURCHEB_long &ncol_global, MPI_Comm &comm);
   
   /**
    * @brief   Initilize MPI, OpenMP, and CUDA. Note that if you have already called MPI_Init, call other init functions instead.
    * @details Initilize MPI, OpenMP, and CUDA. Note that if you have already called MPI_Init, call other init functions instead.
    * @param [in,out]  	argc Input of the main function.
    * @param [in,out]   argv Input of the main function.
    * @return           Return error message.                                                         
    */
   int SchurchebInit(int *argc, char ***argv);
   
   /**
    * @brief   Initilize MPI data struct with MPI_Comm.
    * @details Initilize MPI data struct with MPI_Comm. The Schurcheb package will duplicate this MPI_Comm.
    * @param [in]   comm The comm for Schurcheb, typically should be MPI_COMM_WORLD.
    * @return           Return error message.                                                         
    */
   int SchurchebInitMpi(MPI_Comm comm);
   
   /**
    * @brief   Initilize OpenMP and MKL.
    * @details Initilize OpenMP and MKL.
    * @param [in]   nthreads The max number of OpenMP threads.
    * @return           Return error message.                                                         
    */
   int SchurchebInitOpenMP(int nthreads);
   
   /**
    * @brief   Initilize CUDA.
    * @details Initilize CUDA.
    * @return  Return error message.                                                         
    */
   int SchurchebInitCUDA();
   
   /**
    * @brief   Print the parallel information to output.
    * @details Print the parallel information to output.
    * @return  Return error message.                                                         
    */
   int SchurchebPrintParallelInfo();
   
   /**
    * @brief   Finalize MPI, OpenMP, and CUDA. Note that if you don't want to call MPI_Finalize here, call other finalize functions.
    * @details Finalize MPI, OpenMP, and CUDA. Note that if you don't want to call MPI_Finalize here, call other finalize functions.
    * @return  Return error message.                                                         
    */
   int SchurchebFinalize();
   
   /**
    * @brief   Finalize MPI data structure. Note that MPI_Finalize won't be called here.
    * @details Finalize MPI data structure. Note that MPI_Finalize won't be called here.
    * @return  Return error message.                                                         
    */
   int SchurchebFinalizeMpi();
   
   /**
    * @brief   Finalize OpenMP data structure.
    * @details Finalize OpenMP data structure.
    * @return  Return error message.                                                         
    */
   int SchurchebFinalizeOpenMP();
   
   /**
    * @brief   Finalize CUDA data structure.
    * @details Finalize CUDA data structure.
    * @return  Return error message.                                                         
    */
   int SchurchebFinalizeCUDA();
   
   /**
    * @brief   Get current time using MPI_Wtime.
    * @details Get current time using MPI_Wtime.
    * @param [in]  comm The MPI_Comm. 
    * @param [out] t The time.  
    * @return  Return error message.                                                         
    */
   int SchurchebMpiTime(MPI_Comm comm, double &t);

#ifdef MPI_C_FLOAT_COMPLEX
   
   /**
    * @brief   MPI_Isend.
    * @details MPI_Isend.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   int SchurchebMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Irecv.
    * @details MPI_Irecv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   int SchurchebMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Send.
    * @details MPI_Send.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm. 
    * @return      Return error message.                                                         
    */
   template <typename T>
   int SchurchebMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm);
   
   /**
    * @brief   MPI_Recv.
    * @details MPI_Recv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   int SchurchebMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   
   /**
    * @brief   MPI_Sendrecv.
    * @details MPI_Sendrecv.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  sendcount Number of send/recv elements. 
    * @param [in]  dest Rank of the dest MPI rank. 
    * @param [in]  sendtag Tag of the send message. 
    * @param [in]  recvbuf Pointer to the recv data. 
    * @param [in]  recvcount Number of send/recv elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  recvtag Tag of the recv message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T1, typename T2>
   int SchurchebMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
#else
   
   /**
    * @brief   MPI_Isend.
    * @details MPI_Isend.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Isend.
    * @details MPI_Isend.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiIsend(T *buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Irecv.
    * @details MPI_Irecv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Irecv.
    * @details MPI_Irecv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiIrecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Request *request);
   
   /**
    * @brief   MPI_Send.
    * @details MPI_Send.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm);
   
   /**
    * @brief   MPI_Send.
    * @details MPI_Send.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  dest Rank of the target MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] request MPI_Request.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiSend(T *buf, int count, int dest, int tag, MPI_Comm comm);
   
   /**
    * @brief   MPI_Irecv.
    * @details MPI_Irecv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   
   /**
    * @brief   MPI_Irecv.
    * @details MPI_Irecv.
    * @param [in]  buf Pointer to the data. 
    * @param [in]  count Number of elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  tag Tag of the message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiRecv(T *buf, int count, int source, int tag, MPI_Comm comm, MPI_Status * status);
   
   /**
    * @brief   MPI_Sendrecv.
    * @details MPI_Sendrecv.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  sendcount Number of send/recv elements. 
    * @param [in]  dest Rank of the dest MPI rank. 
    * @param [in]  sendtag Tag of the send message. 
    * @param [in]  recvbuf Pointer to the recv data. 
    * @param [in]  recvcount Number of send/recv elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  recvtag Tag of the recv message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T1, typename T2>
   typename std::enable_if<!SchurchebIsComplex<T1>::value&&!SchurchebIsComplex<T2>::value, int>::type
   int SchurchebMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
   /**
    * @brief   MPI_Sendrecv.
    * @details MPI_Sendrecv.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  sendcount Number of send/recv elements. 
    * @param [in]  dest Rank of the dest MPI rank. 
    * @param [in]  sendtag Tag of the send message. 
    * @param [in]  recvbuf Pointer to the recv data. 
    * @param [in]  recvcount Number of send/recv elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  recvtag Tag of the recv message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T1, typename T2>
   typename std::enable_if<!SchurchebIsComplex<T1>::value&&SchurchebIsComplex<T2>::value, int>::type
   int SchurchebMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
   /**
    * @brief   MPI_Sendrecv.
    * @details MPI_Sendrecv.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  sendcount Number of send/recv elements. 
    * @param [in]  dest Rank of the dest MPI rank. 
    * @param [in]  sendtag Tag of the send message. 
    * @param [in]  recvbuf Pointer to the recv data. 
    * @param [in]  recvcount Number of send/recv elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  recvtag Tag of the recv message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T1, typename T2>
   typename std::enable_if<SchurchebIsComplex<T1>::value&&!SchurchebIsComplex<T2>::value, int>::type
   int SchurchebMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
   /**
    * @brief   MPI_Sendrecv.
    * @details MPI_Sendrecv.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  sendcount Number of send/recv elements. 
    * @param [in]  dest Rank of the dest MPI rank. 
    * @param [in]  sendtag Tag of the send message. 
    * @param [in]  recvbuf Pointer to the recv data. 
    * @param [in]  recvcount Number of send/recv elements. 
    * @param [in]  source Rank of the source MPI rank. 
    * @param [in]  recvtag Tag of the recv message. 
    * @param [in]  comm MPI_Comm.  
    * @param [out] status MPI_Status.  
    * @return      Return error message.                                                         
    */
   template <typename T1, typename T2>
   typename std::enable_if<SchurchebIsComplex<T1>::value&&SchurchebIsComplex<T2>::value, int>::type
   int SchurchebMpiSendRecv(T1 *sendbuf, int sendcount, int dest, int sendtag, T2 *recvbuf, int recvcount, int source, int recvtag, MPI_Comm comm, MPI_Status *status);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX
   
   /**
    * @brief   MPI_Bcast.
    * @details MPI_Bcast.
    * @param [in,out]   buf Pointer to the data. 
    * @param [in]       count Number of elements. 
    * @param [in]       root Root's MPI rank. 
    * @param [in]       comm MPI_Comm.    
    * @return           Return error message.                                                           
    */
   template <typename T>
   int SchurchebMpiBcast(T *buf, int count, int root, MPI_Comm comm);

#else

   /**
    * @brief   MPI_Bcast.
    * @details MPI_Bcast.
    * @param [in,out]   buf Pointer to the data. 
    * @param [in]       count Number of elements. 
    * @param [in]       root Root's MPI rank. 
    * @param [in]       comm MPI_Comm.    
    * @return           Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiBcast(T *buf, int count, int root, MPI_Comm comm);

   /**
    * @brief   MPI_Bcast.
    * @details MPI_Bcast.
    * @param [in,out]   buf Pointer to the data. 
    * @param [in]       count Number of elements. 
    * @param [in]       root Root's MPI rank. 
    * @param [in]       comm MPI_Comm.    
    * @return           Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiBcast(T *buf, int count, int root, MPI_Comm comm);

#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Scan.
    * @details MPI_Scan.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                         
    */
   template <typename T>
   int SchurchebMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);

#else

   /**
    * @brief   MPI_Scan.
    * @details MPI_Scan.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                             
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);

   /**
    * @brief   MPI_Scan.
    * @details MPI_Scan.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                              
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiScan(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);

#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Reduce.
    * @details MPI_Reduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  root Root's MPI rank. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   int SchurchebMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

#else

   /**
    * @brief   MPI_Reduce.
    * @details MPI_Reduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  root Root's MPI rank. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                       
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

   /**
    * @brief   MPI_Reduce.
    * @details MPI_Reduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op. 
    * @param [in]  root Root's MPI rank. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                        
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiReduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, int root, MPI_Comm comm);

#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Allreduce.
    * @details MPI_Allreduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op.
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   int SchurchebMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
#else

   /**
    * @brief   MPI_Allreduce.
    * @details MPI_Allreduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op.
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
   /**
    * @brief   MPI_Allreduce.
    * @details MPI_Allreduce.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  count Number of elements. 
    * @param [in]  op MPI_Op.
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllreduce(T *sendbuf, T *recvbuf, int count, MPI_Op op, MPI_Comm comm);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   In place MPI_Allreduce.
    * @details In place MPI_Allreduce.
    * @param [in,out] buf Pointer to the data. 
    * @param [in]     count Number of elements. 
    * @param [in]     op MPI_Op. 
    * @param [in]     comm MPI_Comm.    
    * @return         Return error message.                                                         
    */
   template <typename T>
   int SchurchebMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm);
   
#else

   /**
    * @brief   In place MPI_Allreduce.
    * @details In place MPI_Allreduce.
    * @param [in,out] buf Pointer to the data. 
    * @param [in]     count Number of elements. 
    * @param [in]     op MPI_Op. 
    * @param [in]     comm MPI_Comm.    
    * @return         Return error message.                                                        
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm);
   
   /**
    * @brief   In place MPI_Allreduce.
    * @details In place MPI_Allreduce.
    * @param [in,out] buf Pointer to the data. 
    * @param [in]     count Number of elements. 
    * @param [in]     op MPI_Op. 
    * @param [in]     comm MPI_Comm.    
    * @return         Return error message.                                                         
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllreduceInplace(T *buf, int count, MPI_Op op, MPI_Comm comm);
   
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Gather.
    * @details MPI_Gather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data.
    * @param [in]  root The MPI rank of the reciver. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                         
    */
   template <typename T>
   int SchurchebMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm);
   
#else

   /**
    * @brief   MPI_Gather.
    * @details MPI_Gather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data.
    * @param [in]  root The MPI rank of the reciver. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                        
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm);
   
   /**
    * @brief   MPI_Gather.
    * @details MPI_Gather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data.
    * @param [in]  root The MPI rank of the reciver. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                        
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiGather(T *sendbuf, int count, T *recvbuf, int root, MPI_Comm comm);
     
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Allgather.
    * @details MPI_Allgather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   int SchurchebMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm);
   
#else

   /**
    * @brief   MPI_Allgather.
    * @details MPI_Allgather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm);
   
   /**
    * @brief   MPI_Allgather.
    * @details MPI_Allgather.
    * @param [in]  sendbuf Pointer to the send data. 
    * @param [in]  count Size of each single send/recv. 
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllgather(T *sendbuf, int count, T *recvbuf, MPI_Comm comm);
     
#endif

#ifdef MPI_C_FLOAT_COMPLEX

   /**
    * @brief   MPI_Allgatherv.
    * @details MPI_Allgatherv.
    * @param [in]  sendbuf Pointer to the send data.
    * @param [in]  count Size of each single send/recv.  
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  recvcounts Number of elements on each processor. 
    * @param [in]  recvdisps Displacement of elements on each processor. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.                                                         
    */
   template <typename T>
   int SchurchebMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   
#else

   /**
    * @brief   MPI_Allgatherv.
    * @details MPI_Allgatherv.
    * @param [in]  sendbuf Pointer to the send data.
    * @param [in]  count Size of each single send/recv.  
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  recvcounts Number of elements on each processor. 
    * @param [in]  recvdisps Displacement of elements on each processor. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
   
   /**
    * @brief   MPI_Allgatherv.
    * @details MPI_Allgatherv.
    * @param [in]  sendbuf Pointer to the send data.
    * @param [in]  count Size of each single send/recv.  
    * @param [out] recvbuf Pointer to the recv data. 
    * @param [in]  recvcounts Number of elements on each processor. 
    * @param [in]  recvdisps Displacement of elements on each processor. 
    * @param [in]  comm MPI_Comm.    
    * @return      Return error message.   
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebMpiAllgatherv(T *sendbuf, int count, T *recvbuf, int *recvcounts, int *recvdisps, MPI_Comm comm);
     
#endif

#ifdef SCHURCHEB_CUDA

   /**
    * @brief   The cuda synchronize.
    * @details The cuda synchronize.
    * @return  Return error message.                                                         
    */
   int SchurchebCudaSynchronize();
#endif

   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template <typename T> 
   MPI_Datatype SchurchebMpiDataType();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype SchurchebMpiDataType<int>();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype SchurchebMpiDataType<long int>();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype SchurchebMpiDataType<float>();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype SchurchebMpiDataType<double>();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype SchurchebMpiDataType<complexs>();
   
   /**
    * @brief   Get the MPI_Datatype.
    * @details Get the MPI_Datatype.
    * @return  Return the MPI_Datatype.                                                       
    */
   template<>
   MPI_Datatype SchurchebMpiDataType<complexd>();
   
}

/*- - - - - - - - - OPENMP default schedule */
#ifdef SCHURCHEB_OPENMP

/* some implementations requires same operation order, use SCHURCHEB_OPENMP_SCHEDULE_STATIC to make sure */
//#define SCHURCHEB_OPENMP_SCHEDULE_DEFAULT       schedule(dynamic)
#define SCHURCHEB_OPENMP_SCHEDULE_DEFAULT       schedule(static)
#define SCHURCHEB_OPENMP_SCHEDULE_STATIC        schedule(static)

#endif

/*- - - - - - - - - MPI calls */

#ifdef SCHURCHEB_DEBUG

#define SCHURCHEB_MPI_CALL(...) {\
   assert( (__VA_ARGS__) == MPI_SUCCESS);\
}

#else

#define SCHURCHEB_MPI_CALL(...) {\
   (__VA_ARGS__);\
}

#endif

/*- - - - - - - - - CUDA calls */

#ifdef SCHURCHEB_CUDA

#ifndef SCHURCHEB_CUDA_VERSION

/* the default CUDA version is 11, note that we only support CUDA 10 and CUDA 11 yet */
#define SCHURCHEB_CUDA_VERSION 11

#endif

#define SCHURCHEB_CUDA_SYNCHRONIZE SchurchebCudaSynchronize();

#ifdef SCHURCHEB_DEBUG

#define SCHURCHEB_CUDA_CALL(...) {\
   assert((__VA_ARGS__) == cudaSuccess);\
}

#define SCHURCHEB_CURAND_CALL(...) {\
   assert((__VA_ARGS__) == CURAND_STATUS_SUCCESS);\
}

#define SCHURCHEB_CUBLAS_CALL(...) {\
   assert( (__VA_ARGS__) == CUBLAS_STATUS_SUCCESS);\
}

#define SCHURCHEB_CUSPARSE_CALL(...) {\
   assert((__VA_ARGS__) == CUSPARSE_STATUS_SUCCESS);\
}

#else

#define SCHURCHEB_CUDA_CALL(...) {\
   (__VA_ARGS__);\
}

#define SCHURCHEB_CURAND_CALL(...) {\
   (__VA_ARGS__);\
}

#define SCHURCHEB_CUBLAS_CALL(...) {\
   (__VA_ARGS__);\
}

#define SCHURCHEB_CUSPARSE_CALL(...) {\
   (__VA_ARGS__);\
}

#endif

//#define SCHURCHEB_THRUST_CALL(thrust_function, ...) thrust::thrust_function(thrust::cuda::par.on(schurcheb::ParallelLogClass::_stream), __VA_ARGS__)

#define SCHURCHEB_THRUST_CALL(thrust_function, ...) thrust::thrust_function( __VA_ARGS__)

#else

#define SCHURCHEB_CUDA_SYNCHRONIZE

#endif

#define SCHURCHEB_GLOBAL_SEQUENTIAL_RUN(...) {\
   for(int pgsri = 0 ; pgsri < schurcheb::parallel_log::_gsize ; pgsri++)\
   {\
      if( schurcheb::parallel_log::_grank == pgsri)\
      {\
         (__VA_ARGS__);\
      }\
      MPI_Barrier(*(schurcheb::parallel_log::_gcomm));\
   }\
}

#endif

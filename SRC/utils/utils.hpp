#ifndef SCHURCHEB_UTILS_H
#define SCHURCHEB_UTILS_H

/**
 * @file utils.hpp
 * @brief Basic ultility functions.
 */

#include <stdio.h>
#include <cstdlib>
#include <utility>
#include <type_traits>
#include <ctype.h>
#include "complex.hpp"
#include <random>
#include <climits>
#include <cassert>

/* do we use long int? 
 * SCHURCHEB_INT32 controls SCHURCHEB_long
 * if always want long int use long int instead
 */
#ifdef SCHURCHEB_INT32
   typedef int SCHURCHEB_long;
#else
   typedef long int SCHURCHEB_long;
#endif


#define SCHURCHEB_FIRM_CHKERR(ierr) {{if(ierr){printf("Err value: %d on MPI rank %d\n",ierr, parallel_log::_grank);assert(!(ierr));};}}

#ifdef SCHURCHEB_DEBUG
   #define SCHURCHEB_ERROR(message) {printf("Error: %s on MPI rank %d\n", message, parallel_log::_grank);assert(0);}
   #define SCHURCHEB_CHKERR(ierr) {{if(ierr){printf("Err value: %d\n",ierr);assert(!(ierr));};}}
   #define SCHURCHEB_PRINT_DEBUG(conda, condb, ...) {if(conda==condb) {printf("DEBUG: ");printf(__VA_ARGS__);}}
#else
   #define SCHURCHEB_ERROR(message) {printf("Error: %s on MPI rank %d\n", message, parallel_log::_grank);}
   #define SCHURCHEB_CHKERR(ierr) {;}
   #define SCHURCHEB_PRINT_DEBUG(conda, condb, ...) {;}
#endif
#ifdef SCHURCHEB_NO_WARNING
   #define SCHURCHEB_WARNING(message) {;}
#else
   #define SCHURCHEB_WARNING(message) {printf("Warning: %s on MPI rank %d\n", message, parallel_log::_grank);}
#endif

#define SCHURCHEB_PRINT(...) fprintf(schurcheb::SCHURCHEB_global::_out_file, __VA_ARGS__);

#ifndef SCHURCHEB_SUCCESS
#define SCHURCHEB_SUCCESS        	                  0
#define SCHURCHEB_RETURN_METIS_INSUFFICIENT_NDOM      1
#define SCHURCHEB_RETURN_METIS_NO_INTERIOR            2
#define SCHURCHEB_RETURN_METIS_PROBLEM_TOO_SMALL      3
#define SCHURCHEB_RETURN_PARILU_NO_INTERIOR           10
#define SCHURCHEB_ERROR_INVALED_OPTION                100
#define SCHURCHEB_ERROR_INVALED_PARAM   	            101
#define SCHURCHEB_ERROR_IO_ERROR        	            102
#define SCHURCHEB_ERROR_ILU_EMPTY_ROW   	            103
#define SCHURCHEB_ERROR_DOUBLE_INIT_FREE              104 // call init function for multiple times
#define SCHURCHEB_ERROR_COMPILER                      105
#define SCHURCHEB_ERROR_FUNCTION_CALL_ERR             106
#define SCHURCHEB_ERROR_MEMORY_LOCATION               107
#endif

#define SCHURCHEB_CAST( type, val) reinterpret_cast<type>((val))

namespace schurcheb
{
   /** 
    * @brief   The data structure for parallel computing, including data structures for MPI and CUDA.
    * @details The data structure for parallel computing, including data structures for MPI and CUDA. \n
    *          All CUDA information are shared, local MPI information can be different.
    */
   typedef class SchurchebGlobalClass 
   {
      public:
      
      /**
       * @brief   Expand factor, used when expand vectors with PushBack.
       * @details Expand factor, used when expand vectors with PushBack.
       */
      static double                                _expand_fact;
      
      /**
       * @brief   Reserved size of COO matrix, default is nrow * _coo_reserve_fact.
       * @details Reserved size of COO matrix, default is nrow * _coo_reserve_fact.
       */
      static int                                   _coo_reserve_fact;
      
      /**
       * @brief   Disable OpenMP in some cases when a loop size is too small.
       * @details Disable OpenMP in some cases when a loop size is too small.
       */
      static int                                   _openmp_min_loopsize;
      
      /**
       * @brief   Numter of Metis refine (for ParMetis).
       * @details Numter of Metis refine (for ParMetis).
       */
      static int                                   _metis_refine;
      
      /**
       * @brief   The tolorance for loading balance when apply the parallel kway partition.
       * @details The tolorance for loading balance when apply the parallel kway partition.
       */
      static double                                _metis_loading_balance_tol;
      
      /**
       * @brief   Min size of the edge separator.
       * @details Min size of the edge separator.
       */
      static int                                   _minsep;
      
      /**
       * @brief   Thick-restart factor for thick-restart Arnoldi.
       * @details Thick-restart factor for thick-restart Arnoldi.
       */
      static double                                _tr_factor;
      
      /**
       * @brief   The tolorance for orthogonalization for Arnoldi.
       * @details The tolorance for orthogonalization for Arnoldi.
       */
      static double                                _orth_tol;
      
      /**
       * @brief   The tolorance for re-orthogonalization for Arnoldi.
       * @details The tolorance for re-orthogonalization for Arnoldi.
       */
      static double                                _reorth_tol;
      
      /**
       * @brief   Default width of the dashline in the output.
       * @details Default width of the dashline in the output.
       */
      static int                                   _dash_line_width;
      
      /**
       * @brief   Used to obtain a seed for the random number engine.
       * @details Used to obtain a seed for the random number engine.
       */
      static std::random_device                    _random_device;
      
      /**
       * @brief   Mersenne_twister_engine.
       * @details Mersenne_twister_engine.
       */
      static std::mt19937                          _mersenne_twister_engine;
      
      /**
       * @brief   Uniform_int_distribution.
       * @details Uniform_int_distribution.
       */
      static std::uniform_int_distribution<int>    _uniform_int_distribution;
      
      /**
       * @brief   The output file, default is stdout.
       * @details The output file, default is stdout.
       */
      static FILE                                  *_out_file;
      
      /**
       * @brief   Gram schmidt option for the eigenvalue solver.
       * @details Gram schmidt option for the eigenvalue solver. \n
       *          0: CGS-2.
       *          1: MGS.
       */
      static int                                   _gram_schmidt;
      
   }SCHURCHEB_global;
   
   /**
    * @brief   Tell if a value is integer.
    * @details Tell if a value is integer.
    */
   template <class T> struct SchurchebIsInteger : public std::false_type {};
   template <class T> struct SchurchebIsInteger<const T> : public SchurchebIsInteger<T> {};
   template <class T> struct SchurchebIsInteger<volatile const T> : public SchurchebIsInteger<T>{};
   template <class T> struct SchurchebIsInteger<volatile T> : public SchurchebIsInteger<T>{};
   template<> struct SchurchebIsInteger<int> : public std::true_type {};
   template<> struct SchurchebIsInteger<long int> : public std::true_type {};
   
   /**
    * @brief   Tell if a value is in double precision.
    * @details Tell if a value is in double precision.
    */
   template <class T> struct SchurchebIsDoublePrecision : public std::false_type {};
   template <class T> struct SchurchebIsDoublePrecision<const T> : public SchurchebIsDoublePrecision<T> {};
   template <class T> struct SchurchebIsDoublePrecision<volatile const T> : public SchurchebIsDoublePrecision<T>{};
   template <class T> struct SchurchebIsDoublePrecision<volatile T> : public SchurchebIsDoublePrecision<T>{};
   template<> struct SchurchebIsDoublePrecision<double> : public std::true_type {};
   template<> struct SchurchebIsDoublePrecision<complexd> : public std::true_type {};
   
   /**
    * @brief   Tell if a value is a parallel data structure.
    * @details Tell if a value is a parallel data structure.
    */
   template <class T> struct SchurchebIsParallel : public std::false_type {};
   template <class T> struct SchurchebIsParallel<const T> : public SchurchebIsParallel<T> {};
   template <class T> struct SchurchebIsParallel<volatile const T> : public SchurchebIsParallel<T>{};
   
   /**
    * @brief   The precision enum.
    * @details The precision enum.
    */
   enum PrecisionEnum
   {
      kUnknownPrecision = -1,
      kInt,
      kLongInt,
      kHalfReal,
      kHalfComplex,
      kSingleReal,
      kSingleComplex,
      kDoubleReal,
      kDoubleComplex
   };
   
   /**
    * @brief   The struct of for sorting.
    * @details The struct of for sorting.
    */
   template <typename T>
   struct CompareStruct
   {
      T     val;
      int   ord;
   };
   
   typedef CompareStruct<int>       compareord_int;
   typedef CompareStruct<long int>  compareord_long;
   typedef CompareStruct<float>     compareord_float;
   typedef CompareStruct<double>    compareord_double;
   
   /**
    * @brief   The operator > for CompareStruct.
    * @details The operator > for CompareStruct.
    * @param [in]   a First value.
    * @param [in]   b Second value.
    * @return       Return true or false.
    */
   template <class T>
   struct CompareStructGreater
   {
       bool operator()(T const &a, T const &b) const { return a.val > b.val; }
   };
   
   /**
    * @brief   The operator < for CompareStruct.
    * @details The operator < for CompareStruct.
    * @param [in]   a First value.
    * @param [in]   b Second value.
    * @return       Return true or false.
    */
   template <class T>
   struct CompareStructLess
   {
       bool operator()(T const &a, T const &b) const { return a.val < b.val; }
   };
   
   /**
    * @brief   Get the larger one out of two numbers.
    * @details Get the larger one out of two numbers.
    * @param [in]   a First value.
    * @param [in]   b Second value.
    * @return       Return the larger value.
    */
   template <typename T>
   T SchurchebMax(T a, T b);
   
   /**
    * @brief   Get the smaller one out of two numbers.
    * @details Get the smaller one out of two numbers.
    * @param [in]   a First value.
    * @param [in]   b Second value.
    * @return       Return the smaller value.
    */
   template <typename T>
   T SchurchebMin(T a, T b);
   
   /**
    * @brief   Get the absolute value of a numbers.
    * @details Get the absolute value of a numbers.
    * @param [in]   a The value.
    * @return       Return the absolute value.
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, T>::type
   SchurchebAbs( const T &a);
   
   /**
    * @brief   Get the absolute value of a numbers.
    * @details Get the absolute value of a numbers.
    * @param [in]   a The value.
    * @return       Return the absolute value.
    */
   float SchurchebAbs( const complexs &a);
   
   /**
    * @brief   Get the absolute value of a numbers.
    * @details Get the absolute value of a numbers.
    * @param [in]   a The value.
    * @return       Return the absolute value.
    */
   double SchurchebAbs( const complexd &a);
   
   /**
    * @brief   Get the real part of a numbers.
    * @details Get the real part value of a numbers.
    * @param [in]   a The value.
    * @return       Return the real part value.
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, T>::type
   SchurchebReal( const T &a);
   
   /**
    * @brief   Get the real part of a numbers.
    * @details Get the real part of a numbers.
    * @param [in]   a The value.
    * @return       Return the real part value.
    */
   float SchurchebReal( const complexs &a);
   
   /**
    * @brief   Get the real part of a numbers.
    * @details Get the real part of a numbers.
    * @param [in]   a The value.
    * @return       Return the real part value.
    */
   double SchurchebReal( const complexd &a);
   
   /**
    * @brief   Get the conjugate of a numbers, for real value we do nothing.
    * @details Get the conjugate of a numbers, for real value we do nothing.
    * @param [in]   a The value.
    * @return       Return the conjugate.
    */
   template <typename T>
   typename std::enable_if<!SchurchebIsComplex<T>::value, T>::type
   SchurchebConj( const T &a);
   
   /**
    * @brief   Get the conjugate of a numbers, for real value we do nothing.
    * @details Get the conjugate of a numbers, for real value we do nothing.
    * @param [in]   a The value.
    * @return       Return the conjugate.
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, T>::type
   SchurchebConj( const T &a);
   
   /**
    * @brief   Generate random integer number at host memory.
    * @details Generate random integer number at host memory.
    * @param [out]   a The random value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<SchurchebIsInteger<T>::value, int>::type
   SchurchebValueRandHost(T &a);
   
   /**
    * @brief   Generate random float number between [0, 1] at host memory.
    * @details Generate random float number between [0, 1] at host memory.
    * @param [out]   a The random value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<SchurchebIsReal<T>::value, int>::type
   SchurchebValueRandHost(T &a);
   
   /**
    * @brief   Generate random single complex number, real and imag part both between [0, 1] at host memory.
    * @details Generate random single complex number, real and imag part both between [0, 1] at host memory.
    * @param [out]   a The random value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebValueRandHost(T &a);
   
   /**
    * @brief   Print spaces using cout.
    * @details Print spaces using cout.
    * @param [in]   width The width of the space.
    * @return       Return error message.
    */
   int SchurchebPrintSpace(int width);
   
   /**
    * @brief   Print a dash line using cout.
    * @details Print a dash line using cout.
    * @param [in]   width The width of the line.
    * @return       Return error message.
    */
   int SchurchebPrintDashLine(int width);
   
   /**
    * @brief   Print a integer value, fixed width.
    * @details Print a integer value, fixed width.
    * @param [in]   val The value.
    * @param [in]   width The width of the value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<SchurchebIsInteger<T>::value, int>::type 
   SchurchebPrintValueHost(T val, int width);
   
   /**
    * @brief   Print a real value, fixed width.
    * @details Print a real value, fixed width.
    * @param [in]   val The value.
    * @param [in]   width The width of the value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<SchurchebIsReal<T>::value, int>::type 
   SchurchebPrintValueHost(T val, int width);
   
   /**
    * @brief   Print a complex value, fixed width.
    * @details Print a complex value, fixed width.
    * @param [in]   val The value.
    * @param [in]   width The width of the value.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type 
   SchurchebPrintValueHost(T val, int width);
   
   /**
    * @brief   Read the first word from a input string, convert to upper case.
    * @details Read the first word from a input string, convert to upper case.
    * @param [in]   pin The input char*.
    * @param [out]  pout The output char**.
    */
   void SchurchebReadFirstWord(char *pin, char **pout);

   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   amount number of data we load after "-xxx".
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int SchurchebReadInputArg(const char *argname, int amount, float *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   amount number of data we load after "-xxx".
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int SchurchebReadInputArg(const char *argname, int amount, double *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   amount number of data we load after "-xxx".
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int SchurchebReadInputArg(const char *argname, int amount, complexs *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   amount number of data we load after "-xxx".
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int SchurchebReadInputArg(const char *argname, int amount, complexd *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   amount number of data we load after "-xxx".
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int SchurchebReadInputArg(const char *argname, int amount, int *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int SchurchebReadInputArg(const char *argname, char *val, int argc, char **argv);
   
   /**
    * @brief   Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @details Read input data. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   val The output value.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int SchurchebReadInputArg(const char *argname, bool *val, int argc, char **argv);
   
   /**
    * @brief   Check if we have an argument. If we want to find "xxx", the user input should be "-xxx".
    * @details Check if we have an argument. If we want to find "xxx", the user input should be "-xxx".
    * @param [in]   argname The target argument.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return 1 when find the data, 0 otherwise.
    */
   int SchurchebReadInputArg(const char *argname, int argc, char **argv);
   
   /**
    * @brief   Plot the data to the terminal output.
    * @details Plot the data to the terminal output.
    * @param [in]   data The data.
    * @param [in]   length length of the data.
    * @param [in]   numx the number of grids on x.
    * @param [in]   numx the number of grids on y.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<!(SchurchebIsComplex<T>::value), int>::type
   SchurchebPlotData(T* ydata, int length, int numx, int numy);
   
   /**
    * @brief   Plot the data to the terminal output.
    * @details Plot the data to the terminal output.
    * @param [in]   argname The target argument.
    * @param [in]   argc From the main function.
    * @param [in]   argv From the main function.
    * @return       Return error message.
    */
   template <typename T>
   typename std::enable_if<SchurchebIsComplex<T>::value, int>::type
   SchurchebPlotData(T* ydata, int length, int numx, int numy);
   
   /**
    * @brief   Set output file of SCHURCHEB_PTINT.
    * @details Set output file of SCHURCHEB_PTINT.
    * @param [in]   filename The pointer of the file.
    * @return       Return error message.
    */
   int SchurchebSetOutputFile(const char *filename);
   
}

#endif

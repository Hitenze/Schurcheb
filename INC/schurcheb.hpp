#ifndef SCHURCHEB_H
#define SCHURCHEB_H

/**
 * @file schurcheb.hpp
 * @brief Global header of the Schur cheb
 */

#define SCHURCHEB_VERSION                             10000

#ifndef SCHURCHEB_SUCCESS
#define SCHURCHEB_SUCCESS        	                  0
#define SCHURCHEB_RETURN_METIS_INSUFFICIENT_NDOM      1
#define SCHURCHEB_RETURN_METIS_NO_INTERIOR            2
#define SCHURCHEB_RETURN_METIS_ISOLATE_NODE           3
#define SCHURCHEB_ERROR_INVALED_OPTION                100
#define SCHURCHEB_ERROR_INVALED_PARAM   	            101
#define SCHURCHEB_ERROR_IO_ERROR        	            102
#define SCHURCHEB_ERROR_ILU_EMPTY_ROW   	            103
#define SCHURCHEB_ERROR_DOUBLE_INIT_FREE              104 // call init function for multiple times
#define SCHURCHEB_ERROR_COMPILER                      105
#define SCHURCHEB_ERROR_FUNCTION_CALL_ERR             106
#define SCHURCHEB_ERROR_MEMORY_LOCATION               107
#endif

#include "../SRC/utils/structs.hpp"
#include "../SRC/utils/utils.hpp"
#include "../SRC/utils/parallel.hpp"
#include "../SRC/utils/memory.hpp"
#include "../SRC/utils/protos.hpp"
#include "../SRC/utils/mmio.hpp"

#include "../SRC/vectors/vector.hpp"
#include "../SRC/vectors/sequential_vector.hpp"
#include "../SRC/vectors/parallel_vector.hpp"
#include "../SRC/vectors/int_vector.hpp"
#include "../SRC/vectors/vectorops.hpp"

#include "../SRC/matrices/matrix.hpp"
#include "../SRC/matrices/arnoldimatrix.hpp"
#include "../SRC/matrices/matrixops.hpp"
#include "../SRC/matrices/coo_matrix.hpp"
#include "../SRC/matrices/csr_matrix.hpp"
#include "../SRC/matrices/dense_matrix.hpp"
#include "../SRC/matrices/parallel_csr_matrix.hpp"

#include "../SRC/solvers/solver.hpp"

#include "../SRC/preconditioners/ilu.hpp"

#include "../SRC/matrices/matrix.hpp"
#include "../SRC/matrices/arnoldimatrix.hpp"
#include "../SRC/matrices/matrixops.hpp"
#include "../SRC/matrices/coo_matrix.hpp"
#include "../SRC/matrices/csr_matrix.hpp"
#include "../SRC/matrices/dense_matrix.hpp"
#include "../SRC/matrices/parallel_csr_matrix.hpp"


#include "../SRC/schurcheb/superlu.hpp"
#include "../SRC/schurcheb/mumps.hpp"
#include "../SRC/schurcheb/pardiso.hpp"
#include "../SRC/schurcheb/dsolver.hpp"
#include "../SRC/schurcheb/arpack.hpp"
#include "../SRC/schurcheb/schurshift.hpp"
#include "../SRC/schurcheb/chebeig.hpp"


#endif

include ./makefile.in

.SUFFIXES : .c .cpp .o

INCLUDES = -I./INC

ifneq ($(USING_SUPERLU),0)
FLAGS += -DSCHURCHEB_SUPERLU
INCLUDES += -I$(SUPERLU_INCLUDE_PATH)
endif

ifneq ($(USING_MUMPS),0)
FLAGS += -DSCHURCHEB_MUMPS
INCLUDES += -I$(MUMPS_INCLUDE_PATH)
endif

SRC = ./SRC/utils/utils.o\
        ./SRC/utils/parallel.o\
        ./SRC/utils/memory.o\
        ./SRC/utils/mmio.o\
        ./SRC/vectors/vector.o\
        ./SRC/vectors/sequential_vector.o\
        ./SRC/vectors/int_vector.o\
        ./SRC/vectors/parallel_vector.o\
        ./SRC/vectors/vectorops.o\
        ./SRC/matrices/matrix.o\
        ./SRC/matrices/arnoldimatrix.o\
        ./SRC/matrices/dense_matrix.o\
        ./SRC/matrices/coo_matrix.o\
        ./SRC/matrices/csr_matrix.o\
        ./SRC/matrices/parallel_csr_matrix.o\
        ./SRC/matrices/matrixops.o\
        ./SRC/preconditioners/ilu.o\
        ./SRC/schurcheb/superlu.o\
        ./SRC/schurcheb/arpack.o\
        ./SRC/schurcheb/mumps.o\
        ./SRC/schurcheb/pardiso.o\
        ./SRC/schurcheb/dsolver.o\
        ./SRC/schurcheb/schurshift.o\
        ./SRC/schurcheb/chebeig.o\

# Rules
default: libschurcheb.a
lib: libschurcheb.a

all: $(ALLEXE)
%.o : %.f
	$(FORT) $(FLAGS) $(INCLUDES) -o $@ -c $<
%.o : %.c
	$(CC) $(FLAGS) $(INCLUDES) -o $@ -c $<
%.o : %.cpp
	$(CXX) $(FLAGS) $(INCLUDES) -o $@ -c $<
ifneq ($(USING_CUDA),0)
%.o : %.cu
	$(NVCC) $(FLAGS) $(INCLUDES) -o $@ -c $<
endif

# Lib

libschurcheb.a: $(SRC)
	$(AR) $@ $(SRC)
	$(RANLIB) $@
	rm -rf build;mkdir build;mkdir build/lib;mkdir build/include;
	cp libschurcheb.a build/lib;

clean:
	rm -f *.a
	rm -rf ./build
	rm -f ./SRC/schurcheb/*.o
	rm -f ./SRC/utils/*.o
	rm -f ./SRC/vectors/*.o
	rm -f ./SRC/matrices/*.o
	rm -f ./SRC/preconditioners/*.o

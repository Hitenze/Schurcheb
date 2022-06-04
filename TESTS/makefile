
include ../makefile.in

.SUFFIXES : .c .cpp .o

INCLUDES = -I../INC -I../SRC

ifneq ($(USING_SUPERLU),0)
FLAGS += -DSCHURCHEB_SUPERLU
INCLUDES += -I$(SUPERLU_INCLUDE_PATH)
endif

ifneq ($(USING_MUMPS),0)
FLAGS += -DSCHURCHEB_MUMPS
INCLUDES += -I$(MUMPS_INCLUDE_PATH)
endif

ifneq ($(USING_MKL),0)
LIBMKL = -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_scalapack_lp64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -liomp5 -lpthread -lmpifort -lm -ldl -lmpifort -lmpifort -lifport -lmpifort     -lmpi     -lmpigi     -ldl     -lrt     -lpthread     -Bdynamic     -Bstatic     -lifport     -lifcoremt     -limf     -lsvml     -Bdynamic     -lm     -Bstatic     -lipgo     -Bdynamic     -liomp5     -Bstatic     -lirc     -Bdynamic     -lpthread     -Bstatic     -lsvml     -Bdynamic     -lc     -lgcc     -lgcc_s     -Bstatic     -lirc_s     -Bdynamic     -ldl     -lc
INCLUDES += -I$(MKLROOT)/include
endif

LIB := $(LIBGFORTRAN)\
		-L../build/lib/ -lschurcheb\
                -L../$(LAPACK_PATH) $(LIBLAPACK)\
                -L$(PARPACK_PATH) $(LIBPARPACK)\
                $(LIBSUPERLU)\
                $(LIBCOMBBLAS)\
                $(LIBMPIF)\
                $(LIBMUMPS)\
                $(LIBSCALAPACK)\
                -L../$(METIS_PATH)/libparmetis -lparmetis\
                -L../$(METIS_PATH)/libmetis -lmetis\
                $(LIBOPENMP) -lgomp $(LIBMKL)

# Rules
ALLEXE = test_min.ex test_schur.ex test_parpack.ex

default: test_min.ex test_schur.ex
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

# Exe

test_min.ex: test_min.o $(EXTRA) $(EXTERNAL)
	$(LINK) $(FLAGS) -o test_min.ex $(INCLUDES) test_min.o $(EXTRA) $(EXTERNAL) $(LIB)

test_schur.ex: test_schur.o $(EXTRA) $(EXTERNAL)
	$(LINK) $(FLAGS) -o test_schur.ex $(INCLUDES) test_schur.o $(EXTRA) $(EXTERNAL) $(LIB)

test_parpack.ex: test_parpack.o $(EXTRA) $(EXTERNAL)
	$(LINK) $(FLAGS) -o test_parpack.ex $(INCLUDES) test_parpack.o $(EXTRA) $(EXTERNAL) $(LIB)

clean:
	rm -f *.ex *.o
	rm -f TempData/*

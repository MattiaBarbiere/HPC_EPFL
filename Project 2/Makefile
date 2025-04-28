# Compilers
NVCC=nvcc
CC=gcc
LD=${NVCC}

# Flags
CXXFLAGS+= -I${OPENBLAS_ROOT}/include
CFLAGS+= -I${OPENBLAS_ROOT}/include
LDFLAGS+=-lm -L${OPENBLAS_ROOT}/lib -lopenblas

# Object files
CU_OBJS=cg_main.o cg.o matrix.o matrix_coo.o
C_OBJS=mmio.o

OBJS=$(CU_OBJS) $(C_OBJS)

all: cgsolver

cgsolver: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LDFLAGS)

# Compile CUDA files
%.o: %.cu
	$(NVCC) $(CXXFLAGS) -c $< -o $@

# Compile C files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f cgsolver *.o *~

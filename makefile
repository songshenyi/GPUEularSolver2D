CC = /export/cuda4.1/cuda/bin/nvcc
LIB = -L/export/cuda4.1/cuda/lib64/
INC = -I/export/cuda4.1/cuda/include/

.SUFFIXES: $(.SUFFIXES) .cu .o

CU_SRC = gpu_solver.cu gpu_init.cu gpu_fvm.cu
OBJL  = $(CU_SRC:.cu=.o)

%.o: %.cu
	$(CC) -O3 -arch=sm_20 $(INC) -c $< -o $@ -Xptxas -dlcm=ca -Xptxas -abi=no

all : $(OBJL) 
	$(CC) $(OBJL) -o solver_gpu.out $(LIB) -lstdc++

clean:
	rm -f *.o solver_gpu.out

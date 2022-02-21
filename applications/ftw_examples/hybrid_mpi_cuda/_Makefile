#!/bin/make

LIBS=-L$(HOME)/lib -L/opt/apps/cuda/3.0/cuda/lib64/

hybrid_mpi_cuda:	hybrid_MPI_cuda.c hybrid_mpi_CUDA.cu
	nvcc hybrid_mpi_CUDA.cu -c 
	mpicc hybrid_MPI_cuda.c hybrid_mpi_CUDA.o $(LD_LIBRARY_PATH) -lcudart


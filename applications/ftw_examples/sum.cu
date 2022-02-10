/* helloCUDA.cu */

/****************************************************************************/
/*                                                                          */
/*  (C) 2010 Texas Advanced Computing Center.                               */
/*                                                                          */
/*  For information, contact Frank Willmore:  willmore@tacc.utexas.edu      */
/*                                                                          */
/*  Shareable in accordance with TACC and University of Texas policies.     */
/*                                                                          */
/****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h> 
#include <assert.h>

#define GRID_DIMENSION 2
#define BLOCK_DIMENSION 4 

__device__ char d_data_array[GRID_DIMENSION * BLOCK_DIMENSION];
__device__ char d_data_array2D[GRID_DIMENSION][BLOCK_DIMENSION];
__device__ int d_sum_array[BLOCK_DIMENSION];
__device__ float d_mean_array[GRID_DIMENSION];
__device__ float d_std_array[GRID_DIMENSION];

__global__ void increment()
{
  int idx;
  idx = blockIdx.x * BLOCK_DIMENSION + threadIdx.x;
  d_data_array[idx]++;
}

__global__ void increment2D()
{
  d_data_array2D[blockIdx.x][threadIdx.x]++;
}

__global__ void printKernel()
{
  d_sum_array[threadIdx.x] += d_data_array2D[blockIdx.x][threadIdx.x];
//  printf("[%2d][%2d]:::\t%d\n", blockIdx.x, threadIdx.x, d_data_array2D[blockIdx.x][threadIdx.x]);
//  printf("sum[%2d]:::\t%d\n", threadIdx.x, d_sum_array[threadIdx.x]);
}

__global__ void calculateSum()
{
  d_sum_array[threadIdx.x] += d_data_array2D[blockIdx.x][threadIdx.x];
}

__global__ void calculateMean(char *h_nrgra, char *d_nrgra)
{
//  unsigned int idx = gridDim.x * GRID_DIMENSION + blockIdx.x;
//  unsigned int idy = gridDim.y * GRID_DIMENSION + blockIdx.y;
//  checksum += data_array[idx][idy];
}

__global__ void calculateStandardDeviation(char *h_nrgra, char *d_nrgra)
{
//shared mem example
//  unsigned int idx = gridDim.x * GRID_DIMENSION + blockIdx.x;
//  unsigned int idy = gridDim.y * GRID_DIMENSION + blockIdx.y;
//  checksum += data_array[idx][idy];
}

int main(int argc, char* argv[])
{
  int i, j;
  FILE *fptr;
  char h_data_array[GRID_DIMENSION * BLOCK_DIMENSION];
  char h_data_array2D[GRID_DIMENSION][BLOCK_DIMENSION];
  int h_sum_array[BLOCK_DIMENSION];
  size_t size = sizeof(h_data_array);

  // generate an array with random data, then copy it to the device
  fptr = fopen("/dev/urandom", "r");
  fread(h_data_array, size, 1, fptr);
  fclose(fptr);

  for (i=0; i< GRID_DIMENSION * BLOCK_DIMENSION; i++) printf("[%2d] = \t%d\n", i, h_data_array[i]);

  dim3 dimGrid(GRID_DIMENSION);
  dim3 dimBlock(BLOCK_DIMENSION);
 
  /////////////////// 1D array /////////////////////////////////

  cudaError_t r = cudaMemcpyToSymbol(d_data_array, h_data_array, size, 0, cudaMemcpyHostToDevice);
  cudaThreadSynchronize(); // block until the device has completed
printf("cuda error:  %s\n", cudaGetErrorString(r));
  assert(r == cudaSuccess);

  increment<<< dimGrid, dimBlock >>>();
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  r = cudaMemcpyFromSymbol(h_data_array, d_data_array, size, 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  for (i=0; i< GRID_DIMENSION * BLOCK_DIMENSION; i++) printf("[%2d] = \t%d\n", i, h_data_array[i]);

  /////////////////// 2D array /////////////////////////////////

  r = cudaMemcpyToSymbol(d_data_array2D, h_data_array, size, 0, cudaMemcpyHostToDevice);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);
 
  increment2D<<< dimGrid, dimBlock >>>();
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  r = cudaMemcpyFromSymbol(h_data_array2D, d_data_array2D, size, 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  for (i=0; i< GRID_DIMENSION; i++) for (j=0; j<BLOCK_DIMENSION; j++) printf("[%2d][%2d] = \t%d\n", i, j, h_data_array2D[i][j]);

  /////////////////// sum //////////////////////////////////////

  memset(h_sum_array, 0, sizeof(h_sum_array));
  for (i=0; i< BLOCK_DIMENSION; i++) printf("memset:::[%2d] = \t%d\n", i, h_sum_array[i]);
  r = cudaMemcpyToSymbol(d_sum_array, h_sum_array, sizeof(h_sum_array), 0, cudaMemcpyHostToDevice);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  calculateSum<<< dimGrid, dimBlock >>>();
  //printKernel<<< dimGrid, dimBlock >>>();
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  r = cudaMemcpyFromSymbol(h_sum_array, d_sum_array, sizeof(h_sum_array), 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  for (i=0; i< BLOCK_DIMENSION; i++) printf("[%2d] = \t%d\n", i, h_sum_array[i]);

}

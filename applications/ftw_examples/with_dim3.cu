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

#define BLOCKS_PER_GRID 16
#define THREADS_PER_BLOCK 16 
#define N_POPULATION 10000

__device__ char d_data_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK][N_POPULATION];
__device__ int d_sum_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];
__device__ float d_mean_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];
__device__ float d_std_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];

// reduce over N_POPULATION

__global__ void calculateMean()
{
  // auto variables other than arrays are register
  int index; 
  int sum=0;
  int sample_number;

  // get the sample number for this thread
  sample_number = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
  
  // calculate the sum for this thread
  for (index=0; index<N_POPULATION; index++) sum += d_data_array[sample_number][index];
 
  // record the sum and mean in global memory
  d_sum_array[sample_number] = sum;
  d_mean_array[sample_number] = (float)sum / N_POPULATION; 
}

// use persistent data (sum) to calculate variance
__global__ void calculateStandardDeviation()
{
  int sample_number;
  float v_sum = 0.0f;
  float delta;
  float variance;
  int index;

  // get the sample number for this thread
  sample_number = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;

  // calculate the sum for this thread
  for (index=0; index<N_POPULATION; index++)
  {
    delta = (float)d_data_array[sample_number][index] - d_mean_array[sample_number];
    v_sum += delta * delta;
  }
 
  variance = v_sum / N_POPULATION;
  d_std_array[sample_number] = sqrt(variance);
}

int main(int argc, char* argv[])
{
  int i, j;
  FILE *fptr;
  char h_data_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK][N_POPULATION];
  int h_sum_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];
  float h_std_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];
  float h_mean_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];
  size_t size = sizeof(h_data_array);

  // generate an array with random data, then copy it to the device
  fptr = fopen("/dev/urandom", "r");
  fread(h_data_array, size, 1, fptr);
  fclose(fptr);

//  for (i=0; i<(BLOCKS_PER_GRID * THREADS_PER_BLOCK); i++) for (j=0; j<N_POPULATION; j++) printf("[%2d] = \t%d\n", i, h_data_array[i][j]);

  dim3 grid_dimension(BLOCKS_PER_GRID);
  dim3 block_dimension(THREADS_PER_BLOCK);
 
  /////////////////// sum //////////////////////////////////////

  cudaError_t r = cudaMemcpyToSymbol(d_data_array, h_data_array, sizeof(h_data_array), 0, cudaMemcpyHostToDevice);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  calculateMean<<< grid_dimension, block_dimension >>>();
  calculateMean<<< BLOCKS_PER_GRID, THREADS_PER_BLOCK >>>();
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  r = cudaMemcpyFromSymbol(h_sum_array, d_sum_array, sizeof(h_sum_array), 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  /////////////////// standard deviation //////////////////////

//  memset(h_std_array, 0, sizeof(h_std_array));
//  r = cudaMemcpyToSymbol(d_std_array, h_std_array, sizeof(h_std_array), 0, cudaMemcpyHostToDevice);
//  cudaThreadSynchronize(); // block until the device has completed
//  assert(r == cudaSuccess);

  calculateStandardDeviation<<< grid_dimension, block_dimension >>>();
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  r = cudaMemcpyFromSymbol(h_mean_array, d_mean_array, sizeof(h_mean_array), 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  r = cudaMemcpyFromSymbol(h_std_array, d_std_array, sizeof(h_std_array), 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  for (i=0; i< BLOCKS_PER_GRID * THREADS_PER_BLOCK; i++) printf("[%2d] = \t%d\t%f\t%f\n", i, h_sum_array[i], h_mean_array[i], h_std_array[i]);
}


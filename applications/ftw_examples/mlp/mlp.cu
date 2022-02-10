/* mlp.cu */

/****************************************************************************/
/*                                                                          */
/*                                                                          */
/*  Multi-level parallel code example.                                      */
/*  MPI(multiprocess) / pthread(multithread) / CUDA (GPU)                   */
/*                                                                          */
/*                                                                          */
/*  (C) 2010 Texas Advanced Computing Center.  All rights reserved.         */
/*                                                                          */
/*  For information, contact Frank Willmore:  willmore@tacc.utexas.edu      */
/*                                                                          */
/****************************************************************************/

#include <stdio.h>
#include <assert.h>

#define BLOCKS_PER_GRID 4
#define THREADS_PER_BLOCK 256 
#define N_POPULATION 10000

__device__ char d_data_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK][N_POPULATION];
__device__ int d_sum_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];
__device__ float d_mean_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];
__device__ float d_std_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];

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
  int index;
  int sample_number;
  float v_sum = 0.0f;
  float delta;
  float variance;

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

void *ThreadMain(int *histogram)
{
  int sample_number;
  FILE *fptr;
  char h_data_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK][N_POPULATION];
  int h_sum_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];
  float h_std_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];
  float h_mean_array[BLOCKS_PER_GRID * THREADS_PER_BLOCK];
  size_t size = sizeof(h_data_array);

  // generate an array with random data, then copy it to the device
  printf("Reading random data from /dev/urandom...\n");
  fptr = fopen("/dev/urandom", "r");
  fread(h_data_array, size, 1, fptr);
  fclose(fptr);
  printf("Read %d bytes from /dev/urandom...\n\n", size);

  /////////////////// sum //////////////////////////////////////

  printf("Calculating sums and means...\n\n");

  cudaError_t result = cudaMemcpyToSymbol(d_data_array, h_data_array, sizeof(h_data_array), 0, cudaMemcpyHostToDevice);
  cudaThreadSynchronize(); // block until the device has completed
  assert(result == cudaSuccess);

  calculateMean<<< BLOCKS_PER_GRID, THREADS_PER_BLOCK >>>();
  cudaThreadSynchronize(); // block until the device has completed
  assert(result == cudaSuccess);

  result = cudaMemcpyFromSymbol(h_sum_array, d_sum_array, sizeof(h_sum_array), 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize(); // block until the device has completed
  assert(result == cudaSuccess);

  /////////////////// standard deviation //////////////////////

  printf("Calculating standard deviations...\n\n");

  calculateStandardDeviation<<< BLOCKS_PER_GRID, THREADS_PER_BLOCK >>>();
  cudaThreadSynchronize(); // block until the device has completed
  assert(result == cudaSuccess);

  result = cudaMemcpyFromSymbol(h_mean_array, d_mean_array, sizeof(h_mean_array), 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize(); // block until the device has completed
  assert(result == cudaSuccess);

  result = cudaMemcpyFromSymbol(h_std_array, d_std_array, sizeof(h_std_array), 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize(); // block until the device has completed
  assert(result == cudaSuccess);

  for (sample_number=0; sample_number < BLOCKS_PER_GRID * THREADS_PER_BLOCK; sample_number++) 
    printf("x_mean[%3d] = %f\t\tx_std[%3d] = %f\n", sample_number, h_mean_array[sample_number], sample_number, h_std_array[sample_number]);

  /////////////////// histogram //////////////////////////////

  printf("\nBuilding histogram of results...\n\n");

  float width=0.1, start_x=-3.0;
  int bin, n_bins = 51;

  for (bin = 0; bin < 51; bin++) histogram[bin] = 0;

  for (sample_number = 0; sample_number < (BLOCKS_PER_GRID * THREADS_PER_BLOCK); sample_number++)
  {
    bin = (int)floor((h_mean_array[sample_number] - start_x)/width);
    histogram[bin]++;
  }

  for (bin = 0; bin < n_bins; bin++) 
  {
    printf("%f\t%d\t", start_x + bin*width, histogram[bin]);
    while (histogram[bin]-- > 0) printf("X");
    printf("\n");
  }
 
  return;
}


/* symbol.cu */

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

__device__ char d_data_array[16];

__global__ void calculate()
{
  int idx;

  idx = threadIdx.x;
  d_data_array[idx]++;
}

int main(int argc, char* argv[])
{
  int i;
  FILE *fptr;
  char h_data_array[16];
  size_t size = sizeof(h_data_array);

printf("size = %d\n", size);

  // generate an array with random data, then copy it to the device
  fptr = fopen("/dev/urandom", "r");
  fread(h_data_array, size, 1, fptr);
  fclose(fptr);

  for (i=0; i< 16; i++) printf("[%2d] = \t%d\n", i, h_data_array[i]);

  cudaError_t r = cudaMemcpyToSymbol(d_data_array, h_data_array, size, 0, cudaMemcpyHostToDevice);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  dim3 dimGrid(1);
  dim3 dimBlock(16);
 
  calculate<<< dimGrid, dimBlock >>>();
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  r = cudaMemcpyFromSymbol(h_data_array, d_data_array, size, 0, cudaMemcpyDeviceToHost);
  cudaThreadSynchronize(); // block until the device has completed
  assert(r == cudaSuccess);

  for (i=0; i< 16; i++) printf("[%2d] = \t%d\n", i, h_data_array[i]);

//  dim3 dimGrid(2, 2);
//  dim3 dimBlock(8, 8, 16);
//  calculateMean<<< 1, dimBlock >>>(h_nrgra, d_nrgra);
//  err = cudaGetLastError();
//  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err)); 
//  cudaMemcpy( d_nrgra, h_nrgra, sizeof(h_nrgra), cudaMemcpyDeviceToHost );
//  err = cudaGetLastError();
//  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err)); 

}


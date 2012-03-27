/* simple.cu */

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

__global__ void kernel()
{
  int idx;
  int array[4];

  idx = threadIdx.x;
  array[0] = idx++;
  array[1] = idx++;
  array[2] = idx++;
  array[3] = idx++;
  idx++;
  idx++;
  idx++;
  idx++;

}

int main()
{
  dim3 dimGrid(1);
  dim3 dimBlock(64);

  kernel<<< dimGrid, dimBlock>>>();
  cudaThreadSynchronize(); // block until the device has completed
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err)); 
}

/******************************************************************************/
/*                                                                            */
/*  (C) 2010 Texas Advanced Computing Center.  All rights reserved.           */
/*  For information, contact Frank Willmore:  willmore@tacc.utexas.edu        */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>
#include <assert.h>

           char h_string[256];
__device__ char d_string[256];

__global__ void toUpper()
{
  if ((d_string[threadIdx.x] <= 122) && (d_string[threadIdx.x]) >=97)
    d_string[threadIdx.x] -= 32;
}

int main(int argc, char* argv[])
{
  sprintf(h_string, "hello world, this is my first CUDA program ever.");
  
  cudaMemcpyToSymbol(d_string, h_string, sizeof(h_string), 0, cudaMemcpyHostToDevice);
  toUpper<<< 1, 256 >>>();
  cudaMemcpyFromSymbol(h_string, d_string, sizeof(h_string), 0, cudaMemcpyDeviceToHost);

  printf("%s\n", h_string);
}

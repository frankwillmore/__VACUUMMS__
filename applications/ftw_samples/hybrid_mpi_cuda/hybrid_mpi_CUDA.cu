/******************************************************************************/
/*                                                                            */
/*  (C) 2010 Texas Advanced Computing Center.  All rights reserved.           */
/*  For information, contact Frank Willmore:  willmore@tacc.utexas.edu        */
/*                                                                            */
/******************************************************************************/

#include <stdio.h>

__device__ char d_string[65536][256];

extern "C" void cmain();

__global__ void toUpper()
{
  if ((d_string[blockIdx.x][threadIdx.x] <= 122) && (d_string[blockIdx.x][threadIdx.x]) >=97)
    d_string[blockIdx.x][threadIdx.x] -= 32;
}

void cmain()
{
  char line[65536][256];
  int n_lines;

  for (n_lines=0; !feof(stdin); n_lines++) fgets(&line[n_lines][0], 256, stdin);

  cudaMemcpyToSymbol(d_string, line, sizeof(line), 0, cudaMemcpyHostToDevice);
  toUpper<<< n_lines, 256 >>>();
  cudaMemcpyFromSymbol(line, d_string, sizeof(line), 0, cudaMemcpyDeviceToHost);

  for (int i=0; i<n_lines; i++) printf("%s", line[i]);
}



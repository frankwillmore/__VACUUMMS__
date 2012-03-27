// CopyBackAndForth.cu

#include <assert.h>
#include <stdio.h>

__device__ char devarray[16];

extern "C" void runTest()
{
   char zerobuf[16];
   memset(zerobuf, '@', sizeof(zerobuf));

   cudaError_t r = cudaMemcpyToSymbol(devarray, zerobuf, sizeof(zerobuf), 0, cudaMemcpyHostToDevice);
   assert(cudaSuccess == r);

   char out[16];
   r = cudaMemcpyFromSymbol(out, devarray, sizeof(devarray), 0, cudaMemcpyDeviceToHost);
   assert(cudaSuccess == r);

   assert(memcmp(out, zerobuf, sizeof(out)) == 0);
}

int main()
{
  runTest();
}

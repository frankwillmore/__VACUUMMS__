#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

void synchronizeAndCheckReturnStatus()
{
  cudaThreadSynchronize();
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) 
  {
    printf("return status:  %s\n", cudaGetErrorString(status)); 
    exit(0);
  }
}

int main()
{
  cudaDeviceProp properties;
  int device;

//  cudaGetDevice(&device);
//  synchronizeAndCheckReturnStatus();
  cudaGetDeviceProperties(&properties, device);
  synchronizeAndCheckReturnStatus();

  printf("Properties of device number   %d:\n\n", device);
  printf("name                          %s\n", properties.name);
  printf("warpSize                      %d\n", properties.warpSize);
  printf("totalGlobalMem                %d\n", properties.totalGlobalMem);
  printf("sharedMemPerBlock             %d\n", properties.sharedMemPerBlock);
  printf("regsPerBlock                  %d\n", properties.regsPerBlock);
  printf("memPitch                      %d\n", properties.memPitch);
  printf("maxThreadsPerBlock            %d\n", properties.maxThreadsPerBlock);
  printf("maxThreadsDim[0]              %d\n", properties.maxThreadsDim[0]);
  printf("maxThreadsDim[1]              %d\n", properties.maxThreadsDim[1]);
  printf("maxThreadsDim[2]              %d\n", properties.maxThreadsDim[2]);
  printf("maxGridSize[0]                %d\n", properties.maxGridSize[0]);
  printf("maxGridSize[1]                %d\n", properties.maxGridSize[1]);
  printf("maxGridSize[2]                %d\n", properties.maxGridSize[2]);
  printf("totalConstMem                 %d\n", properties.totalConstMem);
  printf("major                         %d\n", properties.major);
  printf("minor                         %d\n", properties.minor);
  printf("clockRate                     %d\n", properties.clockRate);
  printf("textureAlignment              %d\n", properties.textureAlignment);
  printf("deviceOverlap                 %d\n", properties.deviceOverlap);
  printf("multiProcessorCount           %d\n", properties.multiProcessorCount);
  printf("kernelExecTimeoutEnabled      %d\n", properties.kernelExecTimeoutEnabled);
  printf("integrated                    %d\n", properties.integrated);
  printf("canMapHostMemory              %d\n", properties.canMapHostMemory);
  printf("computeMode                   %d\n", properties.computeMode);
}


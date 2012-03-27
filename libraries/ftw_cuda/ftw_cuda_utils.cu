/* ftw_cuda.cu */

#include <ftw_cuda.h>
#include <ftw_types.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>


// I took the kernel prototype out of the header file, because the header is included by C/C++ compilers that don't know what a kernel is...

//  This is the routine to call from outside the library
extern "C" int ftw_getNumberOfDevices()
{
	int count = 0;
	cudaGetDeviceCount(&count);
	printf("%d devices present.\n", count);
	return count;
}

extern "C" void ftw_printDeviceInfo(int device = 0)
{	
	int device_number = device;
	struct cudaDeviceProp prop;

	printf("Device #%d Information: \n\n", device_number);
  
	cudaGetDeviceProperties(&prop, device_number);

	printf("canMapHostMemory  		%d\n", prop.canMapHostMemory);
	printf("clockRate         		%d\n", prop.clockRate);
	printf("computeMode	  		%d\n", prop.computeMode);
	printf("concurrentKernels 		%d\n", prop.concurrentKernels);
	printf("deviceOverlap	  		%d\n", prop.deviceOverlap);
	printf("ECCEnabled			%d\n", prop.ECCEnabled); 
	printf("integrated	 		%d\n", prop.integrated);
	printf("kernelExecTimeout Enabled	%d\n", prop.kernelExecTimeoutEnabled);
	printf("major				%d\n", prop.major);
	printf("maxGridSize			%d\t%d\t%d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("maxTexture1D			%d\n", prop.maxTexture1D);
	printf("maxTexture2D			%d\t%d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
//	printf("maxTexture2DArray		%d\t%d\t%d\n", prop.maxTexture2DArray[0], prop.maxTexture2DArray[1], prop.maxTexture2DArray[2]);
	printf("maxTexture3D			%d\t%d\t%d\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);
	printf("maxThreadsDim			%d\t%d\t%d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxThreadsPerBlock		%d\n", prop.maxThreadsPerBlock);
	printf("memPitch			%d\n", prop.memPitch);
	printf("minor				%d\n", prop.minor);
	printf("multProcessorCount		%d\n", prop.multiProcessorCount);
	printf("name				%s\n", prop.name);
	printf("pciBusID			%d\n", prop.pciBusID);
	printf("pciDeviceID			%d\n", prop.pciDeviceID);
	printf("regsPerBlock			%d\n", prop.regsPerBlock );
	printf("sharedMemPerBlock		%lu\n", prop.sharedMemPerBlock );
	printf("surfaceAlignment		%d\n", prop.surfaceAlignment );
	printf("tccDriver			%d\n", prop.tccDriver);
	printf("textureAlignment		%d\n", prop.textureAlignment);
	printf("totalConstMem			%lu\n", prop.totalConstMem);
	printf("totalGlobalMem			%lu\n", prop.totalGlobalMem);
	printf("warpSize			%d\n", prop.warpSize);
}


#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

void checkCUDAError(const char *msg);
long int launch_time;
void setLaunchTime(); 
long int timeSinceLaunch();
void runKernel();
void allocateAndCopy(size_t mem_size);

__global__ void EnergyKernel(float *d_f);
__global__ void EnergyKernel2(float *d_f);
__global__ void EnergyKernel3(float *d_f);
__device__ void innerKernel(float *d_f);

void checkCUDAError(const char *msg);

__device__ float s_f[256];

main()
{
  long int time;

  setLaunchTime();

  printf("priming the GPU...\n");
  allocateAndCopy(4);
  printf("\n\n\n");
  time = timeSinceLaunch();
  runKernel();
  printf("%d\n", timeSinceLaunch() - time);
}

void setLaunchTime() { launch_time = timeSinceLaunch(); }

long int timeSinceLaunch()
{
   struct timeval tv;
   struct timezone tz;
   struct tm *tm;
   gettimeofday(&tv, &tz);
   tm=localtime(&tv.tv_sec);
   return (tm->tm_hour * 3600000000 + tm->tm_min * 60000000 + tm->tm_sec * 1000000 + tv.tv_usec - launch_time);
}

void runKernel()
{
  float *h_f;
  float *d_f;
  int n_floats = 256;
  int i;

  h_f = (float*)malloc(sizeof(float) * n_floats);

  for (i=0; i<n_floats; i++) h_f[i] = 1.0f + i;

  cudaMalloc((void**)&d_f, sizeof(float) * n_floats);
  checkCUDAError("malloc");

  for (i=0; i<n_floats; i++) h_f[i] = 1.0f * i;

  cudaMemcpy( d_f, h_f, sizeof(float) * n_floats, cudaMemcpyHostToDevice );
  checkCUDAError("mem copy");

  dim3 dimGrid(1);
  dim3 dimBlock(n_floats);

printf("launching kernel...\n");

  EnergyKernel3<<< dimGrid, dimBlock >>>( d_f );
  cudaThreadSynchronize(); // block until the device has completed
  checkCUDAError("kernel execution");
  cudaMemcpy( h_f, d_f, sizeof(float) * n_floats, cudaMemcpyDeviceToHost );
  checkCUDAError("mem copy device 2 host");

for (i=0; i<n_floats; i++) printf("%d\t%f\n", i, h_f[i]);

  cudaFree(d_f);
  free(h_f);
}

void allocateAndCopy(size_t mem_size)
{
  float *h_f;
  float *d_f;

  h_f = (float*)malloc(mem_size);
  cudaMalloc((void**)&d_f, mem_size);
  checkCUDAError("malloc");
  cudaFree(d_f);
  free(h_f);
}

void checkCUDAError(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err)
  {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err) );
    //exit(-1);
  }
}

__global__ void EnergyKernel(float *d_f)
{
  __shared__ float junk;
  __shared__ float s_f[256];
  int i;

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  s_f[idx] = d_f[idx];

  junk = 0.0f + idx;
for(int j=0; j<1000; j++)
{
  for (i=0; i<68; i++)  junk += s_f[i] * s_f[i+1];

  junk /= (1.0f + i);
  junk /= (1.0f * (i+1));
  junk /= (1.0f * (i+2));

  junk = sqrt(junk);
  junk = sqrt(junk);
  junk = sqrt(junk);

  junk = sin(junk);
  junk = sin(junk);
  junk = cos(junk);
  junk = cos(junk);

  d_f[idx] = junk;
}
}

__global__ void EnergyKernel2(float *d_f)
{
  __shared__ float junk;
//  __shared__ float s_f[256];
  int i;

  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  s_f[idx] = d_f[idx];
  junk = 0.0f + idx;

  for (i=0; i<1000000; i++)
  {
    junk = (junk + s_f[i&255]) * junk;
  }
  d_f[idx] = junk;
}

__global__ void EnergyKernel3(float *d_f)
{
  __shared__ float junky;
  __shared__ float s_f[256];

  for (int i=0; i<256; i++) s_f[i] = d_f[i];
  for (int i=0; i<256; i++) junky += s_f[i];

  d_f[0] = junky;
  innerKernel<<< 1, 1 >>>( d_f );
}

__device__ void innerKernel(float *d_f)
{
  d_f[0] = 1.5;
}

/* CUDAEnergy.cu */

/*****************************************************************************/
/*                                                                           */ 
/* The function of the CUDA kernel is the same as that of the host routine   */
/* calculateEnergy()                                                         */ 
/*                                                                           */ 
/* This simply returns an array of values for the next 32 points along the   */ 
/* current trajectory of the particle, spaced at intervals specified by the  */ 
/* set of variables (resolution_x, resolution_y, resolution_z).              */ 
/*                                                                           */ 
/* Note that the index of 0 here corresponds to the NEXT point in moving the */
/* test particle forward, thus the addition of one to the idx value.         */
/*                                                                           */ 
/* This implementation is noted because it differs from the traditional C    */
/* practice of indexing the zero point (current position) with the zeroth    */
/* element of the array.                                                     */
/*                                                                           */ 
/*****************************************************************************/

#include <pmdgpu.h>

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg);
extern "C" void calculateEnergyArray(struct ThreadResult *h_thread);

__global__ void EnergyKernel(float *d_f, struct ThreadResult *d_thread)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  float repulsion=0;
  float attraction=0;
  float alpha;
  float dx, dy, dz, d, dd;

  float cuda_x = d_thread->test_x + (idx + 1) * d_thread->resolution_x;
  float cuda_y = d_thread->test_y + (idx + 1) * d_thread->resolution_y;
  float cuda_z = d_thread->test_z + (idx + 1) * d_thread->resolution_z;

  // evaluate energy at (cuda_x, cuda_y, cuda_z);
  for (int i=0; i< d_thread->verlet.close_atoms; i++)
  {
    // central atom
    dx = d_thread->verlet.x[i] - cuda_x;
    dy = d_thread->verlet.y[i] - cuda_y;
    dz = d_thread->verlet.z[i] - cuda_z;
    dd = dx*dx + dy*dy + dz*dz;
    d = sqrt(dd);
    alpha = pow(d_thread->verlet.r_ij[i], 3) / (d * dd);   
    repulsion += d_thread->verlet.epsilon_ij[i] * alpha * alpha * alpha;
    attraction += d_thread->verlet.epsilon_ij[i] * alpha * alpha;

    // forward atom 
    dx = d_thread->verlet.x[i] - (cuda_x + d_thread->orientation_dx1);
    dy = d_thread->verlet.y[i] - (cuda_y + d_thread->orientation_dy1);
    dz = d_thread->verlet.z[i] - (cuda_z + d_thread->orientation_dz1);
    dd = dx*dx + dy*dy + dz*dz;
    d = sqrt(dd);
    alpha = pow(d_thread->verlet.r_ij1[i], 3) / (d * dd);   
    repulsion += d_thread->verlet.epsilon_ij1[i] * alpha * alpha * alpha;
    attraction += d_thread->verlet.epsilon_ij1[i] * alpha * alpha;

    // backward atom
    dx = d_thread->verlet.x[i] - (cuda_x + d_thread->orientation_dx2);
    dy = d_thread->verlet.y[i] - (cuda_y + d_thread->orientation_dy2);
    dz = d_thread->verlet.z[i] - (cuda_z + d_thread->orientation_dz2);
    dd = dx*dx + dy*dy + dz*dz;
    d = sqrt(dd);
    alpha = pow(d_thread->verlet.r_ij2[i], 3) / (d * dd);   
    repulsion += d_thread->verlet.epsilon_ij2[i] * alpha * alpha * alpha;
    attraction += d_thread->verlet.epsilon_ij2[i] * alpha * alpha;
  } 
  
  d_f[idx] = (2 * repulsion - 3 * attraction);
}

extern "C" void calculateEnergyArray(struct ThreadResult *h_thread)
{
//  int i;

  struct ThreadResult *d_thread; // pointer for device memory
  float *d_f;

  // define grid and block size
  int numBlocks = 1;
  int numThreadsPerBlock = 32;

  size_t TmemSize = sizeof(struct ThreadResult);
  cudaMalloc( (void **) &d_thread, TmemSize );
  checkCUDAError("cudaMalloc1");
  size_t memSize = sizeof(float) * numBlocks * numThreadsPerBlock;
  cudaMalloc( (void **) &d_f, memSize );
  checkCUDAError("cudaMalloc2");

  dim3 dimGrid(numBlocks);
  dim3 dimBlock(numThreadsPerBlock);

  cudaMemcpy( d_thread, h_thread, TmemSize, cudaMemcpyHostToDevice );
  checkCUDAError("cudaMemcpy");

  EnergyKernel<<< dimGrid, dimBlock >>>( d_f, d_thread );
  cudaThreadSynchronize(); // block until the device has completed

  checkCUDAError("kernel execution"); // check if kernel execution generated an error
  cudaMemcpy( h_thread->energy_array, d_f, memSize, cudaMemcpyDeviceToHost );
  checkCUDAError("cudaMemcpy");

    // free device memory
    cudaFree(d_thread);
    cudaFree(d_f);

    return;
}

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(-1);
    }                         
}


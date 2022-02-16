/* CUDAEnergy.cu */

/*****************************************************************************/
/*                                                                           */
/*                                                                           */
/*****************************************************************************/

#include <genels.h>

extern "C" void calculateEnergy(struct ZThread *h_thread);
extern "C" void setResolution(float *gross_resolution, float *fine_resolution);

__device__ float _gross_resolution;
__device__ float _fine_resolution;

__global__ void EnergyKernel(struct ZThread *d_thread, struct EnergyArray *d_energy_array)
{
  unsigned int idx = blockIdx.x;
  unsigned int idy = blockIdx.y;
  unsigned int idz = threadIdx.x;

  float repulsion=0;
  float attraction=0;
  float alpha;
  float dx, dy, dz, d, dd;

  float cuda_x = d_thread->i * _gross_resolution + idx * _fine_resolution;
  float cuda_y = d_thread->j * _gross_resolution + idy * _fine_resolution;
  float cuda_z = d_thread->k * _gross_resolution + idz * _fine_resolution;

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
  } 

  d_energy_array->energy[idx][idy][idz] = (2 * repulsion - 3 * attraction);
}

extern "C" void setResolution(float *gross_resolution, float *fine_resolution)
{
  cudaMemcpyToSymbol((const char*)&_gross_resolution, gross_resolution, sizeof(float), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol((const char*)&_fine_resolution, fine_resolution, sizeof(float), 0, cudaMemcpyHostToDevice);
}

extern "C" void calculateEnergy(struct ZThread *thread)
{
  cudaError_t err;
  struct ZThread *d_thread; // pointer for device memory
  struct EnergyArray *d_energy_array;
  size_t TmemSize = sizeof(struct ZThread);

  err = cudaErrorUnknown;
  while (err != cudaSuccess)
  {
    cudaMalloc( (void **) &d_thread, TmemSize );
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "%s\n", cudaGetErrorString(err)); sleep(1);}
  }

  err = cudaErrorUnknown;
  while (err != cudaSuccess)
  {
    cudaMalloc( (void **) &d_energy_array, sizeof(struct EnergyArray) );
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "%s\n", cudaGetErrorString(err)); sleep(1);}
  }

  dim3 dimGrid(FINE_GRID_RESOLUTION, FINE_GRID_RESOLUTION);
  dim3 dimBlock(FINE_GRID_RESOLUTION, 1, 1);

  err = cudaErrorUnknown;
  while (err != cudaSuccess)
  {
    cudaMemcpy( d_thread, thread, TmemSize, cudaMemcpyHostToDevice );
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "%s\n", cudaGetErrorString(err)); sleep(1);}
  }

  EnergyKernel<<< dimGrid, dimBlock >>>( d_thread, d_energy_array );
  cudaThreadSynchronize(); // block until the device has completed
  err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err)); 

  cudaMemcpy( thread, d_thread, TmemSize, cudaMemcpyDeviceToHost );
  err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err)); 

  cudaMemcpy( thread->energy_array, d_energy_array, sizeof(struct EnergyArray), cudaMemcpyDeviceToHost );
  err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err)); 

  // free device memory
  cudaFree(d_thread);
  cudaFree(d_energy_array);
  return;
}


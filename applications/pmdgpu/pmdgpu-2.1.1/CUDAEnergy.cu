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
__device__ void cudaQuaternionRotate(float, float, float, float, float, float, float, float*, float*, float*);
extern "C" void calculateEnergyArray(struct ThreadResult *h_thread);

__global__ void EnergyKernel(float *d_f, struct ThreadResult *d_thread)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  float repulsion=0;
  float attraction=0;
  float alpha;
  float dx, dy, dz, d, dd;
  float rotx1, rotx2, roty1, roty2, rotz1, rotz2;

  float cuda_x = d_thread->test_x + (idx + 1) * d_thread->resolution_x;
  float cuda_y = d_thread->test_y + (idx + 1) * d_thread->resolution_y;
  float cuda_z = d_thread->test_z + (idx + 1) * d_thread->resolution_z;

  float rotation_angle = (idx + 1) * d_thread->rotation_angle_resolution;
//printf("ROTATION ANGLE[%d] = %f\n", idx, rotation_angle);
  // rotate the outer atoms through the angle
  cudaQuaternionRotate(d_thread->orientation_dx1, d_thread->orientation_dy1, d_thread->orientation_dz1, 
                   d_thread->axis_of_rotation_x, d_thread->axis_of_rotation_y, d_thread->axis_of_rotation_z, 
                   rotation_angle, &rotx1, &roty1, &rotz1);
  cudaQuaternionRotate(d_thread->orientation_dx2, d_thread->orientation_dy2, d_thread->orientation_dz2, 
                   d_thread->axis_of_rotation_x, d_thread->axis_of_rotation_y, d_thread->axis_of_rotation_z, 
                   rotation_angle, &rotx2, &roty2, &rotz2);

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
    dx = d_thread->verlet.x[i] - (cuda_x + rotx1);
    dy = d_thread->verlet.y[i] - (cuda_y + roty1);
    dz = d_thread->verlet.z[i] - (cuda_z + rotz1);
    dd = dx*dx + dy*dy + dz*dz;
    d = sqrt(dd);
    alpha = pow(d_thread->verlet.r_ij1[i], 3) / (d * dd);   
    repulsion += d_thread->verlet.epsilon_ij1[i] * alpha * alpha * alpha;
    attraction += d_thread->verlet.epsilon_ij1[i] * alpha * alpha;

    // backward atom
    dx = d_thread->verlet.x[i] - (cuda_x + rotx2);
    dy = d_thread->verlet.y[i] - (cuda_y + roty2);
    dz = d_thread->verlet.z[i] - (cuda_z + rotz2);
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
  cudaError_t err;

  struct ThreadResult *d_thread; // pointer for device memory
  float *d_f;

  // define grid and block size
  int numBlocks = 1;
  int numThreadsPerBlock = 32;

  size_t TmemSize = sizeof(struct ThreadResult);
  err = cudaErrorUnknown;
  while (err != cudaSuccess)
  {
    cudaMalloc( (void **) &d_thread, TmemSize );
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "%s\n", cudaGetErrorString(err)); sleep(1);}
  }

  size_t memSize = sizeof(float) * numBlocks * numThreadsPerBlock;
  err = cudaErrorUnknown;
  while (err != cudaSuccess)
  {
    cudaMalloc( (void **) &d_f, memSize );
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "%s\n", cudaGetErrorString(err)); sleep(1);}
  }

  dim3 dimGrid(numBlocks);
  dim3 dimBlock(numThreadsPerBlock);

  err = cudaErrorUnknown;
  while (err != cudaSuccess)
  {
    cudaMemcpy( d_thread, h_thread, TmemSize, cudaMemcpyHostToDevice );
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "%s\n", cudaGetErrorString(err)); sleep(1);}
  }

  err = cudaErrorUnknown;
  while (err != cudaSuccess)
  {
    EnergyKernel<<< dimGrid, dimBlock >>>( d_f, d_thread );
    cudaThreadSynchronize(); // block until the device has completed
    //checkCUDAError("kernel execution"); // check if kernel execution generated an error
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "%s\n", cudaGetErrorString(err)); sleep(1);}
  }

  err = cudaErrorUnknown;
  while (err != cudaSuccess)
  {
  cudaMemcpy( h_thread->energy_array, d_f, memSize, cudaMemcpyDeviceToHost );
  checkCUDAError("cudaMemcpy");
    err = cudaGetLastError();
    if (err != cudaSuccess) {fprintf(stderr, "%s\n", cudaGetErrorString(err)); sleep(1);}
  }

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

__device__ void cudaQuaternionRotate(float vx, float vy, float vz, float axisx, float axisy, float axisz, float angle, float *vxnew, float *vynew, float *vznew)
{
  float a, b, c, d; // components of quaternion z;
  float t2, t3, t4, t5, t6, t7, t8, t9, t10;  // temporary variables;
  float half_angle = angle * 0.5f;
  float sin_half_angle = sin(half_angle);
//printf("rotating angle %f\n", angle);

  a = cos(half_angle);
  b = axisx * sin_half_angle;
  c = axisy * sin_half_angle;
  d = axisz * sin_half_angle;

  t2 =   a*b;
  t3 =   a*c;
  t4 =   a*d;
  t5 =  -b*b;
  t6 =   b*c;
  t7 =   b*d;
  t8 =  -c*c;
  t9 =   c*d;
  t10 = -d*d;

 *vxnew = 2 * ((t8 + t10) * vx + (t6 -  t4) * vy + (t3 + t7) * vz) + vx;
 *vynew = 2 * ((t4 +  t6) * vx + (t5 + t10) * vy + (t9 - t2) * vz) + vy;
 *vznew = 2 * ((t7 -  t3) * vx + (t2 +  t9) * vy + (t5 + t8) * vz) + vz;
}


/* fv.cu */

//  IN:    A polymer configuration in .gfg format (x, y, z, sigma, epsilon) is read from standard input
//  OUT:   A free volume intensity is written in .fvi format (x, y, z, I) to standard output 

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>

#include <ftw_types.h>

/* macros */

ftw_Configuration h_configuration;
__device__ ftw_Configuration *d_configuration;

ftw_EnergyArray energy_array;
__device__ ftw_EnergyArray *d_energy_array;

int number_of_atoms=0;
float f_resolution;
float box_dimension = 25.0;
float r_i;
float epsilon_i;

__global__ void EnergyKernel(struct Configuration *d_configuration, struct EnergyArray *d_energy_array, int d_number_of_atoms, float d_box_dimension) {
  unsigned int idx = blockIdx.x;
  unsigned int idy = blockIdx.y;
  unsigned int idz = threadIdx.x;

  float repulsion=0;
  float attraction=0;
  float alpha;
  float dx, dy, dz, d, dd;
  float f_resolution = d_box_dimension / RESOLUTION;

  float cuda_x = idx * f_resolution;
  float cuda_y = idy * f_resolution;
  float cuda_z = idz * f_resolution;

  // evaluate energy at (cuda_x, cuda_y, cuda_z);
  for (int i=0; i< d_number_of_atoms; i++) {
    // central atom
    dx = d_configuration->atom[i].x - cuda_x;
    dy = d_configuration->atom[i].y - cuda_y;
    dz = d_configuration->atom[i].z - cuda_z;
    dd = dx*dx + dy*dy + dz*dz; d = sqrt(dd);
    alpha = pow(d_configuration->atom[i].r_ij, 3) / (d * dd);   
    repulsion += d_configuration->atom[i].epsilon_ij * alpha * alpha * alpha;
    attraction += d_configuration->atom[i].epsilon_ij * alpha * alpha;
  } 

//  d_energy_array->energy[idx][idy][idz] = (2 * repulsion - 3 * attraction);
//  Try with repulsion only
  d_energy_array->energy[idx][idy][idz] = (2 * repulsion);
}

int main(int argc, char *argv[]) {

printf("Broke.  \n\n");
exit(1);
  cudaError_t err;
  readCommandLineOptions(argc, argv);
  readPolymerConfiguration();

  /* allocate for energy array and configuration on device */
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_energy_array, sizeof(struct EnergyArray)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_configuration, sizeof(struct Configuration)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy( d_configuration, &h_configuration, sizeof(struct Configuration), cudaMemcpyHostToDevice ));

  dim3 dimGrid(RESOLUTION, RESOLUTION);
  dim3 dimBlock(RESOLUTION, 1, 1);

  EnergyKernel<<< dimGrid, dimBlock >>>( d_configuration, d_energy_array, number_of_atoms, box_dimension );
  cudaThreadSynchronize(); // block until the device has completed
  err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err)); 

  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy( &energy_array, d_energy_array, sizeof(energy_array), cudaMemcpyDeviceToHost ));

  // free device memory
  cudaFree(d_configuration);
  cudaFree(d_energy_array);

  /* now do something with it... */
  for (int i=0; i<RESOLUTION; i++)
    for (int j=0; j<RESOLUTION; j++)
      for (int k=0; k<RESOLUTION; k++)
        printf("%f\t%f\t%f\t%f\n", i*box_dimension / RESOLUTION, j*box_dimension / RESOLUTION, k*box_dimension / RESOLUTION, exp(energy_array.energy[i][j][k]/(0-298000))); 
//        printf("%f\t%f\t%f\t%f\n", i*box_dimension / RESOLUTION, j*box_dimension / RESOLUTION, k*box_dimension / RESOLUTION, exp(energy_array.energy[i][j][k]/(0-298))); 
  return 0; 
}

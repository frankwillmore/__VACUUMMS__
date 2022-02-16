/* essence_kernel.cu */

//  IN:    An array of free volume intensity is entered as ftw_FVI256 *in.  
//  OUT:   An array of free volume intensity is returned as ftw_FVI256 *out.  
//  
//  The kernel operates on the data by distributing the essence of each data point to the six nearest neighbors, according to the relative differences in fvi.

#include <ftw_gfg2fvi.h>
#include <ftw_config_parser.h>
#include <ftw_types.h>

#include <stdlib.h>
#include <math.h>

__global__ void EssenceKernel256(ftw_FVI256 *d_in, ftw_FVI256 *d_out) 
{
  unsigned int idx = blockIdx.x;
  unsigned int idy = blockIdx.y;
  unsigned int idz = threadIdx.x;

  // Six nearest neighbors are...

  unsigned int x1 = 255 & (idx + 1);
  unsigned int y1 = idy;
  unsigned int z1 = idz;
 
  unsigned int x2 = idx;
  unsigned int y2 = 255 & (idy + 1);
  unsigned int z2 = idz;
 
  unsigned int x3 = idx;
  unsigned int y3 = idy;
  unsigned int z3 = 255 & (idz + 1);

  unsigned int x4 = 255 & (idx - 1);
  unsigned int y4 = idy;
  unsigned int z4 = idz;
 
  unsigned int x5 = idx;
  unsigned int y5 = 255 & (idy - 1);
  unsigned int z5 = idz;
 
  unsigned int x6 = idx;
  unsigned int y6 = idy;
  unsigned int z6 = 255 & (idz - 1);
 
  // compute the difference in fvi for each

  // compute the sum of differences

  // generate the output array


 
  float cuda_x = idx * f_resolution_x;
  float cuda_y = idy * f_resolution_y;
  float cuda_z = idz * f_resolution_z;

  // evaluate energy at (cuda_x, cuda_y, cuda_z);
  for (int i=0; i< d_configuration->n_atoms; i++) {
    // central atom
    dx = d_configuration->atom[i].x - cuda_x;
    dy = d_configuration->atom[i].y - cuda_y;
    dz = d_configuration->atom[i].z - cuda_z;
    dd = dx*dx + dy*dy + dz*dz; d = sqrt(dd);
    alpha = pow(d_configuration->atom[i].sigma, 3) / (d * dd);   
    repulsion += d_configuration->atom[i].epsilon * alpha * alpha * alpha;
    attraction += d_configuration->atom[i].epsilon * alpha * alpha;
  } 

  // If NULL pointers are passed for the attraction or repulsion, no values are returned.
  if (d_attraction) d_attraction->energy[idx][idy][idz] = 3 * attraction;
  if (d_repulsion) d_repulsion->energy[idx][idy][idz] = 2 * repulsion;
  if (d_total) d_total->energy[idx][idy][idz] = 2 * repulsion - 3 * attraction;
}




//


//  This routine to be called from outside the library
extern "C" ftw_EnergyArray256 *GFGToEnergyArray256(ftw_GFG65536 *gfg, float sigma, float epsilon)
{
  ftw_EnergyArray256 	*d_energy_array;
  ftw_GFG65536 		*d_configuration;

  // replicate the gfg
  ftw_GFG65536 *h_configuration = replicateGFG65536(gfg); 

  // and cross-parameterize 
  for (int n=0; n<gfg->n_atoms; n++)
  {
    h_configuration->atom[n].sigma = pow(0.5f * (float)(pow(sigma, 6) + pow(h_configuration->atom[n].sigma, 6)), 0.1666666f);
    h_configuration->atom[n].epsilon = sqrt(epsilon * h_configuration->atom[n].epsilon);
  }

  // then do the calc
  cudaError_t err;
  /* allocate for energy array and configuration on device */
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_energy_array, sizeof(ftw_EnergyArray256)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_configuration, sizeof(ftw_GFG65536)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy( d_configuration, h_configuration, sizeof(ftw_GFG65536), cudaMemcpyHostToDevice ));

  dim3 dimGrid(256, 256);
  dim3 dimBlock(256, 1, 1);

  EnergyKernel256<<< dimGrid, dimBlock >>>(d_configuration, NULL, NULL, d_energy_array);
  cudaThreadSynchronize(); // block until the device has completed
  err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err)); 

  // retrieve result
  ftw_EnergyArray256 *h_energy_array = (ftw_EnergyArray256 *)malloc(sizeof(ftw_EnergyArray256));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy(h_energy_array, d_energy_array, sizeof(ftw_EnergyArray256), cudaMemcpyDeviceToHost ));

  // free device memory
  cudaFree(d_configuration);
  cudaFree(d_energy_array);

  free(h_configuration); // free host memory for replicated configuration

  return h_energy_array;
}


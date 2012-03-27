/* maltest.cu */

#include <ftw_config_parser.h>
#include <ftw_param.h>
#include <ftw_types.h>
#include <ftw_gfg2fvi.h>
#include <stdio.h>
#include <math.h>

__global__ void EnergyKernel256_612(ftw_GFG65536 *d_configuration, ftw_EnergyArray256 *d_attraction, ftw_EnergyArray256 *d_repulsion, ftw_EnergyArray256 *d_total)
{
  unsigned int idx = blockIdx.x;
  unsigned int idy = blockIdx.y;
  unsigned int idz = threadIdx.x;

  float repulsion=0;
  float attraction=0;
  float sigma_over_r_sq;
  float dx, dy, dz, dd;
  float f_resolution_x = d_configuration->box_x / 256;
  float f_resolution_y = d_configuration->box_y / 256;
  float f_resolution_z = d_configuration->box_z / 256;

  float cuda_x = idx * f_resolution_x;
  float cuda_y = idy * f_resolution_y;
  float cuda_z = idz * f_resolution_z;

  // evaluate energy at (cuda_x, cuda_y, cuda_z);
  for (int i=0; i< d_configuration->n_atoms; i++) {
    // central atom
    dx = d_configuration->atom[i].x - cuda_x;
    dy = d_configuration->atom[i].y - cuda_y;
    dz = d_configuration->atom[i].z - cuda_z;
    dd = dx*dx + dy*dy + dz*dz;
    sigma_over_r_sq = d_configuration->atom[i].sigma * d_configuration->atom[i].sigma / dd; // squared
    repulsion += d_configuration->atom[i].epsilon * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq;
    attraction += d_configuration->atom[i].epsilon * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq;
  }

  // If NULL pointers are passed for the attraction or repulsion, no values are returned.
  if (d_attraction) d_attraction->energy[idx][idy][idz] = 4 * attraction;
  if (d_repulsion) d_repulsion->energy[idx][idy][idz] = 4 * repulsion;
  if (d_total) d_total->energy[idx][idy][idz] = 4 * repulsion - 4 * attraction;
}

extern "C" ftw_EnergyArray256 *MALGFGToRepulsion256_612(ftw_GFG65536 *gfg, float sigma, float epsilon)
{
  ftw_EnergyArray256    *d_repulsion;
  ftw_GFG65536          *d_configuration;

fprintf(stderr, "banana pointers:  %ld\t%ld\n", d_configuration, d_repulsion);
fflush(stderr);

  // replicate the gfg
  ftw_GFG65536 *h_configuration = replicateGFG65536(gfg);

// and cross-parameterize use 6-12 rule
  for (int n=0; n<gfg->n_atoms; n++)
  {
    h_configuration->atom[n].sigma = 0.5f * (sigma + h_configuration->atom[n].sigma);
    h_configuration->atom[n].epsilon = sqrt(epsilon * h_configuration->atom[n].epsilon);
  }

  // then do the calc
  cudaError_t err;

int *p_int;

err = cudaErrorUnknown; 
fprintf(stderr, "%s\n", cudaGetErrorString(err));
fprintf(stderr, "p_int:  %ld\n", p_int);
err = cudaMalloc( (void **) &p_int, sizeof(int));
fprintf(stderr, "p_int:  %ld\n", p_int);

fprintf(stderr, "%s\n", cudaGetErrorString(err));
fflush(stderr);

  /* allocate for energy array and configuration on device */
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_repulsion, sizeof(ftw_EnergyArray256)));

  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_configuration, sizeof(ftw_GFG65536)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy( d_configuration, h_configuration, sizeof(ftw_GFG65536), cudaMemcpyHostToDevice ));

  dim3 dimGrid(256, 256);
  dim3 dimBlock(256, 1, 1);

  EnergyKernel256_612<<< dimGrid, dimBlock >>>(d_configuration, NULL, d_repulsion, NULL);
  cudaThreadSynchronize(); // block until the device has completed
  err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err));

  // retrieve result
  ftw_EnergyArray256 *h_repulsion = (ftw_EnergyArray256 *)malloc(sizeof(ftw_EnergyArray256));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy(h_repulsion, d_repulsion, sizeof(ftw_EnergyArray256), cudaMemcpyDeviceToHost ));

  // free device memory
  cudaFree(d_configuration);
  cudaFree(d_repulsion);

  free(h_configuration); // free host memory for replicated configuration

  return h_repulsion;
}

main(int argc, char *argv[]) 
{
  int i,j,k;
  double box_x=10, box_y=10, box_z=10;
  int potential = 612;
  int resolution = 256;
  int device_count;
  float temperature = 298.0;
  float sigma=0.0;
  float epsilon=1.0;

  cudaGetDeviceCount(&device_count);
  printf("%d device found.\n", device_count);

fprintf(stderr,"reading configuration\n");

  ftw_GFG65536 *gfg; 
fprintf(stderr, "gfg = %p\n", gfg);
  gfg = readGFG65536(stdin);
fprintf(stderr, "gfg = %p\n", gfg);

  gfg->box_x = box_x;
  gfg->box_y = box_y;
  gfg->box_z = box_z;

fprintf(stderr, "calculating resolution = %d for %d potential\n", resolution, potential);
fprintf(stderr, "setting device 1\n");
cudaSetDevice(1);

int *p_int;

cudaError_t err = cudaErrorUnknown;
fprintf(stderr, "main::%s\n", cudaGetErrorString(err));
fprintf(stderr, "main::p_int:  %ld\n", p_int);
err = cudaMalloc( (void **) &p_int, sizeof(int));
fprintf(stderr, "main::p_int:  %ld\n", p_int);
fprintf(stderr, "main::%s\n", cudaGetErrorString(err));

  ftw_EnergyArray256 *ea;
fprintf(stderr, "ea = %p\n", ea);
//point from integer without cast???  what is this returning???
  ea = MALGFGToRepulsion256_612(gfg, sigma, epsilon);
fprintf(stderr, "ea = %p\n", ea);

  for (i=0; i<resolution; i++) for (j=0; j<resolution; j++) for (k=0; k<resolution; k++)
    printf("%f\t%f\t%f\t%f\n", i*box_x / resolution, j*box_y / resolution, k*box_z / resolution, exp(ea->energy[i][j][k]/(-temperature))); 
}



/* ftw_gfg2fvi.cu */

//  IN:    A pointer to a ***non-replicated*** polymer configuration as *ftw_GFG65536.  
//  OUT:   A free volume intensity is returned as *ftw_FVI256.  
//  Input configuration is not modified.  
//  Cross-interaction values are stored for the replicated config only.

#include <ftw_gfg2fvi.h>
#include <ftw_config_parser.h>
#include <ftw_types.h>

#include <stdlib.h>
#include <math.h>

// I took the kernel prototype out of the header file, because the header is included by C/C++ compilers that don't know what a kernel is...
// NOTE:  this uses COMPASS / LJ 6-9 potential
__global__ void EnergyKernel256(ftw_GFG65536 *d_configuration, ftw_EnergyArray256 *d_attraction, ftw_EnergyArray256 *d_repulsion, ftw_EnergyArray256 *d_total) 
{
  unsigned int idx = blockIdx.x;
  unsigned int idy = blockIdx.y;
  unsigned int idz = threadIdx.x;

  float repulsion=0;
  float attraction=0;
  float alpha;
  float dx, dy, dz, d, dd;
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

__global__ void EnergyKernel512_612(ftw_GFG65536 *d_configuration, ftw_EnergyArray512 *d_attraction, ftw_EnergyArray512 *d_repulsion, ftw_EnergyArray512 *d_total) 
{
  unsigned int idx = blockIdx.x;
  unsigned int idy = blockIdx.y;
  unsigned int idz = threadIdx.x;

  float repulsion=0;
  float attraction=0;
  float sigma_over_r_sq;
  float dx, dy, dz, dd;
  float f_resolution_x = d_configuration->box_x / 512;
  float f_resolution_y = d_configuration->box_y / 512;
  float f_resolution_z = d_configuration->box_z / 512;

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

//  This is the routine to call from outside the library
extern "C" ftw_FVI256 *GFGToFVI256(ftw_GFG65536 *gfg, float sigma, float epsilon) 
{
  // call energy array then process each val
  ftw_EnergyArray256 *era = GFGToRepulsion256(gfg, sigma, epsilon);
  ftw_FVI256 *fvi = (ftw_FVI256*)malloc(sizeof(ftw_FVI256));

  // now process each value...
  for (int i=0; i<256; i++) for (int j=0; j<256; j++) for (int k=0; k<256; k++)
    fvi->intensity[i][j][k] = exp(era->energy[i][j][k]/-298000); // this is arbitrary... should be clarified
  
  return fvi;
}

// Now the C bindings...

//  This routine to be called from outside the library
extern "C" ftw_EnergyArray256 *GFGToRepulsion256_612(ftw_GFG65536 *gfg, float sigma, float epsilon)
{
  ftw_EnergyArray256 	*d_repulsion;
  ftw_GFG65536 		*d_configuration;

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

//  This routine to be called from outside the library
extern "C" ftw_EnergyArray512 *GFGToRepulsion512_612(ftw_GFG65536 *gfg, float sigma, float epsilon)
{
  ftw_EnergyArray512 	*d_repulsion;
  ftw_GFG65536 		*d_configuration;


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
  /* allocate for energy array and configuration on device */
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_repulsion, sizeof(ftw_EnergyArray512)));
fprintf(stderr, "malloc-ing enrgyarray...\n");
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_configuration, sizeof(ftw_GFG65536)));
fprintf(stderr, "malloc-ing gfg ...\n");
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy( d_configuration, h_configuration, sizeof(ftw_GFG65536), cudaMemcpyHostToDevice ));
fprintf(stderr, "mem copying...\n");

  dim3 dimGrid(512, 512);
  dim3 dimBlock(512, 1, 1);

fprintf(stderr, "launching kernel...\n");
  EnergyKernel512_612<<< dimGrid, dimBlock >>>(d_configuration, NULL, d_repulsion, NULL);
fprintf(stderr, "synchronizing...\n");
  cudaThreadSynchronize(); // block until the device has completed
  err = cudaGetLastError();
  if (err != cudaSuccess) printf("%s\n", cudaGetErrorString(err)); 

  // retrieve result
fprintf(stderr, "retrieving result...\n");
  ftw_EnergyArray512 *h_repulsion = (ftw_EnergyArray512 *)malloc(sizeof(ftw_EnergyArray512));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy(h_repulsion, d_repulsion, sizeof(ftw_EnergyArray512), cudaMemcpyDeviceToHost ));

  // free device memory
  cudaFree(d_configuration);
  cudaFree(d_repulsion);

  free(h_configuration); // free host memory for replicated configuration

  return h_repulsion;
}

//  This routine to be called from outside the library
extern "C" ftw_EnergyArray256 *GFGToRepulsion256(ftw_GFG65536 *gfg, float sigma, float epsilon)
{
  ftw_EnergyArray256 	*d_repulsion;
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
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_repulsion, sizeof(ftw_EnergyArray256)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_configuration, sizeof(ftw_GFG65536)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy( d_configuration, h_configuration, sizeof(ftw_GFG65536), cudaMemcpyHostToDevice ));

  dim3 dimGrid(256, 256);
  dim3 dimBlock(256, 1, 1);

  EnergyKernel256<<< dimGrid, dimBlock >>>(d_configuration, NULL, d_repulsion, NULL);
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

// This is for a traditional LJ 6-12 interaction.  Note that sigma is the value where energy is zero, not the well-bottom as for COMPASS...
// This operates on 'chunks' in x-direction because the domain is too large for the GPU memory
__global__ void EnergyKernel1024_612(	ftw_GFG65536 *d_configuration, 
 					ftw_Chunk *d_attraction, 
 					ftw_Chunk *d_repulsion, 
 					ftw_Chunk *d_total, 
 					int chunk, int chunk_size) {
  unsigned int idx = threadIdx.x;
  unsigned int idy = blockIdx.x;
  unsigned int idz = blockIdx.y;

  float repulsion=0;
  float attraction=0;
  float sigma_over_r_sq;
  float dx, dy, dz, dd;
  float f_resolution_x = d_configuration->box_x / 1024;
  float f_resolution_y = d_configuration->box_y / 1024;
  float f_resolution_z = d_configuration->box_z / 1024;

  float cuda_x = (chunk * chunk_size + idx ) * f_resolution_x;
  float cuda_y = idy * f_resolution_y;
  float cuda_z = idz * f_resolution_z;

  // evaluate energy at (cuda_x, cuda_y, cuda_z);
  for (int i=0; i< d_configuration->n_atoms; i++) {
    // central atom
    dx = d_configuration->atom[i].x - cuda_x;
    dy = d_configuration->atom[i].y - cuda_y;
    dz = d_configuration->atom[i].z - cuda_z;
    dd = dx*dx + dy*dy + dz*dz;
    sigma_over_r_sq = d_configuration->atom[i].sigma * d_configuration->atom[i].sigma / dd;
    repulsion  += d_configuration->atom[i].epsilon * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq;
    attraction += d_configuration->atom[i].epsilon * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq;
  } 

  // If NULL pointers are passed, then no values are recorded.
  if (d_attraction) d_attraction->energy[idx][idy][idz] = 4 * attraction;
  if (d_repulsion) d_repulsion->energy[idx][idy][idz] = 4 * repulsion;
  if (d_total) d_total->energy[idx][idy][idz] = 4 * repulsion - 4 * attraction;
}

// This routine to be called from outside the library
extern "C" ftw_EnergyArray1024 *GFGToEnergyArray1024_612(ftw_GFG65536 *gfg, float sigma, float epsilon)
{
  ftw_Chunk 		*d_energy_array_chunk;
  ftw_GFG65536 		*d_configuration;

  ftw_GFG65536 *h_configuration = replicateGFG65536(gfg); // replicate the gfg 
  for (int n=0; n<gfg->n_atoms; n++) // and cross-parameterize 
  {
    h_configuration->atom[n].sigma = pow(0.5f * (float)(pow(sigma, 6) + pow(h_configuration->atom[n].sigma, 6)), 0.1666666f);
    h_configuration->atom[n].epsilon = sqrt(epsilon * h_configuration->atom[n].epsilon);
  }

  // then do the calc
  // (x,y,z) is (blockx, gridx, gridy)... chunking to 4 parts in x, then will combine results
  int chunk_size = 256, chunks = 4;
  dim3 dimGrid(1024, 1024);
  dim3 dimBlock(chunk_size, 1, 1);

  cudaError_t err;
  /* allocate for energy array and configuration on device */
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_energy_array_chunk, sizeof(ftw_Chunk)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_configuration, sizeof(ftw_GFG65536)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy( d_configuration, h_configuration, sizeof(ftw_GFG65536), cudaMemcpyHostToDevice ));

  ftw_EnergyArray1024 *h_energy_array = (ftw_EnergyArray1024 *)malloc(sizeof(ftw_EnergyArray1024)); // host structure, for result

  for (int chunk=0; chunk < chunks; chunk++)
  {
    EnergyKernel1024_612<<< dimGrid, dimBlock >>>(d_configuration, NULL, NULL, d_energy_array_chunk, chunk, chunk_size);
    cudaThreadSynchronize(); // block until the device has completed
    err = cudaGetLastError();
    if (err != cudaSuccess) {printf("CUDA error:  %s\n", cudaGetErrorString(err)); exit(1);}
    // retrieve result
    ftw_EnergyArray1024* h_address = (ftw_EnergyArray1024*)((long)h_energy_array + (long)(sizeof(ftw_Chunk) * chunk));
    for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy(h_address, d_energy_array_chunk, sizeof(ftw_Chunk), cudaMemcpyDeviceToHost ));
  }

  // free device memory
  cudaFree(d_configuration);
  cudaFree(d_energy_array_chunk);

  free(h_configuration); // free host memory for replicated configuration
  return h_energy_array;
}

// This routine to be called from outside the library
extern "C" ftw_EnergyArray1024 *GFGToRepulsion1024_612(ftw_GFG65536 *gfg, float sigma, float epsilon)
{
  ftw_Chunk 		*d_repulsion_chunk;
  ftw_GFG65536 		*d_configuration;

  ftw_GFG65536 *h_configuration = replicateGFG65536(gfg); // replicate the gfg 
  for (int n=0; n<gfg->n_atoms; n++) // and cross-parameterize 
  {
    h_configuration->atom[n].sigma = pow(0.5f * (float)(pow(sigma, 6) + pow(h_configuration->atom[n].sigma, 6)), 0.1666666f);
    h_configuration->atom[n].epsilon = sqrt(epsilon * h_configuration->atom[n].epsilon);
  }

  // then do the calc
  // (x,y,z) is (blockx, gridx, gridy)... chunking to 4 parts in x, then will combine results
  int chunk_size = 256, chunks = 4;
  dim3 dimGrid(1024, 1024);
  dim3 dimBlock(chunk_size, 1, 1);

  cudaError_t err;
  /* allocate for energy array and configuration on device */
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_repulsion_chunk, sizeof(ftw_Chunk)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMalloc( (void **) &d_configuration, sizeof(ftw_GFG65536)));
  for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy( d_configuration, h_configuration, sizeof(ftw_GFG65536), cudaMemcpyHostToDevice ));

  ftw_EnergyArray1024 *h_repulsion = (ftw_EnergyArray1024 *)malloc(sizeof(ftw_EnergyArray1024)); // host structure, for result

  for (int chunk=0; chunk < chunks; chunk++)
  {
    EnergyKernel1024_612<<< dimGrid, dimBlock >>>(d_configuration, NULL, d_repulsion_chunk, NULL, chunk, chunk_size);
    cudaThreadSynchronize(); // block until the device has completed
    err = cudaGetLastError();
    if (err != cudaSuccess) {printf("CUDA error:  %s\n", cudaGetErrorString(err)); exit(1);}
    // retrieve result
    ftw_EnergyArray1024* h_address = (ftw_EnergyArray1024*)((long)h_repulsion + (long)(sizeof(ftw_Chunk) * chunk));
    for(err = cudaErrorUnknown; err != cudaSuccess; err = cudaMemcpy(h_address, d_repulsion_chunk, sizeof(ftw_Chunk), cudaMemcpyDeviceToHost ));
  }

  // free device memory
  cudaFree(d_configuration);
  cudaFree(d_repulsion_chunk);

  free(h_configuration); // free host memory for replicated configuration
  return h_repulsion;
}


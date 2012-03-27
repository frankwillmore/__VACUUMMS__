/* test.c */

#include <cuda.h>

#include <ftw_config_parser.h>
#include <ftw_param.h>
#include <ftw_types.h>
#include <ftw_gfg2fvi.h>
#include <stdio.h>
#include <math.h>

main(int argc, char *argv[]) 
{
  int i,j,k;
  double box_x=10, box_y=10, box_z=10;
  int potential = 69;
  int resolution = 256;
  int device_count;
  float temperature = 298.0;
  float sigma=0.0;
  float epsilon=1.0;

  setCommandLineParameters(argc, argv);

  if (getFlagParam("-usage"))
  {
    printf("usage:     test        -box [10.0 10.0 10.0]\n");
    printf("                          -potential [69] \n");
    printf("                          -resolution [256] \n");
    printf("                          -temperature [298.0] \n");
    printf("                          -sigma [0.0] \n");
    printf("                          -epsilon [1.0] \n");
    printf("                          -check_device \n");
    exit(0);
  }

  if (getFlagParam("-check_device"))
  {
    printf("Checking for devices...\n");
    cudaGetDeviceCount(&device_count);
    printf("Found %d device(s).\n", device_count);
    exit(0);
  }

  cudaGetDeviceCount(&device_count);
  if (device_count == 0)
  {
    printf("No device found.\n");
    exit(1);
  }

//  getVectorParam("-box", &box_x, &box_y, &box_z);
//  getIntParam("-potential", &potential);
//  getIntParam("-resolution", &resolution);
//  getFloatParam("-sigma", &sigma);
//  getFloatParam("-epsilon", &epsilon);

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

cudaError_t err = cudaErrorUnknown;
//fprintf(stderr, "p_int in c = %p\n", p_int);
//err=cudaMalloc*((void **) &p_int, sizeof(int));
//fprintf(stderr, "p_int in c = %p\n", p_int);

  ftw_EnergyArray256 *ea;
fprintf(stderr, "ea = %p\n", ea);
//point from integer without cast???  what is this returning???
  ea = MALGFGToRepulsion256_612(gfg, sigma, epsilon);
fprintf(stderr, "ea = %p\n", ea);

  for (i=0; i<resolution; i++) for (j=0; j<resolution; j++) for (k=0; k<resolution; k++)
    printf("%f\t%f\t%f\t%f\n", i*box_x / resolution, j*box_y / resolution, k*box_z / resolution, exp(ea->energy[i][j][k]/(-temperature))); 
}


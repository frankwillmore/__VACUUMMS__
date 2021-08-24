// gfg2fvi.c

#include <ftw_config_parser.h>
#include <ftw_param.h>
#include <ftw_types.h>
#include <ftw_gfg2fvi.h>

#include <stdio.h>
#include <math.h>

float attenuator = 1.0;

main(int argc, char *argv[]) 
{
  int i,j,k;
  double box_x=10, box_y=10, box_z=10;
  int potential = 69;
  int resolution = 256;
  int device_count;
  float temperature = 298.0;
  float attenuator = 1.0;
  float preexponential = 1.0;
  float sigma=0.0;
  float epsilon=1.0;

  setCommandLineParameters(argc, argv);

  if (getFlagParam("-usage"))
  {
    printf("usage:     gfg2fvi        -box [10.0 10.0 10.0]\n");
    printf("                          -potential [69] \n");
    printf("                          -resolution [256] \n");
    printf("                          -temperature [298.0] \n");
    printf("                          -attenuator [1.0] \n");
    printf("                          -preexponential [1.0] \n");
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

//  For SOME reason, getting the device count causes the device to become busy and never recover.  So it's commented out.
//j  cudaGetDeviceCount(&device_count);
//  if (device_count == 0)
//  {
//    printf("No device found.\n");
//    exit(1);
//  }

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getIntParam("-potential", &potential);
  getIntParam("-resolution", &resolution);
  getFloatParam("-attenuator", &attenuator);
  getFloatParam("-preexponential", &preexponential);
  getFloatParam("-sigma", &sigma);
  getFloatParam("-epsilon", &epsilon);



  fprintf(stderr,"reading configuration\n");

  ftw_GFG65536 *gfg = readGFG65536(stdin);
  gfg->box_x = box_x;
  gfg->box_y = box_y;
  gfg->box_z = box_z;

  fprintf(stderr, "calculating resolution = %d for %d potential\n", resolution, potential);
  fprintf(stderr, "using sigma = %f and epsilon = %f \n", sigma, epsilon);

  // cases
  if ((resolution == 256) && (potential == 69)){
    // 69 means we're using COMPASS forcefield, so we need to do this stuff...
    // first have to change from kcal/mol to K (divide by kB, or multiply by 503.25)
    for (i=0; i < gfg->n_atoms; i++) gfg->atom[i].epsilon *= 503.25;
    if (!getFlagParam("-epsilon")) epsilon = 1.0 / temperature;

    ftw_EnergyArray256 *ea = GFGToRepulsion256(gfg, sigma, epsilon);
    for (i=0; i<resolution; i++) for (j=0; j<resolution; j++) for (k=0; k<resolution; k++)
      printf("%f\t%f\t%f\t%f\n", i*box_x / resolution, j*box_y / resolution, k*box_z / resolution, 
                                 preexponential * exp(ea->energy[i][j][k]/(-temperature * attenuator))); 
  }
  else if ((resolution == 256) && (potential == 612)){
    ftw_EnergyArray256 *ea = GFGToRepulsion256_612(gfg, sigma, epsilon);
    for (i=0; i<resolution; i++) for (j=0; j<resolution; j++) for (k=0; k<resolution; k++)
      printf("%f\t%f\t%f\t%f\n", i*box_x / resolution, j*box_y / resolution, k*box_z / resolution, 
                                 preexponential * exp(ea->energy[i][j][k]/(-temperature * attenuator))); 
  }
  else if ((resolution == 512) && (potential == 612)){
    ftw_EnergyArray512 *ea = GFGToRepulsion512_612(gfg, sigma, epsilon);
    for (i=0; i<resolution; i++) for (j=0; j<resolution; j++) for (k=0; k<resolution; k++)
      printf("%f\t%f\t%f\t%f\n", i*box_x / resolution, j*box_y / resolution, k*box_z / resolution, 
                                 preexponential * exp(ea->energy[i][j][k]/(-temperature * attenuator))); 
  }
  else if ((resolution == 1024) && (potential == 612)){
    ftw_EnergyArray1024 *ea = GFGToRepulsion1024_612(gfg, sigma, epsilon);
    for (i=0; i<resolution; i++) for (j=0; j<resolution; j++) for (k=0; k<resolution; k++)
      printf("%f\t%f\t%f\t%f\n", i*box_x / resolution, j*box_y / resolution, k*box_z / resolution, 
                                 preexponential * exp(ea->energy[i][j][k]/(-temperature * attenuator))); 
  }
  else {
    printf("unknown resolution/potential combination %d/%d\n", resolution, potential);
    exit(1);
  }
}


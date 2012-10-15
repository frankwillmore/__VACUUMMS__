// sgfg2fvi.c -- serial version

#include <ftw_config_parser.h>
#include <ftw_param.h>
#include <ftw_types.h>
#include <stdio.h>
#include <math.h>

main(int argc, char *argv[]) 
{
  int i,j,k;
  double box_x=10, box_y=10, box_z=10;
  int potential = 69;
  float resolution = 0.1;
  float temperature = 1.0;
  float preexponential = 1.0;
  float sigma=0.0;
  float epsilon=1.0;

  setCommandLineParameters(argc, argv);

  if (getFlagParam("-usage"))
  {
    printf("usage:     sgfg2fvi       -box [10.0 10.0 10.0]\n");
    printf("                          -potential [612] \n");
    printf("                          -resolution [0.1] \n");
    printf("                          -temperature [1.0] \n");
    printf("                          -preexponential [1.0] \n");
    printf("                          -sigma [0.0] \n");
    printf("                          -epsilon [1.0] \n");
    exit(0);
  }

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getIntParam("-potential", &potential);
  getFloatParam("-resolution", &resolution);
  getFloatParam("-preexponential", &preexponential);
  getFloatParam("-sigma", &sigma);
  getFloatParam("-epsilon", &epsilon);

  // read configuration
  ftw_GFG65536 *gfg = readGFG65536(stdin);
  int n=0;
  // Berthelot combining rule...
  for (n=0; n<gfg->n_atoms; n++) {
    gfg->atom[n].sigma = 0.5f * (sigma + gfg->atom[n].sigma);
//    gfg->atom[n].epsilon = 503.25 * sqrt(epsilon * gfg->atom[n].epsilon);
    gfg->atom[n].epsilon = sqrt(epsilon * gfg->atom[n].epsilon);
  }
  gfg->box_x = box_x;
  gfg->box_y = box_y;
  gfg->box_z = box_z;

  // replicate, then free the original
  ftw_GFG65536 *configuration = replicateGFG65536(gfg);  
  free(gfg);

  float test_x, test_y, test_z;
  for (test_x =0.0; test_x < box_x; test_x += resolution) for (test_y =0.0; test_y < box_y; test_y += resolution) for (test_z =0.0; test_z < box_z; test_z += resolution){

    //  float repulsion=0;
    float attraction = 0.0;
    float repulsion  = 0.0;
    float total = 0.0;
    float alpha, alpha6;
    float dx, dy, dz, d, dd;

    int i=0;
    for (i=0; i < configuration->n_atoms; i++) {
      dx = configuration->atom[i].x - test_x;
      dy = configuration->atom[i].y - test_y;
      dz = configuration->atom[i].z - test_z;
      dd = dx*dx + dy*dy + dz*dz; 
      alpha = configuration->atom[i].sigma * configuration->atom[i].sigma / dd;
      alpha6 = alpha * alpha * alpha;
      attraction += alpha6;
      repulsion += alpha6 * alpha6;
      //  For 6-9 potential
      //      d = sqrt(dd);
      //      alpha = pow(configuration->atom[i].sigma, 3) / (d * dd);   
      //      repulsion += configuration->atom[i].epsilon * alpha * alpha * alpha;
      //      attraction += configuration->atom[i].epsilon * alpha * alpha;
    } 

    // also for 6-9...
    // repulsion *= 2;
    // attraction *= 3;
    // total = repulsion - attraction;
    total = 4 * (repulsion - attraction);

    printf("%f\t%f\t%f\t%f\n", test_x, test_y, test_z, preexponential * exp( - repulsion / temperature ) ); 
  }
}


// example.c

#include <ftw_config_parser.h>
#include <ftw_param.h>
#include <ftw_types.h>
#include <ftw_gfg2fvi.h>

#include <stdio.h>
#include <math.h>

main(int argc, char *argv[]) 
{
  int i,j,k;
//  int resolution = 256;
  int resolution = 1024;
  double box_x=10, box_y=10, box_z=10;
  double sigma=1, epsilon=1;

  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);

  printf("reading config\n");

  ftw_GFG65536 *gfg = readGFG65536(stdin);
  gfg->box_x = box_x;
  gfg->box_y = box_y;
  gfg->box_z = box_z;


  printf("generating energy array\n");
//  ftw_EnergyArray256 *ea = GFGToEnergyArray256(gfg, sigma, epsilon);
//  for (i=0; i<resolution; i++) for (j=0; j<resolution; j++) for (k=0; k<resolution; k++)
//    printf("%f\t%f\t%f\t%f\n", i*box_x / resolution, j*box_y / resolution, k*box_z / resolution, exp(ea->energy[i][j][k]/(0-298000))); 
//  printf("%f\t%f\t%f\t%f\n", i*box_dimension / resolution, j*box_dimension / resolution, k*box_dimension / resolution, exp(energy_array.energy[i][j][k]/(0-298000))); 

  ftw_EnergyArray1024 *ea = GFGToEnergyArray1024_612(gfg, sigma, epsilon);
  printf("writing trace\n");
  for (i=0; i<resolution; i++) printf("%d\t%f\n", i, ea->energy[i][i][i]);

  ftw_EnergyArray1024 *rea = GFGToRepulsion1024_612(gfg, sigma, epsilon);
  printf("writing trace\n");
  for (i=0; i<resolution; i++) printf("%d\t%f\n", i, rea->energy[i][i][i]);
}


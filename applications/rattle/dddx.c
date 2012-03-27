/* dddx.c Discrete version of ddx */

#include <stdio.h>
#include <math.h>
#include <pthread.h>
#include <semaphore.h>
#include <assert.h>

#include "dddx.h"

#include <ftw_std.h>
#include <ftw_param.h>
#include <ftw_config_parser.h>
#include <ftw_types.h>
#include <ftw_gfg2fvi.h>

double box_x=6, box_y=6, box_z=6;
float min_diameter;
FILE *instream;

dddCavity cavities[100000];

main(int argc, char *argv[]) 
{
  int i,j,k;
  int resolution, resolution_mask;
  float sigma=1, epsilon=1;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("\nusage:\t\t-box [ 6.0 6.0 6.0 ]\n");
    printf("\t\t-min_diameter [ 0.0 ]\n");
    printf("\t\t-resolution_1024 /* 256 is default */\n");
    printf("\n");
    exit(0);
  }
  getFloatParam("-min_diameter", &min_diameter);

  ftw_GFG65536 *gfg = readGFG65536(stdin);
  if(getFlagParam("-box")) getVectorParam("-box", &box_x, &box_y, &box_z);
  gfg->box_x = box_x; gfg->box_y = box_y; gfg->box_z = box_z;

  int cav, n_cavs=0;
  // search the array for minima
  if (getFlagParam("-resolution_1024")) {
    resolution = 1024;  resolution_mask = 1023;
    ftw_EnergyArray1024 *repulsion = GFGToRepulsion1024_612(gfg, sigma, epsilon);

    for (i=0; i<resolution; i++) for (j=0; j<resolution; j++) for (k=0; k<resolution; k++) {
      // look up, down, left, right, front, back...
      if (  (repulsion->energy[ (i - 1) & resolution_mask ][j][k] <= repulsion->energy[i][j][k]) 
         || (repulsion->energy[i][ (j - 1) & resolution_mask ][k] <= repulsion->energy[i][j][k]) 
         || (repulsion->energy[i][j][ (k - 1) & resolution_mask ] <= repulsion->energy[i][j][k]) 
         || (repulsion->energy[ (i + 1) & resolution_mask ][j][k] <= repulsion->energy[i][j][k]) 
         || (repulsion->energy[i][ (j + 1) & resolution_mask ][k] <= repulsion->energy[i][j][k]) 
         || (repulsion->energy[i][j][ (k + 1) & resolution_mask ] <= repulsion->energy[i][j][k]) 
      ) continue;

      cavities[n_cavs].x = i * box_x / resolution;
      cavities[n_cavs].y = j * box_y / resolution;
      cavities[n_cavs].z = k * box_z / resolution;
      cavities[n_cavs].gfg = gfg;

      expandTestParticle((void*)&cavities[n_cavs]);
      if (cavities[n_cavs].diameter > min_diameter) n_cavs++;

    } // end for
  } // end if

  else { // resolution 256
    resolution = 256;  resolution_mask = 255;
    ftw_EnergyArray256 *repulsion = GFGToRepulsion256(gfg, sigma, epsilon);

    for (i=0; i<resolution; i++) for (j=0; j<resolution; j++) for (k=0; k<resolution; k++) {
      // look up, down, left, right, front, back...
      if (  (repulsion->energy[ (i - 1) & resolution_mask ][j][k] <= repulsion->energy[i][j][k]) 
         || (repulsion->energy[i][ (j - 1) & resolution_mask ][k] <= repulsion->energy[i][j][k]) 
         || (repulsion->energy[i][j][ (k - 1) & resolution_mask ] <= repulsion->energy[i][j][k]) 
         || (repulsion->energy[ (i + 1) & resolution_mask ][j][k] <= repulsion->energy[i][j][k]) 
         || (repulsion->energy[i][ (j + 1) & resolution_mask ][k] <= repulsion->energy[i][j][k]) 
         || (repulsion->energy[i][j][ (k + 1) & resolution_mask ] <= repulsion->energy[i][j][k]) 
      ) continue;

      cavities[n_cavs].x = i * box_x / resolution;
      cavities[n_cavs].y = j * box_y / resolution;
      cavities[n_cavs].z = k * box_z / resolution;
      cavities[n_cavs].gfg = gfg;

      expandTestParticle((void*)&cavities[n_cavs]);
      if (cavities[n_cavs].diameter > min_diameter) n_cavs++;

    } // end for
  } // end if-else

  for (cav=0; cav<n_cavs; cav++) printf("%f\t%f\t%f\t%f\n", cavities[cav].x, cavities[cav].y, cavities[cav].z, cavities[cav].diameter);

} // end main()

float calculateEnergy(dddCavity *cavity, float diameter)
{
  float repulsion=0;
  float attraction=0;
  float dx, dy, dz, dd, d6, d12;
  float sigma, sigma6, sigma12;
  int i;

  for (i=0; i<cavity->gfg->n_atoms; i++)
  {
    dx = cavity->gfg->atom[i].x - cavity->x;
    dy = cavity->gfg->atom[i].y - cavity->y;
    dz = cavity->gfg->atom[i].z - cavity->z;
    dd = dx*dx + dy*dy + dz*dz;
    d6 = dd*dd*dd;
    d12 = d6*d6;

    sigma = 0.5 * (cavity->gfg->atom[i].sigma + diameter);
    sigma6 = sigma*sigma*sigma*sigma*sigma*sigma;
    sigma12 = sigma6*sigma6;

    repulsion += sigma12/d12;
    attraction += sigma6/d6;
  }

  return 4.0 * (repulsion - attraction);
}

// Thread function
void *expandTestParticle(void *passval)
{
  dddCavity *cavity = (dddCavity*)passval;
  float slope;
  float step_size;
  float energy, old_energy;
  float e0, e1, r0, r1;
  float diameter = 0.0f;
  old_energy = calculateEnergy(cavity, diameter);
  // We're looking for a zero energy point.  If it's already +, it won't converge, so we eliminate...
  if (old_energy > 0)
  {
    cavity->diameter = -1;
    return;
  }

  while (diameter += .1)
  {
    energy = calculateEnergy(cavity, diameter);
    if (energy > old_energy) break;
    old_energy = energy;
  }

  // Newton's method

  while(1)
  {
    r0 = diameter - .001;
    r1 = diameter + .001;
    
    e0 = calculateEnergy(cavity, r0);
    e1 = calculateEnergy(cavity, r1);
    energy = calculateEnergy(cavity, diameter);

    slope = (e1-e0)/(r1-r0);
    step_size = -energy/slope;

    diameter = diameter + step_size;

    if (step_size*step_size < .00000001) break;
  }
  cavity->diameter = diameter;
}


/* csa.c */

#define MAX_CAVITIES 1310720
#define N_SAMPLES 10000

#define PI 3.14159265358979323846264

#include <ftw_std.h>
#include <ftw_rng2.h>
#include <ftw_param.h>
#include "csa.h"

// Reads a centered cluster and determines the surface area of it.
// Will deliver an erroneous result if cluster is not centered or percolates.

// More generally, it will give the surface area of all the cavities, whether clustered or not.

// In:  one cluster; 1 or more records in .cav format 
// Out: .dst (reports one value of surface area) 

FILE *instream;

int number_of_cavities=0;
double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], d[MAX_CAVITIES];

double sorface_area[MAX_CAVITIES];

double sample_x, sample_y, sample_z, sample_phi, sample_theta;
double total_surface_area=0;

int main(int argc, char* argv[])
{
  double surface_area;
  int successes=0, attempts=0;
  int particle_number;

  instream=stdin;
  setCommandLineParameters(argc, argv);
  if (getFlagParam("-randomize")) initializeRandomNumberGenerator2(-1);
  else initializeRandomNumberGenerator2(0);
  if (getFlagParam("-usage")){
    printf(" Reads a centered cluster and determines the surface area of it.\n");
    printf(" Will deliver an erroneous result if cluster is not centered or percolates.\n");
    printf("\n");
    printf(" More generally, it will give the surface area of all the cavities, whether clustered or not.\n");
    printf("\n");
    printf(" In:  one cluster; 1 or more records in .cav format \n");
    printf(" Out: .dst (reports one value of surface area) \n");
    exit(0);
  }

  readInputStream();

  // figure out total surface area for sampling
  for (particle_number=0; particle_number<number_of_cavities; particle_number++)
  { 
    sorface_area[particle_number] = PI * d[particle_number] * d[particle_number];
    total_surface_area += sorface_area[particle_number];
  }

  for (attempts=0; attempts<N_SAMPLES; attempts++)
  {
    // pick a particle, then pick a point on its surface
    while(1)
    {
      particle_number = rnd2() * number_of_cavities;
      if (rnd2() < (sorface_area[particle_number]/total_surface_area)) break;
    }

    generateSamplePoint(particle_number);

    successes += checkExclusion();
  }

  surface_area = (total_surface_area * successes)/attempts;

  printf("%lf\n", surface_area);
  return 0;
}

void generateSamplePoint(int particle_number)
{
  double sin_theta, cos_theta, sin_phi, cos_phi;
  double radius;
  int not_okay=1;

  sample_theta = 0;

  while (not_okay)
  {
    while (rnd2() >= (sin_theta = sin(sample_theta))) sample_theta = rnd2() * PI;
  
    sample_phi = rnd2() * 2 * PI;
    cos_theta = cos(sample_theta);
    sin_phi = sin(sample_phi);  
    cos_phi = cos(sample_phi);

    radius = d[particle_number] * .5;
    sample_x = x[particle_number] + radius * sin_theta * cos_phi;
    sample_y = y[particle_number] + radius * sin_theta * sin_phi;
    sample_z = z[particle_number] + radius * cos_theta;

    not_okay=0;
//    if (sample_x<0 || sample_x>box_x) not_okay=1;
//    if (sample_y<0 || sample_y>box_y) not_okay=1;
//    if (sample_z<0 || sample_z>box_z) not_okay=1;
  }
}

// returns 1 if point lies within radius of any particle, 0 if it does not.
int checkExclusion()
{
  int i;
  double dx, dy, dz, dd;

  for (i=0; i<number_of_cavities; i++)
  {
    dx = x[i] - sample_x;
    dy = y[i] - sample_y;
    dz = z[i] - sample_z;

    dd = dx*dx + dy*dy + dz*dz;

    //if ( 4*dd +.000000000000000000001 <= (d[i] * d[i])) 
    if ( 4*dd + .000000001 <= (d[i] * d[i])) 
    {
      return 0;
    }
  }
 
  return 1;
}

void readInputStream()
{
  char line[80];
  char *xs, *ys, *zs, *ds;

  while (TRUE)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    x[number_of_cavities] = strtod(xs, NULL);
    y[number_of_cavities] = strtod(ys, NULL);
    z[number_of_cavities] = strtod(zs, NULL);
    d[number_of_cavities] = strtod(ds, NULL);

    number_of_cavities++;

    if (number_of_cavities > MAX_CAVITIES)
    {
      printf("Too many cavities.");
      exit(0);
    }
  }
  
  V printf("%d cavities.\n", number_of_cavities);
}


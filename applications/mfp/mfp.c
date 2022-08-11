/* mfp.c */

#include "io_setup.h"
#include "mfp.h"

#include <stdlib.h>
#include <math.h>
#include <ftw_std.h>
#include <ftw_param.h>
#include <ftw_rng.h>

#define MAX_NUM_MOLECULES 16384
#define MAX_CLOSE 2048
#define PI 3.141592653589796264

/* NOTE:  sigma is specified in Angstroms, epsilon in K, T in K */

double x[MAX_NUM_MOLECULES];
double y[MAX_NUM_MOLECULES];
double z[MAX_NUM_MOLECULES];
double sigma[MAX_NUM_MOLECULES];
double epsilon[MAX_NUM_MOLECULES];

double close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE];
double close_sigma[MAX_CLOSE];
double close_epsilon[MAX_CLOSE];
double close_sigma6[MAX_CLOSE];
double close_sigma12[MAX_CLOSE];

double box_x=6, box_y=6, box_z=6;
double verlet_cutoff=100.0;

double step_size = 0.001;

double T=347;
double energy, new_energy;

int number_of_samples = 1;

double test_x0, test_y0, test_z0;
double test_x, test_y, test_z;
double verlet_center_x, verlet_center_y, verlet_center_z;
double test_diameter = 1.0;
double test_epsilon = 1.0;
int seed = 123450;
int successes;

int number_of_molecules = 0;
int close_molecules;
double drift_x, drift_y, drift_z;
const double rand_step = 1.0/RAND_MAX;

FILE *instream;

int verbose;

int main(int argc, char *argv[])
{
  double sq_distance_from_initial_pt;
  double step_number;
  double dx, dy, dz;
  double phi, theta;
  double x_step, y_step, z_step;


  setCommandLineParameters(argc, argv);
  verbose = getFlagParam("-verbose");
  getIntParam("-seed", &seed);
  if (getFlagParam("-randomize")) randomize();
  else initializeRandomNumberGeneratorTo(seed);
  if (getFlagParam("-usage")) 
  {
    printf("\nusage:  configuration in, list of path lengths out\n\n");
    printf("        -box [ 6.0 6.0 6.0 ]\n");
    printf("        -test_diameter [ 1.0 ]\n");
    printf("        -test_epsilon [ 1.0 ]\n");
    printf("        -T [ 343.0 ]\n");
    printf("        -n [ 1 ]\n");
    printf("        -step_size [ .001 ]\n");
    printf("        -verlet_cutoff [ 100.0 ]\n");
    printf("        -seed [ 123450 ]\n");
    printf("        -randomize\n\n");

    exit(0);
  }

  srand(seed);

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-step_size", &step_size);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getDoubleParam("-test_diameter", &test_diameter);
  getDoubleParam("-test_epsilon", &test_epsilon);
  getDoubleParam("-T", &T);
  getIntParam("-n", &number_of_samples);

  readConfiguration();
  
  while (number_of_samples-- > 0)
  {
    generateTestPoint();
    drift_x=drift_y=drift_z=0;

    V printf("test point:  %lf, %lf, %lf\n", test_x, test_y, test_z);
    
    // pick a direction
    phi = 2*PI*rand()*rand_step;
    theta = PI*rand()*rand_step;
    x_step = step_size*cos(phi)*sin(theta); 
    y_step = step_size*sin(phi)*sin(theta);
    z_step = step_size*cos(theta);

    // extend ray from test point
    while (1)
    {
      test_x += x_step;
      test_y += y_step;
      test_z += z_step;

      dx=test_x - verlet_center_x;
      dy=test_y - verlet_center_y;
      dz=test_z - verlet_center_z;
      if (dx*dx + dy*dy + dz*dz > .01 * verlet_cutoff) makeVerletList();
      V printf("%lf, %lf, %lf:\t%lf\n", test_x, test_y, test_z, calculateEnergy());

      if (calculateEnergy() > (T * 1.5)) break;
    }
  
    dx = test_x - test_x0 + drift_x;
    dy = test_y - test_y0 + drift_y;
    dz = test_z - test_z0 + drift_z;

    printf("%lf\n", 2*sqrt(dx*dx + dy*dy + dz*dz));
  }

  return 0;
}

void generateTestPoint()
{
  test_x = test_x0 = rand() * rand_step * box_x;
  test_y = test_y0 = rand() * rand_step * box_y;
  test_z = test_z0 = rand() * rand_step * box_z;

  makeVerletList();

  while (rand() * rand_step > exp(-(calculateEnergy()/T))) 
  {
    test_x = test_x0 = rand() * rand_step * box_x;
    test_y = test_y0 = rand() * rand_step * box_y;
    test_z = test_z0 = rand() * rand_step * box_z;

    makeVerletList();
  }
}

void makeVerletList()
{
  int i;
  double dx, dy, dz, dd;
  double shift_x, shift_y, shift_z;

  while (test_x > box_x)
  {
    test_x -= box_x;
    drift_x += box_x;
  }
  while (test_y > box_y) 
  {
    test_y -= box_y;
    drift_y += box_y;
  }
  while (test_z > box_z) 
  {
    test_z -= box_z;
    drift_z += box_z;
  }

  while (test_x < 0)
  {
    test_x += box_x;
    drift_x -= box_x;
  }
  while (test_y < 0)
  {
    test_y += box_y;
    drift_y -= box_y;
  }
  while (test_z < 0)
  {
    test_z += box_z;
    drift_z -= box_z;
  }

  verlet_center_x=test_x;
  verlet_center_y=test_y;
  verlet_center_z=test_z;

  close_molecules=0;

  for (i=0; i<number_of_molecules; i++)
  {
    for (shift_x = -box_x; shift_x <= box_x; shift_x += box_x)
    for (shift_y = -box_y; shift_y <= box_y; shift_y += box_y)
    for (shift_z = -box_z; shift_z <= box_z; shift_z += box_z)
    {
      dx = shift_x + x[i] - test_x;
      dy = shift_y + y[i] - test_y;
      dz = shift_z + z[i] - test_z;

      dd = dx*dx + dy*dy + dz*dz;

      if (dd < verlet_cutoff) 
      { 
        close_x[close_molecules] = shift_x + x[i];
        close_y[close_molecules] = shift_y + y[i];
        close_z[close_molecules] = shift_z + z[i];
        close_sigma[close_molecules] = sigma[i];
        close_epsilon[close_molecules] = epsilon[i];
        close_sigma6[close_molecules] = sigma[i]*sigma[i]*sigma[i]*sigma[i]*sigma[i]*sigma[i];
        close_sigma12[close_molecules] = close_sigma6[close_molecules]*close_sigma6[close_molecules];

        close_molecules++;
      }
    }
  }
}

// returns the energy change by insertion of a test particle at test_x, test_y, test_z
double calculateEnergy()
{
  double repulsion=0;
  double attraction=0;
  double dx, dy, dz, dd, d6, d12;
  double sigma, sigma6, sigma12;
  double epsilon;
  int i;
  double retval;


  for (i=0; i<close_molecules; i++)
  {
    dx = close_x[i] - test_x;
    dy = close_y[i] - test_y;
    dz = close_z[i] - test_z;
    dd = dx*dx + dy*dy + dz*dz;
    d6 = dd*dd*dd;
    d12 = d6*d6;

    sigma = 0.5 * (close_sigma[i] + test_diameter);
    epsilon = sqrt(close_epsilon[i] * test_epsilon);
    sigma6 = sigma*sigma*sigma*sigma*sigma*sigma;
    sigma12 = sigma6*sigma6;

    repulsion += epsilon * sigma12/d12;
    attraction += epsilon * sigma6/d6;
  }

  retval = 4.0 * (repulsion - attraction);
  return retval;
}

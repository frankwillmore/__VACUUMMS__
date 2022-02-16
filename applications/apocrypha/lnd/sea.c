/* sea.c */

// sea level is in K 

#include "lnd.h"

#include <ftw_std.h>
#include <ftw_rng.h>
#include <ftw_param.h>

#define MAX_NUM_MOLECULES 16384
#define MAX_CLOSE 16384

double x[MAX_NUM_MOLECULES];
double y[MAX_NUM_MOLECULES];
double z[MAX_NUM_MOLECULES];
double sigma[MAX_NUM_MOLECULES];
double epsilon[MAX_NUM_MOLECULES];

double Xsigma[MAX_NUM_MOLECULES];
double Xepsilon[MAX_NUM_MOLECULES];

/*
double close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE];
double close_sigma[MAX_CLOSE];
double close_sigma6[MAX_CLOSE];
double close_sigma12[MAX_CLOSE];
*/

double box_x=10, box_y=10, box_z=10;

int number_of_samples = 100;

double x_increment, y_increment, z_increment;

double test_x, test_y, test_z;

// NOTE:  sea_level, test_epsilon, and epsilon vals in file all have energy units.  
//        It doesn't matter what the units are, but they need to be the same in all 3.

double sea_level = 347.0;
double test_diameter = 1.0;
double test_epsilon = 1.0;
int seed = 12345;

double verlet_cutoff = 64.0;

int number_of_molecules = 0;
int close_molecules;

FILE *instream;

int main(int argc, char *argv[])
{
  double sq_distance_from_initial_pt;
  int i;

  // overhead

  setCommandLineParameters(argc, argv);
  verbose = getFlagParam("-verbose");
  getIntParam("-seed", &seed);
  if (getFlagParam("-randomize")) randomize();
  else initializeRandomNumberGenerator();

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getDoubleParam("-sea_level", &sea_level);
  getDoubleParam("-test_diameter", &test_diameter);
  getDoubleParam("-test_epsilon", &test_epsilon);
  getIntParam("-n", &number_of_samples);

  if (getFlagParam("-usage"))
  {
    printf("\nusage:\t-box [ 10 10 10 ]\n");
    printf("      \t-verlet_cutoff [ 64.0 ]\n");
    printf("      \t-sea_level [ 347.0 ]\n");
    printf("      \t-test_diameter [ 1.0 ]\n");
    printf("      \t-test_epsilon [ 1.0 ]\n");
    printf("      \t-n [ 100 ]\n\n");
    exit(0);
  }

  x_increment=box_x/(0.0 + number_of_samples);
  y_increment=box_y/(0.0 + number_of_samples);
  z_increment=box_z/(0.0 + number_of_samples);

  readConfiguration();
  for (i=0; i<number_of_molecules; i++)
  {
    while (x[i] >= box_x) x[i] -= box_x;
    while (y[i] >= box_y) y[i] -= box_y;
    while (z[i] >= box_z) z[i] -= box_z;
    while (x[i] < 0) x[i] += box_x;
    while (y[i] < 0) y[i] += box_y;
    while (z[i] < 0) z[i] += box_z;
  }
  
  for (test_x=0; test_x < box_x; test_x += x_increment)
    for (test_y=0; test_y < box_y; test_y += y_increment)
      for (test_z=0; test_z < box_z; test_z += z_increment)
        if (calculateEnergy(test_diameter) < sea_level) printf("%lf\t%lf\t%lf\t%lf\n", test_x, test_y, test_z, test_diameter);

  return 0;
}

double calculateEnergy(double test_diameter)
{
  int i;
  double dx, dy, dz, dd;
  double d6, d12, sigma1, sigma6, sigma12;
  double shift_x, shift_y, shift_z;
  double interaction=0;

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

      if (dd > verlet_cutoff) continue;

      d6 = dd*dd*dd;
      d12 = d6*d6;

      sigma6 = Xsigma[i]*Xsigma[i]*Xsigma[i]*Xsigma[i]*Xsigma[i]*Xsigma[i];
      sigma12 = sigma6*sigma6;

      interaction += Xepsilon[i] * (sigma12/d12 - sigma6/d6);
    }
  }

  return 4.0 * (interaction);
}

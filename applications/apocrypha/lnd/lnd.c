/* lnd.c */

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

double close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE];
double close_sigma[MAX_CLOSE];
double close_sigma6[MAX_CLOSE];
double close_sigma12[MAX_CLOSE];

double box_x=6, box_y=6, box_z=6;

int number_of_samples = 1;

double test_x, test_y, test_z;
double sea_level = 1.0;
double test_diameter = 1.0;
int seed = 12345;

double verlet_cutoff = 25.0;

int number_of_molecules = 0;
int close_molecules;

FILE *instream;

int main(int argc, char *argv[])
{
  double sq_distance_from_initial_pt;

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
  getIntParam("-n", &number_of_samples);

  readConfiguration();
  
  while (number_of_samples>0)
  {
    generateTestPoint();
    if (calculateEnergy(test_diameter) < sea_level)
    {
      printf("%lf\t%lf\t%lf\t%lf\n", test_x, test_y, test_z, test_diameter);
      number_of_samples--;
    }
  }

  return 0;
}

generateTestPoint()
{
  test_x = rnd() * box_x;
  test_y = rnd() * box_y;
  test_z = rnd() * box_z;
}

double calculateEnergy(double test_diameter)
{
  int i;
  double dx, dy, dz, dd;
  double d6, d12, sigma1, sigma6, sigma12;
  double shift_x, shift_y, shift_z;
  double repulsion=0;
  double attraction=0;

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

      sigma1 = 0.5 * (sigma[i] + test_diameter);
      sigma6 = sigma1*sigma1*sigma1*sigma1*sigma1*sigma1;
      sigma12 = sigma6*sigma6;

      repulsion += sigma12/d12;
      attraction += sigma6/d6;
    }
  }

  return 4.0 * (repulsion - attraction);
}

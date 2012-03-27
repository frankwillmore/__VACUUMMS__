/* walk.c */

#include "io_setup.h"
#include "walk.h"

#include <math.h>
#include <ftw_std.h>
#include <ftw_rng.h>
#include <ftw_param.h>

#define MAX_NUM_MOLECULES 16384
#define MAX_CLOSE 2048

/* NOTE:  sigma is specified in Angstroms, epsilon in kcal/mol, T in K */
/* kB is in kcal/molK */
#define kB 0.034787125239584

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

double step_size_factor = 1.0;
int n_steps = 1024;

double T=298, beta;
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

FILE *instream;

int main(int argc, char *argv[])
{
  double sq_distance_from_initial_pt;
  double step_number;
  double dx, dy, dz;

  // overhead

  setCommandLineParameters(argc, argv);
  verbose = getFlagParam("-verbose");
  getIntParam("-seed", &seed);
  if (getFlagParam("-randomize")) randomize();
  else initializeRandomNumberGeneratorTo(seed);

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-step_size_factor", &step_size_factor);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getDoubleParam("-test_diameter", &test_diameter);
  getDoubleParam("-T", &T);
  getIntParam("-n", &number_of_samples);
  getIntParam("-n_steps", &n_steps);

  beta=1/(kB*T);

  readConfiguration();
  
  while (number_of_samples-- > 0)
  {
    generateTestPoint();
    successes=0;
    drift_x=drift_y=drift_z=0;
    
    for (step_number=0; step_number<n_steps; step_number++)
    {
      attemptRandomStep();
      dx=test_x - verlet_center_x;
      dy=test_y - verlet_center_y;
      dz=test_z - verlet_center_z;
      if (dx*dx + dy*dy + dz*dz > .01 * verlet_cutoff) makeVerletList();
    }
  
    dx = test_x - test_x0 + drift_x;
    dy = test_y - test_y0 + drift_y;
    dz = test_z - test_z0 + drift_z;

    printf("%lf\t%d\n", sqrt(dx*dx + dy*dy + dz*dz), successes);
  }

  return 0;
}

attemptRandomStep()
{
  double step_x, step_y, step_z;
double metropolis;

  step_x = (rnd()-.5) * step_size_factor;
  step_y = (rnd()-.5) * step_size_factor;
  step_z = (rnd()-.5) * step_size_factor;

//  energy = new_energy;

  energy = calculateEnergy();

  test_x += step_x;
  test_y += step_y;
  test_z += step_z;
  
  new_energy = calculateEnergy();
  if ((new_energy < energy) || (rnd() < exp(-(new_energy-energy)*beta)))
  {
    successes++;
    return;
  }
  else 
  {
    test_x-=step_x;
    test_y-=step_y;
    test_z-=step_z;
    new_energy=energy;
  }
}

generateTestPoint()
{
  test_x = test_x0 = rnd() * box_x;
  test_y = test_y0 = rnd() * box_y;
  test_z = test_z0 = rnd() * box_z;

  makeVerletList();

  while (rnd() > exp(-beta*calculateEnergy())) 
  {
    test_x = test_x0 = rnd() * box_x;
    test_y = test_y0 = rnd() * box_y;
    test_z = test_z0 = rnd() * box_z;

    makeVerletList();
  }
}

makeVerletList()
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

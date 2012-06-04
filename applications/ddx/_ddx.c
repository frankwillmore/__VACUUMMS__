/* ddx.c */

#include "ddx.h"
#include "io_setup.h"

#include <ftw_std.h>
#include <ftw_rng.h>
#include <ftw_param.h>

#define MAX_NUM_MOLECULES 16384
#define MAX_CLOSE 2048

double x[MAX_NUM_MOLECULES];
double y[MAX_NUM_MOLECULES];
double z[MAX_NUM_MOLECULES];
double sigma[MAX_NUM_MOLECULES];

double close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE];
double close_sigma[MAX_CLOSE];
double close_sigma6[MAX_CLOSE];
double close_sigma12[MAX_CLOSE];

double box_x=6, box_y=6, box_z=6;
double verlet_cutoff=100.0;

double step_size_factor = 1.0;
int n_steps = 250;

int number_of_samples = 1;
int volume_sampling = 0;

double test_x0, test_y0, test_z0;
double test_x, test_y, test_z;
double verlet_center_x, verlet_center_y, verlet_center_z;
double diameter = 1.0;
double min_diameter = 0.0;
int seed = 0;

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
  getDoubleParam("-step_size_factor", &step_size_factor);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getIntParam("-n", &number_of_samples);
  getIntParam("-n_steps", &n_steps);
  volume_sampling = getFlagParam("-volume_sampling");
  getDoubleParam("-min_diameter", &min_diameter);

  if (getFlagParam("-usage"))
  {
    printf("\nusage:\t-box [ 6.0 6.0 6.0 ]\n");
    printf("\t\t-seed [ 0 ]\n");
    printf("\t\t-randomize \n");
    printf("\t\t-step_size_factor [ 1.0 ]\n");
    printf("\t\t-verlet_cutoff [ 100.0 ]\n");
    printf("\t\t-n [ 1 ]\n");
    printf("\t\t-n_steps [ 250 ]\n");
    printf("\t\t-volume_sampling \n");
    printf("\t\t-min_diameter [ 0.0 ]");
    printf("\n");
    exit(0);
  }

  readConfiguration();
  
  while (number_of_samples>0)
  {
    generateTestPoint();
    while (calculateEnergy(0.0) > 0) generateTestPoint();
    
    findEnergyMinimum();
    
    sq_distance_from_initial_pt = (test_x-test_x0)*(test_x-test_x0) + (test_y-test_y0)*(test_y-test_y0) + (test_z-test_z0)*(test_z-test_z0);
    if (!volume_sampling || (sq_distance_from_initial_pt < .25 * diameter * diameter))
    {
      makeVerletList();
      expandTestParticle();
      if (diameter > min_diameter) 
      {

        // correct for box edges...
        while (test_x >= box_x) test_x -= box_x;
        while (test_x < 0) test_x += box_x;
        while (test_y >= box_y) test_y -= box_y;
        while (test_y < 0) test_y += box_y;
        while (test_z >= box_z) test_z -= box_z;
        while (test_z < 0) test_z += box_z;

        printf("%lf\t%lf\t%lf\t%lf\n", test_x, test_y, test_z, diameter);
        number_of_samples--;
      }
    }
  }

  return 0;
}

generateTestPoint()
{
  test_x = test_x0 = rnd() * box_x;
  test_y = test_y0 = rnd() * box_y;
  test_z = test_z0 = rnd() * box_z;

  makeVerletList();
}

makeVerletList()
{
  int i;
  double dx, dy, dz, dd;
  double shift_x, shift_y, shift_z;

  while (test_x > box_x) test_x -= box_x;
  while (test_y > box_y) test_y -= box_y;
  while (test_z > box_z) test_z -= box_z;

  while (test_x < 0) test_x += box_x;
  while (test_y < 0) test_y += box_y;
  while (test_z < 0) test_z += box_z;

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
        close_sigma6[close_molecules] = sigma[i]*sigma[i]*sigma[i]*sigma[i]*sigma[i]*sigma[i];
        close_sigma12[close_molecules] = close_sigma6[close_molecules]*close_sigma6[close_molecules];

        close_molecules++;
      }
    }
  }

}

findEnergyMinimum()
{
  double dx, dy, dz, dd, d6, d14;
  double factor;
  double old_energy, new_energy;
  double grad_x, grad_y, grad_z;
  double length_grad;
  double step_size;
  double step_x, step_y, step_z;
  int i, attempts;
  double drift_sq;

  makeVerletList();

  // begin loop to iterate until minimum found
  for (attempts=0; attempts<n_steps; attempts++)
  {
//printf("attempt #%ld:  ", attempts);
//printTestPoint();
    drift_sq = (test_x-verlet_center_x)*(test_x-verlet_center_x) 
             + (test_y-verlet_center_y)*(test_y-verlet_center_y) 
             + (test_z-verlet_center_z)*(test_z-verlet_center_z);
    if (drift_sq > .01 * verlet_cutoff)
    {
//printf("building list at %d attempts\n", attempts);
      makeVerletList();
    }

    // find the gradient at test_x, test_y, test_Z

    grad_x=0; grad_y=0; grad_z=0;

    for (i=0; i<close_molecules; i++)
    {
      dx = test_x - close_x[i];
      dy = test_y - close_y[i];
      dz = test_z - close_z[i];
      dd = dx*dx + dy*dy + dz*dz;
      d6 = dd*dd*dd;
      d14 = d6*d6*dd;

      factor = 48 * close_sigma12[i] / d14;

      grad_x += dx * factor;
      grad_y += dy * factor;
      grad_z += dz * factor;
    }

    // some rule here to determine new position from grad
    // step size factor is a tunable parameter to give volume sampling.  In limit as -> 0, volume sampling results.
    length_grad = sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
//printf("length_grad = %lf\t", length_grad);
//printTestPoint();

    if (length_grad < .000001) break;

    step_size = step_size_factor/length_grad;
    old_energy = calculateRepulsion();

    step_x = step_size*grad_x;
    step_y = step_size*grad_y;
    step_z = step_size*grad_z;

    while (step_size>0.000001)
    {
      test_x += step_x;
      test_y += step_y;
      test_z += step_z;
 
      // check repulsion at new location
      new_energy = calculateRepulsion();

      // accept/reject the move/try half-step
      if (new_energy > old_energy)
      {
        test_x -= step_x;
        test_y -= step_y;
        test_z -= step_z;
        step_x *= .5;
        step_y *= .5;
        step_z *= .5;
        step_size *=.5;
//printf("step size:  %lf\n", step_size);
      }
      else break;
    }
  }
}

printTestPoint()
{
  printf("test:  %lf\t%lf\t%lf\n", test_x, test_y, test_z);
}

double calculateRepulsion()
{
  double repulsion=0;
  double dx, dy, dz, dd, d6, d12;
  int i;

  for (i=0; i<close_molecules; i++)
  {
    dx = close_x[i] - test_x;
    dy = close_y[i] - test_y;
    dz = close_z[i] - test_z;
    dd = dx*dx + dy*dy + dz*dz;
    d6 = dd*dd*dd;
    d12 = d6*d6;

    repulsion += close_sigma12[i] / d12;
  }
 
  return 4.0 * repulsion;
}

double calculateEnergy(double test_diameter)
{
  double repulsion=0;
  double attraction=0;
  double dx, dy, dz, dd, d6, d12;
  double sigma, sigma6, sigma12;
  int i;

  for (i=0; i<close_molecules; i++)
  {
    dx = close_x[i] - test_x;
    dy = close_y[i] - test_y;
    dz = close_z[i] - test_z;
    dd = dx*dx + dy*dy + dz*dz;
    d6 = dd*dd*dd;
    d12 = d6*d6;

    sigma = 0.5 * (close_sigma[i] + test_diameter);
    sigma6 = sigma*sigma*sigma*sigma*sigma*sigma;
    sigma12 = sigma6*sigma6;

    repulsion += sigma12/d12;
    attraction += sigma6/d6;
  }

  return 4.0 * (repulsion - attraction);
}

expandTestParticle()
{
  double slope;
  double step_size;
  double energy, old_energy;
  double e0, e1, r0, r1;

  diameter = 0;

  // improved initial guess
  old_energy = calculateEnergy(diameter);
//printf("energy at sigma = 0:  %lf\n", old_energy);
  if (old_energy > 0) return -1;
  while (diameter += .1)
  {
    energy = calculateEnergy(diameter);
    if (energy > old_energy) break;
    old_energy = energy;
  }

//printf("starting w/diameter=%lf\n", diameter);
    
  // Newton's method

  while(1)
  {
    r0 = diameter - .001;
    r1 = diameter + .001;
    
    e0 = calculateEnergy(r0);
    e1 = calculateEnergy(r1);
    energy = calculateEnergy(diameter);

    slope = (e1-e0)/(r1-r0);
    step_size = -energy/slope;

    diameter = diameter + step_size;

    if (step_size*step_size < .00000001) break;
  }
}


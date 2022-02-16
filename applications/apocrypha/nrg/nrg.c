/*************************************** nrg.c ********************************************/

/* input:   configuration in .cfg format */
/* output:  list of distances traveled in .dst format */

#include "nrg.h"
#include "io_setup.h"
#include "energy.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ftw_science.h>
#include <ftw_std.h>
#include <ftw_rng.h>

#ifndef MAX_NUMBER_MOLECULES
#define MAX_NUMBER_MOLECULES 1024
#endif

/* non-configurable global params */
double x[MAX_NUMBER_MOLECULES], y[MAX_NUMBER_MOLECULES], z[MAX_NUMBER_MOLECULES];

int rng_seed = 6305;
int n_steps = 1000;
int number_of_molecules;
double temperature;
double box_x, box_y, box_z;

double perturbation_length = 1.0;
double acceptance_ratio;
int accepted_moves;
int n_trials = 1000;

double x_translation, y_translation, z_translation;
double total_translation;

FILE *instream;

int main(int argc, char *argv[])
{
  int i=0;
  double old_energy, new_energy;
  double delta_energy;
  double boltzmann_factor;
  double the_exponential;
  int monte_carlo_steps;
  int trial_number;
  double x0, y0, z0;
  double dx, dy, dz;
  double rvlen;

  double sum_dx, sum_dy, sum_dz;
   
  instream = stdin;

  while (++i < argc)
  {
    if ((argc>1) && (*argv[i] != '-')) instream = fopen(argv[i], "r");
    else if (!strcmp(argv[i], "-v")) verbose = 1;
    else if (!strcmp(argv[i], "-T")) temperature = getDoubleParameter("temperature", argv[++i]);
    else if (!strcmp(argv[i], "-dim")) 
    {
      box_x = getDoubleParameter("box_x", argv[++i]);
      box_y = getDoubleParameter("box_y", argv[++i]);
      box_z = getDoubleParameter("box_z", argv[++i]);
    }
    else if (!strcmp(argv[i], "-n_trials")) n_trials = getDoubleParameter("n_trials", argv[++i]);
    else if (!strcmp(argv[i], "-rng_seed")) rng_seed = getIntParameter("rng_seed", argv[++i]);
    else if (!strcmp(argv[i], "-n_steps")) n_steps = getIntParameter("n_steps", argv[++i]);
    else if (!strcmp(argv[i], "-randomize")) rng_seed = getRandomSeed();
    else if (!strcmp(argv[i], "-perturbation_length")) perturbation_length = getDoubleParameter("perturbation_length", argv[++i]);
  }

  initializeRandomNumberGeneratorTo(rng_seed);
  loadConfiguration();

  for (trial_number = 0; trial_number < n_trials; trial_number++)
  {
    sum_dx = 0;
    sum_dy = 0;
    sum_dz = 0;
    V printf("trial number %d\n", trial_number);
    accepted_moves = 0;

    // pick starting point
    x0 = x[number_of_molecules] = box_x * rnd();
    y0 = y[number_of_molecules] = box_y * rnd();
    z0 = z[number_of_molecules] = box_z * rnd();

    // move it a bunch of times
    for(monte_carlo_steps=0; monte_carlo_steps<n_steps; monte_carlo_steps++)
    {
      dx = 2*(rnd() - .5);
      dy = 2*(rnd() - .5);
      dz = 2*(rnd() - .5);

      rvlen = perturbation_length / sqrt(dx*dx + dy*dy + dz*dz);
      dx *= rvlen;
      dy *= rvlen;
      dz *= rvlen;
      
      // try out the new position
      old_energy = calculateSystemEnergy();

      x[number_of_molecules] += dx;
      y[number_of_molecules] += dy;
      z[number_of_molecules] += dz;
      adjustForBoundary(number_of_molecules);

      new_energy = calculateSystemEnergy();
      delta_energy = new_energy - old_energy;
     
      if (delta_energy <= 0) 
      {
        accepted_moves++;
        sum_dx += dx;
        sum_dy += dy;
        sum_dz += dz;
        continue; /* move accepted */
      }

      the_exponential = 0.0 - delta_energy/temperature;

      if (the_exponential > -75)
      {
        boltzmann_factor = exp(the_exponential);
        if (boltzmann_factor > rnd())
        {
          accepted_moves++;
          sum_dx += dx;
          sum_dy += dy;
          sum_dz += dz;
          continue; /* move accepted */
        }
      }

      /* move rejected: */
      x[number_of_molecules] -= dx;
      y[number_of_molecules] -= dy;
      z[number_of_molecules] -= dz;
      adjustForBoundary(number_of_molecules);
    } 

    // look at end point, compare

    total_translation = sqrt(sum_dx*sum_dx + sum_dy*sum_dy + sum_dz*sum_dz);

    printf("%lf\n", total_translation);
    V printf("%lf\t%lf\t%lf\t%lf\n", sum_dx, sum_dy, sum_dz, total_translation);
    V printf("acceptance ratio:  %lf\n", (0.0 + accepted_moves)/n_steps);
  }

  return 0;
} /* end main */

void adjustForBoundary(int n)
{
  if (x[n] <= 0) x[n] += box_x;
  if (x[n] >= box_x) x[n] -= box_x;
  if (y[n] <= 0) y[n] += box_y;
  if (y[n] >= box_y) y[n] -= box_y;
  if (z[n] <= 0) z[n] += box_z;
  if (z[n] >= box_z) z[n] -= box_z;
}

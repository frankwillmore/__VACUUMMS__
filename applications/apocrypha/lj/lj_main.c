/*************************************** lj_main.c ********************************************/

#include "lj_main.h"
#include "io_setup.h"
#include "graphics.h"
#include "command_line_parser.h"
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
int wsize_x, wsize_y, wsize_z;
int change_flag = TRUE; /* signals need to update display */
int monte_carlo_steps = 0;
int monte_carlo_step_counter = 0;

extern int rng_seed;
extern int end_mcs;
extern int number_of_molecules;
extern double temperature;
extern double box_x, box_y, box_z;
extern double target_acceptance_ratio;

double perturbation_length = 1.0;
double acceptance_ratio;
int attempted_moves;
int accepted_moves;
int number_of_pairs;

int main(int argc, char *argv[])
{
  parseCommandLineOptions(argc, argv);
  initializeRandomNumberGeneratorTo(rng_seed);
  readEnvironmentVariables();
  initializeOutput();
  setInitialConditions();
  if (graphicsModeEnabled()) initializeDisplay();

  for(monte_carlo_steps=0; monte_carlo_steps<=end_mcs; monte_carlo_steps++)
  {
    generateOutput();
    attempted_moves = 0;
    accepted_moves = 0;
    for (monte_carlo_step_counter=0; monte_carlo_step_counter<number_of_molecules; monte_carlo_step_counter++) perturbSystem();
    acceptance_ratio = (0.0 + accepted_moves)/(0.0 + attempted_moves);
    if (acceptance_ratio < target_acceptance_ratio) perturbation_length *= .9;
    else if (perturbation_length*perturbation_length*perturbation_length*16 < box_x*box_y*box_z) perturbation_length *=1.1;
    if (graphicsModeEnabled() && changeFlagIsSet()) drawGraphicalRepresentation();
  } 

  finalizeOutput();
  return 0;
} /* end main */

/* generate a move, apply metropolis test to accept or reject */
void perturbSystem()
{
  double dx, dy, dz;
  double old_x, old_y, old_z;
  double old_energy, new_energy;
  double boltzmann_factor;
  double delta_energy;
  double the_exponential;
  double random_number;
  int particle_number;
  
  /* which molecule and how much to move it */
  attempted_moves++;
  particle_number = floor(number_of_molecules * rnd());
  old_x = x[particle_number];
  old_y = y[particle_number];
  old_z = z[particle_number];
  dx = (rnd() -.5) * perturbation_length;
  dy = (rnd() -.5) * perturbation_length;
  dz = (rnd() -.5) * perturbation_length;
  old_energy = calculateSystemEnergy();
  updatePosition(particle_number, dx, dy, dz);
  new_energy = calculateSystemEnergy();

  delta_energy = new_energy - old_energy;
  if (delta_energy < 0) 
  {
    change_flag = 1;
    accepted_moves++;
    return; /* move accepted */
  }

  // the following uses reduced temperature
  the_exponential = 0.0 - delta_energy/temperature;
//  the_exponential = 0.0 - delta_energy/(avogadros_number * boltzmann_constant * temperature);
////  the_exponential = 0.0 - delta_energy/(number_of_molecules * boltzmann_constant * temperature);
  /* evaluate exponential, unless it's arbitrarily small */
//  if (the_exponential > -75)
  if (the_exponential > -25)
  {
    boltzmann_factor = exp(the_exponential);
    random_number = rnd();
    if (boltzmann_factor > random_number)
    {
      change_flag = 1;
      accepted_moves++;
      return; /* move accepted */
    }
  }

  x[particle_number] = old_x;
  y[particle_number] = old_y;
  z[particle_number] = old_z;
}

void updatePosition(int particle_number, double dx, double dy, double dz)
{
  x[particle_number] += dx;
  if (x[particle_number] > box_x) x[particle_number] -= box_x;
  if (x[particle_number] < 0) x[particle_number] = x[particle_number] + box_x;
  y[particle_number] += dy;
  if (y[particle_number] > box_y) y[particle_number] -= box_y;
  if (y[particle_number] < 0) y[particle_number] += box_y;
  z[particle_number] += dz;
  if (z[particle_number] > box_z) z[particle_number] -= box_z;
  if (z[particle_number] < 0) z[particle_number] += box_z;
}

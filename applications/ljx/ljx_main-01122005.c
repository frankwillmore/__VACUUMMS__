/*************************************** ljx_main.c ********************************************/

#include "ljx_main.h"
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
#define MAX_NUMBER_MOLECULES 16384
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
extern int num_pairs;
extern double temperature;
extern double box_x, box_y, box_z;
extern double target_acceptance_ratio;

extern double pair_list_xoffset[];
extern double pair_list_yoffset[];
extern double pair_list_zoffset[];
extern int pair_list_first[];
extern int pair_list_second[];

double fixed_perturbation_length = .2;
double perturbation_length;
double acceptance_ratio;
int attempted_moves;
int accepted_moves;
int relaxation_allowance = 50;

double dx, dy, dz;
double delta_energy;
int particle_number;

int main(int argc, char *argv[])
{
  parseCommandLineOptions(argc, argv);
  initializeRandomNumberGeneratorTo(rng_seed);
  initializeOutput();
  setInitialConditions();
  if (graphicsModeEnabled()) initializeDisplay();
  
  perturbation_length=fixed_perturbation_length;

  for(monte_carlo_steps=0; monte_carlo_steps<=end_mcs; monte_carlo_steps++)
  {
    updatePairList();
    generateOutput();
    attempted_moves = 0;
    accepted_moves = 0;

    for (monte_carlo_step_counter=0; monte_carlo_step_counter<number_of_molecules; monte_carlo_step_counter++) 
    {
      double boltzmann_factor;
      double the_exponential;
  
      delta_energy = 0;
      attemptMove();
      attempted_moves++;

      if (delta_energy < 0) 
      {
        change_flag = 1;
        accepted_moves++;
        continue; /* move accepted */
      }

      // the following uses reduced temperature
      the_exponential = 0.0 - delta_energy/temperature;
     /* evaluate exponential, unless it's arbitrarily small */
     if (the_exponential > -25)
     {
        boltzmann_factor = exp(the_exponential);
        if (boltzmann_factor > rnd())
        {
          change_flag = 1;
          accepted_moves++;
          continue; /* move accepted */
        }
      }

      // revert move
      x[particle_number] -= dx;
      y[particle_number] -= dy;
      z[particle_number] -= dz;
    }

    if (monte_carlo_steps < relaxation_allowance) 
    {
      acceptance_ratio = (0.0 + accepted_moves)/(0.0 + attempted_moves);
      if (acceptance_ratio < target_acceptance_ratio) perturbation_length *= .9;
      else if (perturbation_length*perturbation_length*perturbation_length*16 < box_x*box_y*box_z) perturbation_length *=1.1;
    }
    else perturbation_length = fixed_perturbation_length;
    if (graphicsModeEnabled() && changeFlagIsSet()) drawGraphicalRepresentation();
  } 

  finalizeOutput();
  return 0;
} /* end main */

// Attempt a move, and return the energy change in doing so.
void attemptMove()
{
  int pair_no;

  delta_energy = 0;
  particle_number = floor(number_of_molecules * rnd());

  // calculate energy of all affected pairs before move...
  for (pair_no=0; pair_no<num_pairs; pair_no++)
  {
    if (pair_list_first[pair_no] == particle_number) delta_energy -= getPairEnergy(pair_no);
    if (pair_list_second[pair_no] == particle_number) delta_energy -= getPairEnergy(pair_no);
  }

  dx = (rnd() -.5) * perturbation_length;
  dy = (rnd() -.5) * perturbation_length;
  dz = (rnd() -.5) * perturbation_length;

  x[particle_number] += dx;
  y[particle_number] += dy;
  z[particle_number] += dz;

  // and after the move...
  for (pair_no=0; pair_no<num_pairs; pair_no++)
  {
    if (pair_list_first[pair_no] == particle_number) delta_energy += getPairEnergy(pair_no);
    if (pair_list_second[pair_no] == particle_number) delta_energy += getPairEnergy(pair_no);
  }

  delta_energy *= 4;
}

/*************************************** sdhs_main.c ********************************************/

#include "sdhs_main.h"
#include "io_setup.h"
#include "graphics.h"

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ftw_science.h>
#include <ftw_std.h>
#include <ftw_rng.h>
#include <ftw_param.h>

#ifndef MAX_NUMBER_MOLECULES
#define MAX_NUMBER_MOLECULES 1024
#endif

/* non-configurable global params */
double x[MAX_NUMBER_MOLECULES], y[MAX_NUMBER_MOLECULES], z[MAX_NUMBER_MOLECULES];
int wsize_x, wsize_y, wsize_z;
int change_flag = TRUE; /* signals need to update display */
int monte_carlo_steps = 0;
int monte_carlo_step_counter = 0;

double acceptance_ratio;
double perturbation_length = .2;
int attempted_moves;
int accepted_moves;
int number_of_pairs;
int verbose = 0;
int graphics = 1;
int side_view = 1;
int particle_scale = 64; /* how many pixels */
int number_of_molecules = 25;
double box_x = 7.0;
double box_y = 7.0;
double box_z = 7.0;
int fg_color = 255;
int bg_color = 0;
int min_color = 64;
int rng_seed = 24375;
/* L-J params are chosen such that intercepts are at 1, 2.5 */
int energy_report_frequency = 100;
int running_average_steps = 1000;
double running_average = 0.0;
int end_mcs = 50000; /* exit simulation after reaching this pt */
int configuration_threshold = 0; /* start to dump config after this many mcs */
int configuration_frequency = 1000;
char *output_file_name = "hs.out";
char *input_file_name;
char *log_file_name = "hs.log";
char *simulation_unique_identifier = "################";
char hostname[50] = "";
double target_acceptance_ratio = .15;

char *display_name_1 = "X-Y projection (front)";
char *display_name_2 = "Z-Y projection (right side)";

int x_laps[MAX_NUMBER_MOLECULES], y_laps[MAX_NUMBER_MOLECULES], z_laps[MAX_NUMBER_MOLECULES];

int mirror_depth = 1;

int main(int argc, char *argv[])
{
  int i;

  setCommandLineParameters(argc, argv);
  verbose = getFlagParam("-v");
  if(getFlagParam("-ng")) graphics = 0;
  if(getFlagParam("-no_side")) {side_view = 0; graphics = 1;}
  getIntParam("-particle_scale", &particle_scale);
  getIntParam("-N", &number_of_molecules);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getIntParam("-fg_color", &fg_color);
  getIntParam("-bg_color", &bg_color);
  getIntParam("-min_color", &min_color);
  getIntParam("-rng_seed", &rng_seed);
  if (getFlagParam("-randomize")) rng_seed = getRandomSeed();
  getIntParam("-end_mcs", &end_mcs);
  getIntParam("-energy_report_frequency", &energy_report_frequency);
  getIntParam("-configuration_threshold", &configuration_threshold);
  getIntParam("-configuration_frequency", &configuration_frequency);
  getStringParam("-log_file_name", &log_file_name);
  getStringParam("-input_file_name", &input_file_name);
  getStringParam("-simulation_unique_identifier", &simulation_unique_identifier);
  getDoubleParam("-target_acceptance_ratio", &target_acceptance_ratio);
  getDoubleParam("-perturbation_length", &perturbation_length);

  initializeRandomNumberGeneratorTo(rng_seed);
  for (i=0; i<MAX_NUMBER_MOLECULES; i++) x_laps[i] = 0;
  for (i=0; i<MAX_NUMBER_MOLECULES; i++) y_laps[i] = 0;
  for (i=0; i<MAX_NUMBER_MOLECULES; i++) z_laps[i] = 0;

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
    if (graphicsModeEnabled() && changeFlagIsSet()) drawGraphicalRepresentation();
  } 

  finalizeOutput();
  return 0;
} /* end main */

/* generate a move, check for overlap */
void perturbSystem()
{
  double dx, dy, dz;
  int particle_number;
  
  /* which molecule and how much to move it */
  attempted_moves++;
  particle_number = floor(number_of_molecules * rnd());
  dx = (rnd() -.5) * perturbation_length;
  dy = (rnd() -.5) * perturbation_length;
  dz = (rnd() -.5) * perturbation_length;

  updatePosition(particle_number, dx, dy, dz);

  if (checkForOverlap(particle_number))
  {
    updatePosition(particle_number, -dx, -dy, -dz);
  }
  else 
  {
    change_flag = 1;
    accepted_moves++;
  }
}

void updatePosition(int particle_number, double dx, double dy, double dz)
{
  x[particle_number] += dx;
  if (x[particle_number] > box_x)
  {
//printf("updating ++x_laps[%d]\n", particle_number);
     x[particle_number] -= box_x;
     x_laps[particle_number]++;
  }
  if (x[particle_number] < 0)
  {
    x[particle_number] += box_x;
    x_laps[particle_number]--;
  }
  y[particle_number] += dy;
  if (y[particle_number] > box_y)
  {
    y[particle_number] -= box_y;
    y_laps[particle_number]++;
  }
  if (y[particle_number] < 0) 
  {
    y[particle_number] += box_y;
    y_laps[particle_number]--;
  }
  z[particle_number] += dz;
  if (z[particle_number] > box_z)
  {
    z[particle_number] -= box_z;
    z_laps[particle_number]++;
  }
  if (z[particle_number] < 0) 
  {
    z[particle_number] += box_z;
    z_laps[particle_number]--;
  }
}

int checkForOverlap(int particle_number)
{
  int i;
  int mirror_x, mirror_y, mirror_z;
  double dist_x, dist_y, dist_z;
  double dist_sq;

  for (i=0; i<number_of_molecules; i++)
  {
    if (i==particle_number) continue;
    for (mirror_x=-mirror_depth; mirror_x<=mirror_depth; mirror_x++)
    for (mirror_y=-mirror_depth; mirror_y<=mirror_depth; mirror_y++)
    for (mirror_z=-mirror_depth; mirror_z<=mirror_depth; mirror_z++)
    {
      dist_x = mirror_x*box_x + x[particle_number] - x[i];
      dist_y = mirror_y*box_y + y[particle_number] - y[i];
      dist_z = mirror_z*box_z + z[particle_number] - z[i];

      dist_sq = dist_x*dist_x + dist_y*dist_y + dist_z*dist_z;
      if (dist_sq < 1) return 1;
    }
  }

  return 0;
}

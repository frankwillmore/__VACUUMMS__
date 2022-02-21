/* ddsg.c */

#include "ddsg.h"
#include "io_setup.h"
#include "convergence.h"

#include <ftw_std.h>
#include <ftw_rng.h>
#include <ftw_param.h>

#define MAX_NUM_MOLECULES 8000

double x[MAX_NUM_MOLECULES];
double y[MAX_NUM_MOLECULES];
double z[MAX_NUM_MOLECULES];

double box_x=7, box_y=7, box_z=7;
double max_scale = 1;
double c_convergence_ratio=.92;
double d_step_size = .1;
int number_of_samples = 1;
int n_trials = 1000;
int d_trials = 100;
int volume_sampling = 0;

int number_of_molecules = 0;
FILE *instream;

int main(int argc, char *argv[])
{
  instream = stdin;

  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  verbose = getFlagParam("-verbose");
  if (getFlagParam("-randomize")) randomize();
  else initializeRandomNumberGenerator();

  getDoubleParam("-c_convergence_ratio", &c_convergence_ratio);
  getDoubleParam("-d_step_size", &d_step_size);
  getIntParam("-n", &number_of_samples);
  getIntParam("-n_trials", &n_trials);
  volume_sampling = getFlagParam("-volume_sampling");

  readConfiguration();
  sampleCavities();
  return 0;
}

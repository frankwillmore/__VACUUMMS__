#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <ftw_std.h>
#include <ftw_rng.h>

#include "command_line_parser.h"

/* parameters configurable on command line and default values */
int verbose = 0;
int graphics = 0;
int side_view = 1;
double temperature = 1;
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
int start_mcs = 0;
int configuration_threshold = 0; /* start to dump config after this many mcs */
int configuration_frequency = 1000;
char *output_file_name = "lj.out";
char *input_file_name;
char *log_file_name = "lj.log";
char simulation_unique_identifier[25] = "################";
char hostname[50] = "";
double target_acceptance_ratio = .15;
extern int relaxation_allowance;
extern double fixed_perturbation_length;
extern double psi_shift;
extern double cutoff_sq;

char *display_name_1 = "X-Y projection (front)";
char *display_name_2 = "Z-Y projection (right side)";

void parseCommandLineOptions(int argc, char *argv[])
{
  int i;

  for (i = 0; i<argc; i++)
  {
    if (!strcmp(argv[i], "-usage")) printUsage();
    if (!strcmp(argv[i], "-v")) verbose = 1;
    if (!strcmp(argv[i], "-X11")) graphics = 1;
    if (!strcmp(argv[i], "-no_side")) {side_view = 0; graphics = 1;}
    if (!strcmp(argv[i], "-T")) temperature = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-fixed_perturbation_length")) fixed_perturbation_length = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-particle_scale")) particle_scale = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-N")) number_of_molecules = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-relaxation_allowance")) relaxation_allowance = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-fg_color")) fg_color = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-bg_color")) bg_color = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-min_color")) min_color = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-rng_seed")) rng_seed = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-randomize")) rng_seed = getRandomSeed();
    if (!strcmp(argv[i], "-end_mcs")) end_mcs = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-start_mcs")) start_mcs = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-energy_report_frequency")) energy_report_frequency = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-configuration_threshold")) configuration_threshold = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-configuration_frequency")) configuration_frequency = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-log_file_name")) log_file_name = argv[++i];
    if (!strcmp(argv[i], "-input_file_name")) input_file_name = argv[++i];
    if (!strcmp(argv[i], "-simulation_unique_identifier")) strcpy(simulation_unique_identifier, argv[++i]);
    if (!strcmp(argv[i], "-target_acceptance_ratio")) target_acceptance_ratio = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-psi_shift")) psi_shift = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-cutoff_sq")) cutoff_sq = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-box"))
    {
       box_x = strtod(argv[++i], NULL);
       box_y = strtod(argv[++i], NULL);
       box_z = strtod(argv[++i], NULL);
    }
  }

  psi_shift *=0.25;
  initializeRandomNumberGeneratorTo(rng_seed);
}

void printUsage()
{
  printf("usage:  lj \n\t[-v]\n\t[-ng]\n\t[-no_side]\n\t-T [1]\n\t-fixed_perturbation_length []\n\t-particle_scale []\n\t-N\n\t-relaxation_allowance\n\t-box\n\t-fg_color [] -bg_color [] \n");
  printf("\t-min_color\n");
  printf("\t-rng_seed [] \n");
  printf("\t-randomize\n");
  printf("\t-end_mcs []\n");
  printf("\t-energy_report_frequency []\n");
  printf("\t-configuration_threshold []\n");
  printf("\t-configuration_frequency []\n");
  printf("\t-input_file_name\n");
  printf("\t-target_acceptance_ratio []\n");
  printf("\t-psi_shift\n");
  printf("\t-cutoff_sq\n\n");
 
  exit(0);
}

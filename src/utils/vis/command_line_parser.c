#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <ftw_std.h>
#include <ftw_rng.h>

#include "command_line_parser.h"

/* parameters configurable on command line and default values */
int verbose = 0;
int side_view = 0;
int particle_scale = 64; /* how many pixels */
double box_x = 4.0;
double box_y = 4.0;
double box_z = 4.0;
int fg_color = 255;
int bg_color = 0;
int min_color = 64;
char *input_file_name;

char *display_name_1 = "X-Y projection (front)";
char *display_name_2 = "Z-Y projection (right side)";

extern double sfactor;
extern double separation_threshold;
extern double diameter_threshold;

void parseCommandLineOptions(int argc, char *argv[])
{
  int i;

  for (i = 0; i<argc; i++)
  {
    if (!strcmp(argv[i], "-v")) verbose = 1;
    if (!strcmp(argv[i], "-side")) side_view = 1;
    if (!strcmp(argv[i], "-particle_scale")) particle_scale = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-box_x")) box_x = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-box_y")) box_y = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-box_z")) box_z = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-dim")) 
    {
      box_x = strtod(argv[++i], NULL);
      box_y = strtod(argv[++i], NULL);
      box_z = strtod(argv[++i], NULL);
    }
    if (!strcmp(argv[i], "-sfactor")) sfactor = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-input_file_name")) input_file_name = argv[++i];
    if (!strcmp(argv[i], "-separation_threshold")) separation_threshold = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-diameter_threshold")) diameter_threshold = strtod(argv[++i], NULL);
  }

}

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "command_line_parser.h"

extern char *input_file_name;
extern int n_bins;
extern double resolution;

void parseCommandLineOptions(int argc, char *argv[])
{
  int i;

  for (i = 0; i<argc; i++)
  {
    if (!strcmp(argv[i], "-input_file_name")) input_file_name = argv[++i];
    if (!strcmp(argv[i], "-resolution")) resolution = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-n_bins")) n_bins = strtol(argv[++i], NULL, 10);
  }
} 

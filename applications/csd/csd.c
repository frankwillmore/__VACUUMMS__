/* csd.c */
/* input file should be of .cav format */
/* output is of .hst format */

#include "csd.h"
#include "command_line_parser.h"

#include <ftw_std.h>
#include <ftw_rng.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int n_bins = 100;
double resolution = .01;
char *input_file_name;
int histogram[1000];

int main(int argc, char *argv[])
{
  FILE *datastream;
  char line[80];
  char *xs, *ys, *zs, *ds;
  double d;
  int i, which_bin;

  parseCommandLineOptions(argc, argv);
  datastream = fopen(input_file_name, "r");

  for (i=0; i<n_bins; i++) histogram[i] = 0;
  while (1)
  {
    fgets(line, 80, datastream);
    if (feof(datastream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    d = strtod(ds, NULL);
    which_bin = (int)(d/resolution);
    histogram[which_bin]++;
  }
 
  fclose(datastream);

  for (i=0; i<n_bins; i++) printf("%lf\t%d\n", i*resolution, histogram[i]);
  
  return 0;
}

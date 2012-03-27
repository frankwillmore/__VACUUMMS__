/* io_setup.c */

#include <ftw_std.h>
#include <ftw_rng.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <arpa/inet.h>
#include <unistd.h>

#include "io_setup.h"
#include "energy.h"

extern double temperature;
extern int number_of_molecules;
extern double box_x, box_y, box_z;
extern double x[], y[], z[];

extern FILE *instream;

void loadConfiguration()
{
  char line[80];
  char *xs, *ys, *zs;

  number_of_molecules = 0;

  while (1)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\n");
    
    x[number_of_molecules] = strtod(xs, NULL);
    y[number_of_molecules] = strtod(ys, NULL);
    z[number_of_molecules] = strtod(zs, NULL);
    number_of_molecules++;
  }
 
  V printf("%d lines read.\n", number_of_molecules);
  fclose(instream);
}


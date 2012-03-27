/* io_setup.c */

#include <ftw_std.h>
#include <ftw_rng.h>
#include <time.h>
#include <string.h>

#include "io_setup.h"

extern int number_of_molecules;
extern double x[], y[], z[];
double sigma[16384], sqrt_epsilon[16384];

extern FILE *instream;

void readConfiguration()
{
  char line[80];
  char *xs, *ys, *zs;
  char *sigmas, *epsilons;

  number_of_molecules = 0;

  while (1)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    sigmas = strtok(NULL, "\t");
    epsilons = strtok(NULL, "\n");

    x[number_of_molecules] = strtod(xs, NULL);
    y[number_of_molecules] = strtod(ys, NULL);
    z[number_of_molecules] = strtod(zs, NULL);
    sigma[number_of_molecules] = strtod(sigmas, NULL);
    sqrt_epsilon[number_of_molecules++] = sqrt(strtod(epsilons, NULL));
    
  }
 
  V printf("%d lines read.\n", number_of_molecules);
  fclose(instream);
}

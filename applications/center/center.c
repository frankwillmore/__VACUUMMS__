/* center.c */

// Takes a given cluster (as a set of cavities) and shifts it around until 
// it's no longer hanging out of the sim box.
// program does not check that all cavities are one cluster
// only makes sure that no pair straddles a boundary

// In:  .cav
// Out: .cav  

#define MAX_CAVITIES 1310720

#include <ftw_std.h>
#include "center.h"

extern double box_x, box_y, box_z;
extern FILE *instream;

int number_of_cavities=0;
double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], r[MAX_CAVITIES];

double shift_resolution = .1;

int main(int argc, char* argv[])
{
  double shift_x, shift_y, shift_z;
  int x_shift_okay=0, y_shift_okay=0, z_shift_okay=0;
  int i;

  parseCommandLineOptions(argc, argv);

  readInputStream();

  for (shift_x=0; shift_x < box_x; shift_x += shift_resolution)
  {
    if (checkXBoundary()==0)
    {
      x_shift_okay=1;
      break;
    }

    for (i=0; i<number_of_cavities; i++)
    {
       x[i] += shift_resolution;
       if (x[i] > box_x) x[i] -= box_x;
    }
  }

  if (x_shift_okay != 1) fprintf(stderr, "not resolved in x.\n");

  for (shift_y=0; shift_y < box_y; shift_y += shift_resolution)
  {
    if (checkYBoundary()==0)
    {
      y_shift_okay=1;
      break;
    }

    for (i=0; i<number_of_cavities; i++)
    {
       y[i] += shift_resolution;
       if (y[i] > box_y) y[i] -= box_y;
    }
  }

  if (y_shift_okay != 1) fprintf(stderr, "not resolved in y.\n");

  for (shift_z=0; shift_z < box_z; shift_z += shift_resolution)
  {
    if (checkZBoundary()==0)
    {
      z_shift_okay=1;
      break;
    }

    for (i=0; i<number_of_cavities; i++)
    {
       z[i] += shift_resolution;
       if (z[i] > box_z) z[i] -= box_z;
    }
  }

  printCavities();

  exit(x_shift_okay & y_shift_okay & z_shift_okay);
}

void printCavities()
{
  int cavity_number;
  for (cavity_number=0; cavity_number<number_of_cavities; cavity_number++)
  printf("%lf\t%lf\t%lf\t%lf\n", x[cavity_number], y[cavity_number], z[cavity_number], 2*r[cavity_number]);
}

int checkXBoundary()
{
  int i;
  for (i=0; i<number_of_cavities; i++)
    if (x[i]+r[i] > box_x || x[i]-r[i] < 0) return 1;
  return 0;
}

int checkYBoundary()
{
  int i;
  for (i=0; i<number_of_cavities; i++)
    if (y[i]+r[i] > box_y || y[i]-r[i] < 0) return 1;
  return 0;
}

int checkZBoundary()
{
  int i;
  for (i=0; i<number_of_cavities; i++)
    if (z[i]+r[i] > box_z || z[i]-r[i] < 0) return 1;
  return 0;
}

void readInputStream()
{
  char line[80];
  char *xs, *ys, *zs, *ds;

  while (TRUE)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    x[number_of_cavities] = strtod(xs, NULL);
    y[number_of_cavities] = strtod(ys, NULL);
    z[number_of_cavities] = strtod(zs, NULL);
    r[number_of_cavities] = strtod(ds, NULL)*.5;

    number_of_cavities++;

    if (number_of_cavities > MAX_CAVITIES)
    {
      printf("Too many cavities.");
      exit(0);
    }
  }
  
  V printf("%d cavities.\n", number_of_cavities);
}


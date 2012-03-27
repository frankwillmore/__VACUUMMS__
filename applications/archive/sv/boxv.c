/* boxv.c */

#define MAX_CAVITIES 1310720
#define N_SUCCESSES 10000

#include <ftw_std.h>
#include <ftw_rng2.h>
#include <ftw_param.h>

#include "boxv.h"

// Reads a centered cluster and determines the volume of it.
// Will deliver an erroneous result if cluster is not centered or percolates.

// More generally, it will give the volume of all the cavities, whether clustered or not.

// In:  one cluster; 1 or more records in .cav format 
// Out: .dst (reports one value of volume) 

extern double box_x, box_y, box_z;
FILE *instream;
extern double sfactor;

int number_of_cavities=0;
double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], d[MAX_CAVITIES];

int n_attempts=1000000;

int main(int argc, char* argv[])
{
  double tx, ty, tz;
  double volume;
  int successes=0, attempts=0;

  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getIntParam("-n_attempts", &n_attempts);
  if (getFlagParam("-randomize")) initializeRandomNumberGenerator2(-1);
  else initializeRandomNumberGenerator2(0);

  instream = stdin;
  readInputStream();

  successes=0;

  for (attempts=0; attempts<n_attempts; attempts++)
  {
    // pick a point and check inclusion

    tx = rnd2() * box_x;
    ty = rnd2() * box_y;
    tz = rnd2() * box_z;

    successes += checkInclusion(tx, ty, tz);
//    if (attempts > 100000000)
//    {
//      printf("too many attempts.");
//      exit(1);
//    }
  }

  volume = (box_x * box_y * box_z * successes)/attempts;

  printf("%lf\n", volume);
  return 0;
}

int checkInclusion(double tx, double ty, double tz)
{
  int i;
  double dx, dy, dz, dd;

  for (i=0; i<number_of_cavities; i++)
  {
    dx = x[i] - tx;
    dy = y[i] - ty;
    dz = z[i] - tz;

    dd = dx*dx + dy*dy + dz*dz;

    if (4*dd < (d[i] * d[i])) return 1;
  }
 
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
    d[number_of_cavities] = strtod(ds, NULL);

    number_of_cavities++;

    if (number_of_cavities > MAX_CAVITIES)
    {
      printf("Too many cavities.");
      exit(0);
    }
  }
  
  V printf("%d cavities.\n", number_of_cavities);
}


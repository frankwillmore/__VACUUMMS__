/* cv.c */

#define MAX_CAVITIES 1310720
#define N_SUCCESSES 10000

#include <ftw_std.h>
#include <ftw_rng2.h>
#include <ftw_param.h>

#include "cv.h"

// Reads a centered cluster and determines the volume of it.
// Will deliver an erroneous result if cluster is not centered or percolates.

// More generally, it will give the volume of all the cavities, whether clustered or not.

// In:  one cluster; 1 or more records in .cav format 
// Out: .dst (reports one value of volume) 

double box_x, box_y, box_z;
FILE *instream;

int number_of_cavities=0;
double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], d[MAX_CAVITIES];

int verbose;

int main(int argc, char* argv[])
{
  double tx, ty, tz;
  double volume;
  int successes=0, attempts=0;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("usage:  cv [ -randomize ]\n\n");
    printf("        will return MC volume of list of spheres entered, box size determined automatically.\n");
    printf("\n");
    printf("// Reads a centered cluster and determines the volume of it.\n");
    printf("// Will deliver an erroneous result if cluster is not centered or percolates.\n");
    printf("\n");
    printf("// More generally, it will give the volume of all the cavities, whether clustered or not.\n");
    printf("\n");
    printf("// In:  one cluster; 1 or more records in .cav format \n");
    printf("// Out: .dst (reports one value of volume) \n");
    exit(1);
  }
  if (getFlagParam("-randomize")) initializeRandomNumberGenerator2(-1);
  else initializeRandomNumberGenerator2(0);

  instream = stdin;
  readInputStream();

  findMinimumBoxSize();

  successes=0;

  for (attempts=0; successes<N_SUCCESSES; attempts++)
  {
    // pick a point and check inclusion

    tx = rnd2() * box_x;
    ty = rnd2() * box_y;
    tz = rnd2() * box_z;

    successes += checkInclusion(tx, ty, tz);
    if (attempts > 100000000)
    {
      printf("too many attempts.");
      exit(1);
    }
  }

  volume = (box_x * box_y * box_z * successes)/attempts;

  printf("%lf\n", volume);
  return 0;
}

void findMinimumBoxSize()
{
  int i;
  double max_x=0, max_y=0, max_z=0;
  double min_x=0, min_y=0, min_z=0;
  double test_x, test_y, test_z;
  double r;
  
  // first find lower extrema
  r = d[0] * .5;
  min_x=x[0]-r;
  min_y=y[0]-r;
  min_z=z[0]-r;

  for (i=1; i<number_of_cavities; i++)
  {
    r = d[i] * .5;

    test_x = x[i] - r;
    if (test_x < min_x) min_x = test_x;
    test_y = y[i] - r;
    if (test_y < min_y) min_y = test_y;
    test_z = z[i] - r;
    if (test_z < min_z) min_z = test_z;
  }

  for (i=0; i<number_of_cavities; i++)
  {
    x[i] -= min_x;
    y[i] -= min_y;
    z[i] -= min_z;
  }

  // now find max box size

  for (i=0; i<number_of_cavities; i++)
  {
    r = d[i] * .5;

    test_x = x[i] + r;
    if (test_x > max_x) max_x = test_x;
    test_y = y[i] + r;
    if (test_y > max_y) max_y = test_y;
    test_z = z[i] + r;
    if (test_z > max_z) max_z = test_z;
  }

  box_x = max_x;
  box_y = max_y;
  box_z = max_z;
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


/* rog.c */

#define MAX_CAVITIES 131072
#define N_POINTS 10000

#include <ftw_std.h>
#include <ftw_rng2.h>
#include <ftw_param.h>

#include "rog.h"

// Reads a centered cluster and determines the radius of gyration
// Will deliver an erroneous result if cluster is not centered or percolates.

// Mechanism:  Samples points in space to see if they are part of cluster
//             If they are, they are added to a set of points used in
//             determining the radius of gyration.

// In:  .cav 
// Out: .dst (reports one value)

double box_x, box_y, box_z;
FILE *instream;

int number_of_cavities=0;
double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], d[MAX_CAVITIES];
double px[N_POINTS], py[N_POINTS], pz[N_POINTS];
int number_of_points=0;
double cm_x, cm_y, cm_z;
double rog=0;

int main(int argc, char* argv[])
{
  int i,j;
  double dx, dy, dz;
  double distance;
  double max_distance = 0;

  instream=stdin;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage")) {
    printf(" Reads a centered cluster and determines the radius of gyration\n");
    printf(" Will deliver an erroneous result if cluster is not centered or percolates.\n");
    printf("\n");
    printf(" Mechanism:  Samples points in space to see if they are part of cluster\n");
    printf("             If they are, they are added to a set of points used in\n");
    printf("             determining the radius of gyration.\n");
    printf("\n");
    printf(" In:  .cav \n");
    printf(" Out: .dst (reports one value)\n");
    printf("\n");
    exit(0);
  }

  if (getFlagParam("-randomize")) initializeRandomNumberGenerator2(-1);
  else initializeRandomNumberGenerator2(0);

  readInputStream();
  findMinimumBoxSize();
  samplePoints();
  findCenterOfMass();
  findRadiusOfGyration();

  printf("%lf\n", rog);
}

void findMinimumBoxSize()
{
  int i;
  double max_x=0, max_y=0, max_z=0;
  double min_x=0, min_y=0, min_z=0;
  double test_x, test_y, test_z;
  double r;

  // first find lower extrema

  for (i=0; i<number_of_cavities; i++)
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
    x[i] -= test_x;
    y[i] -= test_y;
    z[i] -= test_z;
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

void findRadiusOfGyration()
{
  double dx=0, dy=0, dz=0;
  double sum = 0;
  int i;

  for (i=0; i<number_of_points; i++)
  {
    dx = px[i] - cm_x;
    dy = py[i] - cm_y;
    dz = pz[i] - cm_z;
   
    sum += (dx*dx + dy*dy + dz*dz);
  }

  rog = sqrt(sum/number_of_points);
}

void findCenterOfMass()
{
  double x_cum=0, y_cum=0, z_cum=0;
  int i;

  for (i=0; i<number_of_points; i++)
  {
    x_cum+=px[i];
    y_cum+=py[i];
    z_cum+=pz[i];
  }
 
  cm_x = x_cum / number_of_points;
  cm_y = y_cum / number_of_points;
  cm_z = z_cum / number_of_points;
}

void samplePoints()
{
  int i, j;
  double shift_x=0, shift_y=0, shift_z=0;
  double dx, dy, dz, dd;
  int attempts=0;

  while (number_of_points<N_POINTS)
  {
    // pick a point and check inclusion

    px[number_of_points] = rnd2() * box_x;
    py[number_of_points] = rnd2() * box_y;
    pz[number_of_points] = rnd2() * box_z;

    number_of_points += checkInclusion(px[number_of_points], py[number_of_points], pz[number_of_points]);
    attempts++;

    if (attempts > 100000000)
    {
      printf("too many attempts.");
      exit(1);
    }
  }
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


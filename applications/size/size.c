/* size.c */

#define MAX_CAVITIES 16384
#define N_POINTS 10000

#include <ftw_std.h>
#include <ftw_rng.h>
#include <ftw_param.h>
#include "size.h"

// Reads a centered cluster and determines the radius of gyration and span
// Will deliver an erroneous result if cluster is not centered or percolates.

// Mechanism:  Samples points in space to see if they are part of cluster
//             If they are, they are added to a set of points used in
//             determining the radius of gyration.

// In:  .cav 
// Out: %lf\t%lf\n  radius of gyration and span, values repeated proportional to volume of cluster

double box_x=7, box_y=7, box_z=7;

int number_of_cavities=0;
double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], d[MAX_CAVITIES];
double px[N_POINTS], py[N_POINTS], pz[N_POINTS];
int number_of_points=0;
double cm_x, cm_y, cm_z;
double rog=0;
double span=0;
int max_points = N_POINTS;

int main(int argc, char* argv[])
{
  int i;

  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getIntParam("-max_points", &max_points);

  if (getFlagParam("-randomize")) randomize();
  else initializeRandomNumberGenerator();

  readInputStream();
  samplePoints();
  findCenterOfMass();
  rog = findRadiusOfGyration();
  span = findSpan();

  // The following gives volume sampling by repeating the value proportional to volume of cluster
  for (i=0; i<number_of_points; i++) printf("%lf\t%lf\n", rog, span);
}

double findRadiusOfGyration()
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

  return sqrt(sum/number_of_points);
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

  V printf("center of mass:  (%lf, %lf, %lf)\n", cm_x, cm_y, cm_z);
}

void samplePoints()
{
  int i, j;
  double test_x=0, test_y=0, test_z=0;
  double dx, dy, dz, dd;

  for (i=0; i<max_points; i++)
  {
    test_x = box_x * rnd();
    test_y = box_y * rnd();
    test_z = box_z * rnd();

    // check inclusion
    for (j=0; j<number_of_cavities; j++)
    {
      dx = x[j] - test_x;
      dy = y[j] - test_y;
      dz = z[j] - test_z;
      dd = dx*dx + dy*dy + dz*dz;
      if (dd < (.25 * d[j] * d[j]))
      {
        px[number_of_points] = test_x;
        py[number_of_points] = test_y;
        pz[number_of_points] = test_z;
        number_of_points++;
        break;
      }
    }
  }
}

void readInputStream()
{
  char line[80];
  char *xs, *ys, *zs, *ds;

  FILE *instream = stdin;

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

double findSpan(int argc, char* argv[])
{
  double max_distance = 0;
  double dx, dy, dz, distance;
  int i,j;

  for (i=0; i<number_of_cavities; i++) if (d[i] > max_distance) max_distance = d[i];

  for (i=0; i<number_of_cavities-1; i++)
  {
    for (j=i+1; j<number_of_cavities; j++)
    {
      dx = x[i] - x[j];
      dy = y[i] - y[j];
      dz = z[i] - z[j];
      
      distance = sqrt(dx*dx + dy*dy + dz*dz) + .5*(d[i] + d[j]);
      if (distance > max_distance) max_distance = distance;
    }
  }

  return (max_distance);
}


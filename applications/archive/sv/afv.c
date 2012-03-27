/* afv.c - Calculates available free volume (volume included in inserted spheres in land and sea) */
/* input is .sea (set of centers and diameters) as nn.nnn\tnn.nnn\tnn.nnn\tnn.nnn\n */
/* output is one single value:  nn.nnn\n */

#define MAX_CAVITIES 8000000
#define MAX_CLOSE 100000

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ftw_param.h>
#include <math.h>

FILE *instream;

int number_of_cavities=0;
double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], d[MAX_CAVITIES];
double d_max=0, d_max_sq;

double close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE], close_d[MAX_CLOSE];
int number_of_close;
int n_centers=1000;
int n_attempts=1000;

double rnd_step=1.0/(0.0 + RAND_MAX);
double rnd_step_x, rnd_step_y, rnd_step_z;
double rnd_step_center;

double center_x, center_y, center_z;
double box_x, box_y, box_z;

void readInputStream();
inline int checkInclusion(double x, double y, double z);

int main(int argc, char* argv[])
{
  double tx, ty, tz;
  double volume;
  int successes=0, attempts=0;
  int centers;
  int i;
  double dx, dy, dz, dd;
  double zero_x, zero_y, zero_z;

  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getIntParam("-n_centers", &n_centers);
  getIntParam("-n_attempts", &n_attempts);

  if (getFlagParam("-usage")) 
  {
    printf("usage:\t-box [ 10 10 10 ]\n");
    printf("      \t-n_centers [ 1000 ]\n");
    printf("      \t-n_attempts [ 1000 ]\n");
    exit(0);
  }

  rnd_step_x=box_x*rnd_step;
  rnd_step_y=box_y*rnd_step;
  rnd_step_z=box_z*rnd_step;

srand(15);

  instream = stdin;
  readInputStream();

  successes=0;

  for (centers=0; centers<n_centers; centers++)
  {
    center_x=rand()*rnd_step_x;
    center_y=rand()*rnd_step_y;
    center_z=rand()*rnd_step_z;
  
    /* build Verlet list */
    number_of_close=0;
    for (i=0; i<number_of_cavities; i++)
    {
      dx=center_x-x[i];
      dy=center_y-y[i];
      dz=center_z-z[i];
  
      dd = dx*dx + dy*dy + dz*dz;
      if (4*dd < (d_max + d[i])*(d_max + d[i]))
      {
        close_x[number_of_close] = x[i];
        close_y[number_of_close] = y[i];
        close_z[number_of_close] = z[i];
        close_d[number_of_close] = d[i];
        number_of_close++;
      }
    }

    zero_x=center_x-(d_max*0.69336127435063470484335227478596*.5);
    zero_y=center_y-(d_max*0.69336127435063470484335227478596*.5);
    zero_z=center_z-(d_max*0.69336127435063470484335227478596*.5);
    
    if (number_of_close>0) for (attempts=0; attempts<n_attempts; attempts++)
    {
      // pick a point and check inclusion

      tx = zero_x + rand() * rnd_step_center;
      ty = zero_y + rand() * rnd_step_center;
      tz = zero_z + rand() * rnd_step_center;

      successes += checkInclusion(tx, ty, tz);
    }
  }

  volume = (box_x * box_y * box_z * successes)/(n_centers*n_attempts);

  printf("%lf\n", volume);
  return 0;
}

inline int checkInclusion(double tx, double ty, double tz)
{
  int i;
  double dx, dy, dz, dd;

  for (i=0; i<number_of_close; i++)
  {
    dx = close_x[i] - tx;
    dy = close_y[i] - ty;
    dz = close_z[i] - tz;

    dd = dx*dx + dy*dy + dz*dz;

    if (4*dd < (close_d[i] * close_d[i])) return 1;
  }
 
  return 0;
}

void readInputStream()
{
  char line[80];
  char *xs, *ys, *zs, *ds;
  double xx, yy, zz, dd;
  int i;

  while (1)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    xx = strtod(xs, NULL);
    yy = strtod(ys, NULL);
    zz = strtod(zs, NULL);
    dd = strtod(ds, NULL);
  
    if (dd>d_max) d_max = dd;

    x[number_of_cavities] = xx;
    y[number_of_cavities] = yy;
    z[number_of_cavities] = zz;
    d[number_of_cavities] = dd;
    number_of_cavities++;

    x[number_of_cavities] = xx+box_x;
    y[number_of_cavities] = yy;
    z[number_of_cavities] = zz;
    d[number_of_cavities] = dd;
    number_of_cavities++;

    x[number_of_cavities] = xx;
    y[number_of_cavities] = yy+box_y;
    z[number_of_cavities] = zz;
    d[number_of_cavities] = dd;
    number_of_cavities++;

    x[number_of_cavities] = xx+box_x;
    y[number_of_cavities] = yy+box_y;
    z[number_of_cavities] = zz;
    d[number_of_cavities] = dd;
    number_of_cavities++;

    x[number_of_cavities] = xx;
    y[number_of_cavities] = yy;
    z[number_of_cavities] = zz+box_z;
    d[number_of_cavities] = dd;
    number_of_cavities++;

    x[number_of_cavities] = xx+box_x;
    y[number_of_cavities] = yy;
    z[number_of_cavities] = zz+box_z;
    d[number_of_cavities] = dd;
    number_of_cavities++;

    x[number_of_cavities] = xx;
    y[number_of_cavities] = yy+box_y;
    z[number_of_cavities] = zz+box_z;
    d[number_of_cavities] = dd;
    number_of_cavities++;

    x[number_of_cavities] = xx+box_x;
    y[number_of_cavities] = yy+box_y;
    z[number_of_cavities] = zz+box_z;
    d[number_of_cavities] = dd;
    number_of_cavities++;

    if (number_of_cavities > MAX_CAVITIES)
    {
      printf("Too many cavities.");
      exit(1);
    }
  }

  for (i=0; i<number_of_cavities; i++)
  {
    x[i] = x[i]-(.5*box_x);
    y[i] = y[i]-(.5*box_y);
    z[i] = z[i]-(.5*box_z);
  }

  d_max_sq=d_max*d_max;

  rnd_step_center=d_max * 0.69336127435063470484335227478596 * rnd_step;
}

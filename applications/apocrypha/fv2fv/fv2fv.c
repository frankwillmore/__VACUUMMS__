/* fv2fv.c - generates a set of fv points to be fed to 2pc */
/* input is .gfg (set of lj centers and parameters) as x.xxxxxx\ty.yyyyyy\tz.zzzzzz\ts.ssssss\te.eeeeee\n */
/* output is set of {x, y, z} values:  x.xxxxxx\ty.yyyyyy\tz.zzzzzz\n */

#define MAX_ATOMS 8000000

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ftw_param.h>
#include <math.h>

FILE *instream;

int n_atoms=0;
double x[MAX_ATOMS], y[MAX_ATOMS], z[MAX_ATOMS], d_sq[MAX_ATOMS];

int n_points=1000;

double rnd_step=1.0/(0.0 + RAND_MAX);
double rnd_step_x, rnd_step_y, rnd_step_z;
double rnd_step_center;

double center_x, center_y, center_z;
double box_x, box_y, box_z;

void readInputStream();
inline int checkInclusion(double x, double y, double z);

main(int argc, char* argv[])
{
  double tx, ty, tz;
  double volume;
  int successes=0, attempts=0;
  int i;
  double dx, dy, dz, dd;
  double zero_x, zero_y, zero_z;

  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getIntParam("-n_points", &n_points);

  if (getFlagParam("-usage")) 
  {
    printf("usage:\t-box [ 10 10 10 ]\n");
    printf("      \t-n_points [ 1000 ]\n");
    printf("\n");
    printf("input is .gfg (set of lj centers and parameters) as x.xxxxxx\ty.yyyyyy\tz.zzzzzz\ts.ssssss\te.eeeeee\n");
    printf("output is set of {x, y, z} values:  x.xxxxxx\ty.yyyyyy\tz.zzzzzz\n\n"); 

    exit(0);
  }

fprintf(stderr, "box size:  %lf x %lf x %lf\n", box_x, box_y, box_z);

  rnd_step_x=box_x*rnd_step;
  rnd_step_y=box_y*rnd_step;
  rnd_step_z=box_z*rnd_step;

  srand(15);


  instream = stdin;
  readInputStream();

  while (n_points)
  {
    tx=rand()*rnd_step_x;
    ty=rand()*rnd_step_y;
    tz=rand()*rnd_step_z;

//printf("rand:  %lf\t%lf\t%lf\n", tx, ty, tz);
    
    if(!checkInclusion(tx, ty, tz))
    {
      printf("%lf\t%lf\t%lf\n", tx, ty, tz);
      n_points--;
    }
  }
}

// returns 1 for inclusion, 0 for no inclusion
checkInclusion(double tx, double ty, double tz)
{
  int i;
  double dx, dy, dz, dd;

  for (i=0; i<n_atoms; i++)
  {
    dx = x[i] - tx;
    dy = y[i] - ty;
    dz = z[i] - tz;

    dd = dx*dx + dy*dy + dz*dz;

    // return as soon as the test point hits
    if (dd < d_sq[i]) return 1;
  }
 
  // no hit, return 0
  return 0;
}

void readInputStream()
{
  char line[80];
  char *xs, *ys, *zs, *sigmas, *epsilons;
  double xx, yy, zz, dd;
  int i;

  while (1)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    sigmas = strtok(NULL, "\t");
    epsilons = strtok(NULL, "\n");

    xx = strtod(xs, NULL);
    yy = strtod(ys, NULL);
    zz = strtod(zs, NULL);
    dd = 0.25 * strtod(sigmas, NULL) * strtod(sigmas, NULL); // r-squared... its the number to beat
  
    // replicate the box in the forward cardinal directions to form the box-cloud

    x[n_atoms] = xx;
    y[n_atoms] = yy;
    z[n_atoms] = zz;
    d_sq[n_atoms] = dd;
    n_atoms++;

    x[n_atoms] = xx+box_x;
    y[n_atoms] = yy;
    z[n_atoms] = zz;
    d_sq[n_atoms] = dd;
    n_atoms++;

    x[n_atoms] = xx;
    y[n_atoms] = yy+box_y;
    z[n_atoms] = zz;
    d_sq[n_atoms] = dd;
    n_atoms++;

    x[n_atoms] = xx+box_x;
    y[n_atoms] = yy+box_y;
    z[n_atoms] = zz;
    d_sq[n_atoms] = dd;
    n_atoms++;

    x[n_atoms] = xx;
    y[n_atoms] = yy;
    z[n_atoms] = zz+box_z;
    d_sq[n_atoms] = dd;
    n_atoms++;

    x[n_atoms] = xx+box_x;
    y[n_atoms] = yy;
    z[n_atoms] = zz+box_z;
    d_sq[n_atoms] = dd;
    n_atoms++;

    x[n_atoms] = xx;
    y[n_atoms] = yy+box_y;
    z[n_atoms] = zz+box_z;
    d_sq[n_atoms] = dd;
    n_atoms++;

    x[n_atoms] = xx+box_x;
    y[n_atoms] = yy+box_y;
    z[n_atoms] = zz+box_z;
    d_sq[n_atoms] = dd;
    n_atoms++;

    if (n_atoms > MAX_ATOMS)
    {
      printf("Too many atoms.");
      exit(1);
    }
  }

  // now re-center the box-cloud so that the 0 box is in the center
  for (i=0; i<n_atoms; i++)
  {
    x[i] = x[i]-(.5*box_x);
    y[i] = y[i]-(.5*box_y);
    z[i] = z[i]-(.5*box_z);
  }
}


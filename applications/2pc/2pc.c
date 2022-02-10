/*************************************** 2pc.c ********************************************/
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <ftw_science.h>
#include <ftw_std.h>

#ifndef MAX_NUMBER_BINS
#define MAX_NUMBER_BINS 16384
#endif

#ifndef MAX_NUMBER_MOLECULES
#define MAX_NUMBER_MOLECULES 1000000
#endif

/* parameters configurable on command line and default values */
double box_x = 4.0;
double box_y = 4.0;
double box_z = 4.0;

int n_bins = 250;
double resolution = .01;

int mirror_depth = 1;

double x[MAX_NUMBER_MOLECULES], y[MAX_NUMBER_MOLECULES], z[MAX_NUMBER_MOLECULES];
int histogram[MAX_NUMBER_BINS];

int n_molecules;

int main(int argc, char *argv[])
{
  int i,j;
  int xi, yi, zi;
  char line[80];
  double xx, yy, zz;
  char *xs, *ys, *zs;
  double shift_x, shift_y, shift_z;
  double d_min, dd;
  double n_factor;

  for (i = 0; i<argc; i++)
  {
    if (!strcmp(argv[i], "-box")) 
    {
      box_x = strtod(argv[++i], NULL);
      box_y = strtod(argv[++i], NULL);
      box_z = strtod(argv[++i], NULL);
    }
    if (!strcmp(argv[i], "-resolution")) resolution = strtod(argv[++i], NULL);
    if (!strcmp(argv[i], "-n_bins")) n_bins = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-mirror_depth")) mirror_depth = strtol(argv[++i], NULL, 10);
    if (!strcmp(argv[i], "-usage"))
    {
      printf("usage:  2pc\n");
      printf("        -resolution [%lf]\n", resolution);
      printf("        -n_bins [%d]\n", n_bins);
      printf("        -mirror_depth [%d]\n", mirror_depth); 
      printf("        -box [%lf %lf %lf]\n", box_x, box_y, box_z);
      exit(0);
    }
  }

  for (i=0; i<n_bins; i++) histogram[i] = 0;

  /* read data */
  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\n");

    xx = strtod(xs, NULL);
    yy = strtod(ys, NULL);
    zz = strtod(zs, NULL);

    x[n_molecules] = xx;
    y[n_molecules] = yy;
    z[n_molecules] = zz;

    n_molecules++;
  }

//  printf("read %d lines.\n", n_molecules);

  /* generate distribution */
  for (i=0; i<n_molecules; i++)
  {
    for (j=i+1; j<n_molecules; j++)
    {
      d_min = box_x * box_y * box_z;
      for (xi=-mirror_depth; xi<mirror_depth; xi++)
      {
        shift_x = xi * box_x;
        for (yi=-mirror_depth; yi<mirror_depth; yi++)
        {
          shift_y = yi * box_y;
          for (zi=-mirror_depth; zi<mirror_depth; zi++)
          {
            shift_z = zi * box_z;
            dd = (shift_x + x[i] - x[j]) * (shift_x + x[i] - x[j]) 
               + (shift_y + y[i] - y[j]) * (shift_y + y[i] - y[j]) 
               + (shift_z + z[i] - z[j]) * (shift_z + z[i] - z[j]);
      //      if (d_min > dd) d_min = dd;
      histogram[(int)floor(sqrt(dd)/resolution)]++;
          }
        }
      }
      //histogram[(int)floor(sqrt(d_min)/resolution)]++;
    }
  }

  /* generate output */
  for (i=0; i<n_bins; i++) 
  {
    n_factor = 3.14159 * i*resolution*i*resolution * n_molecules * (n_molecules-1) *resolution;
    printf("%lf\t%lf\n", (i*resolution), (histogram[i]/n_factor));
  }

  return 0;

} /* end main */

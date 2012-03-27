/* replicate.c */

// replicates a set of cavities

#include <ftw_std.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define N_CAVITIES 250000

extern FILE *instream;

int number_of_cavities = 0;
extern double box_x, box_y, box_z;
int mirror_depth_x=1, mirror_depth_y=1, mirror_depth_z=1;

double x[N_CAVITIES], y[N_CAVITIES], z[N_CAVITIES], d[N_CAVITIES];

int main(int argc, char *argv[])
{
  char line[80];
  char *xs, *ys, *zs, *ds;
  char *color;
  int i=0;
  int xi, yi, zi;

  instream = stdin;

  if ((argc>1) && ((*argv[1]) != '-'))
  {
    instream = fopen(argv[1], "r");
  }

  parseCommandLineOptions(argc, argv);
  while (++i<argc)
  {
    if (!strcmp(argv[i], "-mirror_depth"))
    { 
       mirror_depth_x = strtol(argv[++i], NULL, 10);
       mirror_depth_y = strtol(argv[++i], NULL, 10);
       mirror_depth_z = strtol(argv[++i], NULL, 10);
    }
  }

  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    addCavity(strtod(xs, NULL), strtod(ys, NULL), strtod(zs, NULL), strtod(ds, NULL));
  }

  for (i=0; i<number_of_cavities; i++)
  for (xi=0; xi<mirror_depth_x; xi++)
  for (yi=0; yi<mirror_depth_y; yi++)
  for (zi=0; zi<mirror_depth_z; zi++)
    printf("%lf\t%lf\t%lf\t%lf\n", x[i] + xi*box_x, y[i] + yi*box_y, z[i] + zi*box_z, d[i]);
}

int addCavity(double cx, double cy, double cz, double cd)
{
  x[number_of_cavities] = cx;
  y[number_of_cavities] = cy;
  z[number_of_cavities] = cz;
  d[number_of_cavities] = cd;
  
  return ++number_of_cavities;
}

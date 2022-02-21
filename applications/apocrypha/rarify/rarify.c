/* rarify.c */

// In:  .sea
// Out: .sea 

#define MAX_CAVITIES 1310720

#include <ftw_std.h>
#include "rarify.h"

extern double box_x, box_y, box_z;

int number_of_cavities=0;

double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], d[MAX_CAVITIES];

double x_spacing, y_spacing, z_spacing;

int n_sections=100;

int main(int argc, char* argv[])
{
  int i,j;

  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  readInputStream();

  x_spacing = box_x/n_sections;
  y_spacing = box_y/n_sections;
  z_spacing = box_z/n_sections;

  for (i=0; i<number_of_cavities; i++)
  {
    int neighbor_count=0;
    for (j=0; j<number_of_cavities; j++)
    {
      neighbor_count += areNeighbors(i,j);
      if (neighbor_count > 5) break;
    }
    if (neighbor_count < 6) printf("%lf\t%lf\t%lf\t%lf\n", x[i], y[i], z[i], d[i]);
  }
}

int areNeighbors(int i, int j)
{
  if (i==j) return 0;

  double dx, dy, dz;

  dx=x[i]-x[j];
  dy=y[i]-y[j];
  dz=z[i]-z[j];
  
  if (dx*dx + dy*dy + dz*dz < x_spacing*x_spacing + .001) return 1;

  return 0;
}

void readInputStream()
{
  char line[80];
  char *xs, *ys, *zs, *ds;

  while (TRUE)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

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


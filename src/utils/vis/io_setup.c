/* io_setup.c */

#include <ftw_std.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "io_setup.h"

extern int number_of_molecules;
extern double box_x, box_y, box_z;
extern char* input_file_name;
extern double x[], y[], z[], d[];
extern int c[];

double separation_threshold=.01;
double diameter_threshold=.01;

FILE *input_file;

int loadConfiguration()
{
  char line[80];
  char *xs, *ys, *zs, *ds, *cs;
  double xx, yy, zz, dd;
  int cc;

  number_of_molecules = 0;

  while (1)
  {
    int i, exclude;
    double d_sq;
    
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\t");
    cs = strtok(NULL, "\n");

    xx = strtod(xs, NULL);
    yy = strtod(ys, NULL);
    zz = strtod(zs, NULL);
    dd = strtod(ds, NULL);
    cc = strtol(cs, NULL, 10);

    exclude = 0;
    if (dd < diameter_threshold) exclude = 1;
    for (i=0; (exclude == 0) && (i<number_of_molecules); i++)
    {
      d_sq = (xx - x[i]) * (xx - x[i]) + (yy - y[i]) * (yy - y[i]) + (zz - z[i]) * (zz - z[i]);
      if (d_sq < (separation_threshold * separation_threshold)) exclude = 1;
    }

    if (exclude == 1) continue;

    x[number_of_molecules] = xx;
    y[number_of_molecules] = yy;
    z[number_of_molecules] = zz;
    d[number_of_molecules] = dd;
    c[number_of_molecules] = cc;
 
    number_of_molecules++;
  }
 
  return number_of_molecules;
}

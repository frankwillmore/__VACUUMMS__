/* end2end.c */

#define MAX_CAVITIES 131072

#include <stdio.h>
#include <ftw_std.h>
#include "end2end.h"

// Reads a centered cluster and determines the end-to-end distance (span) of it.
// Will deliver an erroneous result if cluster is not centered or percolates.

// In:  .cav 
// Out: .dst (reports one value)

//double box_x, box_y, box_z;
//extern FILE *instream;
FILE *instream;

int number_of_cavities=0;
double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], d[MAX_CAVITIES];

int main(int argc, char* argv[])
{
  int i,j;
  double dx, dy, dz;
  double distance;
  double max_distance = 0;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    //printf("end2end    \t-box [ xx.xxx yy.yyy zz.zzz ]\n");
    printf("end2end < cluster.cav  \n");
    printf("Reads a centered cluster and determines the end-to-end distance (span) of it.\n");
    printf("Will deliver an erroneous result if cluster is not centered or percolates.\n");
    printf("\n");
    printf("In:  .cav \n");
    printf("Out: .dst (reports one value)\n");
    printf("\n");
    printf("\n");
    exit(0);
  }
//  getVectorParam("-box", &box_x, &box_y, &box_z);
  instream=stdin;

  readInputStream();

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

  printf("%lf\n", max_distance);
}

void readInputStream()
{
  char line[256];
  char *xs, *ys, *zs, *ds;

  while (TRUE)
  {
    fgets(line, 256, instream);
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


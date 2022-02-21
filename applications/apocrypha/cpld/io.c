/* io.c */

#include <stdio.h>
#include <string.h>

#include "io.h"
#include "ftw_std.h"

extern int n_cavities;
extern double x[], y[], z[], d[];

int readCavityData(FILE *instream)
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

    x[n_cavities] = strtod(xs, NULL);
    y[n_cavities] = strtod(ys, NULL);
    z[n_cavities] = strtod(zs, NULL);
    d[n_cavities] = strtod(ds, NULL); 

    n_cavities++;
  }
 
  fclose(instream);
  
  V printf("%d lines read...\n", n_cavities);
  return n_cavities;
}


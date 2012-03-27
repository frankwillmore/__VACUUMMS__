/* uniq.c */

#include <ftw_param.h>

#define MAX 1000000

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

double x[MAX], y[MAX], z[MAX], d[MAX];
double threshold = .01;
double min_diam = 0.0;

FILE *instream;
double box_x, box_y, box_z;

int main(int argc, char *argv[])
{
  char line[80];
  char *xs, *ys, *zs, *ds;
  double xx, yy, zz, dd, dsq;
  double shift_x, shift_y, shift_z;
  int i, n_cavities = 0;

  instream=stdin;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage")) 
  {
    printf("\n");
    printf("usage:\t-box [ 10.0 10.0 10.0 ]\n");  
    printf("      \t-threshold [ .01 ]\n");  
    printf("      \t-min_diam [ 0.0 ]\n\n");
    exit(0);
  }

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-threshold", &threshold);
  getDoubleParam("-min_diam", &min_diam);

  while (1)
  {
    int exclude;

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

    if (dd < min_diam) exclude=1;
    else
    {
      exclude = 0;
      for (i=0;i<n_cavities && !exclude; i++)
      {
        for (shift_x=-box_x; shift_x<=box_x; shift_x += box_x)
        for (shift_y=-box_y; shift_y<=box_y; shift_y += box_y)
        for (shift_z=-box_z; shift_z<=box_z; shift_z += box_z)
        {
          dsq = (shift_x + xx - x[i]) * (shift_x + xx - x[i])
              + (shift_y + yy - y[i]) * (shift_y + yy - y[i]) 
              + (shift_z + zz - z[i]) * (shift_z + zz - z[i]);
          if (dsq < threshold)
          {
            exclude = 1;
            if (dd > d[i])
            {
              x[i] = xx;
              y[i] = yy;
              z[i] = zz;
              d[i] = dd; 
            }
            break; // no need to keep looking, we know it's a duplicate
          }
        }
      }
    }

    if (!exclude)
    {
      x[n_cavities] = xx;
      y[n_cavities] = yy;
      z[n_cavities] = zz;
      d[n_cavities] = dd;
      n_cavities++;
    }
  }
  
  fclose(instream);

  for (i=0; i<n_cavities; i++) printf("%lf\t%lf\t%lf\t%lf\n", x[i], y[i], z[i], d[i]);
}

/* cull.c */

#include <command_line_parser.h>

#define MAX 1000000

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

double x[MAX], y[MAX], z[MAX], d[MAX];
int include[MAX];
double probe_diameter = 1.0;

FILE *instream;
extern double box_x, box_y, box_z;

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
    printf("      \t-probe_diameter [ 1.0 ]\n");  
    exit(0);
  }

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-probe_diameter", &probe_diameter);
  getDoubleParam("-min_diam", &min_diam);

  n_cavities=-1;
  while (n_cavities++)
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
    include[n_cavities] = 1;
  }
  
  fclose(instream);

int checkDistance(int A, int B)
{
      for (i=0;i<n_cavities && !exclude; i++)
      {
        for (shift_x=0; shift_x<=box_x; shift_x += box_x)
        for (shift_y=0; shift_y<=box_y; shift_y += box_y)
        for (shift_z=0; shift_z<=box_z; shift_z += box_z)
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

  for (i=0; i<n_cavities; i++) printf("%lf\t%lf\t%lf\t%lf\n", x[i], y[i], z[i], d[i]);
}

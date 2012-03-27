/* smooth.c */

/* input:	.hst (histogram) */
/* output:	.hst (histogram, smoothed) */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <ftw_std.h>

FILE *instream;
int n_vals=0;
double x[25000], y[25000], yy[25000];

int all_bins[1024];

int main(int argc, char *argv[])
{
  char line[80];
  char *xs, *ys;
  int which_bin;
  int i=0;

  instream = stdin;

  while (TRUE)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\n");
    x[n_vals] = strtod(xs, NULL);
    y[n_vals] = strtod(ys, NULL);

    n_vals++;
  }

  fclose(instream);
 
  for (i=1; i<n_vals-1; i++) yy[i] = .3333333333 * (y[i-1] + y[i] + y[i+1]);
  yy[0] = .333333333 * (2*y[0] + y[1]); 
  yy[n_vals-1] = .3333333333 * (2*y[n_vals-1] + y[n_vals-2]);

  for (i = 0; i<n_vals; i++) printf("%lf\t%lf\n", x[i], yy[i]);
}

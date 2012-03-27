/* normalize.c */

/* input:	.hst (histogram) */
/* output:	.hst (histogram, smoothed) */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <ftw_std.h>

FILE *instream;
int n_vals=0;
double x[1000], y[1000];

int all_bins[1024];

int main(int argc, char *argv[])
{
  char line[80];
  char *xs, *ys;
  int which_bin;
  int i=0;
  double y_sum=0;

  instream = stdin;

  while (TRUE)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\n");
    x[n_vals] = strtod(xs, NULL);
    y[n_vals] = strtod(ys, NULL);

    y_sum += y[n_vals];

    n_vals++;
  }

  fclose(instream);
 
  for (i = 0; i<n_vals; i++) printf("%lf\t%lf\n", x[i], y[i]/y_sum);
}

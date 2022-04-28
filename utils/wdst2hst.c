/* wdst2hst */

/* input:	.wdst (list of values, weights) */
/* output:	.hst (histogram) */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <ftw_std.h>
#include <ftw_param.h>

FILE *instream;
double width=1;
double start_x=0, end_x;
int n_bins=64;
int normalize=0;

double all_bins[1024];

int main(int argc, char *argv[])
{
  char line[80];
  char *xs, *ws;
  double x, w;
  int which_bin;
  int i=0;
  double sum_of_weights=0;

  instream = stdin;

  while (++i < argc)
  {
    if ((argc>1) && ((*argv[i]) != '-')) instream = fopen(argv[i], "r");
    else if (!strcmp(argv[i], "-width")) width = getDoubleParameter("width", argv[++i]);
    else if (!strcmp(argv[i], "-n_bins")) n_bins = getIntParameter("n_bins", argv[++i]);
    else if (!strcmp(argv[i], "-start_x")) start_x = getDoubleParameter("start_x", argv[++i]);
    else if (!strcmp(argv[i], "-v")) verbose = getIntParameter("verbose", argv[++i]);
    else if (!strcmp(argv[i], "-normalize")) normalize = getFlagParam("-normalize");
  }

  end_x = start_x + n_bins * width;

  while (1)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ws = strtok(NULL, "\n");
    x = strtod(xs, NULL);
    w = strtod(ws, NULL);

    if ((x < start_x) || (x > end_x)) fprintf(stderr, "%lf outside of range, discarding...\n", x);

    else
    {
      which_bin = (int)floor((x - start_x)/width);
      all_bins[which_bin]+=w;
      sum_of_weights+=w;
    }

  }

  if (!normalize) sum_of_weights = 1.0;

  for (which_bin = 0; which_bin<n_bins; which_bin++) printf("%lf\t%lf\n", start_x + which_bin*width, all_bins[which_bin]/sum_of_weights);
}

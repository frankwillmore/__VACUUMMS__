/* dst2hst */

/* input:	.dst (list of values) */
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

int all_bins[1024];

int main(int argc, char *argv[])
{
  char line[80];
  char *xs;
  double x;
  int which_bin;
  int i=0;

  instream = stdin;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage")){
    printf("usage:     dst2hst    -width [1.0]\n");
    printf("                      -n_bins [64]\n");
    printf("                      -start_x [0.0]\n");
    printf("                      \n");
    exit(0);
  }

  while (++i < argc)
  {
    if ((argc>1) && ((*argv[i]) != '-')) instream = fopen(argv[i], "r");
    else if (!strcmp(argv[i], "-width")) width = getDoubleParameter("width", argv[++i]);
    else if (!strcmp(argv[i], "-n_bins")) n_bins = getIntParameter("n_bins", argv[++i]);
    else if (!strcmp(argv[i], "-start_x")) start_x = getDoubleParameter("start_x", argv[++i]);
  }

  end_x = start_x + n_bins * width;

  while (1)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\n");
    x = strtod(xs, NULL);

    if ((x < start_x) || (x > end_x)) fprintf(stderr, "%lf outside of range, discarding...\n", x);

    else
    {
      which_bin = (int)floor((x - start_x)/width);
      all_bins[which_bin]++;
    }

  }

  for (which_bin = 0; which_bin<n_bins; which_bin++) printf("%lf\t%d\n", start_x + which_bin*width, all_bins[which_bin]);
}

/* cpld.c */

#include "ftw_std.h"

#include "cpld.h"
#include "io.h"
#include "edges.h"

#define MAX_CAVITIES 10000

int verbose;
double box_x=6, box_y=6, box_z=6;
double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], d[MAX_CAVITIES];
int connectivity_level[MAX_CAVITIES];
int n_cavities=0;
int mirror_depth=1;
FILE *instream;
extern int histogram[];
int n_bins = 50;
double sfactor = 1;

int main(int argc, char *argv[])
{
  int i=0;

  instream = stdin;

  while (++i<argc)
  {
    if ((argc>1) && ((*argv[i]) != '-')) instream = fopen(argv[i], "r");
    else if (!strcmp(argv[i], "-v")) verbose = 1;
    else if (!strcmp(argv[i], "-box_x")) box_x = getDoubleParameter("box_x", argv[++i]);
    else if (!strcmp(argv[i], "-sfactor")) sfactor = getDoubleParameter("sfactor", argv[++i]);
    else if (!strcmp(argv[i], "-mirror_depth")) mirror_depth = getIntParameter("mirror_depth", argv[++i]);
    else if (!strcmp(argv[i], "-n_bins")) n_bins = getIntParameter("n_bins", argv[++i]);
    else if (!strcmp(argv[i], "-box")) 
    {
      box_x = getDoubleParameter("box_x", argv[++i]);
      box_y = getDoubleParameter("box_y", argv[++i]);
      box_z = getDoubleParameter("box_z", argv[++i]);
    }
    else 
    {
      printf("option %s not understood.\n", argv[i]);
      exit(0);
    }
  }

  readCavityData(instream);
  findAllEdges();
  getConnectivityLevels();
  for (i=0; i<n_bins; i++) printf("%d\t%d\n", i, histogram[i]);

} 

double getDoubleParameter(char *param, char *value)
{
  V printf("%s=%s\n", param, value);
  return strtod(value, NULL);
}

int getIntParameter(char *param, char *value)
{
  V printf("%s=%s\n", param, value);
  return strtol(value, NULL, 10);
}


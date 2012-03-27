/* pack.c - take a gfg and pack it into the periodic cube */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ftw_param.h>
#include <math.h>

FILE *instream;

double box_x, box_y, box_z;

int main(int argc, char* argv[])
{
  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  if (getFlagParam("-usage"))
  {
    printf(" /* pack.c - take a gfg and pack it into the periodic cube */ ");
    exit(0);
  }

  instream = stdin;

  char line[80];
  char *xs, *ys, *zs, *sigmas, *epsilons;
  double xx, yy, zz, sigma, epsilon;
  int i;

  while (1)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    sigmas = strtok(NULL, "\t");
    epsilons = strtok(NULL, "\n");

    xx = strtod(xs, NULL);
    yy = strtod(ys, NULL);
    zz = strtod(zs, NULL);
    sigma = strtod(sigmas, NULL);
    epsilon = strtod(epsilons, NULL);
  
    while (xx >= box_x) xx -= box_x;
    while (yy >= box_y) yy -= box_y;
    while (zz >= box_z) zz -= box_z;

    while (xx < 0) xx += box_x;
    while (yy < 0) yy += box_y;
    while (zz < 0) zz += box_z;

    printf("%lf\t%lf\t%lf\t%lf\t%lf\n", xx, yy, zz, sigma, epsilon);
  }
}

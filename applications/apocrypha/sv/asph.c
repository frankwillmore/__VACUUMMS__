/* asph.c */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <ftw_param.h>

int main(int argc, char* argv[])
{
  char line[80];
  char *cns, *rgs, *sas, *vs;
  double rg, sa, v;
  double aspherical_parameter;
  double factor;
  double correction = 0;

  setCommandLineParameters(argc, argv);
  getDoubleParam("-correction", &correction);

  factor = sqrt(5.0/27.0);

  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    cns = strtok(line, "\t");
    rgs = strtok(NULL, "\t");
    sas = strtok(NULL, "\t");
    vs = strtok(NULL, "\n");

    rg = strtod(rgs, NULL);
    sa = strtod(sas, NULL);
    v = strtod(vs, NULL);

    aspherical_parameter = correction + factor*rg*sa/v;

    printf("%lf\n", aspherical_parameter);
  }

  return 0;
}


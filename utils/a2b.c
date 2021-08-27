/* a2b.c */

// In:  2 files of .cfg format, same number of lines
// Out: a list of distances from pt A (1st file) to pt B (2nd file)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

FILE *instreamA, *instreamB;

int main(int argc, char *argv[])
{
  char line[80];
  double xA, yA, zA, xB, yB, zB;
  double dx, dy, dz;

  instreamA = fopen(argv[1], "r");
  instreamB = fopen(argv[2], "r");

  while (1)
  {
    fgets(line, 80, instreamA);
    if (feof(instreamA)) break;

    xA = strtod(strtok(line, "\t"), NULL);
    yA = strtod(strtok(NULL, "\t"), NULL);
    zA = strtod(strtok(NULL, "\n"), NULL);

    fgets(line, 80, instreamB);
    if (feof(instreamB)) break;

    xB = strtod(strtok(line, "\t"), NULL);
    yB = strtod(strtok(NULL, "\t"), NULL);
    zB = strtod(strtok(NULL, "\n"), NULL);

    dx = xB - xA;
    dy = yB - yA;
    dz = zB - zA;

    printf("%lf\n", sqrt(dx*dx + dy*dy + dz*dz));
  }

  fclose(instreamA);
  fclose(instreamB);
}

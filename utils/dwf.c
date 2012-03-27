/* dwf.c */

/*************************************************************

  For calculating Debye-Waller factor 
  
  IN:   specify two gfg files and box size on commmand line
  OUT:  list of squares of displacements 

**************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ftw_param.h>

char *file1;
char *file2;

double box_x, box_y, box_z;

int main(int argc, char *argv[])
{
  char line[256];
  char *xs, *ys, *zs, *ds, *es;
  double dx, dy, dz, dd;
  double x1, x2, y1, y2, z1, z2;

  FILE *instream1, *instream2;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("usage:      For calculating Debye-Waller factor\n");
    printf("            IN:   specify two gfg files and box size on commmand line\n");
    printf("            OUT:  list of squares of displacements\n");
    printf("            \n");
    printf("		-box [0.0] [0.0] [0.0]\n");
    printf("		-file1 []\n");
    printf("		-file2 []\n");
    printf("\n");
    exit(0);
  }

  if (!getFlagParam("-box"))
  {
    printf("need box dimensions.\n");
    exit(0);
  }
  else (getVectorParam("-box", &box_x, &box_y, &box_z));

  if (!getFlagParam("-file1"))
  {
    printf("need file1.\n");
    exit(0);
  }
  else getStringParam("-file1", &file1);

  if (!getFlagParam("-file2"))
  {
    printf("need file2.\n");
    exit(0);
  }
  else getStringParam("-file2", &file2);

  instream1 = fopen(file1, "r");
  instream2 = fopen(file2, "r");

  fprintf(stderr, "reading %s to %s for box %lfx%lfx%lf\n", file1, file2, box_x, box_y, box_z);

  while (1)
  {
    fgets(line, 256, instream1);
    if (feof(instream1)) break;
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\t");
    es = strtok(NULL, "\n");

    x1 = strtod(xs, NULL);
    y1 = strtod(ys, NULL);
    z1 = strtod(zs, NULL);

    fgets(line, 256, instream2);
    if (feof(instream2)) break;
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\t");
    es = strtok(NULL, "\n");

    x2 = strtod(xs, NULL);
    y2 = strtod(ys, NULL);
    z2 = strtod(zs, NULL);

    dx = x2 - x1; 
    if (dx > (0.5 * box_x)) dx -= box_x;
    if (dx < (-0.5 * box_x)) dx += box_x;
 
    dy = y2 - y1; 
    if (dy > (0.5 * box_y)) dy -= box_y;
    if (dy < (-0.5 * box_y)) dy += box_y;
 
    dz = z2 - z1; 
    if (dz > (0.5 * box_z)) dz -= box_z;
    if (dz < (-0.5 * box_z)) dz += box_z;

    dd = dx*dx + dy*dy + dz*dz;

    // printf("%lf\t%lf\t%lf\t%lf\n", dx, dy, dz, dd);
    printf("%lf\n", dd);
  }

  fclose(instream1);
  fclose(instream2);
}

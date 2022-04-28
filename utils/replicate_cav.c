/* replicate_cav.c */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ftw_config_parser.h>
#include <ftw_param.h>
#include <ftw_types.h>

double box_x, box_y, box_z;

// IN:	input stream of gfg from stdin
// OUT: output stream of replicas
int main(int argc, char* argv[])
{
  int i, j, k;
  int mirror_depth;
  char line[256];
  double x, y, z;
  double box_x, box_y, box_z;

  setCommandLineParameters(argc, argv);
  if (!getFlagParam("-box")) {
    printf("No box specified.\n");
    return(1);
  }
  else getVectorParam("-box", &box_x, &box_y, &box_z);
  getIntParam("-mirror_depth", &mirror_depth);

  while (1)
  {
    fgets(line, 256, stdin);
    if (feof(stdin)) break;
    
    char *xs = strtok(line, "\t");
    char *ys = strtok(NULL, "\t");
    char *zs = strtok(NULL, "\t");
    char *ds = strtok(NULL, "\n");

    double x = strtod(xs, NULL);
    double y = strtod(ys, NULL);
    double z = strtod(zs, NULL);

    for (i=0;i<mirror_depth;i++) for (j=0;j<mirror_depth;j++) for (k=0;k<mirror_depth;k++) printf("%lf\t%lf\t%lf\t%s\n", x + i * box_x, y + j * box_y, z + k * box_z, ds);
  }
 
  fclose(stdin);

}



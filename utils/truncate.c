/* truncate.c */

#include <ftw_std.h>
#include <ftw_param.h>

// Mechanism:  Reads a stream of data, and interprets first three fields as x, y, z. 
//             Any records with x,y,z outside of specified box are discarded.
//             All others are written to standard output.

int main(int argc, char* argv[])
{
  double min_x=0, min_y=0, min_z=0;
  double max_x=0, max_y=0, max_z=0;
  double x, y, z;
  char line[256];
  char *xs, *ys, *zs, *sobrante;

  setCommandLineParameters(argc, argv);
  getVectorParam("-min", &min_x, &min_y, &min_z);
  getVectorParam("-max", &max_x, &max_y, &max_z);

  if (getFlagParam("-usage"))
  {
    printf("\n");
    printf("usage:  truncate\t-min [ nn.nnn nn.nnn nn.nnn ]\n");
    printf("                \t-max [ nn.nnn nn.nnn nn.nnn ]\n\n");
    exit(0);
  }

  while (TRUE)
  {
    fgets(line, 256, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    sobrante = strtok(NULL, "\n");

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);

    if (x < min_x || x > max_x || y < min_y || y > max_y || z < min_z || z > max_z) continue;
    
    printf("%lf\t%lf\t%lf", x, y, z);
    if (sobrante == NULL) printf("\n");
    else printf("\t%s\n", sobrante);
  }
}


/* replicate_gfg.c */

// replicates a set of atoms

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ftw_param.h>
#include <ftw_std.h>

int main(int argc, char *argv[]) {
  char line[256];
  char *xs, *ys, *zs, *sigmas, *epsilons;
  double x, y, z;
  //int i=0;
  int xi, yi, zi;
  double box_x, box_y, box_z;
  int mirror_depth = 1;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("usage:      Replicates a periodic image to the depth specified\n");
    printf("            IN:   atoms as gfg\n");
    printf("            OUT:  atoms as gfg, with replicated images\n");
    printf("            \n");
    printf("		-box [ 0.0 ] [ 0.0 ] [ 0.0 ] \n");
    printf("		-mirror_depth [ 1 ] \n");
    printf("\n");
    exit(0);
  }

  getIntParam("-mirror_depth", &mirror_depth);
  getVectorParam("-box", &box_x, &box_y, &box_z);

  while (1) {
    fgets(line, 256, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    sigmas = strtok(NULL, "\t");
    epsilons = strtok(NULL, "\n");

    x = strtod(xs, NULL); 
    y = strtod(ys, NULL); 
    z = strtod(zs, NULL); 

    for (xi = 0; xi < mirror_depth; xi++)
      for (yi = 0; yi < mirror_depth; yi++)
	for (zi = 0; zi < mirror_depth; zi++)
          printf("%lf\t%lf\t%lf\t%s\t%s\n", x + box_x * xi, y + box_y * yi, z + box_z * zi, sigmas, epsilons);
  }

  return 0;
}


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
main(int argc, char* argv[])
{
  int i;
  char line[256];
  setCommandLineParameters(argc, argv);
  ftw_GFG65536 *gfg = readGFG65536(stdin);
  if (!getFlagParam("-box")) {
    printf("No box specified.\n");
  }
  else getVectorParam("-box", &box_x, &box_y, &box_z);
  gfg->box_x = box_x;
  gfg->box_y = box_y;
  gfg->box_z = box_z;

  ftw_GFG65536 *r_gfg = replicateGFG65536(gfg);
  for (i=0; i<r_gfg->n_atoms; i++) printf("%f\t%f\t%f\t%f\t%f\n", r_gfg->atom[i].x, r_gfg->atom[i].y, r_gfg->atom[i].z, r_gfg->atom[i].sigma, r_gfg->atom[i].epsilon);
  free(gfg);
  free(r_gfg);

  while (TRUE)
  {
    fgets(line, 256, stdin);
    if (feof(stdin)) break;
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);

    for (i=0;i<mirror_depth;i++) for (j=0;j<mirror_depth;j++) for (k=0;k<mirror_depth;k++) printf("%lf\t%lf\t%lf\t%s\n", x + i * box_x, y + j * box_y, z + k * box_z, ds);
  }
 
  fclose(stdin);

}



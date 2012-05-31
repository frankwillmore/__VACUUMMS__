
/* gfg2pov.c */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ftw_param.h>
#include <ftw_pov.h>


int main(int argc, char *argv[])
{
  char line[256];
  char *xs, *ys, *zs, *ds, *es;
  double x, y, z, d, e;
  char *color="White";
  double box_x=10, box_y=10, box_z=10;
  double transmit = 0.0;
  char transmit_str[256];
  double phong = 0.0;
  char phong_str[256];
  char clip_str[256];

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("usage:       gfg2pov     -transmit [ 0.0 ] \n");
    printf("                         -phong [ 0.0 ] \n");
    printf("                         -color [ White ] \n");
    printf("                         -clip \n");
  }
  writePOVHeader();

  getVectorParam("-box", &box_x, &box_y, &box_z);

  if (getFlagParam("-transmit")) // sets up transmit string
  {
    getDoubleParam("-transmit", &transmit);
    sprintf(transmit_str, " transmit %lf ", transmit);
  }
  if (getFlagParam("-phong")) // sets up phong string
  {
    getDoubleParam("-phong", &phong);
    sprintf(phong_str, " finish {phong %lf} ", phong);
  }
  if (getFlagParam("-clip")) // sets up clipping string
  {
    sprintf(clip_str, " clipped_by { box{ <0,0,0> <%lf, %lf, %lf>}}", box_x, box_y, box_z);
  }
  getStringParam("-color", &color);

  printf("// begin gfg2pov records\n");
  while (1) // loop over all lines
  {
    fgets(line, 256, stdin);
    if (feof(stdin)) break;
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\t");
    es = strtok(NULL, "\n");

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);
    d = strtod(ds, NULL);
    e = strtod(es, NULL);

    /* we use diameter, pov uses radius... */
    d *= .5;

    printf("sphere{<%lf, %lf, %lf>, %lf texture{ pigment {color %s %s } %s } %s }\n", x, y, z, d, color, transmit_str, phong_str, clip_str);
  }
 
  fclose(stdin);
  return 0;
}


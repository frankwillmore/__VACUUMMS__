/* gfgc2pov.c */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ftw_param.h>
#include <ftw_pov.h>

double box_x=10, box_y=10, box_z=10;
double transmit = 0.0;
double blob_threshold = 0.65;
char transmit_str[256];
double phong = 0.0;
char phong_str[256];
char *color = "White";

int main(int argc, char *argv[])
{
  char line[256];
  char *xs, *ys, *zs, *ds, *es;
  double x, y, z, d, e;
  double radius;
  char *color="White";

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("usage:       gfgc2pov    -transmit [ 0.0 ] \n");
    printf("                         -phong [ 0.0 ] \n");
    printf("                         -clip \n");
    printf("                         -blob \n");
    printf("                         -blob_threshold 0.65 \n");
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

  if (getFlagParam("-blob")) 
  {
    getDoubleParam("-blob_threshold", &blob_threshold);
    if (getFlagParam("-clip")) printf("intersection { box {<0,0,0>< %lf, %lf, %lf>} ", box_x, box_y, box_z);
    printf("blob { threshold %lf \n", blob_threshold);
  }

  printf("// begin gfgc2pov records\n");
  while (1) // loop over all lines
  {
    fgets(line, 256, stdin);
    if (feof(stdin)) break;
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\t");
    es = strtok(NULL, "\t");
    color = strtok(NULL, "\n");

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);
    d = strtod(ds, NULL);
    e = strtod(es, NULL);

    /* we use diameter, pov uses radius... */
    radius = d * 0.5;

    if (getFlagParam("-blob")) {
      // express atom as blob contribution
      printf("sphere{<%lf, %lf, %lf>, %lf %lf texture{ pigment {color %s %s } %s }}\n", x, y, z, radius, 1.0, color, transmit_str, phong_str);  // 1.0 is relative contribution to blob
    }
    else if (getFlagParam("-clip")) printf("intersection {sphere{<%lf, %lf, %lf>, %lf} box {<0,0,0>< %lf, %lf, %lf>} texture{ pigment {color %s %s } %s }}\n", 
                                      x, y, z, radius, box_x, box_y, box_z, color, transmit_str, phong_str);
    else printf("sphere{<%lf, %lf, %lf>, %lf texture{ pigment {color %s %s } %s } }\n", x, y, z, d, color, transmit_str, phong_str);
  }

  if (getFlagParam("-blob")) 
  {
    printf("} \n");
    if (getFlagParam("-clip")) printf(" texture{ pigment {color %s %s } }}\n", color, transmit_str);
  }

  fclose(stdin);
  return 0;
}


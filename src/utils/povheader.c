/* povheader.c */

#include <stdlib.h>
#include <stdio.h>
#include <ftw_param.h>
#include <string.h>

double camera_x=40, camera_y=40, camera_z=40;
double light_source_x=25, light_source_y=25, light_source_z=25;

int number_of_molecules = 0;
double box_x=10, box_y=10, box_z=10;
double ambient_light = 3.0;

main(int argc, char *argv[])
{
//  int i=0;
//  char line[256];
//  char *xs, *ys, *zs, *ds, *es, *as;
//  char *color = "Red";
//  double x, y, z, d, e, c, a;
  char *light_color = "White";
  char *box_color = "Yellow";
  char *background = "Black";

  setCommandLineParameters(argc, argv);
  getVectorParam("-camera", &camera_x, &camera_y, &camera_z);
  getVectorParam("-light_source", &light_source_x, &light_source_y, &light_source_z);
  getStringParam("-light_color", &light_color);
  getStringParam("-background", &background);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-ambient_light", &ambient_light);
  if (getFlagParam("-usage"))
  {
    printf("usage:  -show_box\n");
    printf("        -light_source [ 25.0 25.0 25.0 ]\n");
    printf("        -light_color [ White ]\n");
    printf("        -standard_light\n");
    printf("        -camera [ 40.0 40.0 40.0 ]\n");
    printf("        -box [ 10.0 10.0 10.0 ]\n");
    printf("        -box_color [ Yellow ]\n");
    printf("        -background [ Black ]\n");
    printf("        -ambient_light [ 3.0 ]\n");
    printf("NOTE:  povheader is deprecated... the new pov utils include header by default.\n");
    printf("\n");
    exit(0);
  }

  printf("#include \"colors.inc\"\n");
  printf("background {color %s}\n", background);
  printf("camera{location<%lf,%lf,%lf> look_at <0,0,0>}\n", camera_x, camera_y, camera_z);

  if (getFlagParam("-ambient_light")) printf("global_settings { ambient_light rgb <%lf,%lf,%lf> }\n", ambient_light, ambient_light, ambient_light);
  if (getFlagParam("-standard_light"))
  {
    printf("light_source{<100,0,0> color %s}\n", light_color);
    printf("light_source{<0,100,0> color %s}\n", light_color);
    printf("light_source{<0,0,100> color %s}\n", light_color);
    printf("light_source{<-100,0,0> color %s}\n", light_color);
    printf("light_source{<0,-100,0> color %s}\n", light_color);
    printf("light_source{<0,0,-100> color %s}\n", light_color);
  }
  if (getFlagParam("-light_source")) printf("light_source{<%lf,%lf,%lf> color %s}\n", light_source_x, light_source_y, light_source_z, light_color);

  if (getFlagParam("-show_box"))
  {
    printf("cylinder { <0,0,0>, <%lf,0,0>, 0.1 open texture { pigment { color %s } }}\n", box_x, box_color);
    printf("cylinder { <0,0,0>, <0,%lf,0>, 0.1 open texture { pigment { color %s } }}\n", box_y, box_color);
    printf("cylinder { <0,0,0>, <0,0,%lf>, 0.1 open texture { pigment { color %s } }}\n", box_z, box_color);
    printf("cylinder { <%lf,%lf,0>, <%lf,0,0>, 0.1 open texture { pigment { color %s } }}\n", box_x, box_y, box_x, box_color);
    printf("cylinder { <%lf,%lf,0>, <0,%lf,0>, 0.1 open texture { pigment { color %s } }}\n", box_x, box_y, box_y, box_color);
    printf("cylinder { <%lf,%lf,0>, <%lf,%lf,%lf>, 0.1 open texture { pigment { color %s } }}\n", box_x, box_y, box_x, box_y, box_z, box_color);
    printf("cylinder { <0,%lf,%lf>, <0,%lf,0>, 0.1 open texture { pigment { color %s } }}\n", box_y, box_z, box_y, box_color);
    printf("cylinder { <0,%lf,%lf>, <%lf,%lf,%lf>, 0.1 open texture { pigment { color %s } }}\n", box_y, box_z, box_x, box_y, box_z, box_color);
    printf("cylinder { <0,%lf,%lf>, <0,0,%lf>, 0.1 open texture { pigment { color %s } }}\n", box_y, box_z, box_z, box_color);
    printf("cylinder { <%lf,0,%lf>, <%lf,0,0>, 0.1 open texture { pigment { color %s } }}\n", box_x, box_z, box_x, box_color);
    printf("cylinder { <%lf,0,%lf>, <0,0,%lf>, 0.1 open texture { pigment { color %s } }}\n", box_x, box_z, box_z, box_color);
    printf("cylinder { <%lf,0,%lf>, <%lf,%lf,%lf>, 0.1 open texture { pigment { color %s } }}\n", box_x, box_z, box_x, box_y, box_z, box_color);

    printf("sphere{<0, 0, 0>, .1 texture{ pigment {color %s}}}\n", box_color);
    printf("sphere{<0, 0, %lf>, .1 texture{ pigment {color %s}}}\n", box_z, box_color);
    printf("sphere{<0, %lf ,0>, .1 texture{ pigment {color %s}}}\n", box_y, box_color);
    printf("sphere{<0, %lf ,%lf>, .1 texture{ pigment {color %s}}}\n", box_y, box_z, box_color);
    printf("sphere{<%lf, 0, 0>, .1 texture{ pigment {color %s}}}\n", box_x, box_color);
    printf("sphere{<%lf, 0, %lf>, .1 texture{ pigment {color %s}}}\n", box_x, box_z, box_color);
    printf("sphere{<%lf, %lf, 0>, .1 texture{ pigment {color %s}}}\n", box_x, box_y, box_color);
    printf("sphere{<%lf, %lf, %lf>, .1 texture{ pigment {color %s}}}\n", box_x, box_y, box_z, box_color);
  }
}

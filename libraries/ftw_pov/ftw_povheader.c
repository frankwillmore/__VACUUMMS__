/* ftw_povheader.c */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ftw_param.h>
#include <ftw_pov.h>

void writePOVHeader()
{
  char *camera_x="40", *camera_y="40", *camera_z="40";
  char *look_at_x="0.0", *look_at_y="0.0", *look_at_z="0.0";
  double light_source_x=125, light_source_y=125, light_source_z=125;
  double box_x=10.0, box_y=10.0, box_z=10.0;
  double ambient_light = 0.0;
  char *light_color = "White";
  char *background = "Black";
  char *shadow = "";
  char *right = "1.0";

  // this will append usage parameters and exit with status 0 when writePOVHeader() is called
  if (getFlagParam("-usage"))
  {
    printf("  *** Header parameters and defaults *** \n");
    printf("        -light_source [ 125.0 125.0 125.0 ]\n");
    printf("        -light_color [ White ]\n");
    printf("        -standard_light\n");
    printf("        -camera [ 40.0 40.0 40.0 ]\n");
    printf("        -look_at [ 0.0 0.0 0.0 ]\n");
    printf("        -box [ 10.0 10.0 10.0 ]\n");
    printf("        -background [ Black ]\n");
    printf("        -ambient_light [ 0.0 ]\n");
    printf("        -right [ 1.0 (for 16/9 aspect ratio) ]\n");
    printf("        -show_floor \n");
    printf("        -show_box \n");
    printf("        -shadowless \n");
    printf("        -no_header \n");
    printf("\n");
    exit(0);
  }
  if (getFlagParam("-no_header")) return;
  if (getFlagParam("-shadowless")) shadow="shadowless";

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-ambient_light", &ambient_light);
  getVectorStringParam("-camera", &camera_x, &camera_y, &camera_z);
  getVectorStringParam("-look_at", &look_at_x, &look_at_y, &look_at_z);
  getVectorParam("-light_source", &light_source_x, &light_source_y, &light_source_z);
  getStringParam("-light_color", &light_color);
  getStringParam("-background", &background);
  getStringParam("-right", &right);

  printf("#include \"colors.inc\"\n");
  printf("background {color %s}\n", background);
  printf("camera{location<%s,%s,%s> look_at <%s,%s,%s> right %s}\n", camera_x, camera_y, camera_z, look_at_x, look_at_y, look_at_z, right);
  if (getFlagParam("-show_box")) printf("box { <0,0,0> <%lf, %lf, %lf> texture { pigment {color White transmit 0.8}} }", box_x, box_y, box_z);
  if (getFlagParam("-show_floor")) printf("plane { <0,1,0> , 0 pigment { checker color White, color Red } }");
  if (getFlagParam("-ambient_light")) printf("global_settings { ambient_light rgb <%lf,%lf,%lf> }\n", ambient_light, ambient_light, ambient_light);
  if (getFlagParam("-standard_light"))
  {
    printf("light_source{<100,0,0> color %s %s}\n", light_color, shadow);
    printf("light_source{<0,100,0> color %s %s}\n", light_color, shadow);
    printf("light_source{<0,0,100> color %s %s}\n", light_color, shadow);
    printf("light_source{<-100,0,0> color %s %s}\n", light_color, shadow);
    printf("light_source{<0,-100,0> color %s %s}\n", light_color, shadow);
    printf("light_source{<0,0,-100> color %s %s}\n", light_color, shadow);
  }
  else printf("light_source{<%lf,%lf,%lf> color %s %s}\n", light_source_x, light_source_y, light_source_z, light_color, shadow);
}

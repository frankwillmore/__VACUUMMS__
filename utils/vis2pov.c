/* vis2pov.c */

#include <stdio.h>
#include <ftw_std.h>

#define MAX_NUM_MOLECULES 1000

double x[MAX_NUM_MOLECULES];
double y[MAX_NUM_MOLECULES];
double z[MAX_NUM_MOLECULES];

double box_x=10, box_y=10, box_z=10;
double max_scale = 1;

double camera_x=15, camera_y=15, camera_z=15;
double light_source_x=15, light_source_y=15, light_source_z=15;

double sfactor=1;

int number_of_molecules = 0;
FILE *instream;

int main(int argc, char *argv[])
{
  int i=0;
  char line[80];
  char *xs, *ys, *zs, *ds, *cs;
  double x, y, z, d, c;
  char *color="White";

  instream = stdin;

  while (++i<argc)
  {
    if ((argc>1) && ((*argv[i]) != '-')) instream = fopen(argv[i], "r");
    else if (!strcmp(argv[i], "-v")) verbose = 1;
    else if (!strcmp(argv[i], "-sfactor")) sfactor = getDoubleParameter("sfactor", argv[++i]); 
    else if (!strcmp(argv[i], "-camera")) 
    {
      camera_x = getDoubleParameter("camera_x", argv[++i]);
      camera_y = getDoubleParameter("camera_y", argv[++i]);
      camera_z = getDoubleParameter("camera_z", argv[++i]);
    }
    else if (!strcmp(argv[i], "-light_source")) 
    {
      light_source_x = getDoubleParameter("light_source_x", argv[++i]);
      light_source_y = getDoubleParameter("light_source_y", argv[++i]);
      light_source_z = getDoubleParameter("light_source_z", argv[++i]);
    }
    else if (!strcmp(argv[i], "-box")) 
    {
      box_x = getDoubleParameter("box_x", argv[++i]);
      box_y = getDoubleParameter("box_y", argv[++i]);
      box_z = getDoubleParameter("box_z", argv[++i]);
    }
    else 
    {
      printf("option %s not understood.\n", argv[i]);
      exit(0);
    }
  }

  printf("#include \"colors.inc\"\n");
  printf("#background {color Black}\n");
  printf("camera{location<%lf,%lf,%lf> look_at <0,0,0>}\n", camera_x, camera_y, camera_z);
  printf("light_source{<%lf,%lf,%lf> color White}\n", light_source_x, light_source_y, light_source_z);

  while (TRUE)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\t");
    cs = strtok(NULL, "\n");

    if (!strcmp(cs, "1")) color = "Red";
    if (!strcmp(cs, "2")) color = "Green";
    if (!strcmp(cs, "3")) color = "Blue";

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);
    d = strtod(ds, NULL); 

    /* vis uses diameter, pov uses radius... */
    d *= .5;
    d *= sfactor;

    printf("sphere{<%lf, %lf, %lf>, %lf texture{ pigment {color %s}}}\n", x, y, z, d, color);
  }
 
  fclose(instream);
  
  //V printf("%d lines read...\n", n_cavities);
 
  return 0;
}

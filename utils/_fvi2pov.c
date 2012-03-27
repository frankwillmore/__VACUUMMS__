/* fvi2pov.c */

//  IN:    .fvi  free volume intensity 
//  OUT:   .pov  povray format

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ftw_param.h>
#include <ftw_pov.h>

double box_x=10, box_y=10, box_z=10;

int main(int argc, char *argv[])
{
  char line[256];
  char *xs, *ys, *zs, *fv_intensity_s;
  double x, y, z;
  char *color="Yellow";
  double fv_intensity;
  double transmittance;
  double intensity_threshold = 0.1;
  double blob_threshold = 0.65;
  double element_radius = 0.125;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("usage:       fvi2pov     IN:      .fvi = free volume intensity (x, y, z, I)\n");
    printf("                         OUT:     .pov = povray format (no header)\n\n");
    printf("\n");
    printf("                         -color [ Yellow ] \n");
    printf("                         -intensity_threshold [0.1] \n");
    printf("                         -blob_threshold [0.65] \n");
    printf("                         -element_radius [0.125] \n");
    printf("                         -blob \n");
    printf("                         -clip \n");
    printf("\n");
  }
  writePOVHeader();

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getStringParam("-color", &color);
  getDoubleParam("-intensity_threshold", &intensity_threshold);
  getDoubleParam("-blob_threshold", &blob_threshold);
  getDoubleParam("-element_radius", &element_radius);

  printf("// using intensity threshold of %lf\n", intensity_threshold);
  printf("// begin gfg2pov records\n");
  
  if (getFlagParam("-clip")) printf("intersection { box {<0,0,0>< %lf, %lf, %lf>} ", box_x, box_y, box_z);
  if (getFlagParam("-blob")) printf("blob { threshold %lf \n", blob_threshold);
  while (1) // loop over all lines
  {
    fgets(line, 256, stdin);
    if (feof(stdin)) break;
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    fv_intensity_s = strtok(NULL, "\n");

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);
    fv_intensity = strtod(fv_intensity_s, NULL);

    // Only put an entry for visible parts... make it easier.
    if (fv_intensity < intensity_threshold) continue; // skip to the next line.

    if (getFlagParam("-blob"))
    {
      // express fvi as blob contribution
      // printf("sphere{<%lf, %lf, %lf>, %lf %lf texture{ pigment {color %s }}}\n", x, y, z, element_radius, fv_intensity, color);
      // changed this to reflect same blobm contribution from all points in isosurface.
      printf("sphere{<%lf, %lf, %lf>, %lf %lf texture{ pigment {color %s }}}\n", x, y, z, element_radius, 1.00000, color);
    }
    else
    {
      // We want transmittance to go to 1 as intensity goes to zero. 
      transmittance = 1.0 - fv_intensity;
      printf("sphere{<%lf, %lf, %lf>, %lf texture{ pigment {color %s transmit %f}}}\n", x, y, z, element_radius, color, transmittance);
    }
  }

  if (getFlagParam("-blob")) printf("} \n");
  if (getFlagParam("-clip")) printf(" texture{ pigment {color %s } }}\n", color);
 
  fclose(stdin);
  return 0;
}


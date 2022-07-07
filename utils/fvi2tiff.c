/* fvi2tiff.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include <ftw_makeTIFF.h>
#include <ftw_param.h>

FILE *instream;

/********************************************************************************/
/*                                                                              */
/*  Reads an fvi format (%f\t%f\t%f\t%f\n") and generates a tif file            */
/*  -o name specified on command line                                           */
/*  -box image dimensions specified on command line                             */
/*                                                                              */
/*  currently only supporting 4 bytes/pixel                                     */
/*                                                                              */
/*                                                                              */
/*                                                                              */
/********************************************************************************/

int main(int argc, char *argv[])
{
  char line[80];
  char *xs, *ys, *zs, *fvis;
  int i, j, k;
  int x_dim, y_dim, z_dim;
  double fvi;
  double _box_x, _box_y, _box_z;
  char *outfile_name = "outfile.tif";

  int sampleperpixel = 4;

  char *image;

  instream = stdin;

  setCommandLineParameters(argc, argv);

  if (getFlagParam("-usage"))
  {
    printf("usage:  fvi2tif       -box lll mmm nnn \n");
    printf("                      -o [outfile.tif] \n"); 
    printf("                      < file.fvi\n");
    exit(0);
  }
  getStringParam("-o", &outfile_name);
  printf("using %s outfile...\n", outfile_name);

  getVectorParam("-box", &_box_x, &_box_y, &_box_z);
  x_dim = floor(_box_x);
  y_dim = floor(_box_y);
  z_dim = floor(_box_z);

//    else if (!strcmp(argv[i], "-o")) outfile_name = argv[++i];
//    else if (!strcmp(argv[i], "-box")) {
//      x_dim = atoi(argv[++i]);
//      y_dim = atoi(argv[++i]);
//      z_dim = atoi(argv[++i]);
//      printf("xyz = %d\t%d\t%d\n", x_dim, y_dim, z_dim );
//    }
//  }

  image = (char*)malloc(x_dim * y_dim * z_dim * sampleperpixel);

  for (k=0; k<z_dim; k++)
  for (j=0; j<y_dim; j++)
  for (i=0; i<x_dim; i++)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    fvis = strtok(NULL, "\n");

    fvi = strtod(fvis, NULL);

    int voxel = sampleperpixel * (i + j*x_dim + k*x_dim*y_dim);
    char fvid = (char)floor(fvi*256);
    image[0 + voxel] = 0;	// RED
    image[1 + voxel] = fvid;	// GREEN
    image[2 + voxel] = 0;	// BLUE
//    image[3 + voxel] = 255;	
    image[3 + voxel] = 0;	// ALPHA
  }

  makeTIFF(outfile_name, x_dim, y_dim, z_dim, image, sampleperpixel);
}


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
  double _dim_x=256, _dim_y=256, _dim_z=256; // to capture command line args as double, to then convert to int 
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
    printf("                      -dims [256] [256] [256] \n"); 
    printf("                      -o [outfile.tif] \n"); 
    printf("                      < file.fvi\n");
    exit(0);
  }
  getStringParam("-o", &outfile_name);
  printf("using %s outfile...\n", outfile_name);

  getVectorParam("-box", &_box_x, &_box_y, &_box_z);
// WTF? what was I doing here? 
//  x_dim = floor(_box_x);
//  y_dim = floor(_box_y);
//  z_dim = floor(_box_z);

  // There's no get for integer triplet, so taking floor
  getVectorParam("-dims", &_dim_x, &_dim_y, &_dim_z);
  x_dim = floor(_dim_x);
  y_dim = floor(_dim_y);
  z_dim = floor(_dim_z);

fprintf(stderr, "Got dims of %dx%dx%d.\n", x_dim, y_dim, z_dim);
fprintf(stderr, "The box size has slices of thickness: %lfx%lfx%lf.\n", _box_x/x_dim, _box_y/y_dim, _box_z/z_dim);

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

// These are read but never used
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    fvis = strtok(NULL, "\n");

    fvi = strtod(fvis, NULL);

    int voxel = sampleperpixel * (i + j*x_dim + k*x_dim*y_dim);
//    char fvid = (char)floor(fvi*256);
    unsigned int fvid = floor(fvi*256);
//printf("%c", fvid);
    image[0 + voxel] = 0;	// RED
    image[1 + voxel] = fvid;	// GREEN
    image[2 + voxel] = 0;	// BLUE
//    image[3 + voxel] = 255;	
//    image[3 + voxel] = 0;	// ALPHA
    image[3 + voxel] = 255;	// ALPHA

// For debugging, this should just have the right vals, assuming they sliced correctly
// fprintf(stderr, "writing value of %s for voxel at %d, %d, %d\n", fvis, i, j, k);
fprintf(stderr, "writing value of %d for voxel at %d, %d, %d\n", fvid, i, j, k);

  }

  makeTIFF(outfile_name, x_dim, y_dim, z_dim, image, sampleperpixel);
}


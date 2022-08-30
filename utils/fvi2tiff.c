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
  int alpha = 255;
  int green = 1, red = 0, blue = 0;
  double _box_x, _box_y, _box_z;
  char *outfile_name = "outfile.tif";

  int sampleperpixel = 4;

  char *image;

  instream = stdin;

  setCommandLineParameters(argc, argv);

  if (getFlagParam("-usage"))
  {
    printf("usage:  fvi2tiff      -box x.xx y.yy z.zz \n");
    printf("                      -dims [256] [256] [256] \n"); 
    printf("                      -alpha [255] \n"); 
    printf("                      -o [outfile.tif] \n"); 
    printf("                      < file.fvi\n\n");
    printf("        Note that it is expected that the spacing of points be uniform, continuous, \n");
    printf("        and compliant with the box size and resolution dims as specified on the\n");
    printf("        command line. No input validation is performed.\n\n");
    printf("        By default, all FVI information is written to the green channel only.\n");
    printf("        To specify an alternate color channel configuration, add: \n");
    printf("                       [+red] [+blue] [-green] \n\n");
    printf("        A dimension size of 1024^3 will gennerate a TIFF that is beyond libtiff's\n");
    printf("        capacity. If working with this resolution, trim the x dimension, back to 1000.\n");
    printf("        since this is the leading dimension, it will stop reading the FVI file..\n\n");
    exit(0);
  }

  red = getFlagParam("+red");
  blue = getFlagParam("+blue");
  if (getFlagParam("-green")) green = 0;

  getIntParam("-alpha", &alpha);
  fprintf(stderr, "Using alpha value of %d\n", alpha);

  getStringParam("-o", &outfile_name);
  fprintf(stderr, "Writing TIFF data to %s outfile...\n", outfile_name);

  // This is recorded but not used. Eventually it will be added to the TIFF file for scale
  getVectorParam("-box", &_box_x, &_box_y, &_box_z);

  // There's no get() for integer triplet, so taking double and then floor()
  getVectorParam("-dims", &_dim_x, &_dim_y, &_dim_z);
  x_dim = floor(_dim_x);
  y_dim = floor(_dim_y);
  z_dim = floor(_dim_z);

  fprintf(stderr, "Got dims of %dx%dx%d, sampleperpixel of %d.\n", x_dim, y_dim, z_dim, sampleperpixel);
  fprintf(stderr, "The expected thickness of slices is: %lfx%lfx%lf.\n", _box_x/x_dim, _box_y/y_dim, _box_z/z_dim);

  // This was a fun bug. For dims of 1024, this works out to 2^32, and therefore zero as an int.
  // So malloc(0) returns a valid pointer, but can't write to the memory. 
  // Added the cast to long to fix 32-bit arithmetic issue.
  long block = (long)x_dim * (long)y_dim * (long)z_dim * (long)sampleperpixel;
  fprintf(stderr, "Allocating block of %ld for TIFF...\n", block);
  image = (char*)malloc(block);
  if (image==NULL) 
  {
      fprintf(stderr, "Couldn't allocate memory.");
      exit(137);
  }
  fprintf(stderr, "Allocated image memory, %ld bytes.\n", block);

  for (i=0; i<x_dim; i++)
  for (j=0; j<y_dim; j++)
  for (k=0; k<z_dim; k++)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    // These are read but never used
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");

    fvis = strtok(NULL, "\n");
    fvi = strtod(fvis, NULL);

    // int voxel = sampleperpixel * (i + j*x_dim + k*x_dim*y_dim);
//    long voxel = (long)sampleperpixel * ((long)i + (long)j*x_dim + (long)k*x_dim*y_dim);
    long voxel = (long)sampleperpixel * ((long)(i*y_dim*z_dim) + (long)(j*z_dim) + (long)k);
    unsigned int fvid = floor(fvi*256);
    if (red) image[0 + voxel] = fvid;
    else image[0 + voxel] = 0;
    if (green) image[1 + voxel] = fvid;
    else image[1 + voxel] = 0;
    if (blue) image[2 + voxel] = fvid;
    else image[2 + voxel] = 0;

    image[3 + voxel] = alpha;
  }


// int makeTIFF(char *filename, int height, int width, int depth, char *image, int sampleperpixel)
// Tried each combo below, settled on the last.
// 1)  makeTIFF(outfile_name, x_dim, y_dim, z_dim, image, sampleperpixel);
// 2)  makeTIFF(outfile_name, y_dim, x_dim, z_dim, image, sampleperpixel);
// 3)  Swap the z and x dims here because of how these are defined in TIFF world
  makeTIFF(outfile_name, z_dim, y_dim, x_dim, image, sampleperpixel);
}


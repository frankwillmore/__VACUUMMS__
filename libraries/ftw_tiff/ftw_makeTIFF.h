#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <tiffio.h>

// makeTIFF turns the 1D array stored in image into the tiff file in filename.
int makeTIFF(char *filename, int height, int width, int depth, char *image, int sampleperpixel);

// readTIFF reads the TIFF file in filename and returns a pointer to the data array.
// The dimensions of the image and samples per pixel are returned by reference.
char *readTIFF(char *filename, int *width, int *height, int *depth, int *samplesperpixel);


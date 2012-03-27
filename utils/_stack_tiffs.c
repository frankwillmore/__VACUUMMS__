/* stack_tiffs.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tiffio.h>

#include <ftw_makeTIFF.h>

int width = 512;
int height = 512;
int depth = 47;
int sample_per_pixel = 4;
char image[1024*1024*94*4];

//  IN:  name of a stack of TIFF files
//  OUT: a 3D-TIFF file
//  e.g. issue command:  stack_tiffs *.tif
//  assuming input is 256x256
main(int argc, char* argv[])
{
  int z;
  char *filename;

  for (z = 1; z < argc; z++)
  {
filename = argv[z];
    TIFF* tif = TIFFOpen(filename, "r");
printf("z-1 = %d, (93-z) - 1 = %d\n", z-1, 93-(z-1));
    if (!tif) 
    { 
      printf("Could not open %s\n", argv[1]);
      exit(1);
    }
    uint32 imagelength;
    tdata_t buf;
    uint32 row;
    uint16 s, nsamples;
    int pixel;
    char value;

    buf = _TIFFmalloc(TIFFScanlineSize(tif));
    for (row = 0; row < 512; row++)
    {
      TIFFReadScanline(tif, buf, row, 0);
      for(pixel =0;pixel<512; pixel++)
      {
        // add 1 to map to green channel
        value = 255 - ((char)(*(char*)(pixel + buf)));
        image[1 + sample_per_pixel * ((z-1)*1024*1024 + row*1024 + pixel)]    = value;

        // periodic images
        image[1 + sample_per_pixel * ((z-1)*1024*1024 + row*1024 + (1023 - pixel))]    = value;
        image[1 + sample_per_pixel * ((z-1)*1024*1024 + (1023-row)*1024 + pixel)]    = value;
        image[1 + sample_per_pixel * ((z-1)*1024*1024 + (1023-row)*1024 + (1023 - pixel))]    = value;
        image[1 + sample_per_pixel * ((93-(z-1))*1024*1024 + row*1024 + pixel)]    = value;
        image[1 + sample_per_pixel * ((93-(z-1))*1024*1024 + row*1024 + (1023 - pixel))]    = value;
        image[1 + sample_per_pixel * ((93-(z-1))*1024*1024 + (1023-row)*1024 + pixel)]    = value;
        image[1 + sample_per_pixel * ((93-(z-1))*1024*1024 + (1023-row)*1024 + (1023 - pixel))]    = value;
      }
    }
    _TIFFfree(buf);
    TIFFClose(tif);
  }

  makeTIFF("new.tif", 2*width, 2*height, 2*depth, image, sample_per_pixel);
  exit(0);
}


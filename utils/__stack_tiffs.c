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
char image[512*512*47*4];

//  IN:  name of a stack of TIFF files
//  OUT: a 3D-TIFF file
//  e.g. issue command:  stack_tiffs *.tif
//  assuming input is 256x256
main(int argc, char* argv[])
{
  int z;
  char *filename;

  for (z = 1; z< argc; z++)
  {
filename = argv[z];
    TIFF* tif = TIFFOpen(filename, "r");
printf("%ld\n", tif);
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

    buf = _TIFFmalloc(TIFFScanlineSize(tif));
    for (row = 0; row < 512; row++)
    {
      TIFFReadScanline(tif, buf, row, 0);
      for(pixel =0;pixel<512; pixel++)
      {
        // add 1 to map to green channel
        image[1 + sample_per_pixel * ((z-1)*512*512 + row*512 + pixel)]    = ((char)(*(char*)(pixel + buf)));
      }
    }
    _TIFFfree(buf);
    TIFFClose(tif);
  }

  makeTIFF("new.tif", width, height, depth, image, sample_per_pixel);
  exit(0);
}


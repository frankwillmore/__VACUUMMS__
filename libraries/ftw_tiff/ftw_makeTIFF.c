#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <tiffio.h>

#include <ftw_makeTIFF.h>

int makeTIFF(char *filename, int height, int width, int depth, char *image, int sampleperpixel)
{
  TIFF *out= TIFFOpen(filename, "w");

  tsize_t linebytes = sampleperpixel * width;     // length in memory of one row of pixel in the image.
  unsigned char *buf = NULL;        // buffer used to store the row of pixel information for writing to file

  // We set the strip size of the file to be size of one row of pixels
  TIFFSetField(out, TIFFTAG_ROWSPERSTRIP, TIFFDefaultStripSize(out, linebytes));

    int page;
    for (page = 0; page < depth; page++)
    {
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, sampleperpixel);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);

//        /* It is good to set resolutions too (but it is not nesessary) */
//        xres = yres = 100;
//        res_unit = RESUNIT_INCH;
//        TIFFSetField(out, TIFFTAG_XRESOLUTION, xres);
//        TIFFSetField(out, TIFFTAG_YRESOLUTION, yres);
//        TIFFSetField(out, TIFFTAG_RESOLUTIONUNIT, res_unit);

        /* We are writing single page of the multipage file */
        TIFFSetField(out, TIFFTAG_SUBFILETYPE, FILETYPE_PAGE);
        /* Set the page number */
        TIFFSetField(out, TIFFTAG_PAGENUMBER, page, depth);

//this was working...
//char *buffer = (char *) malloc(linebytes);
//memset(buffer, 0x5F, linebytes);
//int i;
//for (i = 0; i < width; i++) { buffer[sampleperpixel*i] = i;buffer[sampleperpixel*i+1] = i;buffer[sampleperpixel*i+2] = i;}

char *pbuffer;
        int row;
//        for (row = 0; row < height; row++) TIFFWriteScanline(out, buffer, row, 0);
        for (row = 0; row < height; row++) 
        {
          pbuffer = image + linebytes * (page*height+ row);
// printf("%d\t%d\t%ld\n", page, row, (long)pbuffer);
          TIFFWriteScanline(out, pbuffer, row, 0);
        }

        TIFFWriteDirectory(out);

//  free(buffer);

    } // next page

  TIFFClose(out);
  return 0;
}

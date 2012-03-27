/* readtiff.c */

#include <stdio.h>
#include <stdlib.h>

#include <tiffio.h>

main(int argc, char* argv[])
{
    TIFF* tif = TIFFOpen(argv[1], "r");
    if (tif) 
    {
        uint32 imagelength;
        tdata_t buf;
        uint32 row;
        uint16 s, nsamples;
	int pixel;

        TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
        TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &nsamples);
printf("nsamples = %d\n", nsamples);

        buf = _TIFFmalloc(TIFFScanlineSize(tif));
        for (row = 0; row < imagelength; row++)
        {
            TIFFReadScanline(tif, buf, row, 0);
// need to get width here...
            for(pixel =0;pixel<256; pixel++) 
            {
              printf("%d", ((char)(*(char*)(pixel + buf)))/26 );
            }
printf("\n");
        }
        _TIFFfree(buf);
        TIFFClose(tif);
    }
    else printf("Could not open %s\n", argv[1]);
    exit(0);
}

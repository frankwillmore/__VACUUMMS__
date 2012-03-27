/* stream2slice.c */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <ftw_param.h>

int main(int argc, char *argv[])
{
  char line[256];
  double x;
  double x_sum;
  long line_number=0;
  long slice_number=0;
  long slice_size=65536;
  char *slice_dir = "slice";
  char *file_extension = "";
  char current_slice_filename[256];
  FILE *p_slice;


  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage")){
    printf("usage:     stream2slice    -slice_size [ 65536 ]\n");
    printf("                           -slice_dir [ slice ]\n");
    printf("                           -file_extension [ ] \n");
    printf("                           \n");
    exit(0);
  }

  if (getFlagParam("-slice_size")) getLongParam("-slice_size", &slice_size);
  if (getFlagParam("-slice_dir")) getStringParam("-slice_dir", &slice_dir);
  if (getFlagParam("-file_extension")) getStringParam("-file_extension", &file_extension);

  while (1)
  {
    if ((line_number % slice_size) == 0) 
    {
      // close old slice file
      if (p_slice != NULL)
      {
        fflush(p_slice);
        fclose(p_slice);
      }

      // open a new slice file
      slice_number = (long)floor((double)line_number / (double)slice_size);
      sprintf(current_slice_filename, "%s/%06ld.%s", slice_dir, slice_number, file_extension);
      p_slice = fopen(current_slice_filename, "w"); 
    }

    // now get the line and write to the slice file
    fgets(line, 256, stdin);
    if (feof(stdin)) break;
    fprintf(p_slice, "%s", line);

    line_number++;
  }

  // close slice file if open
  if (p_slice != NULL)
  {
    fflush(p_slice);
    fclose(p_slice);
  }
}

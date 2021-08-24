/* nsplit.c */
/* cut the input file into chunks */
/* ignore -ignore [n] lines */
/* use -interval [n] lines */

#include <stdio.h>
#include <ftw_std.h>
#include <ftw_param.h>

FILE *instream;
FILE *outstream;

int main(int argc, char *argv[])
{
  char line[256];
  char outfile_name[256];
  char *outfile_extension = "xxx";

  int ignore=0;
  int interval=0;

printf("not working...\n");
exit(1);

  instream = stdin;
  setCommandLineParameters(argc, argv);

  if (getFlagParam("-usage")) {
    printf("usage:     nsplit -ignore [0] \n");
    printf("                  -interval [0] \n");
    printf("                  -outfile_extension [xxx] \n");
    printf("\n");
    exit(0);
  }

  getIntParam("-interval", &interval);
  getIntParam("-ignore", &ignore);
  getStringParam("-outfile_extension", &outfile_extension);

  int ignore_counter;
  int interval_counter;
  int outfile_counter;

  ignore_counter = ignore;
  interval_counter = interval;
  outfile_counter = 0;
  sprintf(outfile_name, "%06d.%s", outfile_counter, outfile_extension);
  outstream = fopen(outfile_name, "w");

  while (TRUE)
  {
    // skip lines first
    if (ignore_counter--) {
      fgets(line, 256, instream);
printf("skipping::%s", line);
      if (feof(instream)) break;
      else continue;
    }

    if (interval_counter--){
      fgets(line, 256, instream);
printf("including::%s", line);
      fprintf(outstream, "%s", line); 
      if (feof(instream)) break;
      else continue;
    }

    fclose(outstream);

    // open a new file and reset counters
    ignore_counter = ignore;
    interval_counter = interval;
    outfile_counter++;
    sprintf(outfile_name, "%06d.%s", outfile_counter, outfile_extension);
    outstream = fopen(outfile_name, "w");
  }
 
  fclose(outstream);
  fclose(instream);
  
  return 0;
}

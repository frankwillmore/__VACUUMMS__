#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ftw_param.h>

#define N_STREAMS 64 

FILE *instream[N_STREAMS];

int main(int argc, char *argv[])
{
  char line[80];
  char *xs, *ys, *xsi, *ysi;
  double y_sum;
  int i;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("usage:	Adds a list of tabulated functions.\n");
    printf("		takes a list of files (two fields, separated by tabs) on the command line\n");
    printf("          	And outputs as a tabulated function the sum of the input functions.\n");
    printf("\n");
    exit(0);
  }

  for (i=1; i<argc; i++) instream[i] = fopen(argv[i], "r");

  while (1)
  {
    fgets(line, 80, instream[1]);
    if (feof(instream[1])) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\n");

    y_sum = strtod(ys, NULL);

    for (i=2; i<argc; i++)
    {
      fgets(line, 80, instream[i]);

      xsi = strtok(line, "\t");
      ysi = strtok(NULL, "\n");

      if (strcmp(xs,  xsi)) 
      {
        printf("values %s and %s do not match.\n", xs, xsi);
        exit(0);
      }
      
      y_sum += strtod(ysi, NULL);
    }

    printf("%s\t%lf\n", xs, y_sum);
  }

  for (i=1; i<argc; i++) fclose(instream[i]);
}

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ftw_param.h>

int main(int argc, char *argv[])
{
  double sum=0;
  int i=0;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("usage:	Adds a list of arguments from the command line.\n");
    printf("\n");
    exit(0);
  }

  for (i=1; i<argc; i++)
  {
    sum += strtod(argv[i], NULL);
  }

  printf("%lf\n", sum);
}

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <ftw_param.h>

int main(int argc, char *argv[])
{
  double product=1.0;
  int i=0;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("usage:	multiplies the list of arguments on the command line and writes to stdout.\n");
    printf("            expr_multiply [n.nnnnnn] [n.nnnnnn] ... \n");
    printf("\n");
    exit(0);
  }

  for (i=1; i<argc; i++)
  {
    product *= strtod(argv[i], NULL);
  }

  printf("%lf\n", product);
}

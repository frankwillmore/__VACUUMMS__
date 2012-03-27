/* stream_multiply.c */

/* input:  .dst */

#include <ftw_param.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  char line[80];
  double x;
  char *xs;
  double product;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("usage:	multiplies by the list of arguments on the command line and writes to stdout.\n");
    printf("            stream_multiply [n.nnnnnn] [n.nnnnnn] ... \n");
    printf("\n");
    exit(0);
  }

  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\n");
    product = strtod(xs, NULL);

    int i=0;
    for (i=1; i<argc; i++) product *= strtod(argv[i], NULL);

    printf("%lf\n", product);
  }
}

/* cav2vis.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

FILE *instream;

int main(int argc, char *argv[])
{
  char line[80];
  char *xs, *ys, *zs, *ds;
  char *color;
  int i=0;

  instream = stdin;

  if ((argc>1) && ((*argv[1]) != '-'))
  {
    instream = fopen(argv[1], "r");
  }

  while (++i<argc)
  {
    if ((*argv[i]) != '-') instream = fopen(argv[i], "r");
    else if (!strcmp(argv[i], "-c")) color = argv[++i];
  }

  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    printf("%s\t%s\t%s\t%s\t%s\n", xs, ys, zs, ds, color);
  }
}

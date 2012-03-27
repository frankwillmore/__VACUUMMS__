/* cfg2vis.c */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

FILE *input_file;

int main(int argc, char *argv[])
{
  char line[80];
  char *xs, *ys, *zs;

  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\n");

    printf("%s\t%s\t%s\t1.000000\t1\n", xs, ys, zs);
  }
}

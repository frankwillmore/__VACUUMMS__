/* min.c */

/* input:  .dst */
/* output:  singular value (average of all entries) */

#include <ftw_std.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
  char line[80];
  int n=0;
  double min=0;
  char *xs;
  double x;

  fgets(line, 80, stdin);
  xs = strtok(line, "\n");
  min = strtod(xs, NULL);

  while (TRUE)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\n");
    x = strtod(xs, NULL);
    if (x<min) min=x; 
  }

  printf("%lf\n", min);

}

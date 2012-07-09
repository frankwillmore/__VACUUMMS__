/* recip.c */

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// In:  x, out: 1/x 

int main(int argc, char* argv[])
{
  char line[80];
  double x;
  char *xs;

  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\n");
    x = strtod(xs, NULL);

    printf("%lf\n", 1.0/x);
  }
}

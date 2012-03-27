/* pow.c */

#include <ftw_std.h>
#include <ftw_rng2.h>
#include <ftw_param.h>
#include <errno.h>
#include <math.h>

// In:  x, out: x^y 

int main(int argc, char* argv[])
{
  double x, y, z, d;
  char line[80];
  char *xs;

  y = strtod(argv[1], NULL);

  while (TRUE)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\n");

    x = strtod(xs, NULL);

    printf("%lf\n", pow(x, y));
  }
}


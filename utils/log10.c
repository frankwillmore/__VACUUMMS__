/* log10.c */

/* input:  .dst */
/* output:  log base 10 of input values */

#include <ftw_std.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

FILE *instream;

int main(int argc, char *argv[])
{
  char line[80];
  double x;
  char *xs;

  instream = stdin;
  
  while (TRUE)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\n");
    x = strtod(xs, NULL);
    printf("%lf\n", log10(x));
  }
}

/* avg.c */

/* input:  .dst */
/* output:  singular value (average of all entries) */

#include <ftw_std.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

FILE *instream;

int main(int argc, char *argv[])
{
  char line[80];
  int n=0;
  double sum=0;
  char *xs;

  instream = stdin;
  
  while (TRUE)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\n");
    sum += strtod(xs, NULL);
    n++;
  }

  printf("%lf\n", sum/n);

}

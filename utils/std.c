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
  double avg, std;
  char *xs;
  double x;

  instream = stdin;
  avg=strtod(argv[1], NULL);
  
  while (TRUE)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;

    xs = strtok(line, "\n");
    x = strtod(xs, NULL);
    
    sum += (avg-x)*(avg-x);
    n++;
  }

  std=sqrt(sum/n);
  
  printf("%lf\n", std);
}

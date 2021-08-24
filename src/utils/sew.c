/* sew.c */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N_STREAMS 64 

FILE *instream[N_STREAMS];

int main(int argc, char *argv[])
{
  char line[80];
  char *xs, *ys, *xsi, *ysi;
  double y_sum;
  int i;

  for (i=1; i<argc; i++) instream[i] = fopen(argv[i], "r");

  while (1)
  {
    fgets(line, 80, instream[1]);
    if (feof(instream[1])) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\n");

//    y_sum = strtod(ys, NULL);
    printf("%s\t%s\t", xs, ys);

    for (i=2; i<argc; i++)
    {
      fgets(line, 80, instream[i]);

      xsi = strtok(line, "\t");
      ysi = strtok(NULL, "\n");

      if (strcmp(xs,  xsi)) 
      {
        printf("values %s and %s do not match.\n", xs, xsi);
        exit(1);
      }
      
//      y_sum += strtod(ysi, NULL);
      printf("%s\t", ysi);
    }

    printf("\n");
//    printf("%s\t%lf\n", xs, y_sum);
  }

  for (i=1; i<argc; i++) fclose(instream[i]);
}

/* cfg2gfg.c */

#include <stdio.h>
#include <ftw_std.h>

double epsilon=1.0;

FILE *instream;

int main(int argc, char *argv[])
{
  int i=0;
  char line[80];
  char *xs, *ys, *zs, *ds;
  double x, y, z, d;

  instream = stdin;

  while (++i<argc)
  {
    if ((argc>1) && ((*argv[i]) != '-')) instream = fopen(argv[i], "r");
    else if (!strcmp(argv[i], "-v")) verbose = 1;
    else if (!strcmp(argv[i], "-epsilon")) epsilon = strtod(argv[++i], NULL); 
    else 
    {
      printf("option %s not understood.\n", argv[i]);
      exit(0);
    }
  }

  while (TRUE)
  {
    fgets(line, 80, instream);
    if (feof(instream)) break;
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);
    d = strtod(ds, NULL); 

    printf("%lf\t%lf\t%lf\t%lf\t%lf\n", x, y, z, d, epsilon);
  }
 
  fclose(instream);
  
  return 0;
}

/* clustercat.c */

/* in:  .cls */
/* out:  .ccs  cluster_number\tx\ty\tz\td */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define N_STREAMS 128

FILE *instream[N_STREAMS];

int cluster_number = 0;

int main(int argc, char *argv[])
{
  char line[80];
  char *clus, *cavs, *xs, *ys, *zs, *ds;
  int i;
  int out_cluster_number = 0;
  int in_cluster_number = 0, old_in_cluster_number = 0;

  if (argc>N_STREAMS) 
  {
    printf("too many input files.  exiting...\n"); 
    exit(0);
  }

  for (i=1; i<argc; i++)
  {
    instream[i] = fopen(argv[i], "r");

  while (1)
  {
    fgets(line, 80, instream[i]);
    if (feof(instream[i])) break;

    clus = strtok(line, "\t");
    cavs = strtok(NULL, "\t");
    xs = strtok(NULL, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    in_cluster_number = strtol(clus, NULL, 10);
    if (in_cluster_number != old_in_cluster_number)
    {
      out_cluster_number++;
      old_in_cluster_number = in_cluster_number;
    }

    printf("%ld\t%s\t%s\t%s\t%s\n", out_cluster_number, xs, ys, zs, ds);

  }
    fclose(instream[i]);
  }
}

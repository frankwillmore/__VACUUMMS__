/* cav2cluster.c */

// Finds all clusters, from a given file of cavities. 
// Remember to run uniq and scrub the input! 

// In:  .cav
// Out: .cluster  
// Ouput format is:  ("%d\t%lf\t%lf\t%lf\t%lf\n", cluster_number, x, y, z, d)

#define MAX_PAIRS 50000000
#define MAX_CAVITIES 131072

#include <ftw_std.h>
#include "cav2cluster.h"

double box_x, box_y, box_z;
double sfactor = 1;

int number_of_clusters=0;
int number_of_cavities=0;
int number_of_pairs=0;
double x[MAX_CAVITIES], y[MAX_CAVITIES], z[MAX_CAVITIES], d[MAX_CAVITIES];
int cluster_number[MAX_CAVITIES];
int cavityA[MAX_PAIRS], cavityB[MAX_PAIRS];

int main(int argc, char* argv[])
{
  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("cav2cluster     \t-box [ 10 10 10 ]\n");
    printf("                \t In:  .cav\n");
    printf("                \t Out: .cluster \n"); 
    printf("                \t Ouput format is:  (\"%%d\t%%lf\t%%lf\t%%lf\t%%lf\\n\", cluster_number, x, y, z, d)\n");
    printf("\n");
    exit(0);
  }
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-sfactor", &sfactor);

  readInputStream();
  findAllPairs();
  buildClusters();
  deleteEmptyClusters();
  sortClusters();
  printClusters();
}

void buildClusters()
{
  int i,j;

  for (i=0; i<number_of_cavities; i++) for (j=0; j<number_of_pairs; j++) if (cavityA[j] == i) cluster_number[cavityB[j]] = cluster_number[i];
}

void deleteEmptyClusters()
{
  int i,j;
  int cavs_in_cluster;

  number_of_clusters=number_of_cavities;

  // get rid of empty clusters...
  // first loop over cluster numbers...
  for (i=0; i<number_of_clusters;)
  {
    cavs_in_cluster=0;
    for (j=0; j<number_of_cavities; j++) if (cluster_number[j] == i) cavs_in_cluster++;

    if (cavs_in_cluster == 0)
    {
      // empty cluster, shift all higher cluster numbers down...
      for (j=0; j<number_of_cavities; j++) if (cluster_number[j] > i) cluster_number[j]--;
      number_of_clusters--;
    }
    else i++;
  }
}

void sortClusters()
{
  int i,j;
  double temp_x, temp_y, temp_z, temp_d, temp_cluster_no;
  
  for (i=0; i<number_of_cavities-1; i++)
  for (j=i; j<number_of_cavities; j++)
  {
    if (cluster_number[j] < cluster_number[i])
    {
      // swop the two cavities.

      temp_x = x[i];
      temp_y = y[i];
      temp_z = z[i];
      temp_d = d[i];
      temp_cluster_no = cluster_number[i];

      x[i] = x[j];
      y[i] = y[j];
      z[i] = z[j];
      d[i] = d[j];
      cluster_number[i] = cluster_number[j];

      x[j] = temp_x;
      y[j] = temp_y;
      z[j] = temp_z;
      d[j] = temp_d;
      cluster_number[j] = temp_cluster_no;
    }
  }
}

void printClusters()
{
  int i, j;
  for (i=0; i<number_of_cavities; i++)
    printf("%d\t%05d\t%lf\t%lf\t%lf\t%lf\n", cluster_number[i], i, x[i], y[i], z[i], d[i]);
}

void findAllPairs()
{
  double shift_x, shift_y, shift_z;
  double dx2, dy2, dz2;
  int i, j, k;

  for (i = 0; i < number_of_cavities - 1; i++)
  for (j = i + 1; j < number_of_cavities; j++)
  {
    for (shift_x = -box_x; shift_x <= box_x; shift_x += box_x)
    for (shift_y = -box_y; shift_y <= box_y; shift_y += box_y)
    for (shift_z = -box_z; shift_z <= box_z; shift_z += box_z)
    {
      dx2 = (shift_x + x[i] - x[j]) * (shift_x + x[i] - x[j]);
      dy2 = (shift_y + y[i] - y[j]) * (shift_y + y[i] - y[j]);
      dz2 = (shift_z + z[i] - z[j]) * (shift_z + z[i] - z[j]);

      if ((dx2 + dy2 + dz2) < ((d[i] + d[j]) * (d[i] + d[j]) * .25 * sfactor * sfactor))
      {
        cavityA[number_of_pairs] = i;
        cavityB[number_of_pairs] = j;
        number_of_pairs++;
      }
    }
  }
}

void readInputStream()
{
  char line[80];
  char *xs, *ys, *zs, *ds;

  while (TRUE)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    x[number_of_cavities] = strtod(xs, NULL);
    y[number_of_cavities] = strtod(ys, NULL);
    z[number_of_cavities] = strtod(zs, NULL);
    d[number_of_cavities] = strtod(ds, NULL);
    cluster_number[number_of_cavities] = number_of_cavities;
    number_of_cavities++;

    if (number_of_cavities > MAX_CAVITIES)
    {
      printf("Too many cavities.");
      exit(0);
    }
  }
}

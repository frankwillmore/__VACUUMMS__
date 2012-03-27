/* edges.c */

#include "ftw_std.h"

#include "edges.h"

#define MAX_PAIRS 100000

extern double box_x, box_y, box_z;

extern double x[];
extern double y[];
extern double z[];
extern double d[];
extern int connectivity_level[];
extern int n_cavities;
extern int mirror_depth;

struct pair all_pairs[MAX_PAIRS];
int pair_degeneracy[MAX_PAIRS];
int histogram[1000];
int n_bins;
int n_pairs = 0;
extern double sfactor;

int findAllEdges()
{
  double shift_x, shift_y, shift_z;
  double dx2, dy2, dz2;
  int i, j, k, l;

  for (i = 0; i < n_cavities - 1; i++)
  for (j = i + 1; j < n_cavities; j++)
  {
    for (shift_x = (-box_x * mirror_depth); shift_x <= (box_x * mirror_depth); shift_x += box_x)
    for (shift_y = (-box_y * mirror_depth); shift_y <= (box_y * mirror_depth); shift_y += box_y)
    for (shift_z = (-box_z * mirror_depth); shift_z <= (box_z * mirror_depth); shift_z += box_z)
    {
      dx2 = (shift_x + x[i] - x[j]) * (shift_x + x[i] - x[j]);
      dy2 = (shift_y + y[i] - y[j]) * (shift_y + y[i] - y[j]);
      dz2 = (shift_z + z[i] - z[j]) * (shift_z + z[i] - z[j]);

      if ((dx2 + dy2 + dz2) < ((d[i] + d[j]) * (d[i] + d[j]) * .25 * sfactor * sfactor)) 
      {
        all_pairs[n_pairs].i = i;
        all_pairs[n_pairs].j = j;
        all_pairs[n_pairs].separation = sqrt(dx2 + dy2 + dz2);
        n_pairs++;
      }
    }
  }

  for (k=0; k<n_pairs; k++) pair_degeneracy[k] = 0;

  for (k=0; k<n_pairs-1; k++) 
  for (l=k+1; l<n_pairs; l++)
  {
    if ((all_pairs[k].i == all_pairs[l].i && all_pairs[k].j == all_pairs[l].j) 
    || (all_pairs[k].i == all_pairs[l].j && all_pairs[k].j == all_pairs[l].i)) 
    {
      pair_degeneracy[k]++;
      pair_degeneracy[l]++;
    }
  }

  V printf("%d pairs.\n" , n_pairs);
}

getConnectivityLevels()
{
  int i, j;
  for (i=0; i<n_cavities; i++)
  {
    connectivity_level[i] = 0;
    for (j=0; j<n_pairs; j++)
    {
      if ((all_pairs[j].i == i) || (all_pairs[j].j == i)) connectivity_level[i]++;
    }
  }
 
  for (j=0; j<1000; j++) histogram[j] = 0;
  for (i=0; i<n_cavities; i++) histogram[connectivity_level[i]]++;
}

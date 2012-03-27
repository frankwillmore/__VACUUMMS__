/* CavityGraph.cpp */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#include "CavityGraph.h"

extern int* cavity_marks;

CavityGraph::CavityGraph(char *input_file_name)
{
  FILE *datastream;
  char line[80];
  char *xs, *ys, *zs, *ds;
  double x, y, z, d;
  int line_count=0;
  
  // first find out how many cavities... 
  datastream = fopen(input_file_name, "r");

  while (1)
  {
    fgets(line, 80, datastream);
    if (feof(datastream)) break;
    line_count++;
  }
 
  if (verbose) printf("%d lines read.\n", line_count);

  fclose(datastream);
  cavities = new struct s_cavity[line_count];

  number_of_cavities = 0;
  // now read the data...
  datastream = fopen(input_file_name, "r");

  while (line_count-- > 0)
  {
    fgets(line, 80, datastream);
    
    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    ds = strtok(NULL, "\n");

    x = strtod(xs, NULL);
    y = strtod(ys, NULL);
    z = strtod(zs, NULL);
    d = strtod(ds, NULL); 

    // check uniqueness of cavity
    int uniqueness_flag = 1;
    for (int i=0; i<number_of_cavities; i++)
    {
      double dx, dy, dz;
      dx = cavities[i].x - x;
      dy = cavities[i].y - y;
      dz = cavities[i].z - z;
      if ((dx*dx + dy*dy + dz*dz)<(cavity_uniqueness_threshold)) uniqueness_flag = 0;
    }

    if (uniqueness_flag)
    {
      cavities[number_of_cavities].x = x;
      cavities[number_of_cavities].y = y;
      cavities[number_of_cavities].z = z;
      cavities[number_of_cavities].d = d;

      number_of_cavities++;
    }
  }

  fclose(datastream);

  // now figure out the pairings...
  generatePairList();
}

int CavityGraph::getNumberOfCavities()
{
  return number_of_cavities;
}

int CavityGraph::getNumberOfPairs()
{
  return number_of_pairs;
}

int CavityGraph::getNumberOfUnmarkedAdjacentCavities(int cavity_number)
{
  int num_adjacent = 0;
  int i;

  for (i=0; i<number_of_pairs; i++)
  {
    if ((pairs[i].i == cavity_number) && (cavity_marks[pairs[i].j] == 0)) num_adjacent++;
    if ((pairs[i].j == cavity_number) && (cavity_marks[pairs[i].i] == 0)) num_adjacent++;
  }

  return num_adjacent;
}

int *CavityGraph::getListOfUnmarkedAdjacentCavities(int cavity_number)
{
  int i, j;
  int *list;

  list = new int[getNumberOfUnmarkedAdjacentCavities(cavity_number)];

  for (i=0, j=0; i<number_of_pairs; i++)
  {
    if ((pairs[i].i == cavity_number) && (cavity_marks[pairs[i].j] == 0)) list[j++] = pairs[i].j;
    if ((pairs[i].j == cavity_number) && (cavity_marks[pairs[i].i] == 0)) list[j++] = pairs[i].i;
  }

  return list;
}

double CavityGraph::getCavitySeparation(int cavity1, int cavity2)
{
  double big_sum;
  double separation;
  double dx, dy, dz;
  double dd;

  dx = cavities[cavity2].x - cavities[cavity1].x;
  dy = cavities[cavity2].y - cavities[cavity1].y;
  dz = cavities[cavity2].z - cavities[cavity1].z;
  big_sum = dx*dx + dy*dy + dz*dz;

  separation = sqrt(big_sum);
  return separation;
}

void CavityGraph::generatePairList()
{
  int i, j;

  pairs = new struct s_pair[(number_of_cavities * (number_of_cavities - 1))];

  for (i=0; i < number_of_cavities - 1; i++)
    for (j=i+1; j < number_of_cavities; j++)
      /* check in each of the 27 boxes for proximity */
      if (isAPair(i,j))
      {
        pairs[number_of_pairs].i = i;
        pairs[number_of_pairs].j = j;
        number_of_pairs++;
      }
}

int CavityGraph::isAPair(int i, int j)
{
  double dx, dy, dz;
  double Xi, Yi, Zi, Xj, Yj, Zj;
  double Di, Dj;
  double Dij, Dij2;
  double r2;

  Dij = cscale * (cavities[i].d + cavities[j].d)/2;
  Dij2 = Dij * Dij;

  /* check all 27 boxes */

  box222:
  {
    dx = cavities[j].x - cavities[i].x;
    dy = cavities[j].y - cavities[i].y;
    dz = cavities[j].z - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box111:
  {
    dx = (cavities[j].x - box_x) - cavities[i].x;
    dy = (cavities[j].y - box_y) - cavities[i].y;
    dz = (cavities[j].z - box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box112:
  {
    dx = (cavities[j].x - box_x) - cavities[i].x;
    dy = (cavities[j].y - box_y) - cavities[i].y;
    dz = (cavities[j].z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box113:
  {
    dx = (cavities[j].x - box_x) - cavities[i].x;
    dy = (cavities[j].y - box_y) - cavities[i].y;
    dz = (cavities[j].z + box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box121:
  {
    dx = (cavities[j].x - box_x) - cavities[i].x;
    dy = (cavities[j].y) - cavities[i].y;
    dz = (cavities[j].z - box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box122:
  {
    dx = (cavities[j].x - box_x) - cavities[i].x;
    dy = (cavities[j].y) - cavities[i].y;
    dz = (cavities[j].z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box123:
  {
    dx = (cavities[j].x - box_x) - cavities[i].x;
    dy = (cavities[j].y) - cavities[i].y;
    dz = (cavities[j].z + box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box131:
  {
    dx = (cavities[j].x - box_x) - cavities[i].x;
    dy = (cavities[j].y + box_y) - cavities[i].y;
    dz = (cavities[j].z - box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box132:
  {
    dx = (cavities[j].x - box_x) - cavities[i].x;
    dy = (cavities[j].y + box_y) - cavities[i].y;
    dz = (cavities[j].z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box133:
  {
    dx = (cavities[j].x - box_x) - cavities[i].x;
    dy = (cavities[j].y + box_y) - cavities[i].y;
    dz = (cavities[j].z + box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box211:
  {
    dx = (cavities[j].x) - cavities[i].x;
    dy = (cavities[j].y - box_y) + cavities[i].y;
    dz = (cavities[j].z - box_z) + cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box212:
  {
    dx = (cavities[j].x) - cavities[i].x;
    dy = (cavities[j].y - box_y) + cavities[i].y;
    dz = (cavities[j].z) + cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box213:
  {
    dx = (cavities[j].x) - cavities[i].x;
    dy = (cavities[j].y - box_y) + cavities[i].y;
    dz = (cavities[j].z + box_z) + cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box221:
  {
    dx = (cavities[j].x) - cavities[i].x;
    dy = (cavities[j].y) + cavities[i].y;
    dz = (cavities[j].z - box_z) + cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box223:
  {
    dx = (cavities[j].x) - cavities[i].x;
    dy = (cavities[j].y) + cavities[i].y;
    dz = (cavities[j].z + box_z) + cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box231:
  {
    dx = (cavities[j].x) - cavities[i].x;
    dy = (cavities[j].y + box_y) + cavities[i].y;
    dz = (cavities[j].z - box_z) + cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box232:
  {
    dx = (cavities[j].x) - cavities[i].x;
    dy = (cavities[j].y + box_y) + cavities[i].y;
    dz = (cavities[j].z) + cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box233:
  {
    dx = (cavities[j].x) - cavities[i].x;
    dy = (cavities[j].y + box_y) + cavities[i].y;
    dz = (cavities[j].z + box_z) + cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box311:
  {
    dx = (cavities[j].x + box_x) - cavities[i].x;
    dy = (cavities[j].y - box_y) - cavities[i].y;
    dz = (cavities[j].z - box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box312:
  {
    dx = (cavities[j].x + box_x) - cavities[i].x;
    dy = (cavities[j].y - box_y) - cavities[i].y;
    dz = (cavities[j].z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box313:
  {
    dx = (cavities[j].x + box_x) - cavities[i].x;
    dy = (cavities[j].y - box_y) - cavities[i].y;
    dz = (cavities[j].z + box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box321:
  {
    dx = (cavities[j].x + box_x) - cavities[i].x;
    dy = (cavities[j].y) - cavities[i].y;
    dz = (cavities[j].z - box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box322:
  {
    dx = (cavities[j].x + box_x) - cavities[i].x;
    dy = (cavities[j].y) - cavities[i].y;
    dz = (cavities[j].z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box323:
  {
    dx = (cavities[j].x + box_x) - cavities[i].x;
    dy = (cavities[j].y) - cavities[i].y;
    dz = (cavities[j].z + box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box331:
  {
    dx = (cavities[j].x + box_x) - cavities[i].x;
    dy = (cavities[j].y + box_y) - cavities[i].y;
    dz = (cavities[j].z - box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box332:
  {
    dx = (cavities[j].x + box_x) - cavities[i].x;
    dy = (cavities[j].y + box_y) - cavities[i].y;
    dz = (cavities[j].z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }

  box333:
  {
    dx = (cavities[j].x + box_x) - cavities[i].x;
    dy = (cavities[j].y + box_y) - cavities[i].y;
    dz = (cavities[j].z + box_z) - cavities[i].z;
    r2 = dx * dx + dy * dy + dz * dz;
    if (r2 < Dij2) return 1;
  }


  return 0;
}

int CavityGraph::cavityIsUnique(double x, double y, double z)
{
  int i;
  double dx, dy, dz;

  for (i=0; i < number_of_cavities; i++)
  {
    dx = cavities[i].x - x;
    dy = cavities[i].y - y;
    dz = cavities[i].z - z;
    if ((dx*dx + dy*dy + dz*dz) < (cavity_uniqueness_threshold * cavity_uniqueness_threshold)) return 0;
  }

  return 1;
}



/* rattlj.c */

/****************************************************************************

        *** LJ Version of rattle ***

        This assumes that all sigma/epsilon values are the same
        Even though the value is assigned explicitly when expanding
        a particular rattler.  There is no explicit cross-interaction.

  IN:   GFG (stdin)
  OUT:  N	x	y 	z
  	where N is the rattler number
	and x,y,z are the positions of the set of points in the cluster
	containing N

****************************************************************************/

#include <ftw_std.h>
#include <ftw_rng.h>
#include <ftw_param.h>
#include <string.h>

#define MAX_NUM_RATTLERS 65536
#define MAX_CLOSE 65536

int number_of_rattlers = 0;
int rattlers_of_interest;
double x[MAX_NUM_RATTLERS];
double y[MAX_NUM_RATTLERS];
double z[MAX_NUM_RATTLERS];
double sigma[MAX_NUM_RATTLERS];
double epsilon[MAX_NUM_RATTLERS];

double close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE], close_sigma[MAX_CLOSE];
int close_rattlers;

double box_x=6, box_y=6, box_z=6;
double verlet_cutoff=25.0;
double resolution = 0.01;
double diameter = 1.0;
double diameter_sq;
double T = 1.0;
double test_sigma, test_epsilon;

double test_x, test_y, test_z;
double verlet_center_x, verlet_center_y, verlet_center_z;
int rattler;
int recursion_matrix[256][256][256];

int makeVerletList();
void readConfiguration();
int checkInclusion(double test_x, double test_y, double test_z);
void visit(int _i, int _j, int _k);

int main(int argc, char *argv[])
{
  setCommandLineParameters(argc, argv);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getDoubleParam("-resolution", &resolution);
  getDoubleParam("-diameter", &diameter);
  getDoubleParam("-T", &T);
  diameter_sq = diameter * diameter;
  if (getFlagParam("-usage"))
  {
    printf("\nusage:\t-box [ 6.0 6.0 6.0 ]\n");
    printf("\t\t-verlet_cutoff [ 25.0 ]\n");
    printf("\t\t-resolution [ 0.01 ]\n");
    printf("\t\t-diameter [ 1.0 ]\n");
    printf("\t\t-rattlers_of_interest [ <<all in file>> ]\n");
    printf("\t\t-T [ 1.0 ]\n");
    printf("\n");
    exit(0);
  }

  readConfiguration();
  rattlers_of_interest = number_of_rattlers;
  getIntParam("-rattlers_of_interest", &rattlers_of_interest); // if we only want to look at a few...
  
  // loop over set of rattlers
  for (rattler = 0; rattler < rattlers_of_interest; rattler++)
  {
    int i, j, k;

    // clear recursion_matrix
    for (i=0; i<256; i++) for (j=0; j<256; j++) for (k=0; k<256; k++) recursion_matrix[i][j][k] = 0;
    // start recursion with rattler 0
    test_x = x[rattler];
    test_y = y[rattler];
    test_z = z[rattler];
    test_sigma = sigma[rattler];
    test_epsilon = epsilon[rattler];
    makeVerletList();
    visit(128, 128, 128);
  }

  return 0;
}

// visiting implies that site has already been determined to be included
void visit(int _i, int _j, int _k)
{
  recursion_matrix[_i][_j][_k] = 1; // mark the visited point
  test_x = verlet_center_x + (_i - 128) * resolution;
  test_y = verlet_center_y + (_j - 128) * resolution;
  test_z = verlet_center_z + (_k - 128) * resolution;
  printf("%06d\t%f\t%f\t%f\n", rattler, test_x, test_y, test_z);

  // check each neighbor for inclusion, and if included, then visit it...
  test_x = verlet_center_x + (_i - 129) * resolution;
  if (!recursion_matrix[_i - 1][_j][_k] && checkInclusion(test_x, test_y, test_z)) visit(_i - 1, _j, _k);
  test_x = verlet_center_x + (_i - 127) * resolution;
  if (!recursion_matrix[_i + 1][_j][_k] && checkInclusion(test_x, test_y, test_z)) visit(_i + 1, _j, _k);

  test_x = verlet_center_x + (_i - 128) * resolution;
  test_y = verlet_center_y + (_j - 129) * resolution;
  if (!recursion_matrix[_i][_j - 1][_k] && checkInclusion(test_x, test_y, test_z)) visit(_i, _j - 1, _k);
  test_y = verlet_center_y + (_j - 127) * resolution;
  if (!recursion_matrix[_i][_j + 1][_k] && checkInclusion(test_x, test_y, test_z)) visit(_i, _j + 1, _k);
  
  test_y = verlet_center_y + (_j - 128) * resolution;
  test_z = verlet_center_z + (_k - 129) * resolution;
  if (!recursion_matrix[_i][_j][_k - 1] && checkInclusion(test_x, test_y, test_z)) visit(_i, _j, _k - 1);
  test_z = verlet_center_z + (_k - 127) * resolution;
  if (!recursion_matrix[_i][_j][_k + 1] && checkInclusion(test_x, test_y, test_z)) visit(_i, _j, _k + 1);
}

int checkInclusion(double test_x, double test_y, double test_z)
{
  // printf("Checking inclusion of %lf, %lf, %lf\n", test_x, test_y, test_z);
  int i;
  double dx, dy, dz, dd;
  double r6, r12;
  double interaction = 0;

  for (i=0; i < close_rattlers; i++)
  {
    dx = test_x - close_x[i];
    dy = test_y - close_y[i];
    dz = test_z - close_z[i];
    dd = dx*dx + dy*dy + dz*dz;
    r6 = dd * dd * dd;
    r12 = r6 * r6;
    interaction += 4 * test_epsilon * (1/r12 - 1/r6);
  } 
 
  if (interaction < T) return 1;
  else return 0;
}

void readConfiguration()
{
  char line[80];
  char *xs, *ys, *zs;
  char *sigmas, *epsilons;

  number_of_rattlers = 0;

  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    sigmas = strtok(NULL, "\t");
    epsilons = strtok(NULL, "\n");

    x[number_of_rattlers] = strtod(xs, NULL);
    y[number_of_rattlers] = strtod(ys, NULL);
    z[number_of_rattlers] = strtod(zs, NULL);
    sigma[number_of_rattlers] = strtod(sigmas, NULL);
    epsilon[number_of_rattlers] = strtod(epsilons, NULL);
    number_of_rattlers++;
  }
 
  fclose(stdin);
}

int makeVerletList()
{
  int i;
  double dx, dy, dz, dd;
  double shift_x, shift_y, shift_z;

  while (test_x > box_x) test_x -= box_x;
  while (test_y > box_y) test_y -= box_y;
  while (test_z > box_z) test_z -= box_z;

  while (test_x < 0) test_x += box_x;
  while (test_y < 0) test_y += box_y;
  while (test_z < 0) test_z += box_z;

  verlet_center_x=test_x;
  verlet_center_y=test_y;
  verlet_center_z=test_z;

  close_rattlers=0;
  for (i=0; i<number_of_rattlers; i++)
  {
    if (i == rattler) continue; // don't want to count the rattler

    for (shift_x = -box_x; shift_x <= box_x; shift_x += box_x)
    for (shift_y = -box_y; shift_y <= box_y; shift_y += box_y)
    for (shift_z = -box_z; shift_z <= box_z; shift_z += box_z)
    {
      dx = shift_x + x[i] - test_x;
      dy = shift_y + y[i] - test_y;
      dz = shift_z + z[i] - test_z;

      dd = dx*dx + dy*dy + dz*dz;

      if (dd < verlet_cutoff) 
      { 
        close_x[close_rattlers] = shift_x + x[i];
        close_y[close_rattlers] = shift_y + y[i];
        close_z[close_rattlers] = shift_z + z[i];
        close_sigma[close_rattlers] = sigma[i];

        close_rattlers++;
      }
    }
  }
  return close_rattlers;
}


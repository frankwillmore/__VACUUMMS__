/* grafv.h */

#include "io_setup.h"

#include <math.h>
#include <ftw_std.h>
//#include <ftw_rng.h>
#include <ftw_param.h>

#define MAX_NUM_MOLECULES 16384
#define MAX_CLOSE 2048
#define PI 3.141592653589796264
#define MAX_SHELLS 1000

double x[MAX_NUM_MOLECULES];
double y[MAX_NUM_MOLECULES];
double z[MAX_NUM_MOLECULES];
double sigma[MAX_NUM_MOLECULES];
double epsilon[MAX_NUM_MOLECULES];

double close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE];
double close_sigmasq[MAX_CLOSE];

double box_x=6, box_y=6, box_z=6;
double verlet_cutoff=25.0;

int  points_per_shell = 1024;
int number_of_samples = 1;

double test_x, test_y, test_z;
int seed = 123450;

int number_of_molecules = 0;
int close_molecules;
double Rmax=1.0;
int n_shells=16;
int g[MAX_SHELLS];
double Rstep;
const double rand_step = 1/(0.0 + RAND_MAX);
FILE *instream;

int isOccupied(double, double, double);

int main(int argc, char *argv[])
{
  double sq_distance_from_initial_pt;
  double step_number;
  double dx, dy, dz;
  double phi, theta;
  double x_component, y_component, z_component;
  int shell_no, point_no;

  setCommandLineParameters(argc, argv);
  verbose = getFlagParam("-verbose");
  getIntParam("-seed", &seed);
  srand(seed);
//  if (getFlagParam("-randomize")) randomize();
//  else initializeRandomNumberGeneratorTo(seed);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getIntParam("-n", &number_of_samples);
  getIntParam("-points_per_shell", &points_per_shell);
  getIntParam("-n_shells", &n_shells);
  getDoubleParam("-Rmax", &Rmax);
  if (getFlagParam("-usage"))
  {
    printf("\nusage:\t-seed ddd [123450]\n");
//    printf("      \t-randomize\n");
    printf("      \t-box f.ff f.ff f.ff [6.0 6.0 6.0]\n");
    printf("      \t-verlet_cutoff f.ff [25.0]\n");
    printf("      \t-n ddd [1]\n");
    printf("      \t-points_per_shell ddd [1024]\n");
    printf("      \t-n_shells ddd [16]\n");
    printf("      \t-Rmax f.ff [1.0]\n\n");
    printf("      \tcofiguration (.gfg) in, histogram for g(r), available free volume pair correlation out.\n\n");
    exit(0);
  }

  Rstep=Rmax/n_shells;

  readConfiguration();
  
  while (number_of_samples-- > 0)
  {

    generateTestPoint();
//printf("test point:  %lf, %lf, %lf\n", test_x, test_y, test_z);

    for (shell_no=0; shell_no<=n_shells; shell_no++) g[shell_no] = 0;

    for (point_no=0; point_no<points_per_shell; point_no++)
    {
      // pick a direction
      phi = 2*PI*rand_step*rand();
      theta = PI*rand_step*rand();

      x_component = Rstep*cos(phi)*sin(theta); 
      y_component = Rstep*sin(phi)*sin(theta);
      z_component = Rstep*cos(theta);
      
      for (shell_no=1; shell_no<=n_shells; shell_no++)
      {
        g[shell_no] = g[shell_no] + isOccupied(test_x+shell_no*x_component, test_y+shell_no*y_component, test_z+shell_no*z_component);
        
      }
    }

  }

  for (shell_no=0; shell_no<=n_shells; shell_no++)
    printf("%lf\t%lf\n", shell_no*Rstep, (0.0 + points_per_shell-g[shell_no])/points_per_shell);

  return 0;
}

generateTestPoint()
{
  test_x = rand_step*rand() * box_x;
  test_y = rand_step*rand() * box_y;
  test_z = rand_step*rand() * box_z;

  makeVerletList();

  while (isOccupied(test_x, test_y, test_z)) generateTestPoint();
}

makeVerletList()
{
  int i;
  double dx, dy, dz, dd;
  double shift_x, shift_y, shift_z;

  close_molecules=0;

  for (i=0; i<number_of_molecules; i++)
  {
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
        close_x[close_molecules] = shift_x + x[i];
        close_y[close_molecules] = shift_y + y[i];
        close_z[close_molecules] = shift_z + z[i];
        close_sigmasq[close_molecules] = .25*sigma[i]*sigma[i];

        close_molecules++;
      }
    }
  }
}

int isOccupied(double px, double py, double pz)
{
  double dx, dy, dz, dd;
  int i;


  for (i=0; i<close_molecules; i++)
  {
    dx = close_x[i] - px;
    dy = close_y[i] - py;
    dz = close_z[i] - pz;
    dd = dx*dx + dy*dy + dz*dz;

    if (dd < close_sigmasq[i]) return 1;
  }

  return 0;
}

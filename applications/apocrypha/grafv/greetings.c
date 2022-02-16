#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "mpi.h"
#include <ftw_param.h>
#include <ftw_std.h>
#include <math.h>

#define MAX_CLOSE 1000
#define MAX_SHELLS 1000
#define PI 3.141592653589796264

int number_of_molecules;
double x[10000], y[10000], z[10000];
double sigma[10000], epsilon[10000];

double close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE];
double close_sigmasq[MAX_CLOSE];

double box_x=6, box_y=6, box_z=6;
double verlet_cutoff=25.0;

int points_per_shell = 1024;
int number_of_samples = 1;
double test_x, test_y, test_z;
int seed = 123450;
int number_of_molecules = 0;
int close_molecules;
double Rmax=1.0;
int n_shells=16;
int g[MAX_SHELLS], mpig[MAX_SHELLS];
double Rstep;
const double rand_step = 1/(0.0 + RAND_MAX);

void readConfiguration();
int isOccupied(double, double, double);
void makeVerletList();
void generateTestPoint();

main(int argc, char* argv[]) 
{
  double sq_distance_from_initial_pt;
  double step_number;
  double dx, dy, dz;
  double phi, theta;
  double x_component, y_component, z_component;
  int shell_no, point_no;

  //MPI stuff
  int my_rank;
  int p;
  int source; 
  int dest;
  int tag=0;
  //char message[100];
  MPI_Status status;

  setCommandLineParameters(argc, argv);
  if (getFlagParam("-usage"))
  {
    printf("\nusage:\t-seed ddd [123450]\n");
    printf("      \t-box f.ff f.ff f.ff [6.0 6.0 6.0]\n");
    printf("      \t-verlet_cutoff f.ff [25.0]\n");
    printf("      \t-n ddd [1]\n");
    printf("      \t-points_per_shell ddd [1024]\n");
    printf("      \t-n_shells ddd [16]\n");
    printf("      \t-Rmax f.ff [1.0]\n\n");
    printf("      \tcofiguration (.gfg) in, histogram for g(r), available free volume pair correlation out.\n\n");
    exit(0);
  }
  getIntParam("-seed", &seed);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getIntParam("-n", &number_of_samples);
  getIntParam("-points_per_shell", &points_per_shell);
  getIntParam("-n_shells", &n_shells);
  getDoubleParam("-Rmax", &Rmax);
  Rstep=Rmax/n_shells;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  srand(seed+my_rank);

  if (my_rank==0)
  {
    readConfiguration();
    for (dest=1; dest<p; dest++)
    {
      MPI_Send(&number_of_molecules, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
      MPI_Send(x, number_of_molecules, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
      MPI_Send(y, number_of_molecules, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
      MPI_Send(z, number_of_molecules, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
      MPI_Send(sigma, number_of_molecules, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
    }
  }
  else
  {
    dest=0;
    MPI_Recv(&number_of_molecules, 1, MPI_INT, dest, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(x, number_of_molecules, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(y, number_of_molecules, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(z, number_of_molecules, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(sigma, number_of_molecules, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
  }

  while (number_of_samples-- > 0)
  {
    generateTestPoint();
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

  if (my_rank==0)
  {
    for (dest=1; dest<p; dest++)
    {
      MPI_Recv(mpig, n_shells, MPI_INT, dest, tag, MPI_COMM_WORLD, &status);
      for (shell_no=0; shell_no<n_shells; shell_no++) g[shell_no] += mpig[shell_no];
    }
    for (shell_no=0; shell_no<=n_shells; shell_no++)
      printf("%lf\t%lf\n", shell_no*Rstep, (0.0 + p*points_per_shell-g[shell_no])/(p*points_per_shell));
  }
  else
  {
    MPI_Send(g, n_shells, MPI_INT, 0, tag, MPI_COMM_WORLD);
  }

  MPI_Finalize();
}

void readConfiguration()
{
  char line[80];
  char *xs, *ys, *zs;
  char *sigmas, *epsilons;

  number_of_molecules = 0;

  while (1)
  {
    fgets(line, 80, stdin);
    if (feof(stdin)) break;

    xs = strtok(line, "\t");
    ys = strtok(NULL, "\t");
    zs = strtok(NULL, "\t");
    sigmas = strtok(NULL, "\t");
    epsilons = strtok(NULL, "\n");

    x[number_of_molecules] = strtod(xs, NULL);
    y[number_of_molecules] = strtod(ys, NULL);
    z[number_of_molecules] = strtod(zs, NULL);
    sigma[number_of_molecules] = strtod(sigmas, NULL);
    epsilon[number_of_molecules] = strtod(epsilons, NULL);
    number_of_molecules++;
  }
 
  fclose(stdin);
}

void generateTestPoint()
{
  test_x = rand_step*rand() * box_x;
  test_y = rand_step*rand() * box_y;
  test_z = rand_step*rand() * box_z;

  makeVerletList();

  while (isOccupied(test_x, test_y, test_z)) generateTestPoint();
}

void makeVerletList()
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

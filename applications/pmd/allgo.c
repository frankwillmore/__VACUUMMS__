/* sdlj.c */

void readConfiguration();
double calculateEnergy();

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <ftw_std.h>
#include <ftw_param.h>
#include <time.h>

#define MAX_NUM_MOLECULES 16384
#define MAX_CLOSE 2048
#define TIME_BINS 20400
#define PI 3.141592653589796264

double x[MAX_NUM_MOLECULES];
double y[MAX_NUM_MOLECULES];
double z[MAX_NUM_MOLECULES];
double x_0[MAX_NUM_MOLECULES];
double y_0[MAX_NUM_MOLECULES];
double z_0[MAX_NUM_MOLECULES];
double t[MAX_NUM_MOLECULES];
double drift_x[MAX_NUM_MOLECULES];
double drift_y[MAX_NUM_MOLECULES];
double drift_z[MAX_NUM_MOLECULES];
double sigma[MAX_NUM_MOLECULES];
double epsilon[MAX_NUM_MOLECULES];

double close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE];
double close_sigma[MAX_CLOSE];
double close_epsilon[MAX_CLOSE];

double box_x=6, box_y=6, box_z=6;
double verlet_cutoff=100.0;
double time_series_R_delta_t[TIME_BINS];
double time_series_R_sq_delta_t[TIME_BINS];
double time_series_delta_t[TIME_BINS];

int active_molecule;

double step_size = 0.05;

double T=1.0;
double energy, new_energy;

int number_of_samples = 1;
int time_series=0;

double collision_x, collision_y, collision_z;
double verlet_center_x, verlet_center_y, verlet_center_z;

double test_diameter = 1; 
double test_epsilon = 1;
double target_time = 1000.0; // picoseconds
int bin_size=10;

int seed = 123450;

int report_frequency=10;
int number_of_steps = 1024;
int number_of_molecules = 0;
int close_molecules;

double min_time=0;

/* NOTE:  sigma is specified in Angstroms, epsilon in K, T in K */
const double rand_step = 1.0/RAND_MAX;

int main(int argc, char *argv[])
{
  double sq_distance_from_initial_pt; // 
  double n_steps_taken; // how many steps from initial insertion
  double dx, dy, dz, dd, dt, d; // distance deltas...
  double phi, theta; // used for choosing direction
  double x_step, y_step, z_step; // step size in that direction
  double mid_x, mid_y, mid_z, mid_t;
  double R_t;
  int t_bin;
  double time_elapsed=0.0;
  double collision_t;
  double kinetic_energy;
  double old_energy, new_energy;
  double velocity;
  double time_step;
  double D;
  int bisections;
  int i;
  double bisection_factor;
  double R_sq;

  setCommandLineParameters(argc, argv);
  verbose = getFlagParam("-verbose");
  getIntParam("-seed", &seed);
  if (getFlagParam("-randomize")) seed=time(NULL);
  srand(seed);
  if (getFlagParam("-usage")) 
  {
    printf("\nusage:  configuration in, list of path lengths  and times out\n\n");
    printf("        -box [ 6.0 6.0 6.0 ] (sigma)\n");
    printf("        -T [ 1.0 ]\n");
    printf("        -N [ 1024 ]\n");
    printf("        -step_size [ .050 ]\n");
    printf("        -target_time [ 100.0 ]\n");
    printf("        -bin_size [ 1 ]\n");
    printf("        -seed [ 123450 ]\n");
    printf("        -randomize\n\n");

    exit(0);
  }

  srand(seed);

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-step_size", &step_size);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getDoubleParam("-test_diameter", &test_diameter);
  getDoubleParam("-target_time", &target_time);
  target_time;
  getDoubleParam("-test_epsilon", &test_epsilon);
  getDoubleParam("-T", &T);
  getIntParam("-n", &number_of_samples);
  getIntParam("-N", &number_of_steps);
  getIntParam("-bin_size", &bin_size);
  time_series = getFlagParam("-time_series");

  readConfiguration();

  for (i=0; i<1000; i++)
  {
    //time_series_R_delta_t[i] = 0;
    time_series_R_sq_delta_t[i] = 0;
    time_series_delta_t[i] = 0;
  }

  // loop until time is up
  while (time_elapsed<target_time)
  {
    // pick a particle
    active_molecule = (rand_step*rand())*number_of_molecules;
fprintf(stderr, "active molecule:  %d\n", active_molecule);
    // additional selection criteria so all particles get to move until enough time elapses

    // pick a direction
    phi = 2*PI*rand()*rand_step;
    theta = PI*rand()*rand_step;
    x_step = step_size*cos(phi)*sin(theta); 
    y_step = step_size*sin(phi)*sin(theta);
    z_step = step_size*cos(theta);

    // pick a velocity
    kinetic_energy = -1.5*T*log(rand_step*rand()); 
    velocity = sqrt(2.0*kinetic_energy);
//fprintf(stderr, "v=%lf\n", velocity);
    collision_x = x[active_molecule];
    collision_y = y[active_molecule];
    collision_z = z[active_molecule];
    collision_t = t[active_molecule];
    bisection_factor=1.0;
    old_energy = calculateEnergy();

    // extend ray from collision point
    bisections=0; 
    while (bisections<=15)
    {
      x[active_molecule] += x_step;
      y[active_molecule] += y_step;
      z[active_molecule] += z_step;

      new_energy = calculateEnergy();

      if ((new_energy > (kinetic_energy)) && (new_energy >= old_energy))
      {
        // new_energy = old_energy;
        x[active_molecule]-=x_step;
        y[active_molecule]-=y_step;
        z[active_molecule]-=z_step;

        x_step*=.5;
        y_step*=.5;
        z_step*=.5;
        bisection_factor*=.5;
        bisections++;
      }

//fprintf(stderr, "%f\t%lf\t%lf\n", x[active_molecule], y[active_molecule], z[active_molecule]);
        
    } // end loop over bisections
    
    // got the new position

    mid_x = (x[active_molecule] + collision_x) * .5;
    mid_y = (y[active_molecule] + collision_y) * .5;
    mid_z = (z[active_molecule] + collision_z) * .5;

    dx = mid_x - collision_x;
    dy = mid_y - collision_y;
    dz = mid_z - collision_z;

    dt = sqrt(dx*dx + dy*dy + dz*dz)/velocity;

    // now get displacement from overall starting point
    dx = drift_x[active_molecule] + mid_x - x_0[active_molecule];
    dy = drift_y[active_molecule] + mid_y - y_0[active_molecule];
    dz = drift_z[active_molecule] + mid_z - z_0[active_molecule];
    R_sq = dx*dx + dy*dy + dz*dz;

    t_bin = floor((time_elapsed + .5*dt) + .5);

//  time_series_R_delta_t[t_bin] += R_t*dt;
    time_series_R_sq_delta_t[t_bin] += R_sq*dt;
    time_series_delta_t[t_bin]+= dt;

    t[active_molecule] += dt;

    x[active_molecule] = mid_x;
    y[active_molecule] = mid_y;
    z[active_molecule] = mid_z;

    // correct for periodic boundaries

    while (x[active_molecule] >= box_x) {x[active_molecule] -= box_x; drift_x[active_molecule] += box_x;}
    while (y[active_molecule] >= box_y) {y[active_molecule] -= box_y; drift_y[active_molecule] += box_y;}
    while (z[active_molecule] >= box_z) {z[active_molecule] -= box_z; drift_z[active_molecule] += box_z;}

    while (x[active_molecule] < box_x) {x[active_molecule] += box_x; drift_x[active_molecule] -= box_x;}
    while (y[active_molecule] < box_y) {y[active_molecule] += box_y; drift_y[active_molecule] -= box_y;}
    while (z[active_molecule] < box_z) {z[active_molecule] += box_z; drift_z[active_molecule] -= box_z;}

    time_elapsed=0;
    for (i=0; i<number_of_molecules; i++) time_elapsed += t[i];
    time_elapsed /= number_of_molecules;
fprintf(stderr,"time elapsed=%lf\n", time_elapsed);

  } // end loop until target_time 
  
  for (i=0; i*bin_size < time_elapsed; i++) printf("%d\t%lf\n", i*bin_size, time_series_R_sq_delta_t[i]/time_series_delta_t[i]);

  return 0;
}

// returns the energy of all interactions with active molecule
double calculateEnergy()
{
  double repulsion=0;
  double attraction=0;
  double shift_x, shift_y, shift_z;
  double dx, dy, dz, dd, d6, d12;
  int i;

  for (shift_x=-box_x; shift_x <= box_x; shift_x+=box_x)
  for (shift_y=-box_y; shift_y <= box_y; shift_y+=box_y)
  for (shift_z=-box_z; shift_z <= box_z; shift_z+=box_z)
  for (i=0; i<number_of_molecules; i++)
  {
    if (i != active_molecule)
    {
      dx = shift_x + (drift_x[i] + x[i]) - (drift_x[active_molecule] + x[active_molecule]);
      dy = shift_y + (drift_y[i] + y[i]) - (drift_y[active_molecule] + y[active_molecule]);
      dz = shift_z + (drift_z[i] + z[i]) - (drift_z[active_molecule] + z[active_molecule]);
      dd = dx*dx + dy*dy + dz*dz;

      d6=dd*dd*dd;
      d12=d6*d6;

      attraction += 1.0 / d6;
      repulsion += 1.0 / d12;
    }
  }

  return (4 * (repulsion - attraction));
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

    x_0[number_of_molecules] = x[number_of_molecules] = strtod(xs, NULL);
    y_0[number_of_molecules] = y[number_of_molecules] = strtod(ys, NULL);
    z_0[number_of_molecules] = z[number_of_molecules] = strtod(zs, NULL);

    drift_x[number_of_molecules] = 0;
    drift_y[number_of_molecules] = 0;
    drift_z[number_of_molecules] = 0;
    t[number_of_molecules] = 0;

    number_of_molecules++;
  }
 
fprintf(stderr, "%d lines read.\n", number_of_molecules);
  fclose(stdin);
}

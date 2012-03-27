/* sdlj2p.c */

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
#define TIME_BINS 20400
#define PI 3.141592653589796264

double x[MAX_NUM_MOLECULES];
double y[MAX_NUM_MOLECULES];
double z[MAX_NUM_MOLECULES];
double sigma[MAX_NUM_MOLECULES];
double epsilon[MAX_NUM_MOLECULES];

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
double collision_x, collision_y, collision_z;
double test_diameter = 1; 
double test_epsilon = 1;
double target_time = 1000.0; // picoseconds
int seed = 123450;
int n_steps = 0;
int number_of_molecules = 0;
double psi_shift=0.004079222784;
double cutoff_sq=6.25;
double error_term=0.0;

double drift_x, drift_y, drift_z;

double t=0;
double dd_sum=0;
double dd_avg;

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

  double dx_forward, dy_forward, dz_forward;
  double dx_backward=0, dy_backward=0, dz_backward=0;

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
    printf("        -step_size [ .050 ]\n");
    printf("        -target_time [ 1000.0 ]\n");
    printf("        -bin_size [ 1 ]\n");
    printf("        -seed [ 123450 ]\n");
    printf("        -randomize\n\n");
    printf("        -cutoff_sq [ 6.25 ]\n");
    printf("        -psi_shift [ 0.004079222784 ]\n");
    printf("        -error_term [ 0.0 ]\n");

    exit(0);
  }

  srand(seed);

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-step_size", &step_size);
  getDoubleParam("-target_time", &target_time);
  getDoubleParam("-T", &T);
  getDoubleParam("-cutoff_sq", &cutoff_sq);
  getDoubleParam("-psi_shift", &psi_shift);
  getDoubleParam("-error_term", &error_term);

  readConfiguration();

  // loop until time is up
  t=0.0;
  while (t<target_time)
  {
    // pick a particle
    active_molecule = (rand_step*rand())*number_of_molecules;

    // pick a direction
    phi = 2*PI*rand()*rand_step;
    theta = acos(((rand()*rand_step) * 2) - 1);
    //theta = PI*rand()*rand_step;
    x_step = step_size*cos(phi)*sin(theta); 
    y_step = step_size*sin(phi)*sin(theta);
    z_step = step_size*cos(theta);

    // pick a velocity
    kinetic_energy = -1.5*T*log(rand_step*rand()); 
    velocity = sqrt(2.0*kinetic_energy);
    // now add energy of a second particle
    kinetic_energy -= 1.5*T*log(rand_step*rand()); 

    // extend ray FORWARD from collision point

    collision_x = x[active_molecule];
    collision_y = y[active_molecule];
    collision_z = z[active_molecule];

    drift_x = 0;
    drift_y = 0;
    drift_z = 0;

    bisection_factor=1.0;
    old_energy = calculateEnergy();

    bisections=0; 
    while (bisections<=15)
    {
      collision_x += x_step*bisection_factor;
      collision_y += y_step*bisection_factor;
      collision_z += z_step*bisection_factor;

      recenter_position();

      new_energy = calculateEnergy();

      if ((new_energy > (kinetic_energy)) && (new_energy >= old_energy))
      {
        // new_energy = old_energy;
        collision_x -= (x_step*bisection_factor);
        collision_y -= (y_step*bisection_factor);
        collision_z -= (z_step*bisection_factor);

        bisection_factor*=.5;
        bisections++;
      }
    } 

    // done with forward motion, get displacement

    dx_forward = drift_x + collision_x - x[active_molecule];
    dy_forward = drift_y + collision_y - y[active_molecule];
    dz_forward = drift_z + collision_z - z[active_molecule];

/**/

    // extend ray BACKWARD from collision point

    collision_x = x[active_molecule];
    collision_y = y[active_molecule];
    collision_z = z[active_molecule];

    drift_x = 0;
    drift_y = 0;
    drift_z = 0;
    
    bisection_factor=1.0;
    old_energy = calculateEnergy();

    bisections=0; 
    while (bisections<=15)
    {
      collision_x -= x_step*bisection_factor;
      collision_y -= y_step*bisection_factor;
      collision_z -= z_step*bisection_factor;

      recenter_position();

      new_energy = calculateEnergy();

      if ((new_energy > (kinetic_energy)) && (new_energy >= old_energy))
      {
        // new_energy = old_energy;
        collision_x += (x_step*bisection_factor);
        collision_y += (y_step*bisection_factor);
        collision_z += (z_step*bisection_factor);

        bisection_factor*=.5;
        bisections++;
      }

    } // end loop over bisections
    
    dx_backward = drift_x + collision_x - x[active_molecule];
    dy_backward = drift_y + collision_y - y[active_molecule];
    dz_backward = drift_z + collision_z - z[active_molecule];

/**/

    dx = dx_forward - dx_backward;
    dy = dy_forward - dy_backward;
    dz = dz_forward - dz_backward;

    dd = dx*dx + dy*dy + dz*dz;
    d = sqrt(dd);

// error term

//    d += (rnd() - 0.5) * error_term;
    d += ((rand() * rand_step - 0.5) * error_term);
    dd = d*d;

// error term

    dt = d/velocity;

    t += dt;
    dd_sum += dd;
    dd_avg = dd_sum/++n_steps;

    R_sq = n_steps * dd_avg;
    D = R_sq/(6*t);

    printf("%lf\t%lf\t%lf\n", t, dd_avg, D);

// printf("%lf\t%lf\n", d, dd);

// printf("%lf\t%lf\t%lf\n", dx, dy, dz);

  } // end loop until target_time 
  
//  for (i=0; i*bin_size < time_elapsed; i++) printf("%d\t%lf\n", i*bin_size, time_series_R_sq_delta_t[i]/time_series_delta_t[i]);

  return 0;
}

recenter_position()
{
    while (collision_x >= box_x)
    {
      drift_x += box_x;
      collision_x -= box_x;
    }

    while (collision_x < 0)
    {
      drift_x -= box_x;
      collision_x += box_x;
    }

    while (collision_y >= box_y)
    {
      drift_y += box_y;
      collision_y -= box_y;
    }

    while (collision_y < 0)
    {
      drift_y -= box_y;
      collision_y += box_y;
    }

    while (collision_z >= box_z)
    {
      drift_z += box_z;
      collision_z -= box_z;
    }

    while (collision_z < 0)
    {
      drift_z -= box_z;
      collision_z += box_z;
    }
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
      dx = shift_x + x[i] - collision_x;
      dy = shift_y + y[i] - collision_y;
      dz = shift_z + z[i] - collision_z;
      dd = dx*dx + dy*dy + dz*dz;

      if (dd<cutoff_sq) 
      {
        d6=dd*dd*dd;
        d12=d6*d6;

        attraction += 1.0 / d6;
        repulsion += 1.0 / d12;
        repulsion += psi_shift;

      }
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

    x[number_of_molecules] = strtod(xs, NULL);
    y[number_of_molecules] = strtod(ys, NULL);
    z[number_of_molecules] = strtod(zs, NULL);

    number_of_molecules++;
  }
 
  fclose(stdin);
}

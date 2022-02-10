/* utrwt69.c */

/************************************************************************/
/*									*/
/* (C) 2006 Frank T Willmore.  Freely distributable under the terms of	*/
/* BSD license.  It is requested that any work using or extending this	*/
/* software reference the following publication:			*/
/*									*/
/* "Gas diffusion in glasses via a probabilistic molecular dynamics."   */
/* Willmore FT, Wang XY, Sanchez IC.					*/
/* JOURNAL OF CHEMICAL PHYSICS 126, 234502 JUNE 15 2007			*/
/*									*/
/* To contact the developer, please email frankwillmore@gmail.com.	*/
/*									*/
/************************************************************************/

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

double step_size = 0.10;

double T=298;
double energy, new_energy;

int detail=0;
int gradient=0;
int number_of_samples = 1;
int time_series=0;

double test_x0, test_y0, test_z0;
double test_x, test_y, test_z;
double collision_x, collision_y, collision_z;
double verlet_center_x, verlet_center_y, verlet_center_z;

// Default particle is Helium
double test_diameter = 2.9; // Angstroms
double test_epsilon = 2.52; // in K
double test_mass = 0.004; // kg/mol
double target_time = 100.0; // picoseconds
double threshold_time = 10.0; // picoseconds

int seed = 123450;
int successes;

int bin_size=1;
int report_frequency=10;
int number_of_steps = 1024;
int number_of_molecules = 0;
int close_molecules;
double drift_x, drift_y, drift_z;

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
  double time_elapsed;
  double collision_t;
  double kinetic_energy;
  double old_energy, new_energy;
  double velocity;
  double time_step;
  double D;
  int bisections;
  int i;
  double intercollision_distance;
  double bisection_factor;
  double grad_x, grad_y, grad_z, grad;

  setCommandLineParameters(argc, argv);
  verbose = getFlagParam("-verbose");
  getIntParam("-seed", &seed);
  if (getFlagParam("-randomize")) seed=time(NULL);
  srand(seed);
  if (getFlagParam("-usage")) 
  {
    printf("\nusage:  configuration in, list of path lengths  and times out\n\n");
    printf("        -box [ 6.0 6.0 6.0 ] (Angstroms)\n");
    printf("        -test_diameter [ 2.9 ] (Angstroms)\n");
    printf("        -test_epsilon [ 2.52 ] (K)\n");
    printf("        -test_mass [ 0.004 ] (kg)\n");
    printf("        -T [ 298.0 ]\n");
    printf("        -n [ 1 ]\n");
    printf("        -N [ 1024 ]\n");
    printf("        -step_size [ .100 ]\n");
    printf("        -verlet_cutoff [ 100.0 ]\n");
    printf("        -target_time (in picoseconds) [ 100.0 ]\n");
    printf("        -threshold_time (in picoseconds) [ 10.0 ]\n");
    printf("        -report_frequency (in picoseconds) [ 10 ]\n");
    printf("        -bin_size(in picoseconds) [ 1 ]\n");
    printf("        -seed [ 123450 ]\n");
    printf("        -detail\n");
    printf("        -time_series\n");
    printf("        -gradient\n");
    printf("        -randomize\n\n");

    exit(0);
  }

  srand(seed);

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-step_size", &step_size);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getDoubleParam("-test_diameter", &test_diameter);
  getDoubleParam("-target_time", &target_time);
  target_time*=1.0e-12;
  getDoubleParam("-threshold_time", &threshold_time);
  threshold_time*=1.0e-12;
  getDoubleParam("-test_epsilon", &test_epsilon);
  getDoubleParam("-test_mass", &test_mass);
  getDoubleParam("-T", &T);
  getIntParam("-n", &number_of_samples);
  getIntParam("-N", &number_of_steps);
  getIntParam("-bin_size", &bin_size);
  detail = getFlagParam("-detail");
  time_series = getFlagParam("-time_series");
  gradient = getFlagParam("-gradient");

  readConfiguration();

  for (i=0; i<1000; i++)
  {
    time_series_R_delta_t[i] = 0;
    time_series_R_sq_delta_t[i] = 0;
    time_series_delta_t[i] = 0;
  }

  // loop over # of insertions
  while (number_of_samples-- > 0)
  {
    generateTestPoint();
    new_energy = old_energy = calculateEnergy();
    drift_x=drift_y=drift_z=0;
    time_elapsed=0;

    while (time_elapsed<target_time)
    {
      if (gradient)
      {
        // direction of gradient
        test_x+=.000001;
        grad_x=calculateEnergy();
        test_x-=.000002;
        grad_x-=calculateEnergy(); 
        test_x+=.000001;

        test_y+=.000001;
        grad_y=calculateEnergy();
        test_y-=.000002;
        grad_y-=calculateEnergy(); 
        test_y+=.000001;

        test_z+=.000001;
        grad_z=calculateEnergy();
        test_z-=.000002;
        grad_z-=calculateEnergy(); 
        test_z+=.000001;

        grad=sqrt(grad_x*grad_x + grad_y*grad_y + grad_z*grad_z);
        x_step=-step_size*grad_x/grad;
        y_step=-step_size*grad_y/grad;
        z_step=-step_size*grad_z/grad;
      }
      else
      {
        // pick a direction
        phi = 2*PI*rand()*rand_step;
        theta = PI*rand()*rand_step;
        x_step = step_size*cos(phi)*sin(theta); 
        y_step = step_size*sin(phi)*sin(theta);
        z_step = step_size*cos(theta);
      }

      // pick a velocity
      kinetic_energy = -1.5*log(rand_step*rand())*8.314*T; // in J/mol
      velocity = sqrt(2.0*kinetic_energy/test_mass) * 1.0e+10; // in A/s
      collision_x = test_x;
      collision_y = test_y;
      collision_z = test_z;
      collision_t = time_elapsed;
      intercollision_distance=0;
      bisection_factor=1.0;

      // extend ray from collision point
      for (bisections=0; bisections<=15; )
      {
        test_x += x_step;
        test_y += y_step;
        test_z += z_step;

        dx=test_x - verlet_center_x;
        dy=test_y - verlet_center_y;
        dz=test_z - verlet_center_z;

        if (dx*dx + dy*dy + dz*dz > .01 * verlet_cutoff) makeVerletList();

        old_energy = new_energy;
        new_energy = calculateEnergy();
        if ((new_energy > (kinetic_energy / 8.314)) && (new_energy >= old_energy))
        {
          new_energy = old_energy;

          test_x-=x_step;
          test_y-=y_step;
          test_z-=z_step;

          x_step*=.5;
          y_step*=.5;
          z_step*=.5;
          bisection_factor*=.5;

          bisections++;
        }
        
        // figure out what time it is
        intercollision_distance+=step_size*bisection_factor;
        time_elapsed += step_size*bisection_factor/velocity;
        if (detail && (time_elapsed > threshold_time)) 
        {
          dx = test_x - test_x0 + drift_x;
          dy = test_y - test_y0 + drift_y;
          dz = test_z - test_z0 + drift_z;
          dd = dx*dx + dy*dy + dz*dz;
          d = sqrt(dd);
          D=1.0e-16*dd/(6*time_elapsed);
          printf("%1.12lf\t%6.0lf\t%1.12lf\t%1.12lf\t%1.12lf\t%1.12lf\n", d, 1.0e+12*time_elapsed, D, test_x, test_y, test_z);
        }
      } // end loop over bisections

      if (time_series)
      {
        mid_x = (test_x+collision_x) * .5;
        mid_y = (test_y+collision_y) * .5;
        mid_z = (test_z+collision_z) * .5;
        dx=mid_x-test_x0+drift_x;
        dy=mid_y-test_y0+drift_y;
        dz=mid_z-test_z0+drift_z;
        R_t = sqrt(dx*dx + dy*dy + dz*dz);
        dt = time_elapsed - collision_t;
        mid_t = collision_t + (dt * .5);

        t_bin = floor(1.0e+12*mid_t/bin_size + .5);

        // t_bin = floor(1000000000000*mid_t + .5);

        time_series_R_delta_t[t_bin] += R_t*dt;
        time_series_R_sq_delta_t[t_bin] += R_t*R_t*dt;
        time_series_delta_t[t_bin]+= dt;
      }

    } // end loop until target_time 
  
    // total distance traveled for this insertion
    dx = test_x - test_x0 + drift_x;
    dy = test_y - test_y0 + drift_y;
    dz = test_z - test_z0 + drift_z;
    dd = dx*dx + dy*dy + dz*dz;
    d = sqrt(dd);

  } // end loop over all samples
 
  if (time_series) for (i=0; i*bin_size < (time_elapsed * 1.0e+12); i++) 
    printf("%d\t%lf\n", i*bin_size, time_series_R_sq_delta_t[i]/time_series_delta_t[i]);

  return 0;
}

generateTestPoint()
{
  test_x = test_x0 = rand() * rand_step * box_x;
  test_y = test_y0 = rand() * rand_step * box_y;
  test_z = test_z0 = rand() * rand_step * box_z;

  makeVerletList();

  while (rand() * rand_step > exp(-(calculateEnergy()/T))) 
  {
    test_x = test_x0 = rand() * rand_step * box_x;
    test_y = test_y0 = rand() * rand_step * box_y;
    test_z = test_z0 = rand() * rand_step * box_z;

    makeVerletList();
  }
}

makeVerletList()
{
  int i;
  double dx, dy, dz, dd;
  double shift_x, shift_y, shift_z;
  double close_sigma6;

  while (test_x > box_x)
  {
    test_x -= box_x;
    collision_x -= box_x;
    drift_x += box_x;
  }
  while (test_y > box_y) 
  {
    test_y -= box_y;
    collision_y -= box_y;
    drift_y += box_y;
  }
  while (test_z > box_z) 
  {
    test_z -= box_z;
    collision_z -= box_z;
    drift_z += box_z;
  }

  while (test_x < 0)
  {
    test_x += box_x;
    collision_x += box_x;
    drift_x -= box_x;
  }
  while (test_y < 0)
  {
    test_y += box_y;
    collision_y += box_y;
    drift_y -= box_y;
  }
  while (test_z < 0)
  {
    test_z += box_z;
    collision_z += box_z;
    drift_z -= box_z;
  }

  verlet_center_x=test_x;
  verlet_center_y=test_y;
  verlet_center_z=test_z;

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

//        close_sigma6=((sigma[i]*sigma[i]*sigma[i]*sigma[i]*sigma[i]*sigma[i]) +(test_diameter*test_diameter*test_diameter*test_diameter*test_diameter*test_diameter))*.5;
//        close_sigma[close_molecules] = pow(close_sigma6, .1666666666666666666666666666);
//        close_epsilon[close_molecules] = sqrt(test_epsilon * epsilon[i]) * (test_diameter * sigma[i] * test_diameter * sigma[i] * test_diameter * sigma[i]) / close_sigma6;

        close_sigma[close_molecules] = sigma[i];
        close_epsilon[close_molecules] = epsilon[i];     

        close_molecules++;
      }
    }
  }
}

// returns the energy change by insertion of a test particle at test_x, test_y, test_z
double calculateEnergy()
{
  double repulsion=0;
  double attraction=0;
  double alpha;
  double dx, dy, dz, d, dd;
  int i;

  for (i=0; i<close_molecules; i++)
  {
    dx = close_x[i] - test_x;
    dy = close_y[i] - test_y;
    dz = close_z[i] - test_z;
    dd = dx*dx + dy*dy + dz*dz;

    d=sqrt(dd);
    alpha = close_sigma[i]*close_sigma[i]*close_sigma[i]/(d*dd);

    repulsion += close_epsilon[i] * alpha*alpha*alpha;
    attraction += close_epsilon[i] * alpha*alpha;
  }

  return (2*repulsion - 3*attraction);
}

void readConfiguration()
{
  char line[80];
  char *xs, *ys, *zs;
  char *sigmas, *epsilons;
  double sigmad, epsilond, sigma6;

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
    //sigma[number_of_molecules] = strtod(sigmas, NULL);
    sigmad = strtod(sigmas, NULL);
    sigma6 = ((sigmad*sigmad*sigmad*sigmad*sigmad*sigmad)+(test_diameter*test_diameter*test_diameter*test_diameter*test_diameter*test_diameter))*.5;
    sigma[number_of_molecules] = pow(sigma6, .1666666666666666666666666666);
    epsilond = strtod(epsilons, NULL);
    epsilon[number_of_molecules] = sqrt(test_epsilon*epsilond)*(test_diameter*sigmad*test_diameter*sigmad*test_diameter*sigmad)/sigma6;

    number_of_molecules++;
  }
 
  V printf("%d lines read.\n", number_of_molecules);
  fclose(stdin);
}

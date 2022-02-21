/* utrw.c */

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
#define PI 3.141592653589796264

double x[MAX_NUM_MOLECULES];
double y[MAX_NUM_MOLECULES];
double z[MAX_NUM_MOLECULES];
double sigma[MAX_NUM_MOLECULES];
double epsilon[MAX_NUM_MOLECULES];

double close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE];
double close_sigma[MAX_CLOSE];
double close_epsilon[MAX_CLOSE];
double close_sigma6[MAX_CLOSE];
double close_sigma12[MAX_CLOSE];

double box_x=6, box_y=6, box_z=6;
double verlet_cutoff=100.0;

double step_size = 0.001;

double T=298;
double energy, new_energy;

int detail=0;
int number_of_samples = 1;

double test_x0, test_y0, test_z0;
double test_x, test_y, test_z;
double collision_x, collision_y, collision_z;
double verlet_center_x, verlet_center_y, verlet_center_z;

// Default particle is Helium
double test_diameter = 2.556; // Angstroms
double test_epsilon = 10.223; // in K
double test_mass = 0.004; // kg/mol

int seed = 123450;
int successes;

int number_of_steps = 1024;
int number_of_molecules = 0;
int close_molecules;
double drift_x, drift_y, drift_z;
double collision_energy=1.5;

/* NOTE:  sigma is specified in Angstroms, epsilon in K, T in K */
const double rand_step = 1.0/RAND_MAX;

int main(int argc, char *argv[])
{
  double sq_distance_from_initial_pt; // 
  double n_steps_taken; // how many steps from initial insertion
  double dx, dy, dz, dd, dt, d; // distance deltas...
  double phi, theta; // used for choosing direction
  double x_step, y_step, z_step; // step size in that direction
  double time_elapsed;
  double kinetic_energy;
  double old_energy, new_energy;
  double velocity;
  double time_step;
  double D;
  int bisections;
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
    printf("        -test_diameter [ 2.556 ] (Angstroms)\n");
    printf("        -test_epsilon [ 0.085 ] (k)\n");
    printf("        -test_mass [ 0.004 ] (kg)\n");
    printf("        -T [ 298.0 ]\n");
    printf("        -n [ 1 ]\n");
    printf("        -N [ 1024 ]\n");
    printf("        -step_size [ .001 ]\n");
    printf("        -verlet_cutoff [ 100.0 ]\n");
    printf("        -seed [ 123450 ]\n");
    printf("        -collision_energy [ 1.5 ] \n");
    printf("        -randomize\n\n");

    exit(0);
  }

  srand(seed);

  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-step_size", &step_size);
  getDoubleParam("-collision_energy", &collision_energy);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getDoubleParam("-test_diameter", &test_diameter);
  getDoubleParam("-test_epsilon", &test_epsilon);
  getDoubleParam("-test_mass", &test_mass);
  getDoubleParam("-T", &T);
  getIntParam("-n", &number_of_samples);
  getIntParam("-N", &number_of_steps);
  detail = getFlagParam("-detail");

  readConfiguration();
  
  // loop over # of insertions
  while (number_of_samples-- > 0)
  {
    generateTestPoint();
    new_energy = old_energy = calculateEnergy();
    drift_x=drift_y=drift_z=0;
    time_elapsed=0;

    // looping over the specified number of translation steps for insertion
    for(n_steps_taken=0; n_steps_taken<number_of_steps; n_steps_taken++)
    {

/* 
      // pick a direction
      phi = 2*PI*rand()*rand_step;
      theta = PI*rand()*rand_step;
      x_step = step_size*cos(phi)*sin(theta); 
      y_step = step_size*sin(phi)*sin(theta);
      z_step = step_size*cos(theta);
*/

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

      // pick a velocity
      kinetic_energy = -1.5*log(rand_step*rand())*8.314*T; // in J/mol
      velocity = sqrt(2.0*kinetic_energy/test_mass) * 1.0e+10; // in A/s
      collision_x = test_x;
      collision_y = test_y;
      collision_z = test_z;

      bisections = 0;

      // extend ray from collision point
      while (1)
      {
        test_x += x_step;
        test_y += y_step;
        test_z += z_step;

        dx=test_x - verlet_center_x;
        dy=test_y - verlet_center_y;
        dz=test_z - verlet_center_z;

        if (dx*dx + dy*dy + dz*dz > .01 * verlet_cutoff) makeVerletList();

        // test for next 'collision'
        // if (calculateEnergy() > (collision_energy * T))
        // (J/mol) * (mol*K/J) / K = K

        // if (calculateEnergy() > (kinetic_energy / 8.314))
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
//printf("steps:\t%lf\t%lf\t%lf\n",x_step, y_step, z_step);
          
          if (bisections++ > 15) break;
        }
      }

      dx = test_x - collision_x;
      dy = test_y - collision_y;
      dz = test_z - collision_z;
      dd = dx*dx + dy*dy + dz*dz;
      d = sqrt(dd);
      dt = d/velocity;
      time_elapsed += dt;

if (detail) printf("detail trajectory:  %lf\t%lf\t%lf\t%lf\n", test_x, test_y, test_z, d);
    } 

  
    // total distance traveled for this insertion
    dx = test_x - test_x0 + drift_x;
    dy = test_y - test_y0 + drift_y;
    dz = test_z - test_z0 + drift_z;
    dd = dx*dx + dy*dy + dz*dz;
    d = sqrt(dd);

    // ignore the ones that don't go anywhere
    if (time_elapsed == 0) number_of_samples--;
    else 
    {
      D=1.0e-16*dd/(6*time_elapsed);
      // output the distance traveled, time elapsed, and D in cm^2/s
      printf("%lf\t%1.12lf\t%1.12lf\n", d, time_elapsed, D);
    }

  }

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

  while (test_x > box_x)
  {
    test_x -= box_x;
    collision_x -= box_x;
    drift_x += box_x;
//printf("PBC+x\n");
  }
  while (test_y > box_y) 
  {
    test_y -= box_y;
    collision_y -= box_y;
    drift_y += box_y;
//printf("PBC+y\n");
  }
  while (test_z > box_z) 
  {
    test_z -= box_z;
    collision_z -= box_z;
    drift_z += box_z;
//printf("PBC+z\n");
  }

  while (test_x < 0)
  {
    test_x += box_x;
    collision_x += box_x;
    drift_x -= box_x;
//printf("PBC-x\n");
  }
  while (test_y < 0)
  {
    test_y += box_y;
    collision_y += box_y;
    drift_y -= box_y;
//printf("PBC-y\n");
  }
  while (test_z < 0)
  {
    test_z += box_z;
    collision_z += box_z;
    drift_z -= box_z;
//printf("PBC-z\n");
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
        close_sigma[close_molecules] = sigma[i];
        close_epsilon[close_molecules] = epsilon[i];
        close_sigma6[close_molecules] = sigma[i]*sigma[i]*sigma[i]*sigma[i]*sigma[i]*sigma[i];
        close_sigma12[close_molecules] = close_sigma6[close_molecules]*close_sigma6[close_molecules];

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
  double dx, dy, dz, dd, d6, d12;
  double sigma, sigma6, sigma12;
  double epsilon;
  int i;
  double retval;


  for (i=0; i<close_molecules; i++)
  {
    dx = close_x[i] - test_x;
    dy = close_y[i] - test_y;
    dz = close_z[i] - test_z;
    dd = dx*dx + dy*dy + dz*dz;
    d6 = dd*dd*dd;
    d12 = d6*d6;

    sigma = 0.5 * (close_sigma[i] + test_diameter);
    epsilon = sqrt(close_epsilon[i] * test_epsilon);
    sigma6 = sigma*sigma*sigma*sigma*sigma*sigma;
    sigma12 = sigma6*sigma6;

    repulsion += epsilon * sigma12/d12;
    attraction += epsilon * sigma6/d6;
  }

  retval = 4.0 * (repulsion - attraction);
  return retval;
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
 
  V printf("%d lines read.\n", number_of_molecules);
  fclose(stdin);
}

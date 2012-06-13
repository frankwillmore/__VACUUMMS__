// pddx.c   This is shmem/pthread version of ddx 

#include <stdio.h>
#include <pthread.h>
#include <stdlib.h>
#include <assert.h>
#include <semaphore.h>

#include "ftw_param.h"
#include "ftw_prng.h"
#include "ftw_types.h"
#include "ftw_std.h"

#include "pddx.h"

struct MersenneTwister	*rng; 		// one for each thread... pointer to an array to be malloc'ed
pthread_t 		*threads;	// the threads
sem_t			semaphore;	// semaphore to restrict number of threads running
int 			*thread_args; 	// 

// Accessible to all threads
double 	x[MAX_NUM_MOLECULES];
double 	y[MAX_NUM_MOLECULES];
double 	z[MAX_NUM_MOLECULES];
double 	sigma[MAX_NUM_MOLECULES];
double 	epsilon[MAX_NUM_MOLECULES];
double 	box_x=6, box_y=6, box_z=6;
double 	verlet_cutoff=100.0;
int 	n_steps = 1000;
int 	n_samples = 1;
int 	volume_sampling = 0;
int 	include_center_energy = 0;
int 	show_steps = 0;
int 	number_of_molecules = 0;

double 	min_diameter = 0.0;
double 	characteristic_length = 1.0;
double 	characteristic_energy = 1.0;
double 	precision_parameter = 0.001; // decimal 
int 	seed = 1;

// prototypes...
void 	readConfiguration(FILE *instream);
void* 	ThreadMain(void *threadID);

int main(int argc, char **argv)
{
// handling as thread local...  
Trajectory *trajectories;  // will store data for all trajectories
//  long *passvals;
  int n_threads = 1;

  setCommandLineParameters(argc, argv);
  getIntParam("-seed", &seed);
  getIntParam("-n_threads", &n_threads);
  getVectorParam("-box", &box_x, &box_y, &box_z);
  getDoubleParam("-characteristic_length", &characteristic_length);
  getDoubleParam("-characteristic_energy", &characteristic_energy);
  getDoubleParam("-precision_parameter", &precision_parameter);
  getDoubleParam("-verlet_cutoff", &verlet_cutoff);
  getIntParam("-n_samples", &n_samples);
  getIntParam("-n_steps", &n_steps);
  volume_sampling = getFlagParam("-volume_sampling");
  include_center_energy = getFlagParam("-include_center_energy");
  show_steps = getFlagParam("-show_steps");
  getDoubleParam("-min_diameter", &min_diameter);

  if (getFlagParam("-usage"))
  {
    printf("\nusage:\t-box [ 6.0 6.0 6.0 ]\n");
    printf("\t\t-seed [ 1 ]\n");
    printf("\t\t-characteristic_length [ 1.0 ]\n");
    printf("\t\t-characteristic_energy [ 1.0 ]\n");
    printf("\t\t-precision_parameter [ 0.001 ]\n");
    printf("\t\t-n_steps [ 1000 ] (roughly reciprocal of precision parameter)\n");
    printf("\t\t-n_threads [ 1 ] n");
    printf("\t\t-show_steps (includes steps taken as final column)\n");
    printf("\t\t-verlet_cutoff [ 100.0 ]\n");
    printf("\t\t-n_samples [ 1 ]\n");
    printf("\t\t-volume_sampling \n");
    printf("\t\t-include_center_energy \n");
    printf("\t\t-min_diameter [ 0.0 ]");
    printf("\n");
    exit(0);
  }

  readConfiguration(stdin);

  // make and verify all the threads and resources
  assert(threads = (pthread_t*)malloc(sizeof(pthread_t) * n_samples));
  assert(trajectories = (Trajectory *)malloc(sizeof(Trajectory) * n_samples));

  // initialize the semaphore
  assert(0 == sem_init(&semaphore, 0, n_threads));

  /* create all threads */
//assert(passvals = (long *)malloc(sizeof(long) * n_samples));

  long i;
  for (i=0; i<n_samples; ++i) {
    trajectories[i].thread_id = i;
    assert(0 == pthread_create(&threads[i], NULL, ThreadMain, (void *) &trajectories[i]));
//    passvals[i] = i;
//    assert(0 == pthread_create(&threads[i], NULL, ThreadMain, (void *)&passvals[i])); // pass i as seed value
  }
 
  /* wait for all threads to complete */
  for (i=0; i<n_samples; ++i) assert(0 == pthread_join(threads[i], NULL));

  free(threads);
//free(passvals);
  free(trajectories);
  return 0;
} // end main()

void *ThreadMain(void *passval) { 
//printf("%ld", (long)passval);
//printf("%ld", *(long*)passval);
//return NULL;
  sem_wait(&semaphore); // thread waits to become eligible

//  Trajectory traj;
//  Trajectory *p_traj = &traj;
//  traj.thread_id = *(long*)passval; 

//  assert(trajectories = (Trajectory *)malloc(sizeof(Trajectory) * n_samples));
  Trajectory *p_traj = (Trajectory *)passval;

  MersenneInitialize(&(p_traj->rng), p_traj->thread_id);

  generateTestPoint(p_traj);

  while (calculateEnergy(p_traj, 0.0) > 0) generateTestPoint(p_traj);
   
  findEnergyMinimum(p_traj);
   
  p_traj->sq_distance_from_initial_pt 	= (p_traj->test_x-p_traj->test_x0)*(p_traj->test_x-p_traj->test_x0) 
				   	+ (p_traj->test_y-p_traj->test_y0)*(p_traj->test_y-p_traj->test_y0) 
				   	+ (p_traj->test_z-p_traj->test_z0)*(p_traj->test_z-p_traj->test_z0);
  if (!volume_sampling || (p_traj->sq_distance_from_initial_pt < .25 * p_traj->diameter * p_traj->diameter)) {
    makeVerletList(p_traj);
    expandTestParticle(p_traj);
    if (p_traj->diameter > min_diameter) {
      // correct for box edges...
      while (p_traj->test_x >= box_x) 	p_traj->test_x -= box_x;
      while (p_traj->test_x < 0) 	p_traj->test_x += box_x;
      while (p_traj->test_y >= box_y) 	p_traj->test_y -= box_y;
      while (p_traj->test_y < 0) 	p_traj->test_y += box_y;
      while (p_traj->test_z >= box_z) 	p_traj->test_z -= box_z;
      while (p_traj->test_z < 0) 	p_traj->test_z += box_z;

      printf("%lf\t%lf\t%lf\t%lf", p_traj->test_x, p_traj->test_y, p_traj->test_z, p_traj->diameter);
      if (include_center_energy) printf("\t%lf", calculateEnergy(p_traj, p_traj->diameter));
      if (show_steps) printf("\t%d", p_traj->attempts);
      printf("\n");
    }
  }

  sem_post(&semaphore);
  return NULL;
}

void generateTestPoint(Trajectory *p_traj)
{
  p_traj->test_x = p_traj->test_x0 = prnd(&(p_traj->rng)) * box_x;
  p_traj->test_y = p_traj->test_y0 = prnd(&(p_traj->rng)) * box_y;
  p_traj->test_z = p_traj->test_z0 = prnd(&(p_traj->rng)) * box_z;

  makeVerletList(p_traj);
}

void makeVerletList(Trajectory *p_traj)
{
  int i;
  double dx, dy, dz, dd;
  double shift_x, shift_y, shift_z;

  while (p_traj->test_x > box_x) p_traj->test_x -= box_x;
  while (p_traj->test_y > box_y) p_traj->test_y -= box_y;
  while (p_traj->test_z > box_z) p_traj->test_z -= box_z;

  while (p_traj->test_x < 0) p_traj->test_x += box_x;
  while (p_traj->test_y < 0) p_traj->test_y += box_y;
  while (p_traj->test_z < 0) p_traj->test_z += box_z;

  p_traj->verlet_center_x=p_traj->test_x;
  p_traj->verlet_center_y=p_traj->test_y;
  p_traj->verlet_center_z=p_traj->test_z;

  p_traj->close_molecules=0;
  for (i=0; i<number_of_molecules; i++)
  {
    for (shift_x = -box_x; shift_x <= box_x; shift_x += box_x)
    for (shift_y = -box_y; shift_y <= box_y; shift_y += box_y)
    for (shift_z = -box_z; shift_z <= box_z; shift_z += box_z)
    {
      dx = shift_x + x[i] - p_traj->test_x;
      dy = shift_y + y[i] - p_traj->test_y;
      dz = shift_z + z[i] - p_traj->test_z;

      dd = dx*dx + dy*dy + dz*dz;

      if (dd < verlet_cutoff) 
      { 
        p_traj->close_x[p_traj->close_molecules] = shift_x + x[i];
        p_traj->close_y[p_traj->close_molecules] = shift_y + y[i];
        p_traj->close_z[p_traj->close_molecules] = shift_z + z[i];
        p_traj->close_sigma[p_traj->close_molecules] = sigma[i];
        p_traj->close_sigma6[p_traj->close_molecules] = sigma[i]*sigma[i]*sigma[i]*sigma[i]*sigma[i]*sigma[i];
        p_traj->close_sigma12[p_traj->close_molecules] = p_traj->close_sigma6[p_traj->close_molecules]*p_traj->close_sigma6[p_traj->close_molecules];
        p_traj->close_epsilon[p_traj->close_molecules] = epsilon[i];

        p_traj->close_molecules++;
      }
    }
  }
}

void findEnergyMinimum(Trajectory *p_traj)
{
  double dx, dy, dz, dd, d6, d14;
  double factor;
  double old_energy;
  double new_energy;
  double grad_x, grad_y, grad_z;
  double step_x, step_y, step_z;
  int i;
  double drift_sq;
  int attempts;

  makeVerletList(p_traj);

  // begin loop to iterate until minimum found
  for (attempts=0; attempts<n_steps; attempts++)
  {
    drift_sq = (p_traj->test_x-p_traj->verlet_center_x)*(p_traj->test_x-p_traj->verlet_center_x) 
             + (p_traj->test_y-p_traj->verlet_center_y)*(p_traj->test_y-p_traj->verlet_center_y) 
             + (p_traj->test_z-p_traj->verlet_center_z)*(p_traj->test_z-p_traj->verlet_center_z);

    if (drift_sq > .01 * verlet_cutoff) makeVerletList(p_traj);

    // find the gradient at test_x, test_y, test_Z using the derivative of energy
    grad_x=0; grad_y=0; grad_z=0;

    for (i=0; i<p_traj->close_molecules; i++)
    {
      dx = p_traj->test_x - p_traj->close_x[i];
      dy = p_traj->test_y - p_traj->close_y[i];
      dz = p_traj->test_z - p_traj->close_z[i];
      dd = dx*dx + dy*dy + dz*dz;
      d6 = dd*dd*dd;
      d14 = d6*d6*dd;

      // The analytical expression for the gradient contribution contains a factor of -48.0.
      // The minus is reflected in the sense of the step taken.  The factor of 48 is factored out in the normalization.
      factor = p_traj->close_epsilon[i] * p_traj->close_sigma12[i] / d14;

      grad_x += dx * factor;
      grad_y += dy * factor;
      grad_z += dz * factor;
    }

    // normalize the gradient
    double grad_sq = grad_x * grad_x + grad_y * grad_y + grad_z * grad_z;
    double grad_modulus = sqrt(grad_sq);
    grad_x /= grad_modulus;
    grad_y /= grad_modulus;
    grad_z /= grad_modulus;

    old_energy = calculateRepulsion(p_traj);
//printf("xyz:: %12lf\t%12lf\t%12lf ::%12lf::\t  %12lf\t%12lf\t%12lf\n", test_x, test_y, test_z, old_energy, grad_x, grad_y, grad_z);

    step_x = grad_x * characteristic_energy * characteristic_length; while (step_x * step_x > characteristic_length * characteristic_length * precision_parameter * precision_parameter) {step_x *=.5;}
    step_y = grad_y * characteristic_energy * characteristic_length; while (step_y * step_y > characteristic_length * characteristic_length * precision_parameter * precision_parameter) {step_y *=.5;}
    step_z = grad_z * characteristic_energy * characteristic_length; while (step_z * step_z > characteristic_length * characteristic_length * precision_parameter * precision_parameter) {step_z *=.5;}

//    removed this criteria for assessing minima.... step size no longer shrinks because gradient is now normalized
//    step_sq = step_x * step_x + step_y * step_y + step_z * step_z;
//    if (step_sq < (characteristic_length * characteristic_length * precision_parameter * precision_parameter)) break;  // close enough, exit loop

    p_traj->test_x += step_x;
    p_traj->test_y += step_y;
    p_traj->test_z += step_z;
 
    // check repulsion at new location
    new_energy = calculateRepulsion(p_traj);
    // if the energy fluctuates up by a fraction of the characteristic energy, call it
    if (new_energy - old_energy > precision_parameter * characteristic_energy)
    {
//      printf("exiting from energy increase %lf\n", new_energy);
      break;
    }
  }
}

double calculateRepulsion(Trajectory *p_traj)
{
  double repulsion=0;
  double dx, dy, dz, dd, d6, d12;
  int i;

  for (i=0; i<p_traj->close_molecules; i++)
  {
    dx = p_traj->close_x[i] - p_traj->test_x;
    dy = p_traj->close_y[i] - p_traj->test_y;
    dz = p_traj->close_z[i] - p_traj->test_z;
    dd = dx*dx + dy*dy + dz*dz;
    d6 = dd*dd*dd;
    d12 = d6*d6;

    repulsion += p_traj->close_epsilon[i] * p_traj->close_sigma12[i] / d12;
  }
 
  return 4.0 * repulsion;
}

double calculateEnergy(Trajectory *p_traj, double test_diameter)
{
  double repulsion=0;
  double attraction=0;
  double dx, dy, dz, dd, d6, d12;
  double sigma, sigma6, sigma12;
  int i;

  for (i=0; i<p_traj->close_molecules; i++)
  {
    dx = p_traj->close_x[i] - p_traj->test_x;
    dy = p_traj->close_y[i] - p_traj->test_y;
    dz = p_traj->close_z[i] - p_traj->test_z;
    dd = dx*dx + dy*dy + dz*dz;
    d6 = dd*dd*dd;
    d12 = d6*d6;

    sigma = 0.5 * (p_traj->close_sigma[i] + test_diameter);
    sigma6 = sigma*sigma*sigma*sigma*sigma*sigma;
    sigma12 = sigma6*sigma6;

    repulsion += p_traj->close_epsilon[i] * sigma12/d12;
    attraction += p_traj->close_epsilon[i] * sigma6/d6;
  }

  return 4.0 * (repulsion - attraction);
}

void expandTestParticle(Trajectory *p_traj)
{
  double slope;
  double step_size;
  double energy, old_energy;
  double e0, e1, r0, r1;

  double diameter = 0.0;

  // improved initial guess
  old_energy = calculateEnergy(p_traj, diameter);
//printf("energy at sigma = 0:  %lf\n", old_energy);
  if (old_energy > 0) return;
  while (diameter += .1)
  {
    energy = calculateEnergy(p_traj, diameter);
    if (energy > old_energy) break;
    old_energy = energy;
  }

  while(1) // Newton's method
  {
    r0 = diameter - .001;
    r1 = diameter + .001;
    
    e0 = calculateEnergy(p_traj, r0);
    e1 = calculateEnergy(p_traj, r1);
    energy = calculateEnergy(p_traj, diameter);

    slope = (e1-e0)/(r1-r0);
    step_size = -energy/slope;

    diameter = diameter + step_size;

    if (step_size*step_size < .00000001) break;
  }

  // copy diameter to trajectory data
  p_traj->diameter = diameter;
}

void readConfiguration(FILE *instream)
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
    epsilon[number_of_molecules++] = strtod(epsilons, NULL);
    number_of_molecules++;
  }
 
  fclose(stdin);
}

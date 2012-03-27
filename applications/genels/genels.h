/* genels.h */

/* includes */

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

/* macros */

#define MAX_N_ATOMS 131072
#define GROSS_GRID_RESOLUTION 16
#define FINE_GRID_RESOLUTION 256

/* constants */

//extern const float thirtysecondth;  // one thirty-secondth
//extern const float PI;
//extern const float gas_constant; // in J/mol*K

/* types */

struct CommandLineOptions
{ 
  int verbose;
  float T;
  float verlet_cutoff_sq;
  float box_x;
  float box_y;
  float box_z;
  int n_threads;
  float r_i;
  float epsilon_i;
  int molecule;
  char *config_directory;
  char *config_name;
  int use_stdin;
  float gross_resolution;
  float fine_resolution;
};

struct Atom
{
  float x;
  float y;
  float z;
  float r_ij;
  float epsilon_ij;
};

struct VerletList
{
  float verlet_center_x;
  float verlet_center_y;
  float verlet_center_z;
  float x[768];
  float y[768];
  float z[768];
  float r_ij[768];        // central atom
  float epsilon_ij[768];  // central atom
  int close_atoms;
};

struct EnergyArray
{
  // some meta data as well?
  float energy[FINE_GRID_RESOLUTION][FINE_GRID_RESOLUTION][FINE_GRID_RESOLUTION];
};

struct ZThread
{
  struct VerletList verlet;
  struct EnergyArray *energy_array;
  pthread_t *thread_controller;
  int i, j, k;  // The actual array element numbers
};

/* prototypes */

void *ThreadMain(void *passval);
void makeVerletList(struct ZThread *thread);
void generateOutput(struct ZThread *thread);
void readPolymerConfiguration();
void readCommandLineOptions();

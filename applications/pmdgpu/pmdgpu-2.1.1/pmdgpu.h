/* pmdgpu.h */

/****************************************************************************/
/*                                                                          */
/*                                                                          */
/*                                                                          */
/*                          PMDGPU Version 2.0.0                            */
/*                                                                          */
/*                        (C) 2010 Frank T Willmore                         */
/*                                                                          */
/*                                                                          */
/*                                                                          */
/*  pmdgpu is a re-optimization of the utrwt69 algorithm published in:      */
/*                                                                          */
/*  "Gas diffusion in glasses via a probabilistic molecular dynamics."      */
/*  Willmore FT, Wang XY, Sanchez IC.                                       */
/*  JOURNAL OF CHEMICAL PHYSICS 126, 234502 JUNE 15 2007                    */
/*                                                                          */
/*  PMDGPU is developed by Frank T Willmore in collaboration with           */
/*  Isaac C Sanchez of the University of Texas at Austin department of      */
/*  chemical engineering.  Special thanks to Ying Jiang of the Sanchez      */
/*  research group for contributions to the project.                        */
/*                                                                          */
/*  PMDGPU Version 1.0.0 has been demonstrated to display statistically     */
/*  equivalent results for the diffusion of Helium in HAB6FDACl.  Results   */
/*  were compared for 10 configurations of HAB6FDACl, using n = 50          */
/*  insertions for each configuration for utrwt69 and for pmdgpu.  This     */
/*  was compared to 5 insertions into the same 10 configurations using the  */
/*  Accelrys Materials Studio simulation software package, and were         */
/*  verified to generate the same diffusivity value of 2.5e-04 sq cm/s.     */
/*                                                                          */
/*  PMDGPU Version 2.0.0 introduces MPI support as well as supports the     */
/*  simulation of diatomic molecules, including rotational motions, with    */                                                                   
/*  decoupled sampling of translational and rotational energies.            */                                                         
/*                                                                          */
/*  Allocation suport on the resources Spur and Longhorn is provided by:    */
/*                                                                          */
/*    The Texas Advanced Computing Center                                   */
/*    The National Science Foundation/NSF-Teragrid                          */
/*                                                                          */
/*  correspondence to:  frankwillmore@gmail.com                             */
/*                                                                          */
/****************************************************************************/

/* includes */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

/* macros */

#define MERSENNE_NN 312
#define MERSENNE_MM 156
#define MERSENNE_MATRIX_A 0xB5026F5AA96619E9ULL
#define MERSENNE_UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define MERSENNE_LM 0x7FFFFFFFULL /* Least significant 31 bits */

#define PMD_CO2 1
#define PMD_OXYGEN 2
#define PMD_HELIUM 3
#define PMD_NITROGEN 4
#define PMD_HYDROGEN 5

/* constants */
extern const float gross_resolution;  // in Angstroms
extern const float thirtysecondth;  // one thirty-secondth
extern const float PI;
extern const float gas_constant; // in J/mol*K

/* types */

struct CommandLineOptions
{ 
  int verbose;
  float bin_size;
  float test_mass;
  float test_mass1;
  float test_mass2;
  float T;
  float target_time; 
  float verlet_cutoff_sq;
  float threshold_time;
  int seed;
  float box_x;
  float box_y;
  float box_z;
  int n_threads;
  int n_insertions;
  float resolution_t;
  float r_i;
  float r_i1;
  float r_i2;
  float epsilon_i;
  float epsilon_i1;
  float epsilon_i2;
  float bond_length1;
  float bond_length2;
  int molecule;
  char *config_directory;
  char *config_prefix;
  char *config_suffix;
  int config_start;
  int use_stdin;
  int use_mpi;
  int individual_trajectories;
  float gross_resolution;
};

struct Atom
{
  float x;
  float y;
  float z;
  float r_ij;
  float epsilon_ij;
  float r_ij1;
  float epsilon_ij1;
  float r_ij2;
  float epsilon_ij2;
};

struct MersenneTwister
{
  unsigned long long mt[MERSENNE_NN];
  int mti;
};

struct VerletList
{
  float verlet_center_x;
  float verlet_center_y;
  float verlet_center_z;
  float x[1024];
  float y[1024];
  float z[1024];
  float r_ij[1024];        // central atom
  float epsilon_ij[1024];  // central atom
  float r_ij1[1024];       // forward atom
  float epsilon_ij1[1024]; // forward atom
  float r_ij2[1024];       // backward atom
  float epsilon_ij2[1024]; // backward atom
  int close_atoms;
};

struct ThreadResult
{
  struct MersenneTwister *rng;
  struct VerletList verlet;
  float test_x0;
  float test_y0;
  float test_z0;
  float test_x;
  float test_y;
  float test_z;
  float collision_x;
  float collision_y;
  float collision_z;
  float collision_t;
  float drift_x;
  float drift_y;
  float drift_z;
  int thread_id;
  float resolution_x;
  float resolution_y;
  float resolution_z;
  float translation_phi;
  float translation_theta;
  float orientation_phi;
  float orientation_theta;
  float orientation_dx1;
  float orientation_dy1;
  float orientation_dz1;
  float orientation_dx2;
  float orientation_dy2;
  float orientation_dz2;
  float orientation_rx;
  float orientation_ry;
  float orientation_rz;
  float axis_of_rotation_x;
  float axis_of_rotation_y;
  float axis_of_rotation_z;
  float rotation_angle;
  float rotation_angle_resolution;
  float time_resolution;
  float angular_speed;
  float speed;
  float energy_array[32];
  float testpoint_energy;
  float translational_kinetic_energy;
  float rotational_kinetic_energy;
  float R_t;
  float time_elapsed;
};

/* prototypes */

void *ThreadMain(void *passval);
void readCommandLineOptions(int argc, char* argv[]);
void MersenneInitialize(struct MersenneTwister* MT, int seed);
float rndm(struct MersenneTwister* MT);
void readConfiguration();
void generateTestPoint(struct ThreadResult* thread);
void makeVerletList(struct ThreadResult *thread);
void calculateTestpointEnergy(struct ThreadResult *thread);
void setVelocitySpeed(struct ThreadResult *thread);
void setVelocityDirection(struct ThreadResult *thread);
void setRotationalEnergy(struct ThreadResult *thread);
void setRotationalOrientation(struct ThreadResult *thread);
void setAxisOfRotation(float rx, float ry, float rz, struct MersenneTwister *rng, float *px, float *py, float *pz);
void quaternionRotate(float vx, float vy, float vz, float axisx, float axisy, float axisz, float angle, float *vxnew, float *vynew, float *vznew);


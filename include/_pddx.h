/* pddx.h */

// setting defaults for ~640MB bss for 16M MOLECULES

#ifndef MAX_NUM_MOLECULES
#define MAX_NUM_MOLECULES 16777216
#endif

// setting defaults for 60MB of heap/thread for Verlet list of 1M

#ifndef MAX_CLOSE
#define MAX_CLOSE 1048576
#endif

typedef struct {
  int				thread_id;
  int   			close_molecules;
  int   			attempts;
  double 			test_x0, test_y0, test_z0;
  double 			test_x, test_y, test_z;
  double 			verlet_center_x, verlet_center_y, verlet_center_z;
  double 			diameter;
  double 			close_x[MAX_CLOSE], close_y[MAX_CLOSE], close_z[MAX_CLOSE];
  double 			close_sigma[MAX_CLOSE];
  double 			close_sigma6[MAX_CLOSE];
  double 			close_sigma12[MAX_CLOSE];
  double 			close_epsilon[MAX_CLOSE];
  double 			sq_distance_from_initial_pt;
  struct MersenneTwister 	rng;
} Trajectory;

double calculateRepulsion(Trajectory*);
double calculateEnergy(Trajectory*, double);
void generateTestPoint(Trajectory*);
void findEnergyMinimum(Trajectory*);
void makeVerletList(Trajectory*);
void expandTestParticle(Trajectory*);
void readConfiguration(FILE *instream);
void* ThreadMain(void *threadID);



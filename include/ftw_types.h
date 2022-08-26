/* ftw_types.h */

#include <vacuumms/limits.h>


#ifndef FTW_TYPES
#define FTW_TYPES

typedef struct ftw_vector
{
  double x;
  double y;
  double z;
} ftw_vector;

struct Atom
{
  float x;
  float y;
  float z;
  float sigma;
  float epsilon;
};

typedef struct Atom ftw_Atom;

struct Cavity
{
  float x;
  float y;
  float z;
  float diameter;
};

typedef struct Cavity ftw_Cavity;

/* Structure to hold a set of cavities */
typedef struct CAV65536
{
  ftw_Cavity cavity[65536];
  float box_x;
  float box_y;
  float box_z;
  int n_cavities;
} ftw_CAV65536;

struct Configuration
{
  ftw_Atom *atom;
  double box_x;
  double box_y;
  double box_z;
  int n_atoms;
};

typedef struct Configuration ftw_Configuration;

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

typedef ftw_Configuration ftw_GFG;

/*  Adding the following definition because CUDA doesn't handle deep copies */
// struct GFG65536
// {
//   ftw_Atom atom[65536];
//   float box_x;
//   float box_y;
//   float box_z;
//   int n_atoms;
// };

/* Could it really be as easy as just making this number bigger? */
struct GFG65536
{
  ftw_Atom atom[VACUUMMS_MAX_NUMBER_OF_MOLECULES * 27];
  float box_x;
  float box_y;
  float box_z;
  int n_atoms;
};

typedef struct GFG65536 ftw_GFG65536;

struct EnergyArray256
{
  float energy[256][256][256];
};

typedef struct EnergyArray256 ftw_EnergyArray256;

struct EnergyArray512
{
  float energy[512][512][512];
};

typedef struct EnergyArray512 ftw_EnergyArray512;

struct EnergyArray1024
{
  float energy[1024][1024][1024];
};

typedef struct EnergyArray1024 ftw_EnergyArray1024;

struct FVI256
{
  float intensity[256][256][256];
};

typedef struct FVI256 ftw_FVI256;

struct EnergyArray
{
  float ***energy;
};

typedef struct EnergyArray ftw_EnergyArray;

struct FVI
{
  float ***intensity;
};

typedef struct FVI ftw_FVI;

#endif


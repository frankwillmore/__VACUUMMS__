/* convergence.c */

#include "convergence.h"
#include <ftw_std.h>
#include <ftw_rng.h>

#define sigma0 1.000

extern double max_scale;
extern double d_max_scale;
extern double diameter_scale;
extern double x[], y[], z[];
extern double box_x, box_y, box_z;
extern int number_of_molecules;

extern double c_convergence_ratio;
double r_convergence_ratio;
extern int number_of_samples;

extern int n_trials;
extern int d_trials;
extern double d_step_size;

int mirror_depth = 1;
extern int volume_sampling;

double center_x, center_y, center_z;
double sample_x, sample_y, sample_z;
double diameter;

void sampleCavities()
{
  int i=0;
  double energy;

  r_convergence_ratio = 1/c_convergence_ratio;

  while (i<number_of_samples)
  {
    double dx, dy, dz;
    double shift_x, shift_y, shift_z;
    int valid = 0;

    center_x = sample_x = box_x * rnd();
    center_y = sample_y = box_y * rnd();
    center_z = sample_z = box_z * rnd();
    findCenter();
    findDiameter();

    valid = 1;
    if (volume_sampling)
    {
      valid = 0;
      for (shift_x = -box_x; shift_x < box_x; shift_x += box_x)
      for (shift_y = -box_y; shift_y < box_y; shift_y += box_y)
      for (shift_z = -box_z; shift_z < box_z; shift_z += box_z)
      {
        dx = shift_x + sample_x - center_x;
        dy = shift_y + sample_y - center_y;
        dz = shift_z + sample_z - center_z;
        if ((dx*dx + dy*dy + dz*dz) < (.25*diameter*diameter)) valid = 1;
      }
    }

    if (valid)
    {
       printf("%lf\t%lf\t%lf\t%lf\n", center_x, center_y, center_z, diameter);
       i++;
    }
  }
}

void findCenter()
{
  double new_x, new_y, new_z;
  int trials = 0;
  double scale = max_scale;

  while (trials < n_trials) 
  {
    double dx, dy, dz;

    dx = scale * (rnd() - .5);
    dy = scale * (rnd() - .5);
    dz = scale * (rnd() - .5);

    new_x = sample_x + dx;
    new_y = sample_y + dy;
    new_z = sample_z + dz;
    if (new_x >= box_x) new_x -= box_x;
    if (new_x < 0) new_x += box_x;
    if (new_y >= box_y) new_y -= box_y;
    if (new_y < 0) new_y += box_y;
    if (new_z >= box_z) new_z -= box_z;
    if (new_z < 0) new_z += box_z;

    if (getRepulsiveInteraction(new_x, new_y, new_z, 1) <
        getRepulsiveInteraction(sample_x, sample_y, sample_z, 1)) 
    {
      sample_x = new_x;
      sample_y = new_y;
      sample_z = new_z;
      scale *= r_convergence_ratio;
      if (scale > max_scale) scale = max_scale;
    }
    else 
    {
      scale *= c_convergence_ratio;
    }
    trials++;

    if (scale < .01 * max_scale) break;
  }

}

void findDiameter()
{
  int trials = 0;
  double d_min=0.0, d_mid, d_max;
  double e_mid;

  // initial guess for d_min
  for(d_min = 0;;d_min += d_step_size)
    if (getTotalInteraction(sample_x, sample_y, sample_z, d_min) < 
        getTotalInteraction(sample_x, sample_y, sample_z, d_min + d_step_size))
      break;

  // initial guess for d_max
  for(d_max = d_min;;d_max += d_step_size)
    if (getTotalInteraction(sample_x, sample_y, sample_z, d_max) > 0) 
      break;

  for (trials=0; trials < d_trials; trials++)
  {
    diameter = (d_min + d_max)/2;
    e_mid = getTotalInteraction(sample_x, sample_y, sample_z, diameter);
    
    if (e_mid > 0) d_max = diameter;
    else d_min = diameter;
  }
}

double getTotalInteraction(double sample_x, double sample_y, double sample_z, double sample_d)
{
  return getRepulsiveInteraction(sample_x, sample_y, sample_z, sample_d) 
  + getAttractiveInteraction(sample_x, sample_y, sample_z, sample_d);
}

double getRepulsiveInteraction(double sample_x, double sample_y, double sample_z, double sample_d)
{
  double interaction = 0;
  double dx, dy, dz, d_sq;
  double shift_x, shift_y, shift_z;
  double sigmaij, sigma12;
  int i;

  sigmaij = (sigma0 + sample_d)/2.0;
  sigma12 = sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij;

  for (shift_x = -box_x; shift_x <= box_x; shift_x += box_x)
    for (shift_y = -box_y; shift_y <= box_y; shift_y += box_y)
      for (shift_z = -box_z; shift_z <= box_z; shift_z += box_z)
        for (i=0; i< number_of_molecules; i++)
        {
          dx = shift_x + x[i] - sample_x;
          dy = shift_y + y[i] - sample_y;
          dz = shift_z + z[i] - sample_z;
          d_sq = dx*dx + dy*dy + dz*dz;
          interaction += sigma12/(d_sq*d_sq*d_sq*d_sq*d_sq*d_sq);
        }

  return interaction;
}

double getAttractiveInteraction(double sample_x, double sample_y, double sample_z, double sample_d)
{
  double interaction = 0;
  double dx, dy, dz, d_sq;
  double shift_x, shift_y, shift_z;
  double sigmaij, sigma6;
  int i;

  sigmaij = (sigma0 + sample_d)/2.0;
  sigma6 = sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij;

  for (shift_x = -box_x; shift_x <= box_x; shift_x += box_x)
    for (shift_y = -box_y; shift_y <= box_y; shift_y += box_y)
      for (shift_z = -box_z; shift_z <= box_z; shift_z += box_z)
        for (i=0; i< number_of_molecules; i++)
        {
          dx = shift_x + x[i] - sample_x;
          dy = shift_y + y[i] - sample_y;
          dz = shift_z + z[i] - sample_z;
          d_sq = dx*dx + dy*dy + dz*dz;
          interaction -= sigma6/(d_sq*d_sq*d_sq);
        }

  return interaction;
}


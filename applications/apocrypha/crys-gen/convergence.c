/* convergence.c */

#include "convergence.h"
#include <ftw_std.h>
#include <ftw_rng.h>

#define sigma0 1.000

extern double max_scale;
extern double d_max_scale;
extern double diameter_scale;
extern double x[], y[], z[];
extern double sigma[], sqrt_epsilon[];
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
double attractive_interaction = 0;
double repulsive_interaction = 0;
double net_interaction = 0;

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

  calculateInteractions(1);

  while (trials < n_trials) 
  {
    double dx, dy, dz;
    double old_x, old_y, old_z;
    double old_repulsion;

    dx = scale * (rnd() - .5);
    dy = scale * (rnd() - .5);
    dz = scale * (rnd() - .5);

    old_x = center_x;
    old_y = center_y;
    old_z = center_z;

    center_x += dx;
    center_y += dy;
    center_z += dz;

    if (center_x >= box_x) center_x -= box_x;
    if (center_x < 0) center_x += box_x;
    if (center_y >= box_y) center_y -= box_y;
    if (center_y < 0) center_y += box_y;
    if (center_z >= box_z) center_z -= box_z;
    if (center_z < 0) center_z += box_z;

    // update and calculate new interactions 
    old_repulsion = repulsive_interaction;
    calculateInteractions(1);

    if (old_repulsion < repulsive_interaction) // rejected
    {
      center_x = old_x;
      center_y = old_y;
      center_z = old_z;
      scale *= c_convergence_ratio;
      repulsive_interaction = old_repulsion;
    }
    else // accepted
    {
      scale *= r_convergence_ratio;
      if (scale > max_scale) scale = max_scale;
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
  double old_interaction;

  calculateInteractions(d_min);

  // initial guess for d_min
  for(d_min = d_step_size;;d_min += d_step_size)
  {
    old_interaction = net_interaction;
    calculateInteractions(d_min);
    if (old_interaction < net_interaction) break;
  }

  // initial guess for d_max
  for(d_max = d_min;;d_max += d_step_size)
  {
    calculateInteractions(d_max);
    if (net_interaction > 0) break;
  }

  for (trials=0; trials < d_trials; trials++)
  {
    diameter = (d_min + d_max)/2;
    calculateInteractions(diameter);
    if (net_interaction > 0) d_max = diameter;
    else d_min = diameter;
  }
}

void calculateInteractions(double sample_d)
{
  double dx, dy, dz, d_sq;
  double shift_x, shift_y, shift_z;
  double sigmaij, sigma6, sigma12;
  double d6, d12;
  int i;

  attractive_interaction = 0;
  repulsive_interaction = 0;

//  sigmaij = (sigma0 + sample_d)/2.0;
//  sigma6 = sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij;
//  sigma12 = sigma6*sigma6;

  for (shift_x = -box_x; shift_x <= box_x; shift_x += box_x)
  for (shift_y = -box_y; shift_y <= box_y; shift_y += box_y)
  for (shift_z = -box_z; shift_z <= box_z; shift_z += box_z)
  for (i=0; i< number_of_molecules; i++)
  {
    sigmaij = .5 * (sigma[i] + sample_d);
    sigma6 = sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij;
    sigma12 = sigma6*sigma6;

    dx = shift_x + x[i] - center_x;
    dy = shift_y + y[i] - center_y;
    dz = shift_z + z[i] - center_z;
    d_sq = dx*dx + dy*dy + dz*dz;
    d6 = d_sq * d_sq * d_sq;
    d12 = d6 * d6;
    attractive_interaction -= sqrt_epsilon[i] * sigma6/(d6);
    repulsive_interaction += sqrt_epsilon[i] * sigma12/(d12);
    net_interaction = (attractive_interaction + repulsive_interaction);
  }
}

void calculateRepulsivePotentialGradient()
{
  double dx, dy, dz, d_sq;
  double shift_x, shift_y, shift_z;
  double sigmaij, sigma6, sigma12;
  double d6, d12, d14;
  int i;

  for (i=0; i< number_of_molecules; i++)
  {
    sigmaij = .5 * (sigma[i] + sample_d);
    sigma6 = sigmaij*sigmaij*sigmaij*sigmaij*sigmaij*sigmaij;
    sigma12 = sigma6*sigma6;

    for (shift_x = -box_x; shift_x <= box_x; shift_x += box_x)
    for (shift_y = -box_y; shift_y <= box_y; shift_y += box_y)
    for (shift_z = -box_z; shift_z <= box_z; shift_z += box_z)
    {
      dx = shift_x + x[i] - center_x;
      dy = shift_y + y[i] - center_y;
      dz = shift_z + z[i] - center_z;
      d_sq = dx*dx + dy*dy + dz*dz;
      d6 = d_sq * d_sq * d_sq;
      d12 = d6 * d6;
      repulsive_interaction += sqrt_epsilon[i] * sigma12/(d12);
      net_interaction = (attractive_interaction + repulsive_interaction);

      grad_x = 
    }
  }
}

buildVerletList()
{
}


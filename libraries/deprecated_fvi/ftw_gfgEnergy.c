/* ftw_gfgEnergy.c */

#include <stdlib.h>
#include <math.h>

#include <ftw_gfgEnergy.h>
#include <ftw_config_parser.h>
#include <ftw_types.h>

float ftw_GFG65536Energy_69(ftw_GFG65536 *gfg, float x, float y, float z)
{
  float repulsion=0;
  float attraction=0;
  float alpha;
  float dx, dy, dz, d, dd;

printf("not implemented.\n");
exit(1);
  // evaluate energy at (x, y, z);
  int i;

  for (i=0; i< gfg->n_atoms; i++) {
    // central atom
    dx = gfg->atom[i].x - x;
    dy = gfg->atom[i].y - y;
    dz = gfg->atom[i].z - z;
    dd = dx*dx + dy*dy + dz*dz; d = sqrt(dd);
    alpha = pow(gfg->atom[i].sigma, 3) / (d * dd);   
    repulsion += gfg->atom[i].epsilon * alpha * alpha * alpha;
    attraction += gfg->atom[i].epsilon * alpha * alpha;
  } 

  return repulsion - attraction;
}

float ftw_GFG65536Energy_612(ftw_GFG65536 *gfg, float x_j, float y_j, float z_j, float diameter)
{
  float repulsion=0;
  float attraction=0;
  float sigma_j = diameter;
  float sigma_i6;
  float sigma_j6;
  float sigma_ij2;
  float sigma_over_r_sq;
  float dx, dy, dz, dd;
  int i;

  // evaluate energy at (x, y, z);
  for (i=0; i< gfg->n_atoms; i++) {
    // central atom
    dx = gfg->atom[i].x - x_j;
    dy = gfg->atom[i].y - y_j;
    dz = gfg->atom[i].z - z_j;
    dd = dx*dx + dy*dy + dz*dz;

    sigma_i6 = gfg->atom[i].sigma * gfg->atom[i].sigma * gfg->atom[i].sigma * gfg->atom[i].sigma * gfg->atom[i].sigma * gfg->atom[i].sigma;
    sigma_j6 = sigma_j * sigma_j * sigma_j * sigma_j * sigma_j * sigma_j;
    sigma_ij2 = pow((0.5f * (sigma_i6 + sigma_j6)), 0.333333f);
    sigma_over_r_sq = sigma_ij2 / dd;

    repulsion  += gfg->atom[i].epsilon * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq;
    attraction += gfg->atom[i].epsilon * sigma_over_r_sq * sigma_over_r_sq * sigma_over_r_sq;
  } 

  return 4.0f * (repulsion - attraction);
}

// stripped down version to determine hard-sphere overlap.
// returns Boolean value.
int ftw_GFG65536HS_Overlap(ftw_GFG65536 *gfg, float x_j, float y_j, float z_j, float diameter)
{
  float critical_diameter;
  float dx, dy, dz, dd;
  int i;

  for (i=0; i< gfg->n_atoms; i++) {
    critical_diameter = 0.5f * (diameter + gfg->atom[i].sigma);
    dx = gfg->atom[i].x - x_j;
    dy = gfg->atom[i].y - y_j;
    dz = gfg->atom[i].z - z_j;
    dd = dx*dx + dy*dy + dz*dz;
    if (dd < (critical_diameter * critical_diameter)) 
    {
//printf("found %f < %f\n", dd, critical_diameter * critical_diameter); 
      return 1;
    }
  } 

  return 0; // no overlap found
}

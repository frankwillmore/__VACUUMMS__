/* energy.c */

#include "energy.h"

extern double x[], y[], z[];
extern double box_x, box_y, box_z;
extern int number_of_molecules;

// program uses values of sigma=1,epsilon=1
// psi_shift is the shift-and-truncate correction.
double psi_shift = 1.0198e-3/4;

/* returns energy per mole of LJ centers */
double calculateSystemEnergy()
{
  int i, j;
  double total_energy = 0;

  for (i=0; i<number_of_molecules + 1; i++)
    for (j=i+1; j<number_of_molecules + 1; j++)
    {
      total_energy = total_energy + interactionEnergy(i,j);
    }
  return (total_energy);
}

/* interaction energy (per mole) of pair i,j */
double interactionEnergy(int i,int j)
{
  double dx, dy, dz;
  double r_sq, r6, r12;
  double cutoff_sq = 2.5 * 2.5;
  double energy = 0;

  /* and now the boxes... */

  /* box -1, -1, -1 */
  dx = x[j] - box_x - x[i];
  dy = y[j] - box_y - y[i];
  dz = z[j] - box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box -1, -1, 0 */
  dx = x[j] - box_x - x[i];
  dy = y[j] - box_y - y[i];
  dz = z[j] - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box -1, -1, 1 */
  dx = x[j] - box_x - x[i];
  dy = y[j] - box_y - y[i];
  dz = z[j] + box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box -1, 0, -1 */
  dx = x[j] - box_x - x[i];
  dy = y[j] - y[i];
  dz = z[j] - box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)

  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box -1, 0, 0 */
  dx = x[j] - box_x - x[i];
  dy = y[j] - y[i];
  dz = z[j] - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box -1, 0, 1 */
  dx = x[j] - box_x - x[i];
  dy = y[j] - y[i];
  dz = z[j] + box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box -1, 1, -1 */
  dx = x[j] - box_x - x[i];
  dy = y[j] + box_y - y[i];
  dz = z[j] - box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box -1, 1, 0 */
  dx = x[j] - box_x - x[i];
  dy = y[j] + box_y - y[i];
  dz = z[j] - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box -1, 1, 1 */
  dx = x[j] - box_x - x[i];
  dy = y[j] + box_y - y[i];
  dz = z[j] + box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 0, -1, -1 */
  dx = x[j] - x[i];
  dy = y[j] - box_y - y[i];
  dz = z[j] - box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 0, -1, 0 */
  dx = x[j] - x[i];
  dy = y[j] - box_y - y[i];
  dz = z[j] - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 0, -1, 1 */
  dx = x[j] - x[i];
  dy = y[j] - box_y - y[i];
  dz = z[j] + box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 0, 0, -1 */
  dx = x[j] - x[i];
  dy = y[j] - y[i];
  dz = z[j] - box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 0, 0, 0 */
  dx = x[j] - x[i];
  dy = y[j] - y[i];
  dz = z[j] - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 0, 0, 1 */
  dx = x[j] - x[i];
  dy = y[j] - y[i];
  dz = z[j] + box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 0, 1, -1 */
  dx = x[j] - x[i];
  dy = y[j] + box_y - y[i];
  dz = z[j] - box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 0, 1, 0 */
  dx = x[j] - x[i];
  dy = y[j] + box_y - y[i];
  dz = z[j] - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 0, 1, 1 */
  dx = x[j] - x[i];
  dy = y[j] + box_y - y[i];
  dz = z[j] + box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 1, -1, -1 */
  dx = x[j] + box_x - x[i];
  dy = y[j] - box_y - y[i];
  dz = z[j] - box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 1, -1, 0 */
  dx = x[j] + box_x - x[i];
  dy = y[j] - box_y - y[i];
  dz = z[j] - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 1, -1, 1 */
  dx = x[j] + box_x - x[i];
  dy = y[j] - box_y - y[i];
  dz = z[j] + box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 1, 0, -1 */
  dx = x[j] + box_x - x[i];
  dy = y[j] - y[i];
  dz = z[j] - box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 1, 0, 0 */
  dx = x[j] + box_x - x[i];
  dy = y[j] - y[i];
  dz = z[j] - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 1, 0, 1 */
  dx = x[j] + box_x - x[i];
  dy = y[j] - y[i];
  dz = z[j] + box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 1, 1, -1 */
  dx = x[j] + box_x - x[i];
  dy = y[j] + box_y - y[i];
  dz = z[j] - box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 1, 1, 0 */
  dx = x[j] + box_x - x[i];
  dy = y[j] + box_y - y[i];
  dz = z[j] - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }

  /* box 1, 1, 1 */
  dx = x[j] + box_x - x[i];
  dy = y[j] + box_y - y[i];
  dz = z[j] + box_z - z[i];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy += (1/r12 - 1/r6 + psi_shift);
  }
  
  energy *=4;

  return energy;
}

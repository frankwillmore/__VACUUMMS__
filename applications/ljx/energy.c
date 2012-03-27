/* energy.c */

#include "energy.h"

#define NPAIRS 1048576

extern double x[], y[], z[];
extern double box_x, box_y, box_z;
extern int number_of_molecules;
extern int monte_carlo_steps;
extern int relaxation_allowance;
int num_pairs;

int pair_list_first[NPAIRS];
int pair_list_second[NPAIRS];
double pair_list_xoffset[NPAIRS];
double pair_list_yoffset[NPAIRS];
double pair_list_zoffset[NPAIRS];

int time_to_update=0;

// program uses values of sigma=1,epsilon=1
// psi_shift is the shift-and-truncate correction.
extern double psi_shift;
extern double cutoff_sq;

/* returns energy per mole of LJ centers */
double calculateSystemEnergy()
{
  double energy = 0;
  int pair_no;

  // if (monte_carlo_steps < relaxation_allowance) return calculateSystemEnergyRigorously();

  // now evaluate energies  
  for (pair_no=0; pair_no<num_pairs; pair_no++) energy += getPairEnergy(pair_no);

  return 4*energy;
}

double getPairEnergy(int pair_no)
{
  double dx, dy, dz;
  double energy=0;
  double r_sq, r6, r12;
  
  dx = x[pair_list_second[pair_no]] + pair_list_xoffset[pair_no] - x[pair_list_first[pair_no]];
  dy = y[pair_list_second[pair_no]] + pair_list_yoffset[pair_no] - y[pair_list_first[pair_no]];
  dz = z[pair_list_second[pair_no]] + pair_list_zoffset[pair_no] - z[pair_list_first[pair_no]];
  r_sq = dx*dx + dy*dy + dz*dz;
  if (r_sq < cutoff_sq)
  {
    r6 = r_sq * r_sq * r_sq;
    r12 = r6 * r6;
    energy = (1/r12 - 1/r6 + psi_shift);
  }
  
  return energy;
}

void updatePairList()
{
  int i, j;
  int shift_x, shift_y, shift_z;
  int close_x=0, close_y=0, close_z=0;
  double dsq, min_dsq;
  double dx, dy, dz;

  // correctForBoundaries
  for (i=0; i<number_of_molecules; i++)
  {
    if (x[i] >= box_x) x[i] -= box_x;
    if (y[i] >= box_y) y[i] -= box_y;
    if (z[i] >= box_z) z[i] -= box_z;

    if (x[i] < 0) x[i] += box_x;
    if (y[i] < 0) y[i] += box_y;
    if (z[i] < 0) z[i] += box_z;
  }

  num_pairs = 0;
  for (i=0; i<number_of_molecules - 1; i++)
  for (j=i + 1; j<number_of_molecules; j++)
  {
//    min_dsq = 10;  // set an initial value
    min_dsq = cutoff_sq*1.3;  // set an initial value
    for (shift_x = -1; shift_x<=1; shift_x++)
    for (shift_y = -1; shift_y<=1; shift_y++)
    for (shift_z = -1; shift_z<=1; shift_z++)
    {
      dx = shift_x * box_x + x[j] - x[i];
      dy = shift_y * box_y + y[j] - y[i];
      dz = shift_z * box_z + z[j] - z[i];
      dsq = dx*dx + dy*dy + dz*dz;
      if (dsq < min_dsq)
      {
        close_x = shift_x;
        close_y = shift_y;
        close_z = shift_z;
        min_dsq = dsq;
      }
    }

//    if (min_dsq < 9) // add it to pair list
    if (min_dsq < cutoff_sq*1.25) // add it to pair list
    {
      pair_list_first[num_pairs] = i;
      pair_list_second[num_pairs] = j;
      pair_list_xoffset[num_pairs] = close_x * box_x;
      pair_list_yoffset[num_pairs] = close_y * box_y;
      pair_list_zoffset[num_pairs] = close_z * box_z;
      num_pairs++;
    }
  } // next j, i

  time_to_update = number_of_molecules * 10;
}

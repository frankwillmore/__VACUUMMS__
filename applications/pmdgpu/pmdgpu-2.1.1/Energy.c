/* Energy.cu */

/*************************************************************************/
/*                                                                       */
/*  Host method for calculation energy at the current position of the    */
/*  Test particle, as specified by (test_x, test_y, test_z).             */
/*  Verlet list is updated by ThreadMain().                              */
/*                                                                       */
/*************************************************************************/

#include <pmdgpu.h>

void calculateTestpointEnergy(struct ThreadResult *thread)
{
  float repulsion=0;
  float attraction=0;
  float alpha;
  float dx, dy, dz, d, dd;
  int i;

  for (i=0; i < thread->verlet.close_atoms; i++)
  {
    // central atom
    dx = thread->verlet.x[i] - thread->test_x;
    dy = thread->verlet.y[i] - thread->test_y;
    dz = thread->verlet.z[i] - thread->test_z;
    dd = dx*dx + dy*dy + dz*dz;
    d = sqrt(dd);
    alpha = pow(thread->verlet.r_ij[i], 3) / (d * dd);   
    repulsion += thread->verlet.epsilon_ij[i] * alpha * alpha * alpha;
    attraction += thread->verlet.epsilon_ij[i] * alpha * alpha;

    // forward atom 
    dx = thread->verlet.x[i] - (thread->test_x + thread->orientation_dx1);
    dy = thread->verlet.y[i] - (thread->test_y + thread->orientation_dy1);
    dz = thread->verlet.z[i] - (thread->test_z + thread->orientation_dz1);
    dd = dx*dx + dy*dy + dz*dz;
    d = sqrt(dd);
    alpha = pow(thread->verlet.r_ij1[i], 3) / (d * dd);   
    repulsion += thread->verlet.epsilon_ij1[i] * alpha * alpha * alpha;
    attraction += thread->verlet.epsilon_ij1[i] * alpha * alpha;

    // backward atom
    dx = thread->verlet.x[i] - (thread->test_x + thread->orientation_dx2);
    dy = thread->verlet.y[i] - (thread->test_y + thread->orientation_dy2);
    dz = thread->verlet.z[i] - (thread->test_z + thread->orientation_dz2);
    dd = dx*dx + dy*dy + dz*dz;
    d = sqrt(dd);
    alpha = pow(thread->verlet.r_ij2[i], 3) / (d * dd);   
    repulsion += thread->verlet.epsilon_ij2[i] * alpha * alpha * alpha;
    attraction += thread->verlet.epsilon_ij2[i] * alpha * alpha;
  } 
  
  thread->testpoint_energy = (2 * repulsion - 3 * attraction);
}


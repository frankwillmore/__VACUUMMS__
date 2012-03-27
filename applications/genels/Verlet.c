/* Verlet.c */

/***************************************************************************/
/*                                                                         */
/*                                                                         */
/***************************************************************************/

#include <genels.h>

extern struct Atom configuration[];
extern int number_of_atoms;
extern float gross_resolution;
extern float verlet_cutoff_sq;

void makeVerletList(struct ZThread *thread)
{
  int i;
  float dx, dy, dz, dd;

  thread->verlet.verlet_center_x = (0.5f + thread->i) * gross_resolution;
  thread->verlet.verlet_center_y = (0.5f + thread->j) * gross_resolution;
  thread->verlet.verlet_center_z = (0.5f + thread->k) * gross_resolution;
//printf("verlet center:\t%f\t%f\t%f\n", thread->verlet.verlet_center_x, thread->verlet.verlet_center_y, thread->verlet.verlet_center_z);

  thread->verlet.close_atoms=0;

  for (i=0; i<number_of_atoms; i++)
  {
    dx = configuration[i].x - thread->verlet.verlet_center_x;
    dy = configuration[i].y - thread->verlet.verlet_center_y;
    dz = configuration[i].z - thread->verlet.verlet_center_z;

    dd = dx*dx + dy*dy + dz*dz;

    if (dd < verlet_cutoff_sq)
    {
      thread->verlet.x[thread->verlet.close_atoms] = configuration[i].x;
      thread->verlet.y[thread->verlet.close_atoms] = configuration[i].y;
      thread->verlet.z[thread->verlet.close_atoms] = configuration[i].z;
      thread->verlet.r_ij[thread->verlet.close_atoms] = configuration[i].r_ij;
      thread->verlet.epsilon_ij[thread->verlet.close_atoms] = configuration[i].epsilon_ij;

      thread->verlet.close_atoms++;
//printf("%d\t%d-%d-%d\t%d\n", i, thread->i, thread->j, thread->k, thread->verlet.close_atoms);
      assert(thread->verlet.close_atoms < 768);
    }
  }
//printf("Verlet atoms: %d\n", thread->verlet.close_atoms);
}

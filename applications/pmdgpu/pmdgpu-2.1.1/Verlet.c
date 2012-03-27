/* Verlet.cu */

/***************************************************************************/
/*                                                                         */
/*  This routine updates the Verlet list and recenters the position of     */
/*  test particle with respect to its insertion point.  The value of       */
/*  shift_x + test_x will ALWAYS return the X-distance which the test      */
/*  particle has traveled from its insertion point.  This routine will     */
/*  always bring the test particle into the boundary of the simulation     */
/*  box and will center the Verlet list at the position of the test        */
/*  particle at the point where this routine is called.                    */
/*                                                                         */
/***************************************************************************/

#include <pmdgpu.h>

extern struct CommandLineOptions clo;
extern struct Atom configuration[];
extern long number_of_atoms;
extern pthread_mutex_t verlet_mutex;

void makeVerletList(struct ThreadResult *thread)
{
  int i;
  float dx, dy, dz, dd;
  int shift_x, shift_y, shift_z;

  while (thread->test_x >= clo.box_x)
  {
    thread->test_x -= clo.box_x;
    thread->drift_x += clo.box_x;
  }
  while (thread->test_y >= clo.box_y)
  {
    thread->test_y -= clo.box_y;
    thread->drift_y += clo.box_y;
  }
  while (thread->test_z >= clo.box_z)
  {
    thread->test_z -= clo.box_z;
    thread->drift_z += clo.box_z;
  }

  while (thread->test_x < 0)
  {
    thread->test_x += clo.box_x;
    thread->drift_x -= clo.box_x;
  }
  while (thread->test_y < 0)
  {
    thread->test_y += clo.box_y;
    thread->drift_y -= clo.box_y;
  }
  while (thread->test_z < 0)
  {
    thread->test_z += clo.box_z;
    thread->drift_z -= clo.box_z;
  }

  thread->verlet.verlet_center_x=thread->test_x;
  thread->verlet.verlet_center_y=thread->test_y;
  thread->verlet.verlet_center_z=thread->test_z;

  thread->verlet.close_atoms=0;

  for (i=0; i<number_of_atoms; i++)
  {
    for (shift_x = -1; shift_x <= 1; shift_x++)
    for (shift_y = -1; shift_y <= 1; shift_y++)
    for (shift_z = -1; shift_z <= 1; shift_z++)
    {
      float verlet_x = clo.box_x * shift_x + configuration[i].x;
      float verlet_y = clo.box_y * shift_y + configuration[i].y;
      float verlet_z = clo.box_z * shift_z + configuration[i].z; 

      dx = verlet_x - thread->verlet.verlet_center_x;
      dy = verlet_y - thread->verlet.verlet_center_y;
      dz = verlet_z - thread->verlet.verlet_center_z;

      dd = dx*dx + dy*dy + dz*dz;

      if (dd < clo.verlet_cutoff_sq)
      {
        thread->verlet.x[thread->verlet.close_atoms] = verlet_x;
        thread->verlet.y[thread->verlet.close_atoms] = verlet_y;
        thread->verlet.z[thread->verlet.close_atoms] = verlet_z;

        thread->verlet.r_ij[thread->verlet.close_atoms] = configuration[i].r_ij;
        thread->verlet.r_ij1[thread->verlet.close_atoms] = configuration[i].r_ij1;
        thread->verlet.r_ij2[thread->verlet.close_atoms] = configuration[i].r_ij2;
        thread->verlet.epsilon_ij[thread->verlet.close_atoms] = configuration[i].epsilon_ij;
        thread->verlet.epsilon_ij1[thread->verlet.close_atoms] = configuration[i].epsilon_ij1;
        thread->verlet.epsilon_ij2[thread->verlet.close_atoms] = configuration[i].epsilon_ij2;

        thread->verlet.close_atoms++;
        assert(thread->verlet.close_atoms < 1024);
      }
    }
  }
}


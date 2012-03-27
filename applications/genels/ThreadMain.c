/* ThreadMain.c */

/******************************************************************************/
/*                                                                            */
/*                                                                            */
/******************************************************************************/

#include <genels.h>

extern struct CommandLineOptions clo;
extern sem_t thread_sem;
extern float gross_resolution;
extern float fine_resolution;

void *ThreadMain(void *passval)
{
  struct ZThread *thread;
  thread = (struct ZThread*) passval;

  // distribute threads among available devices
  int cuda_device_count;
  cudaGetDeviceCount(&cuda_device_count);
  cudaSetDevice(thread->k % cuda_device_count);
  setResolution(&gross_resolution, &fine_resolution);

  // Wait for the signal, then start the thread.
  sem_wait(&thread_sem);
  
  thread->energy_array = (struct EnergyArray*)malloc(sizeof(struct EnergyArray));
  makeVerletList(thread);
  calculateEnergy(thread);

  generateOutput(thread);
  free(thread->energy_array);

  // Free memory, alert the next thread that it can run, then exit
  sem_post(&thread_sem);
  pthread_exit(NULL);
}


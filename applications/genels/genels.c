/* genels.c */

#include <genels.h>

struct ZThread thread_array[GROSS_GRID_RESOLUTION][GROSS_GRID_RESOLUTION][GROSS_GRID_RESOLUTION]; 
struct Atom configuration[MAX_N_ATOMS];
int number_of_atoms=0;
float gross_resolution;
float fine_resolution;
float box_dimension;
char els_directory[255];
int concurrency=1;
float r_i;
float epsilon_i;
float verlet_cutoff_sq=81.0;

sem_t thread_sem; // controls how many threads are running

int main(int argc, char *argv[])
{
  pthread_attr_t attr;
  int i, grid_x, grid_y, grid_z;
  int thread_x;
  int sem_res;

  readCommandLineOptions(argc, argv);
  gross_resolution = box_dimension / GROSS_GRID_RESOLUTION;
  fine_resolution = gross_resolution / FINE_GRID_RESOLUTION;
  readPolymerConfiguration();
   
  // thread semaphore keeps a constant # of threads running until all have completed
  // initialize thread semaphore
  sem_res = sem_init(&thread_sem, 0, 0);
  if (sem_res != 0)
  {
    perror("Semaphore initialization failed.  Exiting.\n");
    exit(EXIT_FAILURE);
  }

  // Load up the semaphore with the max # of concurrent threads 
  for (i=0; i<concurrency; i++) sem_post(&thread_sem);  

  /* Initialize and set thread detached attribute */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // start the processing...
  for (grid_x=0; grid_x < GROSS_GRID_RESOLUTION; grid_x++)
  for (grid_y=0; grid_y < GROSS_GRID_RESOLUTION; grid_y++)
  for (grid_z=0; grid_z < GROSS_GRID_RESOLUTION; grid_z++)
  {
    thread_array[grid_x][grid_y][grid_z].i=grid_x;
    thread_array[grid_x][grid_y][grid_z].j=grid_y;
    thread_array[grid_x][grid_y][grid_z].k=grid_z;
    thread_array[grid_x][grid_y][grid_z].thread_controller = (pthread_t*)malloc(sizeof(pthread_t));
    assert(!pthread_create(thread_array[grid_x][grid_y][grid_z].thread_controller, &attr, ThreadMain, (void *)(&thread_array[grid_x][grid_y][grid_z]))); 
  }

  /* Free attribute and wait for the other threads */
  pthread_attr_destroy(&attr);
  for (grid_x=0; grid_x < GROSS_GRID_RESOLUTION; grid_x++)
  for (grid_y=0; grid_y < GROSS_GRID_RESOLUTION; grid_y++)
  for (grid_z=0; grid_z < GROSS_GRID_RESOLUTION; grid_z++)
  {
    assert(!pthread_join(*(thread_array[grid_x][grid_y][grid_z].thread_controller), NULL));
    // process results...
    printf("joined thread (%d, %d, %d)\n", grid_x, grid_y, grid_z);
    fflush(stdout);
  }

  sem_destroy(&thread_sem);
}

/* pmdgpu.c */

/***********************************************************************/
/*                                                                     */
/*  This is the main function which is initialized by the OS.  Its     */
/*  purposes are solely to launch each of the n threads that execute   */
/*  the function ThreadMain(), to ensure that no more than N of these  */
/*  threads run concurrently, to wait for each and all of the n        */
/*  to complete and join, and lastly to display the results to         */
/*  standard output.                                                   */
/*                                                                     */
/***********************************************************************/

#include <pmdgpu.h>
#include <mpi.h>

#define MAX_N_ATOMS 16384

struct CommandLineOptions clo;
struct Atom configuration[MAX_N_ATOMS];
int number_of_atoms=0;

sem_t thread_sem; // controls how many threads are running
pthread_mutex_t time_series_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t verlet_mutex = PTHREAD_MUTEX_INITIALIZER;
float time_series_R_delta_t[64][128];
float time_series_Rsq_delta_t[64][128];
float time_series_delta_t[64][128];

// MPI variables
int 	mpi_rank;
int	mpi_processes;

/* constants */

const float gross_resolution = 1.0;  // in Angstroms
const float thirtysecondth = 0.03125;  // one thirty-secondth
const float PI = 3.14159;
const float gas_constant = 8.314f; // J/mol*K

int main(int argc, char *argv[])
{
  pthread_t *thread_controller;
  pthread_attr_t attr;
  int pthread_return_code;
  int thread_index=0; 
  struct ThreadResult *thread;
  int sem_res;
  int i, j;

  int 		mpi_sender;
  int 		mpi_receiver;
  int 		mpi_tag;
  char 		mpi_message[100];
  MPI_Status 	mpi_status;
  float 	mpi_time_series[64][128];
  
  readCommandLineOptions(argc, argv);
   
  if (clo.use_mpi)
  {
    printf("initializing MPI...\n");
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_processes);
  }

  // thread semaphore keeps a constant # of threads running until all have completed
  // initialize thread semaphore
  sem_res = sem_init(&thread_sem, 0, 0);
  if (sem_res != 0) 
  {
    perror("Semaphore initialization failed.  Exiting.\n");
    exit(EXIT_FAILURE);
  }
  // Load up the semaphore with the max # of concurrent threads 
  for (i=0; i<clo.n_threads; i++) sem_post(&thread_sem);  

  // initialize bins for time series data
  // make twice as many as will be counted... 
  pthread_mutex_lock( &time_series_mutex );
  for (i=0; i<64; i++)
    for (j=0; j<128; j++)
    {
      time_series_R_delta_t[i][j] = 0.0;
      time_series_Rsq_delta_t[i][j] = 0.0;
      time_series_delta_t[i][j] = 0.0;
    }
  pthread_mutex_unlock( &time_series_mutex );

  thread = (struct ThreadResult*)malloc(sizeof(struct ThreadResult) * clo.n_insertions);
  thread_controller = (pthread_t*)malloc(sizeof(pthread_t) * clo.n_insertions);
  
  readConfiguration();

  /* Initialize and set thread detached attribute */
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  // launch one thread for each insertion 
  for(thread_index=0; thread_index<clo.n_insertions; thread_index++) 
  {
    thread[thread_index].thread_id = thread_index;
    pthread_return_code = pthread_create(&(thread_controller[thread_index]), &attr, ThreadMain, (void *)&(thread[thread_index])); 
    assert(!pthread_return_code);
  }

  /* Free attribute and wait for the other threads */
  pthread_attr_destroy(&attr);
  for(thread_index=0; thread_index<clo.n_insertions; thread_index++) 
  {
    pthread_return_code = pthread_join(thread_controller[thread_index], NULL);
    assert(!pthread_return_code);
  }

  sem_destroy(&thread_sem);
  free(thread);
  free(thread_controller);

  // Now get the data from the other processes
  int n_bins = (int)floor(clo.target_time/clo.resolution_t);

  if (!clo.use_mpi)
  {
    pthread_mutex_lock( &time_series_mutex );
    for (i=0; i<=n_bins; i++)  
    {
      printf("%f", i*clo.resolution_t);
      if (clo.individual_trajectories)
      {
        for (j=0; j<clo.n_insertions; j++) printf("\t%f", time_series_Rsq_delta_t[j][i] / time_series_delta_t[j][i]);
        printf("\n");
      }
      else 
      {
        for (j=1; j<clo.n_insertions; j++) 
        {
          time_series_Rsq_delta_t[0][i] += time_series_Rsq_delta_t[j][i];
          time_series_delta_t[0][i] += time_series_delta_t[j][i];
        }
        printf("\t%f\n", time_series_Rsq_delta_t[0][i] / time_series_delta_t[0][i]);
      }
    }
    pthread_mutex_unlock( &time_series_mutex );
    
    exit(0);
  }

  if (mpi_rank == 0)
  {
    // first give our result
    printf("result for configuration %d:\n\n", clo.config_start);
    pthread_mutex_lock( &time_series_mutex );
    for (i=0; i<=n_bins; i++)  
    {
      printf("%f", i*clo.resolution_t);
      if (clo.individual_trajectories)
      {
        for (j=0; j<clo.n_insertions; j++) printf("\t%f", time_series_Rsq_delta_t[j][i] / time_series_delta_t[j][i]);
        printf("\n");
      }
      else 
      {
        for (j=1; j<clo.n_insertions; j++) 
        {
          time_series_Rsq_delta_t[0][i] += time_series_Rsq_delta_t[j][i];
          time_series_delta_t[0][i] += time_series_delta_t[j][i];
        }
        printf("\t%f\n", time_series_Rsq_delta_t[0][i] / time_series_delta_t[0][i]);
      }
    }
    pthread_mutex_unlock( &time_series_mutex );
    
    // Now receive and display the others
    for (mpi_sender = 1; mpi_sender < mpi_processes; mpi_sender++)
    {
      MPI_Recv(mpi_time_series, 8192, MPI_FLOAT, mpi_sender, mpi_tag, MPI_COMM_WORLD, &mpi_status);
      printf("\nresult for configuration %d:\n", clo.config_start + mpi_sender);
      for (i=0; i<=n_bins; i++)  
      {
        printf("%f", i*clo.resolution_t);
        if (clo.individual_trajectories)
        {
          for (j=0; j<clo.n_insertions; j++) printf("\t%f", mpi_time_series[j][i]);
          printf("\n");
        }
        else printf("\t%f\n", mpi_time_series[0][i]);
      }
    }
  }
  else // mpi_rank > 0 , pack up the data to send:
  {
    pthread_mutex_lock( &time_series_mutex );
    if (clo.individual_trajectories)
    {
      for (i=0; i<=n_bins; i++) 
        for (j=0; j<clo.n_insertions; j++)
          mpi_time_series[j][i] = time_series_Rsq_delta_t[j][i] / time_series_delta_t[j][i];
    }
    else 
    {
      for (i=0; i<=n_bins; i++)
      {
        for (j=0; j<clo.n_insertions; j++)
          mpi_time_series[0][i] += time_series_Rsq_delta_t[j][i] / time_series_delta_t[j][i];
        mpi_time_series[0][i] /= clo.n_insertions;
      }
    }
    pthread_mutex_unlock( &time_series_mutex );
    MPI_Send(mpi_time_series, 8192, MPI_FLOAT, 0, mpi_tag, MPI_COMM_WORLD);
  }

  MPI_Finalize();
}

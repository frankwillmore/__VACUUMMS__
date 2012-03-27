/* mlp_mpi.c */

/****************************************************************************/
/*                                                                          */
/*                                                                          */
/*  Multi-level parallel code example.                                      */
/*  MPI(multiprocess) / pthread(multithread) / CUDA (GPU)                   */
/*                                                                          */
/*                                                                          */
/*  (C) 2010 Texas Advanced Computing Center.  All rights reserved.         */
/*                                                                          */
/*  For information, contact Frank Willmore:  willmore@tacc.utexas.edu      */
/*                                                                          */
/****************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <mpi.h>
#include <pthread.h>

#define NUM_THREADS 5
#define N_BINS 51

void *ThreadMain(void *histogram);

int main(int argc, char *argv[])
{
  
  // MPI variables
  int           mpi_rank;
  int           mpi_processes;
  int           mpi_sender;
  int           mpi_receiver;
  int           mpi_tag = 0;
  char          mpi_message[100];
  MPI_Status    mpi_status;

  int bin;
  int histogram[N_BINS];
  
  MPI_Init(&argc, &argv);

  // Now fork some threads...
  pthread_t threads[NUM_THREADS];

  int rc;
  long t;
  for(t=0; t<NUM_THREADS; t++){
    printf("In main: creating thread %ld\n", t);
    rc = pthread_create(&threads[t], NULL, ThreadMain, (void *)t);
    if (rc){ printf("ERROR; return code from pthread_create() is %d\n", rc); exit(-1); }
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_processes);

  if (mpi_rank == 0) {
    for (mpi_sender = 1; mpi_sender < mpi_processes; mpi_sender++){
      MPI_Recv(mpi_message, sizeof(mpi_message), MPI_INT, mpi_sender, mpi_tag, MPI_COMM_WORLD, &mpi_status);
      for (bin=0; bin<N_BINS; bin++) histogram[bin] += mpi_message[bin];
    }
    for (bin=0; bin<N_BINS; bin++){
      printf("%1.2f:", bin);
      while (histogram[bin]--) printf("X");
      printf("\n");
    }
  }
  else MPI_Send(mpi_message, sizeof(mpi_message), MPI_INT, /* to rank 0 */ 0, mpi_tag, MPI_COMM_WORLD);

  MPI_Finalize();
}

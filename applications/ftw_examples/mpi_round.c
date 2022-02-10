#include <mpi.h>
#include <stdio.h>

main(int argc, char *argv[])
{
  // MPI variables
  int           mpi_rank;
  int           mpi_processes;
  int           mpi_sender;
  int           mpi_receiver;
  int           mpi_tag = 0;
  char          mpi_message[100];
  MPI_Status    mpi_status;

  int           relay_count = 0;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_processes);

  sprintf(mpi_message, "I am process %d/%d.", mpi_rank, mpi_processes);

  if (mpi_rank == 0) MPI_Send(mpi_message, sizeof(mpi_message), MPI_CHAR, /* to rank 1 */ 1, mpi_tag, MPI_COMM_WORLD);

  while (relay_count++ < 1000000){
    mpi_sender = (((unsigned int)(mpi_rank - 1)) & (mpi_processes - 1));
// printf("rank %d receiving from rank %d\n", mpi_rank, mpi_sender);
    MPI_Recv(mpi_message, sizeof(mpi_message), MPI_CHAR, mpi_sender, mpi_tag, MPI_COMM_WORLD, &mpi_status);
    mpi_receiver = (unsigned int)(mpi_rank + 1) & (mpi_processes - 1);
// printf("rank %d sending to rank %d\n", mpi_rank, mpi_receiver);
    MPI_Send(mpi_message, sizeof(mpi_message), MPI_CHAR, mpi_receiver, mpi_tag, MPI_COMM_WORLD);
  }

  MPI_Finalize();
}

#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int rank, size;
  long n_itr = 1000;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // loop integer
  long looper;
  MPI_Status status;

  if (rank == 0) {
    looper = 0;
    MPI_Send(&looper, 1, MPI_LONG, 1, 999, MPI_COMM_WORLD);
  }
  for (long i = 0; i < n_itr; i++) {
    MPI_Recv(&looper, 1, MPI_LONG, (rank - 1 + size) % size, 999,
             MPI_COMM_WORLD, &status);
    looper += rank;
    MPI_Send(&looper, 1, MPI_LONG, (rank + 1) % size, 999, MPI_COMM_WORLD);
  }

  MPI_Finalize();

  printf("looper = %ld\n", looper);

  return 0;
}

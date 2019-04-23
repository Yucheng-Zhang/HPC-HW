#include "utils.h"
#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  int rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /*** loop integer ***/
  long n_itr = 10000;
  long looper;
  MPI_Status status;
  Timer t;

  if (rank == 0) {
    looper = 0;
    MPI_Send(&looper, 1, MPI_LONG, 1, 999, MPI_COMM_WORLD);
    t.tic();
  }
  for (long i = 0; i < n_itr; i++) {
    MPI_Recv(&looper, 1, MPI_LONG, (rank - 1 + size) % size, 999,
             MPI_COMM_WORLD, &status);
    looper += rank;
    MPI_Send(&looper, 1, MPI_LONG, (rank + 1) % size, 999, MPI_COMM_WORLD);
  }
  if (rank == 0) {
    double tt = t.toc();
    printf(">> # of processes: %d\n", size);
    printf(">> Value after %d loops: %ld\n", n_itr, looper);
    long v_ref = (size - 1) * size / 2 * n_itr;
    printf(">> It should be: %ld\n", v_ref);
    printf(">> Error = %ld\n", v_ref - looper);
    printf(">> Time: %f s\n", tt);
    printf(">> Latency: %f s\n", tt / (size * n_itr));
  }

  MPI_Finalize();

  return 0;
}

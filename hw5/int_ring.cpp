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
    printf("*** loop an integer (MPI_LONG) ***\n");
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
    printf(">> Value after %ld loops: %ld\n", n_itr, looper);
    long v_ref = (size - 1) * size / 2 * n_itr;
    printf(">> It should be: %ld\n", v_ref);
    printf(">> Error = %ld\n", v_ref - looper);
    printf(">> Time elapsed: %f s\n", tt);
    printf(">> Latency: %f s\n", tt / (size * n_itr));
  }

  /*** loop array ***/
  long n_itr1 = 100;
  long arr_size = (1UL << 18);
  double *looparr = (double *)malloc(arr_size * sizeof(double)); // about 2 MB

  if (rank == 0) {
    printf("*** loop a 2MB array of MPI_DOUBLE ***\n");
    MPI_Send(looparr, arr_size, MPI_DOUBLE, 1, 999, MPI_COMM_WORLD);
    t.tic();
  }
  for (long i = 0; i < n_itr1; i++) {
    MPI_Recv(looparr, arr_size, MPI_DOUBLE, (rank - 1 + size) % size, 999,
             MPI_COMM_WORLD, &status);
    MPI_Send(looparr, arr_size, MPI_DOUBLE, (rank + 1) % size, 999,
             MPI_COMM_WORLD);
  }
  if (rank == 0) {
    double tt = t.toc();
    printf(">> # of processes: %d\n", size);
    printf(">> # of loops: %ld\n", n_itr1);
    printf(">> Time elapsed: %f s\n", tt);
    printf(">> Bandwidth %f MB/s\n", 2 * n_itr1 * size / tt);
  }

  MPI_Finalize();

  return 0;
}

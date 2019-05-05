// Parallel sample sort
#include <algorithm>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N;
  sscanf(argv[1], "d", &N);

  int *vec = (int *)malloc(N * sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int)(rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  // sort locally
  std::sort(vec, vec + N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int *lsp = (int *)malloc((p - 1) * sizeof(int));
  int step = N / p + 1;
  for (int j = 0, i = step - 1; j < p - 1; j++) {
    lsp[j] = vec[i];
    i += step;
  }

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int *gsps = NULL;
  if (rank == 0) {
    gsps = (int *)malloc(p * (p - 1) * sizeof(int));
  }
  MPI_Gather(lsp, p - 1, MPI_INT, gsps, p - 1, MPI_INT, 0, MPI_COMM_WORLD);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int *gsp = (int *)malloc((p - 1) * sizeof(int));
  if (rank == 0) {
    std::sort(gsps, gsps + p * (p - 1));
    for (int j = 0, i = p - 2; j < p - 1; j++) {
      gsp[j] = gsps[i];
      i += p;
    }
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(gsp, p - 1, MPI_INT, 0, MPI_COMM_WORLD);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  int *sdispls = (int *)malloc(p * sizeof(int));
  sdispls[0] = 0;
  for (int i = 0; i < p - 1; i++) {
    sdispls[i + 1] = std::lower_bound(vec, vec + N, gsp[i]) - vec;
  }
  int *scounts = (int *)malloc(p * sizeof(int));
  for (int i = 0; i < p - 1; i++) {
    scounts[i] = sdispls[i + 1] - sdispls[i];
  }
  scounts[p - 1] = N - sdispls[p - 1];

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  int *rcounts = (int *)malloc(p * sizeof(int));
  MPI_Alltoall(scounts, 1, MPI_INT, rcounts, 1, MPI_INT, MPI_COMM_WORLD);

  int rctot = 0;
  int *rdispls = (int *)malloc(p * sizeof(int));
  rdispls[0] = 0;
  for (int i = 1; i < p; i++) {
    rctot += rcounts[i - 1];
    rdispls[i] = rctot;
  }
  rctot += rcounts[p - 1];
  int *vec2 = (int *)malloc(rctot * sizeof(int));

  MPI_Alltoallv(vec, scounts, sdispls, MPI_INT, vec2, rcounts, rdispls, MPI_INT,
                MPI_COMM_WORLD);

  // do a local sort of the received data
  std::sort(vec2, vec2 + rctot);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == rank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }

  // every process writes its result to a file
  char fn[1024];
  snprintf(fn, 1024, "ssort_rank_%02d.txt", rank);
  FILE *fd = fopen(fn, "w");
  for (int i = 0; i < rctot; i++) {
    fprintf(fd, "%d\n", vec2[i]);
  }
  fclose(fd);

  free(vec);
  free(vec2);
  MPI_Finalize();
  return 0;
}

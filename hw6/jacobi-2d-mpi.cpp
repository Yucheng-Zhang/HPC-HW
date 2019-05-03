/* MPI-parallel 2D Jacobi smoother.
 * Yucheng Zhang
 */
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq) {
  int M = lN + 2;
  double Du, gres = 0.0, lres = 0.0;

  for (int i = 1; i <= lN; i++) {
    for (int j = 1; j <= lN; j++) {
      Du = invhsq *
           (-lu[(i - 1) * M + j] - lu[i * M + j - 1] + 4.0 * lu[i * M + j] -
            lu[(i + 1) * M + j] - lu[i * M + j + 1]);
      Du -= 1;
      lres += Du * Du;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}

/* check position of rank */
void check_pos(int rank, int lp, bool *pos) {
  pos[0] = (rank / lp == lp - 1) ? true : false; // up
  pos[1] = (rank / lp == 0) ? true : false;      // down
  pos[2] = (rank % lp == 0) ? true : false;      // left
  pos[3] = (rank % lp == lp - 1) ? true : false; // right
}

int main(int argc, char *argv[]) {
  int mpirank, p, lp, N;
  int lN, max_iters;
  sscanf(argv[1], "%d", &lN);
  sscanf(argv[2], "%d", &max_iters);

  MPI_Status status0, status1, status2, status3;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  lp = int(sqrt(p)); // number of tasks in each direction
  N = lp * lN;
  if (mpirank == 0) {
    printf("N: %d, p: %d, local N: %d\n", N, p, lN);
  }
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of 2D lattice, including ghost points */
  double *lu = (double *)calloc(sizeof(double), (lN + 2) * (lN + 2));
  double *lunew = (double *)calloc(sizeof(double), (lN + 2) * (lN + 2));
  double *lutemp;

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1. / hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  /* check position of rank */
  bool pos[4]; // up, down, left, right
  check_pos(mpirank, lp, pos);

  /* for left and right communication */
  double *clt, *clt1, *crt, *crt1;
  if (!(pos[2])) {
    clt = (double *)calloc(sizeof(double), lN);
    clt1 = (double *)calloc(sizeof(double), lN);
  }
  if (!(pos[3])) {
    crt = (double *)calloc(sizeof(double), lN);
    crt1 = (double *)calloc(sizeof(double), lN);
  }

  for (int iter = 0; iter < max_iters && gres / gres0 > tol; iter++) {

    /* Jacobi step for local points */
    for (int i = 1; i <= lN; i++) {
      for (int j = 1; j <= lN; j++) {
        lunew[i * (lN + 2) + j] =
            0.25 *
            (hsq + lu[(i - 1) * (lN + 2) + j] + lu[i * (lN + 2) + j - 1] +
             lu[(i + 1) * (lN + 2) + j] + lu[i * (lN + 2) + j + 1]);
      }
    }

    /* prepare left and right */
    for (int i = 0; i < lN; i++) {
      clt[i] = lunew[(i + 1) * (lN + 2) + 1];
      crt[i] = lunew[(i + 1) * (lN + 2) + lN];
    }

    /* communicate ghost values */
    if (!(pos[0])) { // send/recv to/from up
      MPI_Send(&(lunew[lN * (lN + 2) + 1]), lN, MPI_DOUBLE, mpirank + lp, 991,
               MPI_COMM_WORLD);
      MPI_Recv(&(lunew[(lN + 1) * (lN + 2) + 1]), lN, MPI_DOUBLE, mpirank + lp,
               992, MPI_COMM_WORLD, &status0);
    }
    if (!(pos[1])) { // send/recv to/from down
      MPI_Send(&(lunew[1 * (lN + 2) + 1]), lN, MPI_DOUBLE, mpirank - lp, 992,
               MPI_COMM_WORLD);
      MPI_Recv(&(lunew[0 * (lN + 2) + 1]), lN, MPI_DOUBLE, mpirank - lp, 991,
               MPI_COMM_WORLD, &status1);
    }
    if (!(pos[2])) { // send/recv to/from left
      MPI_Send(clt, lN, MPI_DOUBLE, mpirank - 1, 993, MPI_COMM_WORLD);
      MPI_Recv(clt1, lN, MPI_DOUBLE, mpirank - 1, 994, MPI_COMM_WORLD,
               &status2);
      // copy to ghost
      for (int i = 1; i <= lN; i++) {
        lunew[i * (lN + 2) + 0] = clt1[i];
      }
    }
    if (!(pos[3])) { // send/recv to/from right
      MPI_Send(crt, lN, MPI_DOUBLE, mpirank + 1, 994, MPI_COMM_WORLD);
      MPI_Recv(crt1, lN, MPI_DOUBLE, mpirank + 1, 993, MPI_COMM_WORLD,
               &status3);
      // copy to ghost
      for (int i = 1; i <= lN; i++) {
        lunew[i * (lN + 2) + lN + 1] = crt1[i];
      }
    }

    /* copy newu to u using pointer flipping */
    lutemp = lu;
    lu = lunew;
    lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
        printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}

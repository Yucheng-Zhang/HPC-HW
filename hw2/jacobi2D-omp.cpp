#include "utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Copy array a to b. */
void cp_arr(double *a, double *b, long L) {
  for (long i = 0; i < L; i++) {
    b[i] = a[i];
  }
}

/* Initialize f. */
void init_f(double *f, long N) {
  for (long i = 0; i < N * N; i++) {
    f[i] = 1.0;
  }
}

/* Initialize u. */
void init_u(double *u, long M) {
  for (long i = 0; i < M * M; i++) {
    u[i] = 0;
  }
}

/* 2D Jacobi method. */
void jacobi2d(double *u, double *f, long N, long n_itr) {
  long M = N + 2;
  double h2 = 1.0 / ((N + 1) * (N + 1));
  double *u_tmp = (double *)malloc(M * M * sizeof(double));
  cp_arr(u, u_tmp, M * M);

  for (long k = 0; k < n_itr; k++) {
    double *tmp = u;
    u = u_tmp;
    u_tmp = tmp;
#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(u, u_tmp, N, M, h2, f)
#endif
    for (long i = 1; i <= N; i++) {
      for (long j = 1; j <= N; j++) {
        u[i * M + j] = 0.25 * (h2 * f[(i - 1) * N + (j - 1)] +
                               u_tmp[(i - 1) * M + j] + u_tmp[i * M + j - 1] +
                               u_tmp[(i + 1) * M + j] + u_tmp[i * M + j + 1]);
      }
    }
  }

  free(u_tmp);
}

/* Calculate the norm of the residual. */
double calc_res(double *u, double *f, long N) {
  long M = N + 2;
  double res = 0;
  double ih2 = (N + 1) * (N + 1);
  double Du;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) shared(M,N,ih2) private(Du) reduction(+:res)
#endif
  for (long i = 1; i <= N; i++) {
    for (long j = 1; j <= N; j++) {
      Du = ih2 * (-u[(i - 1) * M + j] - u[i * M + j - 1] + 4 * u[i * M + j] -
                  u[(i + 1) * M + j] - u[i * M + j + 1]);
      Du -= f[(i - 1) * N + (j - 1)];
      res += Du * Du;
    }
  }
  return sqrt(res);
}

/* Main function. */
int main(int argc, char const *argv[]) {

#ifdef _OPENMP
  omp_set_num_threads(4);
#endif

  long n_itr = 10000;
  long N = 1000;
  long M = N + 2; // 0 and N+1 for the boundary
  double *u = (double *)malloc(M * M * sizeof(double));
  double *f = (double *)malloc(N * N * sizeof(double));

  init_u(u, M);
  init_f(f, N);

  printf(">> N = %ld\n", N);
  double i_res = calc_res(u, f, N);
  printf(">> Initial norm of residual: %.2e\n", i_res);

  Timer t;

  printf(">> Jacobi iteration started\n");
  printf(">> Number of iterations: %ld\n", n_itr);
  t.tic();
  jacobi2d(u, f, N, n_itr);
  double tt = t.toc();

  printf(":: Time elapsed: %lf s\n", tt);

  double f_res = calc_res(u, f, N);
  printf(":: Final norm of residual: %.2e\n", f_res);
  printf(":: Residual is decreased by a factor of: %.2e\n", i_res / f_res);

  free(u);
  free(f);
  return 0;
}

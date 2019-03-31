#include <algorithm>
#include <math.h>
#include <omp.h>
#include <stdio.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long *prefix_sum, const long *A, long n) {
  if (n == 0)
    return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i - 1] + A[i - 1];
  }
}

// Multi-threaded OpenMP scan
void scan_omp(long *prefix_sum, const long *A, long n) {
  if (n == 0)
    return;

  int p, tid;

#pragma omp parallel shared(prefix_sum, A, p) private(tid)
  {
    tid = omp_get_thread_num();
    if (tid == 0) {
      p = omp_get_num_threads();
      printf(">> number of threads: %d\n", p);
    }
#pragma omp barrier
    long ktid = tid * (n / p);
    long ktid1 = (tid + 1) * (n / p);
    prefix_sum[ktid] = 0;
    for (long i = ktid + 1; i < ktid1; i++) {
      prefix_sum[i] = prefix_sum[i - 1] + A[i - 1];
    }
  }

  for (int i = 1; i < p; i++) {
    long ki = i * (n / p);
    long ki1 = (i + 1) * (n / p);
    long corr = prefix_sum[ki - 1] + A[ki - 1];
    for (long j = ki; j < ki1; j++) {
      prefix_sum[j] += corr;
    }
  }
}

int main() {
  long N = 100000000;
  long *A = (long *)malloc(N * sizeof(long));
  long *B0 = (long *)malloc(N * sizeof(long));
  long *B1 = (long *)malloc(N * sizeof(long));
  for (long i = 0; i < N; i++)
    A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++)
    err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}

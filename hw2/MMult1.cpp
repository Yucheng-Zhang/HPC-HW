// g++ -fopenmp -O3 -march=native MMult1.cpp && ./a.out

#include "utils.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>

#define BLOCK_SIZE 64

// Note: matrices are stored in column major order; i.e. the array elements in
// the (m x n) matrix C are stored in the sequence: {C_00, C_10, ..., C_m0,
// C_01, C_11, ..., C_m1, C_02, ..., C_0n, C_1n, ..., C_mn}
void MMult0(long m, long n, long k, double *a, double *b, double *c) {
  for (long j = 0; j < n; j++) {
    for (long p = 0; p < k; p++) {
      for (long i = 0; i < m; i++) {
        double A_ip = a[i + p * m];
        double B_pj = b[p + j * k];
        double C_ij = c[i + j * m];
        C_ij = C_ij + A_ip * B_pj;
        c[i + j * m] = C_ij;
      }
    }
  }
}

void MMult1_v1(long m, long n, long k, double *a, double *b, double *c) {
  const long lb = BLOCK_SIZE;

  for (long jj = 0; jj < n; jj += lb) {
    for (long pp = 0; pp < k; pp += lb) {
      for (long ii = 0; ii < m; ii += lb) {

        for (long j = jj; j < jj + lb; j++) {
          for (long p = pp; p < pp + lb; p++) {
            for (long i = ii; i < ii + lb; i++) {
              double A_ip = a[i + p * m];
              double B_pj = b[p + j * k];
              double C_ij = c[i + j * m];
              C_ij = C_ij + A_ip * B_pj;
              c[i + j * m] = C_ij;
            }
          }
        }
      }
    }
  }
}

void MMult1(long m, long n, long k, double *a, double *b, double *c) {
  const long lb = BLOCK_SIZE;
  // get the number of blocks
  long nb_m = m / lb;
  long nb_n = n / lb;
  long nb_k = k / lb;

#pragma omp parallel for collapse(2) shared(a, b, c, nb_m, nb_n, nb_k, m, n, k)
  for (long ii = 0; ii < nb_m; ii++) {
    for (long jj = 0; jj < nb_n; jj++) { // loop over blocks in C (m x n)
      double C_b[lb * lb];
      long bi0 = ii * lb;
      long bj0 = jj * lb;
      // read in block in C
      for (long i = 0; i < lb; i++) {
        for (long j = 0; j < lb; j++) {
          C_b[i + j * lb] = c[bi0 + i + (bj0 + j) * m];
        }
      }
      // loop over A (m x k), B (k x n) blocks
      for (long kk = 0; kk < nb_k; kk++) {
        double A_b[lb * lb];
        double B_b[lb * lb];
        // read in block in A & B
        for (long i = 0; i < lb; i++) {
          for (long j = 0; j < lb; j++) {
            A_b[i + j * lb] = a[bi0 + i + (kk * lb + j) * m];
            B_b[i + j * lb] = b[kk * lb + i + (bj0 + j) * k];
          }
        }
        // get product of A block and B block
        for (long j = 0; j < lb; j++) {
          for (long p = 0; p < lb; p++) {
            for (long i = 0; i < lb; i++) {
              C_b[i + j * lb] += A_b[i + p * lb] * B_b[p + j * lb];
            }
          }
        }
      }
      // write C block back
      for (long i = 0; i < lb; i++) {
        for (long j = 0; j < lb; j++) {
          c[bi0 + i + (bj0 + j) * m] = C_b[i + j * lb];
        }
      }
    }
  }
}

/* Calculate the flop rate in the matrix operations. */
double calc_flops(long m, long n, long k, long nr, double time) {
  long n_ops = 2 * k * m * n * nr;
  return (double)n_ops / 1e9 / time;
}

/* Calculate the bandwidth in the matrix operations. */
double calc_bw(long m, long n, long k, long nr, double time) {
  long n_rw = 4 * m * n * k * nr;
  return (double)n_rw * sizeof(double) / 1e9 / time;
}

int main(int argc, char **argv) {
  const long PFIRST = BLOCK_SIZE;
  const long PLAST = 2000;
  const long PINC =
      std::max(50 / BLOCK_SIZE, 1) * BLOCK_SIZE; // multiple of BLOCK_SIZE

  printf(" Dimension       Time    Gflop/s       GB/s        Error\n");

  long count = 0;
  double sum_flops = 0.0;
  double sum_bw = 0.0;
  for (long p = PFIRST; p < PLAST; p += PINC) {
    long m = p, n = p, k = p;
    long NREPEATS = 1e9 / (m * n * k) + 1;
    double *a = (double *)aligned_malloc(m * k * sizeof(double));     // m x k
    double *b = (double *)aligned_malloc(k * n * sizeof(double));     // k x n
    double *c = (double *)aligned_malloc(m * n * sizeof(double));     // m x n
    double *c_ref = (double *)aligned_malloc(m * n * sizeof(double)); // m x n

    // Initialize matrices
    for (long i = 0; i < m * k; i++)
      a[i] = drand48();
    for (long i = 0; i < k * n; i++)
      b[i] = drand48();
    for (long i = 0; i < m * n; i++)
      c_ref[i] = 0;
    for (long i = 0; i < m * n; i++)
      c[i] = 0;

    // Compute reference solution
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult0(m, n, k, a, b, c_ref);
    }

    Timer t;
    t.tic();
    for (long rep = 0; rep < NREPEATS; rep++) {
      MMult1(m, n, k, a, b, c);
    }
    double time = t.toc();

    double flops = calc_flops(m, n, k, NREPEATS, time);
    double bandwidth = calc_bw(m, n, k, NREPEATS, time);
    count++;
    sum_flops += flops;
    sum_bw += bandwidth;

    printf("%10ld %10f %10f %10f", p, time, flops, bandwidth);

    double max_err = 0;
    for (long i = 0; i < m * n; i++)
      max_err = std::max(max_err, fabs(c[i] - c_ref[i]));
    printf(" %10e\n", max_err);

    aligned_free(a);
    aligned_free(b);
    aligned_free(c);
  }
  printf(">> Average Gflops/s: %10f\n", sum_flops / count);
  printf(">> Average GB/s: %10f\n", sum_bw / count);

  return 0;
}

// * Using MMult0 as a reference, implement MMult1 and try to rearrange loops to
// maximize performance. Measure performance for different loop arrangements and
// try to reason why you get the best performance for a particular order?
//
//
// * You will notice that the performance degrades for larger matrix sizes that
// do not fit in the cache. To improve the performance for larger matrices,
// implement a one level blocking scheme by using BLOCK_SIZE macro as the block
// size. By partitioning big matrices into smaller blocks that fit in the cache
// and multiplying these blocks together at a time, we can reduce the number of
// accesses to main memory. This resolves the main memory bandwidth bottleneck
// for large matrices and improves performance.
//
// NOTE: You can assume that the matrix dimensions are multiples of BLOCK_SIZE.
//
//
// * Experiment with different values for BLOCK_SIZE (use multiples of 4) and
// measure performance.  What is the optimal value for BLOCK_SIZE?
//
//
// * Now parallelize your matrix-matrix multiplication code using OpenMP.
//
//
// * What percentage of the peak FLOP-rate do you achieve with your code?
//
//
// NOTE: Compile your code using the flag -march=native. This tells the compiler
// to generate the best output using the instruction set supported by your CPU
// architecture. Also, try using either of -O2 or -O3 optimization level flags.
// Be aware that -O2 can sometimes generate better output than using -O3 for
// programmer optimized code.

#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <omp.h>
#include <string>
#include <math.h>
#include "utils.h"

#define BLOCK_DIM 32
__constant__ double h2_d;

/* Initialize f. */
void init_f(double *f, long N) {
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N * N; i++) f[i] = 1.0;
}

/* Initialize u. */
void init_u(double *u, long N) {
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N * N; i++) u[i] = 0;
}

/* Copy array a to b. */
void cp_arr(double *a, double *b, long L) {
  for (long i = 0; i < L; i++) {
    b[i] = a[i];
  }
}

/* CPU reference */
void jacobi2d(double *u, double *f, long N, long n_itr) {
  long M = N + 2;
  double h2 = 1.0 / ((N + 1) * (N + 1));
  double *u_tmp = (double *)malloc(M * M * sizeof(double));
  cp_arr(u, u_tmp, M * M);

  for (long k = 0; k < n_itr; k++) {
    // swap the pointers
    double *tmp = u;
    u = u_tmp;
    u_tmp = tmp;
    #pragma omp parallel for collapse(2) shared(u, u_tmp, N, M, h2, f)
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

void get_u_ref(double* u_ref, double* u_ref1, long N) {
  long M = N + 2;
  #pragma omp parallel for collapse(2)
  for (long i = 0; i < N; i++) {
    for (long j = 0; j < N; j++) {
      u_ref[i * N + j] = u_ref1[(i+1) * M + (j+1)];
    }
  }
}

/* Jacobi GPU kernel */
__global__ void jacobi_kernel(double* u, double* u_tmp, const double* f_d, long N) {
  __shared__ double smem[BLOCK_DIM+2][BLOCK_DIM+2];
  long offset_x = blockIdx.x * (BLOCK_DIM-2);
  long offset_y = blockIdx.y * (BLOCK_DIM-2);

  smem[threadIdx.x][threadIdx.y] = 0;
  if (offset_x + threadIdx.x < N && offset_y + threadIdx.y < N)
    smem[threadIdx.x][threadIdx.y] = u_tmp[(offset_x + threadIdx.x)*N + (offset_y + threadIdx.y)];
  __syncthreads();

  double sum;
  sum = 0.25 * (h2_d * f_d[(offset_x + threadIdx.x)*N + (offset_y + threadIdx.y)] +
                smem[threadIdx.x-1][threadIdx.y] + smem[threadIdx.x+1][threadIdx.y] +
                smem[threadIdx.x][threadIdx.y-1] + smem[threadIdx.x][threadIdx.y+1]);

  if (threadIdx.x+2 < BLOCK_DIM && threadIdx.y+2 < BLOCK_DIM)
    if (offset_x+threadIdx.x+2 <= N && offset_y+threadIdx.y+2 <= N)
      u[(offset_x+threadIdx.x)*N + (offset_y+threadIdx.y)] = sum;
}

int main(int argc, char const *argv[]) {

  Timer t;
  long n_itr = 100;
  long N = 1000;

  double *u;
  cudaMallocHost((void**)&u, N * N * sizeof(double));
  double *f;
  cudaMallocHost((void**)&f, N * N * sizeof(double));

  init_u(u, N);
  init_f(f, N);

  /* CPU reference */
  double *u_ref, *u_ref1;
  long M = N + 2;
  cudaMallocHost((void**)&u_ref, N * N * sizeof(double));
  cudaMallocHost((void**)&u_ref1, M * M * sizeof(double));
  init_u(u_ref, N);
  init_u(u_ref1, M);
  t.tic();
  jacobi2d(u_ref1, f, N, n_itr);
  double tt = t.toc();
  printf("CPU time = %f s\n", tt);
  printf("CPU flops = %f GFlop/s\n", n_itr* 2*(N-2)*(N-2)*4/tt*1e-9);
  get_u_ref(u_ref, u_ref1, N);
  cudaFreeHost(u_ref1);


  /* Allocate GPU memory */
  double *u_d, *u_tmp, *f_d;
  cudaMalloc(&u_d, N * N * sizeof(double));
  cudaMalloc(&u_tmp, N * N * sizeof(double));
  cudaMalloc(&f_d, N * N * sizeof(double));
  /* Copy data to GPU */
  cudaMemcpy(u_d, u, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(u_tmp, u, N * N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(f_d, f, N * N * sizeof(double), cudaMemcpyHostToDevice);
  double h2 = 1.0 / ((N + 1) * (N + 1));
  cudaMemcpyToSymbol(&h2_d, &h2, sizeof(double));

  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim(N/(BLOCK_DIM-2)+1, N/(BLOCK_DIM-2)+1);
  jacobi_kernel<<<gridDim,blockDim>>>(u_d, u_tmp, f_d, N);

  /* Iteration on GPU */
  cudaDeviceSynchronize();
  t.tic();
  for (long i = 0; i < n_itr; i++) {
    jacobi_kernel<<<gridDim,blockDim>>>(u_d, u_tmp, f_d, N);
    /* swap pointers */
    double* p_tmp = u_d;
    u_d = u_tmp;
    u_tmp = p_tmp;
  }
  cudaDeviceSynchronize();
  tt = t.toc();
  printf("GPU time = %f s\n", tt);
  printf("GPU flops = %f GFlop/s\n", n_itr* 2*(N-2)*(N-2)*4/tt*1e-9);

  /* Copy data back to host */
  cudaMemcpy(u, u_d, N * N * sizeof(double), cudaMemcpyDeviceToHost);

  /* Print error */
//  double err = 0;
//  for (long i = 0; i < N*N; i++) err = std::max(err, fabs(u[i]-u_ref[i]));
//  printf("Error = %e\n", err);
  
  /* Free memory */
  cudaFreeHost(u);
  cudaFreeHost(f);
  cudaFreeHost(u_ref);
  cudaFree(u_d);
  cudaFree(u_tmp);
  cudaFree(f_d);

  return 0;
}

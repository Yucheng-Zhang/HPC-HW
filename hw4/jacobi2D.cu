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
  cudaMemcpyToSymbol(h2_d, h2, sizeof(h2));

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

  /* Copy data back to host */
  cudaMemcpy(u, u_d, N * N * sizeof(double), cudaMemcpyDeviceToHost);
  
  /* Free memory */
  cudaFreeHost(u);
  cudaFreeHost(f);
  cudaFree(u_d);
  cudaFree(u_tmp);
  cudaFree(f_d);

  return 0;
}

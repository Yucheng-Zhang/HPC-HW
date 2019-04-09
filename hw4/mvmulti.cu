#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void mvmulti(double* sum_ptr, const double* a, const double* m, long N){
  for (long i = 0; i < N; i++) {
    double sum = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (long j = 0; j < N; j++) {
      sum += m[i * N + j] * a[j];
    }
    sum_ptr[i] = sum;
  }
}

#define BLOCK_SIZE 1024

__global__ void vvmulti_kernel(double* sum, const double* a, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx] * b[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void reduction_kernel(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

int main() {
  long N = (1UL<<14);

  /* Initialize vector and matrix */
  double *a, *m;
  cudaMallocHost((void**)&a, N*sizeof(double));
  cudaMallocHost((void**)&m, N*N*sizeof(double));
  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    a[i] = 1.0 / (i+1);
    for (long j = 0; j < N; j++) {
      m[i * N + j] = 1.0; // row major order
    }
  }

  /* CPU reference */
  double *sum_ref;
  cudaMallocHost((void**)&sum_ref, N*sizeof(double));
  double tt = omp_get_wtime();
  mvmulti(sum_ref, a, m, N);
  printf("CPU Bandwidth = %f GB/s\n", (N+1)*N*sizeof(double)/(omp_get_wtime()-tt)/1e9);

  /* GPU */
  double *sum;
  cudaMallocHost((void**)&sum, N*sizeof(double));

  double *a_d, *m_d, *sum_d;
  cudaMalloc(&a_d, N*sizeof(double));
  cudaMalloc(&m_d, N*N*sizeof(double));
  long N_work = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)) N_work += i;
  cudaMalloc(&sum_d, N_work*sizeof(double)); // extra memory buffer for reduction across thread-blocks
  
  /* Copy Host data to device */
  cudaMemcpyAsync(a_d, a, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cudaMemcpyAsync(m_d, m, N*N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  tt = omp_get_wtime();

  for (long i = 0; i < N; i++){
    long Nb = (N+BLOCK_SIZE-1)/BLOCK_SIZE;
    vvmulti_kernel<<<Nb,BLOCK_SIZE>>>(sum_d, a_d, m_d+i*N, N);
    while (Nb > 1) {
      long N = Nb;
      Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
      reduction_kernel<<<Nb,BLOCK_SIZE>>>(sum_d + N, sum_d, N);
      sum_d += N;
    }
  double sum_tmp;
  cudaMemcpyAsync(&sum_tmp, sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  sum[i] = sum_tmp;
  }
  
  printf("GPU Bandwidth = %f GB/s\n", (N+1)*N*sizeof(double)/(omp_get_wtime()-tt)/1e9);

  double max_err = 0;
  for (long i = 0; i < N; i++)
    max_err = std::max(max_err, fabs(sum[i] - sum_ref[i]));
  printf("Error = %f\n", max_err);

  cudaFree(a_d);
  cudaFree(m_d);
  cudaFree(sum_d);
  cudaFreeHost(a);
  cudaFreeHost(m);

  return 0;
}

#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

struct RGBImage {
  long Xsize;
  long Ysize;
  float* A;
};
void read_image(const int N, RGBImage* I) {
  I->Xsize = N;
  I->Ysize = N;
  I->A = NULL;

  I->A = (float*) malloc(N*N*sizeof(float));
  for (long i = 0; i < N*N; i++) {
    I->A[i] = 0.;
  }
}

void free_image(RGBImage* I) {
  long N = I->Xsize * I->Ysize;
  if (N) free(I->A);
  I->A = NULL;
}


#define FWIDTH 3

// filter for jacobi
float filter[FWIDTH][FWIDTH] = {
     0,      0.25,      0,
  0.25,         0,   0.25,
     0,      0.25,      0};


void CPU_convolution(float* I, const float* I0, long Xsize, long Ysize) {
  constexpr long FWIDTH_HALF = (FWIDTH-1)/2;
  float h2 = 0.25 / ((Xsize + 1)*(Ysize + 1));
  #pragma omp parallel for collapse(2) schedule(static)
  for (long i0 = 0; i0 <= Xsize-FWIDTH; i0++) {
    for (long i1 = 0; i1 <= Ysize-FWIDTH; i1++) {
      float sum = 0;
      for (long j0 = 0; j0 < FWIDTH; j0++) {
        for (long j1 = 0; j1 < FWIDTH; j1++) {
          sum += I0[(i0+j0)*Ysize + (i1+j1)] * filter[j0][j1];
        }
      }
      I[(i0+FWIDTH_HALF)*Ysize + (i1+FWIDTH_HALF)] = (float)fabs(sum) + h2;
    }
  }
}


#define BLOCK_DIM 32
__constant__ float filter_gpu[FWIDTH][FWIDTH];

__global__ void GPU_convolution_no_smem(float* I, const float* I0, long Xsize, long Ysize) {
  constexpr long FWIDTH_HALF = (FWIDTH-1)/2;
  long offset_x = blockIdx.x * (BLOCK_DIM-FWIDTH);
  long offset_y = blockIdx.y * (BLOCK_DIM-FWIDTH);

  float sum = 0;
  for (long j0 = 0; j0 < FWIDTH; j0++) {
    for (long j1 = 0; j1 < FWIDTH; j1++) {
      sum += I0[(offset_x + threadIdx.x + j0)*Ysize + (offset_y + threadIdx.y + j1)] * filter_gpu[j0][j1];
    }
  }

  if (threadIdx.x+FWIDTH < BLOCK_DIM && threadIdx.y+FWIDTH < BLOCK_DIM)
    if (offset_x+threadIdx.x+FWIDTH <= Xsize && offset_y+threadIdx.y+FWIDTH <= Ysize)
      I[(offset_x+threadIdx.x+FWIDTH_HALF)*Ysize + (offset_y+threadIdx.y+FWIDTH_HALF)] = (float)fabs(sum);
}

__global__ void GPU_convolution(float* I, const float* I0, long Xsize, long Ysize) {
  constexpr long FWIDTH_HALF = (FWIDTH-1)/2;
  __shared__ float smem[BLOCK_DIM+FWIDTH][BLOCK_DIM+FWIDTH];
  long offset_x = blockIdx.x * (BLOCK_DIM-FWIDTH);
  long offset_y = blockIdx.y * (BLOCK_DIM-FWIDTH);
  float h2 = 0.25 / ((Xsize + 1)*(Ysize + 1));

  smem[threadIdx.x][threadIdx.y] = 0;
  if (offset_x + threadIdx.x < Xsize && offset_y + threadIdx.y < Ysize)
    smem[threadIdx.x][threadIdx.y] = I0[(offset_x + threadIdx.x)*Ysize + (offset_y + threadIdx.y)];
  __syncthreads();

  float sum = 0;
  for (long j0 = 0; j0 < FWIDTH; j0++) {
    for (long j1 = 0; j1 < FWIDTH; j1++) {
      sum += smem[threadIdx.x+j0][threadIdx.y+j1] * filter_gpu[j0][j1];
    }
  }

  if (threadIdx.x+FWIDTH < BLOCK_DIM && threadIdx.y+FWIDTH < BLOCK_DIM)
    if (offset_x+threadIdx.x+FWIDTH <= Xsize && offset_y+threadIdx.y+FWIDTH <= Ysize)
      I[(offset_x+threadIdx.x+FWIDTH_HALF)*Ysize + (offset_y+threadIdx.y+FWIDTH_HALF)] = (float)fabs(sum) + h2;
}


int main() {
  long repeat = 500;
  long N = 1000;

  // Load image from file
  RGBImage I0, I1, I1_ref;
  read_image(N, &I0);
  read_image(N, &I1);
  read_image(N, &I1_ref);
  long Xsize = I0.Xsize;
  long Ysize = I0.Ysize;

  // Filter on CPU
  Timer t;
  t.tic();
  for (long i = 0; i < repeat; i++) CPU_convolution(I1_ref.A, I0.A, Xsize, Ysize);
  double tt = t.toc();
  printf("CPU time = %fs\n", tt);
  printf("CPU flops = %fGFlop/s\n", repeat * 2*(Xsize-FWIDTH)*(Ysize-FWIDTH)*FWIDTH*FWIDTH/tt*1e-9);

  // Allocate GPU memory
  float *I0gpu, *I1gpu;
  cudaMalloc(&I0gpu, Xsize*Ysize*sizeof(float));
  cudaMalloc(&I1gpu, Xsize*Ysize*sizeof(float));
  cudaMemcpy(I0gpu, I0.A, Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(I1gpu, I1.A, Xsize*Ysize*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(filter_gpu, filter, sizeof(filter_gpu)); // Initialize filter_gpu

  // Create streams
  cudaStream_t streams[1];
  cudaStreamCreate(&streams[0]);

  // Dry run
  dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
  dim3 gridDim(Xsize/(BLOCK_DIM-FWIDTH)+1, Ysize/(BLOCK_DIM-FWIDTH)+1);
  GPU_convolution<<<gridDim,blockDim, 0, streams[0]>>>(I1gpu+0*Xsize*Ysize, I0gpu+0*Xsize*Ysize, Xsize, Ysize);

  // Filter on GPU
  cudaDeviceSynchronize();
  t.tic();
  for (long i = 0; i < repeat; i++) {
    GPU_convolution<<<gridDim,blockDim, 0, streams[0]>>>(I1gpu+0*Xsize*Ysize, I0gpu+0*Xsize*Ysize, Xsize, Ysize);
  }
  cudaDeviceSynchronize();
  tt = t.toc();
  printf("GPU time = %fs\n", tt);
  printf("GPU flops = %fGFlop/s\n", repeat * 2*(Xsize-FWIDTH)*(Ysize-FWIDTH)*FWIDTH*FWIDTH/tt*1e-9);

  // Print error
  float err = 0;
  cudaMemcpy(I1.A, I1gpu, Xsize*Ysize*sizeof(float), cudaMemcpyDeviceToHost);
  for (long i = 0; i < Xsize*Ysize; i++) err = std::max(err, fabs(I1.A[i] - I1_ref.A[i]));
  printf("Error = %e\n", err);

  // Free memory
  cudaStreamDestroy(streams[0]);
  cudaFree(I0gpu);
  cudaFree(I1gpu);
  free_image(&I0);
  free_image(&I1);
  free_image(&I1_ref);
  return 0;
}

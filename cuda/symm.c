#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Array initialization SEQ */
static void init_array_seq(int ni, int nj,
                           float *alpha,
                           float *beta,
                           float *C,
                           float *A,
                           float *B) {
  int i, j;
  *alpha = 32412;
  *beta = 2123;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
      C[i * nj + j] = ((float)i * j) / ni;
      B[i * nj + j] = ((float)i * j) / ni;
    }
  }
  for (i = 0; i < nj; i++) {
    for (j = 0; j < nj; j++) {
      A[i * nj + j] = ((float)i * j) / ni;
    }
  }
}

/* Array initialization CUDA */
static void init_array(int ni, int nj,
                       float *alpha,
                       float *beta,
                       float *C,
                       float *A,
                       float *B) {
  *alpha = 32412;
  *beta = 2123;
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      C[i * nj + j] = ((float)i * j) / ni;
      B[i * nj + j] = ((float)i * j) / ni;
    }
  }
  for (int i = 0; i < nj; i++) {
    for (int j = 0; j < nj; j++) {
      A[i * nj + j] = ((float)i * j) / ni;
    }
  }
}

/* DCE code - Print the array */
static void print_array(int ni, int nj, float *C) {
  int i, j;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
      fprintf(stderr, "%f ", C[i * nj + j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  }
  fprintf(stderr, "\n");
}

/* Sequential kernel */
static void kernel_symm_sequential(int ni, int nj,
                                   float alpha,
                                   float beta,
                                   float *C,
                                   float *A,
                                   float *B) {
  int i, j, k;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
      float acc = 0;
      for (k = 0; k < j - 1; k++) {
        C[k * nj + j] += alpha * A[k * nj + i] * B[i * nj + j];
        acc += B[k * nj + j] * A[k * nj + i];
      }
      C[i * nj + j] = beta * C[i * nj + j] + alpha * A[i * nj + i] * B[i * nj + j] + alpha * acc;
    }
  }
}

/* CUDA kernel */
__global__ void kernel_symm_cuda(int ni, int nj,
                                 float alpha,
                                 float beta,
                                 float *C,
                                 float *A,
                                 float *B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < ni && j < nj) {
    float acc = 0;
    for (int k = 0; k < j - 1; k++) {
      C[k * nj + j] += alpha * A[k * nj + i] * B[i * nj + j];
      acc += B[k * nj + j] * A[k * nj + i];
    }
    C[i * nj + j] = beta * C[i * nj + j] + alpha * A[i * nj + i] * B[i * nj + j] + alpha * acc;
  }
}

/* Compare matrices */
int compare_matrices(int ni, int nj, float *C_seq, float *C_par) {
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++) {
      if (fabs(C_seq[i * nj + j] - C_par[i * nj + j]) > 1e-6) {
        printf("Difference found in C[%d][%d]: %f != %f\n", i, j, C_seq[i * nj + j], C_par[i * nj + j]);
        return 0;
      }
    }
  }
  return 1;
}

/* Main function */
int main(int argc, char **argv) {
  int ni = 1024;  // Matrix dimensions
  int nj = 1024;

  float alpha, beta;
  float *A, *B, *C_seq, *C_par;
  float *d_A, *d_B, *d_C;

  // Allocate host memory
  A = (float *)malloc(ni * nj * sizeof(float));
  B = (float *)malloc(ni * nj * sizeof(float));
  C_seq = (float *)malloc(ni * nj * sizeof(float));
  C_par = (float *)malloc(ni * nj * sizeof(float));

  // Initialize arrays
  init_array_seq(ni, nj, &alpha, &beta, C_seq, A, B);

  // Sequential execution
  double start_time_seq = omp_get_wtime();
  kernel_symm_sequential(ni, nj, alpha, beta, C_seq, A, B);
  double seq_time = omp_get_wtime() - start_time_seq;

  // Allocate device memory
  cudaMalloc((void **)&d_A, ni * nj * sizeof(float));
  cudaMalloc((void **)&d_B, ni * nj * sizeof(float));
  cudaMalloc((void **)&d_C, ni * nj * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_A, A, ni * nj * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, ni * nj * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C_par, ni * nj * sizeof(float), cudaMemcpyHostToDevice);

  // Set up execution configuration
  dim3 blockSize(16, 16);
  dim3 gridSize((ni + 15) / 16, (nj + 15) / 16);

  // CUDA execution
  double start_time_cuda = omp_get_wtime();
  kernel_symm_cuda<<<gridSize, blockSize>>>(ni, nj, alpha, beta, d_C, d_A, d_B);
  cudaDeviceSynchronize();
  double cuda_time = omp_get_wtime() - start_time_cuda;

  // Copy result back to host
  cudaMemcpy(C_par, d_C, ni * nj * sizeof(float), cudaMemcpyDeviceToHost);

  // Compare results
  if (compare_matrices(ni, nj, C_seq, C_par)) {
    printf("The matrices are equal.\n");
  } else {
    printf("The matrices are NOT equal.\n");
  }

  // Print execution times
  printf("Sequential execution time: %f seconds\n", seq_time);
  printf("CUDA execution time: %f seconds\n", cuda_time);

  // Free memory
  free(A);
  free(B);
  free(C_seq);
  free(C_par);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  return 0;
}

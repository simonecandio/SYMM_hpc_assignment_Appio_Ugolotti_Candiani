#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda.h>
#include <symm.hu>

#ifndef CUDA_NTHREADS
#define CUDA_NTHREADS 128
#endif

/* Funzione di inizializzazione sequenziale */
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

/* Kernel sequenziale */
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

/* Kernel CUDA */
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

/*

__global__ void kernel_symm_cuda_optimized(int ni, int nj,

                                           float alpha, float beta,

                                           float *C, float *A, float *B) {

  // Shared memory per blocchi di A e B

  __shared__ float shared_A[16][16];

  __shared__ float shared_B[16][16];



  // Indici globali per thread

  int row = blockIdx.x * blockDim.x + threadIdx.x;

  int col = blockIdx.y * blockDim.y + threadIdx.y;



  // Indici locali per la shared memory

  int tx = threadIdx.x;

  int ty = threadIdx.y;



  // Accumulatore locale per il risultato

  float acc = 0.0f;



  // Carica i dati in shared memory e calcola il prodotto parziale

  for (int t = 0; t < (nj + 15) / 16; t++) {

    // Caricamento della memoria globale in memoria condivisa

    if (row < ni && (t * 16 + ty) < nj) {

      shared_A[tx][ty] = A[row * nj + t * 16 + ty];

    } else {

      shared_A[tx][ty] = 0.0f;

    }



    if (col < nj && (t * 16 + tx) < nj) {

      shared_B[tx][ty] = B[(t * 16 + tx) * nj + col];

    } else {

      shared_B[tx][ty] = 0.0f;

    }



    // Sincronizzazione dei thread per garantire che tutti i dati siano caricati

    __syncthreads();



    // Calcolo del prodotto parziale con unrolling

    for (int k = 0; k < 16; k++) {

      acc += shared_A[tx][k] * shared_B[k][ty];

    }



    // Sincronizzazione prima di ricaricare i nuovi blocchi

    __syncthreads();

  }



  // Scrittura del risultato finale nella matrice C

  if (row < ni && col < nj) {

    C[row * nj + col] = beta * C[row * nj + col] + alpha * acc;

  }

}



*/



/* Compare matrices */

/* Funzione per confrontare le matrici */
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

/* Funzione principale */
int main(int argc, char **argv) {
  int ni = 1024;  // Dimensioni della matrice
  int nj = 1024;

  float alpha, beta;
  float *A, *B, *C_seq, *C_par;
  float *d_A, *d_B, *d_C;

  // Stampa dimensioni della matrice
  printf("Matrix dimensions: ni = %d, nj = %d\n", ni, nj);

  // Allocazione della memoria host
  printf("Allocating host memory...\n");
  A = (float *)malloc(ni * nj * sizeof(float));
  B = (float *)malloc(ni * nj * sizeof(float));
  C_seq = (float *)malloc(ni * nj * sizeof(float));
  C_par = (float *)malloc(ni * nj * sizeof(float));

  // Inizializzazione degli array
  printf("Initializing arrays...\n");
  init_array_seq(ni, nj, &alpha, &beta, C_seq, A, B);

  // Esecuzione sequenziale
  printf("Starting sequential kernel...\n");
  double start_time_seq = omp_get_wtime();
  kernel_symm_sequential(ni, nj, alpha, beta, C_seq, A, B);
  double seq_time = omp_get_wtime() - start_time_seq;
  printf("Sequential execution completed. Time: %f seconds\n", seq_time);

  // Allocazione della memoria device
  printf("Allocating device memory...\n");
  cudaMalloc((void **)&d_A, ni * nj * sizeof(float));
  cudaMalloc((void **)&d_B, ni * nj * sizeof(float));
  cudaMalloc((void **)&d_C, ni * nj * sizeof(float));

  // Copia dei dati sul device
  printf("Copying data to device...\n");
  cudaMemcpy(d_A, A, ni * nj * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, ni * nj * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C_par, ni * nj * sizeof(float), cudaMemcpyHostToDevice);

  // Configurazione del kernel CUDA
  dim3 blockSize(16, 16);
  dim3 gridSize((ni + 15) / 16, (nj + 15) / 16);

  // Esecuzione CUDA
  printf("Launching CUDA kernel...\n");
  double start_time_cuda = omp_get_wtime();
  kernel_symm_cuda<<<gridSize, blockSize>>>(ni, nj, alpha, beta, d_C, d_A, d_B);
  cudaDeviceSynchronize();
  double cuda_time = omp_get_wtime() - start_time_cuda;
  printf("CUDA execution completed. Time: %f seconds\n", cuda_time);

  // Copia del risultato sul host
  printf("Copying result back to host...\n");
  cudaMemcpy(C_par, d_C, ni * nj * sizeof(float), cudaMemcpyDeviceToHost);

  // Confronto dei risultati
  printf("Comparing matrices...\n");
  if (compare_matrices(ni, nj, C_seq, C_par)) {
    printf("The matrices are equal.\n");
  } else {
    printf("The matrices are NOT equal.\n");
  }

  // Stampa dei tempi di esecuzione
  printf("Results:\n");
  printf("Sequential execution time: %f seconds\n", seq_time);
  printf("CUDA execution time: %f seconds\n", cuda_time);

  // Libera la memoria
  printf("Freeing memory...\n");
  free(A);
  free(B);
  free(C_seq);
  free(C_par);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  printf("Program completed successfully.\n");
  return 0;
}

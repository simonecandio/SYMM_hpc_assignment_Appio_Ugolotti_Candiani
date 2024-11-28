#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
//#include "polybench.hu"
#include "symm.hu"

#ifndef CUDA_NTHREADS
#define CUDA_NTHREADS 128
#endif

/* Array initialization (host side) */
static void init_array_seq(int ni, int nj,
                           DATA_TYPE *alpha,
                           DATA_TYPE *beta,
                           DATA_TYPE *C,
                           DATA_TYPE *A,
                           DATA_TYPE *B)
{
    int i, j;
  
    *alpha = 32412.0;
    *beta = 2123.0;

    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            C[i * nj + j] = ((DATA_TYPE)i * j) / ni;
            B[i * nj + j] = ((DATA_TYPE)i * j) / ni;
        }
    }

    for (i = 0; i < nj; i++)
    {
        for (j = 0; j < nj; j++)
        {
            A[i * nj + j] = ((DATA_TYPE)i * j) / ni;
        }
    }
}

/* Sequential symm kernel */
static void kernel_symm_sequential(int ni, int nj,
                                   DATA_TYPE alpha,
                                   DATA_TYPE beta,
                                   DATA_TYPE *C,
                                   DATA_TYPE *A,
                                   DATA_TYPE *B)
{
    int i, j, k;
    for (i = 0; i < ni; i++)
    {
        for (j = 0; j < nj; j++)
        {
            DATA_TYPE acc = 0.0;
            for (k = 0; k < j - 1; k++)
            {
                C[k * nj + j] += alpha * A[k * nj + i] * B[i * nj + j];
                acc += B[k * nj + j] * A[k * nj + i];
}
            C[i * nj + j] = beta * C[i * nj + j] + alpha * A[i * nj + i] * B[i * nj + j] + alpha * acc;
        }
    }
}


__global__ void kernel_symm_cuda(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta,
                                 DATA_TYPE *C, DATA_TYPE *A, DATA_TYPE *B) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // Riga della matrice
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // Colonna della matrice

    if (i < ni && j < nj) {
        DATA_TYPE acc = 0.0;

        // Calcolo della somma accumulativa per l'elemento corrente
        for (int k = 0; k < nj; k++) {
            acc += B[k * nj + j] * A[k * nj + i];
        }

        // Aggiornamento del valore finale della matrice C
        C[i * nj + j] = beta * C[i * nj + j] + alpha * A[i * nj + i] * B[i * nj + j] + alpha * acc;
    }
}



int compare_matrices(int ni, int nj, DATA_TYPE *C1, DATA_TYPE *C2, double tolerance)
{
    for (int i = 0; i < ni; i++)
    {
        for (int j = 0; j < nj; j++)
        {

            if (fabs(C1[i * nj + j] - C2[i * nj + j]) > tolerance)
            {
             //printf("C1[%d][%d] = %f, C2[%d][%d] = %f, diff = %f\n",i, j, C1[i * nj + j], i, j, C2[i * nj + j],fabs(C1[i * nj + j] - C2[i * nj + j]));
             return 0; // Matrices are not equal
            }
        }
    }
    return 1; // Matrices are equal
}



/* Host code for CUDA execution */
int main(int argc, char **argv)
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    DATA_TYPE *C_seq,*C_gpu, *A, *B;
    DATA_TYPE *d_C, *d_A, *d_B;

    int blockSizeX = 16;
    int blockSizeY = 16;

    C_seq = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
    C_gpu = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));
    A = (DATA_TYPE *)malloc(NJ * NJ * sizeof(DATA_TYPE));
    B = (DATA_TYPE *)malloc(NI * NJ * sizeof(DATA_TYPE));

    cudaMalloc((void **)&d_C, NI * NJ * sizeof(DATA_TYPE));
    cudaMalloc((void **)&d_A, NJ * NJ * sizeof(DATA_TYPE));
    cudaMalloc((void **)&d_B, NI * NJ * sizeof(DATA_TYPE));

    /* Initialize arrays on host */
    init_array_seq(ni, nj, &alpha, &beta, C_seq, A, B);

    /* Copy initialized data to GPU */
    cudaMemcpy(d_C, C_seq, NI * NJ * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A, A, NJ * NJ * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, NI * NJ * sizeof(DATA_TYPE), cudaMemcpyHostToDevice);
     /* Execute sequential version */
    printf("Executing sequential kernel...\n");
    double seq_start =  clock();
    kernel_symm_sequential(ni, nj, alpha, beta, C_seq, A, B);
    double seq_end = clock();
    double seq_time =( (double)(seq_end - seq_start)) / CLOCKS_PER_SEC * 1000.0; // Convert to milliseconds
    printf("Sequential kernel execution time: %f ms\n", seq_time);


    /* Define grid and block dimensions */
    dim3 block(blockSizeX, blockSizeY);
    dim3 grid((ni + blockSizeX - 1) / blockSizeX, (nj + blockSizeY - 1) / blockSizeY);


    /* CUDA events for timing */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Start timing */
    cudaEventRecord(start);

    /* Launch CUDA kernel */
    printf("Executing CUDA kernel...\n");
    kernel_symm_cuda<<<grid, block>>>(ni, nj, alpha, beta, d_C, d_A, d_B);
    cudaDeviceSynchronize();

    /* Stop timing */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel execution time (CUDA): %f ms\n", milliseconds);


    /* Copy results back to host */
    cudaMemcpy(C_gpu, d_C, NI * NJ * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost);

    if (compare_matrices(ni, nj, C_seq, C_gpu,1e-3))
        printf("The calculated matrices are equal.\n");
      else
      {
        printf("The computed matrices are NOT equal.\n");
        return 0;
      }

    /* Free memory */
    free(C_seq);
    free(C_gpu);
    free(A);
    free(B);
    cudaFree(d_C);
    cudaFree(d_A);
    cudaFree(d_B);

    /* Destroy CUDA events */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
                                       

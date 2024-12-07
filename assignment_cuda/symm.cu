#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include "symm.hu"


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

__global__ void symm_kernel_row(int nj, int i, DATA_TYPE alpha, DATA_TYPE beta,
                                DATA_TYPE *C, DATA_TYPE *A, DATA_TYPE *B) {

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < nj) {
        DATA_TYPE acc = 0.0;

        // Add the contribution from k to j-1
        for (int k = 0; k < j - 1; ++k) {
            C[k * nj + j] += alpha * A[k * nj + i] * B[i * nj + j];
            acc += B[k * nj + j] * A[k * nj + i];
        }

        // Final calculation for C[i][j]
        C[i * nj + j] = beta * C[i * nj + j] + alpha * A[i * nj + i] * B[i * nj + j] + alpha * acc;
    }
}


int compare_matrices(int ni, int nj, DATA_TYPE *C, DATA_TYPE *C_original){
     for (int i = 0; i < ni; i++)
    {
        for (int j = 0; j < nj; j++)
        {
         if (fabs(C[i * nj + j] - C_original[i * nj + j]) > 1e-3) {
         printf("C_seq[%d][%d] = %f, C_gpu[%d][%d] = %f\n",i, j, C[i * nj + j], i, j, C_original[i * nj + j]);
         return 0;
         }

        }
    }
    return 1;
}


void symmCuda(int ni, int nj, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *C, DATA_TYPE *A, DATA_TYPE *B, DATA_TYPE *C_outputFromGpu,float *gpu_time)
{
   int DIM_THREAD_BLOCK_X = 32;
    DATA_TYPE *A_gpu;
    DATA_TYPE *B_gpu;
    DATA_TYPE *C_gpu;


    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * ni * nj);
    cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * ni * nj);
    cudaMalloc((void **)&C_gpu, sizeof(DATA_TYPE) * ni * nj);
    cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice);
    cudaMemcpy(C_gpu, C, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyHostToDevice);

    dim3 block(DIM_THREAD_BLOCK_X);
    dim3 grid((nj + DIM_THREAD_BLOCK_X - 1) / DIM_THREAD_BLOCK_X);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Start timing */
    cudaEventRecord(start);
    
    // Launch a kernel for each line `i`

    for (int i = 0; i < ni; i++) {
       symm_kernel_row<<<grid, block>>>(nj, i, alpha, beta, C_gpu, A_gpu, B_gpu);
       //cudaDeviceSynchronize();
    }

    /* Stop timing */
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(gpu_time, start, stop);

    //printf("Kernel execution time (CUDA): %f s\n", gpu_time / 1000.0);

    /* Stop and print timer. */


    cudaMemcpy(C_outputFromGpu, C_gpu, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyDeviceToHost);

    cudaFree(C_gpu);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    /* Destroy CUDA events */
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/* Host code for CUDA execution */
int main(int argc, char **argv)
{
    /* Retrieve problem size. */
    int ni = NI;
    int nj = NJ;
    float total_seq_time = 0.0, total_par_time = 0.0, total_speedup = 0.0;
    int num_runs =3; // Number of runs to calculate the average

     for (int run = 0; run < num_runs; run++) {
    float seq_start=0.0,gpu_time=0.0,seq_end=0.0,seq_time=0.0,speedup=0.0;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;

    DATA_TYPE *A = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * ni * nj);
    DATA_TYPE *B = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * ni * nj);
    DATA_TYPE *C = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * ni * nj);
    DATA_TYPE *C_outputFromGpu = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * ni * nj);

    init_array_seq(ni, nj, &alpha, &beta, C, A, B);

    symmCuda(ni, nj, alpha, beta, C, A, B, C_outputFromGpu, &gpu_time);
    //printf("Kernel execution time (CUDA): %f s\n", gpu_time / 1000.0);
    total_par_time += (gpu_time/1000.0);


    seq_start =  clock();
    kernel_symm_sequential(ni, nj, alpha, beta, C, A, B);
    seq_end = clock();
    seq_time =( (double)(seq_end - seq_start)) / CLOCKS_PER_SEC ; 
    //printf("Sequential kernel execution time: %f s\n", seq_time);
    total_seq_time += seq_time;


    if (run == 0) {
     if (compare_matrices(ni, nj, C, C_outputFromGpu)) {
      printf("The matrices are equal.\n");
      }
      else {
      printf("The matrices are NOT equal.\n");
      return 0;
      }
    }

    speedup = seq_time / (gpu_time / 1000.0);
    total_speedup += speedup;

    free(A);
    free(B);
    free(C);
    free(C_outputFromGpu);
}

    /* Calculate the averages */
    double avg_seq_time = total_seq_time / num_runs;
    double avg_par_time = total_par_time / num_runs;
    double avg_speedup = total_speedup / num_runs;

    /* Print average results */
    printf("\nAverage Sequential Execution Time over %d runs: %f seconds\n", num_runs, avg_seq_time);
    printf("Average Parallel Execution Time over %d runs: %f seconds\n", num_runs, avg_par_time);
    printf("Average Speedup over %d runs: %f\n", num_runs, avg_speedup);

    return 0;
}

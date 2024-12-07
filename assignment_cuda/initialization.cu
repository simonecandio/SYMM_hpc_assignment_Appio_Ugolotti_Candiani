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

/* CUDA kernel for matrix initialization */
__global__ void init_array_cuda(int ni, int nj, DATA_TYPE *C, DATA_TYPE *A, DATA_TYPE *B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < ni && col < nj)
    {
        C[row * nj + col] = ((DATA_TYPE)row * col) / ni;
        B[row * nj + col] = ((DATA_TYPE)row * col) / ni;
    }

    if (row < nj && col < nj)
    {
        A[row * nj + col] = ((DATA_TYPE)row * col) / ni;
    }
}

/* Compare matrices */
int compare_matrices(int ni, int nj, DATA_TYPE *mat1, DATA_TYPE *mat2)
{
    for (int i = 0; i < ni; i++)
    {
        for (int j = 0; j < nj; j++)
        {
            if (fabs(mat1[i * nj + j] - mat2[i * nj + j]) > 1e-6)
            {
                printf("Mismatch at [%d][%d]: CPU=%f, GPU=%f\n",
                       i, j, mat1[i * nj + j], mat2[i * nj + j]);
                return 0;
            }
        }
    }
    return 1;
}

/* Initialize matrices using CUDA */
void init_array_cuda_wrapper(int ni, int nj, DATA_TYPE *C, DATA_TYPE *A, DATA_TYPE *B)
{
    DATA_TYPE *d_C, *d_A, *d_B;
    cudaMalloc((void **)&d_C, sizeof(DATA_TYPE) * ni * nj);
    cudaMalloc((void **)&d_A, sizeof(DATA_TYPE) * nj * nj);
    cudaMalloc((void **)&d_B, sizeof(DATA_TYPE) * ni * nj);

    dim3 block(16, 16);
    dim3 grid((nj + block.x - 1) / block.x, (max(ni, nj) + block.y - 1) / block.y);

    init_array_cuda<<<grid, block>>>(ni, nj, d_C, d_A, d_B);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyDeviceToHost);
    cudaMemcpy(A, d_A, sizeof(DATA_TYPE) * nj * nj, cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, sizeof(DATA_TYPE) * ni * nj, cudaMemcpyDeviceToHost);

    cudaFree(d_C);
    cudaFree(d_A);
    cudaFree(d_B);
}

int main(int argc, char **argv)
{
    /* Problem size */
    int ni = NI;
    int nj = NJ;

    /* Variable declaration/allocation */
    DATA_TYPE *A_seq = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * nj * nj);
    DATA_TYPE *B_seq = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * ni * nj);
    DATA_TYPE *C_seq = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * ni * nj);

    DATA_TYPE *A_cuda = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * nj * nj);
    DATA_TYPE *B_cuda = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * ni * nj);
    DATA_TYPE *C_cuda = (DATA_TYPE *)malloc(sizeof(DATA_TYPE) * ni * nj);

    DATA_TYPE alpha, beta;

    /* Initialize arrays sequentially */
    float seq_start = clock();
    init_array_seq(ni, nj, &alpha, &beta, C_seq, A_seq, B_seq);
    float seq_end = clock();
    float seq_time =( (float)(seq_end - seq_start)) / CLOCKS_PER_SEC ; // Convert to milliseconds
    printf("Sequential init execution time: %f s\n", seq_time);
    float par_start =  clock();
    /* Initialize arrays using CUDA */
    init_array_cuda_wrapper(ni, nj, C_cuda, A_cuda, B_cuda);
    float par_end = clock();
    float par_time =( (float)(par_end - par_start)) / CLOCKS_PER_SEC ; // Convert to milliseconds
    printf("Parallel init  execution time: %f s\n", par_time);

    /* Compare matrices */
    printf("Comparing matrix C...\n");
    if (compare_matrices(ni, nj, C_seq, C_cuda))
        printf("Matrix C is identical.\n");
    else
        printf("Matrix C differs.\n");

    printf("Comparing matrix B...\n");
    if (compare_matrices(ni, nj, B_seq, B_cuda))
        printf("Matrix B is identical.\n");
    else
        printf("Matrix B differs.\n");

    printf("Comparing matrix A...\n");
    if (compare_matrices(nj, nj, A_seq, A_cuda))
        printf("Matrix A is identical.\n");
    else
        printf("Matrix A differs.\n");

    /* Free memory */
    free(A_seq);
    free(B_seq);
    free(C_seq);
    free(A_cuda);
    free(B_cuda);
    free(C_cuda);

    return 0;
}

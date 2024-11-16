#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
/* Default data type is double, default size is 4000. */
#include "symm.h"


/* Array initialization SEQ. */
static void init_array_seq(int ni, int nj,
                       DATA_TYPE *alpha,
                       DATA_TYPE *beta,
                       DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(A, NJ, NJ, nj, nj),
                       DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
  int i, j;

  *alpha = 32412;
  *beta = 2123;
    for (i = 0; i < ni; i++)
      for (j = 0; j < nj; j++)
      {
        C[i][j] = ((DATA_TYPE)i * j) / ni;
        B[i][j] = ((DATA_TYPE)i * j) / ni;
      }
    for (i = 0; i < nj; i++)
      for (j = 0; j < nj; j++)
        A[i][j] = ((DATA_TYPE)i * j) / ni;
}  
    




/* Array initialization. */
static void init_array(int ni, int nj,
                       DATA_TYPE *alpha,
                       DATA_TYPE *beta,
                       DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                       DATA_TYPE POLYBENCH_2D(A, NJ, NJ, nj, nj),
                       DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
  int i, j;

  *alpha = 32412;
  *beta = 2123;
   #pragma omp parallel num_threads(4)
   {
    
    #pragma omp for collapse(2) schedule(static)
    for (i = 0; i < ni; i++)
      for (j = 0; j < nj; j++)
      {
        C[i][j] = ((DATA_TYPE)i * j) / ni;
        B[i][j] = ((DATA_TYPE)i * j) / ni;
      }
    
    #pragma omp for collapse(2) schedule(static)
    for (i = 0; i < nj; i++)
      for (j = 0; j < nj; j++)
        A[i][j] = ((DATA_TYPE)i * j) / ni;
    }  
    
}



/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int ni, int nj,
                        DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
    {
      fprintf(stderr, DATA_PRINTF_MODIFIER, C[i][j]);
      if ((i * ni + j) % 20 == 0)
        fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}
/* Main computational kernel (sequential).*/
static void kernel_symm_sequential(int ni, int nj,
                                   DATA_TYPE alpha,
                                   DATA_TYPE beta,
                                   DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                                   DATA_TYPE POLYBENCH_2D(A, NJ, NJ, nj, nj),
                                   DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
  int i, j, k;
  DATA_TYPE acc;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
      acc = 0;
      for (k = 0; k < j - 1; k++) {
        C[k][j] += alpha * A[k][i] * B[i][j];
        acc += B[k][j] * A[k][i];
      }
      C[i][j] = beta * C[i][j] + alpha * A[i][i] * B[i][j] + alpha * acc;
    }
  }
}


/* Main computational kernel. (parallel) */
static void kernel_symm(int ni, int nj,
                        DATA_TYPE alpha,
                        DATA_TYPE beta,
                        DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                        DATA_TYPE POLYBENCH_2D(A, NJ, NJ, nj, nj),
                        DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
  int i, j, k;
  DATA_TYPE acc;
  #pragma scop
  #pragma omp parallel num_threads(4)  
 {	
   #pragma omp for private(i,j,k) collapse(2) reduction(+:acc) schedule(static, 4) 
	 for (i = 0; i < _PB_NI; i++){
     for (j = 0; j < _PB_NJ; j++){
       acc = 0;
       for (k = 0; k < j - 1; k++)
       {        
        C[k][j] += alpha * A[k][i] * B[i][j];
        acc += B[k][j] * A[k][i];
       } 
       C[i][j] = beta * C[i][j] + alpha * A[i][i] * B[i][j] + alpha * acc;
 	   }
   }
 }
  #pragma endscop
}


/*  Second version : Main computational kernel. (parallel)
static void kernel_symm(int ni, int nj,
                        DATA_TYPE alpha,
                        DATA_TYPE beta,
                        DATA_TYPE POLYBENCH_2D(C, NI, NJ, ni, nj),
                        DATA_TYPE POLYBENCH_2D(A, NJ, NJ, nj, nj),
                        DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
  int i, j, k;
  DATA_TYPE acc;
  #pragma scop
  #pragma omp parallel private(i,j,k)  
 {	
   #pragma omp for collapse(2) reduction(+:acc) schedule(dynamic, 16) 
   for (i = 0; i < _PB_NI; i++){
     for (j = 0; j < _PB_NJ; j++){
       acc = 0;
       for (k = 0; k < j - 1; k++)
       {        
        C[k][j] += alpha * A[k][i] * B[i][j];
        acc += B[k][j] * A[k][i];
       }
       #pragma omp critical
       C[i][j] = beta * C[i][j] + alpha * A[i][i] * B[i][j] + alpha * acc;
 	   }
   }
 }
  #pragma endscop
}
*/


int compare_matrices(int ni, int nj,
                     DATA_TYPE POLYBENCH_2D(C_seq, NI, NJ, ni, nj),
                     DATA_TYPE POLYBENCH_2D(C_par, NI, NJ, ni, nj)) {
  int i, j;
  const double epsilon = 1e-5;
  for (i = 0; i < ni; i++) {
    for (j = 0; j < nj; j++) {
      //if (fabs(C_seq[i][j] - C_par[i][j]) > epsilon) {
        if(C_seq[i][j] != C_par[i][j]){
	printf("Difference found in C[%d][%d]: %f != %f\n", i, j, C_seq[i][j], C_par[i][j]);
        return 0;
      }
    }
  }
  return 1;
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  double total_seq_time = 0.0, total_par_time = 0.0, total_speedup = 0.0, num_threads=0.0, amdahl_speedup=0.0;
  int num_runs =3; // Number of runs to calculate the average

  for (int run = 0; run < num_runs; run++) {
    double start_time_seq_init1=0.0, start_time_par_init2=0.0, seq_time_init1=0.0, par_time_init2=0.0, start_time1=0.0, start_time_seq=0.0, seq_time=0.0, T_non_parallel_init=0.0, T_non_parallel_print=0.0, start_time_par=0.0, T_non_parallel=0.0;
    double PARALLEL_FRACTION=0.0, par_time=0.0, speedup=0.0, num_threads=0.0, amdahl_speedup=0.0;
 
    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(C_seq, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(C_par, DATA_TYPE, NI, NJ, ni, nj);
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, NJ, NJ, nj, nj);
    POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, NI, NJ, ni, nj);
  
    start_time_seq_init1 = omp_get_wtime();
    /* Initialize array(s) seq. */
    init_array_seq(ni, nj, &alpha, &beta,
                   POLYBENCH_ARRAY(C_seq),
                   POLYBENCH_ARRAY(A),
                   POLYBENCH_ARRAY(B));
    seq_time_init1 = omp_get_wtime() - start_time_seq_init1;
  

    start_time_seq = omp_get_wtime();
    kernel_symm_sequential(ni, nj, alpha, beta, POLYBENCH_ARRAY(C_seq), POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));
    seq_time = (omp_get_wtime() - start_time_seq) + seq_time_init1;
    total_seq_time += seq_time; 

    /* Initialize array(s) par. */
    start_time_par_init2 = omp_get_wtime();
    init_array(ni, nj, &alpha, &beta,
               POLYBENCH_ARRAY(C_par),
               POLYBENCH_ARRAY(A),
               POLYBENCH_ARRAY(B));
    par_time_init2 = omp_get_wtime() - start_time_par_init2;

    start_time_par = omp_get_wtime();
    kernel_symm(ni, nj,
                alpha, beta,
                POLYBENCH_ARRAY(C_par),
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(B));
    par_time = (omp_get_wtime() - start_time_par) + par_time_init2;
    total_par_time += par_time; 
  
    speedup = seq_time / par_time;
    total_speedup += speedup; 

   //print_array(ni, nj, POLYBENCH_ARRAY(B));
    /* Compare matrices only once to avoid overhead */
    if (run == 0) {
      if (compare_matrices(ni, nj, POLYBENCH_ARRAY(C_seq), POLYBENCH_ARRAY(C_par)))
        printf("The calculated matrices are equal.\n");
      else
      {
        printf("The computed matrices are not equal.\n");
        return 0;
      }
    }

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(C_seq);
    POLYBENCH_FREE_ARRAY(C_par);
    POLYBENCH_FREE_ARRAY(A);
    POLYBENCH_FREE_ARRAY(B);
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

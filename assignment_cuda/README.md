# GROUP 9 MEMBERS:
- Alessandro Appio
- Emanuele Ugolotti
- Simone Candiani

> ## Assigned application
>
> `OpenMP/linear-algebra/kernels/symm`

### Implemented functionality
 - [Creation of a function to compare the sequential and parallelized matrix CUDA](https://github.com/alleappio/hpc_assignment_1/blob/7f3a465ae524391a9890ad755887fb72a265fcad/symm/symm.c#L144C1-L159C2)
 - [Creation of a for loop to measure and calculate the average execution time](https://github.com/alleappio/hpc_assignment_1/blob/b766f69a4e8b23e6035146eb8309045773cef766/symm/symm.c#L167C2-L169C45)
 - [Automated time calculation](https://github.com/alleappio/hpc_assignment_1/blob/7f3a465ae524391a9890ad755887fb72a265fcad/symm/symm.c#L234C1-L242C71)
 - Created CUDA parallelizations:
   * [Creation a function to handle CUDA]
   * [Configuring CUDA blocks and grid]
   * [Creation of a for loop for call in CUDA]
   * [Kernel call in CUDA]
   - For Init:
     * [We decided to use the sequential function]

- [Creation an automatic benchmark for all datasets](https://github.com/alleappio/hpc_assignment_1/blob/develop_candiani/symm/bench.sh)

# Project Results 
## Average Execution Times and Speedups
To get the results run the bench.sh file.

----------------------------------------
### Small Dataset
Running make for dataset SMALL_DATASET
The matrices are equal.

- **Average Sequential Execution Time over 3 runs:** 0.029822 seconds
- **Average Parallel Execution Time over 3 runs:** 0.058014 seconds
- **Average Speedup over 3 runs:** 0.514048
Completed make for dataset SMALL_DATASET
----------------------------------------
### STANDARD Dataset
Running make for dataset STANDARD_DATASET
The matrices are equal.

- **Average Sequential Execution Time over 3 runs:** 38.415722 seconds
- **Average Parallel Execution Time over 3 runs:** 1.841017 seconds
- **Average Speedup over 3 runs:** 20.866573
  
##Completed make for dataset STANDARD_DATASET
----------------------------------------
### LARGE Dataset
Running make for dataset LARGE_DATASET
The matrices are equal.

- **Average Sequential Execution Time over 3 runs:** 103.639361 seconds
- **Average Parallel Execution Time over 3 runs:** 10.927332 seconds
- **Average Speedup over 3 runs:** 6.843851
Completed make for dataset LARGE_DATASET
----------------------------------------
### EXTRALARGE Dataset
Running make for dataset EXTRALARGE_DATASET
The matrices are equal.

- **Average Sequential Execution Time over 3 runs:** 961.69574 seconds
- **Average Parallel Execution Time over 3 runs:** 82.591141 seconds
- **Average Speedup over 3 runs:** 11.644054
Completed make for dataset EXTRALARGE_DATASET


---

### IN THE TABLE


| **Dataset**      | **Average Sequential Time (s)** | **Average Parallel Time (s)** | **Average Speedup** |
|-------------------|---------------------------------|--------------------------------|--------------------|
| **SMALL DATASET** | 0.016330                       | 0.066347                       | 0.246130          |
| **Standard DATASET** | 38.415722                      | 1.841017                      | 20.866573          |
| **Large Dataset** | 74.785027                      | 10.927332                      | 6.843851          |
| **Extra Large Dataset** | 738.156433                    | 82.591141                     | 8.937477          |

---

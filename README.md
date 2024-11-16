# GROUP 9 MEMBERS:
- Alessandro Appio
- Emanuele Ugolotti
- Simone Candiani

> ## Assigned application
>
> `OpenMP/linear-algebra/kernels/symm`

### Implemented functionality
 - [Creation of a function to compare the sequential and parallelized matrix](https://github.com/alleappio/hpc_assignment_1/blob/7f3a465ae524391a9890ad755887fb72a265fcad/symm/symm.c#L144C1-L159C2)
 - [Creation of a for loop to measure and calculate the average execution time](https://github.com/alleappio/hpc_assignment_1/blob/b766f69a4e8b23e6035146eb8309045773cef766/symm/symm.c#L167C2-L169C45)
 - [Automated time calculation](https://github.com/alleappio/hpc_assignment_1/blob/7f3a465ae524391a9890ad755887fb72a265fcad/symm/symm.c#L234C1-L242C71)
 - Created OpenMP parallelizations:
   - For Kernel:
     * [Creation of the parallelized section](https://github.com/alleappio/hpc_assignment_1/blob/b766f69a4e8b23e6035146eb8309045773cef766/symm/symm.c#L124C1-L142C1)
     * [Parallel for loop with collapse, reduction and schedule](https://github.com/alleappio/hpc_assignment_1/blob/b766f69a4e8b23e6035146eb8309045773cef766/symm/symm.c#L127C4-L127C84)
   - For Init:
     * [First loop parallelized](https://github.com/alleappio/hpc_assignment_1/blob/b766f69a4e8b23e6035146eb8309045773cef766/symm/symm.c#L57C5-L57C49)
     * [Second loop parallelized](https://github.com/alleappio/hpc_assignment_1/blob/b766f69a4e8b23e6035146eb8309045773cef766/symm/symm.c#L65)

- [Creation an automatic benchmark for all datasets(excepet EXTRALARGE)](https://github.com/alleappio/hpc_assignment_1/blob/develop_candiani/symm/bench.sh)

# Project Results 
## Average Execution Times and Speedups
To get the results run the bench.sh file.
- To get the extra large result, you need to change the run number in main  
- And run: make EXT_CFLAGS="-DEXTRALARGE_DATASET" clean all run

### Small Dataset
- **Average Sequential Execution Time (3 runs):** 0,029822 seconds
- **Average Parallel Execution Time (3 runs):** 0,016785 seconds
- **Average Speedup (3 runs):** 1,821161

---

### Standard Dataset 
- **Average Sequential Execution Time (3 runs):** 37,747477 seconds
- **Average Parallel Execution Time (3 runs):** 17,310042 seconds
- **Average Speedup (3 runs):** 2,251014

---

### Large Dataset
- **Average Sequential Execution Time (3 runs):** 103.639361 seconds
- **Average Parallel Execution Time (3 runs):** 57.282299 seconds
- **Average Speedup (3 runs):** 1.80927

---

### Extra Large Dataset
- **Average Sequential Execution Time (3 run):** 961,869574 secondi  
- **Average Parallel Execution Time (3 run):** 560,203402 secondi  
- **Average Speedup (3 run):** 1,717000  

---

### IN THE TABLE


| **Dataset**      | **Average Sequential Time (s)** | **Average Parallel Time (s)** | **Average Speedup** |
|-------------------|---------------------------------|--------------------------------|--------------------|
| **SMALL DATASET** | 0,029822                       | 0,016785                       | 1,821161          |
| **Standard DATASET** | 37,747477                      | 17,310042                      | 2,251014          |
| **Large Dataset** | 103.639361                      | 57.282299                      | 1.80927          |
| **Extra Large Dataset** | 961,869574                    | 560,203402                     | 1,717000          |

---




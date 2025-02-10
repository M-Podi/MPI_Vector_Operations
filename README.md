# MPI_Vector_Operations

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Implementation Details](#implementation-details)
- [How to Run](#how-to-run)
- [Output](#output)
- [Acknowledgments](#acknowledgments)

## Introduction
This project implements a parallel computation using MPI (Message Passing Interface) to calculate the following expression:

\[ \text{AVG} = \frac{\sum_{i,j} x_i A_{ij} y_j}{\sum_i x_i y_i} \]

Where:
- \(x\) and \(y\) are vectors of size \(N = 200\).
- \(A\) is an \(N \times N\) matrix.

The vectors \(x\) and \(y\) are stored in `x.dat` and `y.dat` files, and the matrix \(A\) is stored in `mat.dat`.

The computation is distributed across multiple processes in two distinct groups:
1. **Low Group**: Calculates the denominator of the expression.
2. **High Group**: Calculates the numerator of the expression.

## Project Structure
```
|-- main.cpp           # Main source code for the MPI computation
|-- x.dat              # Data file for vector x
|-- y.dat              # Data file for vector y
|-- mat.dat            # Data file for matrix A
|-- result.txt         # Output file for the final result
```

## Requirements
- C++ compiler with MPI support (e.g., `mpic++`)
- MPI library (e.g., OpenMPI)
- `make` utility
  
## Implementation Details
The implementation follows these steps:
1. **Process Grouping**: The processes are split into two groups:
   - Low group (`my_rank < nproc/2`): Calculates the denominator.
   - High group (`my_rank >= nproc/2`): Calculates the numerator.
   
2. **Data Distribution**:
   - Only the process with `my_new_rank = 0` in each group reads the data files.
   - The data is distributed to other processes using collective communication (`MPI_Scatterv`, `MPI_Bcast`).
   
3. **Local Computation**:
   Each process computes its portion of the dot product or matrix-vector multiplication locally.

4. **Result Aggregation**:
   The local results are reduced to a single result using `MPI_Reduce`.
   
5. **Final Result Calculation**:
   The final computation result is written to `result.txt`.

## How to Run
1. **Compile the Code**:
   ```sh
   mpic++ -o mpi_project main.cpp
   ```

2. **Execute the Program**:
   ```sh
   mpirun -np <number_of_processes> ./mpi_project
   ```
   Ensure that the number of processes (`<number_of_processes>`) is an even number.

3. **Input Data Files**:
   Ensure that `x.dat`, `y.dat`, and `mat.dat` are present in the same directory as the executable.

## Output
The output is written to `result.txt` and includes:
- The computed value of `AVG`.
- The total elapsed time for the computation.

Example output:
```
AVG = 1.23456789
Elapsed time: 0.123456 s
```

# Parallel and High Performance Computing @ EPFL
![Demo Animation](./Final%20project/water_drops_animation.gif)
---
Welcome to the repository for the graded projects of the **Parallel and High Performance Computing** course at EPFL, Spring 2025. This repository contains all code, reports, and results for the course assignments.

---

## Table of Contents

- [Overview](#overview)
- [Final Project](#final-project-parallelized-shallow-water-equations-solver-using-mpi-and-cuda)
- [Additional Projects](#additional-projects)

---

## Overview

This repository showcases solutions to various parallel programming and high performance computing problems, including:
- Parallelization using MPI and CUDA
- Performance optimization and profiling
- Theoretical anlysis of run time
- Scalability analysis
- Real-world scientific computing applications

---

## Final Project: Parallelized Shallow Water Equations Solver using MPI and CUDA

**This was the main project for the course.**

### Overview
- **Objective**: Parallelize the solution of shallow water equations using MPI and CUDA to enhance computational performance.
- **Theoretical Analysis**: Computational cost analysis of grid initialization and time-step calculations.

### MPI Parallelization
- **Domain Subdivision**: Divided the 2D grid into subgrids for each processor to handle computations locally.
- **Communication**: Utilized MPI's Cartesian communicator for efficient neighbor communication and halo updates.
- **Performance Metrics**:
  - **Strong Scaling**: Demonstrated significant speedup with increasing processors, aligning with Amdahl's law initially but deviating due to communication overheads.
  ![](./Final%20project/scaling_images/speedup_strong_scaling.png)
  - **Weak Scaling**: Showed efficiency and speedup trends with constant work per processor, highlighting communication costs as processors increased.
  ![](./Final%20project/scaling_images/efficiency_weak_scaling.png)

### CUDA Parallelization
- **Conceptual Similarity**: Subdivided the grid into blocks for parallel computation within each block.
- **Optimizations**:
  - Parallelized the large majority of functions.
  - Implemented a two-kernel approach for global maximum computation, inspired by NVIDIA's reduction techniques.
- **Performance Results**: Illustrated significant time reduction with increasing threads per block, eventually plateauing due to serial bottlenecks.
![](./Final%20project/scaling_images/cuda_scaling.png)

### Results and Analysis
- **Speedup and Efficiency**: Graphical representation of speedup and efficiency for both strong and weak scaling under MPI, and performance gains with CUDA.
- **Bottlenecks**: Identified memory and communication bottlenecks as limiting factors in scalability and performance.

### Conclusion
- **Parallel Potential**: Demonstrated substantial performance improvements through parallelization of shallow water equations.
- **Challenges**: Highlighted the impact of communication overhead and serial code sections on overall performance.
- **Future Work**: Suggested further optimizations and advanced techniques to mitigate current bottlenecks.

---

## Additional Projects
Other than the final project, two other projects were completed with the goal of hands-on learning of MPI and CUDA.

### **Project 1: Parallelized conjugate gradient solver using MPI**
   This project focuses on parallelizing the conjugate gradient method, specifically targeting the matrix vector multiplication function. Using MPI, the matrix indices were divided among processors to dsitribute the workload. Evaluation showed a notable speedup with up to 25 processors, beyond which a plateau was observed. Scaling analysis, applying Amdahl's and Gustafson's laws, revealed that memory and communication bottlenecks limit scalability.

   The full report is avaliable [here](./Project%201/Report_Mattia_Barbiere.pdf) and all the code is localed [here](./Project%201/).


### **Project 2: Parallelized conjugate gradient solver using CUDA**
   This project focuses on transitioning from CBLAS to CUDA to parallelize linear algebra functions on GPUs. The primary goal was to create CUDA kernels for these functions to enhance computational performance. Timing the conjugate gradient function with varying threads per block revealed that performance improved initially but plateaued after 8 threads per block and degraded significantly beyond 256 threads, likely due to memory bottlenecks.

   The full report is avaliable [here](./Project%202/report_Mattia_Barbiere.pdf) and all the code is localed [here](./Project%202/).

---

## Contributor

- [Mattia Barbiere](https://github.com/MattiaBarbiere)

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

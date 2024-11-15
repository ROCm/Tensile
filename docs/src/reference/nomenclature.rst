.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile kernel selection, Tensile solution selection, GEMM, Tensor, ROCm

.. _nomenclature:

============
Nomenclature
============

.. .. list-table:: GEMM data types 
..    :header-rows: 1

..    * - Abbreviation
..      - Description
..      - Bit Size
..    * - HGEMM
..      - Half precision general matrix multiplication
..      - 16-bit
..    * - SGEMM
..      - Single precision general matrix multiplication
..      - 32-bit
..    * - DGEMM
..      - Double precision general matrix multiplication
..      - 64-bit
..    * - CGEMM
..      - Single precision complex general matrix multiplication
..      - 32-bit
..    * - ZGEMM
..      - Double precision complex general matrix multiplication
..      - 32-bit

.. .. list-table:: GEMM Operations
..    :header-rows: 1

..    * - Operation
..      - Equation
..    * - N (N: nontranspose)
..      - C[i,j] = ∑[l] A[i,l] * B[l,j]
..    * - NT (T: transpose)
..      - C[i,j] = ∑[l] A[i,l] * B[j,l]
..    * - TN
..      - C[i,j] = ∑[l] A[l,i] * B[l,j]
..    * - TT
..      - C[i,j] = ∑[l] A[l,i] * B[j,l]
..    * - Batched-GEMM
..      - C[i,j,k] = ∑[l] A[i,l,k] * B[l,j,k]
..    * - 2D Summation
..      - C[i,j] = ∑[k,l] A[i,k,l] * B[j,l,k]
..    * - GEMM with 3 Batched Indices
..      - C[i,j,k,l,m] = ∑[n] A[i,k,m,l,n] * B[j,k,l,n,m]
..    * - 4 Free Indices
..      - C[i,j,k,l,m] = ∑[n,o] A[i,k,m,o,n] * B[j,m,l,n,o]

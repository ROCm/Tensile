.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile kernel selection, Tensile solution selection, GEMM, Tensor, ROCm

.. _nomenclature:

************
Nomenclature
************

.. list-table:: GEMM data types 
   :header-rows: 1

   * - Abbreviation
     - Description
     - Bit Size
   * - HGEMM
     - Half precision general matrix multiplication
     - 16-bit
   * - SGEMM
     - Single precision general matrix multiplication
     - 32-bit
   * - DGEMM
     - Double precision general matrix multiplication
     - 64-bit
   * - CGEMM
     - Single precision complex general matrix multiplication
     - 32-bit
   * - ZGEMM
     - Double precision complex general matrix multiplication
     - 32-bit

.. list-table:: GEMM Operations
   :header-rows: 1

   * - Operation
     - Equation
   * - N (N: nontranspose)
     - C i,j = ∑[l] A[i,l] * B[l,j]
   * - NT (T: transpose)
     - C[i,j] = ∑[l] A[i,l] * B[j,l]
   * - TN
     - C[i,j] = ∑[l] A[l,i] * B[l,j]
   * - TT
     - C[i,j] = ∑[l] A[l,i] * B[j,l]
   * - Batched-GEMM
     - C[i,j,k] = ∑[l] A[i,l,k] * B[l,j,k]
   * - 2D Summation
     - C[i,j] = ∑[k,l] A[i,k,l] * B[j,l,k]
   * - GEMM with 3 Batched Indices
     - C[i,j,k,l,m] = ∑[n] A[i,k,m,l,n] * B[j,k,l,n,m]
   * - 4 Free Indices
     - C[i,j,k,l,m] = ∑[n,o] A[i,k,m,o,n] * B[j,m,l,n,o]

Indices
=======

The indices describe the dimensionality of the problem to be solved. A GEMM operation takes two 2-dimensional matrices as input,
adding up to four input dimensions and contracts them along one dimension, which cancels out two dimensions, leading to a 2-dimensional result.
When an index shows up in multiple tensors, those tensors must be the same size along with the dimension, however, they can have different strides.

There are three categories of indices or dimensions used in the problems supported by Tensile: free, batch and bound.
**Tensile only supports problems with at least one pair of free indices.**

Free indices
------------

Free indices are the paired indices of tensor C with one pair in tensor A and another pair in tensor B. i,j,k, and l are the four free indices of tensor C where indices i and k are present in tensor A while indices j and l are present in tensor B.

Batch indices
-------------

Batch indices are the indices of tensor C that are present in both tensor A and tensor B.
The difference between the GEMM example and the batched-GEMM example is the additional index.
In the batched-GEMM example, the index k is the batch index, which batches together multiple independent GEMMs.

Bound indices
-------------

The bound indices are also known as summation indices. These indices are not present in tensor C but in the summation symbol (Sum[k]) and in tensors A and B. The inner products (pairwise multiply then sum) are performed along these indices.

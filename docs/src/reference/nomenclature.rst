.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile kernel selection, Tensile solution selection, GEMM, Tensor, ROCm

.. _nomenclature:

************
Nomenclature
************

General Matrix Multiplication
=============================

General matrix multiplication (GEMM) is a level 3 BLAS operation that computes the product of two matrices, formalized by the equation,

.. math::
   C = \alpha A B + \beta C

where :math:`\alpha` and :math:`\beta` are scalars and :math:`A` and :math:`B` are optionally transposed input matrices. 

.. list-table:: GEMM data types. 
   :header-rows: 1
   :widths: 30, 50, 20

   * - Abbreviation
     - Description
     - Precision
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
     - 64-bit

.. list-table:: GEMM operations; N (non-transpose) and T (transpose) represent the transpose state of the input matrices.
   :header-rows: 1
   :widths: 30, 70

   * - Operation
     - Equation
   * - NN
     - :math:`C_{i,j} = \sum_l A_{i,l} B_{l,j}`
   * - NT
     - :math:`C_{i,j} = \sum_l A_{i,l} B_{j,l}`
   * - TN
     - :math:`C_{i,j} = \sum_l A_{l,i} B_{l,j}`
   * - TT
     - :math:`C_{i,j} = \sum_l A_{l,i} B_{j,l}`
   * - Batched-GEMM
     - :math:`C_{i,j,k} = \sum_l A_{i,l,k} B_{l,j,k}`
   * - 2D Summation
     - :math:`C_{i,j} = \sum_{k,l} A_{i,k,l} B_{j,l,k}`
   * - 3 Batched Indices
     - :math:`C_{i,j,k,l,m} = \sum_n A_{i,k,m,l,n} B_{j,k,l,n,m}`
   * - 4 Free Indices
     - :math:`C_{i,j,k,l,m} = \sum_{n,o} A_{i,k,m,o,n} B_{j,m,l,n,o}`


Indices
=======

The indices describe the dimensionality of the problem to be solved. A GEMM operation takes two 2-dimensional matrices as input,
adding up to four input dimensions and contracts them along one dimension, which cancels out two dimensions, leading to a 2-dimensional result.
When an index shows up in multiple tensors, those tensors must be the same size along with the dimension, however, they can have different strides.

There are three categories of indices or dimensions used in the problems supported by Tensile: free, batch, and bound.
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

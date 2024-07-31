.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation

.. _concepts:

********************************************************************
Concepts
********************************************************************

Tensile is written in both Python (for library/kernel generation) and C++ (for client headers and library tests)---it is a vital project to the ROCm ecosystem, providing optimized kernels for downstream libraries such as https://github.com/rocm/rocBLAS.

The parts of Tensile that are written in Python consist of applications that, collectively, are responsible for generating optimized assembly kernels and generating library objects to access these kernels from client code.

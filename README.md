# Cobalt

## Description
A core math library back end to domain-specific libraries for multi-dimensional convolutions, tensor contractions, matrix-matrix multiplication, higher-order inner-products and anything else that multiplies two objects together on a gpu using any desired language (OpenCL, HSA, C++ lambda functions).

## Motivation:
The following three multi-dimensional mathematical operations

1) Convolutions (1D, 2D, 3D)
2) Inner-Products (including GEMM)
3) Tensor contractions

not only share the same overall behavior of

1) multiply groups of elements of A with corresponding groups of elements of B
2) sum the group
3) write group's sum to particular location in C

but they all share the same core/inner-most-loop for achieving peak floating-point throughput on GPUs:

1)	load elements of A from global -> local memory
2)	load elements of B from global -> local memory
3)	load m elements of A into registers
4)	load n elements of B into registers
5)	do m\*n multiply-accumulate operations in registers

Cobalt will be a single library back-end, similar to the successful AutoGemm, a "library behind the libraries", to flexibly generate gpu kernels (append domain-specific prefix and suffix to common inner-most-loop) and some necessary host-code to use in domain specific libraries.


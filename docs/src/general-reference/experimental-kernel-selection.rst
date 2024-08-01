.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation

.. _experimental-kernel-selection:

=================================
Experimental kernel selection
=================================

The optimal kernel or solution selection for an arbitrary GEMM or tensor contraction is under active development.
The experimental kernel selection provides an early access to the latest development. These methods are designed to select GPU kernels in an intelligent way to obtain good performance.

Here are the two implemented experimental libraries:

- **Grid experimental kernel selection:** Divides the GEMM space into a grid and assigns the kernels with the highest computational granularity. 
Supported on:
    - MI200
    - FP16
    - NN and NT transposition types
- **Decision trees experimental kernel selection:** Selects high-performance kernels using a pretrained decision tree.
Supported on:
    - MI200
    - FP16
    - NN, NT and TN transposition types

To control experimental kernel selection and select the library, use :ref:`TENSILE_EXPERIMENTAL_SELECTION <tensile-experimental-selection>` environment variable. 

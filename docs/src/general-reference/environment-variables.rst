.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation

.. _environment-variables:

********************************************************************
Environment variables
********************************************************************

This document lists the environment variables that enable test or debug features on the Tensile client.

.. list-table:: environment variables
  :header-rows: 1

  * - Envrionment variable
    - Description
    - Values

  * - TENSILE_DB
    - Enables debugging features based on the supplied value.
      TENSILE_DB is a bit field, so options can be set individually or combined. To enable all debug output, set TENSILE_DB=0xFFFF.
    - 0x2 or 0x4 \- Prints extra information about the solution selection process. Indicates if a kernel was an exact match, or if a sequence of kernels is considered for a closest match.
      0x8 \- Prints extra information about the hardware selection process.
      0x10 \- Prints debug-level information about predicate evaluations.
      0x20 \- Prints a list of loaded or missing code object libraries.
      0x40 \- Prints kernel launch arguments, including the kernel name, work group size and count, and all arguments passed.
      0x80 \- Prints size of allocated tensors.
      0x100 \- Prints debug information about convolution reference calculations.
      0x200 \- Prints more detailed information about convolution reference calculations.
      0x1000 \- Prints information about the loading of embedded, YAML, or MessagePack libraries.
      0x4000 \- Prints solution lookup efficiency.
      0x8000 \- Prints the name of selected kernels.
      0x80000 \- Prints the name of selected kernels and number of common kernel parameters such as Matrix Instruction, MacroTile, ThreadTile, DepthU, and so on.
  
  
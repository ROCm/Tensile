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
      <br>0x8 \- Prints extra information about the hardware selection process.</br>
      <br>0x10 \- Prints debug-level information about predicate evaluations.</br>
      <br>0x20 \- Prints a list of loaded or missing code object libraries.</br>
      <br>0x40 \- Prints kernel launch arguments, including the kernel name, work group size and count, and all arguments passed.</br>
      <br>0x80 \- Prints size of allocated tensors.</br>
      <br>0x100 \- Prints debug information about convolution reference calculations.</br>
      <br>0x200 \- Prints more detailed information about convolution reference calculations.</br>
      <br>0x1000 \- Prints information about the loading of embedded, YAML, or MessagePack libraries.</br>
      <br>0x4000 \- Prints solution lookup efficiency.</br>
      <br>0x8000 \- Prints the name of selected kernels.</br>
      <br>0x80000 \- Prints the name of selected kernels and number of common kernel parameters such as Matrix Instruction, MacroTile, ThreadTile, DepthU, and so on.</br>
  
  * - TENSILE_DB2
    - Enables extended debugging features based on the supplied value. When enabled, Tensile skips launching kernels for debug purposes, but continues to perform other steps such as kernel selection,
      data allocation, and initialization.
    - 1 \- Enable
      <br>2 \- Disable</br>
    
  * - TENSILE_NAIVE_SEARCH
    - Performs a naive search for matching kernels instead of the standard optimized search.
    - 1 \- Enable
      <br>2 \- Disable</br>

  * - TENSILE_TAM_SELECTION_ENABLE
    - Enables tile aware solution selection.
    - 1 \- Enable
      <br>2 \- Disable</br>

  * - TENSILE_SOLUTION_INDEX
    - Prints the index of the selected solution.
    - 1 \- Enable
      <br>2 \- Disable</br>
    
  * - TENSILE_METRIC
    - Overrides the default distance matrix for solution selection with the supplied value.
    - "Euclidean"
      <br>"JSD"</br>
      <br>"Manhattan"</br>
      <br>"Ratio"</br>
      <br>"Random"</br>

  * - TENSILE_EXPERIMENTAL_SELECTION
    - Allows experimental kernel selection for GEMM.
    - 0 or unset \- Default kernel selection
    - 1 \- Grid experimental kernel selection</br>
      <br>2 \- Decision trees experimental kernel selection</br>

  * - TENSILE_PROFILE
    - When enabled, all functions decorated with ``@profile`` are profiled and results are generated as ``.prof`` files.
    - 1, "ON", "TRUE" \- Enable
      <br>Any other value \- Disable</br>  
  
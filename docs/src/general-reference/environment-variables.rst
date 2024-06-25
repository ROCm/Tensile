.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation

.. _environment-variables:

********************************************************************
Environment Variables
********************************************************************


^^^^^^^^^^^^^^^
TENSILE_PROFILE
^^^^^^^^^^^^^^^

Enables profiling when set to "ON", "TRUE", or "1". When enabled, all functions decorated with ``@profile`` will be profiled and results will be generated as .prof files. 

*Example*

.. code-block::
   :name: TENSILE_PROFILE Example

   TENSILE_PROFILE=ON Tensile/bin/Tensile ...

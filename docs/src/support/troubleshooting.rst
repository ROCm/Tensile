.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile developers guide, Tensile contributors guide, Tensile programmers guide, GEMM, Tensor
.. highlight:: none

.. _troubleshooting:

********************************************************************
Troubleshooting
********************************************************************

This topic provides information for programmers and users on how to resolve common issues in Tensile.

============================
Missing toolchain components
============================

.. code-block::

    FileNotFoundError: `amdclang++` either not found or not executable in any search path

In this situation, Tensile cannot locate one or more binaries required fir proper program execution. This includes compilers, assemblers, linkers, and bundlers. 

.. note::

   On Linux, the default installation location is */opt/rocm*. 
   
.. note::

   On Windows, the default installation location is *C:\\Program Files\\AMD\\ROCm\\X.Y*, where *X.Y* identify the major and minor version of the current ROCm installation.
   When the HIP SDK is installed on Windows, the variable ``HIP_PATH`` is set to the installation location of the HIP SDK.

There are two possible reasons for this:

1. ROCm is not installed on the system. To resolve this issue, install ROCm by following the instructions at :ref:`install-rocm`.
2. ROCm is installed, but in a non-default location and the binaries cannot be found in the system PATH.
   In this case, add the installation location to the PATH. For example, on Linux use ``export PATH=$PATH:/<path_to_rocm>/bin`` or on Windows PowerShell use ``$env:PATH += ";<path_to_rocm>\bin"``.

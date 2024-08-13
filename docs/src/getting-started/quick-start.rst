.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile installation, Tensile quick start, GEMM, Tensor, ROCm, API, Documentation

.. _quick-start:

********************************************************************
Quick start
********************************************************************

.. important::

   Ensure you have followed the steps in the :ref:`installation` guide.

To run a benchmark, pass a tuning config to the ``Tensile`` program located in ``Tensile/bin``.

For demonstration purposes, we use the sample tuning file available in ``Tensile/Configs/rocblas_sgemm_example.yaml``.
The sample tuning file allows you to specify the target architecture for which the benchmark will generate a library.
To find your device architecture, run:

.. code-block:: bash

   rocminfo | grep gfx

Specify the device architecture in the sample tuning file using ``ArchitectureName:``. Based on the device architecture, use ``ArchitectureName: "gfx90a"`` or ``ArchitectureName: "gfx1030"``.

You can now run benchmarks using Tensile. From the top-level directory, run:

.. code-block:: bash

   mkdir build && cd build
   ../Tensile/bin/Tensile ../Tensile/Configs/rocblas_sgemm_example.yaml ./

After the benchmark completes, Tensile creates the following directories:

- **0_Build**: Contains a client executable. Use this to launch Tensile from a library viewpoint.
- **1_BenchmarkProblems**: Contains all the problem descriptions and executables generated during benchmarking. Use the ``run.sh`` script to reproduce results.
- **2_BenchmarkData**: Contains the raw performance results of all kernels in CSV and YAML formats.
- **3_LibraryLogic**: Contains the winning (optimal) kernel configurations in YAML format. Typically, rocBLAS takes the YAML files from this folder.
- **4_LibraryClient**: Contains the code objects, kernels, and library code. This is the output of running ``TensileCreateLibrary`` using the ``3_LibraryLogic`` directory as an input.

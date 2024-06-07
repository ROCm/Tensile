.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation

.. _getting-started:

********************************************************************
Getting Started
********************************************************************

Environment requirements
========================

Before working with Tensile, ensure the following dependencies are set up in your enviornment.

Basic dependencies
""""""""""""""""""

1. **Python 3.8+**: Verify your Python version with ``python --version``
2. **CMake 3.13+**: Verify your CMake version with ``cmake --version``

Library dependencies
""""""""""""""""""""

We provide instructions for Ubuntu, for other operating systems use the appropriate package manager.

.. code-block:: bash

  sudo apt install -y            \
    python3-yaml                 \
    libomp-dev                   \
    libboost-program-options-dev \
    libboost-filesystem-dev

Additionally, install *one* of the following, depending on your preferred Tensile backend.

.. code-block:: bash

  sudo apt install libmsgpack-dev    # If using the msgpack backend
  # OR
  sudo apt install libtinfo-dev      # If using the YAML backend

.. note: Tensile uses the LLVM ObjectYAML library for YAML parsing. The LLVM library is bundled with your ROCm installation, but it requires libtinfo to be installed. Alternatively, if you already have LLVM version 6.0 or newer installed, Tensile's build process will find it and libtinfo is not needed.

Python dependencies
"""""""""""""""""""

.. code-block:: bash

  pip3 install joblib                # Tools for pipelining and concurrency

Installation
============

.. code-block:: bash

  git clone -b master https://github.com/ROCm/Tensile.git
  cd Tensile

.. _quick-start-example:

Quick start example
===================

.. important: Ensure you have followed the steps in the **Environment Requirements** and **Installation** sections.

Tensile is a benchmark driven backend library for generating optimal kernels for GEMM-like problems. To run a benchmark, you need to pass a tuning config to the ``Tensile`` program located in *Tensile/bin*.

A sample tuning file has been prepared for this quick start example, it can be found in *Tensile/Configs/rocblas_sgemm_example.yaml*. Note the line at the bottom of this file ``ArchitectureName: "gfx1030"``, this line identifies the target architecture for which the benchmark will generate a library. Verify the architecture of your device by running ``rocminfo | grep gfx``. If you are running on a different architecture, for example, gfx90a, update the line to ``ArchitectureName: "gfx90a"``.

You are now ready to run benchmarks using Tensile! From the top-level directory,

.. code-block:: bash

  mkdir build && cd build
  ../Tensile/bin/Tensile ../Tensile/Configs/rocblas_sgemm_example.yaml ./

After the benchmark completes, Tensile will create the following directories:

- *0_Build* contains a client executable; use this to launch Tensile from a library viewpoint.
- *1_BenchmarkProblems* contains all the problem descriptions and executables generated during benchmarking; use the ``run.sh`` script to reproduce results.
- *2_BenchmarkData* contains the raw performance results for all kernels in CSV and YAML formats.
- *3_LibraryLogic* contains the winning (optimal) kernel configurations in YAML format. Typically, rocBLAS takes the YAML files from this folder.
- *4_LibraryClient* contains the code objects, kernels, and library code. This is the output of running ``TensileCreateLibrary`` using the *3_LibraryLogic* directory as an input


.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation

.. _installation:

********************************************************************
Installation
********************************************************************

You can obtain Tensile as part of ROCm installation or choose to install Tensile from source. This document discusses both the methods to obtain Tensile.

Install ROCm
============

1. Install ROCm for your platform. For installation instructions, refer to `Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_ or Windows `Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html>` installation guide.

   After the installation is complete, you can find the binaries and libraries in ``/opt/rocm``. Installing ROCm provides compilers such as ``amdclang++``, and other useful tools including ``rocminfo`` and ``rocprofv2``.

   .. tip::

      If using Bash, we recommend you to set ``PATH=/opt/rocm/bin/:$PATH`` in your ``~/.bashrc`` and refresh your shell, using ``source ~/.bashrc``.
      Alternatively, export the path for your current shell session only, using ``export PATH=/opt/rocm/bin/:$PATH``.

Install OS dependencies
=======================

Here are the steps to install dependencies for Ubuntu distributions. For other distributions, use the appropriate package manager.

.. note::
   These steps might require elevated privileges.

1. Install dependencies:

.. code-block::

    apt-get install libyaml python3-yaml \
        libomp-dev libboost-program-options-dev libboost-filesystem-dev

2. Install one of the following, depending on your preferred Tensile data format. If both are installed, ``msgpack`` is preferred:

.. code-block::

   apt-get install libmsgpack-dev    # If using the msgpack backend

   # OR

   apt-get install libtinfo-dev      # If using the YAML backend

3. Install build tools. For additional installation methods for the latest versions of CMake, see the `CMake installation <https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html>`_ page.

   .. code-block::

      apt-get install build-essential cmake

4. Verify the versions of installed tools against the following table:

.. table:: C++ build dependencies
   :widths: grid

   ========== =======
   Dependency Version
   ========== =======
   amdclang++ 17.0+
   Make       4.2+
   CMake      3.16+
   ========== =======

Install Tensile from source
============================

To install Tensile from source, it is recommended to create a virtual environment first:

.. code-block:: bash

  python3 -m venv .venv
  source .venv/bin/activate

Then, you can install Tensile using pip or git.

Option 1: Install with pip
---------------------------

.. code-block:: bash

  pip3 install git+https://github.com/ROCmSoftwarePlatform/Tensile.git@develop


Option 2: Install with git
----------------------------

.. code-block:: bash

  git clone git@github.com:ROCm/Tensile.git && cd Tensile
  pip3 install .

You can now run Tensile's Python applications—see :ref:`quick-start`.


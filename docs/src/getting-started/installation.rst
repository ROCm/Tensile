.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation

.. _installation:

********************************************************************
Installation
********************************************************************

Install ROCm
============

1. Install ROCm for your platform (`Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_ or `Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html>`_). 
   
   After the installation is complete, binaries and libraries can be found at */opt/rocm*. ROCm comes packaged with compilers such as **amdclang++**, and other useful tools including **rocminfo** and **rocprofv2**.

   .. tip:: 

      If using Bash, we recommend setting ``PATH=/opt/rocm/bin/:$PATH`` in your *~/.bashrc* and refreshing your shell, e.g., ``source ~/.bashrc``. Alternatively, export the path only for your current shell session with ``export PATH=/opt/rocm/bin/:$PATH``.



Install OS dependencies 
=======================

Steps are provided for Ubuntu distributions. For other distributions, please use the appropriate package manager.

.. note:: 
   This step may require elevated privileges

.. code-block:: 

    apt-get install libyaml python3-yaml \
        libomp-dev libboost-program-options-dev libboost-filesystem-dev

Install *one* of the following, depending on your preferred Tensile data format. If both are installed, *msgpack* is preferred,

   .. code-block::

      apt-get install libmsgpack-dev    # If using the msgpack backend
      # OR
      apt-get install libtinfo-dev      # If using the YAML backend

Install build tools. Additional installation methods for the latest versions for CMake can be found `here <https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html>`_.

   .. code-block::

      apt-get install build-essential cmake


Verify the versions of installed tools against the following table,

   .. table:: C++ build dependencies
      :widths: grid

      ========== =======
      Dependency Version
      ========== =======
      amdclang++ 17.0+  
      Make       4.2+   
      CMake      3.16+  
      ========== =======

Install Tensile
===============

However you choose to install Tensile, we recommend that you start by creating a virtual environment.

.. code-block:: bash

  python3 -m venv .venv
  source .venv/bin/activate


Option 1: Install with pip
==========================

.. code-block:: bash

  pip3 install git+https://github.com/ROCmSoftwarePlatform/Tensile.git@develop


Option 2: Install with git
==========================

.. code-block:: bash

  git clone git@github.com:ROCm/Tensile.git && cd Tensile
  pip3 install .

You can now run Tensile's Python applications—see :ref:`quick-start`.


.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation
.. highlight:: none

.. _developer-guide:

********************************************************************
Developer Guide
********************************************************************

.. _development-environment:

=======================
Development environment
=======================

ROCm is a base requirement for contributing to Tensile. To begin, ensure that ROCm is supported on your platform by reviewing the installation details on the `ROCm documentation <https://rocm.docs.amd.com/>`_ site.

.. note:: 
   Environment setup steps are provided for Ubuntu/Debian platforms. For other operating systems, use the appropriate package manager, or your preferred installation method.


.. table:: Python dependencies
   :widths: grid

   =========== =======
   Dependency  Version
   =========== =======
   Python      3.8+
   Tox         4.0+   
   Joblib      1.4+   
   PyYAML      6.0+   
   MsgPack     1.0+
   =========== =======

.. table:: C++ & build dependencies
   :widths: grid

   ========== =======
   Dependency Version
   ========== =======
   amdclang++ 17.0+  
   Make       4.2+   
   CMake      3.16+  
   ========== =======

------------------------------
Setting up Python dependencies
------------------------------

1. Install OS dependencies

.. code-block:: 

   apt-get install libyaml python3-yaml libomp-dev libboost-program-options-dev libboost-filesystem-dev

2. Install *one* of the following, depending on your preferred Tensile data format

.. code-block::

  apt-get install libmsgpack-dev    # If using the msgpack backend
  # OR
  apt-get install libtinfo-dev      # If using the YAML backend

3. Setup a virtual environment

.. code-block::

   python3 -m venv .venv
   source .venv/bin/activate

4. Install Python dependencies

.. code-block::

   pip3 install -r requirements.txt

---------------------------
Setting up C++ dependencies
---------------------------

1. Install ROCm for your platform (`Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_ or `Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html>`_). After the installation is complete, binaries and libraries can be found at */opt/rocm*---we recommend adding ``PATH=/opt/rocm/bin/:$PATH`` to your *.bashrc* or *.zshrc*. ROCm comes packaged with compilers such as **amdclang++**, and other useful tools including **rocminfo** and **rocprofv2**.

2. Install build tools

.. code-block::

   apt-get install build-essential

3. Install CMake (additional installation methods for the latest versions can be found `here <https://cliutils.gitlab.io/modern-cmake/chapters/intro/installing.html>`_).

.. code-block::

   apt-get install cmake

--------------------
Developing in Docker
--------------------

ROCm development images are available on `Docker Hub <https://hub.docker.com/search?q=rocm%2Fdev>`_ for a variety of OS/ROCm versions. See `Docker images in the ROCm ecosystem <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html#docker-images-in-the-rocm-ecosystem>`_ for more details.


=======
Testing
=======

Tensile uses `pytest <https://docs.pytest.org/>`_ to work with library/kernel tests. In particular, the project makes use of `markers <https://docs.pytest.org/en/stable/how-to/mark.html>`_ to filter which tests are run, as well as for general testing abstraction. Important markers include *pre_checkin*, *extended*, *integration*, and *unit*---refer to `pytest.ini <https://github.com/ROCm/Tensile/blob/develop/pytest.ini>`_ for all supported markers.

In general, a test can be run via the tox **ci** environment by passing the desired test marker with ``-m <MARKER>``,

.. code-block::

   tox run -e ci -- -m <pre_checkin|extended|integration|unit>

Note that ``--`` is used to pass options to the underlying pytest command. By default, ``tox run -e ci`` will run extended tests.

-------------------------------
Unit tests and coverage reports
-------------------------------

Unit tests include all tests located under *Tensile/Tests/unit/*. Although unit tests can be run with the tox **ci** environment, a convenience command is included that adds coverage reporting,

.. code-block::

   tox run -e unittest

By default, coverage results will be dumped to the terminal. To generate reports in other formats (e.g. HTML) use,

.. code-block::

   tox run -e unittest -- --cov-report=html

Files and directories excluded from coverage reporting are itemized in `.coveragerc <https://github.com/ROCm/Tensile/blob/develop/.coveragerc>`_.

------------------
Host library tests
------------------

Host library tests ensure that generated libraries remain operational when being called from client code, e.g., other libraries or applications. These tests are built on `gtest <https://github.com/google/googletest>`_; to run them you must first download the submodule,

.. code-block::

   git submodule update --init

Next, build and run all host library tests,

.. code-block::

   mkdir build && cd build
   cmake -DCMAKE_BUILD_TYPE=<Debug|RelWithDebInfo> -DCMAKE_CXX_COMPILER=amdclang++ -DCODE_OBJECT_VERSION=<V3|V2> -DTensile_ROOT=<Path to repo>/Tensile ../HostLibraryTests
   make -j
   ./TensileTests

For advanced usage, like filtering or repeating test cases, see the `gtest documentation <https://github.com/google/googletest/blob/main/docs/advanced.md>`_.


===============
Static analysis
===============

------
Python
------

**Linting** is evaluated with `flake8 <https://flake8.pycqa.org/en/latest/>`_,

.. code-block::

   tox run -e lint
   # OR
   flake8 Tensile

For convenience, all static analysis checks have been collected under the tox label **static** and can be run with a single command,

.. code-block::

   tox run -m static

---
C++
---

**Formatting** is conducted with `clang-format <https://clang.llvm.org/docs/ClangFormatStyleOptions.html>`_. For example, the following command will format all provided files, however, we recommend that you setup your editor to format on save.

.. code-block::

   clang-format -i style=file <files>

Styling rules are configured in `.clang-format <https://github.com/ROCm/Tensile/blob/develop/.clang-format>`_.


=========
Profiling
=========

------
Python
------

Profiling is enabled through the ``@profile`` decorator, and can be imported from the **Tensile.Utilities.Profile** module. Under the hood, the decorator wraps the function in a `cProfile <https://docs.python.org/3/library/profile.html#module-cProfile>`_ context, and generates a .prof file inside the *profiling-results-<date>* directory.

.. note::
   Due to a current limitation with the profiling decorator, nested profiling is not supported, that is, if `func1` calls `func2` in a loop, and both are marked for profiling, the resulting .prof file for `func1` will display incorrect results.

=============
Documentation
=============

Tensile uses https://github.com/ROCm/rocm-docs-core as the documentation engine (which itself wraps Read the Docs and Sphinx). 

You can build the documentation locally with,

.. code-block::

   tox run -e docs

After the documentation is built, the generated HTML files can be found at *docs/_build/html*. 

==========
Versioning
==========

Tensile follows semantic versioning practices, e.g., **major.minor.patch**.

* **Major** increments are conducted if the public API changes, or if either the benchmark or library configuration (YAML) files change format in a non-backwards-compatible manner.
* **Minor** increments are conducted when new kernel, solution, or benchmarking features are introduced in a backwards-compatible manner.
* **Patch** increments are conducted for bug fixes or minor improvements.

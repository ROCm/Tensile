********************************************************************
Contributor guide
********************************************************************

Welcome to the Tensile project and thank you for thinking about contributing. This guide is intended to help newcomers get acquanited with the the development process, and to serve as a reference for more seasoned Tensile developers. If you haven't already, please review :ref:`getting-started` for an introduction to the project.

====================
Concepts and tooling
====================

Tensile is written in both Python (for library/kernel generation) and C++ (for client headers and library tests)---it is a vital project to the ROCm ecosystem, providing optimized kernels for downstream libraries such as https://github.com/rocm/rocBLAS.

-----------------------------
Library and kernel generation
-----------------------------

The parts of Tensile that are written in Python consist of applications that, collectively, are responsible for generating optimized assembly kernels and generating library objects to access these kernels from client code.

Tensile uses `tox <https://tox.wiki/en/4.15.1/index.html>`_ to standardize workflows relating to these Python applications, including testing, building, and static analysis.

=======================
Development environment
=======================

ROCm is a base requirement for contributing to Tensile. To begin, ensure that ROCm is supported on your platform by following the installation guides on the 
`ROCm documentation <https://rocm.docs.amd.com/>`_ site for 
`Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html>`_ and 
`Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/index.html>`_.

.. table:: C++ dependencies
   :widths: grid

   ========== ======= ============
   Dependency Version Installation
   ========== ======= ============
   amdclang++ 17.0+   Installed via ROCm
   Make       4.2+    See :ref:`setting-up-cpp-dependencies`
   CMake      3.16+   See :ref:`setting-up-cpp-dependencies`
   ========== ======= ============

.. table:: Python dependencies
   :widths: grid

   =========== ======= ============
   Dependency  Version Installation
   =========== ======= ============
   Tox         4.0+    See :ref:`setting-up-python-dependencies`
   Joblib      1.4+    See :ref:`setting-up-python-dependencies`
   PyYAML      6.0+    See :ref:`setting-up-python-dependencies`
   libyaml-dev XXXX    ``apt-get install libyaml-dev``
   =========== ======= ============


.. _setting-up-python-dependencies:

------------------------------
Setting up Python dependencies
------------------------------


.. _setting-up-cpp-dependencies:

---------------------------
Setting up C++ dependencies
---------------------------


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

Here is an example command to display clang-format errors,

.. code-block::

   sh -c '$CLANG_FORMAT_DIR/clang-format -style=file ./Tensile/Source/client/source/HardwareMonitor.cpp | diff - ./Tensile/Source/client/source/HardwareMonitor.cpp'

=========
Profiling
=========

(⚠️ todo after the current profiling PR is merged)

============================
How to submit a Pull Request
============================

Please use the following guidelines when making changes and submitting Pull Requests (PRs):

- Before making a feature branch or testing changes, create your own fork of Tensile---please do not create feature branches directly in the https://github.com/ROCm/Tensile repository.
- All PRs should be made against the https://github.com/ROCm/Tensile (upstream) **develop** branch.
- Before opening a PR:

  1. Ensure that **your develop** branch is up-to-date with the **upstream develop** branch---this may require a rebase or a merge.
  2. Issue ``tox run -m precommit`` and ensure that all checks pass.
  3. If you are updating documentation, issue ``tox run -e docs`` and verify the styling and formatting is what you expect.

- When opening a PR:

  1. Fill in as many details in the PR template as possible.
  2. Title the PR in present imperative tense, e.g., "*Update* kernel parameters" not "Updates" nor "Updated".

------
Labels
------

.. table:: GitHub PR labels

   ============= =======
   Label         Effect
   ============= =======
   ci:profiling  Adds the *profiling* job to the CI pipeline. Profiling artifacts will be saved for 10 days.
   ci:docs-only  Only runs the *docs/readthedocs* job; omits all other pipeline jobs.
   ============= =======

=============
Documentation
=============

Tensile uses https://github.com/ROCm/rocm-docs-core as the documentation engine (which itself wraps Read the Docs and Sphinx). 

You can build the documentation locally with

.. code-block::

   tox run -e docs

After the documentation is built, the generated HTML files can be found at *docs/_build/html*. 

===========================
Conventions and style guide
===========================

-------------------
General conventions
-------------------

1. Always use space indentation (4 spaces)---never commit a tab, e.g., ``\t``.

------------------
Python doc-strings
------------------

Tensile uses `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ to pull in documentation from doc-strings and integrate them into this site. Please use the following guidelines when writing Python functions and modules to maintain quality and consistency.

1. The all parameters and returned values should be identified with type-hints.
2. All functions should have a doc-string describing the parameters, return value, and any exception; however, if the function is small and the implementation is straightforward, a one-line doc-string is sufficient.
3. Do not include types directly in the doc-string, these should be added as type-hints in the function definition.
4. For doc-string styling, use the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.


---
Git
---

When writing commit messages:

1. Use the present imperative tense, e.g., "add" not "adds" nor "added".
2. Don't add a period (``.``) to the end of the message.

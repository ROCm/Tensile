.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation
.. highlight:: none

.. _programmers-guide:

********************************************************************
Programmer's Guide
********************************************************************

.. _development-environment:

=======================
Development environment
=======================

ROCm is a base requirement for contributing to Tensile. To begin, ensure that ROCm is supported on your platform by reviewing the installation details on the `ROCm documentation <https://rocm.docs.amd.com/>`_ site.

Then, follow the steps in the :ref:`installation` guide.

--------------------
Developing in Docker
--------------------

ROCm development images are available on `Docker Hub <https://hub.docker.com/search?q=rocm%2Fdev>`_ for a variety of OS/ROCm versions. See `Docker images in the ROCm ecosystem <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/docker.html#docker-images-in-the-rocm-ecosystem>`_ for more details.

=======
Testing
=======

Tensile uses `pytest <https://docs.pytest.org/>`_ to manage library/kernel tests. In particular, the project makes use of `pytest markers <https://docs.pytest.org/en/stable/how-to/mark.html>`_ to filter which tests are run. Important markers include *pre_checkin*, *extended*, *integration*, and *unit*---refer to `pytest.ini <https://github.com/ROCm/Tensile/blob/develop/pytest.ini>`_ for all supported markers.

In general, a test can be run via the tox **ci** environment by passing the desired test marker with ``-m <MARKER>``,

.. code-block::

   tox run -e ci -- -m {pre_checkin|extended|integration|unit}

Note that ``--`` is used to pass options to the underlying pytest command.

.. note::

   By default ``tox run -e ci`` will run pre-checkin tests.

-------------------------------
Unit tests and coverage reports
-------------------------------

Unit tests include all tests located under *Tensile/Tests/unit/*. A convenience command is included that adds coverage reporting,

.. code-block::

   tox run -e unittest
   # OR for 32 processes
   tox run -e unittest -- -n 32

By default, coverage results will be dumped to the terminal. To generate reports in other formats (e.g. HTML) use,

.. code-block::

   tox run -e unittest -- --cov-report=html

Files and directories excluded from coverage reporting are itemized in `.coveragerc <https://github.com/ROCm/Tensile/blob/develop/.coveragerc>`_.

Although it is encouraged to run unit tests through tox to support consistency, they may also be run directly with pytest for quicker feedback, for example, to debug a run a single test named *test_foo*, the following command may be useful

.. code-block::
   :caption: From *Tensile/Tests/*

   pytest unit/test_TensileCreateLibrary.py -k "test_foo" --capture=no -v


------------------
Host library tests
------------------

Host library tests ensure that generated libraries remain operational when being called from client code, e.g., other libraries or applications. These tests are built on `gtest <https://github.com/google/googletest>`_; to run them you must first download the submodule. From Tensile's project root run,

.. code-block::

   git submodule update --init

Next, you can configure and build the host library tests through tox,

.. code-block::

   tox run -e hostlibtest

.. note::
   Note that this tox command wraps `invoke <https://www.pyinvoke.org/index.html>`_, a tool to manage CLI-invokable tasks. Since tox is, fundamentally, a Python environment manager and test runner, any reusable shell commands that fall outside its purview are managed by invoke (which are then sometimes encapsulated by tox). See `tasks.py <https://github.com/ROCm/Tensile/blob/develop/tasks.py>`_ for more details.

You also can configure, build, and run host library tests directly with `invoke <https://www.pyinvoke.org/index.html>`_,

.. code-block::

   invoke hostlibtest --configure --build --run

An executable *TensileTests* will be generate upon build, which can be used to run the tests.

If you wish to build and run the tests manually, checkout the commands in `tasks.py <https://github.com/ROCm/Tensile/blob/develop/tasks.py>`_. For advanced usage, like filtering or repeating test cases, see the `gtest documentation <https://github.com/google/googletest/blob/main/docs/advanced.md>`_.


===============
Static analysis
===============

------
Python
------

Use the top-level tox label **static** to run all static analysis, **this may reformat your code**, so be sure to commit your changes after running the command,

.. code-block::

   tox run -m static


**Linting** is evaluated with `flake8 <https://flake8.pycqa.org/en/latest/>`_, and **formatting** is conducted with `black <https://black.readthedocs.io/en/stable/>`_ and `isort <https://pycqa.github.io/isort/>`_. To run a check in isolation refer to `tox.ini <https://github.com/ROCm/Tensile/blob/develop/tox.ini>`_, or use one the following commands,

.. code-block::

   tox run -e lint
   tox run -e format     # add `-- --check` to check formatting without applying changes
   tox run -e isort      # add `-- --check` to check imports without applying changes


.. tip::

   To ensure consistent formatting, we recommend setting up your editor to **format on save** using the same formatter settings as in `tox.ini <https://github.com/ROCm/Tensile/blob/develop/tox.ini>`_. Either way, ensuring you commit changes after running  static analysis will reduce wait-times caused by simple CI failures.

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

Tensile uses https://github.com/ROCm/rocm-docs-core as the documentation engine, which itself wraps Read the Docs and Sphinx.

You can build the documentation locally with,

.. code-block::

   tox run -e docs

After the documentation is built, the generated HTML files can be found at *docs/_build/html*.

==========
Versioning
==========

Tensile follows semantic versioning practices, e.g., **major.minor.patch**. See `server.org <https://semver.org/>`_ for more details.

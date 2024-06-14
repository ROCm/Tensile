.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation
.. highlight:: none

.. _contributor-guide:

********************************************************************
Contributor guide
********************************************************************

=================
Project Overview
=================

.. code-block::

   Tensile/
   ├── Tensile/              Source code, tests, and utilities for the Tensile project
   │   └── Tests/                Kernels and application tests
   ├── HostLibraryTests/     Tests for host-side code running the Tensile library
   ├── docker/               A collection of useful Dockerfiles
   ├── docs/                 Documentation source files
   ├── requirements.txt      Python dependencies for running Tensile applications
   ├── pytest.ini            Configuration settings for pytest
   ├── tox.ini               Configuration settings for the Tox environment management tool
   └── setup.py              Package build and installation script


Tensile uses `tox <https://tox.wiki/en/4.15.1/index.html>`_ to standardize various aspects of the project, including packaging, testing, building, and static analysis. Convienience commands for most of the tools that Tensile uses can be found within the `tox.ini <https://github.com/ROCm/Tensile/blob/develop/tox.ini>`_ file, as well as dependencies for these commands. 

=======
Testing
=======

Tensile uses `pytest <https://docs.pytest.org/>`_ to issue and manage kernel tests. In particular, the project makes use of `markers <https://docs.pytest.org/en/stable/how-to/mark.html>`_ to filter which tests are run, as well as for general testing abstraction. Important markers include *pre_checkin*, *extended*, *integration*, and *unit*---refer to `pytest.ini <https://github.com/ROCm/Tensile/blob/develop/pytest.ini>`_ for all supported markers.

In general, a test can be run via the tox **ci** environment by passing the desired test marker with ``-m <MARKER>``,

.. code-block::

   tox run -e ci -- -m <pre_checkin|extended|integration|unit>

Note that ``--`` is used to pass options to the underlying pytest command. By default, ``tox run -e ci`` will run *extended* tests.

-----------------
Pre-checkin Tests
-----------------

(⚠️ Describe what the pre checkin tests actually do)

-----------------
Extended Tests
-----------------

(⚠️ Describe what the extended tests actually do)

-----------------
Integration Tests
-----------------

(⚠️ Describe what the integration tests actually do)

----------
Unit Tests
----------

Unit tests include all tests located under *Tensile/Tests/unit/*. Although unit tests can be run with in the tox **ci** environment, a convenience command is included that adds coverage reporting,

.. code-block::

   tox run -e unittest

Files and directories excluded from coverage reporting are itemized in `.coveragerc <https://github.com/ROCm/Tensile/blob/develop/.coveragerc>`_ (⚠️ in progress)

===============
Static Analysis
===============

------
Python
------

**Formatting** (⚠️ not implemented) is conducted with `black <https://black.readthedocs.io/en/stable/index.html>`_,

.. code-block::

   tox run -e format
   # OR
   black Tensile


**Linting** is evaluated with `flake8 <https://flake8.pycqa.org/en/latest/>`_,

.. code-block::

   tox run -e lint
   # OR
   flake8 Tensile

**Type checking** (⚠️ not implemented yet) is enforced with `mypy <https://www.mypy-lang.org/>`_,

.. code-block::

   tox run -e typecheck
   # OR
   mypy Tensile

For convenience, all static analysis checks have been collected under the tox label **static** and can be run with a single command,

.. code-block::

   tox run -m static

---
C++
---

(⚠️ todo...)

=========
Profiling
=========

(⚠️ todo after the current profiling PR is merged)

===============
Merging Changes
===============

Please use the following guidelines when making changes and submitting Pull Requests (PRs):

- Before making a feature branch or testing changes, create your own fork of Tensile---please do not create feature branches directly in the https://github.com/ROCm/Tensile repository.
- All PRs should be made against the https://github.com/ROCm/Tensile (upstream) **develop** branch.
- Before opening a PR, conducting the following steps:

  1. Ensure that **your develop** branch is up-to-date with the **upstream develop** branch---this may require a rebase or a merge.
  2. Issue ``tox run -m precommit`` and ensure that all checks pass.
  3. If you are updating documentation, issue ``tox run -e docs`` and verify the styling and formatting is what you expect.

- When opening a PR, fill in as many details in the PR template as possible.

------
Labels
------

+---------------+--------------------------------------------------------------------------------------------+
| Label         | Effect                                                                                     |
+===============+============================================================================================+
| ci:profiling  | Adds the *profiling* job to the CI pipeline. Profiling artifacts will be saved for 10 days |
+---------------+--------------------------------------------------------------------------------------------+
| ci:docs-only  | Only runs the *docs/readthedocs* job; omits all other pipeline jobs.                       |
+---------------+--------------------------------------------------------------------------------------------+

=============
Documentation
=============

Tensile uses https://github.com/ROCm/rocm-docs-core as the documentation engine (which itself wraps Read the Docs and Sphinx). 

You can build the documentation locally with

.. code-block::

   tox run -e docs

After the documentation is built, the generated HTML files can be found at *docs/_build/html*. 

------------------
Python Doc-strings
------------------

Tensile uses `autodoc <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_ to pull in documentation from doc-strings and integrate them into this site. Please use the following guidelines when writing Python functions and modules to maintain quality and consistency.

1. The all parameters and returned values should be identified with type-hints.
2. All functions should have a doc-string describing the parameters, return value, and any exception; however, if the function is small and the implementation is straightforward, a one-line doc-string is sufficient.
3. Do not include types directly in the doc-string, these should be added as type-hints in the function definition.
4. For doc-string styling, use the `Google Python Style Guide <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_.

===========
Conventions
===========

1. Always use space indentation (4 spaces)---never commit a tab, e.g., ``\t``.

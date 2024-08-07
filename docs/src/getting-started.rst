.. meta::
  :description: Tensile documentation and API reference
  :keywords: Tensile, GEMM, Tensor, ROCm, API, Documentation

.. _getting-started:

********************************************************************
Getting Started
********************************************************************

Tensile is a tool for creating a benchmark-driven backend library for GEMMs [#gemm]_, GEMM-like problems, *N*-dimensional tensor contractions, and anything else that multiplies two multi-dimensional objects together on AMD GPUs.

  * :ref:`installation`
  * :ref:`quick-start`

Project Overview
================

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




.. rubric:: Footnotes

.. [#gemm] GEMM: General Matrix-Matrix Multiplication

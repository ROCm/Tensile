.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile documentation, GEMM, Tensor, Tensile API

.. _index:

********************************************************************
Tensile documentation
********************************************************************

ensile is a tool for creating a benchmark-driven backend library for General Matrix-Matrix Multiplications (GEMMs), GEMM-like problems such as batched GEMM, N-dimensional tensor contractions, and anything else that multiplies two multidimensional objects together on an AMD GPU.

Tensile is written in Python for library and kernel generation and in C++ for client headers and library tests. It is a vital
project in the ROCm ecosystem, providing optimized kernels for downstream libraries such as :doc:`rocBLAS <rocblas:index>`.

The parts of Tensile that are written in Python consist of applications that are collectively responsible
for generating optimized kernels and library objects to access these kernels from client code.

The code is open source and hosted at https://github.com/ROCm/Tensile

.. grid:: 2
  :gutter: 2

  .. grid-item-card:: Install

    * :ref:`Installation <installation>`

  .. grid-item-card:: Conceptual

    * :ref:`solution-selection-catalogs`

  .. grid-item-card:: Reference

    * :ref:`Environment variables <environment-variables>`
    * :ref:`API reference <api-reference>`
    * :ref:`CLI reference <cli-reference>`

  .. grid-item-card:: Contribution

    * :ref:`Programmer's guide <programmers-guide>`
    * :ref:`Contribution guidelines <contribution-guidelines>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.

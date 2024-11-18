.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile, GEMM, Tensor, Tensile API documentation, Tensile library creation

.. _solution-catalogs:

***************************
Solution selection catalogs
***************************

 Tensile provides a mechanism by which only a subset of the code object files produced during a build are loaded at runtime. This is necessary to avoid the overhead associated with loading code object files including initialization time and the memory footprint of the loaded code object files. However, this introduces the problem of knowing which code object file to load. Solution selection is the process 
by which the **TensileHost** library determines what kernel is preferred and, in turn, 
what code object file contains the selected kernel. This process uses
a hierarchical structure
to efficiently search for kernels based on hardware, problem size, and transpose, among others. 
This is the role of the **solution selection catalog** [1]_---a serialized file that uses a hierarchical schema to organize kernel metadata for efficient lookup at runtime.

.. note::
    Throughout this document we will refer to catalog files with the .yaml extension. In practice
    solution selection catalogs are usually serialized with `MessagePack <https://msgpack.org/>`_, which uses the .dat extension.

Catalog hierarchy
=================

.. figure:: ../../assets/msl.svg
    :alt: Master Solution Library hierarchy
    :align: center

    Solution selection catalog heirarchy for gfx900 and gfx90a

**Level 1: Hardware**

At runtime, only kernels compatible with the device can execute. As such, the top level of the hierarchy involves hardware comparisons using GFX architecture.

**Level 2: Operation**

This layer is a mapping from a GEMM transpose setting, defined using 
Einstein tensor notation (e.g. *Contraction_l_Alik_Bjlk_Cijk_Dijk*), to a list of problem properties (Level 3).

**Level 3: Problem**

This layer matches against specific problem properties such as input and output types, and features like high precision accumulation and stochastic rounding.

**Level 4: Exact solution**

Finally, exact solutions contain fine-grained details about each solution that can be used during solution selection to locate the best kernel and to assert that the requested problem predicates are satisfied. Each kernel will have an index and a performance ranking. During solution selection, the highest ranked kernel from this pool will be selected.


Build modes
===========

Tensile comes equipped with multiple build modes, which affect the way solution selection catalogs are generated.

Mode 1: Merge files
-------------------

When ``--merge-files`` is enabled, one solution catalog is generated for each architecture, named

.. centered:: TensileLibrary_<gfx>.yaml

The catalog contains information about supported GEMM types and 
solution metadata that is used to locate the optimal kernel for a requested GEMM. This pattern
has the drawback that all code object libraries are loaded eagerly,
thereby increasing both the initialization time and memory footprint of the calling application.

**Example**

Say you're building libraries for gfx908 and gfx90a with ``--merge-files``. The build output directory would look like this

.. code-block:: bash

    build/
    └── library/
        ├── Kernels.so-000-gfx1030.hsaco
        ├── Kernels.so-000-gfx1030.hsaco
        ├── Kernels.so-000-gfx1030.hsaco
        ├── Kernels.so-000-gfx900.hsaco
        ├── Kernels.so-000-gfx906.hsaco
        ├── TensileLibrary_gfx1030.co
        ├── TensileLibrary_gfx1030.yaml
        ├── TensileLibrary_gfx900.co
        ├── TensileLibrary_gfx900.yaml
        ├── TensileLibrary_gfx906.co
        └── TensileLibrary_gfx906.yaml


Mode 2: Lazy library loading
----------------------------

If ``--lazy-library-loading`` is enabled, then a "parent" catalog is generated for each architecture, named

.. centered:: TensileLibrary_lazy_<gfx>.yaml

This file, contains a
reference to each of it's "child" catalogs, but doesn't have details about the exact solutions. These settings are instead 
held in the "child" catalogs, which use the naming convention 

.. centered:: TensileLibrary_Type_<precision>_<problem type>_<gfx>.yaml

Here, *precision* is the data type, *problem type* is the GEMM type, including transpose and accumulate settings, and *gfx* is the hardware GFX archiecture.

For example, *TensileLibrary_Type_HH_Contraction_l_Alik_Bjlk_Cijk_Dijk_<gfx>.yaml* identifies a code object library for half precision
contractions on two transpose matrices, otherwise known as HGEMM TT.
In this way, the child catalogs contain the solution metadata, while the parent catalog is responsible for organizing the child catalogs
by hardware, problem type, transpose, precision, and other predicates.
This has the benefit of reducing the memory footprint of the calling application, as code object libraries are compiled separately and loaded only when required.

**Example: Build outputs**

.. code-block:: bash
  :caption: Lazy library loading build outputs for *DD_Contraction_l_Alik_Bjlk_Cijk_Dijk*

  build/
  └── library/
      ├── Kernels.so-000-gfx1030.hsaco
      ├── Kernels.so-000-gfx900.hsaco
      ├── Kernels.so-000-gfx906.hsaco
      ├── TensileLibrary_lazy_gfx1030.yaml                   # [A]
      ├── TensileLibrary_lazy_gfx900.yaml                                    
      ├── TensileLibrary_lazy_gfx906.yaml                                    
      ├...
      ├── TensileLibrary_Type_..._fallback_gfx1030.hsaco
      ├── TensileLibrary_Type_..._fallback_gfx900.hsaco
      ├── TensileLibrary_Type_..._fallback_gfx906.hsaco
      ├── TensileLibrary_Type_..._fallback.yaml              # [B]
      ├── TensileLibrary_Type_..._gfx900.co
      ├── TensileLibrary_Type_..._gfx900.hsaco
      ├── TensileLibrary_Type_..._gfx900.yaml                # [C]
      ├── TensileLibrary_Type_..._gfx906.co
      ├── TensileLibrary_Type_..._gfx906.yaml                # [D]

Line **[A]** shows the parent catalog for gfx1030, the first of the three parent catalogs generated.
Line **[B]** shows a fallback child catalog, which reference each of the archiecture specific fallback kernels
in the associated .hsaco files.
This means that at least some of the parameter/problem type combinations for *DD_Contraction_l_Alik_Bjlk_Cijk_Dijk*
haven't been explicitly tuned for these architectures.
Note that the matching .hsaco files (above **[B]**) are code object libraries for HIP source kernels.
These files are referenced by the fallback catalog.
Line **[C]** shows a child catalog for gfx900 that references both HIP source and assembly source kernels, found in the associated .hsaco and .co files, respectively.
Line **[D]** shows a child catalog for gfx906, similar to the gfx900 catalog. However, notice that there is only one associated
.co file. This means that there are only assembly source kernels in this catalog.

**Example: Parent solution selection catalog**

.. code-block:: yaml
  :caption: build/library/TensileLibrary_lazy_gfx900.yaml

  library:
    rows:                                                    # [A_]
    - library:
        map:
          Contraction_l_Alik_Bjlk_Cijk_Dijk:                 # [B_]
            ...
            rows:                                            # [C_]
            - library: {type: Placeholder, value: TensileLibrary_Type_SS_..._fallback}
              predicate:
                type: And
                value:
                - type: TypesEqual
                  value: [Float, Float, Float, Float]
                - {type: HighPrecisionAccumulate, value: false}
                - {type: F32XdlMathOp, value: Float}
                - {type: StochasticRounding, value: false}
            - ...
            type: Problem
            ...
          Contraction_l_Alik_Bljk_Cijk_Dijk:
            rows:
              - ...
            type: Problem                                    # [_C]
        property: {type: OperationIdentifier}
        type: ProblemMap                                     # [_B]
      predicate: {type: TruePred}
    type: Hardware                                           # [_A]
  solutions: []

Line **[A]** shows the top level of the parent catalog, which contains a single row for each hardware architecture.
Line **[B]** shows the problem map for the operation *Contraction_l_Alik_Bjlk_Cijk_Dijk*.
Line **[C]** shows the problem type and predicates used to match against exact solutions contained in the child catalogs.

--------------------

.. [1] Previously these files were called *master solution libraries* because they contain two top level keys, "solutions" and "library". The term *solution selection catalog* was later adopted to clarify the purpose of this file within the larger context of the Tensile C++ API.
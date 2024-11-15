.. meta::
  :description: Tensile is a tool for creating a benchmark-driven backend library for GEMM
  :keywords: Tensile, GEMM, Tensor, Tensile API documentation, Tensile library creation

.. _solution-catalogs:

********************************************************************
Solution catalogs
********************************************************************

After kernels are compiled and linked into code objects (.co files), we still have the problem of how these kernels are executed at runtime. The naive approach
would be to search through all of the code object libraries until an appropriate kernel is found. A more sophisticated approach is to use a heirarchical structure
that allows calling code to search by hardware, problem size, transpose, and other predicates. This is the role of a **solution catalog** (previously called 
master solution library); it is a YAML file that uses a heirarchical schema to organize kernel metadata for quick lookup at runtime.

At a minimum, during build, one solution catalog is generated for each device architecture provided, named *TensileLibrary_<gfx>.yaml*. 
In this case, the generated solution catalog *TensileLibrary_<gfx>.yaml* contains all information about the supported libraries, as well as references to the
list of solutions and the solution metadata, which is used to locate the optimal kernel for the requested GEMM. This pattern is the original implementation,
and while it is still occasionally used, has the drawback that all libraries need to be loaded eagerly into memory.

If lazy library loading is enabled, then the file is instead called *TensileLibrary_lazy_<gfx>.yaml* and serves as a "parent" catalog, containing a
reference to each of it's "child" catalogs, which use the naming convention *TensileLibrary_Type_<precision>_<problem type>_<gfx>.yaml* (note that there
will also be a code object file generated alongside it with the same name.) 
For example, *TensileLibrary_Type_HH_Contraction_l_Alik_Bjlk_Cijk_Dijk_<gfx>.yaml* identifies a code object library for half precision
contractions on two transpose matrices, otherwise known as HGEMM TT.
In this way, the child catalogs are responsible for holding the actual solution metadata, while the parent catalog is responsible for organizing the child catalogs
by hardware, problem type, transpose, precision, and other predicates.
This has the benefit of reducing the memory footprint of the calling application, as code object libraries are compiled separately and loaded only when required.

.. image:: ../../assets/msl.svg
    :alt: Master Solution Library hierarchy
    :align: center
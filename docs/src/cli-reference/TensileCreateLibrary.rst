
.. _tensilecreatelibrary-cli-reference:

====================
TensileCreateLibrary
====================

Syntax
------

.. code-block::

    TensileCreateLibrary [OPTIONS...] <LOGIC DIRECTORY> <OUTPUT DIRECTORY> <RUNTIME LANGUAGE>

Required Arguments
------------------

When invoking *TensileCreateLibrary*, the following arguments are required.

\<LOGIC DIRECTORY\>
    Absolute path for logic files. These files are generally found in one of two ways: (i) they are
    generated via the `Tensile` program and placed in the build directory under *3_LibraryLogic* (see :ref:`quick-start-example`).
    (ii) they are found within a project that hosts pre-generated logic files, e.g., `rocBLAS <https://github.com/ROCm/rocBLAS/tree/develop/library/src/blas3/Tensile/Logic>`_.
\<OUTPUT DIRECTORY\>
    Absolute or relative path to the output directory where build artifacts are placed.
\<RUNTIME LANGUAGE\>
    One of: {HIP, HSA}

Options
-------

When invoking *TensileCreateLibrary*, one can provide zero or more options.

\-\-architecture=ARCHITECTURE
    Architectures to generate a library for. When specifying multiple options, use quoted, semicolon delimited 
    architectures, e.g., --architecture='gfx908;gfx1012'.
    Supported architectures include: all gfx000 gfx803 gfx900 gfx900:xnack- gfx906 gfx906:xnack+ gfx906:xnack- gfx908 gfx908:xnack+
    gfx908:xnack- gfx90a gfx90a:xnack+ gfx90a:xnack- gfx940 gfx940:xnack+ gfx940:xnack- gfx941 gfx941:xnack+
    gfx941:xnack- gfx942 gfx942:xnack+ gfx942:xnack- gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1034 gfx1035
    gfx1100 gfx1101 gfx1102.
\-\-build-client
    Build Tensile client executable; used for stand alone benchmarking (default).
\-\-client-config 
    Creates best-solution.ini in the output directory for the library and code object files created (default).
\-\-code-object-version={default,V4,V5}
    HSA code-object version.
\-\-cxx-compiler={amdclang++, hipcc} or on Windows {clang++, hipcc}
    C++ compiler used when generating binaries.
\-\-embed-library=EMBEDLIBRARY
    Embed (new) library files into static variables. Specify the name of the library.
\-\-embed-library-key=EMBEDLIBRARYKEY
    Access key for embedding library files.
\-\-generate-manifest-and-exit
    Similar to dry-run option for *make*, will compute the outputs
    of *TensileCreateLibrary* and write the expected outputs to a 
    manifest file but does not exectue the commands to generate the 
    output.
\-\-generate-sources-and-exit
    Skip building source code object files and assembly code object files.
    Output source files only and exit. 
\-\-ignore-asm-cap-cache
    Ignore asm capability cache and derive the asm capabilities at runtime.    
\-\-jobs=CPUTHREADS, \-j CPUTHREADS
    Number of parallel jobs to launch. If this options is set higher than *nproc* the number of parallel 
    jobs will be equal to the number of cores. If the this option is set below 1 (e.g. 0 or -1), the number
    of parallel jobs will be set to the number of cores, up to a maximum of 64. (default = -1).    
\-\-lazy-library-loading
    Loads Tensile libraries when needed instead of upfront.
\-\-library-format={yaml,msgpack}
    Select which library format to use (default = msgpack).
\-\-no-enumerate
    Do not run rocm_agent_enumerator.
\-\-no-merge-files
    Store every solution and kernel in separate file.
\-\-no-short-file-names
    Disables short files names.
\-\-num-merged-files=NUMMERGEDFILES
    Number of files the kernels should be written into.
\-\-merge-files
    Store all solutions in single file (default).
\-\-short-file-names
    On Windows kernel names can get too long. 
    Converts solution and kernel names to serial ids (default).
\-\-separate-architectures
    Separates TensileLibrary file by architecture to reduce the time to load the library file.
    This option writes each architecture into a different TensileLibrary_gfxXXX.dat file.
\-\-verbose=PRINTLEVEL, \-v PRINTLEVEL
    Set printout verbosity level {0, 1, 2}.
\-\-version=VERSION
    Version string to embed into library file.
\-\-write-master-solution-index
    Output master solution index in csv format including number 
    of kernels per architecture post build in csv format.

Examples
--------

No options
^^^^^^^^^^

The following command will invoke *TensileCreateLibrary*
with no options passing the Logic directory containing 
logic files and creates a directory *tensile-output* 
in the directory where the *TensileCreateLibrary* 
command was invoked. The *tensile-output* directory
will contain the artifacts.

.. code-block::

    TensileCreateLibrary /home/myuser/Logic tensile-output HIP

Adding TensileCreateLibrary options 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example illustrates how to add options when
invoking *TensileCreateLibrary*. In some cases, such as ``--separate-architectures``
no arguments are required; whereas, for ``--jobs`` an argument is required.

.. code-block::

    TensileCreateLibrary --separate-architectures --jobs=32 /home/myuser/Logic tensile-output HIP

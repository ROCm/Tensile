
.. _tensilecreatelibrary-cli-reference:

====================
TensileCreateLibrary
====================

Syntax
------

.. code-block::

    TensileCreateLibrary [OPTION...] [LOGIC DIRECTORY] [OUTPUT DIRECTORY] [RUNTIME LANGUAGE]

When invoking *TensileCreateLibrary*, one can provide zero or more options.
The absolute path to Logic files, the (absolute|relative) path to an 
output directory and the runtime language {OCL, HIP, HSA} are required. 

Options
-------

\-\-architecture=ARCHITECTURE
    Supported archs: all gfx000 gfx803 gfx900 gfx900:xnack- gfx906 gfx906:xnack+ gfx906:xnack- gfx908 gfx908:xnack+
    gfx908:xnack- gfx90a gfx90a:xnack+ gfx90a:xnack- gfx940 gfx940:xnack+ gfx940:xnack- gfx941 gfx941:xnack+
    gfx941:xnack- gfx942 gfx942:xnack+ gfx942:xnack- gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1034 gfx1035
    gfx1100 gfx1101 gfx1102.
\-\-build-client
    Build Tensile client executable analogous to rocBLAS used for stand alone benchmarking (default).
\-\-client-config 
    Creates best-solution.ini in the output directory for the library and code object files created (default).
\-\-code-object-version={default,V4,V5}
    HSA code-object version.
\-\-cxx-compiler={amdclang++, hipcc}
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
    Number of parallel jobs to launch.    
\-\-lazy-library-loading
    Loads Tensile libraries when needed instead of upfront.
\-\-library-format={yaml,msgpack}
    Select which library format to use (default: msgpack).
\-\-library-print-debug
    Solutions will print enqueue info when enqueueing a kernel.
\-\-no-enumerate
    Do not run rocm_agent_enumerator.
\-\-no-library-print-debug
    Solutions will not print enqueue info when enqueueing a kernel (default).
\-\-no-merge-files
    Store every solution and kernel in separate file.
\-\-no-short-file-names
    Disables short files names.
\-\-num-merged-files=NUMMERGEDFILES
    Number of files the kernels should be written into.
\-\-merge-files
    Store all solutions in single file (default).
\-\-package-library 
    Enable the packaging of the client library code objects into separate (deprecated).
    architectures as sub-directories within the Tensile/library directory.
\-\-short-file-names
    On windows kernel names can get too long. 
    Converts solution and kernel names to serial ids (default).
\-\-separate-architectures
    Separates TensileLibrary file by architecture to reduce the time to load the library file.
    The separate-architectures option writes each architecture into a different TensileLibrary_gfxXXX.dat 
    file.
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

    TensileCreateLibrary <absolute path to>/Logic tensile-output HIP

Adding TensileCreateLibrary options 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following example illustrates how to add options When
invoking *TensileCreateLibrary*. In some cases, such as ``--seprate-architectures``
no arguments are required; whereas, for ``--jobs`` an argument is required.

.. code-block::

    TensileCreateLibrary --separate-architectures --jobs=32 <absolute path to>Logic tensile-output HIP


.. \-\-cmake-cxx-compiler=CMAKECXXCOMPILER
    This doesn't appear to do much and I would like to consider removing

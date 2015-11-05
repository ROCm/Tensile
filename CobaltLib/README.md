# CobaltLib

CobaltLib is the C++ interface for domain-specific libraries to use Cobalt for multiplication on GPUs.

## How to Build
CobaltLib is a library which writes itself at build-time using CMake.
CMake allows a user or domain-library to select which language back-end to use (OpenCL, HSA); language is specified at compile-time.
CMake runs CobaltGen to generate appropriate C++ source files in the build directory according to configuration/profile information stored in CobaltGen/profiles.

## How to Customize
To write a customized profile for a specific application and a specific device:

1. Run CMake to specify that CobaltLib should be compiled in "log-only" mode; this causes Cobalt to not execute any GPU code, only to keep track of what problems it attempted to solve.
2. Compile CobaltLib in "log-only" mode, and ensure the application links to it.
3. Run the application, using logger-only CobaltLib. This produces a problem.xml file describing what problem were attempted.
4. Run CobaltGen on problem.xml to do exhaustive benchmarking as to which are the best GPU kernels to launch for each problem. This produces solution.xml.
5. Re-run CMake to specifiy CobaltLib should now be compiled in "solution" mode.
6. Compile CobaltLib. During compilation, CMake will run CobaltGen on solutions.xml to produce additional library source files which will be compiled into CobaltLib. Ensure the application links to it.
7. The application can now be run. All the calls to the CobaltLib API will call into customized/optimized GPU code.

# Cobalt

A tool for creating a library back end to domain-specific libraries for matrix-matrix multiplication, N-dimensional tensor contractions, and anything else that multiplies two multi-dimensional objects together on a GPU.

Cobalt automates the process of:

1. Enumerating what "problems" and application seeks to solve.
2. Creating a list of "solution" candidates for each problem.
3. Benchmarking all solutions to determine the fastest.
4. Writing the customized library backend, consisting of:
  1. GPU kernels, written in HIP or OpenCL.
  2. Solution c++ classes, which enqueue 1 or more kernels.
  3. A getSolutionForProblem() function which looks up the fastest solution for a problem.

### Usage -1- CMake

The Cobalt build process has been instrumented with CMake; it comes pre-setup to automate the process, however it also allows for users to override options, which will be explain below.

"Unresolved External Symbol" - On several of the build steps, the first time your try and build a particular target (CobaltBenchmark or CobaltLib), the compiler will complain about an unresolved external symbols; the second time you try to build that target it will build fine. The reason is because for some of the build steps Cobalt's python code generate source files and generates cmake files specifying which generated source files need to be added to the target; this happens the first time you build the target. The second time your build the target, CMake loads in the generated cmake file, then it knows all the generated source files it needs to fully build the target.

The CMake setup for Cobalt includes the following options:
- Cobalt\_BACKEND - "HIP" or "OpenCL\_1.2"
- Cobalt\_ENABLE\_LOGGER - Activate the logger in CobaltLib; this slows down performance (because almost every API call writes to a file) but is helpful if you want extra profiling data.
- Cobalt\_OPTIMIZE\_BETA - when true, it will write specific kernels for beta=0 which is slightly (un-noticeably) faster, but it could double the number of kernels you need.
- Cobalt\_OPTIMIZE\_ALPHA - similar to above, but for when alpha=1.
- Cobalt\_DIR\_PROBLEMS - the directory to which the clients will write their Problem.xml files, and the directory from which the CobaltGen python scripts will read the Problem.xml files (see Usage-2 below). 
- Cobalt\_DIR\_SOLUTIONS - the directory to which the CobaltBenchmark will write its benchmark data, and it's the directory from which the GobaltGen python scripts will read the solution benchmark times (see Usage-4 below).

### Usage -2- Create Problem.xml

The first step is to enumerate what problems you want Cobalt to solve, and write them to a file. The two ways of doing this are

1. Use one of Cobalt clients or create your own dummy application/client whose sole purpose is to create problems.
2. Incorporate Cobalt into your own application (see code example below), link it with CobaltLogger, then run your application; Cobalt will log all the problems your application requested solutions to.

In either case, you'll need to create a CobaltProblem object. The following code snipet creates a CobaltProblem for sgemm\_NT for M=64, N=256, K=1024, looks up the solution and enqueues it.

### Usage -3- Code Example

```c++
/*
 * compiler has defined:
 * Cobalt_BACKEND_OPENCL12=1
 * Cobalt_BACKEND_HIP=0
 */

#include "Cobalt.h"

  CobaltStatus status;

  /* Setup
   * Must be called before any other calls to Cobalt.
   * The input parameter is an xml file name to where Cobalt's logger
   * will record problem, solution and benchmark data.
   */
  std::string logFilePath = "/path/to/log/file.xml";
  status = cobaltSetup( logFilePath.c_str() );
  cobaltStatusCheck( status );

  /* TensorC
   * A tensor consists of a precision and a list of dimension
   * with each dimension having a stride and a size.
   * The dimensions should be ordered from smallest to largest stride;
   * this is important for Cobalt's nomenclature.
   */
  CobaltTensor tensorC = cobaltCreateEmptyTensor();
  tensorC.dataType = cobaltDataTypeSingle;
  tensorC.numDimensions = 2;
  tensorC.dimensions[0].stride = 1;
  tensorC.dimensions[0].size = 64;
  tensorC.dimensions[1].stride = 64;
  tensorC.dimensions[1].size = 256;

  /* TensorA */
  CobaltTensor tensorA = cobaltCreateEmptyTensor();
  tensorA.dataType = cobaltDataTypeSingle;
  tensorA.numDimensions = 2;
  tensorA.dimensions[0].stride = 1;
  tensorA.dimensions[0].size = 64;
  tensorA.dimensions[1].stride = 64;
  tensorA.dimensions[1].size = 1024;

  /* TensorB */
  CobaltTensor tensorB = cobaltCreateEmptyTensor();
  tensorB.dataType = cobaltDataTypeSingle;
  tensorB.numDimensions = 2;
  tensorB.dimensions[0].stride = 1;
  tensorB.dimensions[0].size = 256;
  tensorB.dimensions[1].stride = 256;
  tensorB.dimensions[1].size = 1024;

  /* Operation */
  CobaltOperationType operationType = cobaltOperationTypeContraction;
  CobaltDataType alphaType = cobaltDataTypeSingle;
  CobaltDataType betaType = cobaltDataTypeSingle;
  bool useOffsets = true; // always true, for now

  /* Index Assignments
   * Cobalt automatically assigns integers to indices/dimensions as follows:
   * Indices of tensor C are numbered 0->N-1 (number of dimensions of C).
   * Indices of summation are numbered beginning with N.
   * So, for the equation
   * C[i,j] = Sum(k) A[i,k] * B[j,k] (sgemm_NT)
   * i = dim 0
   * j = dim 1
   * k = dim 2
   * Next, to describe the problem we want to solve (which dimensions
   * get multiplied together, and to which C dimension do they get written),
   * we specify the index assignments of A and B.
   * A[i,k] is represented as {0, 2} // i->0, k->2
   * B[j,k] is represented as {1, 2} // j->1, k->2
   */
  std::vector<unsigned int> indexAssignmentsA(2);
  indexAssignmentsA[0] = 0;
  indexAssignmentsA[1] = 2;
  std::vector<unsigned int> indexAssignmentsB(2);
  indexAssignmentsB[0] = 1;
  indexAssignmentsB[1] = 2;

  /* Device Profile
   * Because the optimal kernel may depend on which device is being targeted,
   * the target device is part of the "problem".
   * The CobaltDeviceProfile is the list of CobaltDevices that you want
   * to run on; current only one device per profile is supported (no multi-gpu).
   * You can create your own CobaltDeviceProfile object or have
   * Cobalt's api enumerate them for you (shown below).
   */
  unsigned int numProfiles;
  status = cobaltEnumerateDeviceProfiles(nullptr, &numProfiles);
  cobaltStatusCheck( status );
  CobaltDeviceProfile *deviceProfiles = new CobaltDeviceProfile[numProfiles];
  cobaltEnumerateDeviceProfiles(deviceProfiles, &numProfiles);
  CobaltDeviceProfile *deviceProfile = deviceProfiles[0];

  /* Problem
   * The "problem" is the culmination of all prior objects;
   * it describes exactly what the user wants to be solved.
   *
   * Performance: this function constructs a c++ object whose constructor
   * creates some internal state, so it takes medium time.
   * It recommended to avoid creating a new problem for every enqueue;
   * try architecting your code to enqueue multiple times for each
   * problem creation / solution lookup.
   */
  CobaltProblem problem;
  status = cobaltCreateProblem(
      &problem,
      tensorC,
      tensorA,
      tensorB,
      &indexAssignmentsA[0], // plain pointer
      &indexAssignmentsB[0], // plain pointer
      operationType,
      alphaType,
      betaType,
      useOffsets,
      deviceProfile);
  cobaltStatusCheck( status );

  /* Validate Problem
   * Since there's so much opportunity for mistakes in setting up the problem,
   * such as if index assigningments were {0, 2} and {0, 1},
   * there is a specific API for verifying that the created problem
   * is a valid one.
   *
   * Performance: this operation takes a bit of time (and more checks
   * may be added later), so it recommended only for initial debugging help,
   * then removed for high-performance.
   */
  status = cobaltValidateProblem( problem );
  cobaltStatusCheck( status );


  /* Solution
   * After creating the problem description, we need Cobalt's backend
   * to lookup and return which solution is the fastest for that problem.
   *
   * Performance: this function evaluates many conditionals,
   * including a string comparison (device name) just to lookup
   * the Solution object. After finding the object, it calls a constructor
   * which prepares every argument for every kernel, including
   * number and size of work-groups.
   * This function takes a relatively long time.
   * It recommended to avoid looking up a solution for every enqueue;
   * try architecting your code to enqueue multiple times for each
   * problem creation / solution lookup.
   */
  CobaltSolution solution;
  status = cobaltGetSolutionForProblem( &solution, problem );
  cobaltStatusCheck( status );

  /* Destroy Problem
   * Problem can be destroyed as soon as getSolution returns since
   * solution object holds its own copy of the problem.
   */
  status = cobaltDestroyProblem( problem );
  cobaltStatusCheck( status );

  /* User Data
   * These are the data and OpenCL objects the user creates.
   */
  float alpha = 1.f;
  float beta = 0.f;
  cl_mem tensorOnDeviceC = nullptr; // user allocates
  cl_mem tensorOnDeviceA = nullptr; // user allocates
  cl_mem tensorOnDeviceB = nullptr; // user allocates
  cl_command_queue queue0 = nullptr; // user allocates
  cl_command_queue queue1 = nullptr; // user allocates
  cl_command_queue queue2 = nullptr; // user allocates
  cl_command_queue queue3 = nullptr; // user allocates

  /* User Data -> Cobalt Data
   * User's data must be stored in Cobalt's containers for the API
   * (this is so we can support multiple backends with one API).
   */
  CobaltTensorData      tensorDataC{ tensorOnDeviceC, 0 };
  CobaltTensorDataConst tensorDataA{ tensorOnDeviceA, 0 };
  CobaltTensorDataConst tensorDataB{ tensorOnDeviceB, 0 };
  CobaltScalarData alpha{ &alpha };
  CobaltScalarData beta{ &beta };
  CobaltControl control = cobaltCreateEmptyControl();
  control.numQueues = 4;
  control.queues[0] = queue0;
  control.queues[1] = queue1;
  control.queues[2] = queue2;
  control.queues[3] = queue3;

  /* Enqueue
   * Enqueues all the kernels belonging to the solution.
   * It is asynchronous / non-blocking, so the user is responsible for
   * waiting for the solution to complete (described below).
   *
   * Performance: very fast! This is the API you want to call the most.
   * For the OpenCL backend, it calls clSetKernelArg() for each
   * kernel argument then clEnqueueNDRangeKernel(); that's it.
   * For the HIP backend, it just calls the enqueue command.
   */
  status = cobaltEnqueueSolution(
    solution,
    tensorDataC,
    tensorDataA,
    tensorDataB,
    alpha,
    beta,
    &control );
  cobaltStatusCheck( status );

  /* Wait
   * The user is responsible for flushing / finishing the queues or waiting
   * on specific event objects, if desired.
   */
  for (unsigned int i = 0; i < control.numQueuesUsed; i++) {
    clFinish( control.queues[i] );
  }

  /* Destroy Solution
   * Deletes memory associated with the solution object, but it keeps
   * the kernels in the global kernel map in case others solutions
   * use them.
   */
  status = cobaltDestroySolution( solution );
  cobaltStatusCheck( status );

  /* Teardown
   * After tearing down, Cobalt cleans up all its own memory
   * (calls clReleaseKernel on the cached kernels), except maybe
   * for a few small global static objects on the stack and get deleted
   * upon exit of the application.
   */
  status = cobaltTeardown();
  cobaltStatusCheck( status );

```

A few notes on Cobalt's behavior during runtime:

For OpenCL only, the kernels are compiled during the first call to cobaltEnqueueSolution(); however, you can call enqueue with the tensorData pointers set to nullptr, and Cobalt will compile the kernels then return, giving users the option to pre-compile kernels when they want.

After a kernel has been compiled, it is stored in a thread-local map, so it never has to be compiled again. Its gets mapped using the key {cl\_context, cl\_device, \*kernelString}, so using the same solution for a different device or context will trigger having to re-compile the kernel.

Cobalt is designed to be thread safe. Any bugs you may find to the contrary, please help us fix them.


### Usage -4- Build & Run CobaltBenchmark

After generating a Problems.xml file, build the target "CobaltBenchmark;" this will cause CobaltGenBenchmark.py to generate the benchmark as follows:

1. CobaltGenBenchmark reads in all the "\*.xml" files in the target directory and reads in all the problems.
2. For each problem, it generates all the solution candidates (using SolutionCandidateGenerator.py) which should be benchmarked; see the section on "how many kernels" to learn about how many will be generated.
3. The script writes out all the kernel files, solution files, other benchmark files (which create problems and solutions for the benchmark) and a cmake file which tells the CobaltBenchmark executable which additional source files it needs.
4. CMake will then build CobaltBenchmark (may have to build twice) with all the generated source files.
5. You will have to run the CobaltBenchmark executable; as it runs, it will write multiple SolutionBenchmarkTime.xml files recording how fast each solution is for respective problems.

Note - If you're benchmarking many problems (which we do have to do for generating a full BLAS backend), CobaltBenchmark may run out of memory; for this reason, you can run "CobaltBenchmark -h" to see commandline parameters which allow you to only run sub-portions of the benchmark at a time.

### Usage -5- Build CobaltLib

Once you have run the benchmark, and have that performance data, you're ready to build the final target CobaltLib which will cause CMake to run CobaltGenBackend.py to generate the library as follows:

1. CobaltGenBenchmark.py reads in all the "\*.xml" files in the target directory and reads in all the problem-solution-pairs (PSPs) which consist of a problem, solution and the benchmark time.
2. The PSPs are stored into a heirarchal data structure to organize data by device, problem description and problem size.
3. The script SolutionSelectionWriter.py writes a heirarchy of solution selection logic which looks like:
  1. getSolution\_Top(problem) - match which device the problem targets
  2. getSolution\_Device( problem ) - determine which ExactMatch (involves every part of a problem description except for the sizes of the tensor dimensions) the problem targets
  3. getSolution\_Device\_ExactMatch( problem ) - determine which solution is best for the problem size.
4. The scripts also write out the kernels and solutions which are actually used by the library backend, and a cmake script which specifies that the library has to be built with all these generated files.
5. The build system builds CobaltLib.lib.


## Included Clients

The Cobalt repository includes clients for using and demonstrating Cobalt.

**GEMM:** The GEMM client is used to generate Problem.xml files to help with building Cobalt for BLAS libraries. It is also able to generate batched GEMMs and GEMMs where the leading stride is greater than 1.

**DNN:** The DNN client is used to generate Problem.xml files for doing convolutions. Cobalt's current support for convolutions isn't a straight forward convolution kernel, but rather a 7-dimensional tensor contraction mapped to a convolution. This client shows how to set that up that complicated problem.

**Simple:** The Simple client is a sandbox for trying out new kernel ideas to include in Cobalt.


## How many kernels?

Cobalt has many different parameters which it can tweak when writing kernels, they are:
- work-group dim0
- work-group dim1
- micro-tile dim0
- micro-tile dim1 
- branch type (branch in kernel or multiple kernels)
- num loads parallel-to-coalesced for tensorA
- load size parallel-to-coalesced for tensorA
- num loads parallel-to-coalesced for tensorB
- load size parallel-to-coalesced for tensorB
- num loads perpendicular-to-coalesced for tensorA
- load size perpendicular-to-coalesced for tensorA
- num loads perpendicular-to-coalesced for tensorB
- load size perpendicular-to-coalesced for tensorB
- unroll of inner loop
- second inner loop with unroll=1 or not
- preprocessor define offsets
- preprocessor define initial strides
- preprocessor define all sizes and strides

The combinatorics of all these options lead to huge number of kernels that Cobalt can generate. Cobalt therefore provides three levels of thoroughness in generating candidate kernels to benchmark: exhaustive, thorough, fast (few).

**Exhaustive:** This mode won't bother generating kernels, it will just count them; it's just for fun. It can generate tens of millions of kernels per problem type. GEMM itself has many tens of problem types and beyond GEMM (higher dimensionality) there are many more problem types.

**Thorough:** This mode is appropriate when working on a new problem type that you have no idea what will be the fastest and you want to be thorough in which kernels you explore. It will generate a few thousand kernels to test.

**Fast:** Once we know what kernel should and should not be fast, we encode that into fast mode. The purpose of fast mode is to generate the fewest numbers of kernels which will definitely include the fastest. This will generate tens to a few hundred kernels for "normal" problem sizes, but more for "unusual" problem sizes.

## Normal vs Unusual Problem Sizes

The Cobalt backend can work in two ways for two categories of problems: normal and unusual.

**Normal:** for a BLAS library we need to support every problem size conceivable, but we don't want to have to benchmark every problem size. Therefore, we benchmark a few standard problem sizes (multiples of 16 and M=N=K) and we have Cobalt's library backend create problem size ranges, i.e., if a problem's size (M\*N\*batch) falls withing this range, then its best solution is that.

**Unusual:** for DNN libraries, there are many problem sizes which can be small or skinny or not a multiple of 16 (or prime numbers). For these "unusual" problem, the Cobalt backend creates a 1:1 mapping of problem -> solution since performance changes non-linearly with change in these problem sizes.

Yes, Cobalt can contain both types in its backend.

## Languages

**API is C89.** The CobaltLIb API (CobaltLib/include/Cobalt.h) is written in C89 so that it will be usable with most software. All of the structs (Except for CobaltProlem and CobaltSolution which are pimpls) are completely allocated on the stack; they all include fixed length arrays for dimensions, devices, queues and so forth.

**Backend is C++11.** The CobaltLib backend (behind the API) is written in C++11 to take full advantage of modern language features. When the createProblem API is called, the input parameters are used to construct a Cobalt::Problem object, whose pointer is stored in the CobaltProlem struct. When the getSolution API is called, the Cobalt::Problem object is passed into the constructor of a Cobal::Solution object, whose pointer is stored in the CobaltSolution struct.

**Generator is Python.** The backend generator, CobaltGen, is a suite of python scripts used to generate part of the C++11 backend.

## Minimum Required Software

**CMake** - CMake 2.8

**Python** - Python 2.7

**Compilers**

- For Cobalt\_BACKEND = OpenCL1.2
  - Visual Studio 14 (2015)
  - GCC 4.8
- For Cobalt\_BACKEND = HIP
  - HIP 1.1.2

## Conributing Code

See [Contributing][] document for requirements on contributing code to this open source project.


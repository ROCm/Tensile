# Cobalt

### Overview

A tool for creating a library back end to domain-specific libraries for matrix-matrix multiplication, N-dimensional tensor contractions, and anything else that multiplies two multi-dimensional objects together on a GPU.

Cobalt automates the process of:

1) Enumerating what "problems" and application seeks to solve.
2) Creating a list of "solution" candidates for each problem.
3) Benchmarking all solutions to determine the fastest.
4) Writing the customized library backend, consisting of:
  1) GPU kernels, written in HIP or OpenCL.
  2) Solution c++ classes, which enqueue 1 or more kernels.
  3) A getSolutionForProblem() function which looks up the fastest solution for a problem.

## Usage

The following are the steps for using Cobalt; they refer to the code example farther down.

### Usage-0: CMake

The Cobalt build process has been instrumented with CMake; it comes pre-setup to automate the process, however it also allows for users to override options, which will be explain below.

"Unresolved External Symbol" - On several of the build steps, the first time your try and build a particular target (CobaltBenchmark or CobaltLib), the compiler will complain about an unresolved external symbols; the second time you try to build that target it will build fine. The reason is because for some of the build steps Cobalt's python code generate source files and generates cmake files specifying which generated source files need to be added to the target; this happens the first time you build the target. The second time your build the target, CMake loads in the generated cmake file, then it knows all the generated source files it needs to fully build the target.

### Usage-1: Create Problem.xml

The first step is to enumerate what problems you want Cobalt to solve, and write them to a file. The two ways of doing this are
1) Use one of Cobalt clients or create your own dummy application/client whose sole purpose is to create problems.
2) Incorporate Cobalt into your own application, link it with CobaltLogger, then run your application; Cobalt will log all the problems your application requested solutions to.

In either case, you'll need to create a CobaltProblem object. The following code snipet creates a CobaltProblem for sgemm\_NT for M=64, N=256, K=1024, looks up the solution and enqueues it.

#### Usage-1.1 Create CobaltProblem Object


### Usage-Code Example
```
/*
 * compiler has defined:
 * Cobalt_BACKEND_OPENCL12=1
 * Cobalt_BACKEND_HIP=0
 */

#include "Cobalt.h"

  CobaltStatus status;

  /* setup */
  std::string logFilePath = "/path/to/log/file.xml";
  status = cobaltSetup( logFilePath.c_str() );

  /* tensorC */
  CobaltTensor tensorC = cobaltCreateEmptyTensor();
  tensorC.dataType = cobaltDataTypeSingle;
  tensorC.numDimensions = 2;
  tensorC.dimensions[0].stride = 1;
  tensorC.dimensions[0].size = 64;
  tensorC.dimensions[1].stride = 64;
  tensorC.dimensions[1].size = 256;

  /* tensorA */
  CobaltTensor tensorA = cobaltCreateEmptyTensor();
  tensorA.dataType = cobaltDataTypeSingle;
  tensorA.numDimensions = 2;
  tensorA.dimensions[0].stride = 1;
  tensorA.dimensions[0].size = 64;
  tensorA.dimensions[1].stride = 64;
  tensorA.dimensions[1].size = 1024;

  /* tensorB */
  CobaltTensor tensorB = cobaltCreateEmptyTensor();
  tensorB.dataType = cobaltDataTypeSingle;
  tensorB.numDimensions = 2;
  tensorB.dimensions[0].stride = 1;
  tensorB.dimensions[0].size = 256;
  tensorB.dimensions[1].stride = 256;
  tensorB.dimensions[1].size = 1024;

  /* operation sgemm_NT: C[i,j] = Sum(k) A[i,k] * B[j,k] */
  CobaltOperationType operationType = cobaltOperationTypeContraction;
  CobaltDataType alphaType = cobaltDataTypeSingle;
  CobaltDataType betaType = cobaltDataTypeSingle;
  bool useOffsets = true;
  std::vector<unsigned int> indexAssignmentsA(2);
  indexAssignmentsA[0] = 0;
  indexAssignmentsA[1] = 2;
  std::vector<unsigned int> indexAssignmentsB(2);
  indexAssignmentsB[0] = 1;
  indexAssignmentsB[1] = 2;

  /* device profile */
  unsigned int numProfiles;
  status = cobaltEnumerateDeviceProfiles(nullptr, &numProfiles);
  CobaltDeviceProfile *deviceProfiles = new CobaltDeviceProfile[numProfiles];
  cobaltEnumerateDeviceProfiles(deviceProfiles, &numProfiles);
  CobaltDeviceProfile *deviceProfile = deviceProfiles[0];

  /* problem */
  status = cobaltCreateProblem(
      problem,
      tensorC,
      tensorA,
      tensorB,
      &indexAssignmentsA[0],
      &indexAssignmentsB[0],
      operationType,
      alphaType,
      betaType,
      useOffsets,
      deviceProfile);

  /* solution */
  CobaltSolution solution;
  status = cobaltGetSolutionForProblem( &solution, problem );

  /* user data */
  float alpha = 1.f;
  float beta = 0.f;
  cl_mem tensorOnDeviceC = nullptr; // user allocates
  cl_mem tensorOnDeviceA = nullptr; // user allocates
  cl_mem tensorOnDeviceB = nullptr; // user allocates
  cl_command_queue queue0 = nullptr; // user allocates
  cl_command_queue queue1 = nullptr; // user allocates
  cl_command_queue queue2 = nullptr; // user allocates
  cl_command_queue queue3 = nullptr; // user allocates

  /* user data -> cobalt data */
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

  /* enqueue */
  status = cobaltEnqueueSolution(
    solution,
    tensorDataC,
    tensorDataA,
    tensorDataB,
    alpha,
    beta,
    &control );

  /* wait */
  for (unsigned int i = 0; i < control.numQueuesUsed; i++) {
    clFinish( control.queues[i] );
  }

  /* teardown */
  status = cobaltTeardown();

```

## Included Clients

The Cobalt repository includes clients for using and demonstrating Cobalt.

GEMM - The GEMM client is used to generate Problem.xml files to help with building Cobalt for BLAS libraries. It is also able to generate batched GEMMs and GEMMs where the leading stride is greater than 1.

DNN - The DNN client is used to generate Problem.xml files for doing convolutions. Cobalt's current support for convolutions isn't a straight forward convolution kernel, but rather a 7-dimensional tensor contraction mapped to a convolution. This client shows how to set that up that complicated problem.

Simple - The Simple client is a sandbox for trying out new kernel ideas to include in Cobalt.


## How many kernels?
fast
thorough
exhaustive

## Limitations
memory
write minimum xml

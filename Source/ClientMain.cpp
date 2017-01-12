/*******************************************************************************
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

/*******************************************************************************
 * Tensile Benchmark
 ******************************************************************************/
#ifdef WIN32
#define _CRTDBG_MAP_ALLOC
#endif

#ifdef _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>
#ifndef DBG_NEW
#define DBG_NEW new ( _NORMAL_BLOCK , __FILE__ , __LINE__ )
#define new DBG_NEW
#endif
_CrtMemState s1;
_CrtMemState s2;
_CrtMemState s3;
#endif

#include "TensileBenchmark.h"
#include "Tensile.h"
#include "Tools.h"
#include "Solution.h"
#include "TensileSolutionCandidates.h"
#include "SolutionTensorContractionCPU.h"
#include "StructOperations.h"
#include "MathTemplates.h"

#include <tuple>
#include <string>
#include <cstring>
#include <cstdio>

#define ULL (unsigned long long)

#if 1
static Tensile::Tensor::FillType tensorFillTypeC = Tensile::Tensor::fillTypeRandom;
static Tensile::Tensor::FillType tensorFillTypeA = Tensile::Tensor::fillTypeRandom;
static Tensile::Tensor::FillType tensorFillTypeB = Tensile::Tensor::fillTypeRandom;
#else
static Tensile::Tensor::FillType tensorFillTypeC = Tensile::Tensor::fillTypeIndex;
static Tensile::Tensor::FillType tensorFillTypeA = Tensile::Tensor::fillTypeIndex;
static Tensile::Tensor::FillType tensorFillTypeB = Tensile::Tensor::fillTypeIndex;
#endif

//#define MAX_PROBLEMS 1
//#define MAX_SOLUTIONS_PER_PROBLEM 1

#define ALPHA 1
#define BETA  0

static size_t overrideExactMatchStartIdx = 0;
static size_t overrideExactMatchEndIdx = 0;
static size_t overrideProblemStartIdx = 0;
static size_t overrideProblemEndIdx = 0;
static size_t overrideSolutionStartIdx = 0;
static size_t overrideSolutionEndIdx = 0;
static unsigned int platformIdx = 0;
static unsigned int deviceIdx = 0;
static bool deviceOverride = false;
static TensileDeviceProfile deviceProfile;

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {
#ifdef _CRTDBG_MAP_ALLOC
  _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
#endif
  srand(4123570);
  // parse commandline options
  parseCommandLineOptions(argc, argv);


  // create TensileControl
  initControls();

  // initialize initial buffer values for validation
  initTensorData();
  
#ifdef MAX_PROBLEMS
  if (problemEndIdx > MAX_PROBLEMS) {
    problemEndIdx = MAX_PROBLEMS;
  }
#endif

  size_t exactMatchStartIdx = overrideExactMatchStartIdx ? overrideExactMatchStartIdx : 0;
  size_t exactMatchEndIdx = overrideExactMatchEndIdx ? overrideExactMatchEndIdx : benchmarkNumExactMatches;
  for (size_t exactMatchIdx = exactMatchStartIdx; exactMatchIdx < exactMatchEndIdx; exactMatchIdx++) {

    std::string exactMatchName = benchmarkExactMatchNames[exactMatchIdx];
    size_t problemStartIdx = overrideProblemStartIdx ? overrideProblemStartIdx : 0;
    size_t problemEndIdx = overrideProblemEndIdx ? overrideProblemEndIdx : benchmarkExactMatchNumProblems[exactMatchIdx];

    // setup TensileLib
    std::string logFilePath = Tensile_DIR_SOLUTIONS;
    logFilePath += "/";
    logFilePath += exactMatchName;
    logFilePath += ".xml";
    tensileSetup(logFilePath.c_str());


    for (size_t problemIdx = problemStartIdx; problemIdx < problemEndIdx; problemIdx++) {

#ifdef _CRTDBG_MAP_ALLOC
      if (problemIdx > 0) {
        _CrtMemCheckpoint(&s2);
        int diff = _CrtMemDifference(&s3, &s1, &s2);
        _CrtMemDumpStatistics(&s3);
        printf("Difference[%llu] = %i\n", ULL problemIdx - 1, diff);
      }
      _CrtMemCheckpoint(&s1);
#endif

      // info about problem
      TensileProblem problem;
      std::vector<Tensile::Solution *> solutionCandidates;
      initializeSolutionCandidates(deviceProfile, &problem, &solutionCandidates, exactMatchIdx, problemIdx);
      bool isFloatC = tensileDataTypeIsFloat(problem->pimpl->getDataTypeC());
      bool isFloatA = tensileDataTypeIsFloat(problem->pimpl->getDataTypeA());
      bool isFloatB = tensileDataTypeIsFloat(problem->pimpl->getDataTypeB());
      bool isFloatAlpha = tensileDataTypeIsFloat(problem->pimpl->getDataTypeAlpha());
      bool isDoubleAlpha = tensileDataTypeIsDouble(problem->pimpl->getDataTypeAlpha());
      bool isFloatBeta = tensileDataTypeIsFloat(problem->pimpl->getDataTypeBeta());
      bool isDoubleBeta = tensileDataTypeIsDouble(problem->pimpl->getDataTypeBeta());
      size_t sizeC = problem->pimpl->tensorC.numBytes();
      size_t sizeA = problem->pimpl->tensorA.numBytes();
      size_t sizeB = problem->pimpl->tensorB.numBytes();
      void *initialDataC = isFloatC ? initialTensorDataFloatC.data : initialTensorDataDoubleC.data;
      void *initialDataA = isFloatA ? initialTensorDataFloatA.data : initialTensorDataDoubleA.data;
      void *initialDataB = isFloatB ? initialTensorDataFloatB.data : initialTensorDataDoubleB.data;
      TensileScalarData alpha{ isFloatAlpha ? static_cast<void*>(alphaFloat)
          : isDoubleAlpha ? static_cast<void*>(alphaDouble) : static_cast<void*>(alphaFloat) };
      TensileScalarData beta{ isFloatBeta ? static_cast<void*>(betaFloat)
          : isDoubleBeta ? static_cast<void*>(betaDouble) : static_cast<void*>(betaFloat) };

      // re-initialize device input buffers
#if Tensile_BACKEND_OCL
      clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataA.data), CL_FALSE, deviceTensorDataA.offset, sizeA, initialDataA, 0, nullptr, nullptr);
      clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataB.data), CL_FALSE, deviceTensorDataB.offset, sizeB, initialDataB, 0, nullptr, nullptr);
#elif Tensile_BACKEND_HIP
      status = hipMemcpy(deviceTensorDataA.data, initialDataA,
        sizeA, hipMemcpyHostToDevice);
      status = hipMemcpy(deviceTensorDataB.data, initialDataB,
        sizeB, hipMemcpyHostToDevice);
      status = hipStreamSynchronize(nullptr);
#endif

      // calculate reference C once for problem
      if (doValidation) {

        // get reference solution
        Tensile::Solution *solutionReference;
        problem->pimpl->deviceProfile = deviceProfileReference;
        std::tie(solutionReference, tensileStatus) = getSolutionCPU(*(problem->pimpl));

        // re-initialize reference buffers
        memcpy(referenceTensorDataC.data, initialDataC, sizeC);
        TensileTensorDataConst referenceTensorDataA{ initialDataA, 0 };
        TensileTensorDataConst referenceTensorDataB{ initialDataB, 0 };

        // enqueue reference solution
        printf("Status: Enqueueing reference for %s ...", problem->pimpl->toString().c_str());
        solutionReference->enqueue(
          referenceTensorDataC,
          referenceTensorDataA,
          referenceTensorDataB,
          alpha,
          beta,
          ctrlValidation);
        ctrl.validate = &referenceTensorDataC;
        printf("done.\n");

      } else {
        printf("Status: Problem[%llu/%llu] %s\n", ULL problemIdx, ULL problemEndIdx, problem->pimpl->toString().c_str());
      }


      size_t solutionStartIdx = 0;
      if (overrideSolutionStartIdx) {
        solutionStartIdx = overrideSolutionStartIdx;
      }
      size_t solutionEndIdx = solutionCandidates.size();
      if (overrideSolutionEndIdx) {
        solutionEndIdx = overrideSolutionEndIdx;
      }
      for (size_t solutionIdx = solutionStartIdx; solutionIdx < solutionEndIdx;
        solutionIdx++) {
        printf("G[%llu/%llu] P[%llu/%llu] S[%llu/%llu] ", ULL exactMatchIdx, ULL exactMatchEndIdx, ULL problemIdx, ULL problemEndIdx, ULL solutionIdx, ULL solutionEndIdx);

        // get solution candidate
        Tensile::Solution *solution = solutionCandidates[solutionIdx];

        //for (unsigned int j = 0; j < 4; j++) {
        if (doValidation) {
          // re-initialize device buffers
#if Tensile_BACKEND_OCL
          clEnqueueWriteBuffer(ctrl.queues[0],
            static_cast<cl_mem>(deviceTensorDataC.data),
            CL_FALSE, deviceTensorDataC.offset, sizeC, initialDataC,
            0, nullptr, nullptr);
          clEnqueueWriteBuffer(ctrl.queues[0],
            static_cast<cl_mem>(deviceTensorDataA.data),
            CL_FALSE, deviceTensorDataA.offset, sizeA, initialDataA,
            0, nullptr, nullptr);
          clEnqueueWriteBuffer(ctrl.queues[0],
            static_cast<cl_mem>(deviceTensorDataB.data),
            CL_FALSE, deviceTensorDataB.offset, sizeB, initialDataB,
            0, nullptr, nullptr);
          clFinish(ctrl.queues[0]);
#elif Tensile_BACKEND_HIP
          status = hipMemcpy(deviceTensorDataC.data, initialDataC,
            sizeC, hipMemcpyHostToDevice);
          status = hipMemcpy(deviceTensorDataA.data, initialDataA,
            sizeA, hipMemcpyHostToDevice);
          status = hipMemcpy(deviceTensorDataB.data, initialDataB,
            sizeB, hipMemcpyHostToDevice);
          status = hipStreamSynchronize(nullptr);
#endif
        }
        // for validation ctrl.benchmark = 0; 1 call to enqueueEntry below
        ctrl.benchmark = 1;
        unsigned int numSamples = 1;
        if (doValidation) {
          //ctrl.benchmark = 0;
          numSamples = 1;
        }
        TensileTensorDataConst constA{ deviceTensorDataA.data,
            deviceTensorDataA.offset };
        TensileTensorDataConst constB{ deviceTensorDataB.data,
            deviceTensorDataB.offset };
        for (unsigned int s = 0; s < numSamples; s++) {
          solution->enqueueEntry(
            deviceTensorDataC,
            constA,
            constB,
            alpha,
            beta,
            ctrl,
            true /*do print*/);
        }
        if (doValidation) {
          for (unsigned int i = 0; i < ctrl.numQueues; i++) {
#if Tensile_BACKEND_OCL
            status = clFinish(ctrl.queues[i]);
#elif Tensile_BACKEND_HIP
            status = hipStreamSynchronize(ctrl.queues[i]);
#endif
            CL_CHECK(status)
        }
        }
        //}

#if 0
      // peek at gpu result
        status = clEnqueueReadBuffer(ctrl.queues[0], (cl_mem)deviceTensorDataC.data, CL_TRUE, deviceTensorDataC.offset, sizeC, deviceTensorDataOnHostC.data, 0, nullptr, nullptr);
        CL_CHECK(status)
          status = clFinish(ctrl.queues[0]);
        CL_CHECK(status

          // print gpu in tensor form
          printf("\nTensorC-GPU:\n");
        printf(problemReference->pimpl->tensorC.toString(deviceTensorDataOnHostC).c_str());
#endif
        delete solution;
        solutionCandidates[solutionIdx] = nullptr;
      } // solution loop
      tensileDestroyProblem(problem);

    } // problem loop

    // write exactMatch XML to file
    tensileTeardown();

  } // exact match index

  destroyTensorData();
  destroyControls();
#ifdef _CRTDBG_MAP_ALLOC
  _CrtDumpMemoryLeaks();
#endif
  return 0;
} // end main

void initTensorData() {
  printf("Status: Initializing tensor data %.1f MB", 2*(tensorSizeMaxC + tensorSizeMaxA + tensorSizeMaxB)/(1024.f*1024.f) );
  // dummy tensors for filling initial data


  Tensile::factor(tensorSizeMaxC, tensorSizeMaxC_0, tensorSizeMaxC_1);
  Tensile::factor(tensorSizeMaxA, tensorSizeMaxA_0, tensorSizeMaxA_1);
  Tensile::factor(tensorSizeMaxB, tensorSizeMaxB_0, tensorSizeMaxB_1);

  //printf("maxC: %llu %s %u * %u\n", tensorSizeMaxC, factoredC ? "==" : "!=", tensorSizeMaxC_0, tensorSizeMaxC_1);
  //printf("maxA: %llu %s %u * %u\n", tensorSizeMaxA, factoredA ? "==" : "!=", tensorSizeMaxA_0, tensorSizeMaxA_1);
  //printf("maxB: %llu %s %u * %u\n", tensorSizeMaxB, factoredB ? "==" : "!=", tensorSizeMaxB_0, tensorSizeMaxB_1);


  initialTensorFloatC.dataType = tensileDataTypeSingle;
  initialTensorFloatC.numDimensions = 2;
  initialTensorFloatC.dimensions[0].stride = 1;
  initialTensorFloatC.dimensions[0].size = tensileMax(1, tensorSizeMaxC_0 / 4) /*bytes per float*/;
  initialTensorFloatC.dimensions[1].stride = initialTensorFloatC.dimensions[0].size;
  initialTensorFloatC.dimensions[1].size = tensorSizeMaxC_1;
  initialTensorFloatA.dataType = tensileDataTypeSingle;
  initialTensorFloatA.numDimensions = 2;
  initialTensorFloatA.dimensions[0].stride = 1;
  initialTensorFloatA.dimensions[0].size = tensileMax(1, tensorSizeMaxA_0 / 4) /*bytes per float*/;
  initialTensorFloatA.dimensions[1].stride = initialTensorFloatA.dimensions[0].size;
  initialTensorFloatA.dimensions[1].size = tensorSizeMaxA_1;
  initialTensorFloatB.dataType = tensileDataTypeSingle;
  initialTensorFloatB.numDimensions = 2;
  initialTensorFloatB.dimensions[0].stride = 1;
  initialTensorFloatB.dimensions[0].size = tensileMax(1, tensorSizeMaxB_0 / 4) /*bytes per float*/;
  initialTensorFloatB.dimensions[1].stride = initialTensorFloatB.dimensions[0].size;
  initialTensorFloatB.dimensions[1].size = tensorSizeMaxB_1;

  initialTensorDoubleC.dataType = tensileDataTypeDouble;
  initialTensorDoubleC.numDimensions = 1;
  initialTensorDoubleC.dimensions[0].stride = 1;
  initialTensorDoubleC.dimensions[0].size = tensileMax(1, tensorSizeMaxC_0 / 8) /*bytes per double*/;
  initialTensorDoubleC.dimensions[1].stride = initialTensorDoubleC.dimensions[0].size;
  initialTensorDoubleC.dimensions[1].size = tensorSizeMaxC_1;
  initialTensorDoubleA.dataType = tensileDataTypeDouble;
  initialTensorDoubleA.numDimensions = 1;
  initialTensorDoubleA.dimensions[0].stride = 1;
  initialTensorDoubleA.dimensions[0].size = tensileMax(1, tensorSizeMaxA_0 / 8) /*bytes per double*/;
  initialTensorDoubleA.dimensions[1].stride = initialTensorDoubleA.dimensions[0].size;
  initialTensorDoubleA.dimensions[1].size = tensorSizeMaxA_1;
  initialTensorDoubleB.dataType = tensileDataTypeDouble;
  initialTensorDoubleB.numDimensions = 1;
  initialTensorDoubleB.dimensions[0].stride = 1;
  initialTensorDoubleB.dimensions[0].size = tensileMax(1, tensorSizeMaxB_0 / 8) /*bytes per double*/;
  initialTensorDoubleB.dimensions[1].stride = initialTensorDoubleB.dimensions[0].size;
  initialTensorDoubleB.dimensions[1].size = tensorSizeMaxB_1;

  // initial tensor data for host buffers
  initialTensorDataFloatC.data = new float[initialTensorFloatC.dimensions[1].stride * initialTensorFloatC.dimensions[1].size];
  initialTensorDataFloatA.data = new float[initialTensorFloatA.dimensions[1].stride * initialTensorFloatA.dimensions[1].size];
  initialTensorDataFloatB.data = new float[initialTensorFloatB.dimensions[1].stride * initialTensorFloatB.dimensions[1].size];
  initialTensorDataDoubleC.data = new double[initialTensorDoubleC.dimensions[1].stride * initialTensorDoubleC.dimensions[1].size];
  initialTensorDataDoubleA.data = new double[initialTensorDoubleA.dimensions[1].stride * initialTensorDoubleA.dimensions[1].size];
  initialTensorDataDoubleB.data = new double[initialTensorDoubleB.dimensions[1].stride * initialTensorDoubleB.dimensions[1].size];
  initialTensorDataFloatC.offset = 0;
  initialTensorDataFloatA.offset = 0;
  initialTensorDataFloatB.offset = 0;
  initialTensorDataDoubleC.offset = 0;
  initialTensorDataDoubleA.offset = 0;
  initialTensorDataDoubleB.offset = 0;

  printf("."); fillTensor( initialTensorFloatC, initialTensorDataFloatC, tensorFillTypeC, nullptr);
  printf("."); fillTensor( initialTensorFloatA, initialTensorDataFloatA, tensorFillTypeA, nullptr);
  printf("."); fillTensor( initialTensorFloatB, initialTensorDataFloatB, tensorFillTypeB, nullptr);
  printf("."); fillTensor( initialTensorDoubleC, initialTensorDataDoubleC, tensorFillTypeC, nullptr);
  printf("."); fillTensor( initialTensorDoubleA, initialTensorDataDoubleA, tensorFillTypeA, nullptr);
  printf("."); fillTensor( initialTensorDoubleB, initialTensorDataDoubleB, tensorFillTypeB, nullptr);
  printf("."); 

#if Tensile_BACKEND_OCL
  deviceTensorDataC.data = static_cast<void *>(clCreateBuffer(context, CL_MEM_READ_WRITE, tensorSizeMaxC, nullptr, &status));
  deviceTensorDataA.data = static_cast<void *>(clCreateBuffer(context, CL_MEM_READ_ONLY, tensorSizeMaxA, nullptr, &status));
  deviceTensorDataB.data = static_cast<void *>(clCreateBuffer(context, CL_MEM_READ_ONLY, tensorSizeMaxB, nullptr, &status));
#elif Tensile_BACKEND_HIP
  status = hipMalloc( &deviceTensorDataC.data, tensorSizeMaxC );
  status = hipMalloc( &deviceTensorDataA.data, tensorSizeMaxA );
  status = hipMalloc( &deviceTensorDataB.data, tensorSizeMaxB );
#endif

  deviceTensorDataC.offset = 0;
  deviceTensorDataOnHostC.data = malloc(tensorSizeMaxC);
  deviceTensorDataOnHostC.offset = 0;
  deviceTensorDataA.offset = 0;
  deviceTensorDataOnHostA.data = malloc(tensorSizeMaxA);
  deviceTensorDataOnHostA.offset = 0;
  deviceTensorDataB.offset = 0;
  deviceTensorDataOnHostB.data = malloc(tensorSizeMaxB);
  deviceTensorDataOnHostB.offset = 0;

  // reference tensor data
  referenceTensorDataC.data = malloc(tensorSizeMaxC);
  referenceTensorDataC.offset = 0;

  // scalars
  alphaFloat = new float[2];
  alphaFloat[0] = ALPHA;
  alphaFloat[1] = ALPHA;
  betaFloat = new float[2];
  betaFloat[0] = BETA;
  betaFloat[1] = BETA;
  alphaDouble = new double[2];
  alphaDouble[0] = ALPHA;
  alphaDouble[1] = ALPHA;
  betaDouble = new double[2];
  betaDouble[0] = BETA;
  betaDouble[1] = BETA;
  printf("done.\n");
}

void destroyTensorData() {

  delete[] static_cast<float *>(initialTensorDataFloatC.data);
  delete[] static_cast<float *>(initialTensorDataFloatA.data);
  delete[] static_cast<float *>(initialTensorDataFloatB.data);
  delete[] static_cast<double *>(initialTensorDataDoubleC.data);
  delete[] static_cast<double *>(initialTensorDataDoubleA.data);
  delete[] static_cast<double *>(initialTensorDataDoubleB.data);

  delete[] static_cast<float *>(deviceTensorDataOnHostC.data);
  delete[] static_cast<float *>(deviceTensorDataOnHostA.data);
  delete[] static_cast<float *>(deviceTensorDataOnHostB.data);
  delete[] static_cast<float *>(referenceTensorDataC.data);

  delete[] alphaFloat;
  delete[] betaFloat;
  delete[] alphaDouble;
  delete[] betaDouble;

#if Tensile_BACKEND_OCL
  clReleaseMemObject(static_cast<cl_mem>(deviceTensorDataC.data));
  clReleaseMemObject(static_cast<cl_mem>(deviceTensorDataA.data));
  clReleaseMemObject(static_cast<cl_mem>(deviceTensorDataB.data));
#elif Tensile_BACKEND_OCL
  hipFree(deviceTensorDataC.data);
  hipFree(deviceTensorDataA.data);
  hipFree(deviceTensorDataB.data);
#endif

}

void fillTensor(TensileTensor inputTensor, TensileTensorData tensorData, Tensile::Tensor::FillType fillType, void *src) {
  Tensile::Tensor tensor(inputTensor);
  tensor.fill(tensorData, fillType, src);
}

void initControls() {
#if Tensile_BACKEND_OCL
  // setup opencl objects
  status = clGetPlatformIDs(0, nullptr, &numPlatforms);
  platforms = new cl_platform_id[numPlatforms];
  status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
  platform = platforms[platformIdx];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  devices = new cl_device_id[numDevices];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
  device = devices[deviceIdx];
  size_t nameLength;
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, 0, nullptr, &nameLength );
  char *deviceName = new char[nameLength+1];
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, nameLength, deviceName, 0 );
  Tensile::makeFileNameSafe( deviceName );
  printf("Device[%u/%u][%u/%u]: %s\n", platformIdx, numPlatforms, deviceIdx, numDevices, deviceName);
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);

  // device control
  ctrl = tensileCreateEmptyControl();
  for (ctrl.numQueues = 0; ctrl.numQueues < ctrl.maxQueues; ctrl.numQueues++) {
    ctrl.queues[ctrl.numQueues] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  }
#elif Tensile_BACKEND_HIP
  status = hipGetDeviceCount( &numDevices );
  device = 0;
  status = hipSetDevice( device );
  ctrl = tensileCreateEmptyControl();
  for (ctrl.numQueues = 0; ctrl.numQueues < ctrl.maxQueues; ctrl.numQueues++) {
    status = hipStreamCreate( &ctrl.queues[ctrl.numQueues] );
  }
#endif

  // host control
  ctrlValidation = tensileCreateEmptyControl();

  // reference device
  deviceProfileReference.numDevices = 1;
#ifdef WIN32
  sprintf_s(deviceProfileReference.devices[0].name, deviceProfileReference.devices[0].maxNameLength, "cpu");
#else
  std::sprintf(deviceProfileReference.devices[0].name, "cpu" );
#endif
}

void destroyControls() {

#if Tensile_BACKEND_OCL
  for (unsigned int i = 0; i < ctrl.numQueues; i++) {
    clReleaseCommandQueue(ctrl.queues[i]);
  }
  clReleaseContext(context);
  for (unsigned int i = 0; i < numDevices; i++) {
    clReleaseDevice(devices[i]);
  }
  delete[] devices;
  delete[] platforms;
#elif Tensile_BACKEND_HIP
  for (unsigned int i = 0; i < ctrl.numQueues; i++) {
    hipStreamDestroy(ctrl.queues[i]);
  }
#endif

}

void parseCommandLineOptions(int argc, char *argv[]) {
  doValidation = false;
  doValidationKernels = false;
  for (int argIdx = 0; argIdx < argc; argIdx++) {
    char *arg = argv[argIdx];
    if (std::strcmp(arg, "--validate") == 0) {
      doValidation = true;
    }
    if (std::strcmp(arg, "--start-problem") == 0) {
      overrideProblemStartIdx = atol(argv[argIdx+1]);
      argIdx++;
    }
    if (std::strcmp(arg, "--end-problem") == 0) {
      overrideProblemEndIdx = atol(argv[argIdx+1]);
      argIdx++;
    }
    if (std::strcmp(arg, "--start-solution") == 0) {
      overrideSolutionStartIdx = atol(argv[argIdx+1]);
      argIdx++;
    }
    if (std::strcmp(arg, "--end-solution") == 0) {
      overrideSolutionEndIdx = atol(argv[argIdx+1]);
      argIdx++;
    }
    if (std::strcmp(arg, "--start-group") == 0) {
      overrideExactMatchStartIdx = atol(argv[argIdx + 1]);
      argIdx++;
    }
    if (std::strcmp(arg, "--end-group") == 0) {
      overrideExactMatchEndIdx = atol(argv[argIdx + 1]);
      argIdx++;
    }
    if (std::strcmp(arg, "--device") == 0) {
      deviceIdx = atol(argv[argIdx+1]);
      deviceOverride = true;
      argIdx++;
    }
    if (std::strcmp(arg, "--platform") == 0) {
      platformIdx = atol(argv[argIdx+1]);
      deviceOverride = true;
      argIdx++;
    }
    if (std::strcmp(arg, "-h") == 0) {
      printf("Usage: TensileBenchmark [options]\n");
      printf("Options:\n");
      printf("  --validate         | validate results (very slow); default is off.\n");
      printf("  --start-group x    | start at problem-group idx (for continuing or debugging); default is 0.\n");
      printf("  --end-group x      | end at problem-group idx (for continuing or debugging); default is last.\n");
      printf("  --start-problem x  | start at problem idx (for continuing or debugging); default is 0.\n");
      printf("  --end-problem x    | end at problem idx (for debugging); default is last.\n");
      printf("  --start-solution x | start at solution idx (for debugging); default is 0.\n");
      printf("  --end-solution x   | end at solution idx (for debugging); default is last.\n");
      printf("  --platform x       | use platform idx (for OpenCL only); default specified by benchmark problem.\n");
      printf("  --device x         | use device idx; default specified by benchmark problem.\n");
      argIdx++;
    }
    
  }

  if (!deviceOverride) {
    // select device based on TensileProblem
#if Tensile_BACKEND_OCL
    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    for (unsigned int p = 0; p < numPlatforms; p++) {
      status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
      devices = new cl_device_id[numDevices];
      status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
      for (unsigned int d = 0; d < numDevices; d++) {
        deviceProfile.numDevices = 1;
        status = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, deviceProfile.devices[0].maxNameLength, deviceProfile.devices[0].name, 0);
        Tensile::makeFileNameSafe( deviceProfile.devices[0].name );
        status = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(deviceProfile.devices[0].clockFrequency), &deviceProfile.devices[0].clockFrequency, 0);
        status = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(deviceProfile.devices[0].numComputeUnits), &deviceProfile.devices[0].numComputeUnits, 0);
        if (strcmp(deviceProfile.devices[0].name, benchmarkDeviceName) == 0 && deviceProfile.devices[0].numComputeUnits == benchmarkDeviceNumComputeUnits && deviceProfile.devices[0].clockFrequency == benchmarkDeviceClockFrequency) {
          platformIdx = p;
          deviceIdx = d;
          printf("Using benchmark-requested device: P[%u] D[%u]: %s %u CUs @ %u MHz (%.0f GFlop/s).\n",
              platformIdx,
              deviceIdx,
              deviceProfile.devices[0].name,
              deviceProfile.devices[0].numComputeUnits,
              deviceProfile.devices[0].clockFrequency,
              1.0*deviceProfile.devices[0].numComputeUnits*deviceProfile.devices[0].clockFrequency*deviceProfile.devices[0].flopsPerClock / 1000.f);
          break;
        }
      }
      delete[] devices;
    }
    delete[] platforms;
#else
    status = hipGetDeviceCount( &numDevices );
    printf("%i HIP devices\n", numDevices);
    for (int i = 0; i < numDevices; i++) {
      hipDeviceProp_t deviceProperties;
      hipGetDeviceProperties( &deviceProperties, i);
      deviceProfile.numDevices = 1;
      strcpy( deviceProfile.devices[0].name, deviceProperties.name );
      Tensile::makeFileNameSafe( deviceProfile.devices[0].name );
      deviceProfile.devices[0].numComputeUnits = deviceProperties.multiProcessorCount;
      deviceProfile.devices[0].clockFrequency = deviceProperties.clockRate / 1000; // kHz -> MHz
      if (strcmp(deviceProfile.devices[0].name, benchmarkDeviceName) == 0
          && deviceProfile.devices[0].numComputeUnits == benchmarkDeviceNumComputeUnits
          && deviceProfile.devices[0].clockFrequency == benchmarkDeviceClockFrequency) {
        platformIdx = 0;
        deviceIdx = i;
        printf("Using benchmark-requested device: D[%u]: %s %u CUs @ %u MHz (%.0f GFlop/s).\n",
            deviceIdx,
            deviceProfile.devices[0].name,
            deviceProfile.devices[0].numComputeUnits,
            deviceProfile.devices[0].clockFrequency,
            1.0*deviceProfile.devices[0].numComputeUnits*deviceProfile.devices[0].clockFrequency*deviceProfile.devices[0].flopsPerClock / 1000.f);
        break;
      }
    }
    
#endif
  } else {
#if Tensile_BACKEND_OCL
    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    status = clGetDeviceIDs(platforms[platformIdx], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
    devices = new cl_device_id[numDevices];
    status = clGetDeviceIDs(platforms[platformIdx], CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
    deviceProfile.numDevices = 1;
    status = clGetDeviceInfo(devices[deviceIdx], CL_DEVICE_NAME, deviceProfile.devices[0].maxNameLength, deviceProfile.devices[0].name, 0);
    Tensile::makeFileNameSafe( deviceProfile.devices[0].name );
    status = clGetDeviceInfo(devices[deviceIdx], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(deviceProfile.devices[0].clockFrequency), &deviceProfile.devices[0].clockFrequency, 0);
    status = clGetDeviceInfo(devices[deviceIdx], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(deviceProfile.devices[0].numComputeUnits), &deviceProfile.devices[0].numComputeUnits, 0);
    printf("Using user-overrided device: P[%u] D[%u]: %s %u CUs @ %u MHz (%.0f GFlop/s).\n",
        platformIdx,
        deviceIdx,
        deviceProfile.devices[0].name,
        deviceProfile.devices[0].numComputeUnits,
        deviceProfile.devices[0].clockFrequency,
        1.0*deviceProfile.devices[0].numComputeUnits*deviceProfile.devices[0].clockFrequency*deviceProfile.devices[0].flopsPerClock / 1000.f);
#else
    hipDeviceProp_t deviceProperties;
    hipGetDeviceProperties( &deviceProperties, deviceIdx);
    deviceProfile.numDevices = 1;
    strcpy( deviceProfile.devices[0].name, deviceProperties.name );
    Tensile::makeFileNameSafe( deviceProfile.devices[0].name );
    deviceProfile.devices[0].numComputeUnits = deviceProperties.multiProcessorCount;
    deviceProfile.devices[0].clockFrequency = deviceProperties.clockRate/1000; // kHz -> MHz
    printf("Using user-overrided device: D[%u]: %s %u CUs @ %u MHz (%.0f GFlop/s).\n",
        deviceIdx,
        deviceProfile.devices[0].name,
        deviceProfile.devices[0].numComputeUnits,
        deviceProfile.devices[0].clockFrequency,
        1.0*deviceProfile.devices[0].numComputeUnits*deviceProfile.devices[0].clockFrequency*deviceProfile.devices[0].flopsPerClock / 1000.f);
#endif
  }
}

#ifdef Tensile_ENABLE_FP16
bool tensileDataTypeIsHalf(TensileDataType
dataType
  ) {
  return dataType == tensileDataTypeHalf
      || dataType == tensileDataTypeComplexHalf
      || dataType == tensileDataTypeComplexConjugateHalf;
}
#endif
bool tensileDataTypeIsFloat(TensileDataType dataType) {
  return dataType == tensileDataTypeSingle
    || dataType == tensileDataTypeComplexSingle
    || dataType == tensileDataTypeComplexConjugateSingle;
}
bool tensileDataTypeIsDouble(TensileDataType dataType) {
  return dataType == tensileDataTypeDouble
    || dataType == tensileDataTypeComplexDouble
    || dataType == tensileDataTypeComplexConjugateDouble;
}
#if 0

3 Clients
(1) BenchmarkProblem Step
options:
 - validate
 - api time or runtime time
 - num flushes, finishes
 - problem size range
 - output csv filename
hardcoded:
 - problem type
 - included file of solutions
output:
 - csv of 
process:
 - get max problem size
 - allocate matrices
 - for each problem in range
   - call included file which calls all solutions
   - returns vector of times; one for each solution


(2) BenchmarkProblem Final
options:
 - same as above


(3) Client
options:
 - same as above
 - problem type
hardcoded:
 - library backend
output:
 - csv of performance; 1 solution many problems

int main() {
  parseCommandLineParameters();
  allocateRuntimeData();
  allocateTensorData();
  performBenchmark();
  writeBenchmarkData();
  teardown();
}

#endif

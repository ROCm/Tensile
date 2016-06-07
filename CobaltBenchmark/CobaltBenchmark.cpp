/*******************************************************************************
 * Cobalt Benchmark
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

#include "CobaltBenchmark.h"
#include "Cobalt.h"
#include "Tools.h"
#include "Solution.h"
#include "CobaltSolutionCandidates.h"
#include "SolutionTensorContractionCPU.h"
#include "StructOperations.h"
#include "MathTemplates.h"

#include <tuple>
#include <string>
#include <cstring>
#include <cstdio>

#define ULL (unsigned long long)

Cobalt::Tensor::FillType tensorFillTypeC = Cobalt::Tensor::fillTypeRandom;
Cobalt::Tensor::FillType tensorFillTypeA = Cobalt::Tensor::fillTypeRandom;
Cobalt::Tensor::FillType tensorFillTypeB = Cobalt::Tensor::fillTypeRandom;

//#define MAX_PROBLEMS 1
//#define MAX_SOLUTIONS_PER_PROBLEM 1

#define ALPHA 1
#define BETA  0

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {
#ifdef _CRTDBG_MAP_ALLOC
  _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
#endif
  // parse commandline options
  parseCommandLineOptions(argc, argv);

  // setup CobaltLib
  std::string logFilePath = Cobalt_DIR_SOLUTIONS;
  //logFilePath += "/CobaltBenchmark_log.xml";
  cobaltSetup(logFilePath.c_str());

  // create CobaltControl
  initControls();

  // initialize initial buffer values for validation
  initTensorData();
  
  size_t problemStartIdx = 0;
  size_t problemEndIdx = numProblems;
#ifdef MAX_PROBLEMS
  if (problemEndIdx > MAX_PROBLEMS) {
    problemEndIdx = MAX_PROBLEMS;
  }
#endif
  // for each problem
  for ( size_t problemIdx = problemStartIdx; problemIdx < problemEndIdx;
      problemIdx++ ) {

#ifdef _CRTDBG_MAP_ALLOC
    if (problemIdx > 0) {
      _CrtMemCheckpoint( &s2 );
      int diff = _CrtMemDifference(&s3, &s1, &s2);
      _CrtMemDumpStatistics(&s3);
      printf("Difference[%llu] = %i\n", ULL problemIdx-1, diff);
    }
    _CrtMemCheckpoint(&s1);
#endif

    // info about problem
    CobaltProblem problem;
    std::vector<Cobalt::Solution *> solutionCandidates;
    initializeSolutionCandidates(&problem, &solutionCandidates, problemIdx);
    bool isFloatC = cobaltDataTypeIsFloat(problem->pimpl->getDataTypeC());
    bool isFloatA = cobaltDataTypeIsFloat(problem->pimpl->getDataTypeA());
    bool isFloatB = cobaltDataTypeIsFloat(problem->pimpl->getDataTypeB());
    bool isFloatAlpha = cobaltDataTypeIsFloat(problem->pimpl->getDataTypeAlpha());
    bool isDoubleAlpha = cobaltDataTypeIsDouble(problem->pimpl->getDataTypeAlpha());
    bool isFloatBeta = cobaltDataTypeIsFloat(problem->pimpl->getDataTypeBeta());
    bool isDoubleBeta = cobaltDataTypeIsDouble(problem->pimpl->getDataTypeBeta());
    size_t sizeC = problem->pimpl->tensorC.numBytes();
    size_t sizeA = problem->pimpl->tensorA.numBytes();
    size_t sizeB = problem->pimpl->tensorB.numBytes();
    void *initialDataC = isFloatC ? initialTensorDataFloatC.data : initialTensorDataDoubleC.data;
    void *initialDataA = isFloatA ? initialTensorDataFloatA.data : initialTensorDataDoubleA.data;
    void *initialDataB = isFloatB ? initialTensorDataFloatB.data : initialTensorDataDoubleB.data;
    CobaltScalarData alpha{ isFloatAlpha ? static_cast<void*>(alphaFloat)
        : isDoubleAlpha ? static_cast<void*>(alphaDouble) : nullptr };
    CobaltScalarData beta{ isFloatBeta ? static_cast<void*>(betaFloat)
        : isDoubleBeta ? static_cast<void*>(betaDouble) : nullptr };

    // re-initialize device input buffers
#if Cobalt_BACKEND_OPENCL12
    clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataA.data), CL_FALSE, deviceTensorDataA.offset, sizeA, initialDataA, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataB.data), CL_FALSE, deviceTensorDataB.offset, sizeB, initialDataB, 0, nullptr, nullptr);
#elif Cobalt_BACKEND_HIP
    status = hipMemcpy( deviceTensorDataA.data, initialDataA,
        sizeA, hipMemcpyHostToDevice );
    status = hipMemcpy( deviceTensorDataB.data, initialDataB,
        sizeB, hipMemcpyHostToDevice );
    status = hipStreamSynchronize(nullptr);
#endif

    // calculate reference C once for problem
    if (doValidation) {

      // get reference solution
      Cobalt::Solution *solutionReference;
      problem->pimpl->deviceProfile = deviceProfileReference;
      std::tie(solutionReference,cobaltStatus) = getSolutionCPU( *(problem->pimpl) );

      // re-initialize reference buffers
      memcpy(referenceTensorDataC.data, initialDataC, sizeC);
      CobaltTensorDataConst referenceTensorDataA{ initialDataA, 0 };
      CobaltTensorDataConst referenceTensorDataB{ initialDataB, 0 };

      // enqueue reference solution
      printf("Status: Enqueueing reference for %s ...", problem->pimpl->toString().c_str());
      solutionReference->enqueue(
        referenceTensorDataC,
        referenceTensorDataA,
        referenceTensorDataB,
        alpha,
        beta,
        ctrlValidation );
      ctrl.validate = &referenceTensorDataC;
      printf("done.\n");

    } else {
      printf("Status: Problem[%llu/%llu] %s\n", ULL problemIdx, ULL problemEndIdx, problem->pimpl->toString().c_str());
    }


    size_t solutionStartIdx = 0;
    size_t solutionEndIdx = solutionCandidates.size();
#ifdef MAX_SOLUTIONS_PER_PROBLEM
      if (solutionEndIdx > MAX_SOLUTIONS_PER_PROBLEM) {
        solutionEndIdx = MAX_SOLUTIONS_PER_PROBLEM;
      }
#endif
    for ( size_t solutionIdx = solutionStartIdx; solutionIdx < solutionEndIdx;
        solutionIdx++ ) {
      printf("P[%llu/%llu] S[%llu/%llu] ", ULL problemIdx, ULL problemEndIdx, ULL solutionIdx, ULL solutionEndIdx);

      // get solution candidate
      Cobalt::Solution *solution = solutionCandidates[ solutionIdx ];

      //for (unsigned int j = 0; j < 4; j++) {
      if (doValidation) {
        // re-initialize device buffers
#if Cobalt_BACKEND_OPENCL12
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
#elif Cobalt_BACKEND_HIP
        status = hipMemcpy( deviceTensorDataC.data, initialDataC,
            sizeC, hipMemcpyHostToDevice );
        status = hipMemcpy( deviceTensorDataA.data, initialDataA,
            sizeA, hipMemcpyHostToDevice );
        status = hipMemcpy( deviceTensorDataB.data, initialDataB,
            sizeB, hipMemcpyHostToDevice );
        status = hipStreamSynchronize(nullptr);
#endif
      }
      // for validation ctrl.benchmark = 0; 1 call to enqueueEntry below
      ctrl.benchmark = 4;
      unsigned int numSamples = 4;
      if (doValidation) {
        //ctrl.benchmark = 0;
        numSamples = 1;
      }
      CobaltTensorDataConst constA{ deviceTensorDataA.data,
          deviceTensorDataA.offset };
      CobaltTensorDataConst constB{ deviceTensorDataB.data,
          deviceTensorDataB.offset };
      for (unsigned int s = 0; s < numSamples; s++) {
        solution->enqueueEntry(
            deviceTensorDataC,
            constA,
            constB,
            alpha,
            beta,
            ctrl );
      }
      if (doValidation) {
        for (unsigned int i = 0; i < ctrl.numQueues; i++) {
#if Cobalt_BACKEND_OPENCL12
          status = clFinish(ctrl.queues[i]);
#elif Cobalt_BACKEND_HIP
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
        printf( problemReference->pimpl->tensorC.toString(deviceTensorDataOnHostC).c_str() );
#endif
      delete solution;
      solutionCandidates[ solutionIdx ] = nullptr;
    } // solution loop
    cobaltDestroyProblem( problem );
    
  } // problem loop
  destroyTensorData();
  destroyControls();
  cobaltTeardown();
#ifdef _CRTDBG_MAP_ALLOC
  int leaks = _CrtDumpMemoryLeaks();
#endif
  return 0;
} // end main

void initTensorData() {
  printf("Status: Initializing tensor data %.1f MB", 2*(tensorSizeMaxC + tensorSizeMaxA + tensorSizeMaxB)/(1024.f*1024.f) );
  // dummy tensors for filling initial data


  bool factoredC = Cobalt::factor(tensorSizeMaxC, tensorSizeMaxC_0, tensorSizeMaxC_1);
  bool factoredA = Cobalt::factor(tensorSizeMaxA, tensorSizeMaxA_0, tensorSizeMaxA_1);
  bool factoredB = Cobalt::factor(tensorSizeMaxB, tensorSizeMaxB_0, tensorSizeMaxB_1);

  //printf("maxC: %llu %s %u * %u\n", tensorSizeMaxC, factoredC ? "==" : "!=", tensorSizeMaxC_0, tensorSizeMaxC_1);
  //printf("maxA: %llu %s %u * %u\n", tensorSizeMaxA, factoredA ? "==" : "!=", tensorSizeMaxA_0, tensorSizeMaxA_1);
  //printf("maxB: %llu %s %u * %u\n", tensorSizeMaxB, factoredB ? "==" : "!=", tensorSizeMaxB_0, tensorSizeMaxB_1);


  initialTensorFloatC.dataType = cobaltDataTypeSingle;
  initialTensorFloatC.numDimensions = 2;
  initialTensorFloatC.dimensions[0].stride = 1;
  initialTensorFloatC.dimensions[0].size = tensorSizeMaxC_0 / 4 /*bytes per float*/;
  initialTensorFloatC.dimensions[1].stride = initialTensorFloatC.dimensions[0].size;
  initialTensorFloatC.dimensions[1].size = tensorSizeMaxC_1;
  initialTensorFloatA.dataType = cobaltDataTypeSingle;
  initialTensorFloatA.numDimensions = 2;
  initialTensorFloatA.dimensions[0].stride = 1;
  initialTensorFloatA.dimensions[0].size = tensorSizeMaxA_0 / 4 /*bytes per float*/;
  initialTensorFloatA.dimensions[1].stride = initialTensorFloatA.dimensions[0].size;
  initialTensorFloatA.dimensions[1].size = tensorSizeMaxA_1;
  initialTensorFloatB.dataType = cobaltDataTypeSingle;
  initialTensorFloatB.numDimensions = 2;
  initialTensorFloatB.dimensions[0].stride = 1;
  initialTensorFloatB.dimensions[0].size = tensorSizeMaxB_0 / 4 /*bytes per float*/;
  initialTensorFloatB.dimensions[1].stride = initialTensorFloatB.dimensions[0].size;
  initialTensorFloatB.dimensions[1].size = tensorSizeMaxB_1;

  initialTensorDoubleC.dataType = cobaltDataTypeDouble;
  initialTensorDoubleC.numDimensions = 1;
  initialTensorDoubleC.dimensions[0].stride = 1;
  initialTensorDoubleC.dimensions[0].size = tensorSizeMaxC_0 / 8 /*bytes per double*/;
  initialTensorDoubleC.dimensions[1].stride = initialTensorDoubleC.dimensions[0].size;
  initialTensorDoubleC.dimensions[1].size = tensorSizeMaxC_1;
  initialTensorDoubleA.dataType = cobaltDataTypeDouble;
  initialTensorDoubleA.numDimensions = 1;
  initialTensorDoubleA.dimensions[0].stride = 1;
  initialTensorDoubleA.dimensions[0].size = tensorSizeMaxA_0 / 8 /*bytes per double*/;
  initialTensorDoubleA.dimensions[1].stride = initialTensorDoubleA.dimensions[0].size;
  initialTensorDoubleA.dimensions[1].size = tensorSizeMaxA_1;
  initialTensorDoubleB.dataType = cobaltDataTypeDouble;
  initialTensorDoubleB.numDimensions = 1;
  initialTensorDoubleB.dimensions[0].stride = 1;
  initialTensorDoubleB.dimensions[0].size = tensorSizeMaxB_0 / 8 /*bytes per double*/;
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

#if Cobalt_BACKEND_OPENCL12
  deviceTensorDataC.data = static_cast<void *>(clCreateBuffer(context, CL_MEM_READ_WRITE, tensorSizeMaxC, nullptr, &status));
  deviceTensorDataA.data = static_cast<void *>(clCreateBuffer(context, CL_MEM_READ_ONLY, tensorSizeMaxA, nullptr, &status));
  deviceTensorDataB.data = static_cast<void *>(clCreateBuffer(context, CL_MEM_READ_ONLY, tensorSizeMaxB, nullptr, &status));
#elif Cobalt_BACKEND_HIP
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

#if Cobalt_BACKEND_OPENCL12
  clReleaseMemObject(static_cast<cl_mem>(deviceTensorDataC.data));
  clReleaseMemObject(static_cast<cl_mem>(deviceTensorDataA.data));
  clReleaseMemObject(static_cast<cl_mem>(deviceTensorDataB.data));
#elif Cobalt_BACKEND_OPENCL12
  hipFree(deviceTensorDataC.data);
  hipFree(deviceTensorDataA.data);
  hipFree(deviceTensorDataB.data);
#endif

}

void fillTensor(CobaltTensor inputTensor, CobaltTensorData tensorData, Cobalt::Tensor::FillType fillType, void *src) {
  Cobalt::Tensor tensor(inputTensor);
  tensor.fill(tensorData, fillType, src);
}

void initControls() {
#if Cobalt_BACKEND_OPENCL12
  // setup opencl objects
  status = clGetPlatformIDs(0, nullptr, &numPlatforms);
  platforms = new cl_platform_id[numPlatforms];
  status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
  platform = platforms[0];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  devices = new cl_device_id[numDevices];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
  device = devices[0];
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);

  // device control
  ctrl = cobaltCreateEmptyControl();
  for (ctrl.numQueues = 0; ctrl.numQueues < ctrl.maxQueues; ctrl.numQueues++) {
    ctrl.queues[ctrl.numQueues] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  }
#elif Cobalt_BACKEND_HIP
  status = hipGetDeviceCount( &numDevices );
  device = 0;
  status = hipSetDevice( device );
  ctrl = cobaltCreateEmptyControl();
  for (ctrl.numQueues = 0; ctrl.numQueues < ctrl.maxQueues; ctrl.numQueues++) {
    status = hipStreamCreate( &ctrl.queues[ctrl.numQueues] );
  }
#endif

  // host control
  ctrlValidation = cobaltCreateEmptyControl();

  // reference device
  deviceProfileReference.numDevices = 1;
  std::sprintf(deviceProfileReference.devices[0].name, "cpu");
}

void destroyControls() {

#if Cobalt_BACKEND_OPENCL12
  for (unsigned int i = 0; i < ctrl.numQueues; i++) {
    clReleaseCommandQueue(ctrl.queues[i]);
  }
  clReleaseContext(context);
  for (unsigned int i = 0; i < numDevices; i++) {
    clReleaseDevice(devices[i]);
  }
  delete[] devices;
  delete[] platforms;
#elif Cobalt_BACKEND_HIP
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
    if (std::strcmp(arg, "--validate-kernels") == 0) {
      doValidationKernels = true;
    }
  }
}

bool cobaltDataTypeIsHalf(CobaltDataType dataType) {
  return dataType == cobaltDataTypeHalf
      || dataType == cobaltDataTypeComplexHalf
      || dataType == cobaltDataTypeComplexConjugateHalf;
}
bool cobaltDataTypeIsFloat(CobaltDataType dataType) {
  return dataType == cobaltDataTypeSingle
    || dataType == cobaltDataTypeComplexSingle
    || dataType == cobaltDataTypeComplexConjugateSingle;
}
bool cobaltDataTypeIsDouble(CobaltDataType dataType) {
  return dataType == cobaltDataTypeDouble
    || dataType == cobaltDataTypeComplexDouble
    || dataType == cobaltDataTypeComplexConjugateDouble;
}

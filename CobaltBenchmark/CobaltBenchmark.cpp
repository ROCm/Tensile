/*******************************************************************************
 * Cobalt Benchmark
 ******************************************************************************/
#define _CRTDBG_MAP_ALLOC

#ifdef _CRTDBG_MAP_ALLOC
#include <stdlib.h>
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

Cobalt::Tensor::FillType tensorFillTypeC = Cobalt::Tensor::fillTypeRandom;
Cobalt::Tensor::FillType tensorFillTypeA = Cobalt::Tensor::fillTypeRandom;
Cobalt::Tensor::FillType tensorFillTypeB = Cobalt::Tensor::fillTypeRandom;

//#define MAX_PROBLEMS 16
//#define MAX_SOLUTIONS_PER_PROBLEM 16

// alpha = 2
// beta = 2

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
  std::string logFilePath = CobaltBenchmark_DIR_SOLUTIONS;
  logFilePath += "/CobaltBenchmark_log.xml";
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
      printf("Difference[%llu] = %i\n", problemIdx-1, diff);
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
    CobaltScalarData alpha;
    alpha.data = isFloatAlpha ? alphaFloat.data : isDoubleAlpha ? alphaDouble.data : nullptr;
    CobaltScalarData beta;
    beta.data = isFloatBeta ? betaFloat.data : isDoubleBeta ? betaDouble.data : nullptr; 

    // re-initialize device input buffers
    clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataA.data), CL_FALSE, deviceTensorDataA.offset, sizeA, initialDataA, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataB.data), CL_FALSE, deviceTensorDataB.offset, sizeB, initialDataB, 0, nullptr, nullptr);

    // calculate reference C once for problem
    if (doValidation) {

      // get reference solution
      Cobalt::Solution *solutionReference;
      problem->pimpl->deviceProfile = deviceProfileReference;
      std::tie(solutionReference,cobaltStatus) = getSolutionCPU( *(problem->pimpl) );

      // re-initialize reference buffers
      memcpy(referenceTensorDataC.data, initialDataC, sizeC);
      referenceTensorDataA.data = initialDataA;
      referenceTensorDataB.data = initialDataB;

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
      printf("Status: Problem[%llu/%llu] %s\n", problemIdx, problemEndIdx, problem->pimpl->toString().c_str());
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
      printf("S[%llu/%llu] ", solutionIdx, solutionEndIdx);

      // get solution candidate
      Cobalt::Solution *solution = solutionCandidates[ solutionIdx ];

      // re-initialize device buffers
      clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataC.data), CL_FALSE, deviceTensorDataC.offset, sizeC, initialDataC, 0, nullptr, nullptr);
      clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataA.data), CL_FALSE, deviceTensorDataA.offset, sizeA, initialDataA, 0, nullptr, nullptr);
      clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataB.data), CL_FALSE, deviceTensorDataB.offset, sizeB, initialDataB, 0, nullptr, nullptr);
      clFinish(ctrl.queues[0]);

      // ensure kernels are compiled before timing
      // for validation ctrl.benchmark = 0; 1 call to enqueueEntry below
      ctrl.benchmark = 4; // 5;
      unsigned int numSamples = 1; // 4;
      if (doValidation) {
        ctrl.benchmark = 0;
        numSamples = 1;
      }
      for (unsigned int s = 0; s < numSamples; s++) {
        solution->enqueueEntry(
            deviceTensorDataC,
            deviceTensorDataA,
            deviceTensorDataB,
            alpha,
            beta,
            ctrl );
      }
      for (unsigned int i = 0; i < ctrl.numQueues; i++) {
        status = clFinish(ctrl.queues[i]);
        CL_CHECK(status)
      }

#if 0
      // peek at gpu result
      status = clEnqueueReadBuffer(ctrl.queues[0], (cl_mem)deviceTensorDataC.data, CL_TRUE, deviceTensorDataC.offset, sizeC, deviceTensorDataOnHostC.data, 0, nullptr, nullptr);
      CL_CHECK(status)
      status = clFinish(ctrl.queues[0]);
      CL_CHECK(status)

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
  // device tensor data; max sized; initial data get clWriteBuffer each time
  deviceTensorDataC.data = static_cast<void *>(clCreateBuffer(context, CL_MEM_READ_WRITE, tensorSizeMaxC, nullptr, &status));
  deviceTensorDataC.offset = 0;
  deviceTensorDataOnHostC.data = malloc(tensorSizeMaxC);
  deviceTensorDataOnHostC.offset = 0;
  deviceTensorDataA.data = static_cast<void *>(clCreateBuffer(context, CL_MEM_READ_ONLY, tensorSizeMaxA, nullptr, &status));
  deviceTensorDataA.offset = 0;
  deviceTensorDataOnHostA.data = malloc(tensorSizeMaxA);
  deviceTensorDataOnHostA.offset = 0;
  deviceTensorDataB.data = static_cast<void *>(clCreateBuffer(context, CL_MEM_READ_ONLY, tensorSizeMaxB, nullptr, &status));
  deviceTensorDataB.offset = 0;
  deviceTensorDataOnHostB.data = malloc(tensorSizeMaxB);
  deviceTensorDataOnHostB.offset = 0;

  // reference tensor data
  referenceTensorDataC.data = malloc(tensorSizeMaxC);
  referenceTensorDataC.offset = 0;
  referenceTensorDataA.data = nullptr;
  referenceTensorDataA.offset = 0;
  referenceTensorDataB.data = nullptr;
  referenceTensorDataB.offset = 0;

  // scalars
  alphaFloat.data = new float[2];
  static_cast<float *>(alphaFloat.data)[0] = 2.f;
  static_cast<float *>(alphaFloat.data)[1] = 2.f;
  //alphaFloat.dataType = cobaltDataTypeComplexSingle;
  betaFloat.data = new float[2];
  static_cast<float *>(betaFloat.data)[0] = 2.f;
  static_cast<float *>(betaFloat.data)[1] = 2.f;
  alphaDouble.data = new double[2];
  static_cast<double *>(alphaDouble.data)[0] = 2.0;
  static_cast<double *>(alphaDouble.data)[1] = 2.0;
  betaDouble.data = new double[2];
  static_cast<double *>(betaDouble.data)[0] = 2.0;
  static_cast<double *>(betaDouble.data)[1] = 2.0;
  printf("done.\n");
}

void destroyTensorData() {


  delete[] initialTensorDataFloatC.data;
  delete[] initialTensorDataFloatA.data;
  delete[] initialTensorDataFloatB.data;
  delete[] initialTensorDataDoubleC.data;
  delete[] initialTensorDataDoubleA.data;
  delete[] initialTensorDataDoubleB.data;

  delete[] deviceTensorDataOnHostC.data;
  delete[] deviceTensorDataOnHostA.data;
  delete[] deviceTensorDataOnHostB.data;
  delete[] referenceTensorDataC.data;


  clReleaseMemObject(static_cast<cl_mem>(deviceTensorDataC.data));
  clReleaseMemObject(static_cast<cl_mem>(deviceTensorDataA.data));
  clReleaseMemObject(static_cast<cl_mem>(deviceTensorDataB.data));


  delete[] alphaFloat.data;
  delete[] betaFloat.data;
  delete[] alphaDouble.data;
  delete[] betaDouble.data;
}

void fillTensor(CobaltTensor inputTensor, CobaltTensorData tensorData, Cobalt::Tensor::FillType fillType, void *src) {
  Cobalt::Tensor tensor(inputTensor);
  tensor.fill(tensorData, fillType, src);
}

void initControls() {
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

  // host control
  ctrlValidation = cobaltCreateEmptyControl();

  // reference device
  deviceProfileReference.numDevices = 1;
  sprintf_s(deviceProfileReference.devices[0].name, "cpu");
}

void destroyControls() {
  for (unsigned int i = 0; i < ctrl.numQueues; i++) {
    clReleaseCommandQueue(ctrl.queues[i]);
  }
  clReleaseContext(context);

  for (unsigned int i = 0; i < numDevices; i++) {
    clReleaseDevice(devices[i]);
  }
  delete[] devices;
  delete[] platforms;
}

void parseCommandLineOptions(int argc, char *argv[]) {
  doValidation = false;
  doValidationKernels = false;
  for (int argIdx = 0; argIdx < argc; argIdx++) {
    char *arg = argv[argIdx];
    if (strcmp(arg, "--validate") == 0) {
      doValidation = true;
    }
    if (strcmp(arg, "--validate-kernels") == 0) {
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
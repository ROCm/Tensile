/*******************************************************************************
 * Cobalt Benchmark
 ******************************************************************************/

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
// alpha = 2
// beta = 2

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {

  // parse commandline options
  parseCommandLineOptions(argc, argv);

  // setup CobaltLib
  std::string logFilePath = Cobalt_DIR_SOLUTIONS;
  logFilePath += "/CobaltBenchmark_log.xml";
  cobaltSetup(logFilePath.c_str());

  // create CobaltControl
  initControls();

  // initialize initial buffer values for validation
  initTensorData();
  
  size_t problemStartIdx = 0;
  size_t problemEndIdx = numProblems;

  // for each problem
  for ( size_t problemIdx = problemStartIdx; problemIdx < problemEndIdx;
      problemIdx++ ) {

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
    clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataA.data), CL_TRUE, deviceTensorDataA.offset, sizeA, initialDataA, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataB.data), CL_TRUE, deviceTensorDataB.offset, sizeB, initialDataB, 0, nullptr, nullptr);

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

    }


    size_t solutionStartIdx = 0;
    size_t solutionEndIdx = solutionCandidates.size();
    for ( size_t solutionIdx = solutionStartIdx; solutionIdx < solutionEndIdx;
        solutionIdx++ ) {

      // get solution candidate
      Cobalt::Solution *solution = solutionCandidates[ solutionIdx ];

      // re-initialize device C buffers
      clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataC.data), CL_TRUE, deviceTensorDataC.offset, sizeC, initialDataC, 0, nullptr, nullptr);
      clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataA.data), CL_TRUE, deviceTensorDataA.offset, sizeA, initialDataA, 0, nullptr, nullptr);
      clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataB.data), CL_TRUE, deviceTensorDataB.offset, sizeB, initialDataB, 0, nullptr, nullptr);
      clFinish(ctrl.queues[0]);

      // ensure kernels are compiled before timing
      ctrl.benchmark = 1;
      solution->enqueueEntry(
          deviceTensorDataC,
          deviceTensorDataA,
          deviceTensorDataB,
          alpha,
          beta,
          ctrl );

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
    cobaltDestroyProblem( &problem );
    
  } // problem loop
  cobaltTeardown();
  return 0;
} // end main

void initTensorData() {
  printf("Status: Initializing tensor data %.1f MB ...", (tensorSizeMaxC + tensorSizeMaxA + tensorSizeMaxB)/(1024.f*1024.f) );
  // dummy tensors for filling initial data
  initialTensorFloatC.dataType = cobaltDataTypeSingle;
  initialTensorFloatC.numDimensions = 1;
  initialTensorFloatC.dimensions[0].stride = 1;
  initialTensorFloatC.dimensions[0].size = tensorSizeMaxC / 4 /*bytes per float*/;
  initialTensorFloatA.dataType = cobaltDataTypeSingle;
  initialTensorFloatA.numDimensions = 1;
  initialTensorFloatA.dimensions[0].stride = 1;
  initialTensorFloatA.dimensions[0].size = tensorSizeMaxA / 4 /*bytes per float*/;
  initialTensorFloatB.dataType = cobaltDataTypeSingle;
  initialTensorFloatB.numDimensions = 1;
  initialTensorFloatB.dimensions[0].stride = 1;
  initialTensorFloatB.dimensions[0].size = tensorSizeMaxB / 4 /*bytes per float*/;

  initialTensorDoubleC.dataType = cobaltDataTypeDouble;
  initialTensorDoubleC.numDimensions = 1;
  initialTensorDoubleC.dimensions[0].stride = 1;
  initialTensorDoubleC.dimensions[0].size = tensorSizeMaxC / 8 /*bytes per double*/;
  initialTensorDoubleA.dataType = cobaltDataTypeDouble;
  initialTensorDoubleA.numDimensions = 1;
  initialTensorDoubleA.dimensions[0].stride = 1;
  initialTensorDoubleA.dimensions[0].size = tensorSizeMaxA / 8 /*bytes per double*/;
  initialTensorDoubleB.dataType = cobaltDataTypeDouble;
  initialTensorDoubleB.numDimensions = 1;
  initialTensorDoubleB.dimensions[0].stride = 1;
  initialTensorDoubleB.dimensions[0].size = tensorSizeMaxB / 8 /*bytes per double*/;

  // initial tensor data for host buffers
  initialTensorDataFloatC.data = new float[initialTensorFloatC.dimensions[0].size];
  initialTensorDataFloatA.data = new float[initialTensorFloatA.dimensions[0].size];
  initialTensorDataFloatB.data = new float[initialTensorFloatB.dimensions[0].size];
  initialTensorDataDoubleC.data = new double[initialTensorDoubleC.dimensions[0].size];
  initialTensorDataDoubleA.data = new double[initialTensorDoubleA.dimensions[0].size];
  initialTensorDataDoubleB.data = new double[initialTensorDoubleB.dimensions[0].size];
  initialTensorDataFloatC.offset = 0;
  initialTensorDataFloatA.offset = 0;
  initialTensorDataFloatB.offset = 0;
  initialTensorDataDoubleC.offset = 0;
  initialTensorDataDoubleA.offset = 0;
  initialTensorDataDoubleB.offset = 0;

  fillTensor( initialTensorFloatC, initialTensorDataFloatC, tensorFillTypeC, nullptr);
  fillTensor( initialTensorFloatA, initialTensorDataFloatA, tensorFillTypeA, nullptr);
  fillTensor( initialTensorFloatB, initialTensorDataFloatB, tensorFillTypeB, nullptr);
  fillTensor( initialTensorDoubleC, initialTensorDataDoubleC, tensorFillTypeC, nullptr);
  fillTensor( initialTensorDoubleA, initialTensorDataDoubleA, tensorFillTypeA, nullptr);
  fillTensor( initialTensorDoubleB, initialTensorDataDoubleB, tensorFillTypeB, nullptr);

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

void parseCommandLineOptions(int argc, char *argv[]) {
  doValidation = false;
  doValidationKernels = false;
  for (int argIdx = 0; argIdx < argc; argIdx++) {
    char *arg = argv[argIdx];
    printf(arg);
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
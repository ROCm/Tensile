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

#if 0

// maximums host buffers
initialHostC float and double
initialHostA float and double
initialHostB float and double

cl_mem initialDeviceA = copy initialHostA
cl_mem initialDeviceB = copy initialHostB


for problems
  solve problem on cpu
    cpuC = initialC
  for solutions
    solve problem on gpu for validation
      devC = clMemcopy initialHostC
    benchmark solution

#endif

Cobalt::Tensor::FillType tensorFillTypeC = Cobalt::Tensor::fillTypeRandom;
Cobalt::Tensor::FillType tensorFillTypeA = Cobalt::Tensor::fillTypeRandom;
Cobalt::Tensor::FillType tensorFillTypeB = Cobalt::Tensor::fillTypeRandom;


/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {

  // parse commandline options
  parseCommandLineOptions(argc, argv);

  // setup CobaltLib
  cobaltSetup("CobaltBenchmark");

  // create CobaltControl
  initControls();

  // initialize initial buffer values for validation
  initTensorData();

  // initialize Candidates
  printf("Status: Initializing solution candidates...");
  initializeSolutionCandidates();
  printf("done.\n");


  size_t problemStartIdx = 0;
  size_t problemEndIdx = numProblems;
  size_t solutionStartIdx = 0;
  size_t solutionEndIdx = 0;

  // for each problem
  for ( size_t problemIdx = problemStartIdx; problemIdx < problemEndIdx;
      problemIdx++ ) {

    // info about problem
    CobaltProblem problemReference = problems[problemIdx];
    bool isFloatC = cobaltDataTypeIsFloat(problemReference->pimpl->getDataTypeC());
    bool isFloatA = cobaltDataTypeIsFloat(problemReference->pimpl->getDataTypeA());
    bool isFloatB = cobaltDataTypeIsFloat(problemReference->pimpl->getDataTypeB());
    bool isFloatAlpha = cobaltDataTypeIsFloat(problemReference->pimpl->getDataTypeAlpha());
    bool isFloatBeta = cobaltDataTypeIsFloat(problemReference->pimpl->getDataTypeBeta());
    size_t sizeC = problemReference->pimpl->tensorC.numBytes();
    size_t sizeA = problemReference->pimpl->tensorA.numBytes();
    size_t sizeB = problemReference->pimpl->tensorB.numBytes();
    void *initialDataC = isFloatC ? initialTensorDataFloatC.data : initialTensorDataDoubleC.data;
    void *initialDataA = isFloatA ? initialTensorDataFloatA.data : initialTensorDataDoubleA.data;
    void *initialDataB = isFloatB ? initialTensorDataFloatB.data : initialTensorDataDoubleB.data;
    CobaltScalarData alpha;
    alpha.data = isFloatAlpha ? alphaFloat.data : alphaDouble.data;
    alpha.dataType = problemReference->pimpl->getDataTypeAlpha();
    CobaltScalarData beta;
    beta.data = isFloatBeta ? betaFloat.data : betaDouble.data;
    beta.dataType = problemReference->pimpl->getDataTypeBeta();

    //if (isFloatC) {
    //  float tmpC[1024];
    //  memcpy(tmpC, initialDataC, 1024*sizeof(float));
    //  printf("floatC[0]=%f\n", tmpC[0]);
    //} else {
    //  double tmpC[1024];
    //  memcpy(tmpC, initialDataC, 1024 * sizeof(double));
    //  printf("doubleC[0]=%f\n", tmpC[0]);
    //}
    //if (isFloatA) {
    //  float tmpA[1024];
    //  memcpy(tmpA, initialDataA, 1024 * sizeof(float));
    //  printf("floatA[0]=%f\n", tmpA[0]);
    //} else {
    //  double tmpA[1024];
    //  memcpy(tmpA, initialDataA, 1024 * sizeof(double));
    //  printf("doubleA[0]=%f\n", tmpA[0]);
    //}
    //if (isFloatB) {
    //  float tmpB[1024];
    //  memcpy(tmpB, initialDataB, 1024 * sizeof(float));
    //  printf("floatB[0]=%f\n", tmpB[0]);
    //} else {
    //  double tmpB[1024];
    //  memcpy(tmpB, initialDataB, 1024 * sizeof(double));
    //  printf("doubleB[0]=%f\n", tmpB[0]);
    //}
    //
    //if (isFloatBeta) {
    //  float tmpBeta[2];
    //  memcpy(tmpBeta, beta.data, 2 * sizeof(float));
    //  printf("beta={%f, %f}\n", tmpBeta[0], tmpBeta[1]);
    //}
    //else {
    //  double tmpBeta[2];
    //  memcpy(tmpBeta, beta.data, 2 * sizeof(double));
    //  printf("beta={%f, %f}\n", tmpBeta[0], tmpBeta[1]);
    //}

    // re-initialize device input buffers
    clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataA.data), CL_TRUE, deviceTensorDataA.offset, sizeA, initialDataA, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataB.data), CL_TRUE, deviceTensorDataB.offset, sizeB, initialDataB, 0, nullptr, nullptr);

    // calculate reference C once for problem
    if (doValidation) {

      // get reference solution
      Cobalt::Solution *solutionReference;
      problemReference->pimpl->deviceProfile = deviceProfileReference;
      std::tie(solutionReference,cobaltStatus) = getSolutionCPU( *(problemReference->pimpl) );

      // re-initialize reference buffers
      memcpy(referenceTensorDataC.data, initialDataC, sizeC);
      referenceTensorDataA.data = initialDataA;
      referenceTensorDataB.data = initialDataB;
      //memcpy(referenceTensorDataA.data, initialDataA, sizeA);
      //memcpy(referenceTensorDataB.data, initialDataB, sizeB);

      // enqueue reference solution
      printf("Status: Enqueueing reference for %s ...", problemReference->pimpl->toString().c_str());
      solutionReference->enqueue(
        referenceTensorDataC,
        referenceTensorDataA,
        referenceTensorDataB,
        alpha,
        beta,
        ctrlValidation );
      printf("done.\n");

      //printf("\nTensorA:\n");
      //printf( problemReference->pimpl->tensorA.toString(referenceTensorDataA).c_str() );
      //printf("\nTensorB:\n");
      //printf( problemReference->pimpl->tensorB.toString(referenceTensorDataB).c_str() );
      //printf("\nTensorC-CPU:\n");
      //printf( problemReference->pimpl->tensorC.toString(referenceTensorDataC).c_str() );
    }




    solutionEndIdx += numSolutionsPerProblem[problemIdx];
    for ( size_t solutionIdx = solutionStartIdx; solutionIdx < solutionEndIdx;
        solutionIdx++ ) {

      // get solution candidate
      Cobalt::Solution *solution = solutionCandidates[ solutionIdx ];

      // re-initialize device C buffers
      clEnqueueWriteBuffer(ctrl.queues[0], static_cast<cl_mem>(deviceTensorDataC.data), CL_TRUE, deviceTensorDataC.offset, sizeC, initialDataC, 0, nullptr, nullptr);


      // ensure kernels are compiled before timing
      solution->enqueue(
          deviceTensorDataC,
          deviceTensorDataA,
          deviceTensorDataB,
          alpha,
          beta,
          ctrl );

      // validate before enqueueing multiple times
      if (doValidation) {
        // get gpu result
        for ( unsigned int q = 0; q < ctrl.numQueues; q++) {
          status = clFinish( ctrl.queues[q] );
          CL_CHECK(status)
        }

        // print tensorC-gpu
        status = clEnqueueReadBuffer(ctrl.queues[0], (cl_mem)deviceTensorDataC.data, CL_TRUE, 0, sizeC,
            deviceTensorDataOnHostC.data, 0, nullptr, nullptr );
        CL_CHECK(status)
        status = clFinish(ctrl.queues[0]);
        CL_CHECK(status)

        // print cpu in tensor form
        // printf("\nTensorC-CPU:\n");
        // printf(problemReference->pimpl->tensorC.toString(referenceTensorDataC).c_str());

        // print gpu in tensor form
        // printf("\nTensorC-GPU:\n");
        // printf( problemReference->pimpl->tensorC.toString(deviceTensorDataOnHostC).c_str() );
        
        //printf("\nComparing...\n");
        bool equal = compareTensors(deviceTensorDataOnHostC, referenceTensorDataC, problemReference->pimpl->tensorC, ctrl );
        //printf("\nDone Comparing.\n");
        printf("P[%04llu] S[%03llu] %7s - %s\n", problemIdx, solutionIdx-solutionStartIdx, equal?"PASSED":"FAILED", solution->toString(0).c_str() );
        if (!equal) {
          printf("Oops!\n");
        }
      }

      // time solution
      double time = timeSolution( solution, deviceTensorDataC, deviceTensorDataA, deviceTensorDataB, alpha, beta, ctrl );
      //printf("P[%04llu] S[%03llu] %7.3f - %s\n", problemIdx, solutionIdx-solutionStartIdx, time, solution->toString(0).c_str() );
      // write time to result xml file

    } // solution loop

    solutionStartIdx = solutionEndIdx;
    
  } // problem loop
  cobaltTeardown();
  return 0;
} // end main

void initTensorData() {
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
  deviceTensorDataB.data = static_cast<void *>(clCreateBuffer(context, CL_MEM_READ_ONLY, tensorSizeMaxB, nullptr, &status));
  deviceTensorDataB.offset = 0;

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
  for (ctrl.numQueues = 0; ctrl.numQueues < ctrl.maxQueues; ctrl.numQueues++) {
    ctrl.queues[ctrl.numQueues] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  }
  ctrl.inputEvents = nullptr;
  ctrl.numInputEvents = 0;
  ctrl.numDependencies = 0;
  ctrl.numOutputEvents = 0;
  ctrl.outputEvents = nullptr;

  // host control
  ctrlValidation.inputEvents = nullptr;
  ctrlValidation.numInputEvents = 0;
  ctrlValidation.numDependencies = 0;
  ctrlValidation.numOutputEvents = 0;
  ctrlValidation.outputEvents = nullptr;
  ctrlValidation.numQueues = 0;

  // reference device
  deviceProfileReference.numDevices = 1;
  sprintf_s(deviceProfileReference.devices[0].name, "cpu");
}

 /*******************************************************************************
  * Compare Tensors
  ******************************************************************************/
bool compareTensors(
  CobaltTensorData gpu,
  CobaltTensorData cpu,
  Cobalt::Tensor tensor,
  CobaltControl ctrl) {

  switch (tensor.getDataType()) {
  case cobaltDataTypeSingle:
    return compareTensorsTemplate<float>((float *)gpu.data, (float *)cpu.data, tensor);
  case cobaltDataTypeDouble:
    return compareTensorsTemplate<double>((double *)gpu.data, (double *)cpu.data, tensor);
  case cobaltDataTypeComplexSingle:
    return compareTensorsTemplate<CobaltComplexFloat>((CobaltComplexFloat *)gpu.data, (CobaltComplexFloat *)cpu.data, tensor);
  case cobaltDataTypeComplexDouble:
    return compareTensorsTemplate<CobaltComplexDouble>((CobaltComplexDouble *)gpu.data, (CobaltComplexDouble *)cpu.data, tensor);
  default:
    printf("ERROR\n");
    return false;
  }
}
template<typename DataType>
bool compareTensorsTemplate(
  DataType *gpuData,
  DataType *cpuData,
  Cobalt::Tensor tensor) {
  unsigned int maxToPrint = 4;
  unsigned int printCount = 0;
  bool equal = true;

  std::vector<unsigned int> coords(tensor.numDims());
  for (unsigned int i = 0; i < tensor.numDims(); i++) {
    coords[i] = 0;
  }
  bool done = false;

  while (!done) { // exit criteria specified below

    for (coords[0] = 0; coords[0] < tensor[0].size; coords[0]++) {
      size_t index = tensor.getIndex(coords);
      if (!(Cobalt::almostEqual(cpuData[index], gpuData[index]))) {
        equal = false;
        if (printCount < maxToPrint) {
          printMismatch<DataType>(index, gpuData[index], cpuData[index]);
          printCount++;
        } else {
          done = true;
          break;
        }
      } /*else {
        if (printCount < maxToPrint) {
          printMatch<DataType>(index, gpuData[index], cpuData[index]);
          printCount++;
        } else {
          break;
        }
      }*/
    } // d0

    // if 1-dimensional done
    if (coords.size() == 1) {
      done = true;
      break;
    } else { // 2+ dimensions
      bool dimIncremented = false; // for printing extra line
                                   // increment coord
      coords[1]++;
      for (unsigned int d = 1; d < tensor.numDims(); d++) {
        if (coords[d] >= tensor[d].size) {
          if (d == tensor.numDims() - 1) {
            // exit criteria - last dimension full
            done = true;
            break;
          }
          dimIncremented = true;
          coords[d] = 0;
          coords[d + 1]++;
        }
      }
    }
  }

  return equal;
}


/*******************************************************************************
* timeSolution - milliseconds
******************************************************************************/
double timeSolution(
  Cobalt::Solution *solution,
  CobaltTensorData tensorDataC,
  CobaltTensorData tensorDataA,
  CobaltTensorData tensorDataB,
  CobaltScalarData alpha,
  CobaltScalarData beta,
  CobaltControl &ctrl) {

  size_t numEnqueuesPerSample = 1;
  const size_t numSamples = 1;

  double sampleTimes[numSamples];
  Cobalt::Timer timer;

  for (size_t sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {

    // start timer
    timer.start();
    for (size_t i = 0; i < numEnqueuesPerSample; i++) {
      solution->enqueue(tensorDataC, tensorDataA, tensorDataB, alpha, beta, ctrl);
    }
    // wait for queue
    for (size_t i = 0; i < ctrl.numQueues; i++) {
      clFinish(ctrl.queues[i]);
    }
    // stop timer
    double time = timer.elapsed_ms();
    //printf("%f\n", time);
    sampleTimes[sampleIdx] = time;
  } // samples

    // for median, selection sort and take middle
  for (size_t i = 0; i < numSamples; i++) {
    size_t fastestIdx = i;
    for (size_t j = i + 1; j < numSamples; j++) {
      if (sampleTimes[j] < sampleTimes[fastestIdx]) {
        fastestIdx = j;
      }
    }
    // swap i and fastest
    double tmp = sampleTimes[i];
    sampleTimes[i] = sampleTimes[fastestIdx];
    sampleTimes[fastestIdx] = tmp;
  }
  return sampleTimes[numSamples / 2];
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


 /*******************************************************************************
  * Print Match / Mismatch
  ******************************************************************************/
template<>
void printMismatch<float>(size_t index, float gpuData, float cpuData) {
  printf("%5llu: %.6f != %.6f\n", index, gpuData, cpuData);
}
template<>
void printMismatch<double>(size_t index, double gpuData, double cpuData) {
  printf("%6llu: %.6f != %.6f\n", index, gpuData, cpuData);
}
template<>
void printMismatch<CobaltComplexFloat>(size_t index, CobaltComplexFloat gpuData, CobaltComplexFloat cpuData) {
  printf("%6llu: %.6f, %.6f != %.6f, %.6f\n", index, gpuData.s[0], gpuData.s[1], cpuData.s[0], cpuData.s[1]);
}
template<>
void printMismatch<CobaltComplexDouble>(size_t index, CobaltComplexDouble gpuData, CobaltComplexDouble cpuData) {
  printf("%6llu: %.6f, %.6f != %.6f, %.6f\n", index, gpuData.s[0], gpuData.s[1], cpuData.s[0], cpuData.s[1]);
}
template<>
void printMatch<float>(size_t index, float gpuData, float cpuData) {
  printf("%5llu: %.6f == %.6f\n", index, gpuData, cpuData);
}
template<>
void printMatch<double>(size_t index, double gpuData, double cpuData) {
  printf("%6llu: %.6f == %.6f\n", index, gpuData, cpuData);
}
template<>
void printMatch<CobaltComplexFloat>(size_t index, CobaltComplexFloat gpuData, CobaltComplexFloat cpuData) {
  printf("%6llu: %.6f, %.6f == %.6f, %.6f\n", index, gpuData.s[0], gpuData.s[1], cpuData.s[0], cpuData.s[1]);
}
template<>
void printMatch<CobaltComplexDouble>(size_t index, CobaltComplexDouble gpuData, CobaltComplexDouble cpuData) {
  printf("%6llu: %.6f, %.6f == %.6f, %.6f\n", index, gpuData.s[0], gpuData.s[1], cpuData.s[0], cpuData.s[1]);
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
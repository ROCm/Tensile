/*******************************************************************************
 * Cobalt Benchmark
 ******************************************************************************/

#include "Cobalt.h"
#include "Tools.h"
#include "Solution.h"
#include "CobaltSolutionCandidates.h"
#include "SolutionTensorContractionCPU.h"
#include "StructOperations.h"

#include <tuple>


template<typename DataType>
void printMismatch( size_t index, DataType gpuData, DataType cpuData );

template<>
void printMismatch<float>( size_t index, float gpuData, float cpuData ) {
  printf("%5zi: %.6f != %.6f\n", index, gpuData, cpuData );
}
template<>
void printMismatch<double>( size_t index, double gpuData, double cpuData ) {
  printf("%6zi: %.6f != %.6f\n", index, gpuData, cpuData );
}
template<>
void printMismatch<CobaltComplexFloat>( size_t index, CobaltComplexFloat gpuData, CobaltComplexFloat cpuData ) {
  printf("%6zi: %.6f, %.6f != %.6f, %.6f\n", index, gpuData.s0, gpuData.s1, cpuData.s0, cpuData.s1 );
}
template<>
void printMismatch<CobaltComplexDouble>( size_t index, CobaltComplexDouble gpuData, CobaltComplexDouble cpuData ) {
  printf("%6zi: %.6f, %.6f != %.6f, %.6f\n", index, gpuData.s0, gpuData.s1, cpuData.s0, cpuData.s1 );
}

template<typename DataType>
void printMatch(size_t index, DataType gpuData, DataType cpuData);

template<>
void printMatch<float>(size_t index, float gpuData, float cpuData) {
  printf("%5zi: %.6f == %.6f\n", index, gpuData, cpuData);
}
template<>
void printMatch<double>(size_t index, double gpuData, double cpuData) {
  printf("%6zi: %.6f == %.6f\n", index, gpuData, cpuData);
}
template<>
void printMatch<CobaltComplexFloat>(size_t index, CobaltComplexFloat gpuData, CobaltComplexFloat cpuData) {
  printf("%6zi: %.6f, %.6f == %.6f, %.6f\n", index, gpuData.s0, gpuData.s1, cpuData.s0, cpuData.s1);
}
template<>
void printMatch<CobaltComplexDouble>(size_t index, CobaltComplexDouble gpuData, CobaltComplexDouble cpuData) {
  printf("%6zi: %.6f, %.6f == %.6f, %.6f\n", index, gpuData.s0, gpuData.s1, cpuData.s0, cpuData.s1);
}

template<typename DataType>
bool compareTensorsTemplate(
  DataType *gpuData,
  DataType *cpuData,
  Cobalt::Tensor tensor) {
  
  unsigned int maxToPrint = 64;
  unsigned int printCount = 0;
  bool equal = true;
  for ( unsigned long long i = 0; i < tensor.numElements(); i++) {
    if ( !(cpuData[i] == gpuData[i]) ) {
      equal = false;
      if (printCount < maxToPrint) {
        printMismatch<DataType>(i, gpuData[i], cpuData[i]);
        printCount++;
      } else {
        break;
      }
    }
    else {
      printMatch<DataType>(i, gpuData[i], cpuData[i]);
    }
  }
  return equal;
}


bool compareTensors(
    CobaltTensorData gpu,
    CobaltTensorData cpu,
    Cobalt::Tensor tensor,
    CobaltControl ctrl ) {

  switch( tensor.getDataType() ) {
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

  size_t numEnqueuesPerSample = 6;
  const size_t numSamples = 5;

  double sampleTimes[numSamples];
  Cobalt::Timer timer;

  for ( size_t sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {

    // start timer
    timer.start();
    for (size_t i = 0; i < numEnqueuesPerSample; i++) {
      solution->enqueue( tensorDataC, tensorDataA, tensorDataB, alpha, beta, ctrl );
    }
    // wait for queue
    for (size_t i = 0; i < ctrl.numQueues; i++) {
      clFinish( ctrl.queues[i] );
    }
    // stop timer
    double time = timer.elapsed_ms();
    //printf("%f\n", time);
    sampleTimes[sampleIdx] = time;
  } // samples

  // for median, selection sort and take middle
  for (size_t i = 0; i < numSamples; i++) {
    size_t fastestIdx = i;
    for (size_t j = i+1; j < numSamples; j++) {
      if (sampleTimes[j] < sampleTimes[fastestIdx]) {
        fastestIdx = j;
      }
    }
    // swap i and fastest
    double tmp = sampleTimes[i];
    sampleTimes[i] = sampleTimes[fastestIdx];
    sampleTimes[fastestIdx] = tmp;
  }
  return sampleTimes[ numSamples/2 ];
}

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {

  bool doValidation = false;
  bool doValidationKernels = false;
  for ( int argIdx = 0; argIdx < argc; argIdx++) {
    char *arg = argv[argIdx];
    printf(arg);
    if ( strcmp(arg, "--validate")==0 ) {
      doValidation = true;
    }
    if (strcmp( arg, "--validate-kernels")==0 ) {
      doValidationKernels = true;
    }
  }

  // setup Cobalt
  cobaltSetup("CobaltBenchmark");

  // creat CobaltControl
  CobaltStatus cobaltStatus;
  CobaltControl ctrl;
  CobaltControl ctrlValidation;
  CobaltTensorData tensorDataC;
  CobaltTensorData tensorDataA;
  CobaltTensorData tensorDataB;

  // setup opencl
  cl_int status;
  cl_uint numPlatforms;
  status = clGetPlatformIDs( 0, nullptr, &numPlatforms );
  cl_platform_id *platforms = new cl_platform_id[ numPlatforms ];
  status = clGetPlatformIDs( numPlatforms, platforms, nullptr );
  cl_platform_id platform = platforms[0];
  cl_uint numDevices;
  status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices );
  cl_device_id *devices = new cl_device_id[ numDevices ];
  status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr );
  cl_device_id device = devices[0];
  cl_context context = clCreateContext( nullptr, 1, &device, nullptr, nullptr, &status );
  for ( ctrl.numQueues = 0; ctrl.numQueues < ctrl.maxQueues; ctrl.numQueues++) {
    ctrl.queues[ctrl.numQueues] = clCreateCommandQueue( context, device, CL_QUEUE_PROFILING_ENABLE, &status );
  }
  ctrl.inputEvents = nullptr;
  ctrl.numInputEvents = 0;
  ctrl.numDependencies = 0;
  ctrl.numOutputEvents = 0;
  ctrl.outputEvents = nullptr;

  // create buffers opencl
  float *tensorDataHostC = new float[tensorSizeMaxC];
  float *tensorDataHostA = new float[tensorSizeMaxA];
  float *tensorDataHostB = new float[tensorSizeMaxB];
  for (size_t i = 0; i < tensorSizeMaxC; i++) tensorDataHostC[i] = 0.f;
  for (size_t i = 0; i < tensorSizeMaxA; i++) tensorDataHostA[i] = (float) (i+1);
  for (size_t i = 0; i < tensorSizeMaxB; i++) tensorDataHostB[i] = (float) (i+1);
  tensorDataC.data = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, tensorSizeMaxC, tensorDataHostC, &status );
  tensorDataC.offset = 0;
  tensorDataA.data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorSizeMaxA, tensorDataHostA, &status );
  tensorDataA.offset = 0;
  tensorDataB.data = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, tensorSizeMaxB, tensorDataHostB, &status );
  tensorDataB.offset = 0;

  // create buffers cpu validation
  float *tensorDataHostValidationC = nullptr;
  CobaltTensorData tensorDataValidationC;
  CobaltTensorData tensorDataValidationA;
  CobaltTensorData tensorDataValidationB;
  if (doValidation) {
    tensorDataHostValidationC = new float[tensorSizeMaxC];
    tensorDataValidationC.data = tensorDataHostValidationC;
    tensorDataValidationC.offset = 0;
    tensorDataValidationA.data = tensorDataHostA;
    tensorDataValidationA.offset = 0;
    tensorDataValidationB.data = tensorDataHostB;
    tensorDataValidationB.offset = 0;
    ctrlValidation.inputEvents = nullptr;
    ctrlValidation.numInputEvents = 0;
    ctrlValidation.numDependencies = 0;
    ctrlValidation.numOutputEvents = 0;
    ctrlValidation.outputEvents = nullptr;
    ctrlValidation.numQueues = 0;
    
  }

  // scalars
  CobaltScalarData alpha;
  float alphaArray[] = { 2.0, 3.0, 4.0, 5.0 };
  alpha.data = &alphaArray[0];
  alpha.dataType = cobaltDataTypeSingle;
  CobaltScalarData beta;
  float betaArray[] = { 6.0, 7.0, 8.0, 9.0 };
  beta.data = &betaArray[0];
  beta.dataType = cobaltDataTypeSingle;

  // initialize Candidates
  initializeSolutionCandidates();

  // reference device
  CobaltDeviceProfile deviceProfileReference;
  deviceProfileReference.numDevices = 1;
  sprintf_s(deviceProfileReference.devices[0].name, "cpu" );

  size_t problemStartIdx = 0;
  size_t problemEndIdx = numProblems;
  size_t solutionStartIdx = 0;
  size_t solutionEndIdx = 0;

  // for each problem
  for ( size_t problemIdx = problemStartIdx; problemIdx < problemEndIdx;
      problemIdx++ ) {
        
    CobaltProblem problemReference;
    if (doValidation) {
      printf("doing validation\n");
      // get cpu result
      Cobalt::Solution *solutionReference;
      problemReference = problems[problemIdx];
      printf( problemReference->pimpl->toString().c_str() );

      problemReference->pimpl->deviceProfile = deviceProfileReference;
      std::tie(solutionReference,cobaltStatus) = getSolutionCPU( *(problemReference->pimpl) );
      solutionReference->enqueue( tensorDataValidationC, tensorDataValidationA,
          tensorDataValidationB, alpha, beta, ctrlValidation );

      // print tensorA
      printf("\nTensorA:\n");
      printf( problemReference->pimpl->tensorA.toString(tensorDataValidationA).c_str() );

      // print tensorB
      printf("\nTensorB:\n");
      printf( problemReference->pimpl->tensorB.toString(tensorDataValidationB).c_str() );

      // print tensorC-cpu
      printf("\nTensorC-CPU:\n");
      printf( problemReference->pimpl->tensorC.toString(tensorDataValidationC).c_str() );

    }




    solutionEndIdx += numSolutionsPerProblem[problemIdx];
    for ( size_t solutionIdx = solutionStartIdx; solutionIdx < solutionEndIdx;
        solutionIdx++ ) {

      // get solution candidate
      Cobalt::Solution *solution = solutionCandidates[ solutionIdx ];
      // ensure kernels are compiled before timing
      solution->enqueue( tensorDataC, tensorDataA, tensorDataB, alpha, beta, ctrl );

      // validate before enqueueing multiple times
      if (doValidation) {
        // get gpu result
        for ( unsigned int q = 0; q < ctrl.numQueues; q++) {
          status = clFinish( ctrl.queues[q] );
          CL_CHECK(status)
        }

        CobaltTensorData tmpC;
        tmpC.offset = 0;
        tmpC.data = new float[problemReference->pimpl->tensorC.numElements()];
        memset(tmpC.data, 0, problemReference->pimpl->tensorC.numElements());

        // print tensorC-gpu
        printf("\nTensorC-GPU:\n");
        status = clEnqueueReadBuffer(
          ctrl.queues[0],
          (cl_mem)tensorDataC.data,
          CL_TRUE,
          0,
          problemReference->pimpl->tensorC.numElements()*Cobalt::sizeOf(problemReference->pimpl->tensorC.getDataType()),
          tmpC.data,
          0, nullptr, nullptr );
        CL_CHECK(status)
        status = clFinish(ctrl.queues[0]);
        CL_CHECK(status)
        float *tmp = static_cast<float *>(tmpC.data);
        printf("%f, %f, %f, %f\n", tmp[0], tmp[1], tmp[2], tmp[3]);
        printf( problemReference->pimpl->tensorC.toString(tmpC).c_str() );
        
        printf("\nComparing...\n");
        compareTensors( tmpC, tensorDataValidationC, problemReference->pimpl->tensorC, ctrl );
        printf("\nDone Comparing.\n");
        delete[] tmpC.data;
      }

      // time solution
      double time = timeSolution( solution, tensorDataC, tensorDataA, tensorDataB, alpha, beta, ctrl );
      std::string solutionName = solution->toString(0);
      printf("P[%04zu] S[%03zu] %7.3f %s\n", problemIdx, solutionIdx-solutionStartIdx, time, solutionName.c_str() );
      // write time to result xml file

    } // solution loop

    solutionStartIdx = solutionEndIdx;
    
  } // problem loop
  cobaltTeardown();
  return 0;
}




/*******************************************************************************
 * Cobalt Benchmark
 ******************************************************************************/

#include "Cobalt.h"
#include "Tools.h"
#include "CobaltSolutionCandidates.h"

bool compareTensors(
  CobaltTensorData gpu,
  CobaltTensorData cpu,
  CobaltTensor tensor,
  CobaltControl ctrl ) {
  unsigned long long tensorSize = 0;
  for ( unsigned int d = 0; d < tensor.numDimensions; d++) {
    if (tensor.dimensions[d].size * tensor.dimensions[d].stride > tensorSize) {
      tensorSize = tensor.dimensions[d].size * tensor.dimensions[d].stride;
    }
  }
  float *gpuData = new float[tensorSize];
  clEnqueueReadBuffer( ctrl.queues[0], (cl_mem) gpu.data, true, 0, tensorSizeMaxC, gpuData, 0, nullptr, nullptr);
  float *cpuData = (float *)cpu.data;
  unsigned int maxToPrint = 0;
  unsigned int printCount = 0;
  bool equal = true;
  for ( unsigned long long i = 0; i < tensorSize; i++) {
    if (cpuData[i] != gpuData[i]) {
      equal = false;
      if (printCount < maxToPrint) {
        printf("%6i: %f != %f\n", cpuData[i], gpuData[i]);
        printCount++;
      } else {
        break;
      }
    }
  }
  return false;
}

/*******************************************************************************
 * timeSolution - milliseconds
 ******************************************************************************/
double timeSolution(
    CobaltSolutionBase *solution,
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl &ctrl) {

  size_t numEnqueuesPerSample = 6;
  const size_t numSamples = 5;

  double sampleTimes[numSamples];
  Timer timer;

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
  for (size_t i = 0; i < tensorSizeMaxA; i++) tensorDataHostA[i] = 1.f;
  for (size_t i = 0; i < tensorSizeMaxB; i++) tensorDataHostB[i] = 1.f;
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

    solutionEndIdx += numSolutionsPerProblem[problemIdx];
    for ( size_t solutionIdx = solutionStartIdx; solutionIdx < solutionEndIdx;
        solutionIdx++ ) {

      // get solution candidate
      CobaltSolutionBase *solution = solutionCandidates[ solutionIdx ];
      // ensure kernels are compiled before timing
      solution->enqueue( tensorDataC, tensorDataA, tensorDataB, alpha, beta, ctrl );

      // validate before enqueueing multiple times
      if (doValidation) {
        printf("doing validation\n");
        // get cpu result
        CobaltSolution *solutionReference;
        CobaltProblem problemReference = problems[problemIdx];
        problemReference.deviceProfile = deviceProfileReference;
        cobaltStatus = cobaltGetSolution( problemReference, &solutionReference );
        cobaltStatus = cobaltEnqueueSolution( solutionReference, tensorDataValidationC, tensorDataValidationA,
          tensorDataValidationB, alpha, beta, &ctrlValidation );
        // get gpu result
        for ( unsigned int q = 0; q < ctrl.numQueues; q++) {
          status = clFinish( ctrl.queues[q] );
        }
        compareTensors( tensorDataC, tensorDataValidationC, problemReference.tensorC, ctrl );
      }

      // time solution
      double time = timeSolution( solution, tensorDataC, tensorDataA, tensorDataB, alpha, beta, ctrl );
      std::string solutionName = solution->toString(0);
      printf("P[%04u] S[%03u] %7.3f %s\n", problemIdx, solutionIdx-solutionStartIdx, time, solutionName.c_str() );
      // write time to result xml file

    } // solution loop

    solutionStartIdx = solutionEndIdx;
    
  } // problem loop
  cobaltTeardown();
  return 0;
}




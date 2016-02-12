/*******************************************************************************
 * Cobalt Benchmark
 ******************************************************************************/

#include "Cobalt.h"
#include "Tools.h"
#include "CobaltSolutionCandidates.h"



/*******************************************************************************
 * timeSolution - milliseconds
 ******************************************************************************/
double timeSolution(
    CobaltSolution *solution,
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltControl &ctrl) {

  size_t numEnqueuesPerSample = 6;
  const size_t numSamples = 5;

  double sampleTimes[numSamples];
  Timer timer;

  for ( size_t sampleIdx = 0; sampleIdx < numSamples; sampleIdx++) {

    // start timer
    timer.start();
    for (size_t i = 0; i < numEnqueuesPerSample; i++) {
      solution->enqueue( tensorDataC, tensorDataA, tensorDataB, ctrl );
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
int main( void ) {

  // creat CobaltControl
  CobaltControl ctrl;
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

  // create buffers
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

  // initialize Candidates
  initializeSolutionCandidates();

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
      CobaltSolution *solution = solutionCandidates[ solutionIdx ];

      // time solution
      solution->enqueue( tensorDataC, tensorDataA, tensorDataB, ctrl );
      double time = timeSolution( solution, tensorDataC, tensorDataA, tensorDataB, ctrl );
      std::string solutionName = solution->toString(0);
      printf("P[%04u] S[%03u] %7.3f %s\n", problemIdx, solutionIdx-solutionStartIdx, time, solutionName.c_str() );
      // write time to result xml file

    } // solution loop

    solutionStartIdx = solutionEndIdx;
    
  } // problem loop

  return 0;
}




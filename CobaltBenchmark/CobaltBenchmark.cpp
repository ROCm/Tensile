/*******************************************************************************
 * Cobalt Benchmark
 ******************************************************************************/

#include "Cobalt.h"
#include "Tools.h"
#include "CobaltSolutionCandidates.h"



/*******************************************************************************
 * timeSolution
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
    // stop timer
    double time = timer.elapsed();
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
  tensorDataC.data = clCreateBuffer(context, CL_MEM_READ_WRITE, tensorSizeMaxC, nullptr, &status );
  tensorDataC.offset = 0;
  tensorDataA.data = clCreateBuffer(context, CL_MEM_READ_WRITE, tensorSizeMaxA, nullptr, &status );
  tensorDataA.offset = 0;
  tensorDataB.data = clCreateBuffer(context, CL_MEM_READ_WRITE, tensorSizeMaxB, nullptr, &status );
  tensorDataB.offset = 0;

  // initialize Candidates
  initializeSolutionCandidates();

  size_t problemStartIdx = 0;
  size_t problemEndIdx = numProblems;
  size_t solutionStartIdx = 0;
  size_t solutionEndIdx;

  // for each problem
  for ( size_t problemIdx = problemStartIdx; problemIdx < problemEndIdx;
      problemIdx++ ) {

    solutionEndIdx = numSolutionsPerProblem[problemIdx];
    for ( size_t solutionIdx = solutionStartIdx; solutionIdx < solutionEndIdx;
        solutionIdx++ ) {

      // get solution candidate
      CobaltSolution *solution = solutionCandidates[ solutionIdx ];

      // time solution
      timeSolution( solution, tensorDataC, tensorDataA, tensorDataB, ctrl );

      // write time to result xml file

    } // solution loop

    solutionStartIdx = solutionEndIdx;
    
  } // problem loop

  return 0;
}




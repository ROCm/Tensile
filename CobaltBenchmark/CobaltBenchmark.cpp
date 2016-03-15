/*******************************************************************************
 * Cobalt Benchmark
 ******************************************************************************/

#include "Cobalt.h"
#include "Tools.h"
#include "Solution.h"
#include "CobaltSolutionCandidates.h"
#include "SolutionTensorContractionCPU.h"
#include "StructOperations.h"
#include "MathTemplates.h"

#include <tuple>


template<typename DataType>
void printMismatch( size_t index, DataType gpuData, DataType cpuData );

template<>
void printMismatch<float>( size_t index, float gpuData, float cpuData ) {
  printf("%5llu: %.6f != %.6f\n", index, gpuData, cpuData );
}
template<>
void printMismatch<double>( size_t index, double gpuData, double cpuData ) {
  printf("%6llu: %.6f != %.6f\n", index, gpuData, cpuData );
}
template<>
void printMismatch<CobaltComplexFloat>( size_t index, CobaltComplexFloat gpuData, CobaltComplexFloat cpuData ) {
  printf("%6llu: %.6f, %.6f != %.6f, %.6f\n", index, gpuData.s[0], gpuData.s[1], cpuData.s[0], cpuData.s[1] );
}
template<>
void printMismatch<CobaltComplexDouble>( size_t index, CobaltComplexDouble gpuData, CobaltComplexDouble cpuData ) {
  printf("%6llu: %.6f, %.6f != %.6f, %.6f\n", index, gpuData.s[0], gpuData.s[1], cpuData.s[0], cpuData.s[1] );
}

template<typename DataType>
void printMatch(size_t index, DataType gpuData, DataType cpuData);

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

template<typename DataType>
bool compareTensorsTemplate(
  DataType *gpuData,
  DataType *cpuData,
  Cobalt::Tensor tensor) {
  // TODO - respect dimension strides so we don't read gaps
  unsigned int maxToPrint = 16;
  unsigned int printCount = 0;
  bool equal = true;
  for ( unsigned long long i = 0; i < tensor.numElements(); i++) {
    if ( !(almostEqual(cpuData[i], gpuData[i]) ) ) {
      equal = false;
      if (printCount < maxToPrint) {
        printMismatch<DataType>(i, gpuData[i], cpuData[i]);
        printCount++;
      } else {
        break;
      }
    } /*else {
      printMatch<DataType>(i, gpuData[i], cpuData[i]);
    }*/
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

  size_t numEnqueuesPerSample = 1;
  const size_t numSamples = 1;

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
  printf("Status: Initializing solution candidates...");
  initializeSolutionCandidates();
  printf("done.\n");

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

    // calculate reference C once for problem
    if (doValidation) {
      // get cpu result
      Cobalt::Solution *solutionReference;
      problemReference = problems[problemIdx];

      problemReference->pimpl->deviceProfile = deviceProfileReference;
      std::tie(solutionReference,cobaltStatus) = getSolutionCPU( *(problemReference->pimpl) );

      printf("Status: Enqueueing reference for %s ...", problemReference->pimpl->toString().c_str());
      solutionReference->enqueue( tensorDataValidationC, tensorDataValidationA,
          tensorDataValidationB, alpha, beta, ctrlValidation );
      printf("done.\n");

      // print tensorA
      //printf("\nTensorA:\n");
      //printf( problemReference->pimpl->tensorA.toString(tensorDataValidationA).c_str() );

      // print tensorB
      //printf("\nTensorB:\n");
      //printf( problemReference->pimpl->tensorB.toString(tensorDataValidationB).c_str() );

      // print tensorC-cpu
      //printf("\nTensorC-CPU:\n");
      //printf( problemReference->pimpl->tensorC.toString(tensorDataValidationC).c_str() );

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

        CobaltTensorData gpuTensorDataOnHostC;
        gpuTensorDataOnHostC.offset = 0;
        gpuTensorDataOnHostC.data;
        switch(problemReference->pimpl->tensorC.getDataType()) {
        case cobaltDataTypeSingle:
          gpuTensorDataOnHostC.data = new float[problemReference->pimpl->tensorC.numElements()];
          break;
        case cobaltDataTypeDouble:
          gpuTensorDataOnHostC.data = new double[problemReference->pimpl->tensorC.numElements()];
          break;
        case cobaltDataTypeComplexSingle:
        case cobaltDataTypeComplexConjugateSingle:
          gpuTensorDataOnHostC.data = new CobaltComplexFloat[problemReference->pimpl->tensorC.numElements()];
          break;
        case cobaltDataTypeComplexDouble:
        case cobaltDataTypeComplexConjugateDouble:
          gpuTensorDataOnHostC.data = new CobaltComplexDouble[problemReference->pimpl->tensorC.numElements()];
          break;
        }
        memset(gpuTensorDataOnHostC.data, 0, problemReference->pimpl->tensorC.numElements()*Cobalt::sizeOf(problemReference->pimpl->tensorC.getDataType()));

        // print tensorC-gpu
        status = clEnqueueReadBuffer(
          ctrl.queues[0],
          (cl_mem)tensorDataC.data,
          CL_TRUE,
          0,
          problemReference->pimpl->tensorC.numElements()*Cobalt::sizeOf(problemReference->pimpl->tensorC.getDataType()),
          gpuTensorDataOnHostC.data,
          0, nullptr, nullptr );
        CL_CHECK(status)
        status = clFinish(ctrl.queues[0]);
        CL_CHECK(status)
        //float *gpuDataC = static_cast<float *>(gpuTensorDataOnHostC.data);
        //float *cpuDataC = static_cast<float *>(tensorDataValidationC.data);
        //printf("\nTensorC-GPU:\n");
        //printf(problemReference->pimpl->toString().c_str());
        //printf("\n");
        // print raw gpu buffer
        //printf("gpuC={");
        //for (unsigned int i = 0; i < problemReference->pimpl->tensorC.numElements(); i++) {
        //  printf("%4.0f, ", gpuDataC[i]);
        //}
        //printf("}\n");
        //printf("cpuC={");
        //for (unsigned int i = 0; i < problemReference->pimpl->tensorC.numElements(); i++) {
        //  printf("%4.0f, ", cpuDataC[i]);
        //}
        //printf("}\n");


        // print cpu in tensor form
        //printf("\nTensorC-CPU:\n");
        //printf(problemReference->pimpl->tensorC.toString(tensorDataValidationC).c_str());

        // print gpu in tensor form
        //printf("\nTensorC-GPU:\n");
        //printf( problemReference->pimpl->tensorC.toString(gpuTensorDataOnHostC).c_str() );
        
        //printf("\nComparing...\n");
        bool equal = compareTensors(gpuTensorDataOnHostC, tensorDataValidationC, problemReference->pimpl->tensorC, ctrl );
        //printf("\nDone Comparing.\n");
        delete[] gpuTensorDataOnHostC.data;
        printf("P[%04llu] S[%03llu] %7s - %s\n", problemIdx, solutionIdx-solutionStartIdx, equal?"PASSED":"FAILED", solution->toString(0).c_str() );
        if (!equal) {
          printf("Oops!\n");
        }
      }

      // time solution
      double time = timeSolution( solution, tensorDataC, tensorDataA, tensorDataB, alpha, beta, ctrl );
      //printf("P[%04llu] S[%03llu] %7.3f - %s\n", problemIdx, solutionIdx-solutionStartIdx, time, solution->toString(0).c_str() );
      // write time to result xml file

    } // solution loop

    solutionStartIdx = solutionEndIdx;
    
  } // problem loop
  cobaltTeardown();
  return 0;
}





#include "Solution.h"
#include "Logger.h"
#include "StructOperations.h"
#include "MathTemplates.h"

#if Cobalt_BACKEND_OPENCL12
#include "CL/cl.h"
#elif Cobalt_BACKEND_HIP
#include <hip_runtime.h>
#endif

namespace Cobalt {

/*******************************************************************************
 *******************************************************************************
 **
 **   Solution
 **
 *******************************************************************************
 ******************************************************************************/

/*******************************************************************************
 * Solution constructor
 ******************************************************************************/
Solution::Solution( const Problem & inputProblem)
  : problem(inputProblem) { }


/*******************************************************************************
 * toStringXML
 ******************************************************************************/
std::string Solution::toStringXML( size_t indentLevel ) const {
  std::string state = Cobalt::indent(indentLevel) + "<S>\n";
  state += problem.toStringXML(indentLevel+1);
  state += toStringDetailXML(indentLevel+1);
  state += Cobalt::indent(indentLevel) + "</S>\n";
  return state;
}


/*******************************************************************************
 * toStringXML
 ******************************************************************************/
Problem Solution::getProblem() const {
  return problem;
}

bool Solution::operator<( const Solution & other ) const {
  return problem < other.getProblem();
}

/******************************************************************************
 * enqueueEntry
 * enter enqueue process here; can do validation and benchmarking
 *****************************************************************************/
CobaltStatus Solution::enqueueEntry(
  CobaltTensorData tensorDataC,
  CobaltTensorDataConst tensorDataA,
  CobaltTensorDataConst tensorDataB,
  CobaltScalarData alpha,
  CobaltScalarData beta,
  CobaltControl & ctrl) {
#if Cobalt_BACKEND_OPENCL12
      cl_int status;
#elif Cobalt_BACKEND_HIP
      hipError_t status;
#endif

  Logger::TraceEntry entry;
  entry.type = Logger::TraceEntryType::enqueueSolution;
  entry.solution = this;
    
  // validation
  if (ctrl.validate) {
    // enqueue gpu solution
    enqueue(tensorDataC, tensorDataA, tensorDataB, alpha, beta, ctrl);
    for (size_t i = 0; i < ctrl.numQueues; i++) {
#if Cobalt_BACKEND_OPENCL12
      clFlush(ctrl.queues[i]);
#elif Cobalt_BACKEND_HIP
      // automatically flushed?
#endif
    }
    // allocate memory for gpu result on host
    size_t sizeC = Solution::problem.tensorC.numBytes();
    CobaltTensorData gpuOnHostC;
    gpuOnHostC.offset = 0;
    gpuOnHostC.data = malloc(sizeC);
    // wait for gpu solution
    printf("Validation: syncing %u queues\n", ctrl.numQueuesUsed);
    for (size_t i = 0; i < ctrl.numQueuesUsed; i++) {
#if Cobalt_BACKEND_OPENCL12
      clFinish(ctrl.queues[i]);
#elif Cobalt_BACKEND_HIP
      status = hipStreamSynchronize( ctrl.queues[i] );
#endif
    }
    // copy results back
    printf("Validation: reading %u bytes\n", (unsigned int)sizeC);
#if Cobalt_BACKEND_OPENCL12
    clEnqueueReadBuffer(ctrl.queues[0], (cl_mem)tensorDataC.data,
        CL_TRUE, tensorDataC.offset, sizeC, gpuOnHostC.data,
        0, nullptr, nullptr);
#elif Cobalt_BACKEND_HIP
    status = hipMemcpy(gpuOnHostC.data, tensorDataC.data, sizeC, hipMemcpyDeviceToHost);
    status = hipStreamSynchronize(nullptr);
#endif
    // compare results
    CobaltTensorData *ref = static_cast<CobaltTensorData *>(ctrl.validate);
    CobaltTensorDataConst constRef{ ref->data, ref->offset};
    CobaltTensorDataConst constGPU{ gpuOnHostC.data, gpuOnHostC.offset};
    bool equal = compareTensors(constGPU, constRef,
        Solution::problem.tensorC, ctrl);
    entry.validationStatus = equal ? ValidationStatus::statusValid
        : ValidationStatus::statusInvalid;
    printf("%s validation %s;", equal ? "PASSED" : "FAILED",
        toString(0).c_str() );
    // cleanup
    delete static_cast<float *>(gpuOnHostC.data);
  } else {
    printf("%s;", toString(0).c_str());
  }

  // benchmarking
  if (ctrl.benchmark) {
    Cobalt::Timer timer;
    CobaltStatus returnStatus;

    // warmup 
    returnStatus =
        enqueue(tensorDataC, tensorDataA, tensorDataB, alpha, beta, ctrl); 
    for (size_t i = 0; i < ctrl.numQueuesUsed; i++) {
#if Cobalt_BACKEND_OPENCL12
      clFinish(ctrl.queues[i]);
#elif Cobalt_BACKEND_HIP
      status = hipStreamSynchronize( ctrl.queues[i] );
#endif
    }

    // start timer
    timer.start();
    // enqueue solution
    for (size_t i = 0; i < ctrl.benchmark; i++) {
      returnStatus = enqueue(tensorDataC, tensorDataA, tensorDataB,
          alpha, beta, ctrl);
    } // samples
    // wait for queues
    for (size_t i = 0; i < ctrl.numQueuesUsed; i++) {
#if Cobalt_BACKEND_OPENCL12
      status = clFinish(ctrl.queues[i]);
#elif Cobalt_BACKEND_HIP
      status = hipStreamSynchronize( ctrl.queues[i] );
#endif
    }
    // stop timer
    double time = timer.elapsed_ms();
    time /= ctrl.benchmark;
    printf(" t = %7.3f ms (avg of %u)", time, ctrl.benchmark);
    entry.benchmarkTimes.push_back(time);

    if (!ctrl.validate) {
      entry.status = returnStatus;
    }
  }
  printf("\n");

  // if we didn't do any of the previous, enqueue here
  if( !ctrl.validate && !ctrl.benchmark ) {
    entry.status = enqueue(tensorDataC, tensorDataA, tensorDataB,
        alpha, beta, ctrl);
  }

#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.log(entry);
#endif

  return entry.status;
}
  


/*******************************************************************************
 *******************************************************************************
 **
 **   Solution-LogOnly
 **
 *******************************************************************************
 ******************************************************************************/
#if Cobalt_LOGGER_ENABLED
/*******************************************************************************
 * LogSolution:: constructor
 ******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
SolutionLogOnly<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::SolutionLogOnly( const Problem & inputProblem)
  : SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>(inputProblem) {
}

/*******************************************************************************
* LogSolution:: destructor
******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
SolutionLogOnly<TypeC, TypeA, TypeB, TypeAlpha, TypeBeta>::~SolutionLogOnly() {
  // nothing
}

/*******************************************************************************
 * LogSolution:: toString
 ******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
std::string SolutionLogOnly<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toString( size_t indentLevel ) const {
  return Solution::toStringXML(0);
}

/*******************************************************************************
* LogSolution:: toStringDetailXML
******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
std::string SolutionLogOnly<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toStringDetailXML( size_t indentLevel ) const {
  return "";
}

/*******************************************************************************
 * LogSolution:: enqueue
 ******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
CobaltStatus SolutionLogOnly<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::enqueue(
    CobaltTensorData tensorDataC,
    CobaltTensorDataConst tensorDataA,
    CobaltTensorDataConst tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl & ctrl ) {
  printf("ERROR - CobaltSolutionLogOnly::enqueue() should never be called.\n");
  return cobaltStatusSuccess;
}
#endif


/*******************************************************************************
 *******************************************************************************
 **
 **   Solution-Template
 **
 *******************************************************************************
 ******************************************************************************/

/*******************************************************************************
 * CobaltSolutionTemplate constructor
 ******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::SolutionTemplate( const Problem & inputProblem)
  : Solution(inputProblem) { }


/*******************************************************************************
 *******************************************************************************
 **
 **   Solution-GPU
 **
 *******************************************************************************
 ******************************************************************************/

/******************************************************************************
 * constructor
 *****************************************************************************/
template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
SolutionGPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::
SolutionGPU( const Problem & inputProblem)
    : SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>(inputProblem) {
  // nothing
}

template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
SolutionGPU<TypeC, TypeA, TypeB, TypeAlpha, TypeBeta>::
~SolutionGPU() {
  // no kernels to be released?
}

/******************************************************************************
 * assignKernelArgs
 *****************************************************************************/
template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
void SolutionGPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::
assignKernelArgs() {

  /****************************************
   * Step 1: Calculate per-kernel values
   ***************************************/

  // work-group sizes
  localWorkSize[0] = workGroup[0]; // * microTile[0];
  localWorkSize[1] = workGroup[1]; // * microTile[1];
  localWorkSize[2] = 1;

  // num kernels
  unsigned int numEdgeKernels0 = edge[0] ? 1 : 0;
  unsigned int numEdgeKernels1 = edge[1] ? 1 : 0;
  unsigned int numMainKernels0 = kernelGrid[0] - numEdgeKernels0;
  unsigned int numMainKernels1 = kernelGrid[1] - numEdgeKernels1;

  // 3rd dim is size of all other dimension
  unsigned int sizeOfAllOtherDimensions = 1;
  for (unsigned int i = 0; i < Solution::problem.tensorC.numDims(); i++) {
    if (i != indexAssignmentCd0 && i != indexAssignmentCd1) {
      sizeOfAllOtherDimensions *= Solution::problem.tensorC[i].size;
    }
  }
  unsigned int sizeOfCAlongD0 =
      Solution::problem.tensorC[indexAssignmentCd0].size;
  unsigned int sizeOfCAlongD1 =
      Solution::problem.tensorC[indexAssignmentCd1].size;
  unsigned int macroTileSizeAlongD0 =
      static_cast<unsigned int>(workGroup[0] * microTile[0]);
  unsigned int macroTileSizeAlongD1 =
      static_cast<unsigned int>(workGroup[1] * microTile[1]);
  unsigned int totalWorkGroupsAlongD0 = sizeOfCAlongD0 / macroTileSizeAlongD0;
  unsigned int totalWorkGroupsAlongD1 = sizeOfCAlongD1 / macroTileSizeAlongD1;

  // add extra work-group here for single-branch-kernel solution here
  if (!edge[0] && totalWorkGroupsAlongD0*macroTileSizeAlongD0
      < sizeOfCAlongD0) {
    totalWorkGroupsAlongD0++;
  }
  if (!edge[1] && totalWorkGroupsAlongD1*macroTileSizeAlongD1
      < sizeOfCAlongD1) {
    totalWorkGroupsAlongD1++;
  }

  // divide work groups among kernels in kernelGrid
  unsigned int mainWorkGroupsAlongD0 = totalWorkGroupsAlongD0 / numMainKernels0;
  unsigned int mainWorkGroupsAlongD1 = totalWorkGroupsAlongD1 / numMainKernels1;
  globalWorkSize[0][0] = /*localWorkSize[0] */ mainWorkGroupsAlongD0;
  globalWorkSize[0][1] = /*localWorkSize[1] */ mainWorkGroupsAlongD1;
  globalWorkSize[0][2] = /*localWorkSize[2] */ sizeOfAllOtherDimensions;
  
  unsigned int kernelNumElementsDim0[maxNumKernels];
  unsigned int kernelNumElementsDim1[maxNumKernels];
  unsigned int kernelNumElementsDimU[maxNumKernels];

  kernelNumElementsDim0[0] =
      edge[0] ? mainWorkGroupsAlongD0 * macroTileSizeAlongD0 : sizeOfCAlongD0;
  kernelNumElementsDim1[0] =
      edge[1] ? mainWorkGroupsAlongD1 * macroTileSizeAlongD1 : sizeOfCAlongD1;
  kernelNumElementsDimU[0] =
      Solution::problem.tensorA[indexAssignmentAdU].size/kernelGrid[2];


  // add extra work-group for multi-kernel solution here
  if (edge[0] && totalWorkGroupsAlongD0*macroTileSizeAlongD0
      < sizeOfCAlongD0) {
    totalWorkGroupsAlongD0++;
  }
  if (edge[1] && totalWorkGroupsAlongD1*macroTileSizeAlongD1
      < sizeOfCAlongD1) {
    totalWorkGroupsAlongD1++;
  }

  // kernel - edge0
  if (edge[0]) {
    globalWorkSize[1][0] = /*localWorkSize[0] */ (totalWorkGroupsAlongD0 - numMainKernels0*mainWorkGroupsAlongD0);
    globalWorkSize[1][1] = /*localWorkSize[1] */ mainWorkGroupsAlongD1;
    globalWorkSize[1][2] = /*localWorkSize[2] */ sizeOfAllOtherDimensions;
    kernelNumElementsDim0[1] = sizeOfCAlongD0
        - numMainKernels0*mainWorkGroupsAlongD0*macroTileSizeAlongD0;
    kernelNumElementsDim1[1] = kernelNumElementsDim1[0]; // sizeOfCAlongD1;
    kernelNumElementsDimU[1] = kernelNumElementsDimU[0];
  } else {
    globalWorkSize[1][0] = 0;
    globalWorkSize[1][1] = 0;
    globalWorkSize[1][2] = 0;
    kernelNumElementsDim0[1] = 0;
    kernelNumElementsDim1[1] = 0;
    kernelNumElementsDimU[1] = 0;
  }

  // kernel - edge1
  if (edge[1]) {
    globalWorkSize[2][0] = /*localWorkSize[0] */ mainWorkGroupsAlongD0;
    globalWorkSize[2][1] = /*localWorkSize[1] */ (totalWorkGroupsAlongD1 - numMainKernels1*mainWorkGroupsAlongD1);
    globalWorkSize[2][2] = /*localWorkSize[2] */ sizeOfAllOtherDimensions;
    kernelNumElementsDim0[2] = kernelNumElementsDim0[0]; // sizeOfCAlongD0;
    kernelNumElementsDim1[2] = sizeOfCAlongD1
        - numMainKernels1*mainWorkGroupsAlongD1*macroTileSizeAlongD1;
    kernelNumElementsDimU[2] = kernelNumElementsDimU[0];
  } else {
    globalWorkSize[2][0] = 0;
    globalWorkSize[2][1] = 0;
    globalWorkSize[2][2] = 0;
    kernelNumElementsDim0[2] = 0;
    kernelNumElementsDim1[2] = 0;
    kernelNumElementsDimU[2] = 0;
  }

  // kernel - edge01
  if (edge[0] && edge[1]) {
    globalWorkSize[3][0] = /*localWorkSize[0] */ (totalWorkGroupsAlongD0 - numMainKernels0*mainWorkGroupsAlongD0);
    globalWorkSize[3][1] = /*localWorkSize[1] */ (totalWorkGroupsAlongD1 - numMainKernels1*mainWorkGroupsAlongD1);
    globalWorkSize[3][2] = /*localWorkSize[2] */ sizeOfAllOtherDimensions;
    kernelNumElementsDim0[3] = kernelNumElementsDim0[1]; // same Dim0 as edge0
    kernelNumElementsDim1[3] = kernelNumElementsDim1[2]; // same Dim1 as edge1
    kernelNumElementsDimU[3] = kernelNumElementsDimU[0];
  } else {
    globalWorkSize[3][0] = 0;
    globalWorkSize[3][1] = 0;
    globalWorkSize[3][2] = 0;
    kernelNumElementsDim0[3] = 0;
    kernelNumElementsDim1[3] = 0;
    kernelNumElementsDimU[3] = 0;
  }

  /****************************************
   * Step 2: Calculate per-enqueue values
   ***************************************/

  // kernel 0 - main
  // kernel 1 - edge 0
  // kernel 2 - edge 1
  // kernel 3 - edge 0,1
  for (unsigned int i = 0; i < maxNumKernels; i++) {
    numEnqueues[i] = 0;
  }
  for( unsigned int d1 = 0; d1 < kernelGrid[1]; d1++) {
    for (unsigned int d0 = 0; d0 < kernelGrid[0]; d0++) {
      for (unsigned int dU = 0; dU < kernelGrid[2]; dU++) {
        // which kernel is getting enqueued for this kernel grid entry
        unsigned int kernelIdx = 0;

        if (d0 == kernelGrid[0]-1 && edge[0]) { // edge 0 kernel
          kernelIdx += 1;
        }
        if (d1 == kernelGrid[1]-1 && edge[1]) { // edge 1 kernel
          kernelIdx += 2;
        }

        if ( kernelNumElementsDim0[kernelIdx] == 0
            || kernelNumElementsDim1[kernelIdx] == 0
            || kernelNumElementsDimU[kernelIdx] == 0 ) {
          continue;
        }

        // is this kernelIdx necessary (no edges or corner only)?
        if (globalWorkSize[kernelIdx][0] *
            globalWorkSize[kernelIdx][1] *
            globalWorkSize[kernelIdx][2] == 0) {
          continue;
        }

        // init enqueue args for offsets
        for ( unsigned int i = 0; i < 3; i++) {
          enqueueArgs[kernelIdx][numEnqueues[kernelIdx]][i] = 0;
        }
        // copy kernel args into enqueue args
        for ( unsigned int i = 3; i < numKernelArgs; i++) {
          enqueueArgs[kernelIdx][numEnqueues[kernelIdx]][i] = *kernelArgs[i];
        }

        // tensorC grid offsets
        unsigned int tensorOffsetCd0 = d0*kernelNumElementsDim0[0]
            *Solution::problem.tensorC[indexAssignmentCd0].stride;
        unsigned int tensorOffsetCd1 = d1*kernelNumElementsDim1[0]
            *Solution::problem.tensorC[indexAssignmentCd1].stride;

        // tensorA,B grid offsets
        unsigned int tensorOffsetAdU = dU*kernelNumElementsDimU[0]
            *Solution::problem.tensorA[indexAssignmentAdU].stride;
        unsigned int tensorOffsetBdU = dU*kernelNumElementsDimU[0]
            *Solution::problem.tensorB[indexAssignmentAdU].stride;
        unsigned int tensorOffsetAd0or1 = 0;
        unsigned int tensorOffsetBd0or1 = 0;
        if (d0InTensorA) {
          tensorOffsetAd0or1 = d0*(kernelNumElementsDim0[0])
              *Solution::problem.tensorA[indexAssignmentAd0or1].stride;
          tensorOffsetBd0or1 = d1*(kernelNumElementsDim1[0])
              *Solution::problem.tensorB[indexAssignmentBd0or1].stride;
        } else {
          tensorOffsetAd0or1 = d1*(kernelNumElementsDim1[0])
              *Solution::problem.tensorA[indexAssignmentAd0or1].stride;
          tensorOffsetBd0or1 = d0*(kernelNumElementsDim0[0])
              *Solution::problem.tensorB[indexAssignmentBd0or1].stride;
        }

        // tensor total offsets
        unsigned int tensorOffsetC = tensorOffsetCd0 + tensorOffsetCd1;
        unsigned int tensorOffsetA = tensorOffsetAd0or1 + tensorOffsetAdU;
        unsigned int tensorOffsetB = tensorOffsetBd0or1 + tensorOffsetBdU;
        enqueueArgs[kernelIdx][numEnqueues[kernelIdx]][0] = tensorOffsetC;
        enqueueArgs[kernelIdx][numEnqueues[kernelIdx]][1] = tensorOffsetA;
        enqueueArgs[kernelIdx][numEnqueues[kernelIdx]][2] = tensorOffsetB;
        
        // size overrides (due to kernel grid)
        enqueueArgs[kernelIdx][numEnqueues[kernelIdx]][kernelArgIdxDim0] =
            kernelNumElementsDim0[kernelIdx];
        enqueueArgs[kernelIdx][numEnqueues[kernelIdx]][kernelArgIdxDim1] =
            kernelNumElementsDim1[kernelIdx];
        enqueueArgs[kernelIdx][numEnqueues[kernelIdx]][kernelArgIdxSummation] =
            kernelNumElementsDimU[kernelIdx];

        // add enqueue
        numEnqueues[kernelIdx]++;
      }
    }
  }

  // manipulate which kernels get executed (for debugging)
  //numEnqueues[0] = 0;
  //numEnqueues[1] = 0;
  //numEnqueues[2] = 0;
  //numEnqueues[3] = 0;

} // assign kernel arguments


/*******************************************************************************
 *******************************************************************************
 **
 **   Solution-OpenCL
 **
 *******************************************************************************
 ******************************************************************************/

/*******************************************************************************
 * CobaltSolutionOpenCL constructor
 ******************************************************************************/
#if Cobalt_BACKEND_OPENCL12
template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::
SolutionOpenCL( const Problem & inputProblem)
    : SolutionGPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>(inputProblem) {

  for (size_t i = 0; i < this->maxNumKernels; i++) {
    kernels[i] = nullptr;
  }
}

template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
SolutionOpenCL<TypeC, TypeA, TypeB, TypeAlpha, TypeBeta>::
~SolutionOpenCL() {
  // opencl kernels released in cobaltTeardown()
}

#if 0
/******************************************************************************
 * assignWorkSizes - global and local
 *****************************************************************************/
template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
void SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::
assignWorkSizes() {

  // work-group sizes
  for (unsigned int i = 0; i < maxNumKernels; i++) {
    localWorkSize[i][0] = workGroup[0]; // * microTile[0];
    localWorkSize[i][1] = workGroup[1]; // * microTile[1];
    localWorkSize[i][2] = 1;
  }


  unsigned int numEdgeKernels0 = edge[0] ? 1 : 0;
  unsigned int numEdgeKernels1 = edge[1] ? 1 : 0;
  unsigned int numMainKernels0 = kernelGrid[0] - numEdgeKernels0;
  unsigned int numMainKernels1 = kernelGrid[1] - numEdgeKernels1;

  // 3rd dim is size of all other dimension
  unsigned int sizeOfAllOtherDimensions = 1;
  for (unsigned int i = 0; i < Solution::problem.tensorC.numDims(); i++) {
    if (i != indexAssignmentCd0 && i != indexAssignmentCd1) {
      sizeOfAllOtherDimensions *= Solution::problem.tensorC[i].size;
    }
  }
  unsigned int sizeOfCAlongD0 =
      Solution::problem.tensorC[indexAssignmentCd0].size;
  unsigned int sizeOfCAlongD1 =
      Solution::problem.tensorC[indexAssignmentCd1].size;
  unsigned int macroTileSizeAlongD0 =
      static_cast<unsigned int>(workGroup[0] * microTile[0]);
  unsigned int macroTileSizeAlongD1 =
      static_cast<unsigned int>(workGroup[1] * microTile[1]);
  unsigned int totalWorkGroupsAlongD0 = sizeOfCAlongD0 / macroTileSizeAlongD0;
  unsigned int totalWorkGroupsAlongD1 = sizeOfCAlongD1 / macroTileSizeAlongD1;

  // add extra work-group here for single-branch-kernel solution here
  if (!edge[0] && totalWorkGroupsAlongD0*macroTileSizeAlongD0
      < sizeOfCAlongD0) {
    totalWorkGroupsAlongD0++;
  }
  if (!edge[1] && totalWorkGroupsAlongD1*macroTileSizeAlongD1
      < sizeOfCAlongD1) {
    totalWorkGroupsAlongD1++;
  }

  // divide work groups among kernels in kernelGrid
  unsigned int mainWorkGroupsAlongD0 = totalWorkGroupsAlongD0 / numMainKernels0;
  unsigned int mainWorkGroupsAlongD1 = totalWorkGroupsAlongD1 / numMainKernels1;
  globalWorkSize[0][0] = localWorkSize[0][0] * mainWorkGroupsAlongD0;
  globalWorkSize[0][1] = localWorkSize[0][1] * mainWorkGroupsAlongD1;
  globalWorkSize[0][2] = localWorkSize[0][2] * sizeOfAllOtherDimensions;
  
  kernelNumElementsDim0[0] =
      edge[0] ? mainWorkGroupsAlongD0 * macroTileSizeAlongD0 : sizeOfCAlongD0;
  kernelNumElementsDim1[0] =
      edge[1] ? mainWorkGroupsAlongD1 * macroTileSizeAlongD1 : sizeOfCAlongD1;
  kernelNumElementsDimU[0] =
      Solution::problem.tensorA[indexAssignmentAdU].size/kernelGrid[2];


  // add extra work-group for multi-kernel solution here
  if (edge[0] && totalWorkGroupsAlongD0*macroTileSizeAlongD0 < sizeOfCAlongD0) {
    totalWorkGroupsAlongD0++;
  }
  if (edge[1] && totalWorkGroupsAlongD1*macroTileSizeAlongD1 < sizeOfCAlongD1) {
    totalWorkGroupsAlongD1++;
  }

  // kernel - edge0
  if (edge[0]) {
    globalWorkSize[1][0] = localWorkSize[1][0] * (totalWorkGroupsAlongD0 - numMainKernels0*mainWorkGroupsAlongD0);
    globalWorkSize[1][1] = localWorkSize[1][1] * mainWorkGroupsAlongD1;
    globalWorkSize[1][2] = localWorkSize[1][2] * sizeOfAllOtherDimensions;
    kernelNumElementsDim0[1] = sizeOfCAlongD0 - numMainKernels0*mainWorkGroupsAlongD0*macroTileSizeAlongD0;
    kernelNumElementsDim1[1] = kernelNumElementsDim1[0]; // sizeOfCAlongD1;
    kernelNumElementsDimU[1] = kernelNumElementsDimU[0];
  } else {
    globalWorkSize[1][0] = 0;
    globalWorkSize[1][1] = 0;
    globalWorkSize[1][2] = 0;
    kernelNumElementsDim0[1] = 0;
    kernelNumElementsDim1[1] = 0;
    kernelNumElementsDimU[1] = 0;
  }

  // kernel - edge1
  if (edge[1]) {
    globalWorkSize[2][0] = localWorkSize[2][0] * mainWorkGroupsAlongD0;
    globalWorkSize[2][1] = localWorkSize[2][1] * (totalWorkGroupsAlongD1 - numMainKernels1*mainWorkGroupsAlongD1);
    globalWorkSize[2][2] = localWorkSize[2][2] * sizeOfAllOtherDimensions;
    kernelNumElementsDim0[2] = kernelNumElementsDim0[0]; // sizeOfCAlongD0;
    kernelNumElementsDim1[2] = sizeOfCAlongD1 - numMainKernels1*mainWorkGroupsAlongD1*macroTileSizeAlongD1;
    kernelNumElementsDimU[2] = kernelNumElementsDimU[0];
  } else {
    globalWorkSize[2][0] = 0;
    globalWorkSize[2][1] = 0;
    globalWorkSize[2][2] = 0;
    kernelNumElementsDim0[2] = 0;
    kernelNumElementsDim1[2] = 0;
    kernelNumElementsDimU[2] = 0;
  }

  // kernel - edge01
  if (edge[0] && edge[1]) {
    globalWorkSize[3][0] = localWorkSize[3][0] * (totalWorkGroupsAlongD0 - numMainKernels0*mainWorkGroupsAlongD0);
    globalWorkSize[3][1] = localWorkSize[3][1] * (totalWorkGroupsAlongD1 - numMainKernels1*mainWorkGroupsAlongD1);
    globalWorkSize[3][2] = localWorkSize[3][2] * sizeOfAllOtherDimensions;
    kernelNumElementsDim0[3] = kernelNumElementsDim0[1]; // same Dim0 as edge0 (kernel 1)
    kernelNumElementsDim1[3] = kernelNumElementsDim1[2]; // same Dim1 as edge1 (kernel 2)
    kernelNumElementsDimU[3] = kernelNumElementsDimU[0];
  } else {
    globalWorkSize[3][0] = 0;
    globalWorkSize[3][1] = 0;
    globalWorkSize[3][2] = 0;
    kernelNumElementsDim0[3] = 0;
    kernelNumElementsDim1[3] = 0;
    kernelNumElementsDimU[3] = 0;
  }

}
#endif

#define TIME_KERNEL_COMPILATION 1



/******************************************************************************
 * makeKernel
 *****************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
void SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::makeKernel(
  cl_kernel *kernel,
  cl_command_queue queue,
  const char *kernelSource,
  const char *sourceBuildOptions)
{
  // initialize kernel map
  if (!kernelMap) {
    kernelMap = new KernelMap();
  }

  // was kernel already taken care of for this solution instance?
  if (*kernel) {
    return;
  }

  // get context and device from queue
  cl_int err;
  cl_context clContext;
  cl_device_id clDevice;
  err = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(clContext), &clContext, NULL);
  CL_CHECK(err)
    err = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(clDevice), &clDevice, NULL);
  CL_CHECK(err)

  // is kernel already compiled?
  KernelMapKey key;
  key.kernelSource = kernelSource;
  key.context = clContext;
  key.device = clDevice;
  KernelMap::iterator idx = kernelMap->find(key);
  if (idx != kernelMap->end()) {
    *kernel = idx->second;
    //printf("kernel already compiled %p %p\n", kernel, *kernel);
    return;
  }
  
  //printf("building kernel\n");
  cl_program clProgram;
  clProgram = clCreateProgramWithSource(
    clContext,
    1, &kernelSource,
    NULL, &err );
  CL_CHECK(err)
  // driver leaks ~200kB at this call
  err = clBuildProgram(
    clProgram,
    1, &clDevice,
    sourceBuildOptions, NULL, NULL );
  CL_CHECK(err)

  // print build failure
  if (err != CL_SUCCESS) {
    printf("clBuildProgram Failed; Error = %d\n", err);
    printf("\nKernel Source:\n\n");
    printf("%s\n", kernelSource);

    size_t len = 0;
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char* buildLog = new char[len];
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, len*sizeof(char), buildLog, 0);
    printf("\n\n\nBuild Log:\n\n");
    printf("%s\n", buildLog);
    printf("\n");
    delete[] buildLog;
  }
  err = clCreateKernelsInProgram(
    clProgram,
    1, kernel,
    NULL );
  CL_CHECK(err)
	err = clReleaseProgram(clProgram);
	CL_CHECK(err)

  // put kernel in map
  (*kernelMap)[key] = *kernel;
}


/******************************************************************************
 * enqueue
 *****************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
CobaltStatus SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::enqueue(
    CobaltTensorData tensorDataC,
    CobaltTensorDataConst tensorDataA,
    CobaltTensorDataConst tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl & ctrl ) {

  // compile kernels
  const char *buildOptions = "-cl-std=CL2.0";
  for (size_t i = 0; i < this->maxNumKernels; i++) {
    if (kernelSources[i]) {
      makeKernel( &kernels[i], ctrl.queues[0], kernelSources[i], buildOptions );
    }
  }

  size_t *globalWorkOffset = NULL;

  unsigned int kernelSerialIdx = 0;
  for (unsigned int kernelIdx = 0; kernelIdx < this->maxNumKernels;
      kernelIdx++) {
    for (unsigned int enqueueIdx = 0;
        enqueueIdx < this->numEnqueues[kernelIdx]; enqueueIdx++) {

      // data pointers
      status = clSetKernelArg( kernels[kernelIdx], 0,
          sizeof(cl_mem), &tensorDataC.data ); CL_CHECK(status)
      status = clSetKernelArg( kernels[kernelIdx], 1,
          sizeof(cl_mem), &tensorDataA.data ); CL_CHECK(status)
      status = clSetKernelArg( kernels[kernelIdx], 2,
          sizeof(cl_mem), &tensorDataB.data ); CL_CHECK(status)
      status = clSetKernelArg( kernels[kernelIdx], 3,
          sizeof(TypeAlpha), alpha.data ); CL_CHECK(status)
      status = clSetKernelArg( kernels[kernelIdx], 4,
          sizeof(TypeBeta), beta.data ); CL_CHECK(status)

      // uint args
      for (unsigned int i = 0; i < this->numKernelArgs; i++) {
        status = clSetKernelArg( kernels[kernelIdx], i+5, sizeof(unsigned int),
            &this->enqueueArgs[kernelIdx][enqueueIdx][i] ); CL_CHECK(status)
      }

      // out cl_event
      cl_event *outEvent = nullptr;
      if (ctrl.numOutputEvents) {
        outEvent = &(ctrl.outputEvents[kernelSerialIdx%ctrl.numOutputEvents]);
      }

      /*
      size_t gws[3] = { this->globalWorkSize[kernelIdx][0],
          this->globalWorkSize[kernelIdx][1],
          this->globalWorkSize[kernelIdx][2] };
      size_t lws[3] = { this->localWorkSize[0],
          this->localWorkSize[1],
          this->localWorkSize[2] };
       */

      // print args
#if 0
      printf("openclKernelLaunch(%u:%u:%u):\n    g{%u,%u,%u};\n    l{%u,%u,%u};\n    p{%p,%p,%p};\n    ab{%f,%f};\n    o{%u,%u,%u};\n    s{%u,%u,%u,%u,%u,%u}\n",
          kernelIdx,
          enqueueIdx,
          this->numKernelArgs,
          (unsigned int)this->globalWorkSize[kernelIdx][0],
          (unsigned int)this->globalWorkSize[kernelIdx][1],
          (unsigned int)this->globalWorkSize[kernelIdx][2],
          (unsigned int)this->localWorkSize[0],
          (unsigned int)this->localWorkSize[1],
          (unsigned int)this->localWorkSize[2],
          static_cast<TypeC*>(tensorDataC.data),
          static_cast<const TypeA*>(tensorDataA.data),
          static_cast<const TypeB*>(tensorDataB.data),
          *static_cast<const TypeAlpha*>(alpha.data),
          *static_cast<const TypeBeta*>(beta.data),
          this->enqueueArgs[kernelIdx][enqueueIdx][0],
          this->enqueueArgs[kernelIdx][enqueueIdx][1],
          this->enqueueArgs[kernelIdx][enqueueIdx][2],
          this->enqueueArgs[kernelIdx][enqueueIdx][3],
          this->enqueueArgs[kernelIdx][enqueueIdx][4],
          this->enqueueArgs[kernelIdx][enqueueIdx][5],
          this->enqueueArgs[kernelIdx][enqueueIdx][6],
          this->enqueueArgs[kernelIdx][enqueueIdx][7],
          this->enqueueArgs[kernelIdx][enqueueIdx][8] ); 
#endif
      // enqueue
      status = clEnqueueNDRangeKernel(
          ctrl.queues[kernelSerialIdx%ctrl.numQueues],
          kernels[kernelIdx],
          this->workDim,
          globalWorkOffset,
          this->globalWorkSize[kernelIdx],
          this->localWorkSize,
          ctrl.numInputEvents,
          ctrl.inputEvents,
          outEvent );
      CL_CHECK(status)
      clFinish(ctrl.queues[kernelSerialIdx%ctrl.numQueues]);
      kernelSerialIdx++;
    }
  }

  if (kernelSerialIdx > ctrl.numQueues) {
    ctrl.numQueuesUsed = ctrl.numQueues;
  } else {
    ctrl.numQueuesUsed = kernelSerialIdx;
  }

  return cobaltStatusSuccess;


#if 0
  // user is allowed to pass in null for alpha & beta, in which case we'll provide
  // the default values here
  TypeC fallbackAlpha;
  TypeC fallbackBeta;
  size_t sizeOfAlpha = Solution::problem.alphaSize(); // sizeof(TypeAlpha);
  size_t sizeOfBeta = Solution::problem.betaSize(); // sizeof(TypeBeta);
  if (!alpha.data && requireAlpha) {
    fallbackAlpha = Cobalt::getOne<TypeC>();
    alpha.data = &fallbackAlpha;
    sizeOfAlpha = sizeof(TypeC);
  }
  if (!beta.data && requireBeta) {
    fallbackBeta = Cobalt::getZero<TypeC>();
    beta.data = &fallbackBeta;
    sizeOfBeta = sizeof(TypeC);
  }
  TypeC betaOne = Cobalt::getOne<TypeC>(); // if summation unrolled

  if (argOffsets && !Solution::problem.useOffsets) {
    printf("SolutionOpenCL::enqueue() solution requires offsets but problem specifies not to use them; write code to provide dummyOffsets=0 for this scenario.\n");
  }

  cl_int status;
  cl_uint workDim = 3;
  size_t *globalWorkOffset = NULL;

  // compile kernels
  const char *buildOptions = "-cl-std=CL2.0";
  for (size_t i = 0; i < numKernels; i++) {
    if (kernelSources[i]) {
      makeKernel( &kernels[i], ctrl.queues[0], kernelSources[i], buildOptions );
    }
  }
  

  // kernel 0 - main
  // kernel 1 - edge 0
  // kernel 2 - edge 1
  // kernel 3 - edge 0,1
  unsigned int kernelSerialIdx = 0;
  for( unsigned int d1 = 0; d1 < kernelGrid[1]; d1++) {
    for (unsigned int d0 = 0; d0 < kernelGrid[0]; d0++) {
      for (unsigned int dU = 0; dU < kernelGrid[2]; dU++) {
        // which kernel is getting enqueued for this kernel grid entry
        unsigned int kernelIdx = 0;

        if (d0 == kernelGrid[0]-1 && edge[0]) { // edge 0 kernel
          kernelIdx += 1;
        }
        if (d1 == kernelGrid[1]-1 && edge[1]) { // edge 1 kernel
          kernelIdx += 2;
        }

        if ( kernelNumElementsDim0[kernelIdx] == 0
            || kernelNumElementsDim1[kernelIdx] == 0
            || kernelNumElementsDimU[kernelIdx] == 0 ) {
          continue;
        }

        // if this kernelIdx isn't necessary (no edges or corner only...) continue
        if (globalWorkSize[kernelIdx][0] *
            globalWorkSize[kernelIdx][1] *
            globalWorkSize[kernelIdx][2] == 0) {
          continue;
        }

        unsigned int argIdx = 0;

        // data pointers
        status = clSetKernelArg( kernels[kernelIdx], argIdx++, sizeof(cl_mem), &tensorDataC.data );
        CL_CHECK(status)
        status = clSetKernelArg( kernels[kernelIdx], argIdx++, sizeof(cl_mem), &tensorDataA.data );
        CL_CHECK(status)
        status = clSetKernelArg( kernels[kernelIdx], argIdx++, sizeof(cl_mem), &tensorDataB.data );
        CL_CHECK(status)

        if (argOffsets || kernelIdx > 0) {
          // tensorC offsets
          unsigned int tensorOffsetCd0 = d0*kernelNumElementsDim0[0]*Solution::problem.tensorC[indexAssignmentCd0].stride;
          unsigned int tensorOffsetCd1 = d1*kernelNumElementsDim1[0]*Solution::problem.tensorC[indexAssignmentCd1].stride;

          // tensorA,B offsets
          unsigned int tensorOffsetAdU = dU*kernelNumElementsDimU[0]*Solution::problem.tensorA[indexAssignmentAdU].stride;
          unsigned int tensorOffsetBdU = dU*kernelNumElementsDimU[0]*Solution::problem.tensorB[indexAssignmentAdU].stride;
          unsigned int tensorOffsetAd0or1 = 0;
          unsigned int tensorOffsetBd0or1 = 0;
          if (d0InTensorA) {
            tensorOffsetAd0or1 = d0*(kernelNumElementsDim0[0])*Solution::problem.tensorA[indexAssignmentAd0or1].stride;
            tensorOffsetBd0or1 = d1*(kernelNumElementsDim1[0])*Solution::problem.tensorB[indexAssignmentBd0or1].stride;
          } else {
            tensorOffsetAd0or1 = d1*(kernelNumElementsDim1[0])*Solution::problem.tensorA[indexAssignmentAd0or1].stride;
            tensorOffsetBd0or1 = d0*(kernelNumElementsDim0[0])*Solution::problem.tensorB[indexAssignmentBd0or1].stride;
          }
          // data offsets
          unsigned int tensorOffsetC = tensorDataC.offset + tensorOffsetCd0 + tensorOffsetCd1;
          unsigned int tensorOffsetA = tensorDataA.offset + tensorOffsetAd0or1 + tensorOffsetAdU;
          unsigned int tensorOffsetB = tensorDataB.offset + tensorOffsetBd0or1 + tensorOffsetBdU;

          
          status = clSetKernelArg( kernels[kernelIdx], argIdx++, sizeof(unsigned int), &tensorOffsetC );
          CL_CHECK(status)
          status = clSetKernelArg( kernels[kernelIdx], argIdx++, sizeof(unsigned int), &tensorOffsetA );
          CL_CHECK(status)
          status = clSetKernelArg( kernels[kernelIdx], argIdx++, sizeof(unsigned int), &tensorOffsetB );
          CL_CHECK(status)
        }

        if (argSizes || kernelIdx > 0) {
          // data sizes (truncated due to grid)
          unsigned int readShift = argIdx;
          //if (!argOffsets && kernelIdx == 0) {
          //  readShift = 3;
          //}
          for ( unsigned int i = 0; i < numKernelArgs; i++) {
            status = clSetKernelArg( kernels[kernelIdx], argIdx+i, kernelArgSizes[i], kernelArgs[i] );
            CL_CHECK(status)
          }

          // size overrides
          status = clSetKernelArg( kernels[kernelIdx], kernelArgIdxDim0+readShift, kernelArgSizes[kernelArgIdxDim0], &kernelNumElementsDim0[kernelIdx] );
          CL_CHECK(status)
          status = clSetKernelArg( kernels[kernelIdx], kernelArgIdxDim1+readShift, kernelArgSizes[kernelArgIdxDim1], &kernelNumElementsDim1[kernelIdx] );
          CL_CHECK(status)
          status = clSetKernelArg( kernels[kernelIdx], kernelArgIdxSummation+readShift, kernelArgSizes[kernelArgIdxSummation], &kernelNumElementsDimU[kernelIdx] );
          CL_CHECK(status)
          argIdx += numKernelArgs;
        }

        // alpha
        if (requireAlpha) {
          status = clSetKernelArg( kernels[kernelIdx], argIdx++, sizeOfAlpha, alpha.data );
          CL_CHECK(status)
        }

        // beta
        if (requireBeta) {
          status = clSetKernelArg( kernels[kernelIdx], argIdx++, sizeOfBeta, (dU>0) ? static_cast<void *>(&betaOne) : beta.data );
          CL_CHECK(status)
        }

        // enqueue
#if 0
        printf("enq[%u,%u,%u] k%u: s{%u, %u, %u}, g{%llu, %llu, %llu} l{%llu, %llu, %llu}\n",
          d0, d1, dU,
          kernelIdx,
          //tensorOffsetC, tensorOffsetA, tensorOffsetB,
          kernelNumElementsDim0[kernelIdx], kernelNumElementsDim1[kernelIdx], kernelNumElementsDimU[kernelIdx],
          globalWorkSize[kernelIdx][0],
          globalWorkSize[kernelIdx][1],
          globalWorkSize[kernelIdx][2],
          localWorkSize[kernelIdx][0],
          localWorkSize[kernelIdx][1],
          localWorkSize[kernelIdx][2]);
#endif
        cl_event *outEvent = nullptr;
        if (ctrl.numOutputEvents) {
          outEvent = &(ctrl.outputEvents[kernelSerialIdx%ctrl.numOutputEvents]);
        }
#if 1
        status = clEnqueueNDRangeKernel(
          ctrl.queues[kernelSerialIdx%ctrl.numQueues],
          kernels[kernelIdx],
          workDim,
          globalWorkOffset,
          globalWorkSize[kernelIdx],
          localWorkSize[kernelIdx],
          ctrl.numInputEvents,
          ctrl.inputEvents,
          outEvent );
#endif
        CL_CHECK(status)
        //status = clFinish(ctrl.queues[kernelSerialIdx%ctrl.numQueues]);
        //CL_CHECK(status)
        kernelSerialIdx++;
      }
    }
  }
  if (kernelSerialIdx > ctrl.numQueues) {
    ctrl.numQueuesUsed = ctrl.numQueues;
  } else {
    ctrl.numQueuesUsed = kernelSerialIdx;
  }

  //ctrl.numOutputEvents = kernelSerialIdx;;
  return cobaltStatusSuccess;
#endif
} // enqueue
#endif

/*******************************************************************************
 *******************************************************************************
 **
 **   Solution-HIP
 **
 *******************************************************************************
 ******************************************************************************/
#if Cobalt_BACKEND_HIP
/******************************************************************************
 * constructor
 *****************************************************************************/
template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
SolutionHIP<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::
SolutionHIP( const Problem & inputProblem)
    : SolutionGPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>(inputProblem) { }

template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
SolutionHIP<TypeC, TypeA, TypeB, TypeAlpha, TypeBeta>::
~SolutionHIP() {
  // no kernels to be released?
}


/******************************************************************************
 * enqueue
 * each solution defines its own enqueue, so it can get the arguments right
 *****************************************************************************/
/*
template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
CobaltStatus SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::
enqueue(
    CobaltTensorData tensorDataC,
    CobaltTensorDataConst tensorDataA,
    CobaltTensorDataConst tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl & ctrl ) {

  for (unsigned int i = 0; i < numEnqueues; i++) {

    hipLaunchKernel(
        HIP_KERNEL_NAME(sgemm),
        dim3(globalWorkSize[i][0], globalWorkSize[i][1], globalWorkSize[i][2]),
        dim3(localWorkSize[i][0], localWorkSize[i][1], localWorkSize[i][2]),
        groupMemBytes,
        ctrl.queues[i%ctrl.numQueues],
        static_cast<TypeC*>(tensorDataC.data),
        static_cast<TypeA*>(tensorDataA.data),
        static_cast<TypeB*>(tensorDataB.data),
        static_cast<TypeAlpha*>(alpha.data),
        static_cast<TypeBeta*>(beta.data)
        enqueueArgs[i][0]+tensorDataC.offset,
        enqueueArgs[i][1]+tensorDataA.offset,
        enqueueArgs[i][2]+tensorDataB.offset,
        enqueueArgs[i][3]
        );
  } // for enqueues

  return cobaltStatusSuccess;
} // enqueue
*/

#endif






} // namespace

#if Cobalt_BACKEND_OPENCL12
bool operator<(const KernelMapKey & l, const KernelMapKey & r) {

  if (l.kernelSource < r.kernelSource) {
    return true;
  }
  else if (r.kernelSource < l.kernelSource) {
    return false;
  }
  if (l.context < r.context) {
    return true;
  }
  else if (r.context < l.context) {
    return false;
  }
  if (l.device < r.device) {
    return true;
  }
  else if (r.device < l.device) {
    return false;
  }
  return false;
}
#endif

/*******************************************************************************
 * Explicit Template Instantiation
 ******************************************************************************/
// used for cpu classes
template class Cobalt::SolutionTemplate<float,float,float,float,float>;
template class Cobalt::SolutionTemplate<double,double,double,double,double>;
template class Cobalt::SolutionTemplate<CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat>;
template class Cobalt::SolutionTemplate<CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble>;
// used for gpu classes
//template class CobaltSolutionOpenCL<float,float,float,void,void>;
//template class CobaltSolutionOpenCL<double,double,double,void,void>;
//template class CobaltSolutionOpenCL<CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,void,void>;
//template class CobaltSolutionOpenCL<CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,void,void>;

#include "SolutionTemplateInstantiations.inl"

template class Cobalt::SolutionLogOnly<void,void,void,void,void>;

#if Cobalt_BACKEND_OPENCL12
#if defined( _WIN32 )
__declspec(thread) KernelMap *kernelMap = 0;
#else
__thread KernelMap *kernelMap = 0;
#endif
#endif

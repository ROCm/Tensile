
#include "Solution.h"
#include "Logger.h"


/*******************************************************************************
 * CobaltSolution:: constructor
 ******************************************************************************/
CobaltSolution::CobaltSolution( CobaltProblem inputProblem)
  : problem(inputProblem) {
}

#ifdef Cobalt_BACKEND_OPENCL12
/*******************************************************************************
 * CobaltSolutionOpenCL:: constructor
 ******************************************************************************/
CobaltSolutionOpenCL::CobaltSolutionOpenCL( CobaltProblem inputProblem)
  : CobaltSolution(inputProblem) {
}

CobaltStatus CobaltSolutionOpenCL::enqueue(
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltTensorData tensorDataC,
    CobaltControl & ctrl ) {
  cl_int status;
  cl_uint workDim = 2;
  size_t *globalWorkOffset = NULL;

  // kernel 0 - main
  // kernel 1 - edge 0
  // kernel 2 - edge 1
  // kernel 3 - edge 0,1
  size_t kernelSerialIdx = 0;
  for( size_t d0 = 0; d0 < kernelGrid[0]; d0++) {
    for (size_t d1 = 0; d1 < kernelGrid[1]; d1++) {
      for (size_t dk = 0; dk < kernelGrid[2]; dk++) {
        // which kernel is getting enqueued for this kernel grid entry
        size_t kernelIdx = 0;
        if (d0 == kernelGrid[0]-1 && edge[0]) {
          kernelIdx += 1;
        }
        if (d1 == kernelGrid[1]-1 && edge[1]) {
          kernelIdx += 2;
        }
        // data pointers
        status = clSetKernelArg( kernels[kernelIdx], 0, sizeof(cl_mem), tensorDataA.data );
        status = clSetKernelArg( kernels[kernelIdx], 1, sizeof(cl_mem), tensorDataB.data );
        status = clSetKernelArg( kernels[kernelIdx], 2, sizeof(cl_mem), tensorDataC.data );
        // data offsets
        // data sizes (truncated due to grid)

        status = clEnqueueNDRangeKernel(
          ctrl.queues[kernelSerialIdx%ctrl.numQueues],
          kernels[kernelIdx],
          workDim,
          globalWorkOffset,
          globalWorkSize[kernelIdx],
          localWorkSize[kernelIdx],
          ctrl.numInputEvents,
          ctrl.inputEvents,
          &ctrl.outputEvents[kernelSerialIdx] );
          }
        }
  }


  for (size_t i = 0; i < numKernels; i++) {

  }
  ctrl.numOutputEvents = numKernels;
  return cobaltStatusSuccess;
}

/*******************************************************************************
 * LogSolution:: toString
 ******************************************************************************/
std::string CobaltSolutionOpenCL::toString( size_t indentLevel ) const {
  std::string state = indent(indentLevel) + "<Solution>\n";
  state += ::toStringXML(problem, indentLevel+1);
  state += indent(indentLevel) + "</Solution>";
  return state;
}

/*******************************************************************************
 * CobaltSolutionOpenCLDummy:: constructor
 ******************************************************************************/
CobaltSolutionOpenCLDummy::CobaltSolutionOpenCLDummy( CobaltProblem inputProblem)
  : CobaltSolutionOpenCL(inputProblem) {
  problem.tensorC.dimensions[0].stride;
#if 0
  indexAssignmentTileDim0 = 3;
  tensorAssignedDim0 = 0;
  indexAssignmentTileDim1 = 1;
  tensorAssignedDim1 = 1;
  indexAssignmentUnroll = 2;
  // may as well write everything else out in case want to print


  numKernels = 4;
  if (!CT_SSS_Cijk_Aij_bK_DeviceProfile_kernel) {
    // not yet compiled, so compile it
  }
  kernels[0] = CT_SSS_Cji=Si_Aik_Bkj_a1b0_j8x1_i8x1_k8__kernel;
  kernels[1] = CT_SSS_Cji=Si_Aik_Bkj_a1b0_j8x1_i8y1_k8__kernel;
  kernels[2] = CT_SSS_Cji=Si_Aik_Bkj_a1b0_j8y1_i8x1_k8__kernel;
  kernels[3] = CT_SSS_Cji=Si_Aik_Bkj_a1b0_j8y1_i8y1_k8__kernel;

  localWorkSize[0] = { 8, 8, 1 };
  localWorkSize[1] = { 8, 8, 1 };
  localWorkSize[2] = { 8, 8, 1 };
  localWorkSize[3] = { 8, 8, 1 };

  globalWorkSize[0][0] = tensorAssignedDim0 == TensorOrdinalA
      ? problem.tensorA.dimensions[indexAssignmentTileDim0].size / macroTileDim0
      : problem.tensorB.dimensions[indexAssignmentTileDim0].size / macroTileDim0;
  globalWorkSize[0][1] = tensorAssignedDim1 == TensorOrdinalA
      ? problem.tensorA.dimensions[indexAssignmentTileDim1].size / macroTileDim0
      : problem.tensorB.dimensions[indexAssignmentTileDim1].size / macroTileDim0;
  globalWorkSize[1] = { 1, globalWorkSize[0][1] };
  globalWorkSize[2] = { globalWorkSize[0][0], 1 };
  globalWorkSize[3] = { 1, 1 };
  
  // set kernel args
  // 3 pointers
  // 3 offsets
  // C strides
  // A strides
  // B strides
  // dimension sizes, I, J, K...
  // alpha
  // beta
  /*
  strideCI
  strideCJ
  strideAI
  strideAK
  strideBK
  strideBJ
  sizeI  <-- even these for ifs and splits
  sizeJ
  sizeK

  */
  size_t alphaSize
      = problem.operation.alphaType == cobaltDataTypeSingle ? sizeof(float)
      : problem
  // set globalWorkSize...
#endif
}

#endif

#if Cobalt_LOGGER_ENABLED
/*******************************************************************************
 * LogSolution:: constructor
 ******************************************************************************/
CobaltSolutionLogOnly::CobaltSolutionLogOnly( CobaltProblem inputProblem)
  : CobaltSolution(inputProblem) {
}

//CobaltSolutionLogOnly::~CobaltSolutionLogOnly() {
//}

/*******************************************************************************
 * LogSolution:: enqueue
 ******************************************************************************/
CobaltStatus CobaltSolutionLogOnly::enqueue(
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltTensorData tensorDataC,
    CobaltControl & ctrl ) {
  printf("CobaltSolution::enqueue() virtual function not overrided\n");
  return cobaltStatusSuccess;
}

/*******************************************************************************
 * LogSolution:: toString
 ******************************************************************************/
std::string CobaltSolutionLogOnly::toString( size_t indentLevel ) const {
  std::string state = indent(indentLevel) + "<Solution>\n";
  state += ::toStringXML(problem, indentLevel+1);
  state += indent(indentLevel) + "</Solution>";
  return state;
}


#endif

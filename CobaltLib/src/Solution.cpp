
#include "Solution.h"
#include "Logger.h"


/*******************************************************************************
 * CobaltSolution:: constructor
 ******************************************************************************/
CobaltSolution::CobaltSolution( CobaltProblem inputProblem)
  : problem(inputProblem) {
}

#ifdef COBALT_BACKEND_OPENCL
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

  for (size_t i = 0; i < numKernels; i++) {
    status = clSetKernelArg( kernels[i], 0, sizeof(cl_mem), tensorDataA.data );
    status = clSetKernelArg( kernels[i], 1, sizeof(cl_mem), tensorDataB.data );
    status = clSetKernelArg( kernels[i], 2, sizeof(cl_mem), tensorDataC.data );
    status = clEnqueueNDRangeKernel(
      ctrl.queues[i%ctrl.numQueues],
      kernels[i],
      workDim,
      globalWorkOffset,
      globalWorkSize[i],
      localWorkSize[i],
      ctrl.numEventsInWaitList,
      ctrl.eventWaitList,
      &ctrl.event );

  }

}

/*******************************************************************************
 * CobaltSolutionOpenCLDummy:: constructor
 ******************************************************************************/
CobaltSolutionOpenCLDummy::CobaltSolutionOpenCLDummy( CobaltProblem inputProblem)
  : CobaltSolutionOpenCL(inputProblem) {
  numKernels = 1;
  if (!CT_SSS_Cijk_Aij_bK_DeviceProfile_kernel) {
    // not yet compiled, so compile it
  }
  kernels[0] = CT_SSS_Cijk_Aij_bK_kernel;
  // set kernel args
  // set globalWorkSize...
}

/*CobaltStatus CobaltSolutionOpenCL::enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltControl & ctrl ) {
  // delete me
}*/
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

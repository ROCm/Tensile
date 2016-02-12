
#include "Solution.h"
#include "Logger.h"


/*******************************************************************************
 * CobaltSolution:: constructor
 ******************************************************************************/
CobaltSolution::CobaltSolution( CobaltProblem inputProblem)
  : problem(inputProblem) {
}


/*******************************************************************************
 * CobaltSolution:: toStringXML
 ******************************************************************************/
std::string CobaltSolution::toStringXML( size_t indentLevel ) const {
  std::string state = indent(indentLevel) + "<Solution>\n";
  state += ::toStringXML(problem, indentLevel+1);
  state += indent(indentLevel) + "</Solution>";
  return state;
}

#ifdef Cobalt_BACKEND_OPENCL12
/*******************************************************************************
 * CobaltSolutionOpenCL:: constructor
 ******************************************************************************/
CobaltSolutionOpenCL::CobaltSolutionOpenCL( CobaltProblem inputProblem)
  : CobaltSolution(inputProblem) {
}

/******************************************************************************
 * Make Gemm Kernel
 *****************************************************************************/
void CobaltSolutionOpenCL::makeKernel(
  cl_kernel *kernel,
  cl_command_queue queue,
  const char *kernelSource,
  const char *sourceBuildOptions)
{
  cl_int err;
  if (*kernel) {
    // kernel has already been built, return
#ifdef AUTOGEMM_PRINT_DEBUG
    // get kernel name
    size_t kernelNameLength;
    err = clGetKernelInfo(
      *clKernel,
      CL_KERNEL_FUNCTION_NAME,
      sizeof(kernelNameLength),
      NULL,
      &kernelNameLength );
    CL_CHECK(err)
    char *kernelName = new char[kernelNameLength];
    err = clGetKernelInfo(
      *clKernel,
      CL_KERNEL_FUNCTION_NAME,
      kernelNameLength*sizeof(char),
      kernelName,
      NULL );
    CL_CHECK(err)
    printf("makeGemmKernel: \"%s\" already built; returning.\n", kernelName);
    delete[] kernelName;
#endif
    return;
  } else {
    // kernel has not been built, so build it (from binary, preferably)
    cl_context clContext;
    cl_device_id clDevice;
    err = clGetCommandQueueInfo( queue, CL_QUEUE_CONTEXT, sizeof(clContext), &clContext, NULL);
    CL_CHECK(err)
    err = clGetCommandQueueInfo( queue, CL_QUEUE_DEVICE, sizeof(clDevice), &clDevice, NULL);
    CL_CHECK(err)
    cl_program clProgram;
    
    clProgram = clCreateProgramWithSource(
      clContext,
      1, &kernelSource,
      NULL, &err );
    CL_CHECK(err)
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
      //printf("\n\nKernel String:\n\n");
      //printf("%s\n", kernelSource);
    }

    err = clCreateKernelsInProgram(
      clProgram,
      1, kernel,
      NULL );
    CL_CHECK(err)
	err = clReleaseProgram(clProgram);
	CL_CHECK(err)
    
#ifdef AUTOGEMM_PRINT_DEBUG
    // get kernel name
    size_t kernelNameLength;
    err = clGetKernelInfo(
      *clKernel,
      CL_KERNEL_FUNCTION_NAME,
      sizeof(kernelNameLength),
      NULL,
      &kernelNameLength );
    CL_CHECK(err)
    char *kernelName = new char[kernelNameLength];
    err = clGetKernelInfo(
      *clKernel,
      CL_KERNEL_FUNCTION_NAME,
      kernelNameLength*sizeof(char),
      kernelName,
      NULL );
    CL_CHECK(err)
    printf("makeGemmKernel: \"%s\" now built; returning.\n", kernelName);
    delete[] kernelName;
#endif
  }
}

CobaltStatus CobaltSolutionOpenCL::enqueue(
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltTensorData tensorDataC,
    CobaltControl & ctrl ) {
  cl_int status;
  cl_uint workDim = 2;
  size_t *globalWorkOffset = NULL;

  // compile kernels
  char *buildOptions = "";
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
  for( unsigned int d0 = 0; d0 < kernelGrid[0]; d0++) {
    for (unsigned int d1 = 0; d1 < kernelGrid[1]; d1++) {
      for (unsigned int dU = 0; dU < kernelGrid[2]; dU++) {
        // which kernel is getting enqueued for this kernel grid entry
        unsigned int kernelIdx = 0;
        if (d0 == kernelGrid[0]-1 && edge[0]) {
          kernelIdx += 1;
        }
        if (d1 == kernelGrid[1]-1 && edge[1]) {
          kernelIdx += 2;
        }
        // data pointers
        status = clSetKernelArg( kernels[kernelIdx], 0, sizeof(cl_mem), tensorDataC.data );
        status = clSetKernelArg( kernels[kernelIdx], 1, sizeof(cl_mem), tensorDataA.data );
        status = clSetKernelArg( kernels[kernelIdx], 2, sizeof(cl_mem), tensorDataB.data );

        // tensorC offsets
        unsigned int tensorOffsetCd0 = d0*problem.tensorC.dimensions[indexAssignmentCd0].stride/kernelGrid[0];
        unsigned int tensorOffsetCd1 = d1*problem.tensorC.dimensions[indexAssignmentCd1].stride/kernelGrid[1];

        // tensorA,B offsets
        unsigned int tensorOffsetAdU = dU*problem.tensorA.dimensions[indexAssignmentAdU].stride/kernelGrid[2];
        unsigned int tensorOffsetBdU = dU*problem.tensorB.dimensions[indexAssignmentBdU].stride/kernelGrid[2];
        unsigned int tensorOffsetAd0or1 = 0;
        unsigned int tensorOffsetBd0or1 = 0;
        if (d0InTensorA) {
          tensorOffsetAd0or1 = d0*problem.tensorA.dimensions[indexAssignmentAd0or1].stride/kernelGrid[0];
          tensorOffsetBd0or1 = d1*problem.tensorB.dimensions[indexAssignmentBd0or1].stride/kernelGrid[1];
        } else {
          tensorOffsetAd0or1 = d1*problem.tensorA.dimensions[indexAssignmentAd0or1].stride/kernelGrid[1];
          tensorOffsetBd0or1 = d0*problem.tensorB.dimensions[indexAssignmentBd0or1].stride/kernelGrid[0];
        }
        // data offsets
        unsigned int tensorOffsetC = tensorDataC.offset + tensorOffsetCd0 + tensorOffsetCd1;
        unsigned int tensorOffsetA = tensorDataA.offset + tensorOffsetAd0or1 + tensorOffsetAdU;
        unsigned int tensorOffsetB = tensorDataB.offset + tensorOffsetBd0or1 + tensorOffsetBdU;
        
        status = clSetKernelArg( kernels[kernelIdx], 3, sizeof(unsigned int), &tensorOffsetC );
        status = clSetKernelArg( kernels[kernelIdx], 4, sizeof(unsigned int), &tensorOffsetA );
        status = clSetKernelArg( kernels[kernelIdx], 5, sizeof(unsigned int), &tensorOffsetB );

        // data sizes (truncated due to grid)
        for (cl_uint i = 6; i < numKernelArgs; i++) {
          status = clSetKernelArg( kernels[kernelIdx], i, kernelArgSizes[i], kernelArgs[i] );
        }

        // d0 size override
        unsigned int sizeDim0 = *(unsigned int *)kernelArgs[kernelArgIdxDim0] / kernelGrid[0];
        status = clSetKernelArg( kernels[kernelIdx], kernelArgIdxDim0, kernelArgSizes[kernelArgIdxDim0], &sizeDim0 );
        // d1 size override
        unsigned int sizeDim1 = *(unsigned int *)kernelArgs[kernelArgIdxDim1] / kernelGrid[1];
        status = clSetKernelArg( kernels[kernelIdx], kernelArgIdxDim1, kernelArgSizes[kernelArgIdxDim1], &sizeDim1 );

        // d0 size override
        unsigned int sizeSummation = *(unsigned int *)kernelArgs[kernelArgIdxSummation] / kernelGrid[2];
        status = clSetKernelArg( kernels[kernelIdx], kernelArgIdxSummation, kernelArgSizes[kernelArgIdxSummation], &sizeSummation );

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
#endif


#if Cobalt_LOGGER_ENABLED
/*******************************************************************************
 * LogSolution:: constructor
 ******************************************************************************/
CobaltSolutionLogOnly::CobaltSolutionLogOnly( CobaltProblem inputProblem)
  : CobaltSolution(inputProblem) {
}


/*******************************************************************************
 * LogSolution:: toString
 ******************************************************************************/
std::string CobaltSolutionLogOnly::toString( size_t indentLevel ) const {
  return toStringXML(indentLevel);
}

/*******************************************************************************
 * LogSolution:: enqueue
 ******************************************************************************/
CobaltStatus CobaltSolutionLogOnly::enqueue(
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltTensorData tensorDataC,
    CobaltControl & ctrl ) {
  printf("CobaltSolutionLogOnly::enqueue()\n");
  return cobaltStatusSuccess;
}


#endif


#include "Solution.h"
#include "Logger.h"
#include "StructOperations.h"
#include "MathTemplates.h"

namespace Cobalt {

/*******************************************************************************
 * Solution constructor
 ******************************************************************************/
Solution::Solution( const Problem & inputProblem)
  : problem(inputProblem) { }


/*******************************************************************************
 * toStringXML
 ******************************************************************************/
std::string Solution::toStringXML( size_t indentLevel ) const {
  std::string state = Cobalt::indent(indentLevel) + "<Solution>\n";
  state += problem.toStringXML(indentLevel+1);
  state += Cobalt::indent(indentLevel) + "</Solution>\n";
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



/*******************************************************************************
 * CobaltSolutionTemplate constructor
 ******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::SolutionTemplate( const Problem & inputProblem)
  : Solution(inputProblem) { }




/*******************************************************************************
 * CobaltSolutionOpenCL constructor
 ******************************************************************************/
#ifdef Cobalt_BACKEND_OPENCL12
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::SolutionOpenCL( const Problem & inputProblem)
  : SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>(inputProblem) { }


/******************************************************************************
 * assignWorkSizes - global and local
 *****************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
void SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::assignWorkSizes() {

  // work-group sizes
  for (unsigned int i = 0; i < maxNumKernels; i++) {
    localWorkSize[i][0] = workGroup[0]; // * microTile[0];
    localWorkSize[i][1] = workGroup[1]; // * microTile[1];
    localWorkSize[i][2] = 1;
  }

  // kernel - main
  unsigned int sizeOfAllOtherDimensions = 1;
  for (unsigned int i = 0; i < problem.tensorC.numDims(); i++) {
    if (i != indexAssignmentCd0 && i != indexAssignmentCd1) {
      sizeOfAllOtherDimensions *= problem.tensorC[i].size;
    }
  }
  unsigned int sizeOfCAlongD0 = problem.tensorC[indexAssignmentCd0].size;
  unsigned int sizeOfCAlongD1 = problem.tensorC[indexAssignmentCd1].size;
  unsigned int macroTileSizeAlongD0 = static_cast<unsigned int>(workGroup[0] * microTile[0]); // macroTile
  unsigned int macroTileSizeAlongD1 = static_cast<unsigned int>(workGroup[1] * microTile[1]); // macroTile
  unsigned int numWorkGroupsAlongD0 = sizeOfCAlongD0 / macroTileSizeAlongD0;
  unsigned int numWorkGroupsAlongD1 = sizeOfCAlongD1 / macroTileSizeAlongD1;
  // branch kernel
  if ( !edge[0] && numWorkGroupsAlongD0*macroTileSizeAlongD0 < sizeOfCAlongD0) {
    numWorkGroupsAlongD0++;
  }
  if (!edge[1] && numWorkGroupsAlongD1*macroTileSizeAlongD1 < sizeOfCAlongD1) {
    numWorkGroupsAlongD1++;
  }
  //size_t numWorkGroups = numWorkGroupsAlongD0 * numWorkGroupsAlongD1 * sizeOfAllOtherDimensions;
  // divide work groups among kernels in kernelGrid
  numWorkGroupsAlongD0 /= edge[0] ? (kernelGrid[0]-1) : (kernelGrid[0]);
  numWorkGroupsAlongD1 /= edge[1] ? (kernelGrid[1]-1) : (kernelGrid[1]);
  globalWorkSize[0][0] = localWorkSize[0][0] * numWorkGroupsAlongD0;
  globalWorkSize[0][1] = localWorkSize[0][1] * numWorkGroupsAlongD1;
  globalWorkSize[0][2] = localWorkSize[0][2] * sizeOfAllOtherDimensions;
  
  kernelNumElementsDim0[0] = edge[0] ? numWorkGroupsAlongD0 * macroTileSizeAlongD0 : sizeOfCAlongD0;
  kernelNumElementsDim1[0] = edge[1] ? numWorkGroupsAlongD1 * macroTileSizeAlongD1 : sizeOfCAlongD1;
  kernelNumElementsDimU[0] = problem.tensorA[indexAssignmentAdU].size/kernelGrid[2];

  

  // kernel - edge0
  if (edge[0]) {
    globalWorkSize[1][0] = localWorkSize[1][0]; // * numWorkGroupsAlongD0;
    globalWorkSize[1][1] = localWorkSize[1][1] * numWorkGroupsAlongD1;
    globalWorkSize[1][2] = localWorkSize[1][2] * sizeOfAllOtherDimensions;
    kernelNumElementsDim0[1] = sizeOfCAlongD0 % macroTileSizeAlongD0;
    kernelNumElementsDim1[1] = sizeOfCAlongD1;
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
    globalWorkSize[2][0] = localWorkSize[2][0] * numWorkGroupsAlongD0;
    globalWorkSize[2][1] = localWorkSize[2][1]; // * numWorkGroupsAlongD1;
    globalWorkSize[2][2] = localWorkSize[2][2] * sizeOfAllOtherDimensions;
    kernelNumElementsDim0[2] = sizeOfCAlongD0;
    kernelNumElementsDim1[2] = sizeOfCAlongD1 % macroTileSizeAlongD1;
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
    globalWorkSize[3][0] = localWorkSize[3][0];
    globalWorkSize[3][1] = localWorkSize[3][1];
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
      printf("\n");
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

  
/******************************************************************************
 * enqueue
 *****************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
CobaltStatus SolutionOpenCL<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::enqueue(
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl & ctrl ) {
  //printf("Status: Enqueueing %s\n", toString(0).c_str());

  // user is allowed to pass in null for alpha & beta, in which case we'll provide
  // the default values here
  TypeC fallbackAlpha;
  TypeC fallbackBeta;
  size_t sizeOfAlpha = problem.alphaSize(); // sizeof(TypeAlpha);
  size_t sizeOfBeta = problem.betaSize(); // sizeof(TypeBeta);
  if (!alpha.data && requireAlpha) {
    fallbackAlpha = Cobalt::getOne<TypeC>();
    alpha.data = &fallbackAlpha;
    sizeOfAlpha = sizeof(TypeAlpha);
  }
  if (!beta.data && requireBeta) {
    fallbackBeta = Cobalt::getZero<TypeC>();
    beta.data = &fallbackBeta;
    sizeOfBeta = sizeof(TypeBeta);
  }

  cl_int status;
  cl_uint workDim = 3;
  size_t *globalWorkOffset = NULL;

  // compile kernels
  char *buildOptions = "";
  for (size_t i = 0; i < numKernels; i++) {
    if (kernelSources[i]) {
      makeKernel( &kernels[i], ctrl.queues[0], kernelSources[i], buildOptions );
    }
  }
  
  unsigned int numElements0 = *(unsigned int *)kernelArgs[kernelArgIdxDim0];
  unsigned int numElements1 = *(unsigned int *)kernelArgs[kernelArgIdxDim1];
  unsigned int numEdgeKernels0 = edge[0] ? 1 : 0;
  unsigned int numEdgeKernels1 = edge[1] ? 1 : 0;
  unsigned int numMainKernels0 = kernelGrid[0] - numEdgeKernels0;
  unsigned int numMainKernels1 = kernelGrid[1] - numEdgeKernels1;
  unsigned int numElementsPerMainKernel0 = numElements0 / numMainKernels0;


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

        // if this kernelIdx isn't necessary (no edges or corner only...) continue
        if (globalWorkSize[kernelIdx][0] *
            globalWorkSize[kernelIdx][1] *
            globalWorkSize[kernelIdx][2] == 0) {
          continue;
        }


        // data pointers
        status = clSetKernelArg( kernels[kernelIdx], 0, sizeof(cl_mem), &tensorDataC.data );
        CL_CHECK(status)
        status = clSetKernelArg( kernels[kernelIdx], 1, sizeof(cl_mem), &tensorDataA.data );
        CL_CHECK(status)
        status = clSetKernelArg( kernels[kernelIdx], 2, sizeof(cl_mem), &tensorDataB.data );
        CL_CHECK(status)

        // tensorC offsets
        unsigned int tensorOffsetCd0 = d0*kernelNumElementsDim0[0]*problem.tensorC[indexAssignmentCd0].stride;
        unsigned int tensorOffsetCd1 = d1*kernelNumElementsDim1[0]*problem.tensorC[indexAssignmentCd1].stride;

        // tensorA,B offsets
        unsigned int tensorOffsetAdU = dU*kernelNumElementsDimU[0]*problem.tensorA[indexAssignmentAdU].stride;
        unsigned int tensorOffsetBdU = dU*kernelNumElementsDimU[0]*problem.tensorB[indexAssignmentAdU].stride;
        unsigned int tensorOffsetAd0or1 = 0;
        unsigned int tensorOffsetBd0or1 = 0;
        if (d0InTensorA) {
          tensorOffsetAd0or1 = d0*(kernelNumElementsDim0[0])*problem.tensorA[indexAssignmentAd0or1].stride;
          tensorOffsetBd0or1 = d1*(kernelNumElementsDim1[0])*problem.tensorB[indexAssignmentBd0or1].stride;
        } else {
          tensorOffsetAd0or1 = d1*(kernelNumElementsDim1[0])*problem.tensorA[indexAssignmentAd0or1].stride;
          tensorOffsetBd0or1 = d0*(kernelNumElementsDim0[0])*problem.tensorB[indexAssignmentBd0or1].stride;
        }
        // data offsets
        unsigned int tensorOffsetC = tensorDataC.offset + tensorOffsetCd0 + tensorOffsetCd1; // TODO - fix these for multi kernels
        unsigned int tensorOffsetA = tensorDataA.offset + tensorOffsetAd0or1 + tensorOffsetAdU;
        unsigned int tensorOffsetB = tensorDataB.offset + tensorOffsetBd0or1 + tensorOffsetBdU;

        
        status = clSetKernelArg( kernels[kernelIdx], 3, sizeof(unsigned int), &tensorOffsetC );
        CL_CHECK(status)
        status = clSetKernelArg( kernels[kernelIdx], 4, sizeof(unsigned int), &tensorOffsetA );
        CL_CHECK(status)
        status = clSetKernelArg( kernels[kernelIdx], 5, sizeof(unsigned int), &tensorOffsetB );
        CL_CHECK(status)

        // data sizes (truncated due to grid)
        for (cl_uint i = 6; i < numKernelArgs; i++) {
          status = clSetKernelArg( kernels[kernelIdx], i, kernelArgSizes[i], kernelArgs[i] );
          CL_CHECK(status)
        }

        // size overrides
        status = clSetKernelArg( kernels[kernelIdx], kernelArgIdxDim0, kernelArgSizes[kernelArgIdxDim0], &kernelNumElementsDim0[kernelIdx] );
        CL_CHECK(status)
        status = clSetKernelArg( kernels[kernelIdx], kernelArgIdxDim1, kernelArgSizes[kernelArgIdxDim1], &kernelNumElementsDim1[kernelIdx] );
        CL_CHECK(status)
        status = clSetKernelArg( kernels[kernelIdx], kernelArgIdxSummation, kernelArgSizes[kernelArgIdxSummation], &kernelNumElementsDimU[kernelIdx] );
        CL_CHECK(status)

        // alpha
        unsigned int argIdx = numKernelArgs;
        if (requireAlpha) {
          status = clSetKernelArg( kernels[kernelIdx], argIdx, sizeOfAlpha, alpha.data );
          CL_CHECK(status)
          argIdx++;
        }

        // beta
        if (requireBeta) {
          status = clSetKernelArg( kernels[kernelIdx], argIdx, sizeOfBeta, beta.data );
          CL_CHECK(status)
          argIdx++;
        }

        // enqueue
#if 0
        printf("enq[%u,%u,%u] k%u: o{%u, %u, %u} s{%u, %u, %u}, g{%llu, %llu, %llu} l{%llu, %llu, %llu}\n",
          d0, d1, dU,
          kernelIdx,
          tensorOffsetC, tensorOffsetA, tensorOffsetB,
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
        status = clFinish(ctrl.queues[kernelSerialIdx%ctrl.numQueues]);
        CL_CHECK(status)
        kernelSerialIdx++;
      }
    }
  }

  //ctrl.numOutputEvents = kernelSerialIdx;;
  return cobaltStatusSuccess;
}
#endif





/*******************************************************************************
 * LogSolution:: constructor
 ******************************************************************************/
#if Cobalt_LOGGER_ENABLED
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
SolutionLogOnly<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::SolutionLogOnly( const Problem & inputProblem)
  : SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>(inputProblem) {
}


/*******************************************************************************
 * LogSolution:: toString
 ******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
std::string SolutionLogOnly<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::toString( size_t indentLevel ) const {
  return toStringXML(0);
}

/*******************************************************************************
 * LogSolution:: enqueue
 ******************************************************************************/
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
CobaltStatus SolutionLogOnly<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::enqueue(
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl & ctrl ) {
  printf("ERROR - CobaltSolutionLogOnly::enqueue() should never be called.\n");
  return cobaltStatusSuccess;
}


#endif

}



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
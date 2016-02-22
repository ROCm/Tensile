#ifndef SOLUTION_H
#define SOLUTION_H

#include "Cobalt.h"

#include <string>

namespace Cobalt {

/*******************************************************************************
 * Solution - base private class
 ******************************************************************************/
class Solution {
public:
  Solution( CobaltProblem inputProblem );
  
  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataC,
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltScalarData alpha,
      CobaltScalarData beta,
      CobaltControl & ctrl ) = 0;

  virtual std::string toString( size_t indentLevel ) const = 0;
  
  std::string toStringXML( size_t indentLevel ) const;

  CobaltProblem getProblem() const;

protected:
  CobaltProblem problem;

};




/*******************************************************************************
 * CobaltSolutionTemplate - parent class for all solutions
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
class SolutionTemplate : public Solution {
public:
  SolutionTemplate( CobaltProblem inputProblem );
  
  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataC,
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltScalarData alpha,
      CobaltScalarData beta,
      CobaltControl & ctrl ) = 0;

  virtual std::string toString( size_t indentLevel ) const = 0;

};


/*******************************************************************************
 * CobaltSolutionOpenCL
 ******************************************************************************/
#ifdef Cobalt_BACKEND_OPENCL12
#include "CL/cl.h"
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
class SolutionOpenCL : public SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  SolutionOpenCL( CobaltProblem inputProblem );

  void makeKernel(
  cl_kernel *kernel,
  cl_command_queue queue,
  const char *kernelSource,
  const char *sourceBuildOptions);
  
  CobaltStatus enqueue(
      CobaltTensorData tensorDataC,
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltScalarData alpha,
      CobaltScalarData beta,
      CobaltControl & ctrl );

  virtual std::string toString( size_t indentLevel ) const = 0;

protected:
  // constants
  static const unsigned int workDim = 3;
  static const unsigned int maxNumKernels = 4;
  const static unsigned int maxKernelArgs = 3+3+2+4*CobaltTensor::maxDimensions;
  // kernels
  cl_uint numKernels;
  cl_kernel kernels[maxNumKernels];
  const char *kernelSources[maxNumKernels];
  unsigned int kernelGrid[workDim];
  unsigned int edge[workDim];
  // kernel dimensions
  size_t globalWorkSize[maxNumKernels][workDim];
  size_t localWorkSize[maxNumKernels][workDim];
  // kernel argumets
  cl_uint numKernelArgs;
  void *kernelArgs[maxKernelArgs];
  size_t kernelArgSizes[maxKernelArgs];

  unsigned int indexAssignmentCd0;
  unsigned int indexAssignmentCd1;
  bool d0InTensorA;
  unsigned int indexAssignmentAd0or1;
  unsigned int indexAssignmentAdU;
  unsigned int indexAssignmentBd0or1;
  unsigned int indexAssignmentBdU;

  unsigned int kernelArgIdxDim0;
  unsigned int kernelArgIdxDim1;
  unsigned int kernelArgIdxSummation;

};

#endif

/*******************************************************************************
 * CobaltSolutionLogOnly - used in LOG_ONLY mode
 ******************************************************************************/
#if Cobalt_LOGGER_ENABLED
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
class SolutionLogOnly : public SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  SolutionLogOnly( CobaltProblem inputProblem );
  
  CobaltStatus enqueue(
      CobaltTensorData tensorDataC,
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltScalarData alpha,
      CobaltScalarData beta,
      CobaltControl & ctrl );

  std::string toString( size_t indentLevel ) const;

};
#endif

}


#include <assert.h>
#define CL_CHECK(RET) \
  if(RET != CL_SUCCESS) { \
    printf("OpenCL error %i on line %u\n", RET, __LINE__); \
    /*assert(false);*/ \
    }



/*******************************************************************************
 * CobaltSolution - public pimpl
 ******************************************************************************/
struct _CobaltSolution {
  Cobalt::Solution *pimpl;
};


#endif

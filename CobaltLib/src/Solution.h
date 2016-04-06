#ifndef SOLUTION_H
#define SOLUTION_H

#include "Cobalt.h"
#include "Problem.h"

#include <string>

namespace Cobalt {

/*******************************************************************************
 * Solution - base abstract class without templates
 ******************************************************************************/
class Solution {
public:
  Solution( const Problem & inputProblem );
  virtual ~Solution() { /*nothing*/ };
  
  CobaltStatus enqueueEntry(
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl & ctrl);

  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataC,
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltScalarData alpha,
      CobaltScalarData beta,
      CobaltControl & ctrl ) = 0;

  virtual std::string toString( size_t indentLevel ) const = 0;
  
  std::string toStringXML( size_t indentLevel ) const;

  Problem getProblem() const;

  virtual bool operator<( const Solution & other) const;

protected:
  Problem problem;

};

struct CobaltSolutionPtrComparator
    : std::binary_function<const Solution *,
    const Solution *, bool> {
  bool  operator() (const Solution *l, const Solution *r) const {
    return *l < *r;
  }
};




/*******************************************************************************
 * SolutionTemplate - parent class for all solutions; with templates
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
class SolutionTemplate : public Solution {
public:
  SolutionTemplate( const Problem & inputProblem );
  virtual ~SolutionTemplate() { /*nothing*/ };

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
 * CobaltSolutionOpenCL - parent class for OpenCL solutions
 ******************************************************************************/
#ifdef Cobalt_BACKEND_OPENCL12
#include "CL/cl.h"
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
class SolutionOpenCL : public SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  SolutionOpenCL( const Problem & inputProblem );
  ~SolutionOpenCL();

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
  size_t workGroup[workDim];
  size_t microTile[workDim];
  size_t globalWorkSize[maxNumKernels][workDim];
  size_t localWorkSize[maxNumKernels][workDim];
  unsigned int kernelNumElementsDim0[maxNumKernels];
  unsigned int kernelNumElementsDim1[maxNumKernels];
  unsigned int kernelNumElementsDimU[maxNumKernels];
  void assignWorkSizes();
  // kernel argumets
  cl_uint numKernelArgs;
  const void *kernelArgs[maxKernelArgs];
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

  bool requireAlpha;
  bool requireBeta;

  // preprocessor optimizations
  bool argOffsets;
  bool argSizes;

};

#endif

/*******************************************************************************
 * CobaltSolutionLogOnly - used in LOG_ONLY mode
 ******************************************************************************/
#if Cobalt_LOGGER_ENABLED
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
class SolutionLogOnly : public SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  SolutionLogOnly( const Problem & inputProblem );
  ~SolutionLogOnly();

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
    printf("OpenCL Error %i on line %u of %s\n", RET, __LINE__, __FILE__); \
    /*assert(false);*/ \
    }



/*******************************************************************************
 * CobaltSolution - public pimpl
 ******************************************************************************/
struct _CobaltSolution {
  Cobalt::Solution *pimpl;
};


#endif

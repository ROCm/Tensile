/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef SOLUTION_H
#define SOLUTION_H

#include "Tensile.h"
#include "Problem.h"
#include <map>
#include <string>

namespace Tensile {

/*******************************************************************************
 * Solution - base abstract class without templates
 ******************************************************************************/
class Solution {
public:
  Solution( const Problem & inputProblem );
  virtual ~Solution() { /*nothing*/ };
  TensileStatus enqueueEntry(
      TensileTensorData tensorDataC,
      TensileTensorDataConst tensorDataA,
      TensileTensorDataConst tensorDataB,
      TensileScalarData alpha,
      TensileScalarData beta,
      TensileControl & ctrl,
      bool doPrint );
  virtual TensileStatus enqueue(
      TensileTensorData tensorDataC,
      TensileTensorDataConst tensorDataA,
      TensileTensorDataConst tensorDataB,
      TensileScalarData alpha,
      TensileScalarData beta,
      TensileControl & ctrl ) = 0;
  virtual std::string toString( size_t indentLevel ) const = 0;
  std::string toStringXML( size_t indentLevel ) const;
  virtual std::string toStringDetailXML( size_t indentLevel ) const = 0;
  Problem getProblem() const;
  virtual bool operator<( const Solution & other) const;
protected:
  Problem problem;
};

struct TensileSolutionPtrComparator
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
  virtual TensileStatus enqueue(
      TensileTensorData tensorDataC,
      TensileTensorDataConst tensorDataA,
      TensileTensorDataConst tensorDataB,
      TensileScalarData alpha,
      TensileScalarData beta,
      TensileControl & ctrl ) = 0;
  virtual std::string toString( size_t indentLevel ) const = 0;
  virtual std::string toStringDetailXML( size_t indentLevel ) const = 0;
};



/*******************************************************************************
 * SolutionGPU - parent of SolutionOpenCL and SolutionHIP
 ******************************************************************************/
template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
class SolutionGPU
    : public SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  SolutionGPU( const Problem & inputProblem );
  ~SolutionGPU();

  virtual TensileStatus enqueue(
      TensileTensorData tensorDataC,
      TensileTensorDataConst tensorDataA,
      TensileTensorDataConst tensorDataB,
      TensileScalarData alpha,
      TensileScalarData beta,
      TensileControl & ctrl ) = 0;

  virtual std::string toString( size_t indentLevel ) const = 0;
  virtual std::string toStringDetailXML( size_t indentLevel ) const = 0;

protected:
  void assignKernelArgs();

  // kernels
  static const unsigned int workDim = 3;
  static const unsigned int maxNumKernels = 4;
  static const unsigned int maxNumEnqueues = 4*4;
  const static unsigned int maxNumKernelArgs = 3+4*TensileTensor::maxDimensions;
  unsigned int numKernels;
  unsigned int numEnqueues[maxNumKernels];
  unsigned int numKernelArgs; // integers only
  unsigned int kernelGrid[workDim];
  unsigned int edge[workDim];

  // kernel dimensions
  unsigned int workGroup[workDim];
  unsigned int microTile[workDim];
  unsigned int macroTile[workDim];
  size_t globalWorkSize[maxNumKernels][workDim]; // all enqueues of a kernels are currently restricted to have identical grid size
  size_t localWorkSize[workDim];

  // kernel arguments; true if argument is argument, false if argument is pre-processor defined
  bool requireAlpha;
  bool requireBeta;
  bool argOffsets;
  bool argLeadingStrides;
  bool argSizes;
  const unsigned int *kernelArgs[maxNumKernelArgs];
  unsigned int enqueueArgs[maxNumKernels][maxNumEnqueues][maxNumKernelArgs];

  // index assignments
  bool d0InTensorA;
  unsigned int indexAssignmentCd0;
  unsigned int indexAssignmentCd1;
  unsigned int indexAssignmentAd0or1;
  unsigned int indexAssignmentAdU;
  unsigned int indexAssignmentBd0or1;
  unsigned int indexAssignmentBdU;

  unsigned int kernelArgIdxDim0;
  unsigned int kernelArgIdxDim1;
  unsigned int kernelArgIdxSummation;

};



/*******************************************************************************
 * TensileSolutionOpenCL - parent class for OpenCL solutions
 ******************************************************************************/
#if Tensile_BACKEND_OPENCL12
#include "CL/cl.h"
template<
    typename TypeC,
    typename TypeA,
    typename TypeB,
    typename TypeAlpha,
    typename TypeBeta>
class SolutionOpenCL
    : public SolutionGPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  SolutionOpenCL( const Problem & inputProblem );
  ~SolutionOpenCL();

  void makeKernel(
  cl_kernel *kernel,
  cl_command_queue queue,
  const char *kernelSource,
  const char *sourceBuildOptions);
  
  TensileStatus enqueue(
      TensileTensorData tensorDataC,
      TensileTensorDataConst tensorDataA,
      TensileTensorDataConst tensorDataB,
      TensileScalarData alpha,
      TensileScalarData beta,
      TensileControl & ctrl );

  virtual std::string toString( size_t indentLevel ) const = 0;
  virtual std::string toStringDetailXML( size_t indentLevel ) const = 0;

protected:
  cl_kernel kernels[SolutionGPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::maxNumKernels];
  const char *kernelSources[SolutionGPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta>::maxNumKernels];
  cl_int status;

  // constants
  //static const unsigned int workDim = 3;
  // kernels
  //cl_uint numKernels;
  //unsigned int kernelGrid[workDim];
  //unsigned int edge[this->workDim];
  // kernel dimensions
  //size_t workGroup[workDim];
  //size_t microTile[this->workDim];
  //size_t globalWorkSize[this->maxNumKernels][this->workDim];
  //size_t localWorkSize[this->maxNumKernels][this->workDim];
  //unsigned int kernelNumElementsDim0[maxNumKernels]; // local to assign
  //unsigned int kernelNumElementsDim1[maxNumKernels];
  //unsigned int kernelNumElementsDimU[maxNumKernels];
  //void assignWorkSizes(); // moved up to GPU and improved
  // kernel argumets
  //cl_uint numKernelArgs;
  //const void *kernelArgs[this->maxNumKernelArgs];
  //size_t kernelArgSizes[this->maxNumKernelArgs];

};
#endif

/*******************************************************************************
 * TensileSolutionHIP - parent class for HIP solutions
 * for HIP kernels:
 *    - all kernels will accept offsets, to simplify calling
 *    - all kernels will accept alpha/beta, to simplify calling
 *    - leading strides are optional
 ******************************************************************************/
#if Tensile_BACKEND_HIP
template<typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta>
class SolutionHIP : public SolutionGPU<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  SolutionHIP( const Problem & inputProblem );
  ~SolutionHIP();

  virtual TensileStatus enqueue(
      TensileTensorData tensorDataC,
      TensileTensorDataConst tensorDataA,
      TensileTensorDataConst tensorDataB,
      TensileScalarData alpha,
      TensileScalarData beta,
      TensileControl & ctrl ) = 0;

  virtual std::string toString( size_t indentLevel ) const = 0;
  virtual std::string toStringDetailXML( size_t indentLevel ) const = 0;

protected:
  hipError_t status;
  // pointers to kernel functions
  

};
#endif



/*******************************************************************************
 * TensileSolutionLogOnly - used in LOG_ONLY mode
 ******************************************************************************/
#if Tensile_LOGGER_ENABLED
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
class SolutionLogOnly : public SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  SolutionLogOnly( const Problem & inputProblem );
  ~SolutionLogOnly();

  TensileStatus enqueue(
      TensileTensorData tensorDataC,
      TensileTensorDataConst tensorDataA,
      TensileTensorDataConst tensorDataB,
      TensileScalarData alpha,
      TensileScalarData beta,
      TensileControl & ctrl );

  std::string toString( size_t indentLevel ) const;
  std::string toStringDetailXML( size_t indentLevel ) const;

};
#endif






} // end namespace


// cache kernels so they only get compiled once
#if Tensile_BACKEND_OPENCL12
typedef struct KernelMapKey_ {
  cl_context context; // address of context
  cl_device_id device; // address of device
  const char *kernelSource; // address of kernel source
} KernelMapKey;
typedef std::map<KernelMapKey, cl_kernel> KernelMap;
bool operator<(const KernelMapKey & l, const KernelMapKey & r);

#ifdef WIN32
__declspec(thread) extern KernelMap *kernelMap;
#else
extern __thread KernelMap *kernelMap;
#endif

#elif Tensile_BACKEND_HIP
// not needed; pre-compiled
#endif



#include <assert.h>
#if Tensile_BACKEND_OPENCL12
#define CL_CHECK(RET) \
  if(RET != CL_SUCCESS) { \
    printf("OpenCL Error %i on line %u of %s\n", RET, __LINE__, __FILE__); \
    /*assert(false);*/ \
    }
#elif Tensile_BACKEND_HIP
#define CL_CHECK(RET) \
  if(RET != hipSuccess) { \
    printf("HIP Error %i on line %u of %s\n", RET, __LINE__, __FILE__); \
    /*assert(false);*/ \
    }
#endif




/*******************************************************************************
 * TensileSolution - public pimpl
 ******************************************************************************/
struct _TensileSolution {
  Tensile::Solution *pimpl;
};


#endif


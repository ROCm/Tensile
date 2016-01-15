
#ifndef SOLUTION_H
#define SOLUTION_H

#include "Cobalt.h"

#include <string>


/*******************************************************************************
 * CobaltSolution (abstract)
 ******************************************************************************/
struct CobaltSolution {

  CobaltSolution( CobaltProblem inputProblem );
  //virtual ~CobaltSolution() = 0;

  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltControl & ctrl ) = 0;

  virtual std::string toString( size_t indentLevel ) const = 0;

  CobaltProblem problem; // problem used to get this solution

}; // class Solution
//#define COBALT_BACKEND_OPENCL
#ifdef COBALT_BACKEND_OPENCL
#include "CL/cl.h"

/*******************************************************************************
 * CobaltSolutionOpenCL - used in LOG_ONLY mode
 ******************************************************************************/
class CobaltSolutionOpenCL : public CobaltSolution {
public:
  CobaltSolutionOpenCL( CobaltProblem inputProblem );

  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltControl & ctrl );

  virtual std::string toString( size_t indentLevel ) const;

  size_t numKernels;
  const size_t maxNumKernels = 4;
  const size_t workDim = 2;
  cl_kernel kernels[maxNumKernels];
  size_t globalWorkSize[maxNumKernels][workDim];
  size_t localWorkSize[maxNumKernels][workDim];
};

class CobaltSolutionOpenCLDummy : public CobaltSolutionOpenCL {
public:
  CobaltSolutionOpenCLDummy( CobaltProblem inputProblem );

  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltControl & ctrl );

  virtual std::string toString( size_t indentLevel ) const;
};

#endif

#if Cobalt_LOGGER_ENABLED
/*******************************************************************************
 * CobaltSolutionLogOnly - used in LOG_ONLY mode
 ******************************************************************************/
class CobaltSolutionLogOnly : public CobaltSolution {
public:
  CobaltSolutionLogOnly( CobaltProblem inputProblem );

  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltControl & ctrl );

  virtual std::string toString( size_t indentLevel ) const;

};
#endif

#endif
/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef REFERENCE_TENSOR_CONTRACTION_H
#define REFERENCE_TENSOR_CONTRACTION_H
#include "Cobalt.h"
#include "Solution.h"
#include <assert.h>
#include <tuple>

namespace Cobalt {

/*******************************************************************************
 * CobaltSolutionTensorContractionCPU
 * - compute tensor contraction on cpu using simple/slow loops
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
class SolutionTensorContractionCPU : public SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  SolutionTensorContractionCPU( const Problem & inputProblem );

  CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorDataConst tensorDataB,
      CobaltTensorDataConst tensorDataC,
      CobaltScalarData alpha,
      CobaltScalarData beta,
      CobaltControl & ctrl );
  
  std::string toString( size_t indentLevel ) const;
  std::string toStringDetailXML( size_t indentLevel ) const;
};


/*******************************************************************************
 * cobaltGetSolution
 ******************************************************************************/
std::tuple<Cobalt::Solution *, CobaltStatus> getSolutionCPU(
    const Cobalt::Problem & problem );

} // namespace


#endif

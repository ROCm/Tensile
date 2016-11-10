/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef REFERENCE_TENSOR_CONTRACTION_H
#define REFERENCE_TENSOR_CONTRACTION_H
#include "Tensile.h"
#include "Solution.h"
#include <assert.h>
#include <tuple>

namespace Tensile {

/*******************************************************************************
 * TensileSolutionTensorContractionCPU
 * - compute tensor contraction on cpu using simple/slow loops
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB, typename TypeAlpha, typename TypeBeta >
class SolutionTensorContractionCPU : public SolutionTemplate<TypeC,TypeA,TypeB,TypeAlpha,TypeBeta> {
public:
  SolutionTensorContractionCPU( const Problem & inputProblem );

  TensileStatus enqueue(
      TensileTensorData tensorDataA,
      TensileTensorDataConst tensorDataB,
      TensileTensorDataConst tensorDataC,
      TensileScalarData alpha,
      TensileScalarData beta,
      TensileControl & ctrl );
  
  std::string toString( size_t indentLevel ) const;
  std::string toStringDetailXML( size_t indentLevel ) const;
};


/*******************************************************************************
 * tensileGetSolution
 ******************************************************************************/
std::tuple<Tensile::Solution *, TensileStatus> getSolutionCPU(
    const Tensile::Problem & problem );

} // namespace


#endif


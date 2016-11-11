/*******************************************************************************
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

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


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
  SolutionTensorContractionCPU( CobaltProblem inputProblem );


  CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltScalarData alpha,
      CobaltScalarData beta,
      CobaltControl & ctrl );
  
  std::string toString( size_t indentLevel ) const;
 


  //CobaltStatus gemm_batched(
  //  CobaltTensorData tensorDataC,
  //  CobaltTensorData tensorDataA,
  //  CobaltTensorData tensorDataB,
  //  CobaltScalarData alpha,
  //  CobaltScalarData beta,
  //  CobaltControl & ctrl );
  //
  //CobaltStatus gemm(
  //  CobaltTensorData tensorDataC,
  //  CobaltTensorData tensorDataA,
  //  CobaltTensorData tensorDataB,
  //  CobaltScalarData alpha,
  //  CobaltScalarData beta,
  //  CobaltControl & ctrl );
};


/*******************************************************************************
 * cobaltGetSolution
 * need to list all wanted template variants for compiler in this file
 ******************************************************************************/
std::tuple<Cobalt::Solution *, CobaltStatus> getSolutionCPU(
    const Cobalt::Problem & problem );

} // namespace


#endif

#include "Cobalt.h"
#include "Solution.h"
#include <assert.h>


/*******************************************************************************
 * CobaltSolutionTensorContractionCPU
 * - compute tensor contraction on cpu using simple/slow loops
 ******************************************************************************/
class CobaltSolutionTensorContractionCPU : public CobaltSolution {
public:
  CobaltSolutionTensorContractionCPU( CobaltProblem inputProblem );

  CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltScalarData alpha,
      CobaltScalarData beta,
      CobaltControl & ctrl );

  std::string toString( size_t indentLevel ) const;
};

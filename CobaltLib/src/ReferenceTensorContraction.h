
#include "Cobalt.h"
#include "Solution.h"
#include <assert.h>


/*******************************************************************************
 * ReferenceTensorContraction
 * - compute tensor contraction on cpu using simple/slow loops
 ******************************************************************************/
class ReferenceTensorContraction : CobaltSolution {
public:
  ReferenceTensorContraction( CobaltProblem inputProblem );

  virtual CobaltStatus enqueue(
      CobaltTensorData tensorDataA,
      CobaltTensorData tensorDataB,
      CobaltTensorData tensorDataC,
      CobaltControl & ctrl );

  virtual std::string toString( size_t indentLevel ) const;
};

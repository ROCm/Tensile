
//#include "Status.h"
//#include "Problem.h"
#include "Cobalt.h"
#include <assert.h>


CobaltStatus referenceTensorContraction(
  CobaltProblem problem,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltTensorData tensorDataC ) {
  CobaltStatus status;
  status.numCodes = 0;
  // verify that C has the right number of dimensions,
  // i.e., C.dim.size() == A.dim.size() + B.dim.size() - 2*contractedIndices.size()
  //assert( problem.c.dimensions.size() == problem.a.dimensions.size()
  //    + problem.b.dimensions.size() - 2*problem.operation.contractedIndices.size() );
  //
  //// verify contracted indices are of equal size
  //for (size_t i = 0; i < problem.operation.numContractedIndices; i++) {
  //  assert(problem.a.dimensions[problem.operation.contractedIndices[i].a ].size
  //      == problem.b.dimensions[problem.operation.contractedIndices[i].b].size);
  //}

  // verify that each A index accounted for,
  // i.e., A.dim.size() == count(contractedIndices[A]) + count(cIndices[A])
  // verify that each B index accounted for
  // verify that each C index accounted for

  // verify that each contracted index of A appears only once in contractedIndices
  // verify that each contracted index of B appears only once in contractedIndices
  // verify that each index of C

  //A[0,1,2];
  //B[0,1,2];
  //C[0,1];
  //operation.contractedIndices(1,0);
  //operation.cIndices[0](A,0);
  //operation.cIndices[1](B,1);

  //for ( size_t i = 0; i < 
  return status;
} // referenceTensorContraction

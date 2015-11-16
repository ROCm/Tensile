#include "Cobalt.h"
#include "Logger.h"
#include <assert.h>
#include <stdio.h>

/*******************************************************************************
 * cobaltSetup()
 ******************************************************************************/
CobaltStatus cobaltSetup() {
  CobaltStatus status;
  status.numCodes = 0;
  return status;
}

/*******************************************************************************
 * cobaltTeardown
 ******************************************************************************/
CobaltStatus cobaltTeardown() {
  CobaltStatus status;
  status.numCodes = 0;
  return status;
}

/*******************************************************************************
 * cobaltGetSolution
 ******************************************************************************/
CobaltStatus cobaltGetSolution(
   const CobaltProblem & problem,
    CobaltSolution *solution ) {
  CobaltStatus status;
  status.numCodes = 0;

  // request solution
#if Cobalt_SOLUTIONS_ENABLED
  functionStatus = cobaltGetSolution(this, solution);
#else
  solution = new LogSolution(problem);
  if (status.numCodes < status.maxCodes) {
    status.codes[status.numCodes] = cobaltCodeSolutionsDisabled;
    status.numCodes++;
  }
#endif

#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.logGetSolution(solution, status);
#endif

  return status;
}

/*******************************************************************************
 * cobaltEnqueueSolution
 ******************************************************************************/
CobaltStatus cobaltEnqueueSolution(
    CobaltSolution *solution,
    CobaltTensorData a,
    CobaltTensorData b,
    CobaltTensorData c,
    CobaltControl *ctrl ) {
  CobaltStatus status;
  status.numCodes = 0;

#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.logEnqueueSolution(solution, status, ctrl);
#endif

  return status;
}


/*******************************************************************************
 * cobaltValidateProblem
 ******************************************************************************/
static const char *freeIndexChars = "ijklmnopqrstuvwxyz";
static const char *boundIndexChars = "zyxwvutsrqponmlkjihgfedcba";
CobaltStatus cobaltValidateProblem( CobaltProblem problem ) {
  CobaltStatus status;
  status.numCodes = 0;

  std::string problemState = toString( problem, 0);
  printf("cobaltValidateProblem():\n%s\n", problemState.c_str() );

  // operation agrees with tensorA,B.numDimensions.
  assert(problem.tensorA.numDimensions <= problem.tensorA.maxDimensions);
  assert(problem.tensorB.numDimensions <= problem.tensorB.maxDimensions);
  assert(problem.tensorA.numDimensions == problem.operation.numOperationIndexAssignmentsA);
  assert(problem.tensorB.numDimensions == problem.operation.numOperationIndexAssignmentsB);

  // tensorA,B have same num free indices
  size_t numFreeIndicesA = 0;
  for( size_t i = 0; i < problem.operation.numOperationIndexAssignmentsA; i++) {
    if (problem.operation.operationIndexAssignmentsA[i].type
      == cobaltOperationIndexAssignmentTypeFree ) {
        numFreeIndicesA++;
    }
  }
  size_t numFreeIndicesB = 0;
  for( size_t i = 0; i < problem.operation.numOperationIndexAssignmentsB; i++) {
    if (problem.operation.operationIndexAssignmentsB[i].type
      == cobaltOperationIndexAssignmentTypeFree ) {
        numFreeIndicesB++;
    }
  }
  assert( numFreeIndicesA == numFreeIndicesB );

  // for each tensorC index; tensorA or B has it as a free assinment
  assert(problem.tensorC.numDimensions > 0);
  for (size_t i = 0; i < problem.tensorC.numDimensions; i++) {
    bool freeIndexOfA = false;
    for (size_t j = 0; j < problem.tensorA.numDimensions; j++) {
      if (problem.operation.operationIndexAssignmentsA[i].type==cobaltOperationIndexAssignmentTypeFree
          && problem.operation.operationIndexAssignmentsA[i].index == j) {
        freeIndexOfA = true;
      }
    }
    bool freeIndexOfB = false;
    for (size_t j = 0; j < problem.tensorB.numDimensions; j++) {
      if (problem.operation.operationIndexAssignmentsB[i].type==cobaltOperationIndexAssignmentTypeFree
          && problem.operation.operationIndexAssignmentsB[i].index == j) {
        freeIndexOfA = true;
      }
    }
    assert( freeIndexOfA || freeIndexOfB );
  }

  // for each A index bound to B, B is bound back
  size_t numBoundIndices = 0;
  for( size_t i = 0; i < problem.operation.numOperationIndexAssignmentsA; i++) {
    if (problem.operation.operationIndexAssignmentsA[i].type
        == cobaltOperationIndexAssignmentTypeBound ) {
      size_t boundIndex = problem.operation.operationIndexAssignmentsA[i].index;
      assert( boundIndex < problem.tensorB.numDimensions );
      assert( problem.operation.operationIndexAssignmentsB[boundIndex].type == cobaltOperationIndexAssignmentTypeBound);
      assert( problem.operation.operationIndexAssignmentsB[boundIndex].index == i);
      numBoundIndices++;
    }
  }
  // for each B index bound to A, A is bound back
  for( size_t i = 0; i < problem.operation.numOperationIndexAssignmentsB; i++) {
    if (problem.operation.operationIndexAssignmentsB[i].type
        == cobaltOperationIndexAssignmentTypeBound ) {
      size_t boundIndex = problem.operation.operationIndexAssignmentsB[i].index;
      assert( boundIndex < problem.tensorA.numDimensions );
      assert( problem.operation.operationIndexAssignmentsA[boundIndex].type == cobaltOperationIndexAssignmentTypeBound);
      assert( problem.operation.operationIndexAssignmentsA[boundIndex].index == i);
    }
  }

  // comparison operators working
  assert( problem < problem == false);

  // pretty print
  std::cout << "C[" << freeIndexChars[0];
  for (size_t i = 1; i < problem.tensorC.numDimensions; i++) {
    std::cout << "," << freeIndexChars[i];
  }
  std::cout << "] = Sum(" << boundIndexChars[0];
  for (size_t i = 1; i < numBoundIndices; i++) {
    std::cout << "," << boundIndexChars[i];
  }
  std::cout << ") A[";
  size_t currentBoundIndex = 0;
  size_t aIndexToBoundIndex[CobaltTensor::maxDimensions];
  for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
    if (problem.operation.operationIndexAssignmentsA[i].type == cobaltOperationIndexAssignmentTypeBound) {
      aIndexToBoundIndex[i] = currentBoundIndex;
      currentBoundIndex++;
      std::cout << boundIndexChars[aIndexToBoundIndex[i]];
    } else {
      // free index, which one?
      std::cout << freeIndexChars[problem.operation.operationIndexAssignmentsA[i].index];
    }
    if (i < problem.tensorA.numDimensions-1) {
      std::cout << ",";
    }
  }
  std::cout << "] * B[";
  for (size_t i = 0; i < problem.tensorB.numDimensions; i++) {
    if (problem.operation.operationIndexAssignmentsB[i].type == cobaltOperationIndexAssignmentTypeBound) {
      std::cout << boundIndexChars[aIndexToBoundIndex[problem.operation.operationIndexAssignmentsB[i].index]];
    } else {
      // free index, which one?
      std::cout << freeIndexChars[problem.operation.operationIndexAssignmentsB[i].index];
    }
    if (i < problem.tensorB.numDimensions-1) {
      std::cout << ",";
    }
  }
  std::cout << "]\n";
  return status;
}
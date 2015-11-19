#include "Cobalt.h"
#include <stdio.h>

CobaltTensor createTensorForMatrix(
    bool colMajor, bool trans, size_t numRows, size_t numCols );
CobaltOperation createOperationForGEMM();
void gemm( bool colMajor, bool transA, bool transB,
    size_t M, size_t N, size_t K );


/*******************************************************************************
 * main
 ******************************************************************************/
int main( char * argv[], int argc ) {
  cobaltSetup();
  for (size_t order = 1; order < 2; order++) {
    for (size_t transA = 0; transA < 2; transA++) {
      for (size_t transB = 0; transB < 2; transB++) {
        for (size_t size = 256; size <= 1024; size += 256) {
          gemm(
            order==1, // true means colMajor
            transA==1, // true means do transpose
            transB==1, // true means do transpose
            size, // M
            size, // N
            size);// K
        } // size
      } // transB
    } // transA
  } // order
  cobaltTeardown();
  return 0;

} // main


/*******************************************************************************
 * gemm
 ******************************************************************************/
void gemm(
    bool colMajor,
    bool transA,
    bool transB,
    size_t M,
    size_t N,
    size_t K ) {

  // Matrix A
  size_t numRowsA;
  size_t numColsA;
  if (transA==colMajor) {
    numRowsA = K;
    numColsA = M;
  } else {
    numRowsA = M;
    numColsA = K;
  }

  // Matrix B
  size_t numRowsB;
  size_t numColsB;
  if (transB==colMajor) {
    numRowsB = N;
    numColsB = K;
  } else {
    numRowsB = K;
    numColsB = N;
  }

  // Matrix C
  size_t numRowsC = M;
  size_t numColsC = N;
  if (colMajor) {
    numRowsC = M;
    numColsC = N;
  } else {
    numRowsC = N;
    numColsC = M;
  }

  // problem - tensor
  CobaltProblem problem;
  problem.tensorA = createTensorForMatrix(colMajor, transA, numRowsA, numColsA);
  problem.tensorB = createTensorForMatrix(colMajor, transB, numRowsB, numColsB);
  problem.tensorC = createTensorForMatrix(colMajor,  false, numRowsC, numColsC);

  // problem - operation GEMM
  problem.operation = createOperationForGEMM();

  // problem - device problem
  problem.deviceProfile.numDevices = 1;
  sprintf_s(problem.deviceProfile.devices[0].name, "Hawaii" );
  problem.deviceProfile.devices[0].numComputeUnits = 44;
  problem.deviceProfile.devices[0].clockFrequency = 900; // MHz

  // problem - validate
  CobaltStatus validationStatus = cobaltValidateProblem( problem );
  size_t strLength;
  cobaltStatusToString(validationStatus, nullptr, &strLength);
  char *statusStr = new char[strLength];
  cobaltStatusToString(validationStatus, statusStr, &strLength);
  printf("%s\n", statusStr );
  delete[] statusStr;

  // get solution
  CobaltSolution *solution = nullptr;
  CobaltStatus getSolutionStatus = cobaltGetSolution( problem, solution );

  // control
  CobaltControl ctrl;
  ctrl.numDependencies = 0;

  // data
  CobaltTensorData dataA;
  dataA.data = nullptr;
  CobaltTensorData dataB;
  dataB.data = nullptr;
  CobaltTensorData dataC;
  dataC.data = nullptr;

  // enqueue solution
  CobaltStatus enqueueSolutionStatus = cobaltEnqueueSolution(
      solution, dataA, dataB, dataC, &ctrl);
}

// TODO - debug this
CobaltTensor createTensorForMatrix(
    bool colMajor,
    bool trans,
    size_t numRows,
    size_t numCols
    ) {
  CobaltTensor tensor;
  tensor.precision = cobaltPrecisionSingle;
  tensor.numDimensions = 2;
  if (colMajor != trans) {
    // 0th dimension is col
    tensor.dimensions[0].stride = 1; // incr to get to next row
    tensor.dimensions[0].size = numRows; // how many times can we incr in this dimension
    // 1th dimensions is row
    tensor.dimensions[1].stride = numRows; // incr to get to next col
    tensor.dimensions[1].size = numCols; // how many times can we incr in this dimension
  } else {
    // 0th dimension is col
    tensor.dimensions[0].stride = 1; // incr to get to next col
    tensor.dimensions[0].size = numCols; // how many time can we incr in this dimension
    // 1th dimensions is row
    tensor.dimensions[1].stride = numCols; // incr to get to next row
    tensor.dimensions[1].size = numRows; // how many times can we incr in this dimension
  }
  return tensor;
}

CobaltOperation createOperationForGEMM() {
  CobaltOperation operation;
  operation.type = cobaltOperationTypeTensorContraction;
  // C[i,j] = Sum_k A[i,k] * B[k,j]
  //   0,u freeA
  //   u,1 freeB
  //       boundA 1
  //       boundB 0
  // numFreeIndices = 2 b/c tensorC rank 2
  operation.numFreeIndicesAB = 1;
  operation.numSummationIndices = 1;
  operation.indexAssignmentsA[0] = 0; // i
  operation.indexAssignmentsA[1] = 2; // j
  operation.indexAssignmentsB[0] = 2; // i
  operation.indexAssignmentsB[1] = 1; // j
  //operation.numSummationIndices = 1;
  //operation.boundIndicesA[0] = 0; // k
  //operation.boundIndicesB[0] = 1; // k

  // C[i,j] = Sum_k A[i,k] * B[k,j]

  //operation.numOperationIndexAssignmentsA = 2;
  //operation.operationIndexAssignmentsA[0].type
  //    = cobaltOperationIndexAssignmentTypeFree;
  //operation.operationIndexAssignmentsA[0].index = 0;
  //operation.operationIndexAssignmentsA[1].type
  //    = cobaltOperationIndexAssignmentTypeBound;
  //operation.operationIndexAssignmentsA[1].index = 0;
  //operation.numOperationIndexAssignmentsB = 2;
  //operation.operationIndexAssignmentsB[0].type
  //    = cobaltOperationIndexAssignmentTypeBound;
  //operation.operationIndexAssignmentsB[0].index = 1;
  //operation.operationIndexAssignmentsB[1].type
  //    = cobaltOperationIndexAssignmentTypeFree;
  //operation.operationIndexAssignmentsB[1].index = 1;
  return operation;
}
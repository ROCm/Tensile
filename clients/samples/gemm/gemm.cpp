#include "Cobalt.h"
#include <stdio.h>

CobaltTensor createTensorForMatrix(
    CobaltDataType dataType,
    size_t initialStride,
    size_t dim0,
    size_t dim1,
    size_t dimBatch );
CobaltOperation createOperationGEMM(
  CobaltDataType dataType,
  bool transA,
  bool transB,
  bool alpha,
  bool beta,
  bool batched );
CobaltProblem createProblemGEMM(
    bool transA,
    bool transB,
    size_t M,
    size_t N,
    size_t K,
    size_t initialStride,
    size_t numBatches,
    bool alpha,
    bool beta,
    CobaltDataType dataType );

/*******************************************************************************
 * main
 ******************************************************************************/
int main( char * argv[], int argc ) {
  // transA, transB, strideMultiple, M, N, K
  const size_t numSizes = 1;
  size_t sizes[] = {960-1, 960, 4096};
  const size_t numStrides = 2;
  size_t initialStrides[] = { 1, 64 };
  const size_t numBatchSizes = 2;
  size_t batches[] = { 1, 3 };
  const size_t numDataTypes = 1;
  const CobaltDataType dataTypes[] = {
    cobaltDataTypeSingle,
    cobaltDataTypeDouble,
    cobaltDataTypeComplexSingle,
    cobaltDataTypeComplexDouble };
  const size_t numAlphas = 1;
  const bool alphas[] = { false, true };
  const size_t numBetas = 1;
  const bool betas[] = { false, true };
  size_t numProblems = 0;
  cobaltSetup("GEMM");
  for (size_t transA = 0; transA < 2; transA++) {
    for (size_t transB = 0; transB < 2; transB++) {
      for (size_t mIdx = 0; mIdx < numSizes; mIdx++) {
        for (size_t nIdx = 0; nIdx < numSizes; nIdx++) {
          for (size_t kIdx = 0; kIdx < numSizes; kIdx++) {
            for (size_t sIdx = 0; sIdx < numStrides; sIdx++) {
              for (size_t dtIdx = 0; dtIdx < numDataTypes; dtIdx++) {
                for (size_t bIdx = 0; bIdx < numBatchSizes; bIdx++) {
                  for (size_t alphaIdx = 0; alphaIdx < numAlphas; alphaIdx++) {
                    for (size_t betaIdx = 0; betaIdx < numBetas; betaIdx++) {
                      size_t M = sizes[mIdx];
                      size_t N = sizes[nIdx];
                      size_t K = sizes[kIdx];
                      if (M != N || M != K || N != K) continue;
                      size_t initStride = initialStrides[sIdx];
                      CobaltDataType dataType = dataTypes[dtIdx];
                      size_t numBatches = batches[bIdx];
                      bool alpha = alphas[alphaIdx];
                      bool beta = betas[betaIdx];
                      CobaltProblem problem = createProblemGEMM(
                          transA==1, // true means do transpose
                          transB==1, // true means do transpose
                          M, N, K,
                          initStride,
                          numBatches,
                          alpha,
                          beta,
                          dataType );
                      CobaltSolution *solution;
                      CobaltStatus status = cobaltGetSolution( problem, &solution );
                      numProblems++;
                    } // beta
                  } // alpha
                } // batch
              } // data type
            } // stride
          } // K
        } // N
      } // M
    } // transB
  } // transA
  printf("Num Problems: %u\n", numProblems );
  cobaltTeardown();

  return 0;

} // main


/*******************************************************************************
 * createProblemGEMM
 ******************************************************************************/
CobaltProblem createProblemGEMM(
    bool transA,
    bool transB,
    size_t M,
    size_t N,
    size_t K,
    size_t initialStride,
    size_t numBatches,
    bool alpha,
    bool beta,
    CobaltDataType dataType ) {

  // problem - tensor
  CobaltProblem problem;
  problem.tensorA = createTensorForMatrix(
    dataType,
    initialStride,
    transA ? K : M,
    transA ? M : K,
    numBatches );
  problem.tensorB = createTensorForMatrix(
    dataType,
    initialStride,
    transB ? K : N,
    transB ? N : K,
    numBatches );
  problem.tensorC = createTensorForMatrix(
    dataType,
    initialStride,
    M,
    N,
    numBatches );

  // problem - operation GEMM
  problem.operation = createOperationGEMM(
    dataType, transA, transB, alpha, beta, numBatches > 1);

  // problem - device problem
  problem.deviceProfile.numDevices = 1;
  sprintf_s(problem.deviceProfile.devices[0].name, "Hawaii" );
  problem.deviceProfile.devices[0].numComputeUnits = 44;
  problem.deviceProfile.devices[0].clockFrequency = 900; // MHz

  // problem - validate
  CobaltStatus validationStatus = cobaltValidateProblem( problem );
  unsigned int strLength;
  cobaltStatusToString(validationStatus, nullptr, &strLength);
  char *statusStr = new char[strLength];
  cobaltStatusToString(validationStatus, statusStr, &strLength);
  printf("%s\n", statusStr );
  delete[] statusStr;
  if (validationStatus != cobaltStatusSuccess) {
    cobaltValidateProblem( problem );
  }

  return problem;
}

// TODO - debug this
CobaltTensor createTensorForMatrix(
    CobaltDataType dataType,
    size_t initialStride,
    size_t dim0,
    size_t dim1,
    size_t dimBatch
    ) {
  CobaltTensor tensor;
  tensor.dataType = dataType;
  tensor.numDimensions = 2;
  tensor.dimensions[0].stride = (unsigned int) initialStride;
  tensor.dimensions[0].size = (unsigned int) dim0;
  tensor.dimensions[1].stride = (tensor.dimensions[0].stride*tensor.dimensions[0].size);
  tensor.dimensions[1].size = (unsigned int) dim1;

  if (dimBatch > 1) {
    tensor.numDimensions++;
    tensor.dimensions[2].stride = tensor.dimensions[1].stride*tensor.dimensions[1].size;
    tensor.dimensions[2].size = (unsigned int) dimBatch;
  }
  return tensor;
}

CobaltOperation createOperationGEMM(
  CobaltDataType dataType,
  bool transA,
  bool transB,
  bool alpha,
  bool beta,
  bool batched ) {
  CobaltOperation operation;
  operation.type = cobaltOperationTypeContraction;
  operation.useAlpha = alpha;
  operation.alphaType = dataType;
  //operation.alpha = nullptr;
  operation.useBeta = beta;
  operation.betaType = dataType;
  //operation.beta = nullptr;
  operation.numIndicesFree = 2;
  operation.numIndicesSummation = 1;
  if (batched) {
  operation.numIndicesBatch = 1;
  operation.indexAssignmentsA[0] = transA ? 3 : 0;
  operation.indexAssignmentsA[1] = transA ? 0 : 3;
  operation.indexAssignmentsA[2] = 2;
  operation.indexAssignmentsB[0] = transB ? 1 : 3;
  operation.indexAssignmentsB[1] = transB ? 3 : 1;
  operation.indexAssignmentsB[2] = 2;
  } else {
  operation.numIndicesBatch = 0;
  operation.indexAssignmentsA[0] = transA ? 2 : 0;
  operation.indexAssignmentsA[1] = transA ? 0 : 2;
  operation.indexAssignmentsB[0] = transB ? 1 : 2;
  operation.indexAssignmentsB[1] = transB ? 2 : 1;
  }
  return operation;
}
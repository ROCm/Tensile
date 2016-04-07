#include "Cobalt.h"
#include <stdio.h>
#include <string>
#include <vector>
#include <array>

CobaltTensor createTensorForMatrix(
    CobaltDataType dataType,
    size_t initialStride,
    size_t dim0,
    size_t dim1,
    size_t dimBatch );
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
    CobaltDataType dataTypeC,
    CobaltDataType dataTypeA,
    CobaltDataType dataTypeB
  );


const size_t sgemmSizeBoundsSize = 6;
const size_t sgemmSizeBounds[][2] = {
  {16 * 1, 1536},
  {16 * 2, 2048},
  {16 * 3, 2048},
  {16 * 4, 3072},
  {16 * 5, 3072},
  {16 * 6, 4096} };

#if 0
  sgemm
  16*1, 0, 1.5k
  16*2, 0, 2k
  16*3, 0, 2k
  16*4, 0, 3k
  16*5, 0, 3k
  16*6, 0, 4k

  dgemm
  16*1, 0, 1.5k
  16*2, 0, 3k
  16*3, 0, 4k
  16*4, 0, 5k
  16*5, 0, 4k

  cgemm
  16*1, 0, 1k
  16*2, 0, 2.5k
  16*3, 0, 3k
  16*4, 0, 4k
  16*5, 0, 4k
  16*6, 0, 4k

  zgemm
  16*1,2,3,4, 0, 3k
  
#endif

/*******************************************************************************
 * main
 ******************************************************************************/
int main( char * argv[], int argc ) {
  // transA, transB, strideMultiple, M, N, K
  std::vector<std::array<size_t,3>> sizes;
  for (size_t i = 16; i < 2048+16*12; i+= 16) {
    bool useSize = false;
    for (size_t j = 0; j < sgemmSizeBoundsSize; j++) {
      if (i < sgemmSizeBounds[j][1] && i % sgemmSizeBounds[j][0]==0) {
        useSize = true;
        break;
      }
    }
    if (useSize) {
      sizes.push_back({ i, i, i });
    }
  }

  const size_t numStrides = 1;
  size_t initialStrides[] = { 1, 2 }; // , 64 };
  const size_t numBatchSizes = 1;
  size_t batches[] = { 1, 2 };
  const size_t numDataTypes = 1;
  const CobaltDataType dataTypes[][3] = {
    { cobaltDataTypeSingle, cobaltDataTypeSingle, cobaltDataTypeSingle },
    { cobaltDataTypeDouble, cobaltDataTypeDouble, cobaltDataTypeDouble },
    
    { cobaltDataTypeComplexSingle, cobaltDataTypeComplexSingle, cobaltDataTypeComplexSingle },
    { cobaltDataTypeComplexDouble, cobaltDataTypeComplexDouble, cobaltDataTypeComplexDouble },

    { cobaltDataTypeComplexSingle, cobaltDataTypeComplexConjugateSingle, cobaltDataTypeComplexSingle },
    { cobaltDataTypeComplexSingle, cobaltDataTypeComplexSingle, cobaltDataTypeComplexConjugateSingle },
    { cobaltDataTypeComplexSingle, cobaltDataTypeComplexConjugateSingle, cobaltDataTypeComplexConjugateSingle },

    { cobaltDataTypeComplexDouble, cobaltDataTypeComplexConjugateDouble, cobaltDataTypeComplexDouble },
    { cobaltDataTypeComplexDouble, cobaltDataTypeComplexDouble, cobaltDataTypeComplexConjugateDouble },
    { cobaltDataTypeComplexDouble, cobaltDataTypeComplexConjugateDouble, cobaltDataTypeComplexConjugateDouble }
  };
  const size_t numAlphas = 1;
  const bool alphas[] = { true, false };
  const size_t numBetas = 1;
  const bool betas[] = { true, false };
  const size_t numTransA = 2;
  const bool transAs[] = {false, true};
  const size_t numTransB = 2;
  const bool transBs[] = {true, false};
  size_t numProblems = 0;
  std::string logFilePath = Cobalt_DIR_PROBLEMS;
  logFilePath += "/GEMM_log.xml";
  cobaltSetup(logFilePath.c_str());
  for (size_t transA = 0; transA < numTransA; transA++) {
    for (size_t transB = 0; transB < numTransB; transB++) {
      for (size_t mIdx = 0; mIdx < sizes.size(); mIdx++) {
        for (size_t sIdx = 0; sIdx < numStrides; sIdx++) {
          for (size_t dtIdx = 0; dtIdx < numDataTypes; dtIdx++) {
            for (size_t bIdx = 0; bIdx < numBatchSizes; bIdx++) {
              for (size_t alphaIdx = 0; alphaIdx < numAlphas; alphaIdx++) {
                for (size_t betaIdx = 0; betaIdx < numBetas; betaIdx++) {
                  size_t numBatches = batches[bIdx];
                  size_t M = sizes[mIdx][0];
                  size_t N = sizes[mIdx][1];
                  size_t K = sizes[mIdx][2];
                  //if (M != N || M != K || N != K) continue;
                  size_t initStride = initialStrides[sIdx];
                  bool alpha = alphas[alphaIdx];
                  bool beta = betas[betaIdx];
                  CobaltProblem problem = createProblemGEMM(
                      transAs[transA],
                      transBs[transB],
                      M, N, K,
                      initStride,
                      numBatches,
                      alpha,
                      beta,
                      dataTypes[dtIdx][0],
                      dataTypes[dtIdx][1],
                      dataTypes[dtIdx][2]
                    );
                  //unsigned int nameSize;
                  //cobaltProblemToString(problem, nullptr, &nameSize);
                  //char *nameStr = new char[nameSize];
                  //cobaltProblemToString(problem, nameStr, &nameSize);
                  //delete[] nameStr;
                  
                  CobaltStatus status;
                  CobaltSolution solution = cobaltGetSolutionForProblem( problem, &status );
                  numProblems++;
                } // beta
              } // alpha
            } // batch
          } // data type
        } // stride
      } // M
    } // transB
  } // transA
  printf("Num Problems: %llu\n", numProblems );
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
    CobaltDataType dataTypeC,
    CobaltDataType dataTypeA,
    CobaltDataType dataTypeB
  ) {

  // problem - tensor
  CobaltTensor tensorC = createTensorForMatrix(
    dataTypeC,
    initialStride,
    M,
    N,
    numBatches);
  CobaltTensor tensorA = createTensorForMatrix(
    dataTypeA,
    initialStride,
    transA ? K : M,
    transA ? M : K,
    numBatches );
  CobaltTensor tensorB = createTensorForMatrix(
    dataTypeB,
    initialStride,
    transB ? N : K,
    transB ? K : N,
    numBatches );

  // operation
  CobaltOperationType operationType = cobaltOperationTypeContraction;
  CobaltDataType alphaType = alpha ? dataTypeC : cobaltDataTypeNone;
  CobaltDataType betaType = beta ? dataTypeC : cobaltDataTypeNone;
  unsigned int indexAssignmentsA[CobaltTensor::maxDimensions];
  unsigned int indexAssignmentsB[CobaltTensor::maxDimensions];
  if (numBatches > 1) {
    indexAssignmentsA[0] = transA ? 3 : 0;
    indexAssignmentsA[1] = transA ? 0 : 3;
    indexAssignmentsA[2] = 2;
    indexAssignmentsB[0] = transB ? 1 : 3;
    indexAssignmentsB[1] = transB ? 3 : 1;
    indexAssignmentsB[2] = 2;
  } else {
    indexAssignmentsA[0] = transA ? 2 : 0;
    indexAssignmentsA[1] = transA ? 0 : 2;
    indexAssignmentsB[0] = transB ? 1 : 2;
    indexAssignmentsB[1] = transB ? 2 : 1;
  }

  // problem - device problem
  CobaltDeviceProfile deviceProfile = cobaltCreateEmptyDeviceProfile();
  deviceProfile.numDevices = 1;
  sprintf_s(deviceProfile.devices[0].name, "Hawaii" );


  CobaltStatus status;
  CobaltProblem problem = cobaltCreateProblem(
      tensorC,
      tensorA,
      tensorB,
      indexAssignmentsA,
      indexAssignmentsB,
      operationType,
      alphaType,
      betaType,
      deviceProfile,
      &status );
  cobaltStatusCheck(status);
  unsigned int problemStringSize;
  cobaltProblemToString(problem, nullptr, &problemStringSize);
  char *problemString = new char[problemStringSize];
  cobaltProblemToString(problem, problemString, &problemStringSize);
  printf("%4llux%4llux%4llu %s\n", M, N, K, problemString);
  delete[] problemString;


  // problem - validate
  CobaltStatus validationStatus = cobaltValidateProblem( problem );
  cobaltStatusCheck(validationStatus);
  //unsigned int strLength;
  //cobaltStatusToString(validationStatus, nullptr, &strLength);
  //char *statusStr = new char[strLength];
  //cobaltStatusToString(validationStatus, statusStr, &strLength);
  //printf("%s\n", statusStr );
  //delete[] statusStr;
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
  CobaltTensor tensor = cobaltCreateEmptyTensor();
  tensor.dataType = dataType;
  tensor.numDimensions = 2;
  tensor.dimensions[0].stride = (unsigned int)initialStride;
  tensor.dimensions[0].size = (unsigned int)dim0;
  tensor.dimensions[1].stride = (unsigned int)(tensor.dimensions[0].stride*tensor.dimensions[0].size*initialStride);
  tensor.dimensions[1].size = (unsigned int)dim1;

  if (dimBatch > 1) {
    tensor.numDimensions++;
    tensor.dimensions[2].stride = (unsigned int)(tensor.dimensions[1].stride*tensor.dimensions[1].size*initialStride);
    tensor.dimensions[2].size = (unsigned int) dimBatch;
  }
  return tensor;
}

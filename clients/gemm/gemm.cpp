#include "Cobalt.h"
#include "dnn_gemms.h"
#include <cstdio>
#include <string>
#include <vector>
#include <array>

#define ULL (unsigned long long)
void createAppXMLForExactMatch(
    CobaltDataType typeC,
    CobaltDataType typeA,
    CobaltDataType typeB,
    bool transA,
    bool transB,
    bool alpha,
    bool beta,
    size_t numBatches,
    size_t initStride,
    const std::vector<std::array<size_t, 3>> & sizes
);
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
    bool useOffsets,
    CobaltDataType dataTypeC,
    CobaltDataType dataTypeA,
    CobaltDataType dataTypeB
  );
unsigned int addGEMMCombinatorics();
unsigned int addGEMMList();
char cobaltDataTypeToChar(CobaltDataType t);
size_t numProblems;

// Device Profiles
unsigned int numProfiles;
unsigned int selectedProfile;
CobaltDeviceProfile *profiles;


/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char * argv[] ) {
  selectedProfile = 0;
  if (argc > 1) {
    selectedProfile = atol(argv[1]);
  }


  cobaltEnumerateDeviceProfiles(nullptr, &numProfiles);
  profiles = new CobaltDeviceProfile[numProfiles];
  cobaltEnumerateDeviceProfiles(profiles, &numProfiles);
  printf("CobaltDeviceProfiles:\n");
  for (unsigned int i = 0; i < numProfiles; i++) {
    printf("  (%2u) %11s: %3u CUs @ %5u MHz = %5.0f GFlop/s%s\n",
      i,
      profiles[i].devices[0].name,
      profiles[i].devices[0].numComputeUnits,
      profiles[i].devices[0].clockFrequency,
      1.0*profiles[i].devices[0].numComputeUnits*profiles[i].devices[0].clockFrequency*profiles[i].devices[0].flopsPerClock / 1000.f,
      (i==selectedProfile) ? " (selected)" : ""
      );
  }
  if (selectedProfile >= numProfiles) {
    printf("selectedProfile (%u) exceeds numProfiles (%u); aborting!", selectedProfile, numProfiles);
    return 1;
  }

#if 0
  unsigned int numGemmList = addGEMMList();
  printf("GEMM List: %u\n", numGemmList );
#else
  numProblems = 0;
  addGEMMCombinatorics();
  printf("Num GEMM Problems: %u\n", static_cast<unsigned int>(numProblems));
#endif
  //cobaltTeardown();
}

// initially just for DNN, hence single precision
unsigned int addGEMMList() {

  std::string logFilePath = Cobalt_DIR_PROBLEMS;
  logFilePath += "/GEMM.xml";
  cobaltSetup(logFilePath.c_str());

  for (unsigned int i = 0; i < num_gemm_params; i++) {
    // get parameters from list
    size_t M = gemm_params[i][0];
    size_t N = gemm_params[i][1];
    size_t K = gemm_params[i][2];
    bool transA = gemm_params[i][3] == 1;
    bool transB = gemm_params[i][4] == 1;
    size_t initialStride = 1;
    size_t numBatches = gemm_params[i][10];
    bool alpha = gemm_params[i][8] == 1;
    bool beta = gemm_params[i][9] == 1;
    bool useOffsets = true;
    CobaltDataType dataTypeC = cobaltDataTypeSingle;
    CobaltDataType dataTypeA = cobaltDataTypeSingle;
    CobaltDataType dataTypeB = cobaltDataTypeSingle;
    
    // create problem from parameters
    CobaltProblem problem = createProblemGEMM(
      transA,
      transB,
      M, N, K,
      initialStride,
      numBatches,
      alpha,
      beta,
      useOffsets,
      dataTypeC,
      dataTypeA,
      dataTypeB );

    // send problem to logger
    CobaltSolution solution;
    CobaltStatus status = cobaltGetSolutionForProblem( &solution, problem );
  }
  cobaltTeardown();

  return num_gemm_params;
}


unsigned int addGEMMCombinatorics() {
  // transA, transB, strideMultiple, M, N, K


#if 0
  sizes.push_back( {5760, 5760, 5760 });
  //sizes.push_back({ 96*3, 96*2, 96*1 });
#endif

#if 0
  // validation
  sizes.push_back( { 96*3  , 96*2  , 96*1   });
  sizes.push_back( { 96*3-1, 96*2  , 96*1   });
  sizes.push_back( { 96*3  , 96*2-1, 96*1   });
  sizes.push_back( { 96*3  , 96*2  , 96*1-1 });
  sizes.push_back( { 96*3-1, 96*2-1, 96*1   });
  sizes.push_back( { 96*3-1, 96*2  , 96*1-1 });
  sizes.push_back( { 96*3  , 96*2-1, 96*1-1 });
  sizes.push_back( { 96*3-1, 96*2-1, 96*1-1 });
#endif

  // each api is own test

  // how many problem options
  const size_t numStrides     = 2; // 2
  const size_t numBatchSizes  = 2; // 2
  const size_t numDataTypes   = 2; // 10
  const size_t numAlphas      = 1; // 1
  const size_t numBetas       = 1; // 2
  const size_t numTransA      = 2; // 2
  const size_t numTransB      = 2; // 2

  // problem options
  size_t initialStrides[] = { 1, 2 }; // , 64 };
  size_t batches[] = { 1, 2 };
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
  const bool alphas[] = { true };
  const bool betas[] = { true, false };
  const bool transAs[] = { false, true };
  const bool transBs[] = { true, false };

  // create problem for each combination
  unsigned int numProblems = 0;
  for (size_t transAIdx = 0; transAIdx < numTransA; transAIdx++) {
    for (size_t transBIdx = 0; transBIdx < numTransB; transBIdx++) {
      for (size_t sIdx = 0; sIdx < numStrides; sIdx++) {
        for (size_t dtIdx = 0; dtIdx < numDataTypes; dtIdx++) {
          for (size_t bIdx = 0; bIdx < numBatchSizes; bIdx++) {
            for (size_t alphaIdx = 0; alphaIdx < numAlphas; alphaIdx++) {
              for (size_t betaIdx = 0; betaIdx < numBetas; betaIdx++) {

#if 1
				  std::vector<std::array<size_t, 3>> sizes;
				  size_t stride = 16;
				  size_t stride_incr = 16; // 0->1440, 16->108, 32->76
				  size_t sizeMax = 5760 / batches[bIdx];
				  for (size_t i = stride; i <= sizeMax; i += stride, stride += stride_incr) {
					  sizes.push_back({ i, i, i }); // exact tile, exact unroll
					  sizes.push_back({ i, i, i - 1 }); // exact tile, fallback unroll
					  sizes.push_back({ i - 1, i - 1, i }); // fallback tile, exact unroll
					  sizes.push_back({ i - 1, i - 1, i - 1 }); // fallback tile, fallback unroll
				  }
#endif



                createAppXMLForExactMatch(
                    dataTypes[dtIdx][0],
                    dataTypes[dtIdx][1],
                    dataTypes[dtIdx][2],
                    transAs[transAIdx],
                    transBs[transBIdx],
                    alphas[alphaIdx],
                    betas[betaIdx],
                    batches[bIdx],
                    initialStrides[sIdx],
                    sizes );

              } // beta
            } // alpha
          } // batch
        } // data type
      } // stride
    } // transB
  } // transA

  return numProblems;

} // main


void createAppXMLForExactMatch(
    CobaltDataType typeC,
    CobaltDataType typeA,
    CobaltDataType typeB,
    bool transA,
    bool transB,
    bool alpha,
    bool beta,
    size_t numBatches,
    size_t initStride,
    const std::vector<std::array<size_t, 3>> & sizes
) {
  std::string logFilePath = Cobalt_DIR_PROBLEMS;
  std::string logFileName = "GEMM";
  logFileName += "_";
  logFileName += cobaltDataTypeToChar(typeC);
  logFileName += cobaltDataTypeToChar(typeA);
  logFileName += cobaltDataTypeToChar(typeB);
  logFileName += alpha ? cobaltDataTypeToChar(typeC) : cobaltDataTypeToChar(cobaltDataTypeNone);
  logFileName += beta ? cobaltDataTypeToChar(typeC) : cobaltDataTypeToChar(cobaltDataTypeNone);
  logFileName += "_";
  logFileName += transA ? "T" : "N";
  logFileName += transB ? "T" : "N";
  if (initStride > 1) {
    logFileName += "_strided";
  }
  if (numBatches > 1) {
    logFileName += "_batched";
  }
  logFileName += "_";
  logFileName += std::to_string(sizes.size());
  logFilePath += "/" + logFileName + ".xml";
  printf("%s\n", logFileName.c_str());
  cobaltSetup(logFilePath.c_str());
  
  for (size_t mIdx = 0; mIdx < sizes.size(); mIdx++) {
  
    size_t M = sizes[mIdx][0];
    size_t N = sizes[mIdx][1];
    size_t K = sizes[mIdx][2];
    CobaltProblem problem = createProblemGEMM(
        transA,
        transB,
        M, N, K,
        initStride,
        numBatches,
        alpha,
        beta,
        true, // useOffset
        typeC,
        typeA,
        typeB );
    CobaltSolution solution;
    CobaltStatus status = cobaltGetSolutionForProblem(&solution, problem);
    numProblems++;
  } // size
  
  cobaltTeardown();
}


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
    bool useOffsets,
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

  size_t sizeC = numBatches>1 ? tensorC.dimensions[2].size*tensorC.dimensions[2].stride
                              : tensorC.dimensions[1].size*tensorC.dimensions[1].stride;
  size_t sizeA = numBatches>1 ? tensorA.dimensions[2].size*tensorA.dimensions[2].stride
                              : tensorA.dimensions[1].size*tensorA.dimensions[1].stride;
  size_t sizeB = numBatches>1 ? tensorB.dimensions[2].size*tensorB.dimensions[2].stride
                              : tensorB.dimensions[1].size*tensorB.dimensions[1].stride;

  //printf("sizeTotal = %.1f MB\n", (sizeC+sizeA+sizeB)/1024.0/1024.0);
  //printf("    sizeC = %.1f MB\n", sizeC/1024.0/1024.0);
  //printf("    sizeA = %.1f MB\n", sizeA/1024.0/1024.0);
  //printf("    sizeB = %.1f MB\n", sizeB/1024.0/1024.0);
  //printf("\n");

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
  
  CobaltProblem problem;
  CobaltStatus status = cobaltCreateProblem(
      &problem,
      tensorC,
      tensorA,
      tensorB,
      indexAssignmentsA,
      indexAssignmentsB,
      operationType,
      alphaType,
      betaType,
      useOffsets,
      profiles[selectedProfile] );
  cobaltStatusCheck(status);
  unsigned int problemStringSize;
  cobaltProblemToString(problem, nullptr, &problemStringSize);
  char *problemString = new char[problemStringSize];
  cobaltProblemToString(problem, problemString, &problemStringSize);
  printf("%4llux%4llux%4llu %s\n", ULL M, ULL N, ULL K, problemString);
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

char cobaltDataTypeToChar(CobaltDataType t) {
  switch (t) {
  case cobaltDataTypeHalf:                    return 'H';
  case cobaltDataTypeSingle:                  return 'S';
  case cobaltDataTypeDouble:                  return 'D';
  case cobaltDataTypeComplexHalf:             return 'Q';
  case cobaltDataTypeComplexSingle:           return 'C';
  case cobaltDataTypeComplexDouble:           return 'Z';
  case cobaltDataTypeComplexConjugateHalf:    return 'W';
  case cobaltDataTypeComplexConjugateSingle:  return 'X';
  case cobaltDataTypeComplexConjugateDouble:  return 'Y';
  default:                                    return '0';
  }
}

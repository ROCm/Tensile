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

#include "Tensile.h"
#include "dnn_gemms.h"
#include <cstdio>
#include <string>
#include <vector>
#include <array>

#define ULL (unsigned long long)
void createAppXMLForExactMatch(
    TensileDataType typeC,
    TensileDataType typeA,
    TensileDataType typeB,
    bool transA,
    bool transB,
    bool alpha,
    bool beta,
    size_t numBatches,
    size_t initStride,
    const std::vector<std::array<size_t, 3>> & sizes
);
TensileTensor createTensorForMatrix(
    TensileDataType dataType,
    size_t initialStride,
    size_t dim0,
    size_t dim1,
    size_t dimBatch );
TensileProblem createProblemGEMM(
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
    TensileDataType dataTypeC,
    TensileDataType dataTypeA,
    TensileDataType dataTypeB
  );
unsigned int addGEMMCombinatorics();
unsigned int addGEMMList();
char tensileDataTypeToChar(TensileDataType t);
size_t numProblems;

// Device Profiles
unsigned int numProfiles;
unsigned int selectedProfile;
TensileDeviceProfile *profiles;


/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char * argv[] ) {
  selectedProfile = 0;
  if (argc > 1) {
    selectedProfile = atol(argv[1]);
  }


  tensileEnumerateDeviceProfiles(nullptr, &numProfiles);
  profiles = new TensileDeviceProfile[numProfiles];
  tensileEnumerateDeviceProfiles(profiles, &numProfiles);
  printf("TensileDeviceProfiles:\n");
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
  //tensileTeardown();
}

// initially just for DNN, hence single precision
unsigned int addGEMMList() {

  std::string logFilePath = Tensile_DIR_PROBLEMS;
  logFilePath += "/GEMM.xml";
  tensileSetup(logFilePath.c_str());

  for (unsigned int i = 0; i < num_gemm_params; i++) {
    // get parameters from list
    size_t M = gemm_params[i][0];
    size_t N = gemm_params[i][1];
    size_t K = gemm_params[i][2];
    bool transA = gemm_params[i][3] == 1;
    bool transB = gemm_params[i][4] == 1;
    size_t initialStride = 1;
    size_t numBatches = gemm_params[i][5];
    bool alpha = true;
    bool beta = true;
    bool useOffsets = true;
    TensileDataType dataTypeC = tensileDataTypeSingle;
    TensileDataType dataTypeA = tensileDataTypeSingle;
    TensileDataType dataTypeB = tensileDataTypeSingle;
    
    // create problem from parameters
    TensileProblem problem = createProblemGEMM(
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
    TensileSolution solution;
    TensileStatus status = tensileGetSolutionForProblem( &solution, problem );
    tensileStatusCheck(status);
  }
  tensileTeardown();

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
  const unsigned int numSizeLimits = 8;
  const size_t sizeLimits[][2] = {
      {  16, 1024 },
      {  32, 1536 },
      {  48, 3360 },
      {  64, 5760 },
      {  80, 5760 },
      {  96, 5760 },
      { 112, 5760 },
      { 128, 5760 }
  };

  // how many problem options
  const size_t numStrides     = 1; // 2
  const size_t numBatchSizes  = 1; // 2
  const size_t numDataTypes   = 1; // 10
  const size_t numAlphas      = 1; // 1
  const size_t numBetas       = 1; // 2
  const size_t numTransA      = 1; // 2
  const size_t numTransB      = 1; // 2

  // problem options
  size_t initialStrides[] = { 1, 2 }; // , 64 };
  size_t batches[] = { 1, 2 };
  const TensileDataType dataTypes[][3] = {
    { tensileDataTypeSingle, tensileDataTypeSingle, tensileDataTypeSingle },
    { tensileDataTypeDouble, tensileDataTypeDouble, tensileDataTypeDouble },
    
    { tensileDataTypeComplexSingle, tensileDataTypeComplexSingle, tensileDataTypeComplexSingle },
    { tensileDataTypeComplexDouble, tensileDataTypeComplexDouble, tensileDataTypeComplexDouble },

    { tensileDataTypeComplexSingle, tensileDataTypeComplexConjugateSingle, tensileDataTypeComplexSingle },
    { tensileDataTypeComplexSingle, tensileDataTypeComplexSingle, tensileDataTypeComplexConjugateSingle },
    { tensileDataTypeComplexSingle, tensileDataTypeComplexConjugateSingle, tensileDataTypeComplexConjugateSingle },

    { tensileDataTypeComplexDouble, tensileDataTypeComplexConjugateDouble, tensileDataTypeComplexDouble },
    { tensileDataTypeComplexDouble, tensileDataTypeComplexDouble, tensileDataTypeComplexConjugateDouble },
    { tensileDataTypeComplexDouble, tensileDataTypeComplexConjugateDouble, tensileDataTypeComplexConjugateDouble }
  };
  const bool alphas[] = { true };
  const bool betas[] = { false };
  const bool transAs[] = { false, true };
  const bool transBs[] = { false, false };

  // create problem for each combination
  numProblems = 0;
  for (size_t transAIdx = 0; transAIdx < numTransA; transAIdx++) {
    for (size_t transBIdx = 0; transBIdx < numTransB; transBIdx++) {
      for (size_t sIdx = 0; sIdx < numStrides; sIdx++) {
        for (size_t dtIdx = 0; dtIdx < numDataTypes; dtIdx++) {
          for (size_t bIdx = 0; bIdx < numBatchSizes; bIdx++) {
            for (size_t alphaIdx = 0; alphaIdx < numAlphas; alphaIdx++) {
              for (size_t betaIdx = 0; betaIdx < numBetas; betaIdx++) {

                std::vector<std::array<size_t, 3>> sizes;
#if 0
                size_t stride = 16;
                size_t stride_incr = 0; // 0->1440, 16->108, 32->76
                size_t sizeMax = 5760 / batches[bIdx];
                for (size_t i = stride; i <= sizeMax; i += stride, stride += stride_incr) {
                  bool sizeValid = false;
                  for (unsigned int s = 0; s < numSizeLimits; s++) {
                    if (i % sizeLimits[s][0] == 0 && i <= sizeLimits[s][1]) {
                      sizeValid = true;
                      break;
                    }
                  }
                  if (sizeValid) {
                    sizes.push_back({ i, i, i }); // exact tile, exact unroll
                    sizes.push_back({ i, i, i - 1 }); // exact tile, fallback unroll
                    sizes.push_back({ i - 1, i - 1, i - 1 }); // fallback tile, fallback unroll
                  }
                }
#else
                //sizes.push_back({ 5760, 5760, 5760 });
                sizes.push_back({ 16, 16, 16 }); // all good sizes and square
                //sizes.push_back({ 131, 257, 37 }); // all bad sizes and non-square
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
    TensileDataType typeC,
    TensileDataType typeA,
    TensileDataType typeB,
    bool transA,
    bool transB,
    bool alpha,
    bool beta,
    size_t numBatches,
    size_t initStride,
    const std::vector<std::array<size_t, 3>> & sizes
) {
  std::string logFilePath = Tensile_DIR_PROBLEMS;
  std::string logFileName = "GEMM";
  logFileName += "_";
  logFileName += tensileDataTypeToChar(typeC);
  logFileName += tensileDataTypeToChar(typeA);
  logFileName += tensileDataTypeToChar(typeB);
  logFileName += alpha ? tensileDataTypeToChar(typeC) : tensileDataTypeToChar(tensileDataTypeNone);
  logFileName += beta ? tensileDataTypeToChar(typeC) : tensileDataTypeToChar(tensileDataTypeNone);
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
  tensileSetup(logFilePath.c_str());
  
  for (size_t mIdx = 0; mIdx < sizes.size(); mIdx++) {
  
    size_t M = sizes[mIdx][0];
    size_t N = sizes[mIdx][1];
    size_t K = sizes[mIdx][2];
    TensileProblem problem = createProblemGEMM(
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
    TensileSolution solution;
    TensileStatus status = tensileGetSolutionForProblem(&solution, problem);
    tensileStatusCheck(status);
    numProblems++;
  } // size
  
  tensileTeardown();
}


/*******************************************************************************
 * createProblemGEMM
 ******************************************************************************/
TensileProblem createProblemGEMM(
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
    TensileDataType dataTypeC,
    TensileDataType dataTypeA,
    TensileDataType dataTypeB
  ) {

  // problem - tensor
  TensileTensor tensorC = createTensorForMatrix(
    dataTypeC,
    initialStride,
    M,
    N,
    numBatches);
  TensileTensor tensorA = createTensorForMatrix(
    dataTypeA,
    initialStride,
    transA ? K : M,
    transA ? M : K,
    numBatches );
  TensileTensor tensorB = createTensorForMatrix(
    dataTypeB,
    initialStride,
    transB ? N : K,
    transB ? K : N,
    numBatches );

  //size_t sizeC = numBatches>1 ? tensorC.dimensions[2].size*tensorC.dimensions[2].stride
  //                            : tensorC.dimensions[1].size*tensorC.dimensions[1].stride;
  //size_t sizeA = numBatches>1 ? tensorA.dimensions[2].size*tensorA.dimensions[2].stride
  //                            : tensorA.dimensions[1].size*tensorA.dimensions[1].stride;
  //size_t sizeB = numBatches>1 ? tensorB.dimensions[2].size*tensorB.dimensions[2].stride
  //                            : tensorB.dimensions[1].size*tensorB.dimensions[1].stride;

  //printf("sizeTotal = %.1f MB\n", (sizeC+sizeA+sizeB)/1024.0/1024.0);
  //printf("    sizeC = %.1f MB\n", sizeC/1024.0/1024.0);
  //printf("    sizeA = %.1f MB\n", sizeA/1024.0/1024.0);
  //printf("    sizeB = %.1f MB\n", sizeB/1024.0/1024.0);
  //printf("\n");

  // operation
  TensileOperationType operationType = tensileOperationTypeContraction;
  TensileDataType alphaType = alpha ? dataTypeC : tensileDataTypeNone;
  TensileDataType betaType = beta ? dataTypeC : tensileDataTypeNone;
  unsigned int indexAssignmentsA[TensileTensor::maxDimensions];
  unsigned int indexAssignmentsB[TensileTensor::maxDimensions];
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
  
  TensileProblem problem;
  TensileStatus status = tensileCreateProblem(
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
  tensileStatusCheck(status);
  unsigned int problemStringSize;
  tensileProblemToString(problem, nullptr, &problemStringSize);
  char *problemString = new char[problemStringSize];
  tensileProblemToString(problem, problemString, &problemStringSize);
  printf("%4llux%4llux%4llu %s\n", ULL M, ULL N, ULL K, problemString);
  delete[] problemString;

  // problem - validate
  TensileStatus validationStatus = tensileValidateProblem( problem );
  tensileStatusCheck(validationStatus);
  //unsigned int strLength;
  //tensileStatusToString(validationStatus, nullptr, &strLength);
  //char *statusStr = new char[strLength];
  //tensileStatusToString(validationStatus, statusStr, &strLength);
  //printf("%s\n", statusStr );
  //delete[] statusStr;
  if (validationStatus != tensileStatusSuccess) {
    tensileValidateProblem( problem );
  }

  return problem;
}


TensileTensor createTensorForMatrix(
    TensileDataType dataType,
    size_t initialStride,
    size_t dim0,
    size_t dim1,
    size_t dimBatch
    ) {
  TensileTensor tensor = tensileCreateEmptyTensor();
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

char tensileDataTypeToChar(TensileDataType t) {
  switch (t) {
  case tensileDataTypeSingle:                  return 'S';
  case tensileDataTypeDouble:                  return 'D';
  case tensileDataTypeComplexSingle:           return 'C';
  case tensileDataTypeComplexDouble:           return 'Z';
  case tensileDataTypeComplexConjugateSingle:  return 'X';
  case tensileDataTypeComplexConjugateDouble:  return 'Y';
#ifdef Tensile_ENABLE_FP16
  case tensileDataTypeHalf:                    return 'H';
  case tensileDataTypeComplexHalf:             return 'Q';
  case tensileDataTypeComplexConjugateHalf:    return 'W';
#endif
  default:                                    return '0';
  }
}

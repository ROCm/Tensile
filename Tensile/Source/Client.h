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

#include "TensileTypes.h"
#include "Tools.h"
#include "ReferenceCPU.h"
#include "MathTemplates.h"
#include "ClientParameters.h"
#include "DeviceStats.h"
#include "TensorUtils.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <ctime>
#include <sys/time.h>
#include <unistd.h>
#include <set>
#include <assert.h>

TensileTimer timer;
TensileTimer apiTimer;
std::ofstream file;

// benchmark parameters
unsigned int deviceIdx;
unsigned int initAlpha;
unsigned int initBeta;
unsigned int initA;
unsigned int initB;
unsigned int initC;
unsigned int initD;
unsigned int initAB;
unsigned int specializeAB; // True if the init mode requires different values for each matrix dim
unsigned int cEqualD;
unsigned int platformIdx;
unsigned int printValids;
unsigned int printMax;
unsigned int numBenchmarks;
unsigned int numElementsToValidate;
unsigned int numEnqueuesPerSync;
unsigned int numSyncsPerBenchmark;
unsigned int useGPUTimer;
unsigned int sleepPercent;
#if Tensile_CLIENT_BENCHMARK
unsigned int solutionStartIdx;
unsigned int numSolutions;
unsigned int runBenchmarkSolutions = 0;
#endif

// benchmark parameters commandline strings
const std::string keyDeviceIdx = "--device-idx";
const std::string keyHelp1 = "-h";
const std::string keyHelp2 = "--help";
const std::string keyInitD = "--init-d";
const std::string keyInitC = "--init-c";
const std::string keyInitA = "--init-a";
const std::string keyInitB = "--init-b";
const std::string keyInitAlpha = "--init-alpha";
const std::string keyInitBeta = "--init-beta";
const std::string keyCEqualD = "--c-equal-d";
const std::string keyPlatformIdx = "--platform-idx";
const std::string keyPrintValids = "--print-valids";
const std::string keyPrintMax = "--print-max";
const std::string keyNumBenchmarks = "--num-benchmarks";
const std::string keyNumElementsToValidate = "--num-elements-to-validate";
const std::string keyNumEnqueuesPerSync = "--num-enqueues-per-sync";
const std::string keyNumSyncsPerBenchmark = "--num-syncs-per-benchmark";
const std::string keyUseGPUTimer = "--use-gpu-timer";
const std::string keySleepPercent = "--sleep-percent";
const std::string keyLda = "--lda";
const std::string keyLdb = "--ldb";
const std::string keyLdc = "--ldc";
const std::string keyLdd = "--ldd";
const std::string keyStrideA = "--stride_a";
const std::string keyStrideB = "--stride_b";
const std::string keyStrideC = "--stride_c";
const std::string keyStrideD = "--stride_d";
#if Tensile_CLIENT_BENCHMARK
const std::string keySolutionStartIdx = "--solution-start-idx";
const std::string keyNumSolutions = "--num-solutions";
const std::string keyBenchmarkSolutions = "--benchmark-solutions";
#endif

// benchmark parameters default values
const unsigned int defaultDeviceIdx = 0;
const unsigned int defaultInitAlpha = 2;
const unsigned int defaultInitBeta = 2;
const unsigned int defaultInitD = 0;
const unsigned int defaultInitC = 3;
const unsigned int defaultInitA = 3;
const unsigned int defaultInitB = 3;
const unsigned int defaultCEqualD = 0;
const unsigned int defaultPlatformIdx = 0;
const unsigned int defaultPrintValids = 0;
const unsigned int defaultPrintMax = 0;
const unsigned int defaultNumBenchmarks = 1;
const unsigned int defaultNumElementsToValidate = 0;
const unsigned int defaultNumEnqueuesPerSync = 1;
const unsigned int defaultNumSyncsPerBenchmark = 1;
const unsigned int defaultUseGPUTimer = 1;
const unsigned int defaultSleepPercent = 0;
unsigned int lda = std::numeric_limits<unsigned int>::max();
unsigned int ldb = std::numeric_limits<unsigned int>::max();
unsigned int ldc = std::numeric_limits<unsigned int>::max();
unsigned int ldd = std::numeric_limits<unsigned int>::max();
unsigned int strideA = std::numeric_limits<unsigned int>::max();
unsigned int strideB = std::numeric_limits<unsigned int>::max();
unsigned int strideC = std::numeric_limits<unsigned int>::max();
unsigned int strideD = std::numeric_limits<unsigned int>::max();
#if Tensile_CLIENT_BENCHMARK
const unsigned int defaultSolutionStartIdx = 0;
const unsigned int defaultNumSolutions = maxNumSolutions;
const unsigned int defaultBenchmarkSolutions = 0;
#endif

// benchmark parameters for library client
#if Tensile_CLIENT_LIBRARY
const std::string keyFunctionIdx = "--function-idx";
const std::string keySizes = "--sizes";
const unsigned int defaultFunctionIdx = 0;
const unsigned int defaultSize = 128;
#endif

#if Tensile_CLIENT_BENCHMARK
const size_t solutionKeySize = 4;
std::set<unsigned int> invalidSolutions;
std::map<std::vector<unsigned int>, std::set<std::pair<unsigned int,double>>> solutionBenchmarks;
std::map<std::vector<unsigned int>, std::pair<unsigned int,double>> solutionMaxPeformance;
#endif

int expectedClockRate; // MHz
void initControls();
void destroyControls();

double globalFastestGFlops = 0.0;
double globalFastestTime  = 0.0;
unsigned int globalFastestIdx = 0;
double fastestGFlops = 0.0;
unsigned int fastestIdx = 0;


SolutionLock *solutionLocks;

#if Tensile_RUNTIME_LANGUAGE_OCL
/*******************************************************************************
* Given a finished/synced OpenCL event
* return performance in nano-seconds of time executing the kernel associated with the event
******************************************************************************/
cl_ulong getEventDeltaTime(cl_event event)
{
  cl_ulong start, end = 0;
  cl_int cl_error = CL_SUCCESS;

  if (cl_error = ::clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL) != CL_SUCCESS)
  {
    std::cout << "::clGetEventProfilingInfo error code: " << cl_error << std::endl;
    start = 0;
  }

  if (cl_error = ::clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL) != CL_SUCCESS)
  {
    std::cout << "::clGetEventProfilingInfo error code: " << cl_error << std::endl;
    end = 0;
  }
  return (end - start);
}
#else
/*******************************************************************************
* Given a finished/synced OpenCL event
* return performance in nano-seconds of time executing the kernel associated with the event
******************************************************************************/
float getEventDeltaTime(hipEvent_t start, hipEvent_t stop)
{
    float result = 0.0f;
    hipEventElapsedTime( &result, start, stop );
    return result;
}

#endif


template<typename DataType>
void copyData(
    DataType *initialA,
    DataType *initialB) {
#if Tensile_RUNTIME_LANGUAGE_OCL
  status = clEnqueueWriteBuffer(stream, static_cast<cl_mem>(deviceA), CL_TRUE,
      0, maxSizeA*bytesPerElement[dataTypeIdx], initialA, 0, NULL, NULL);
  tensileStatusCheck(status);
    std::cout << ".";
  status = clEnqueueWriteBuffer(stream, static_cast<cl_mem>(deviceB), CL_TRUE,
      0, maxSizeB*bytesPerElement[dataTypeIdx], initialB, 0, NULL, NULL);
  tensileStatusCheck(status);
    std::cout << ".";
#else
  status = hipMemcpy(deviceA, initialA, maxSizeA*bytesPerElement[dataTypeIdx],
      hipMemcpyHostToDevice);
  status = hipMemcpy(deviceB, initialB, maxSizeB*bytesPerElement[dataTypeIdx],
      hipMemcpyHostToDevice);
#endif
}

#if Tensile_CLIENT_BENCHMARK
std::vector<unsigned int> makeKeyForSolution(unsigned int solutionIdx, unsigned int summationSize) {
    const unsigned int *metaData = solutionMetaData[solutionIdx];
    unsigned int transformA = metaData[6];
    unsigned int transformB = metaData[7];
    unsigned int mt0 = metaData[0];
    unsigned int mt1 = metaData[1];
    unsigned int gsu = metaData[8];
    unsigned int lsu = metaData[9];

    size_t key_size = 7;    
    std::vector<unsigned int> key = std::vector<unsigned int>(key_size);
    key[0] = mt0;
    key[1] = mt1;
    key[2] = gsu;
    key[3] = lsu;
    key[4] = transformA;
    key[5] = transformB; 
    key[6] = summationSize;

    return key;
}
#endif

// Specialize the AB data for the problem size.
// This is used with the SerialInK data init type - each matrix needs different data.
// Since this takes extra time, it is recommended to use this data init mode only
// for debug
template<typename DataType>
void specializeData(
    DataType *initialData,
    unsigned int totalIndices,
    unsigned int numIndicesC,
    unsigned int numIndicesAB,
    const unsigned int *allSizes,
    const unsigned int *indexAssignments) {

  assert(totalIndices != 0);

  const unsigned int numIndicesSummation = totalIndices - numIndicesC;

  const unsigned int db = 0; // 0x1=header, 0x2=offset/value on each store, 0x4=loop debug
  TensorDims td("specialize_matrix", numIndicesAB, numIndicesC, allSizes, indexAssignments);

  if (db & 0x1) {
    td.print();
  }


  // Bucketize the sizes for the free and bound (summation) indices:
  //std::vector<unsigned int> freeIndexSizes(numIndicesC, 0);
  std::vector<unsigned int> freeIndexSizes;
  std::vector<unsigned int> boundIndexSizes;
  for (size_t i = 0; i < numIndicesAB; i++) {
    if (indexAssignments[i] < numIndicesC) {
      freeIndexSizes.push_back(allSizes[indexAssignments[i]]);
    } else {
      boundIndexSizes.push_back(allSizes[indexAssignments[i]]);
    }
  }
  const unsigned numIndicesFree = freeIndexSizes.size();
  assert(boundIndexSizes.size() == numIndicesSummation);


  // Counter for free-coord, these change in the free index loop
  std::vector<unsigned int> freeCoord(numIndicesFree, 0);

  // Counters to track coordinate, these change in the bound index loop
  std::vector<unsigned int> boundCoord( numIndicesSummation, 0);
  std::vector<unsigned int> coords( numIndicesAB, 0 );

  DataType val = static_cast<DataType>(0); // running initializer value
  bool moreIndicesC = true;
  while (moreIndicesC) { // iterate over entire free index range
    // reset summation indices
    for (unsigned int b = 0; b < numIndicesSummation; b++) {
      boundCoord[b] = 0;
    }

    while (true) { // iterate over entire bound index range


      // convert free/bound coord into tensorA,B 
      unsigned int f=0;
      unsigned int b=0;
      for (unsigned int i = 0; i < numIndicesAB; i++) {
        if (indexAssignments[i] < numIndicesC) {
          coords[i] = freeCoord[f++];
        } else {
          coords[i] = boundCoord[b++];
        }
      }

      size_t serialIdx = 0;
      for (unsigned int i = 0; i < numIndicesAB; i++) {
        serialIdx += coords[i]*td.memoryStrides[i];
      }

      if (db & 0x2) {
        std::cout << "[" << serialIdx << "] = " << val << "\n";
      }
      initialData[serialIdx] = val++;  // actually initialize the element

      // increment bound coord
      boundCoord[numIndicesSummation-1]++;
      for ( size_t bi = numIndicesSummation - 1; bi > 0 ; bi--) {
        if ( boundCoord[bi] >= boundIndexSizes[bi]) {
          boundCoord[bi] = 0;
          boundCoord[bi-1]++;
        }
      }

      if (boundCoord[0] >= boundIndexSizes[0]) {
        if (db & 0x4) {
          std::cout << "boundsBreak, boundCoord=" << boundCoord[0] << "\n";
        }
        break; // bound index range exit criteria
      }
    } // bound range
    
    // increment free coord
    // skip = 1, validate everything
    freeCoord[0]++;
    // bump free counters to next level:
    for (size_t f = 0; f < numIndicesFree-1; f++) {
      if (db & 0x4) {
        std::cout << "wrapcheck" << f << ":" << freeCoord[f] << " >= ? " << freeIndexSizes[f] << "\n";
      }
      if (freeCoord[f] >= freeIndexSizes[f]) {
        if (db & 0x4) {
          std::cout << "wrapdo" << f << ":" << freeCoord[f] << " >= ? " << freeIndexSizes[f] << "\n";
        }

        freeCoord[f] = 0;
        freeCoord[f+1]++;
      }
    }

    // When last free coord hits the max, exit the loop:
    if (db & 0x4) {
      std::cout << "done?" << freeCoord[numIndicesFree-1] << " >= ? " << freeIndexSizes[numIndicesFree-1] << "\n";
    }
    if (freeCoord.back() >= freeIndexSizes.back()) {
      moreIndicesC = false;
      break; // free index range exit criteria
    }
  }
}

/*******************************************************************************
 * Call Library
 * return true if errors/invalids
 ******************************************************************************/
#if Tensile_CLIENT_LIBRARY
template<typename DataType, typename DestDataType, typename ComputeDataType>
bool callLibrary(
    DestDataType *initialD,
    DestDataType *initialC,
    DataType *initialA,
    DataType *initialB,
    ComputeDataType alpha,
    ComputeDataType beta,
    unsigned int lda,
    unsigned int ldb,
    unsigned int ldc,
    unsigned int ldd,
    unsigned int strideA,
    unsigned int strideB,
    unsigned int strideC,
    unsigned int strideD,
    DestDataType *referenceD,
    DestDataType *referenceC,
    DestDataType *deviceOnHostD,
    DestDataType *deviceOnHostC)
{
  size_t totalFlops = numFlopsPerMac[dataTypeIdx];
  for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
    totalFlops *= userSizes[i]; }

  // Compute stridesC for validation
  // strideC accounts for memory strides (ie ldc)
  // while elementStride is a pure element space
  std::vector<unsigned int> strides(totalIndices[problemTypeIdx]);
  std::vector<unsigned int> stridesC(numIndicesC[problemTypeIdx]);
  std::vector<unsigned int> elementStridesC(numIndicesC[problemTypeIdx]);

  for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
    strides[i] = std::max(minStrides[i], userSizes[i]);
  }
  elementStridesC[0] = 1;
  stridesC[0] = 1;
  for (unsigned int i = 1; i < numIndicesC[problemTypeIdx]; i++) {
    elementStridesC[i] = elementStridesC[i-1] * userSizes[i-1];
    stridesC[i] = stridesC[i-1] * strides[i-1];
  }

  if (specializeAB) {
    if (initA == 5) {
      specializeData(initialA, totalIndices[problemTypeIdx],
                     numIndicesC[problemTypeIdx],
                     numIndicesAB[problemTypeIdx],
                     userSizes, indexAssignmentsA[problemTypeIdx]);
    }
    if (initB == 5) {
      specializeData(initialB, totalIndices[problemTypeIdx],
                     numIndicesC[problemTypeIdx],
                     numIndicesAB[problemTypeIdx],
                     userSizes, indexAssignmentsB[problemTypeIdx]);
    }
    copyData<DataType> (initialA, initialB);
  }

  if (printTensorA) {
    printTensor("A", initialA, numIndicesAB[problemTypeIdx],
                  numIndicesC[problemTypeIdx],
                  userSizes,
                  indexAssignmentsA[problemTypeIdx]);
  }
  if (printTensorB) {
    printTensor("B", initialB, numIndicesAB[problemTypeIdx],
                  numIndicesC[problemTypeIdx],
                  userSizes, 
                  indexAssignmentsB[problemTypeIdx]);
  }

  if (printTensorC & 0x1) {
    std::vector<unsigned int> indexAssignmentsC;
    for (unsigned  int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
      indexAssignmentsC.push_back(i);
    }
    printTensor("C_in", initialC, numIndicesC[problemTypeIdx],
                  numIndicesC[problemTypeIdx], userSizes,
                  indexAssignmentsC.data());
  }

  if (printTensorD & 0x1) {
    std::vector<unsigned int> indexAssignmentsC;
    for (unsigned  int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
      indexAssignmentsC.push_back(i);
    }
    printTensor("D_in", initialD, numIndicesC[problemTypeIdx],
                  numIndicesC[problemTypeIdx], userSizes,
                  indexAssignmentsC.data());
  }

  size_t currentElementSizeC = 1;
  size_t currentMemorySizeC = 1;
  for (unsigned int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
    currentElementSizeC *= userSizes[i];
    currentMemorySizeC *= strides[i];
  }
  size_t sizeToCopy = currentMemorySizeC*bytesPerElement[dataTypeIdx];
#if Tensile_RUNTIME_LANGUAGE_OCL
  status = clEnqueueWriteBuffer(stream, static_cast<cl_mem>(deviceC), CL_TRUE,
      0, sizeToCopy, initialC, 0, NULL, NULL);
  tensileStatusCheck(status);
  if(!cEqualD)
    status = clEnqueueWriteBuffer(stream, static_cast<cl_mem>(deviceD), CL_TRUE,
        0, sizeToCopy, initialD, 0, NULL, NULL);
#else
  status = hipMemcpy(deviceC, initialC, sizeToCopy, hipMemcpyHostToDevice);
  tensileStatusCheck(status);
  if(!cEqualD)
    status = hipMemcpy(deviceD, initialD, sizeToCopy, hipMemcpyHostToDevice);
#endif
  tensileStatusCheck(status);

  size_t numInvalids = 0;
  size_t numChecked = 0;

  // do validation
  bool solutionIsValid = true;
  if (numElementsToValidate) {
    memcpy(referenceC, initialC, sizeToCopy);
    if(!cEqualD)
      memcpy(referenceD, initialD, sizeToCopy);
    // calculate validation stride
    if (numElementsToValidate >= currentElementSizeC) {
      validationStride = 1;
    } else {
      if (numElementsToValidate) {
        validationStride = currentElementSizeC / numElementsToValidate;
        // find next prime number
        while (true) { // break statement at end
          bool prime = true;
          for (unsigned int i = 2; i < validationStride; i++) {
            if ( validationStride % i == 0) {
              prime = false;
              break;
            }
          }
          if (prime) {
            break;
          } else {
            validationStride++;
          }
        }
      } else {
        validationStride = 0;
      }
    }

    // call reference function
    TensileStatus referenceStatus = generatedCallToReferenceCPU(
        userSizes, minStrides, referenceD, referenceC,
        initialA, initialB,
        lda, ldb, ldc, ldd,
        strideA, strideB, strideC, strideD,
        alpha, beta, useHighPrecisionAccumulate);

    // call device function
    TensileStatus tensileCallStatus = generatedCallTo_tensile<DataType, DestDataType, ComputeDataType>(userSizes, minStrides,
        alpha, beta, lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD);
    if (tensileCallStatus != tensileStatusSuccess) {
      solutionIsValid = false;
    }

    // copy data back to host
#if Tensile_RUNTIME_LANGUAGE_OCL
    clEnqueueReadBuffer(stream, static_cast<cl_mem>(deviceC), CL_TRUE, 0,
        sizeToCopy, deviceOnHostC, 0, NULL,
        NULL);
    clEnqueueReadBuffer(stream, static_cast<cl_mem>(deviceD), CL_TRUE, 0,
        sizeToCopy, deviceOnHostD, 0, NULL,
        NULL);
#else
    hipMemcpy(deviceOnHostC, deviceC, sizeToCopy, hipMemcpyDeviceToHost);
    hipMemcpy(deviceOnHostD, deviceD, sizeToCopy, hipMemcpyDeviceToHost);
#endif

    if (printTensorC & 0x2) {
      std::vector<unsigned int> indexAssignmentsC;
      for (unsigned  int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
        indexAssignmentsC.push_back(i);
      }
      printTensor("C_result", deviceOnHostC, numIndicesC[problemTypeIdx],
                  numIndicesC[problemTypeIdx], userSizes,
                  indexAssignmentsC.data());
    }

    if (printTensorD & 0x2) {
      std::vector<unsigned int> indexAssignmentsC;
      for (unsigned  int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
        indexAssignmentsC.push_back(i);
      }
      printTensor("D_result", deviceOnHostD, numIndicesC[problemTypeIdx],
                  numIndicesC[problemTypeIdx], userSizes,
                  indexAssignmentsC.data());
    }

    // compare
    bool firstPrint = true;
    unsigned int printIdx = 0;
    for (size_t e = 0; e < currentElementSizeC; e+= validationStride) {

      // Compute the actual serialIdxX accouting for strides:
      size_t serialIdxC = 0;
      size_t r = e;
      for (int j = numIndicesC[problemTypeIdx]-1; j >=0; j--) {
        serialIdxC += r / elementStridesC[j] * stridesC[j];
        r = r % elementStridesC[j];
      }

      bool equalC, equalD;
      equalD = tensileAlmostEqual<DataType>( // need AlmostEqual for StaggerU
          deviceOnHostD[serialIdxC], referenceD[serialIdxC]);
      equalC = tensileAlmostEqual<DataType>( // need AlmostEqual for StaggerU
          deviceOnHostC[serialIdxC], referenceC[serialIdxC]);
      numChecked++;

      if (!equalC || !equalD) numInvalids++;

      if (!equalC || !equalD || printValids) {
        if (printIdx < printMax) {
          if (firstPrint) {
            std::cout << "Index:  Device | Reference" << std::endl;
            firstPrint = false;
          }
          std::cout << "[" << (numChecked-1) << "] " 
            << " e=" << e
            << " serialIdxC=" << serialIdxC << ": "
            << tensileToString(deviceOnHostD[serialIdxC])
            << (equalD ? "==" : "!=") << tensileToString(referenceD[serialIdxC])
            << " , "
            << tensileToString(deviceOnHostC[serialIdxC])
            << (equalC ? "==" : "!=") << tensileToString(referenceC[serialIdxC])
            << std::endl;
          printIdx++;
        }
      }
    } // compare loop

  } // if validate
  if (numInvalids) {
    solutionIsValid = false;
  }

#if Tensile_RUNTIME_LANGUAGE_OCL
  cl_event l_outputEvent[numSyncsPerBenchmark][numEnqueuesPerSync];
#else
  hipEvent_t l_eventStart[numSyncsPerBenchmark][numEnqueuesPerSync];
  hipEvent_t l_eventStop[numSyncsPerBenchmark][numEnqueuesPerSync];
  for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
    for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
      hipEventCreateWithFlags(&l_eventStart[syncIdx][enqIdx], hipEventDefault);
      hipEventCreateWithFlags(&l_eventStop[syncIdx][enqIdx], hipEventDefault);
    }
  }
#endif

  // time solution
  timer.start();
  double apiTimeUs = 0;
  // device stats
  unsigned long long avgCoreClock = 0;
  unsigned long long avgMemClock = 0;
  double avgTemp = 0;
  unsigned long long avgFanSpeed = 0;
  for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
    unsigned long long syncCoreClock = 0;
    unsigned long long syncMemClock = 0;
    double syncTemp = 0;
    unsigned long long syncFanSpeed = 0;
    apiTimer.start();
    for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
#if Tensile_RUNTIME_LANGUAGE_OCL
      generatedCallTo_tensile(userSizes, minStrides, alpha, beta,
          lda, ldb, ldc, ldd,
          strideA, strideB, strideC, strideD, 0, NULL,
          &l_outputEvent[syncIdx][enqIdx]);
#else
      generatedCallTo_tensile<DataType, DestDataType, ComputeDataType>(userSizes, minStrides, alpha, beta,
          lda, ldb, ldc, ldd,
          strideA, strideB, strideC, strideD,
          numEnqueuesPerSync, &l_eventStart[syncIdx][enqIdx], &l_eventStop[syncIdx][enqIdx]);
#endif
    }
    double currentApiTimeUs = apiTimer.elapsed_us() / numEnqueuesPerSync;
    apiTimeUs += currentApiTimeUs;

    // sync
#if Tensile_RUNTIME_LANGUAGE_OCL
    status = clFinish(stream);
#else
    unsigned int numDeviceStatsQueries = 0;
    do { // device stats
      int currentCoreClock = tensileGetDeviceCoreClock(0);
      int currentMemClock = tensileGetDeviceMemClock(0);
      float currentTemp = tensileGetDeviceTemp(0);
      int currentFanSpeed = tensileGetDeviceFanSpeed(0);
      //std::cout << "clock: " << currentCoreClock << " Mhz" << std::endl;
      syncCoreClock += currentCoreClock;
      syncMemClock += currentMemClock;
      syncTemp += currentTemp;
      syncFanSpeed += currentFanSpeed;
      numDeviceStatsQueries++;
    } while (hipEventQuery(l_eventStop[syncIdx][numEnqueuesPerSync-1]) != hipSuccess);
    syncCoreClock /= numDeviceStatsQueries;
    syncMemClock /= numDeviceStatsQueries;
    syncTemp /= numDeviceStatsQueries;
    syncFanSpeed /= numDeviceStatsQueries;
    avgCoreClock += syncCoreClock;
    avgMemClock += syncMemClock;
    avgTemp += syncTemp;
    avgFanSpeed += syncFanSpeed;
#endif
    tensileStatusCheck(status);
  } // sync loop

  apiTimeUs /= numSyncsPerBenchmark;

  double timeNs = 0.0;
#if Tensile_RUNTIME_LANGUAGE_OCL
  if (useGPUTimer) {
    // Loop through the event array and collect kernel performance data
    // Release events when done with them
    cl_ulong kernel_time_sum = 0;
    for (auto& event_array : l_outputEvent) {
      for (auto event : event_array) {
        // getEventDeltaTime returns unsigned long in nano-seconds on opencl
        kernel_time_sum += getEventDeltaTime(event);
        ::clReleaseEvent(event);
      }
    }
    timeNs = static_cast<double>(kernel_time_sum);
  } else {
    timeNs = timer.elapsed_ns();
  }
#else
  if (useGPUTimer) {
    // Loop through the event array and collect kernel performance data
    // Release events when done with them
    float kernel_time_sum = 0;
    for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
      for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
        // getEventDeltaTime returns unsigned long in milli-seconds on hip
        kernel_time_sum += getEventDeltaTime(l_eventStart[syncIdx][enqIdx],
            l_eventStop[syncIdx][enqIdx]);
        ::hipEventDestroy(l_eventStart[syncIdx][enqIdx]);
        ::hipEventDestroy(l_eventStop[syncIdx][enqIdx]);
      }
    }
    // convert to nano-seconds
    timeNs = static_cast<double>(kernel_time_sum) * TensileTimer::million;
  } else {
    timeNs = timer.elapsed_ns();
  }
#endif
  if (sleepPercent) {
    unsigned int sleepMicroSeconds = (timeNs*10*sleepPercent)/1e6;
    usleep(sleepMicroSeconds);
  }

  timeNs /= (numSyncsPerBenchmark * numEnqueuesPerSync);
  // device stats
  avgCoreClock /= numSyncsPerBenchmark;
  avgMemClock /= numSyncsPerBenchmark;
  avgTemp /= numSyncsPerBenchmark;
  avgFanSpeed /= numSyncsPerBenchmark;

  float perfScaling = 1.f;
  double gflops = solutionIsValid ? perfScaling * totalFlops / timeNs : 0;

  bool newFastest = false;
  if (gflops > fastestGFlops) {
    fastestGFlops = gflops;
    fastestIdx = functionIdx;
    newFastest = true;
    if (fastestGFlops > globalFastestGFlops) {
      globalFastestGFlops = fastestGFlops;
      globalFastestTime = timeNs;
      globalFastestIdx = fastestIdx;
    }
  }

  const char * solutionName = generatedCallTo_tensileGetSolutionName<DataType, DestDataType, ComputeDataType>(
      userSizes, minStrides, alpha, beta,
      lda, ldb, ldc, ldd,
      strideA, strideB, strideC, strideD);

  std::cout << std::setw(10) << std::fixed << std::setprecision(3)
      << gflops*perfScaling << ", "
      << std::setw(10) << std::fixed << std::setprecision(3)
      << gflops << ", "
      << solutionName << (newFastest ? "*" : " ") << ", "
      << std::setw(10) << std::fixed << std::setprecision(4)
      << timeNs * TensileTimer::reciprical_million << ", "
      << std::setw(7) << std::fixed << std::setprecision(3)
      << apiTimeUs << ", ";
  if (numElementsToValidate) {
    std::cout << (numInvalids ? "FAILED" : "PASSED")
      << ": " << (numChecked-numInvalids) << "/" << numChecked << ", ";
  }
  // device stats
  std::cout << avgCoreClock << ", ";
  std::cout << avgMemClock << ", ";
  std::cout << avgTemp << ", ";
  std::cout << avgFanSpeed << ", ";
  
  std::cout << functionIdx << "/" << numFunctions;
  std::cout << std::endl;


#if 0
  if (numElementsToValidate) {
    std::cout << "Function[" << std::setw(2) << functionIdx << "/"
      << numFunctions << "]:"
      << std::setw(10) << std::fixed << std::setprecision(3)
      << gflops << " GFlop/s";
    if (newFastest) {
      std::cout << "*";
    } else {
      std::cout << " ";
    }
    std::cout << " |"
      << std::setw(9) << std::fixed << std::setprecision(3)
      << timeNs * TensileTimer::reciprical_million
      << " ms | v: " << (numInvalids ? "FAILED" : "PASSED")
      << " " << (numChecked-numInvalids) << "/" << numChecked;
    std::cout << " | api:" << std::setw(7) << std::fixed
      << std::setprecision(3) << apiTimeUs << " us";
    std::cout << " | " << solutionName;
    std::cout << std::endl;
  } else {
    std::cout << "Function[" << functionIdx << "/" << numFunctions << "]:"
      << std::setw(10) << std::fixed << std::setprecision(3)
      << gflops << " GFlop/s";
    if (newFastest) {
      std::cout << "*";
    } else {
      std::cout << " ";
    }
    std::cout << " |"
      << std::setw(9) << std::fixed << std::setprecision(3)
      << timeNs * TensileTimer::reciprical_million << " ms";
    if (newFastest) {
      std::cout << "*";
    }
    std::cout << " | api:" << std::setw(7) << std::fixed
      << std::setprecision(3) << apiTimeUs << " us";
    std::cout << " | " << solutionName;
    std::cout << std::endl;
  }
#endif
  return (numInvalids > 0);
} // callLibrary
#endif

#if Tensile_CLIENT_BENCHMARK
template<typename DataType, typename DestDataType, typename ComputeDataType>
bool benchmarkAllSolutions(
    DestDataType *initialD,
    DestDataType *initialC,
    DataType *initialA,
    DataType *initialB,
    ComputeDataType alpha,
    ComputeDataType beta,
    DestDataType *referenceD,
    DestDataType *referenceC,
    DestDataType *deviceOnHostD,
    DestDataType *deviceOnHostC,
    double *problem_gpu_time_ms) {

  bool returnInvalids = false;

#if Tensile_RUNTIME_LANGUAGE_OCL
    cl_event l_outputEvent[numSyncsPerBenchmark][numEnqueuesPerSync];
#else
    hipEvent_t l_eventStart[numSyncsPerBenchmark][numEnqueuesPerSync];
    hipEvent_t l_eventStop[numSyncsPerBenchmark][numEnqueuesPerSync];
    for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
      for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
        hipEventCreateWithFlags( &l_eventStart[syncIdx][enqIdx],
            hipEventDefault );
        hipEventCreateWithFlags( &l_eventStop[syncIdx][enqIdx],
            hipEventDefault );
      }
    }
#endif


  fastestGFlops = 0;
  *problem_gpu_time_ms = 0;

  std::ofstream gFile;

  gFile.open(granularityFileName);
  gFile << "Solution, M, N, MT0, MT1, LSU, GSU";
  for (unsigned int summationIdx = 0; summationIdx < numSummations; summationIdx ++) {
    gFile << ", K=" << summations[summationIdx];
  }
  gFile << std::endl;

  unsigned int sizes[8];

  for (unsigned int solutionIdx = 0; solutionIdx < numSolutions; solutionIdx ++) {

    const unsigned int *metaData = solutionMetaData[solutionIdx];
    unsigned int transformA = metaData[6];
    unsigned int transformB = metaData[7];
    unsigned int mt0 = metaData[0];
    unsigned int mt1 = metaData[1];
    unsigned int m = 36*mt0;
    unsigned int n = 36*mt1;
    unsigned int gsu = metaData[8];
    unsigned int lsu = metaData[9];

    gFile << solutions[solutionIdx]._name;
    gFile << ", " << m;
    gFile << ", " << n;
    gFile << ", " << mt0;
    gFile << ", " << mt1;
    gFile << ", " << lsu;
    gFile << ", " << gsu;

    std::cout << "Performace for solution: " << solutions[solutionIdx]._name << std::endl;
    for (unsigned int summationIdx = 0; summationIdx < numSummations; summationIdx ++) {

      unsigned int sizes[8];
      unsigned int k = summations[summationIdx];

      sizes[0] = m;
      sizes[1] = n;
      sizes[2] = 1;
      sizes[3] = k;
      sizes[indexAssignmentsLD[0]] = m;
      sizes[indexAssignmentsLD[1]] = m; 

      
      if (transformA == 0) {
        sizes[indexAssignmentsLD[2]] = m;
      } else {
        sizes[indexAssignmentsLD[2]] = k;
      }

      if (transformB == 0) {
        sizes[indexAssignmentsLD[3]] = k;
      } else {
        sizes[indexAssignmentsLD[3]] = n;
      }

      if (numIndicesLD == 4)
      {
        ldd = sizes[indexAssignmentsLD[0]];
        ldc = sizes[indexAssignmentsLD[1]];
        lda = sizes[indexAssignmentsLD[2]];
        ldb = sizes[indexAssignmentsLD[3]];
      }

      // Compute stridesC for validation
      // strideC accounts for memory strides (ie ldc)
      // while elementStride is a pure element space
      std::vector<unsigned int> strides(totalIndices[problemTypeIdx]);
      std::vector<unsigned int> stridesD(numIndicesC[problemTypeIdx]);
      std::vector<unsigned int> stridesC(numIndicesC[problemTypeIdx]);
      std::vector<unsigned int> elementStrides(numIndicesC[problemTypeIdx]);

      for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
        strides[i] = std::max(minStrides[i], sizes[i]);
      }
      elementStrides[0] = 1;
      stridesD[0] = 1;
      stridesC[0] = 1;

      elementStrides[1] = sizes[0];
      stridesD[1] = (ldd != std::numeric_limits<unsigned int>::max()) ? ldd : strides[0];
      stridesC[1] = (ldc != std::numeric_limits<unsigned int>::max()) ? ldc : strides[0];

      for (unsigned int i = 2; i < numIndicesC[problemTypeIdx]; i++) {
        elementStrides[i] = elementStrides[i-1] * sizes[i-1];
        stridesD[i] = stridesD[i-1] * strides[i-1];
        stridesC[i] = stridesC[i-1] * strides[i-1];
      }

      bool returnInvalids = false;
      size_t currentElementSizeC = elementStrides[numIndicesC[problemTypeIdx]-1];
      size_t currentMemorySizeD = stridesD[numIndicesC[problemTypeIdx]-1];
      size_t currentMemorySizeC = stridesC[numIndicesC[problemTypeIdx]-1];

      size_t sizeToCopyD = currentMemorySizeD*bytesPerElement[dataTypeIdx];
      size_t sizeToCopyC = currentMemorySizeC*bytesPerElement[dataTypeIdx];

      size_t totalFlops = numFlopsPerMac[0];
      for (unsigned int i = 0; i < totalIndices[0]; i++) {
        totalFlops *= sizes[i]; }

      bool solutionIsValid = true;

      // validate solution
      size_t numInvalids = 0;
      size_t numChecked = 0;
      TensileStatus callStatus = tensileStatusSuccess;

      // time solution
      timer.start();
        // device stats
      unsigned long long avgCoreClock = 0;
      unsigned long long avgMemClock = 0;
      double avgTemp = 0;
      unsigned long long avgFanSpeed = 0;
      for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
        unsigned long long syncCoreClock = 0;
        unsigned long long syncMemClock = 0;
        double syncTemp = 0;
        unsigned long long syncFanSpeed = 0;
        for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
#if Tensile_RUNTIME_LANGUAGE_OCL
          TensileStatus status = generatedCallToSolution( solutions[solutionIdx], &solutionLocks[solutionIdx],
              sizes, minStrides, lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD, alpha, beta,
              0, NULL, &l_outputEvent[syncIdx][enqIdx] );
#else
          TensileStatus status = generatedCallToSolution( solutions[solutionIdx], &solutionLocks[solutionIdx],
              sizes, minStrides, lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD, alpha, beta,
              numEnqueuesPerSync, &l_eventStart[syncIdx][enqIdx],
              &l_eventStop[syncIdx][enqIdx] );

          if(status == hipErrorFileNotFound)
          {
              printf("Kernel file not found; exiting.\n");
              exit(1);
          }
#endif
          if (status != tensileStatusSuccess) {
            solutionIsValid = false;
          }

        }
        // sync
#if Tensile_RUNTIME_LANGUAGE_OCL
        status = clFinish(stream);
#else
        unsigned int numDeviceStatsQueries = 0;
        do { // device stats
          int currentCoreClock = tensileGetDeviceCoreClock(0);
          int currentMemClock = tensileGetDeviceMemClock(0);
          float currentTemp = tensileGetDeviceTemp(0);
          int currentFanSpeed = tensileGetDeviceFanSpeed(0);
          //std::cout << "clock: " << currentCoreClock << " Mhz" << std::endl;
          syncCoreClock += currentCoreClock;
          syncMemClock += currentMemClock;
          syncTemp += currentTemp;
          syncFanSpeed += currentFanSpeed;
          numDeviceStatsQueries++;
        } while (hipEventQuery(l_eventStop[syncIdx][numEnqueuesPerSync-1]) != hipSuccess);
        syncCoreClock /= numDeviceStatsQueries;
        syncMemClock /= numDeviceStatsQueries;
        syncTemp /= numDeviceStatsQueries;
        syncFanSpeed /= numDeviceStatsQueries;

        avgCoreClock += syncCoreClock;
        avgMemClock += syncMemClock;
        avgTemp += syncTemp;
        avgFanSpeed += syncFanSpeed;
#endif
        tensileStatusCheck(status);
      } // sync loop

      double timeNs = 0.0;
#if Tensile_RUNTIME_LANGUAGE_OCL
      if (useGPUTimer) {
        // Loop through the multi-dimensional event array and collect kernel performance data
        // Release events when done with them
        cl_ulong kernel_time_sum = 0;
        for (auto& event_array : l_outputEvent) {
          for (auto event : event_array) {
            // getEventDeltaTime returns unsigned long in nano-seconds on opencl
            kernel_time_sum += getEventDeltaTime(event);
          }
        }
        timeNs = static_cast<double>(kernel_time_sum);
      } else {
        timeNs = timer.elapsed_ns();
      }
#else
      if (useGPUTimer) {
        // Loop through the event array and collect kernel performance data
        // Release events when done with them
        float kernel_time_sum = 0;
        for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++){
          for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
            // getEventDeltaTime returns unsigned long in milli-seconds on hip
            float kernel_time = getEventDeltaTime(l_eventStart[syncIdx][enqIdx],
                l_eventStop[syncIdx][enqIdx] );
            //std::cout << "kernelTime: " << kernel_time << std::endl;
            kernel_time_sum += kernel_time;
          }
        }
        timeNs = static_cast<double>(kernel_time_sum)
          * TensileTimer::million;  // convert to nano-seconds
      } else {
        timeNs = timer.elapsed_ns();
      }

#endif
      if (sleepPercent) {
        unsigned int sleepMicroSeconds = (timeNs*10*sleepPercent)/1e6;
        usleep(sleepMicroSeconds);
      }

      *problem_gpu_time_ms += timeNs/1e6;
      //printf ("problem: %6.2f ms+ %6.2fns\n", *problem_gpu_time_ms, timeNs);

      timeNs /= (numSyncsPerBenchmark * numEnqueuesPerSync);
      // device status
      avgCoreClock /= numSyncsPerBenchmark;
      avgMemClock /= numSyncsPerBenchmark;
      avgTemp /= numSyncsPerBenchmark;
      avgFanSpeed /= numSyncsPerBenchmark;

      float perfScaling = 1.f;
      double gflops = solutionIsValid ? perfScaling * totalFlops / timeNs : 0;
      //std::cout << gflops << " gflops = " << totalFlops << " flops / " << timeNs << " ns" << std::endl;
      bool newFastest = false;
      if (gflops > fastestGFlops) {
        fastestGFlops = gflops;
        fastestIdx = solutionIdx;
        newFastest = true;
//        if (fastestGFlops > globalFastestGFlops) {
//          globalFastestGFlops = fastestGFlops;
//          globalFastestTime = timeNs;
//          globalFastestIdx = fastestIdx;
//        }
      }

      std::cout << "Solution Size: " << sizes[0];
      for (unsigned int i = 1; i < totalIndices[problemTypeIdx]+numIndicesLD; i++) {
        std::cout << ", " << sizes[i];
      }
      std::cout << std::endl;

      // print results to stdout
      //if (newFastest || numInvalids>0 || !printWinnersOnly || callStatus != tensileStatusSuccess)
      {
        std::cout << std::setw(10) << std::fixed << std::setprecision(3)
            << gflops*perfScaling << ", "
            << std::setw(10) << std::fixed << std::setprecision(3)
            << gflops << ", "
            //<< solutions[solutionIdx]._name << (newFastest ? "*" : " ") << ", "
            << std::setw(9) << std::fixed << std::setprecision(3)
            << timeNs * TensileTimer::reciprical_million << ", ";

        if (callStatus == tensileStatusSuccess) {
          if (numElementsToValidate) {
            std::cout << (numInvalids ? "FAILED" : "PASSED")
              << ": " << (numChecked-numInvalids) << "/" << numChecked << ", ";
          } else {
            std::cout << "NO_CHECK, "; // did not validate any results, may work or maybe not
          }
        } else if (callStatus == tensileStatusAssertFailure) {
          std::cout << "DID_NOT_SATISFY_ASSERTS, ";
        } else {
          std::cout << "INVALID_KERNEL, "; // launch function returned error?
        }

        // device stats
        std::cout << avgCoreClock << ", ";
        std::cout << avgMemClock << ", ";
        std::cout << avgTemp << ", ";
        std::cout << avgFanSpeed << ", ";

        std::cout << solutionIdx << "/" << numSolutions << ", ";

        struct timeval tmnow;
        struct tm *tm;
        gettimeofday(&tmnow, NULL); // microsecond resolution
        tm = localtime(&tmnow.tv_sec);
        char prev_fill = std::cout.fill('0');
        std::cout << (tm->tm_year + 1900) << "-"
          << std::setw(2) << (tm->tm_mon + 1) << "-"
          << std::setw(2) << tm->tm_mday << " "
          << std::setw(2) << tm->tm_hour << ":"
          << std::setw(2) << tm->tm_min << ":"
          << std::setw(2) << tm->tm_sec << "."
          << std::setw(6) << static_cast<int>(tmnow.tv_usec) << ", ";
        (void) std::cout.fill(prev_fill);

        std::cout << std::endl;
      }

      // write results to file
      if ((numInvalids > 0) || (callStatus != tensileStatusSuccess)) {
        gflops = -1.0;
        invalidSolutions.insert(solutionIdx);
      }

      gFile << ", " << gflops;
      float granularityPerf = static_cast<float>(gflops);
   
      std::vector<unsigned int> key = makeKeyForSolution(solutionIdx, k);
      std::map<std::vector<unsigned int>, std::set<std::pair<unsigned int,double>>>::iterator found;      
      found = solutionBenchmarks.find(key);

      std::pair<unsigned int,double> value = std::make_pair(solutionIdx, granularityPerf);
      
      if (found == solutionBenchmarks.end()) {
          std::set<std::pair<unsigned int,double>> valueSet; // = std::set<std::pair<unsigned int,double>>>();
          valueSet.insert(value);
          solutionBenchmarks[key] = valueSet;           
      } else {
          (*found).second.insert(value);
      }
    }
    gFile << std::endl;
  } // solution loop

  if (useGPUTimer) {
#if Tensile_RUNTIME_LANGUAGE_OCL
    for (auto& event_array : l_outputEvent) {
      for (auto event : event_array) {
        ::clReleaseEvent(event);
      }
    }
#else
    for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++){
      for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
        ::hipEventDestroy( l_eventStart[syncIdx][enqIdx] );
        ::hipEventDestroy( l_eventStop[syncIdx][enqIdx] );
      }
    }
#endif

    gFile << std::endl;
  }
  file << std::endl;
  gFile.close();

  return returnInvalids;
} // benchmark solutions
#endif // benchmark client

/*******************************************************************************
 * benchmark all solutions for problem size
 * return true if error/invalids
 * writes to these globalVariables:
 * - globalFastestGFlops
 * - globalFastestTime
 * - globalFastestIdx
 *
 * - invalidSolutions
 *
 * - reads: problemSizes?
 *
 * - writes one index in solutionPerf[problemIdx]
 ******************************************************************************/
#if Tensile_CLIENT_BENCHMARK
template<typename DataType, typename DestDataType, typename ComputeDataType>
bool benchmarkAllSolutionsForSize(
    unsigned int problemIdx,
    //unsigned int *sizes,
    DestDataType *initialD,
    DestDataType *initialC,
    DataType *initialA,
    DataType *initialB,
    ComputeDataType alpha,
    ComputeDataType beta,
    DestDataType *referenceD,
    DestDataType *referenceC,
    DestDataType *deviceOnHostD,
    DestDataType *deviceOnHostC,
    double *problem_gpu_time_ms) {
  const unsigned int *sizes = problemSizes[problemIdx];

  if (numIndicesLD == 4)
  {
    ldd = sizes[indexAssignmentsLD[0]];
    ldc = sizes[indexAssignmentsLD[1]];
    lda = sizes[indexAssignmentsLD[2]];
    ldb = sizes[indexAssignmentsLD[3]];
  }

  // Compute stridesC for validation
  // strideC accounts for memory strides (ie ldc)
  // while elementStride is a pure element space
  std::vector<unsigned int> strides(totalIndices[problemTypeIdx]);
  std::vector<unsigned int> stridesD(numIndicesC[problemTypeIdx]);
  std::vector<unsigned int> stridesC(numIndicesC[problemTypeIdx]);
  std::vector<unsigned int> elementStrides(numIndicesC[problemTypeIdx]);

  for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
    strides[i] = std::max(minStrides[i], sizes[i]);
  }
  elementStrides[0] = 1;
  stridesD[0] = 1;
  stridesC[0] = 1;

  elementStrides[1] = sizes[0];
  stridesD[1] = (ldd != std::numeric_limits<unsigned int>::max()) ? ldd : strides[0];
  stridesC[1] = (ldc != std::numeric_limits<unsigned int>::max()) ? ldc : strides[0];

  for (unsigned int i = 2; i < numIndicesC[problemTypeIdx]; i++) {
    elementStrides[i] = elementStrides[i-1] * sizes[i-1];
    stridesD[i] = stridesD[i-1] * strides[i-1];
    stridesC[i] = stridesC[i-1] * strides[i-1];
  }

  bool returnInvalids = false;
  size_t currentElementSizeC = elementStrides[numIndicesC[problemTypeIdx]-1];
  size_t currentMemorySizeD = stridesD[numIndicesC[problemTypeIdx]-1];
  size_t currentMemorySizeC = stridesC[numIndicesC[problemTypeIdx]-1];

  size_t sizeToCopyD = currentMemorySizeD*bytesPerElement[dataTypeIdx];
  size_t sizeToCopyC = currentMemorySizeC*bytesPerElement[dataTypeIdx];

  file << problemIdx << ", " << sizes[0];
  for (unsigned int i = 1; i < totalIndices[problemTypeIdx]+numIndicesLD; i++) {
    file << ", " << sizes[i];
  }
  size_t totalFlops = numFlopsPerMac[dataTypeIdx];
  for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
    totalFlops *= sizes[i]; }
  file << ", " << totalFlops;

  if (specializeAB) {
    if (initA==5) {
      specializeData(initialA, totalIndices[problemTypeIdx],
                      numIndicesC[problemTypeIdx],
                      numIndicesAB[problemTypeIdx],
                      sizes, indexAssignmentsA[problemTypeIdx]);
    }
    if (initB==5) {
      specializeData(initialB, totalIndices[problemTypeIdx],
                      numIndicesC[problemTypeIdx],
                      numIndicesAB[problemTypeIdx],
                      sizes, indexAssignmentsB[problemTypeIdx]);
    }
    copyData<DataType> (initialA, initialB);
  }

  if (printTensorA) {
    printTensor("A", initialA, numIndicesAB[problemTypeIdx],
                numIndicesC[problemTypeIdx], sizes,
                indexAssignmentsA[problemTypeIdx]);
  }
  if (printTensorB) {
    printTensor("B", initialB, numIndicesAB[problemTypeIdx],
                numIndicesC[problemTypeIdx], sizes, 
                indexAssignmentsB[problemTypeIdx]);
  }
  if (printTensorC & 0x1) {
    std::vector<unsigned int> indexAssignmentsC;
    for (unsigned  int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
      indexAssignmentsC.push_back(i);
    }
    printTensor("C_in", initialC, numIndicesC[problemTypeIdx],
                numIndicesC[problemTypeIdx], sizes,
                indexAssignmentsC.data());
  }
  if (printTensorD & 0x1) {
    std::vector<unsigned int> indexAssignmentsC;
    for (unsigned  int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
      indexAssignmentsC.push_back(i);
    }
    printTensor("D_in", initialD, numIndicesC[problemTypeIdx],
                numIndicesC[problemTypeIdx], sizes,
                indexAssignmentsC.data());
  }

  // pre-compute referenceCPU if validating
  if (numElementsToValidate) {
    memcpy(referenceC, initialC, sizeToCopyC);
    if(!cEqualD)
      memcpy(referenceD, initialD, sizeToCopyD);
    if (numElementsToValidate >= currentElementSizeC) {
      validationStride = 1;
    } else {
      if (numElementsToValidate) {
        validationStride = currentElementSizeC / numElementsToValidate;
        // find next prime number
        while (true) { // break statement at end
          bool prime = true;
          for (unsigned int i = 2; i < validationStride; i++) {
            if ( validationStride % i == 0) {
              prime = false;
              break;
            }
          }
          if (prime) {
            break;
          } else {
            validationStride++;
          }
        }
      } else {
        validationStride = 0;
      }
    }
    generatedCallToReferenceCPU( sizes, minStrides, referenceD, referenceC, initialA, initialB,
        lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD, alpha, beta, useHighPrecisionAccumulate);

  }
#if Tensile_RUNTIME_LANGUAGE_OCL
    cl_event l_outputEvent[numSyncsPerBenchmark][numEnqueuesPerSync];
#else
    hipEvent_t l_eventStart[numSyncsPerBenchmark][numEnqueuesPerSync];
    hipEvent_t l_eventStop[numSyncsPerBenchmark][numEnqueuesPerSync];
    for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
      for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
        hipEventCreateWithFlags( &l_eventStart[syncIdx][enqIdx],
            hipEventDefault );
        hipEventCreateWithFlags( &l_eventStop[syncIdx][enqIdx],
            hipEventDefault );
      }
    }
#endif


  fastestGFlops = 0;
  *problem_gpu_time_ms = 0;
  for (unsigned int solutionIdx = solutionStartIdx; solutionIdx < solutionStartIdx + numSolutions; solutionIdx ++) {
    bool solutionIsValid = true;

    // validate solution
    size_t numInvalids = 0;
    size_t numChecked = 0;
    TensileStatus callStatus = tensileStatusSuccess;
    if (numElementsToValidate) {
      // copy data in language
#if Tensile_RUNTIME_LANGUAGE_OCL
      status = clEnqueueWriteBuffer(stream, static_cast<cl_mem>(deviceC), CL_TRUE, 0,
          sizeToCopyC, initialC, 0, NULL, NULL);
      tensileStatusCheck(status);
      if(!cEqualD)
        status = clEnqueueWriteBuffer(stream, static_cast<cl_mem>(deviceD), CL_TRUE, 0,
            sizeToCopyD, initialD, 0, NULL, NULL);
#else
      status = hipMemcpy(deviceC, initialC, sizeToCopyC, hipMemcpyHostToDevice);
      tensileStatusCheck(status);
      if(!cEqualD)
        status = hipMemcpy(deviceD, initialD, sizeToCopyD, hipMemcpyHostToDevice);
#endif
      tensileStatusCheck(status);

      // enqueue device solution
      callStatus = generatedCallToSolution( solutions[solutionIdx] , &solutionLocks[solutionIdx],
                                            sizes, minStrides,
                                            lda, ldb, ldc, ldd,
                                            strideA, strideB, strideC, strideD,
                                            alpha, beta );

      if (callStatus == tensileStatusSuccess) {
        // copy data back to host
#if Tensile_RUNTIME_LANGUAGE_OCL
        clEnqueueReadBuffer(stream, static_cast<cl_mem>(deviceC), CL_TRUE, 0,
            sizeToCopyC, deviceOnHostC, 0, NULL, NULL);
        if(!cEqualD)
          clEnqueueReadBuffer(stream, static_cast<cl_mem>(deviceD), CL_TRUE, 0,
              sizeToCopyD, deviceOnHostD, 0, NULL, NULL);
#else
        hipMemcpy(deviceOnHostC, deviceC, sizeToCopyC, hipMemcpyDeviceToHost);
        if(!cEqualD)
          hipMemcpy(deviceOnHostD, deviceD, sizeToCopyD, hipMemcpyDeviceToHost);
#endif
        if (printTensorC & 0x2) {
          std::vector<unsigned int> indexAssignmentsC;
          for (unsigned  int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
            indexAssignmentsC.push_back(i);
          }
          printTensor("C_out", deviceOnHostC, numIndicesC[problemTypeIdx],
                      numIndicesC[problemTypeIdx], sizes,
                      indexAssignmentsC.data());
        }
        if (printTensorD & 0x2) {
          std::vector<unsigned int> indexAssignmentsC;
          for (unsigned  int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
            indexAssignmentsC.push_back(i);
          }
          printTensor("D_out", deviceOnHostD, numIndicesC[problemTypeIdx],
                      numIndicesC[problemTypeIdx], sizes,
                      indexAssignmentsC.data());
        }

        // compare
        //
        bool firstPrint = true;
        unsigned int printIdx = 0;
        for (size_t e = 0; e < currentElementSizeC; e+= validationStride) {

          // Compute the actual serialIdxX accouting for strides:
          size_t serialIdxD = 0;
          size_t serialIdxC = 0;
          size_t r = e;
          for (int j = numIndicesC[problemTypeIdx]-1; j >=0; j--) {
            serialIdxD += r / elementStrides[j] * stridesD[j];
            serialIdxC += r / elementStrides[j] * stridesC[j];
            r = r % elementStrides[j];
          }

          bool equalC, equalD;
          equalD = tensileAlmostEqual<DataType>( // need AlmostEqual for StaggerU
              deviceOnHostD[serialIdxD], referenceD[serialIdxD]);
          equalC = tensileAlmostEqual<DataType>( // need AlmostEqual for StaggerU
              deviceOnHostC[serialIdxC], referenceC[serialIdxC]);
          numChecked++;
          
          if (!equalC || !equalD) numInvalids++;

          if (!equalC || !equalD || printValids) {
            if (printIdx < printMax) {
              if (firstPrint) {
                std::cout << "Index:  Device | Reference" << std::endl;
                firstPrint = false;
              }
              std::cout << "[" << (numChecked-1) << "] " 
                << " e=" << e
                << " serialIdxD=" << serialIdxD << ": "
                << tensileToString(deviceOnHostD[serialIdxD])
                << (equalD ? "==" : "!=") << tensileToString(referenceD[serialIdxD])
                << " , serialIdxC=" << serialIdxC << ": "
                << tensileToString(deviceOnHostC[serialIdxC])
                << (equalC ? "==" : "!=") << tensileToString(referenceC[serialIdxC])
                << std::endl;
              printIdx++;
            }
          }
        } // compare loop
        if (numInvalids) {
          returnInvalids = true;
          solutionIsValid = false;
        }
      } // if callStatus == success
      else
      {
#if Tensile_RUNTIME_LANGUAGE_OCL
#else
        if(callStatus == hipErrorFileNotFound)
        {
            printf("Kernel file not found; exiting.\n");
            exit(1);
        }
#endif
      }
    } // if numElementsToValidate > 0


    // time solution
    timer.start();
      // device stats
    unsigned long long avgCoreClock = 0;
    unsigned long long avgMemClock = 0;
    double avgTemp = 0;
    unsigned long long avgFanSpeed = 0;
    for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
      unsigned long long syncCoreClock = 0;
      unsigned long long syncMemClock = 0;
      double syncTemp = 0;
      unsigned long long syncFanSpeed = 0;
      for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
#if Tensile_RUNTIME_LANGUAGE_OCL
        TensileStatus status = generatedCallToSolution( solutions[solutionIdx], &solutionLocks[solutionIdx],
            sizes, minStrides, lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD, alpha, beta,
            0, NULL, &l_outputEvent[syncIdx][enqIdx] );
#else
        TensileStatus status = generatedCallToSolution( solutions[solutionIdx], &solutionLocks[solutionIdx],
            sizes, minStrides, lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD, alpha, beta,
            numEnqueuesPerSync, &l_eventStart[syncIdx][enqIdx],
            &l_eventStop[syncIdx][enqIdx] );

        if(status == hipErrorFileNotFound)
        {
            printf("Kernel file not found; exiting.\n");
            exit(1);
        }
#endif
        if (status != tensileStatusSuccess) {
          solutionIsValid = false;
        }

      }
      // sync
#if Tensile_RUNTIME_LANGUAGE_OCL
      status = clFinish(stream);
#else
      unsigned int numDeviceStatsQueries = 0;
      do { // device stats
        int currentCoreClock = tensileGetDeviceCoreClock(0);
        int currentMemClock = tensileGetDeviceMemClock(0);
        float currentTemp = tensileGetDeviceTemp(0);
        int currentFanSpeed = tensileGetDeviceFanSpeed(0);
        //std::cout << "clock: " << currentCoreClock << " Mhz" << std::endl;
        syncCoreClock += currentCoreClock;
        syncMemClock += currentMemClock;
        syncTemp += currentTemp;
        syncFanSpeed += currentFanSpeed;
        numDeviceStatsQueries++;
      } while (hipEventQuery(l_eventStop[syncIdx][numEnqueuesPerSync-1]) != hipSuccess);
      syncCoreClock /= numDeviceStatsQueries;
      syncMemClock /= numDeviceStatsQueries;
      syncTemp /= numDeviceStatsQueries;
      syncFanSpeed /= numDeviceStatsQueries;

      avgCoreClock += syncCoreClock;
      avgMemClock += syncMemClock;
      avgTemp += syncTemp;
      avgFanSpeed += syncFanSpeed;
#endif
      tensileStatusCheck(status);
    } // sync loop

    double timeNs = 0.0;
#if Tensile_RUNTIME_LANGUAGE_OCL
    if (useGPUTimer) {
      // Loop through the multi-dimensional event array and collect kernel performance data
      // Release events when done with them
      cl_ulong kernel_time_sum = 0;
      for (auto& event_array : l_outputEvent) {
        for (auto event : event_array) {
          // getEventDeltaTime returns unsigned long in nano-seconds on opencl
          kernel_time_sum += getEventDeltaTime(event);
        }
      }
      timeNs = static_cast<double>(kernel_time_sum);
    } else {
      timeNs = timer.elapsed_ns();
    }
#else
    if (useGPUTimer) {
      // Loop through the event array and collect kernel performance data
      // Release events when done with them
      float kernel_time_sum = 0;
      for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++){
        for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
          // getEventDeltaTime returns unsigned long in milli-seconds on hip
          float kernel_time = getEventDeltaTime(l_eventStart[syncIdx][enqIdx],
              l_eventStop[syncIdx][enqIdx] );
          //std::cout << "kernelTime: " << kernel_time << std::endl;
          kernel_time_sum += kernel_time;
        }
      }
      timeNs = static_cast<double>(kernel_time_sum)
        * TensileTimer::million;  // convert to nano-seconds
    } else {
      timeNs = timer.elapsed_ns();
    }

#endif
    if (sleepPercent) {
      unsigned int sleepMicroSeconds = (timeNs*10*sleepPercent)/1e6;
      usleep(sleepMicroSeconds);
    }

    *problem_gpu_time_ms += timeNs/1e6;
    //printf ("problem: %6.2f ms+ %6.2fns\n", *problem_gpu_time_ms, timeNs);

    timeNs /= (numSyncsPerBenchmark * numEnqueuesPerSync);
    // device status
    avgCoreClock /= numSyncsPerBenchmark;
    avgMemClock /= numSyncsPerBenchmark;
    avgTemp /= numSyncsPerBenchmark;
    avgFanSpeed /= numSyncsPerBenchmark;

    float perfScaling = 1.f;
    double gflops = solutionIsValid ? perfScaling * totalFlops / timeNs : 0;
    //std::cout << gflops << " gflops = " << totalFlops << " flops / " << timeNs << " ns" << std::endl;
    bool newFastest = false;
    if (gflops > fastestGFlops) {
      fastestGFlops = gflops;
      fastestIdx = solutionIdx;
      newFastest = true;
      if (fastestGFlops > globalFastestGFlops) {
        globalFastestGFlops = fastestGFlops;
        globalFastestTime = timeNs;
        globalFastestIdx = fastestIdx;
      }
    }

    // print results to stdout
    if (newFastest || numInvalids>0 || !printWinnersOnly || callStatus != tensileStatusSuccess) {
      std::cout << std::setw(10) << std::fixed << std::setprecision(3)
          << gflops*perfScaling << ", "
          << std::setw(10) << std::fixed << std::setprecision(3)
          << gflops << ", "
          << solutions[solutionIdx]._name << (newFastest ? "*" : " ") << ", "
          << std::setw(9) << std::fixed << std::setprecision(3)
          << timeNs * TensileTimer::reciprical_million << ", ";

      if (callStatus == tensileStatusSuccess) {
        if (numElementsToValidate) {
          std::cout << (numInvalids ? "FAILED" : "PASSED")
            << ": " << (numChecked-numInvalids) << "/" << numChecked << ", ";
        } else {
          std::cout << "NO_CHECK, "; // did not validate any results, may work or maybe not
        }
      } else if (callStatus == tensileStatusAssertFailure) {
        std::cout << "DID_NOT_SATISFY_ASSERTS, ";
      } else {
        std::cout << "INVALID_KERNEL, "; // launch function returned error?
      }

      // device stats
      std::cout << avgCoreClock << ", ";
      std::cout << avgMemClock << ", ";
      std::cout << avgTemp << ", ";
      std::cout << avgFanSpeed << ", ";

      std::cout << solutionIdx << "/" << numSolutions << ", ";

      struct timeval tmnow;
      struct tm *tm;
      gettimeofday(&tmnow, NULL); // microsecond resolution
      tm = localtime(&tmnow.tv_sec);
      char prev_fill = std::cout.fill('0');
      std::cout << (tm->tm_year + 1900) << "-"
        << std::setw(2) << (tm->tm_mon + 1) << "-"
        << std::setw(2) << tm->tm_mday << " "
        << std::setw(2) << tm->tm_hour << ":"
        << std::setw(2) << tm->tm_min << ":"
        << std::setw(2) << tm->tm_sec << "."
        << std::setw(6) << static_cast<int>(tmnow.tv_usec) << ", ";
      (void) std::cout.fill(prev_fill);

      std::cout << std::endl;
    }

    // write results to file
    if ((numInvalids > 0) || (callStatus != tensileStatusSuccess)) {
      gflops = -1.0;
      invalidSolutions.insert(solutionIdx);
    }
    file << ", " << gflops;
    solutionPerf[problemIdx][solutionIdx ] = static_cast<float>(gflops);
  } // solution loop

  if (useGPUTimer) {
#if Tensile_RUNTIME_LANGUAGE_OCL
    for (auto& event_array : l_outputEvent) {
      for (auto event : event_array) {
        ::clReleaseEvent(event);
      }
    }
#else
    for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++){
      for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
        ::hipEventDestroy( l_eventStart[syncIdx][enqIdx] );
        ::hipEventDestroy( l_eventStop[syncIdx][enqIdx] );
      }
    }
#endif
  }
  file << std::endl;

  return returnInvalids;
} // benchmark solutions
#endif // benchmark client

#if Tensile_CLIENT_BENCHMARK
template<typename DataType, typename DestDataType, typename ComputeDataType>
bool benchmarkSolutions(
    DestDataType *initialD,
    DestDataType *initialC,
    DataType *initialA,
    DataType *initialB,
    ComputeDataType alpha,
    ComputeDataType beta,
    DestDataType *referenceD,
    DestDataType *referenceC,
    DestDataType *deviceOnHostD,
    DestDataType *deviceOnHostC) {
  bool returnInvalids = false;

  // write benchmark data column headers
  //std::cout << std::endl;
  //std::cout << "Solutions: " << std::endl;
  //for (unsigned int sIdx = 0; sIdx < numSolutions; sIdx++) {
  //  std::cout << "(" << sIdx << ") " << solutions[sIdx]._name << std::endl;
  //}
  //file.open(resultsFileName);
  //file << "GFlops";
  //for ( unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
  //  file << ", Size" << indexChars[i];
  //}
  //file << ", LDD, LDC, LDA, LDB, TotalFlops";
  //for ( unsigned int s = 0; s < numSolutions; s++) {
  //  file << ", " << solutions[s]._name;
  //}
  //file << std::endl;

#if Tensile_RUNTIME_LANGUAGE_OCL
  if (!numElementsToValidate) {
    std::cout << "Pre-compiling " << numSolutions << " OpenCL kernels";
    for (unsigned int sIdx = 0; sIdx < numSolutions; sIdx++) {
      generatedCallToSolution( solutions[sIdx], &solutionLocks[sIdx], problemSizes[0], minStrides,
          lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD, alpha, beta );
      status = clFinish(stream); tensileStatusCheck(status);
      tensileStatusCheck(status);
      std::cout << ".";
    }
    std::cout << std::endl;
  }
#endif // opencl
  std::cout << std::endl;

  TensileTimer totalKernelTimer;
  totalKernelTimer.start();
  // iterate over all problem sizes
  double gpu_time_ms = 0;

  {

    // benchmark all solutions 
    double problem_gpu_time_ms;
    bool invalids = benchmarkAllSolutions( initialD, initialC,
        initialA, initialB, alpha, beta, referenceD, referenceC, deviceOnHostD, deviceOnHostC,
        &problem_gpu_time_ms);
    if (invalids) returnInvalids = true;
    gpu_time_ms += problem_gpu_time_ms;
    //printf ("gpu_time: %6.2f ms+ %6.2fns\n", gpu_time_ms, problem_gpu_time_ms);

    //std::map<std::vector<unsigned int>, std::set<std::pair<unsigned int,double>>>::iterator benchmarkIter = solutionBenchmarks.begin();
    std::map<std::vector<unsigned int>, std::set<std::pair<unsigned int,double>>>::iterator iter;   

    for (iter = solutionBenchmarks.begin(); iter != solutionBenchmarks.end(); iter++)
    {  
        //std::cout << "****** size ****** -> " << (*(*it).second.begin()).second << std::endl;


        std::vector<unsigned int> key = (*iter).first;

        double maxPerf = -1.0;
        std::pair<unsigned int,double> maxPair;

        std::set<std::pair<unsigned int,double>>::iterator benchmarksIter;
        for(benchmarksIter = (*iter).second.begin(); benchmarksIter != (*iter).second.end(); benchmarksIter++) 
        {
            //std::cout << "*** final perf **** -> " << (*benchmarksIter).second << std::endl;
            std::pair<unsigned int,double> myPerf = *benchmarksIter;
            if (myPerf.second > maxPerf)
            {
               maxPerf = myPerf.second;
               maxPair = myPerf;
            }
        }
        solutionMaxPeformance[key] = maxPair;
    }

    //std::map<std::vector<unsigned int>, std::pair<unsigned int,double>>::iterator piter;
    //for (piter = solutionMaxPeformance.begin(); piter != solutionMaxPeformance.end(); piter++)
    //{
    //   //std::vector<unsigned int> key = (*piter).first;
    //   std::pair<unsigned int,double> value = (*piter).second;
    //   std::cout << "**** max for key  *** -> (" << value.first << "," << value.second << ")" << std::endl; 
    //}


  } // for problemIdx
  auto timeK = totalKernelTimer.elapsed_sec();
  std::cout <<  "\nRun kernels elapsed gpu_time:" << gpu_time_ms/1000.0
            << " secs  total_time:" << timeK << " secs "
            << std::setprecision(2) << gpu_time_ms/timeK*(100.0/1000)
            << "% gpu utilization\n";

  //file.close();

  return returnInvalids;
} // benchmarkSolutions
#endif // benchmark

/*******************************************************************************
 * Benchmark Problem Sizes
 ******************************************************************************/
#if Tensile_CLIENT_BENCHMARK
template<typename DataType, typename DestDataType, typename ComputeDataType>
bool benchmarkProblemSizes(
    DestDataType *initialD,
    DestDataType *initialC,
    DataType *initialA,
    DataType *initialB,
    ComputeDataType alpha,
    ComputeDataType beta,
    DestDataType *referenceD,
    DestDataType *referenceC,
    DestDataType *deviceOnHostD,
    DestDataType *deviceOnHostC) {
  bool returnInvalids = false;

  // write benchmark data column headers
  std::cout << std::endl;
  file.open(resultsFileName);
  file << "GFlops";
  for ( unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
    file << ", Size" << indexChars[i];
  }
  file << ", LDD, LDC, LDA, LDB, TotalFlops";
  for ( unsigned int s = 0; s < numSolutions; s++) {
    file << ", " << solutions[s]._name;
  }
  file << std::endl;

//#if Tensile_RUNTIME_LANGUAGE_OCL
  //if (!numElementsToValidate) {
  //  std::cout << "Pre-compiling " << numSolutions << " OpenCL kernels";
    //for (unsigned int sIdx = 0; sIdx < numSolutions; sIdx++) {
  //  for (std::set<unsigned int>::iterator it=validSolutions.begin(); it!=validSolutions.end(); ++it) {
  //    unsigned int sIdx = *it;
  //    generatedCallToSolution( solutions[sIdx], &solutionLocks[sIdx], problemSizes[0], minStrides,
  //        lda, ldb, ldc, ldd, strideA, strideB, strideC, strideD, alpha, beta );
  //    status = clFinish(stream); tensileStatusCheck(status);
  //    tensileStatusCheck(status);
  //    std::cout << ".";
  //  }
 //   std::cout << std::endl;
 // }
//#endif // opencl
  std::cout << std::endl;

	TensileTimer totalKernelTimer;
  totalKernelTimer.start();
  // iterate over all problem sizes
  double gpu_time_ms = 0;

  for (unsigned int problemIdx = 0; problemIdx < numProblems; problemIdx++ ) {
 
    // print size
    std::cout << "Problem[" << problemIdx << "/" << numProblems << "]: " << problemSizes[problemIdx][0];
    for (unsigned int i = 1; i < totalIndices[problemTypeIdx]+numIndicesLD; i++) {
      std::cout << ", " << problemSizes[problemIdx][i];
    }
    std::cout << std::endl;

    // benchmark all solutions for this problem size
    double problem_gpu_time_ms;
    bool invalids = benchmarkAllSolutionsForSize(problemIdx, initialD, initialC,
        initialA, initialB, alpha, beta, referenceD, referenceC, deviceOnHostD, deviceOnHostC,
        &problem_gpu_time_ms);
    if (invalids) returnInvalids = true;
    gpu_time_ms += problem_gpu_time_ms;
    printf ("gpu_time: %6.2f ms+ %6.2fns\n", gpu_time_ms, problem_gpu_time_ms);

  } // for problemIdx
	auto timeK = totalKernelTimer.elapsed_sec();
  std::cout <<  "\nRun kernels elapsed gpu_time:" << gpu_time_ms/1000.0
            << " secs  total_time:" << timeK << " secs "
            << std::setprecision(2) << gpu_time_ms/timeK*(100.0/1000)
            << "% gpu utilization\n";

  // close file
  file.close();
  return returnInvalids;
} // benchmarkProblemSizes
#endif // benchmark

enum InitOp {None, Abs, AltSign};
template<typename DataType>
void initInput(
    const std::string &tag,
    unsigned dataInitType,
    DataType **initial,
    size_t     maxSize,
    InitOp     initOp)
{
  if (dataInitType == 0) {
    for (size_t i = 0; i < maxSize; i++) {
      (*initial)[i] = tensileGetZero<DataType>(); }
    std::cout << ".";
  } else if (dataInitType == 1) {
    for (size_t i = 0; i < maxSize; i++) {
      (*initial)[i] = tensileGetOne<DataType>(); }
    std::cout << ".";
  } else if (dataInitType == 2) {
    for (size_t i = 0; i < maxSize; i++) {
      (*initial)[i] = tensileGetTypeForInt<DataType>(i); }
    std::cout << ".";
  } else if (dataInitType == 3) {
    for (size_t i = 0; i < maxSize; i++) {
      DataType v = tensileGetRandom<DataType>();
      v = (v >= static_cast<DataType>(0)) ? v : static_cast<DataType>(0) - v;
      if (initOp == AltSign) {
        v = ((i & 0x1) == 0) ? v : static_cast<DataType>(0) - v;
      }
      (*initial)[i] = v;
    }
    std::cout << ".";
  } else if (dataInitType == 4) {
    for (size_t i = 0; i < maxSize; i++) {
      (*initial)[i] = tensileGetNaN<DataType>(); }
    std::cout << ".";
  } else if (dataInitType == 5) {
    // Will initialize later for each matrix dim:
    specializeAB = true;
  } else if (dataInitType == 6) {
    for (size_t i = 0; i < maxSize; i++) {
      DataType v = tensileGetTrig<DataType>(i);   // initialize with sin to get value between -1 and 1. 
      v = (v >= static_cast<DataType>(0)) ? v : static_cast<DataType>(0) - v;
      if (initOp == AltSign) {
        v = ((i & 0x1) == 0) ? v : static_cast<DataType>(0) - v;
      }
      (*initial)[i] = v;
    }
    std::cout << ".";
  } else {
    std::cout << "FATAL ERROR: Bad " << tag << " = " << dataInitType << "\n";
    exit(0);
  }
}


/*******************************************************************************
 * initialize data
 ******************************************************************************/
template<typename DataType, typename DestDataType, typename ComputeDataType>
void initData(
    DestDataType **initialD,
    DestDataType **initialC,
    DataType **initialA,
    DataType **initialB,
    ComputeDataType *alpha,
    ComputeDataType *beta,
    DestDataType **referenceD,
    DestDataType **referenceC,
    DestDataType **deviceOnHostD,
    DestDataType **deviceOnHostC) {
  //int seed = time(NULL);
  int seed = 0x1000;
  srand(seed);

  // initialize alpha
  if (initAlpha == 0) {
    *alpha = tensileGetZero<ComputeDataType>();
  } else if (initAlpha == 1) {
    *alpha = tensileGetOne<ComputeDataType>();
  } else if (initAlpha == 2) {
    *alpha = tensileGetTypeForInt<ComputeDataType>(2);
  } else if (initAlpha == 3) {
    *alpha = tensileGetRandom<ComputeDataType>();
  } else {
    *alpha = tensileGetNaN<ComputeDataType>();
  }

  // initialize beta
  if (useBeta[problemTypeIdx]) {
    if (initBeta == 0) {
      *beta = tensileGetZero<ComputeDataType>();
    } else if (initBeta == 1) {
      *beta = tensileGetOne<ComputeDataType>();
    } else if (initBeta == 2) {
      *beta = tensileGetTypeForInt<ComputeDataType>(2);
    } else if (initBeta == 3) {
      *beta = tensileGetRandom<ComputeDataType>();
    } else {
      *beta = tensileGetNaN<ComputeDataType>();
    }
  } else {
    *beta = tensileGetZero<ComputeDataType>();
  }

  size_t numElements = 3*maxSizeC+maxSizeA+maxSizeB;
  if(!cEqualD)
    numElements += 3*maxSizeD;
  std::cout << "Initializing "
    << (bytesPerElement[dataTypeIdx]*numElements/1000000)
    << " MBytes";
  std::cout << ".";

  // initial and reference buffers
  *referenceC = new DestDataType[maxSizeC];
  std::cout << ".";
  if(cEqualD)
    *referenceD = *referenceC;
  else
    *referenceD = new DestDataType[maxSizeD];
  std::cout << ".";
  *deviceOnHostC = new DestDataType[maxSizeC];
  std::cout << ".";
  if(cEqualD)
    *deviceOnHostD = *deviceOnHostC;
  else
    *deviceOnHostD = new DestDataType[maxSizeD];
  std::cout << ".";
  *initialC = new DestDataType[maxSizeC];
  std::cout << ".";
  if(cEqualD)
    *initialD = *initialC;
  else
    *initialD = new DestDataType[maxSizeD];
  std::cout << ".";
  *initialA = new DataType[maxSizeA];
  std::cout << ".";
  *initialB = new DataType[maxSizeB];
  std::cout << ".";

  // initialize buffers
  initInput("DataInitTypeA", initA, initialA, maxSizeA, Abs);
  initInput("DataInitTypeB", initB, initialB, maxSizeB, AltSign);
  initInput("DataInitTypeC", initC, initialC, maxSizeC, None);
  if(!cEqualD)
    initInput("DataInitTypeD", initD, initialD, maxSizeD, None);

  // create device buffers and copy data
#if Tensile_RUNTIME_LANGUAGE_OCL
  deviceC = clCreateBuffer(context, CL_MEM_READ_WRITE,
      maxSizeC*bytesPerElement[dataTypeIdx], NULL, &status);
  tensileStatusCheck(status);
    std::cout << ".";
  if(cEqualD)
    deviceD = deviceC;
  else
  {
    deviceD = clCreateBuffer(context, CL_MEM_READ_WRITE,
        maxSizeD*bytesPerElement[dataTypeIdx], NULL, &status);
    tensileStatusCheck(status);
  }
  std::cout << ".";
  deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY,
      maxSizeA*bytesPerElement[dataTypeIdx], NULL, &status);
  tensileStatusCheck(status);
  std::cout << ".";
  deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY,
      maxSizeB*bytesPerElement[dataTypeIdx], NULL, &status);
  tensileStatusCheck(status);
  std::cout << ".";
#else
  status = hipMalloc( &deviceC, maxSizeC*bytesPerElement[dataTypeIdx] );
  tensileStatusCheck(status);
  std::cout << ".";
  if(cEqualD)
    deviceD = deviceC;
  else
  {
    status = hipMalloc( &deviceD, maxSizeD*bytesPerElement[dataTypeIdx] );
    tensileStatusCheck(status);
  }
  std::cout << ".";
  status = hipMalloc( &deviceA, maxSizeA*bytesPerElement[dataTypeIdx] );
  tensileStatusCheck(status);
  std::cout << ".";
  status = hipMalloc( &deviceB, maxSizeB*bytesPerElement[dataTypeIdx] );
  tensileStatusCheck(status);
  std::cout << ".";
#endif

  if (!specializeAB) {
    // Specialized data is initialized and copied before each matrix run:
    copyData<DataType>(*initialA, *initialB);
  }

  std::cout << std::endl;
}





/*******************************************************************************
 * destroy data
 ******************************************************************************/
template<typename DataType, typename DestDataType>
void destroyData(
    DestDataType *initialD,
    DestDataType *initialC,
    DataType *initialA,
    DataType *initialB,
    DestDataType *referenceD,
    DestDataType *referenceC,
    DestDataType *deviceOnHostD,
    DestDataType *deviceOnHostC) {

  delete[] initialC;
  if(!cEqualD)
    delete[] initialD;
  delete[] initialA;
  delete[] initialB;
  delete[] referenceC;
  if(!cEqualD)
    delete[] referenceD;
  delete[] deviceOnHostC;
  if(!cEqualD)
    delete[] deviceOnHostD;

#if Tensile_RUNTIME_LANGUAGE_OCL
  clReleaseMemObject(static_cast<cl_mem>(deviceC));
  if(!cEqualD)
    clReleaseMemObject(static_cast<cl_mem>(deviceD));
  clReleaseMemObject(static_cast<cl_mem>(deviceA));
  clReleaseMemObject(static_cast<cl_mem>(deviceB));
#else
  hipFree(deviceC);
  if(!cEqualD)
    hipFree(deviceD);
  hipFree(deviceA);
  hipFree(deviceB);
#endif

}


void printClientUsage(std::string executableName) {
  std::cout << "Usage: " << executableName << std::endl;
  std::cout << "  " << keyDeviceIdx << " [" << defaultDeviceIdx << "]" << std::endl;
  std::cout << "  " << keyInitC << " [" << defaultInitC << "]" << std::endl;
  std::cout << "  " << keyInitA << " [" << defaultInitA << "]" << std::endl;
  std::cout << "  " << keyInitB << " [" << defaultInitB << "]" << std::endl;
  std::cout << "  " << keyInitAlpha << " [" << defaultInitAlpha << "]" << std::endl;
  std::cout << "  " << keyInitBeta << " [" << defaultInitBeta << "]" << std::endl;
  std::cout << "  " << keyCEqualD << " [" << defaultCEqualD << "]" << std::endl;
#if Tensile_RUNTIME_LANGUAGE_OCL
  std::cout << "  " << keyPlatformIdx << " [" << defaultPlatformIdx << "]" << std::endl;
#endif
  std::cout << "  " << keyPrintValids << " [" << defaultPrintValids << "]" << std::endl;
  std::cout << "  " << keyPrintMax << " [" << defaultPrintMax << "]" << std::endl;
  std::cout << "  " << keyNumBenchmarks << " [" << defaultNumBenchmarks << "]" << std::endl;
  std::cout << "  " << keyNumElementsToValidate << " [" << defaultNumElementsToValidate << "]" << std::endl;
  std::cout << "  " << keyNumEnqueuesPerSync << " [" << defaultNumEnqueuesPerSync << "]" << std::endl;
  std::cout << "  " << keyNumSyncsPerBenchmark << " [" << defaultNumSyncsPerBenchmark << "]" << std::endl;
  std::cout << "  " << keyUseGPUTimer << " [" << defaultUseGPUTimer << "]" << std::endl;
  std::cout << "  " << keySleepPercent << " [" << defaultSleepPercent << "]" << std::endl;
  std::cout << "  " << keyLda << " [defaut is size of array A]" << std::endl;
  std::cout << "  " << keyLdb << " [defaut is size of array B]" << std::endl;
  std::cout << "  " << keyLdc << " [defaut is size of array C]" << std::endl;
  std::cout << "  " << keyLdd << " [defaut is size of array D]" << std::endl;
  std::cout << "  " << keyStrideA << " [defaut is size of array A]" << std::endl;
  std::cout << "  " << keyStrideB << " [defaut is size of array B]" << std::endl;
  std::cout << "  " << keyStrideC << " [defaut is size of array C]" << std::endl;
  std::cout << "  " << keyStrideD << " [defaut is size of array D]" << std::endl;
#if Tensile_CLIENT_LIBRARY
  std::cout << "  " << keyFunctionIdx << " [" << defaultFunctionIdx << "]" << std::endl;
  std::cout << "  " << keySizes << " [" << defaultSize << " " << defaultSize << " 1 " << defaultSize << "]" << std::endl;
  std::cout << "FunctionIdx:" << std::endl;
  for (unsigned int i = 0; i < numFunctions; i++) {
    std::cout << "  (" << i << ") " << functionNames[i] << std::endl;
  }
#else
  std::cout << "  " << keySolutionStartIdx << " [" << defaultSolutionStartIdx << "]" << std::endl;  
  std::cout << "  " << keyNumSolutions << " [" << defaultNumSolutions << "]" << std::endl;  
  std::cout << "  " << keyBenchmarkSolutions << " [" << defaultBenchmarkSolutions << "]" << std::endl;
#endif
}


/*******************************************************************************
 * Parse Command Line Parameters
 ******************************************************************************/
void parseCommandLineParameters( int argc, char *argv[] ) {
  std::string executableName(argv[0]);

  // set benchmark parameters to default values before parsing user values
  deviceIdx = defaultDeviceIdx;
  initAlpha = defaultInitAlpha;
  initBeta = defaultInitBeta;
  initD = defaultInitD;
  initC = defaultInitC;
  initA = defaultInitA;
  initB = defaultInitB;
  cEqualD = defaultCEqualD;
  specializeAB = false;
  platformIdx = defaultPlatformIdx;
  printValids = defaultPrintValids;
  printMax = defaultPrintMax;
  numBenchmarks = defaultNumBenchmarks;
  numElementsToValidate = defaultNumElementsToValidate;
  numEnqueuesPerSync = defaultNumEnqueuesPerSync;
  numSyncsPerBenchmark = defaultNumSyncsPerBenchmark;
  useGPUTimer = defaultUseGPUTimer;
  sleepPercent = defaultSleepPercent;
#if Tensile_CLIENT_LIBRARY
  functionIdx = defaultFunctionIdx;
  dataTypeIdx = functionInfo[functionIdx][0];
  problemTypeIdx = functionInfo[functionIdx][2];
  for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
    userSizes[i] = defaultSize;
  }
#else
  solutionStartIdx = defaultSolutionStartIdx;
  numSolutions = defaultNumSolutions;
  runBenchmarkSolutions = defaultBenchmarkSolutions;
#endif

  try {
    // check for help
    for (unsigned int argIdx = 1; argIdx < argc; argIdx++) {
      if (keyHelp1 == argv[argIdx] || keyHelp2 == argv[argIdx]) {
          printClientUsage(executableName);
          exit(0);
      }
    }
#if Tensile_CLIENT_LIBRARY
    // first, get functionIdx
    functionIdx = defaultFunctionIdx;
    for (unsigned int argIdx = 1; argIdx < argc; argIdx++) {
      if (keyFunctionIdx == argv[argIdx]) {
        functionIdx = static_cast<unsigned int>(atoi(argv[argIdx+1]));
        if (functionIdx >= numFunctions) {
          std::cout << "FATAL ERROR: FunctionIdx=" << functionIdx << " >= "
            << "NumFunctions=" << numFunctions << std::endl;
          printClientUsage(executableName);
          exit(0);
        }
      }
    }
    dataTypeIdx = functionInfo[functionIdx][0];
    problemTypeIdx = functionInfo[functionIdx][2];
    for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
      userSizes[i] = defaultSize;
    }
#endif

    // second, get other arguments
    for (unsigned int argIdx = 1; argIdx < argc; argIdx++) {
      // device idx
      if (keyDeviceIdx == argv[argIdx]) {
        argIdx++;
        deviceIdx = static_cast<unsigned int>(atoi(argv[argIdx]));

      // init alpha
      } else if (keyInitAlpha == argv[argIdx]) {
        argIdx++;
        initAlpha = static_cast<unsigned int>(atoi(argv[argIdx]));

      // init beta
      } else if (keyInitBeta == argv[argIdx]) {
        argIdx++;
        initBeta = static_cast<unsigned int>(atoi(argv[argIdx]));

      // init d
      } else if (keyInitD == argv[argIdx]) {
        argIdx++;
        initD = static_cast<unsigned int>(atoi(argv[argIdx]));

      // init c
      } else if (keyInitC == argv[argIdx]) {
        argIdx++;
        initC = static_cast<unsigned int>(atoi(argv[argIdx]));

      // init ab
      } else if (keyInitA == argv[argIdx]) {
        argIdx++;
        initA = static_cast<unsigned int>(atoi(argv[argIdx]));

      } else if (keyInitB == argv[argIdx]) {
        argIdx++;
        initB = static_cast<unsigned int>(atoi(argv[argIdx]));

      // c == d
      } else if (keyCEqualD == argv[argIdx]) {
        argIdx++;
        cEqualD = static_cast<unsigned int>(atoi(argv[argIdx]));

      // platform idx
      } else if (keyPlatformIdx == argv[argIdx]) {
        argIdx++;
        platformIdx = static_cast<unsigned int>(atoi(argv[argIdx]));

      // print valids
      } else if (keyPrintValids == argv[argIdx]) {
        argIdx++;
        printValids = static_cast<unsigned int>(atoi(argv[argIdx]));

      // print max
      } else if (keyPrintMax == argv[argIdx]) {
        argIdx++;
        printMax = static_cast<unsigned int>(atoi(argv[argIdx]));

      // num benchmarks
      } else if (keyNumBenchmarks == argv[argIdx]) {
        argIdx++;
        numBenchmarks = static_cast<unsigned int>(atoi(argv[argIdx]));

      // num elements to validate
      } else if (keyNumElementsToValidate == argv[argIdx]) {
        argIdx++;
        numElementsToValidate = static_cast<unsigned int>(atoi(argv[argIdx]));

      // num enqueues per sync
      } else if (keyNumEnqueuesPerSync == argv[argIdx]) {
        argIdx++;
        numEnqueuesPerSync = static_cast<unsigned int>(atoi(argv[argIdx]));

      // num syncs per benchmark
      } else if (keyNumSyncsPerBenchmark == argv[argIdx]) {
        argIdx++;
        numSyncsPerBenchmark = static_cast<unsigned int>(atoi(argv[argIdx]));

      // use gpu timer
      } else if (keyUseGPUTimer == argv[argIdx]) {
        argIdx++;
        useGPUTimer = static_cast<unsigned int>(atoi(argv[argIdx]));

      // sleep percent
      } else if (keySleepPercent == argv[argIdx]) {
        argIdx++;
        sleepPercent = static_cast<unsigned int>(atoi(argv[argIdx]));

      } else if (keyLda == argv[argIdx]) {
        argIdx++;
        lda = static_cast<unsigned int>(atoi(argv[argIdx]));

      } else if (keyLdb == argv[argIdx]) {
        argIdx++;
        ldb = static_cast<unsigned int>(atoi(argv[argIdx]));

      } else if (keyLdc == argv[argIdx]) {
        argIdx++;
        ldc = static_cast<unsigned int>(atoi(argv[argIdx]));

      } else if (keyLdd == argv[argIdx]) {
        argIdx++;
        ldd = static_cast<unsigned int>(atoi(argv[argIdx]));

      } else if (keyStrideA == argv[argIdx]) {
        argIdx++;
        strideA = static_cast<unsigned int>(atoi(argv[argIdx]));

      } else if (keyStrideB == argv[argIdx]) {
        argIdx++;
        strideB = static_cast<unsigned int>(atoi(argv[argIdx]));

      } else if (keyStrideC == argv[argIdx]) {
        argIdx++;
        strideC = static_cast<unsigned int>(atoi(argv[argIdx]));

      } else if (keyStrideD == argv[argIdx]) {
        argIdx++;
        strideD = static_cast<unsigned int>(atoi(argv[argIdx]));
      }
#if Tensile_CLIENT_LIBRARY
      // function idx
      else if (keyFunctionIdx == argv[argIdx]) {
        argIdx++;
        // handled above

      // sizes
      } else if (keySizes == argv[argIdx]) {
        argIdx++;
        for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
          userSizes[i] = static_cast<unsigned int>(atoi(argv[argIdx]));
          argIdx++;
        }
        argIdx--; // b/c incremented at end of loop
      }
#else
      // solution start idx
      else if (keySolutionStartIdx == argv[argIdx]) {
        argIdx++;
        solutionStartIdx = static_cast<unsigned int>(atoi(argv[argIdx]));
        if (solutionStartIdx >= maxNumSolutions) {
          std::cout << "Tensile::FATAL: " << keySolutionStartIdx << " " << solutionStartIdx << " must be less than maxNumSolutions " << maxNumSolutions  << std::endl;
          throw -1;
        }

      // num solutions
      } else if (keyNumSolutions == argv[argIdx]) {
        argIdx++;
        numSolutions = static_cast<unsigned int>(atoi(argv[argIdx]));
        if (numSolutions > maxNumSolutions) {
          std::cout << "Tensile::FATAL: " << keyNumSolutions << " " << numSolutions << " must be less than maxNumSolutions " << maxNumSolutions  << std::endl;
          throw -1;
        }
      } else if (keyBenchmarkSolutions == argv[argIdx]) {
        argIdx++;
        runBenchmarkSolutions = static_cast<unsigned int>(atoi(argv[argIdx]));
      }
#endif
      // unrecognized
      else {
       std::cout << "Unrecognized: " << argv[argIdx] << std::endl;
       printClientUsage(executableName);
       exit(0);
      }
    } // loop
#if Tensile_CLIENT_BENCHMARK
    if (solutionStartIdx + numSolutions > maxNumSolutions) {
      std::cout << "Tensile::FATAL: " << keySolutionStartIdx << " " << solutionStartIdx << " + " << keyNumSolutions << " " << numSolutions << " must be less than maxNumSolutions " << maxNumSolutions  << std::endl;
      throw -1;
    }
#endif
  } catch (...) {
    printClientUsage(executableName);
    exit(0);
  }

#if Tensile_CLIENT_LIBRARY
  // max tensor sizes
  maxSizeC = 1;
  for (unsigned int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
    maxSizeC *= userSizes[i];
  }
  maxSizeD = maxSizeC;

  maxSizeA = 1;
  maxSizeB = 1;
  for (unsigned int i = 0; i < numIndicesAB[problemTypeIdx]; i++) {
    maxSizeA *= userSizes[indexAssignmentsA[problemTypeIdx][i]];
    maxSizeB *= userSizes[indexAssignmentsB[problemTypeIdx][i]];
  }
#endif

}

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
#include <unistd.h>

TensileTimer timer;
TensileTimer apiTimer;
std::ofstream file;

// benchmark parameters
unsigned int deviceIdx;
unsigned int initAlpha;
unsigned int initBeta;
unsigned int initC;
unsigned int initAB;
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
#endif

// benchmark parameters commandline strings
std::string keyDeviceIdx = "--device-idx";
std::string keyHelp1 = "-h";
std::string keyHelp2 = "--help";
std::string keyInitC = "--init-c";
std::string keyInitAB = "--init-ab";
std::string keyInitAlpha = "--init-alpha";
std::string keyInitBeta = "--init-beta";
std::string keyPlatformIdx = "--platform-idx";
std::string keyPrintValids = "--print-valids";
std::string keyPrintMax = "--print-max";
std::string keyNumBenchmarks = "--num-benchmarks";
std::string keyNumElementsToValidate = "--num-elements-to-validate";
std::string keyNumEnqueuesPerSync = "--num-enqueues-per-sync";
std::string keyNumSyncsPerBenchmark = "--num-syncs-per-benchmark";
std::string keyUseGPUTimer = "--use-gpu-timer";
std::string keySleepPercent = "--sleep-percent";
#if Tensile_CLIENT_BENCHMARK
std::string keySolutionStartIdx = "--solution-start-idx";
std::string keyNumSolutions = "--num-solutions";
#endif

// benchmark parameters default values
unsigned int defaultDeviceIdx = 0;
unsigned int defaultInitAlpha = 2;
unsigned int defaultInitBeta = 2;
unsigned int defaultInitC = 3;
unsigned int defaultInitAB = 3;
unsigned int defaultPlatformIdx = 0;
unsigned int defaultPrintValids = 0;
unsigned int defaultPrintMax = 0;
unsigned int defaultNumBenchmarks = 1;
unsigned int defaultNumElementsToValidate = 0;
unsigned int defaultNumEnqueuesPerSync = 1;
unsigned int defaultNumSyncsPerBenchmark = 1;
unsigned int defaultUseGPUTimer = 1;
unsigned int defaultSleepPercent = 0;
#if Tensile_CLIENT_BENCHMARK
unsigned int defaultSolutionStartIdx = 0;
unsigned int defaultNumSolutions = maxNumSolutions;
#endif

// benchmark parameters for library client
#if Tensile_CLIENT_LIBRARY
std::string keyFunctionIdx = "--function-idx";
std::string keySizes = "--sizes";
unsigned int defaultFunctionIdx = 0;
unsigned int defaultSize = 128;
#endif

#if Tensile_CLIENT_BENCHMARK
std::set<unsigned int> invalidSolutions;
#endif

int expectedClockRate; // MHz
void initControls();
void destroyControls();

double globalFastestGFlops = 0.0;
unsigned int globalFastestIdx = 0;
double fastestGFlops = 0.0;
unsigned int fastestIdx = 0;

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


/*******************************************************************************
 * Call Library
 * return true if errors/invalids
 ******************************************************************************/
#if Tensile_CLIENT_LIBRARY
template<typename DataType>
bool callLibrary(
    DataType *initialC,
    DataType *initialA,
    DataType *initialB,
    DataType alpha,
    DataType beta,
    DataType *referenceC,
    DataType *deviceOnHostC ) {

  size_t currentSizeC = 1;
  for (unsigned int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
    currentSizeC *= userSizes[i];
  }
  size_t totalFlops = numFlopsPerMac[dataTypeIdx];
  for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
    totalFlops *= userSizes[i]; }

  if (printTensorA) {
    printTensor("A", initialA, numIndicesAB[problemTypeIdx],
                  numIndicesC[problemTypeIdx],
                  sizes, 
                  indexAssignmentsA[problemTypeIdx]);
  }
  if (printTensorB) {
    printTensor("B", initialB, numIndicesAB[problemTypeIdx],
                  numIndicesC[problemTypeIdx],
                  indexAssignmentsB[problemTypeIdx]);
  }

  size_t sizeToCopy = currentSizeC*bytesPerElement[dataTypeIdx];
#if Tensile_RUNTIME_LANGUAGE_OCL
  status = clEnqueueWriteBuffer(stream, static_cast<cl_mem>(deviceC), CL_TRUE,
      0, sizeToCopy, initialC, 0, NULL, NULL);
#else
  status = hipMemcpy(deviceC, initialC, sizeToCopy, hipMemcpyHostToDevice);
#endif
  tensileStatusCheck(status);

  size_t numInvalids = 0;
  size_t numChecked = 0;

  // do validation
  bool solutionIsValid = true;
  if (numElementsToValidate) {
    memcpy(referenceC, initialC, sizeToCopy);
    // calculate validation stride
    if (numElementsToValidate >= currentSizeC) {
      validationStride = 1;
    } else {
      if (numElementsToValidate) {
        validationStride = currentSizeC / numElementsToValidate;
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
    TensileStatus referenceStatus = generatedCallToReferenceCPU( userSizes, referenceC,
        initialA, initialB,
        alpha, beta);

    // call device function
    TensileStatus tensileStatus = generatedCallTo_tensile( userSizes, alpha, beta);
    if (tensileStatus == tensileStatusFailure) {
      solutionIsValid = false;
    }

    // copy data back to host
#if Tensile_RUNTIME_LANGUAGE_OCL
    clEnqueueReadBuffer(stream, static_cast<cl_mem>(deviceC), CL_TRUE, 0,
        sizeToCopy, deviceOnHostC, 0, NULL,
        NULL);
#else
    hipMemcpy(deviceOnHostC, deviceC, sizeToCopy, hipMemcpyDeviceToHost);
#endif

    // compare
    bool firstPrint = true;
    unsigned int printIdx = 0;
    for (size_t i = 0; i < currentSizeC; i+= validationStride) {
      bool equal;
      equal = tensileEqual<DataType>( // was AlmostEqual
          deviceOnHostC[i], referenceC[i]);
      numChecked++;
      if (!equal) numInvalids++;

      if (!equal || printValids) {
        if (printIdx < printMax) {
          if (firstPrint) {
            std::cout << "  Device | Reference" << std::endl;
            firstPrint = false;
          }
          std::cout << "[" << (numChecked-1) << "] " << i << ": "
            << tensileToString(deviceOnHostC[i])
            << (equal ? "==" : "!=") << tensileToString(referenceC[i])
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
      generatedCallTo_tensile(userSizes, alpha, beta, 0, NULL,
          &l_outputEvent[syncIdx][enqIdx]);
#else
      generatedCallTo_tensile(userSizes, alpha, beta, numEnqueuesPerSync,
          &l_eventStart[syncIdx][enqIdx], &l_eventStop[syncIdx][enqIdx]);
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
#if Tensile_RUNTIME_LANGUAGE_HIP
  perfScaling = 1.f * expectedClockRate / avgCoreClock; // if clock speeds drop
#endif
  double gflops = solutionIsValid ? perfScaling * totalFlops / timeNs : 0;

  bool newFastest = false;
  if (gflops > fastestGFlops) {
    fastestGFlops = gflops;
    fastestIdx = functionIdx;
    newFastest = true;
    if (fastestGFlops > globalFastestGFlops) {
      globalFastestGFlops = fastestGFlops;
      globalFastestIdx = fastestIdx;
    }
  }

  const char * solutionName = generatedCallTo_tensileGetSolutionName(
      userSizes, alpha, beta);

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

/*******************************************************************************
 * benchmark all solutions for problem size
 * return true if error/invalids
 ******************************************************************************/
#if Tensile_CLIENT_BENCHMARK
template<typename DataType>
bool benchmarkAllSolutionsForSize(
    unsigned int problemIdx,
    //unsigned int *sizes,
    DataType *initialC,
    DataType *initialA,
    DataType *initialB,
    DataType alpha,
    DataType beta,
    DataType *referenceC,
    DataType *deviceOnHostC) {
  const unsigned int *sizes = problemSizes[problemIdx];

  bool returnInvalids = false;
  size_t currentSizeC = 1;
  for (unsigned int i = 0; i < numIndicesC[problemTypeIdx]; i++) {
    currentSizeC *= sizes[i];
  }
  size_t sizeToCopy = currentSizeC*bytesPerElement[dataTypeIdx];

  file << problemIdx << ", " << sizes[0];
  for (unsigned int i = 1; i < totalIndices[problemTypeIdx]; i++) {
    file << ", " << sizes[i];
  }
  size_t totalFlops = numFlopsPerMac[dataTypeIdx];
  for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
    totalFlops *= sizes[i]; }
  file << ", " << totalFlops;

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

  // pre-compute referenceCPU if validating
  if (numElementsToValidate) {
    memcpy(referenceC, initialC, sizeToCopy);
    if (numElementsToValidate >= currentSizeC) {
      validationStride = 1;
    } else {
      if (numElementsToValidate) {
        validationStride = currentSizeC / numElementsToValidate;
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
    generatedCallToReferenceCPU( sizes, referenceC, initialA, initialB,
        alpha, beta);

  }


  fastestGFlops = 0;
  for (unsigned int solutionIdx = solutionStartIdx; solutionIdx < solutionStartIdx + numSolutions; solutionIdx ++) {
    bool solutionIsValid = true;


    // copy data in language
#if Tensile_RUNTIME_LANGUAGE_OCL
    status = clEnqueueWriteBuffer(stream, static_cast<cl_mem>(deviceC), CL_TRUE, 0,
        sizeToCopy, initialC, 0, NULL, NULL);
#else
    status = hipMemcpy(deviceC, initialC, sizeToCopy, hipMemcpyHostToDevice);
#endif
    tensileStatusCheck(status);

    // validate solution
    size_t numInvalids = 0;
    size_t numChecked = 0;
    if (numElementsToValidate) {

      // enqueue device solution
      generatedCallToSolution( solutionIdx , sizes, alpha, beta );

      // copy data back to host
#if Tensile_RUNTIME_LANGUAGE_OCL
      clEnqueueReadBuffer(stream, static_cast<cl_mem>(deviceC), CL_TRUE, 0,
          sizeToCopy, deviceOnHostC, 0, NULL, NULL);
#else
      hipMemcpy(deviceOnHostC, deviceC, sizeToCopy, hipMemcpyDeviceToHost);
#endif

      // compare
      //unsigned int maxPrint = 16;
      //bool printTrue = false;
      bool firstPrint = true;
      unsigned int printIdx = 0;
      for (size_t i = 0; i < currentSizeC; i+= validationStride) {
        bool equal;
        equal = tensileEqual<DataType>( // was AlmostEqual
            deviceOnHostC[i], referenceC[i]);
        numChecked++;
        if (!equal) numInvalids++;

        if (!equal || printValids) {
          if (printIdx < printMax) {
            if (firstPrint) {
              std::cout << "  Device | Reference" << std::endl;
              firstPrint = false;
            }
            std::cout << "[" << (numChecked-1) << "] " << i << ": "
              << tensileToString(deviceOnHostC[i])
              << (equal ? "==" : "!=") << tensileToString(referenceC[i])
              << std::endl;
            printIdx++;
          }
        }
      } // compare loop
      if (numInvalids) {
        returnInvalids = true;
        solutionIsValid = false;
      }
    } // if numElementsToValidate > 0

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
        TensileStatus status = generatedCallToSolution( solutionIdx , sizes, alpha, beta,
            0, NULL, &l_outputEvent[syncIdx][enqIdx] );
#else
        TensileStatus status = generatedCallToSolution( solutionIdx, sizes, alpha, beta,
            numEnqueuesPerSync, &l_eventStart[syncIdx][enqIdx],
            &l_eventStop[syncIdx][enqIdx] );
#endif
        if (status == tensileStatusFailure) {
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
      for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++){
        for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
          // getEventDeltaTime returns unsigned long in milli-seconds on hip
          float kernel_time = getEventDeltaTime(l_eventStart[syncIdx][enqIdx],
              l_eventStop[syncIdx][enqIdx] );
          //std::cout << "kernelTime: " << kernel_time << std::endl;
          kernel_time_sum += kernel_time;
          ::hipEventDestroy( l_eventStart[syncIdx][enqIdx] );
          ::hipEventDestroy( l_eventStop[syncIdx][enqIdx] );
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

    timeNs /= (numSyncsPerBenchmark * numEnqueuesPerSync);
    // device status
    avgCoreClock /= numSyncsPerBenchmark;
    avgMemClock /= numSyncsPerBenchmark;
    avgTemp /= numSyncsPerBenchmark;
    avgFanSpeed /= numSyncsPerBenchmark;

    float perfScaling = 1.f;
#if Tensile_RUNTIME_LANGUAGE_HIP
    perfScaling = 1.f * expectedClockRate / avgCoreClock; // if clock speeds drop
#endif
    double gflops = solutionIsValid ? perfScaling * totalFlops / timeNs : 0;
    //std::cout << gflops << " gflops = " << totalFlops << " flops / " << timeNs << " ns" << std::endl;
    bool newFastest = false;
    if (gflops > fastestGFlops) {
      fastestGFlops = gflops;
      fastestIdx = solutionIdx;
      newFastest = true;
      if (fastestGFlops > globalFastestGFlops) {
        globalFastestGFlops = fastestGFlops;
        globalFastestIdx = fastestIdx;
      }
    }

    // print results to stdout
    std::cout << std::setw(10) << std::fixed << std::setprecision(3)
        << gflops*perfScaling << ", "
        << std::setw(10) << std::fixed << std::setprecision(3)
        << gflops << ", "
        << solutionNames[solutionIdx] << (newFastest ? "*" : " ") << ", "
        << std::setw(9) << std::fixed << std::setprecision(3)
        << timeNs * TensileTimer::reciprical_million << ", ";
    if (numElementsToValidate) {
      std::cout << (numInvalids ? "FAILED" : "PASSED")
        << ": " << (numChecked-numInvalids) << "/" << numChecked << ", ";
    }
    // device stats
    std::cout << avgCoreClock << ", ";
    std::cout << avgMemClock << ", ";
    std::cout << avgTemp << ", ";
    std::cout << avgFanSpeed << ", ";

    std::cout << solutionIdx << "/" << numSolutions << ", ";
    std::cout << std::endl;

    // write results to file
    if (numInvalids > 0) {
      gflops = -1.0;
      invalidSolutions.insert(solutionIdx);
    }
    file << ", " << gflops;
    solutionPerf[problemIdx][solutionIdx ] = static_cast<float>(gflops);
  } // solution loop
  file << std::endl;

  return returnInvalids;
} // benchmark solutions
#endif // benchmark client


/*******************************************************************************
 * Benchmark Problem Sizes
 ******************************************************************************/
#if Tensile_CLIENT_BENCHMARK
template<typename DataType>
bool benchmarkProblemSizes(
    DataType *initialC,
    DataType *initialA,
    DataType *initialB,
    DataType alpha,
    DataType beta,
    DataType *referenceC,
    DataType *deviceOnHostC) {
  bool returnInvalids = false;

  // write benchmark data column headers
  std::cout << std::endl;
  std::cout << "Solutions: " << std::endl;
  for (unsigned int sIdx = 0; sIdx < numSolutions; sIdx++) {
    std::cout << "(" << sIdx << ") " << solutionNames[sIdx] << std::endl;
  }
  //std::cout << "ResultsFileName: " << resultsFileName << std::endl;
  file.open(resultsFileName);
  file << "GFlops";
  for ( unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
    file << ", Size" << indexChars[i];
  }
  file << ", TotalFlops";
  for ( unsigned int s = 0; s < numSolutions; s++) {
    file << ", " << solutionNames[s];
  }
  file << std::endl;

#if Tensile_RUNTIME_LANGUAGE_OCL
  if (!numElementsToValidate) {
    std::cout << "Pre-compiling " << numSolutions << " OpenCL kernels";
    for (unsigned int sIdx = 0; sIdx < numSolutions; sIdx++) {
      generatedCallToSolution( sIdx, problemSizes[0], alpha, beta );
      status = clFinish(stream); tensileStatusCheck(status);
      tensileStatusCheck(status);
      std::cout << ".";
    }
    std::cout << std::endl;
  }
#endif // opencl
  std::cout << std::endl;

  // iterate over all problem sizes
  for (unsigned int problemIdx = 0; problemIdx < numProblems; problemIdx++ ) {

    // print size
    std::cout << "Problem[" << problemIdx << "/" << numProblems << "]: " << problemSizes[problemIdx][0];
    for (unsigned int i = 1; i < totalIndices[problemTypeIdx]; i++) {
      std::cout << ", " << problemSizes[problemIdx][i];
    }
    std::cout << std::endl;

    // benchmark all solutions for this problem size
    bool invalids = benchmarkAllSolutionsForSize( problemIdx, initialC,
        initialA, initialB, alpha, beta, referenceC, deviceOnHostC);
    if (invalids) returnInvalids = true;
  } // for problemIdx

  // close file
  file.close();
  return returnInvalids;
} // benchmarkProblemSizes
#endif // benchmark




/*******************************************************************************
 * initialize data
 ******************************************************************************/
template<typename DataType>
void initData(
    DataType **initialC,
    DataType **initialA,
    DataType **initialB,
    DataType *alpha,
    DataType *beta,
    DataType **referenceC,
    DataType **deviceOnHostC) {
  int seed = time(NULL);
  srand(seed);

  // initialize alpha
  if (initAlpha == 0) {
    *alpha = tensileGetZero<DataType>();
  } else if (initAlpha == 1) {
    *alpha = tensileGetOne<DataType>();
  } else if (initAlpha == 2) {
    *alpha = tensileGetTypeForInt<DataType>(2);
  } else if (initAlpha == 3) {
    *alpha = tensileGetRandom<DataType>();
  } else {
    *alpha = tensileGetNaN<DataType>();
  }

  // initialize beta
  if (useBeta[problemTypeIdx]) {
    if (initBeta == 0) {
      *beta = tensileGetZero<DataType>();
    } else if (initBeta == 1) {
      *beta = tensileGetOne<DataType>();
    } else if (initBeta == 2) {
      *beta = tensileGetTypeForInt<DataType>(2);
    } else if (initBeta == 3) {
      *beta = tensileGetRandom<DataType>();
    } else {
      *beta = tensileGetNaN<DataType>();
    }
  } else {
    *beta = tensileGetZero<DataType>();
  }

  std::cout << "Initializing "
    << (bytesPerElement[dataTypeIdx]*(maxSizeC+maxSizeA+maxSizeB)/1000000)
    << " MBytes";
  std::cout << ".";

  // initial and reference buffers
  *referenceC = new DataType[maxSizeC];
  *deviceOnHostC = new DataType[maxSizeC];
  *initialC = new DataType[maxSizeC];
  std::cout << ".";
  *initialA = new DataType[maxSizeA];
  std::cout << ".";
  *initialB = new DataType[maxSizeB];
  std::cout << ".";

  // initialize buffers C
  if (initC == 0) {
    for (size_t i = 0; i < maxSizeC; i++) {
      (*initialC)[i] = tensileGetZero<DataType>(); }
    std::cout << ".";
  } else if (initC == 1) {
    for (size_t i = 0; i < maxSizeC; i++) {
      (*initialC)[i] = tensileGetOne<DataType>(); }
    std::cout << ".";
  } else if (initC == 2) {
    for (size_t i = 0; i < maxSizeC; i++) {
      (*initialC)[i] = tensileGetTypeForInt<DataType>(i); }
    std::cout << ".";
  } else if (initC == 3) {
    for (size_t i = 0; i < maxSizeC; i++) {
      (*initialC)[i] = tensileGetRandom<DataType>(); }
    std::cout << ".";
  } else {
    for (size_t i = 0; i < maxSizeC; i++) {
      (*initialC)[i] = tensileGetNaN<DataType>(); }
    std::cout << ".";
  }

  // initialize buffers
  if (initAB == 0) {
    for (size_t i = 0; i < maxSizeA; i++) {
      (*initialA)[i] = tensileGetZero<DataType>(); }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      (*initialB)[i] = tensileGetZero<DataType>(); }
    std::cout << ".";
  } else if (initAB == 1) {
    for (size_t i = 0; i < maxSizeA; i++) {
      (*initialA)[i] = tensileGetOne<DataType>(); }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      (*initialB)[i] = tensileGetOne<DataType>(); }
    std::cout << ".";
  } else if (initAB == 2) {
    for (size_t i = 0; i < maxSizeA; i++) {
      (*initialA)[i] = tensileGetTypeForInt<DataType>(i); }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      (*initialB)[i] = tensileGetTypeForInt<DataType>(i); }
    std::cout << ".";
  } else {
    for (size_t i = 0; i < maxSizeA; i++) {
      (*initialA)[i] = tensileGetRandom<DataType>(); }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      (*initialB)[i] = tensileGetRandom<DataType>(); }
    std::cout << ".";
  }





  // create device buffers and copy data
#if Tensile_RUNTIME_LANGUAGE_OCL
  deviceC = clCreateBuffer(context, CL_MEM_READ_WRITE,
      maxSizeC*bytesPerElement[dataTypeIdx], NULL, &status);
  tensileStatusCheck(status);
    std::cout << ".";
  deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY,
      maxSizeA*bytesPerElement[dataTypeIdx], NULL, &status);
  tensileStatusCheck(status);
    std::cout << ".";
  deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY,
      maxSizeB*bytesPerElement[dataTypeIdx], NULL, &status);
  tensileStatusCheck(status);
    std::cout << ".";
  status = clEnqueueWriteBuffer(stream, static_cast<cl_mem>(deviceA), CL_TRUE,
      0, maxSizeA*bytesPerElement[dataTypeIdx], *initialA, 0, NULL, NULL);
  tensileStatusCheck(status);
    std::cout << ".";
  status = clEnqueueWriteBuffer(stream, static_cast<cl_mem>(deviceB), CL_TRUE,
      0, maxSizeB*bytesPerElement[dataTypeIdx], *initialB, 0, NULL, NULL);
  tensileStatusCheck(status);
    std::cout << ".";
#else
  status = hipMalloc( &deviceC, maxSizeC*bytesPerElement[dataTypeIdx] );
  tensileStatusCheck(status);
  std::cout << ".";
  status = hipMalloc( &deviceA, maxSizeA*bytesPerElement[dataTypeIdx] );
  tensileStatusCheck(status);
  std::cout << ".";
  status = hipMalloc( &deviceB, maxSizeB*bytesPerElement[dataTypeIdx] );
  tensileStatusCheck(status);
  std::cout << ".";
  status = hipMemcpy(deviceA, *initialA, maxSizeA*bytesPerElement[dataTypeIdx],
      hipMemcpyHostToDevice);
  status = hipMemcpy(deviceB, *initialB, maxSizeB*bytesPerElement[dataTypeIdx],
      hipMemcpyHostToDevice);
#endif

  copyData<DataType>(*initialA, *initialB);

  std::cout << std::endl;
}

/*******************************************************************************
 * destroy data
 ******************************************************************************/
template<typename DataType>
void destroyData(
    DataType *initialC,
    DataType *initialA,
    DataType *initialB,
    DataType *referenceC,
    DataType *deviceOnHostC) {

  delete[] initialC;
  delete[] initialA;
  delete[] initialB;
  delete[] referenceC;
  delete[] deviceOnHostC;

#if Tensile_RUNTIME_LANGUAGE_OCL
  clReleaseMemObject(static_cast<cl_mem>(deviceC));
  clReleaseMemObject(static_cast<cl_mem>(deviceA));
  clReleaseMemObject(static_cast<cl_mem>(deviceB));
#else
  hipFree(deviceC);
  hipFree(deviceA);
  hipFree(deviceB);
#endif

}


void printClientUsage(std::string executableName) {
  std::cout << "Usage: " << executableName << std::endl;
  std::cout << "  " << keyDeviceIdx << " [" << defaultDeviceIdx << "]" << std::endl;  
  std::cout << "  " << keyInitC << " [" << defaultInitC << "]" << std::endl;  
  std::cout << "  " << keyInitAB << " [" << defaultInitAB << "]" << std::endl;  
  std::cout << "  " << keyInitAlpha << " [" << defaultInitAlpha << "]" << std::endl;  
  std::cout << "  " << keyInitBeta << " [" << defaultInitBeta << "]" << std::endl;  
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
#if Tensile_CLIENT_LIBRARY
  std::cout << "  " << keyFunctionIdx << " [" << defaultFunctionIdx << "]" << std::endl;  
  std::cout << "  " << keySizes << " [" << defaultSize << " " << defaultSize << " " << defaultSize << "]" << std::endl;  
  std::cout << "FunctionIdx:" << std::endl;
  for (unsigned int i = 0; i < numFunctions; i++) {
    std::cout << "  (" << i << ") " << functionNames[i] << std::endl;
  }
#else
  std::cout << "  " << keySolutionStartIdx << " [" << defaultSolutionStartIdx << "]" << std::endl;  
  std::cout << "  " << keyNumSolutions << " [" << defaultNumSolutions << "]" << std::endl;  
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
  initC = defaultInitC;
  initAB = defaultInitAB;
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

      // init c
      } else if (keyInitC == argv[argIdx]) {
        argIdx++;
        initC = static_cast<unsigned int>(atoi(argv[argIdx]));

      // init ab
      } else if (keyInitAB == argv[argIdx]) {
        argIdx++;
        initAB = static_cast<unsigned int>(atoi(argv[argIdx]));

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
  maxSizeA = 1;
  maxSizeB = 1;
  for (unsigned int i = 0; i < numIndicesAB[problemTypeIdx]; i++) {
    maxSizeA *= userSizes[indexAssignmentsA[problemTypeIdx][i]];
    maxSizeB *= userSizes[indexAssignmentsB[problemTypeIdx][i]];
  }
#endif

}

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
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>

TensileTimer timer;
TensileTimer apiTimer;
std::ofstream file;

#if Tensile_CLIENT_BENCHMARK
unsigned int invalidSolutions[numSolutions];
unsigned int numInvalidSolutions = 0;
#endif

void initControls();
void destroyControls();

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

  // copy data to device
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
    generatedCallToReferenceCPU( userSizes, referenceC,
        initialA, initialB,
        alpha, beta);

    // call device function
    generatedCallTo_tensile( userSizes, alpha, beta);

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
      equal = tensileAlmostEqual<DataType>(
          deviceOnHostC[i], referenceC[i]);
      numChecked++;
      if (!equal) numInvalids++;

      if (!equal || validationPrintValids) {
        if (printIdx < validationMaxToPrint) {
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
  for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
    apiTimer.start();
    for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
      if (measureKernelTime) {
#if Tensile_RUNTIME_LANGUAGE_OCL
        generatedCallTo_tensile(userSizes, alpha, beta, 0, NULL,
            &l_outputEvent[syncIdx][0]);
#else
        generatedCallTo_tensile(userSizes, alpha, beta, numEnqueuesPerSync,
            &l_eventStart[syncIdx][0], &l_eventStop[syncIdx][0]);
#endif
      } else {
        generatedCallTo_tensile(userSizes, alpha, beta);
      }
    }
    double currentApiTimeUs = apiTimer.elapsed_us() / numEnqueuesPerSync;
    apiTimeUs += currentApiTimeUs;
    // sync
#if Tensile_RUNTIME_LANGUAGE_OCL
    status = clFinish(stream);
#else
    status = hipStreamSynchronize(stream);
#endif
    tensileStatusCheck(status);
  } // sync loop

  apiTimeUs /= numSyncsPerBenchmark;

  double timeNs = 0.0;
#if Tensile_RUNTIME_LANGUAGE_OCL
  if (measureKernelTime) {
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
  }
  else {
    timeNs = timer.elapsed_ns();
  }
#else
  if (measureKernelTime) {
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
  }
  else {
    timeNs = timer.elapsed_ns();
  }

#endif

  timeNs /= (numSyncsPerBenchmark / numEnqueuesPerSync);

  double gflops = totalFlops / timeNs;

  bool newFastest = false;
  if (gflops > fastestGFlops) {
    fastestGFlops = gflops;
    fastestIdx = functionIdx;
    newFastest = true;
  }

  const char * solutionName = generatedCallTo_tensileGetSolutionName(
      userSizes, alpha, beta);

  std::cout << std::setw(10) << std::fixed << std::setprecision(3)
      << gflops << ", "
      << solutionName << (newFastest ? "*" : " ") << ", "
      << std::setw(9) << std::fixed << std::setprecision(3)
      << timeNs * TensileTimer::reciprical_million << ", "
      << std::setw(7) << std::fixed << std::setprecision(3)
      << apiTimeUs << ", ";
  if (numElementsToValidate) {
    std::cout << (numInvalids ? "FAILED" : "PASSED")
      << ": " << (numChecked-numInvalids) << "/" << numChecked << ", ";
  }
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
    unsigned int *sizes,
    DataType *initialC,
    DataType *initialA,
    DataType *initialB,
    DataType alpha,
    DataType beta,
    DataType *referenceC,
    DataType *deviceOnHostC) {

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
  for (unsigned int solutionIdx = 0; solutionIdx < numSolutions; solutionIdx ++) {

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
        equal = tensileAlmostEqual<DataType>(
            deviceOnHostC[i], referenceC[i]);
        numChecked++;
        if (!equal) numInvalids++;

        if (!equal || validationPrintValids) {
          if (printIdx < validationMaxToPrint) {
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
      if (numInvalids) returnInvalids = true;
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
    for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
      for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
        if (measureKernelTime) {
#if Tensile_RUNTIME_LANGUAGE_OCL
          generatedCallToSolution( solutionIdx , sizes, alpha, beta,
              0, NULL, &l_outputEvent[syncIdx][0] );
#else
          generatedCallToSolution( solutionIdx, sizes, alpha, beta,
              numEnqueuesPerSync, &l_eventStart[syncIdx][0],
              &l_eventStop[syncIdx][0] );
#endif
        } else {
          generatedCallToSolution( solutionIdx, sizes, alpha, beta );
        }
      }
      // sync
#if Tensile_RUNTIME_LANGUAGE_OCL
      status = clFinish(stream);
#else
      status = hipStreamSynchronize(stream);
#endif
      tensileStatusCheck(status);
    } // sync loop

    double timeNs = 0.0;
#if Tensile_RUNTIME_LANGUAGE_OCL
    if (measureKernelTime) {
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
    if (measureKernelTime) {
      // Loop through the event array and collect kernel performance data
      // Release events when done with them
      float kernel_time_sum = 0;
      for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++){
        for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
          // getEventDeltaTime returns unsigned long in milli-seconds on hip
          kernel_time_sum += getEventDeltaTime(l_eventStart[syncIdx][enqIdx],
              l_eventStop[syncIdx][enqIdx] );
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

    timeNs /= (numSyncsPerBenchmark / numEnqueuesPerSync);

    double gflops = totalFlops / timeNs;
    bool newFastest = false;
    if (gflops > fastestGFlops) {
      fastestGFlops = gflops;
      fastestIdx = solutionIdx;
      newFastest = true;
    }

    // print results to stdout
    std::cout << std::setw(10) << std::fixed << std::setprecision(3)
        << gflops << ", "
        << solutionNames[solutionIdx] << (newFastest ? "*" : " ") << ", "
        << std::setw(9) << std::fixed << std::setprecision(3)
        << timeNs * TensileTimer::reciprical_million << ", ";
    if (numElementsToValidate) {
      std::cout << (numInvalids ? "FAILED" : "PASSED")
        << ": " << (numChecked-numInvalids) << "/" << numChecked << ", ";
    }
    std::cout << solutionIdx << "/" << numSolutions << ", ";
    std::cout << std::endl;

    // write results to file
    if (numInvalids > 0) {
      gflops = -1.0;
      invalidSolutions[numInvalidSolutions++] = solutionIdx;
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
  std::cout << "ResultsFileName: " << resultsFileName << std::endl;
  file.open(resultsFileName);
  file << "Milliseconds ";
  for ( unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
    file << ", Size" << indexChars[i];
  }
  file << ", TotalFlops";
  for ( unsigned int s = 0; s < numSolutions; s++) {
    file << ", " << solutionNames[s];
  }
  file << std::endl;

  // initialize index sizes
  for ( unsigned int i = 0; i < numIndicesSized; i++) {
    currentSizedIndexSizes[i] = indicesSized[i][0];
    currentSizedIndexIncrements[i] = indicesSized[i][1];
  }

  // run each solution to pre-compile opencl kernels if not validating
  unsigned int currentSizedIdx = 0;
  unsigned int currentMappedIdx = 0;
#if Tensile_RUNTIME_LANGUAGE_OCL
  if (!numElementsToValidate) {
    std::cout << "Pre-compiling " << numSolutions << " OpenCL kernels";
    for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
      if (indexIsSized[i]) {
        fullSizes[i] = currentSizedIndexSizes[currentSizedIdx++];
      }
#if Tensile_INDICES_MAPPED
      else {
        fullSizes[i] = fullSizes[indicesMapped[currentMappedIdx++]];
      }
#endif
    }
    for (unsigned int sIdx = 0; sIdx < numSolutions; sIdx++) {
      generatedCallToSolution( sIdx, fullSizes, alpha, beta );
      status = clFinish(stream); tensileStatusCheck(status);
      tensileStatusCheck(status);
      std::cout << ".";
    }
    std::cout << std::endl;
  }
#endif // opencl

  // iterate over all problem sizes
  bool moreProblemSizes = true;
  unsigned int problemIdx = 0;
  do {

    // convert current sized and mapped indices to full sizes
    currentSizedIdx = 0;
    currentMappedIdx = 0;
    for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
      if (indexIsSized[i]) {
        fullSizes[i] = currentSizedIndexSizes[currentSizedIdx++];
      }
#if Tensile_INDICES_MAPPED
      else {
        fullSizes[i] = fullSizes[indicesMapped[currentMappedIdx++]];
      }
#endif
    }
    // print size
    std::cout << "Problem[" << problemIdx << "/" << numProblems << "]: " << fullSizes[0];
    for (unsigned int i = 1; i < totalIndices[problemTypeIdx]; i++) {
      std::cout << ", " << fullSizes[i];
    }
    std::cout << std::endl;

    // benchmark all solutions for this problem size
    bool invalids = benchmarkAllSolutionsForSize( problemIdx, fullSizes, initialC, initialA,
        initialB, alpha, beta, referenceC, deviceOnHostC);
    if (invalids) returnInvalids = true;

    // increment sizes for next benchmark
    currentSizedIndexSizes[0] += currentSizedIndexIncrements[0];
    currentSizedIndexIncrements[0] += indicesSized[0][2];
    for (unsigned int i = 1; i < numIndicesSized+1; i++) {
      // if prior index past max, reset to min and increment next index
      if (currentSizedIndexSizes[i-1] > indicesSized[i-1][3]) {
        // reset prior index
        currentSizedIndexSizes[i-1] = indicesSized[i-1][0];
        currentSizedIndexIncrements[i-1] = indicesSized[i-1][1];
        // increment next index
        if ( i >= numIndicesSized) {
          moreProblemSizes = false;
        } else {
          currentSizedIndexSizes[i] += currentSizedIndexIncrements[i];
          currentSizedIndexIncrements[i] += indicesSized[i][2];
        }
      }
    }
    problemIdx++;
  } while(moreProblemSizes);

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
  std::cout << "InitData(" << maxSizeC << ", " << maxSizeA << ", "
    << maxSizeB << ")" << std::endl;

  *alpha = tensileGetOne<DataType>();
  if (useBeta[problemTypeIdx]) {
    *beta = tensileGetOne<DataType>();
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
  if (dataInitTypeC == 0) {
    for (size_t i = 0; i < maxSizeC; i++) {
      (*initialC)[i] = tensileGetRandom<DataType>(); }
    std::cout << ".";
  } else if (dataInitTypeC == 1) {
    for (size_t i = 0; i < maxSizeC; i++) {
      (*initialC)[i] = tensileGetOne<DataType>(); }
    std::cout << ".";
  } else if (dataInitTypeC == 2) {
    for (size_t i = 0; i < maxSizeC; i++) {
      (*initialC)[i] = tensileGetTypeForInt<DataType>(i); }
    std::cout << ".";
  } else {
    for (size_t i = 0; i < maxSizeC; i++) {
      (*initialC)[i] = tensileGetZero<DataType>(); }
    std::cout << ".";
  }

  // initialize buffers
  if (dataInitTypeAB == 0) {
    for (size_t i = 0; i < maxSizeA; i++) {
      (*initialA)[i] = tensileGetRandom<DataType>(); }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      (*initialB)[i] = tensileGetRandom<DataType>(); }
    std::cout << ".";
  } else if (dataInitTypeAB == 1) {
    for (size_t i = 0; i < maxSizeA; i++) {
      (*initialA)[i] = tensileGetOne<DataType>(); }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      (*initialB)[i] = tensileGetOne<DataType>(); }
    std::cout << ".";
  } else if (dataInitTypeAB == 2) {
    for (size_t i = 0; i < maxSizeA; i++) {
      (*initialA)[i] = tensileGetTypeForInt<DataType>(i); }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      (*initialB)[i] = tensileGetTypeForInt<DataType>(i); }
    std::cout << ".";
  } else {
    for (size_t i = 0; i < maxSizeA; i++) {
      (*initialA)[i] = tensileGetZero<DataType>(); }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      (*initialB)[i] = tensileGetZero<DataType>(); }
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
#if Tensile_CLIENT_LIBRARY
unsigned int defaultNumElementsToValidate = 128;
void printLibraryClientUsage(std::string executableName) {
  std::cout << "Usage: " << executableName
    << " FunctionIdx SizeI SizeJ SizeK [SizeL ...] [NumElementsToValidate="
    << defaultNumElementsToValidate << "]" << std::endl;
  std::cout << "Functions:" << std::endl;
  for (unsigned int i = 0; i < numFunctions; i++) {
    std::cout << "  (" << i << ") " << functionNames[i] << std::endl;
  }
}
#endif

void parseCommandLineParameters( int argc, char *argv[] ) {
  std::string executableName = argv[0];

#if Tensile_CLIENT_BENCHMARK
  dataTypeIdx = 0;
  problemTypeIdx = 0;
#endif

#if Tensile_CLIENT_LIBRARY
  if (argc < 2) {
    std::cout << "FATAL ERROR: no FunctionIdx provided" << std::endl;
    printLibraryClientUsage(executableName);
    exit(0);
  }
  try {
    functionIdx = static_cast<unsigned int>(atoi(argv[1]));
    if (functionIdx >= numFunctions) {
      std::cout << "FATAL ERROR: FunctionIdx=" << functionIdx << " >= "
        << "NumFunctions=" << numFunctions << std::endl;
    }
    std::cout << "FunctionIdx: " << functionIdx << std::endl;
    dataTypeIdx = functionInfo[functionIdx][0];
    problemTypeIdx = functionInfo[functionIdx][2];
    if (static_cast<unsigned int>(argc - 2) < totalIndices[problemTypeIdx]) {
      std::cout << "FATAL ERROR: " << totalIndices[problemTypeIdx]
        << " sizes required for function[" << functionIdx << "]; only "
        << static_cast<unsigned int>(argc - 2) << " provided." << std::endl;
      printLibraryClientUsage(executableName);
      exit(0);
    }

    for (unsigned int i = 0; i < totalIndices[problemTypeIdx]; i++) {
      userSizes[i] = static_cast<unsigned int>(atoi(argv[2+i]));
      std::cout << "  Size" << indexChars[i] << ": " << userSizes[i]
        << std::endl;
    }
    if (static_cast<unsigned int>(argc) > 2+totalIndices[problemTypeIdx]) {
      numElementsToValidate = static_cast<unsigned int>(
          atoi(argv[2+totalIndices[problemTypeIdx]]));
      std::cout << "  NumElementsToValidate: " << numElementsToValidate
          << std::endl;
    } else {
      numElementsToValidate = defaultNumElementsToValidate;
      std::cout << "  NumElementsToValidate: " << numElementsToValidate
          << " (unspecified)" << std::endl;
    }

  } catch (...) {
    printLibraryClientUsage(executableName);
    exit(0);
  }

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

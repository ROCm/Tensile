#include "BenchmarkClient.h"
#include <string>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <iomanip>

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {
  if (sizeof(size_t) != 8) {
    std::cout << "WARNING: Executable not 64-bit." << std::endl;
  }
  initControls();
  initData();
  std::cout << std::endl;
  std::cout << "Solutions: " << std::endl;
  for (unsigned int sIdx = 0; sIdx < numSolutions; sIdx++) {
    std::cout << "  " << solutionNames[sIdx] << std::endl;
  }

  std::cout << "ResultsFileName: " << resultsFileName << std::endl;
  file.open(resultsFileName);
  // write column headers
  file << "Milliseconds ";
  char indexChars[19] = "IJKLMNOPQRSTUVWXYZ";
  for ( unsigned int i = 0; i < totalIndices; i++) {
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
#if Tensile_BACKEND_OCL
  if (!numElementsToValidate) {
    std::cout << "Pre-compiling " << numSolutions << " OpenCL kernels";
    for (unsigned int i = 0; i < totalIndices; i++) {
      if (indexIsSized[i]) {
        fullSizes[i] = currentSizedIndexSizes[currentSizedIdx++];
      } else {
        fullSizes[i] = fullSizes[indicesMapped[currentMappedIdx++]];
      }
    }
    for (unsigned int sIdx = 0; sIdx < numSolutions; sIdx++) {
      generatedCallToSolution( sIdx, fullSizes );
#if Tensile_BACKEND_OCL
      status = clFinish(stream); tensileStatusCheck(status);
#else
      status = hipStreamSynchronize(stream); tensileStatusCheck(status);
#endif
      tensileStatusCheck(status);
      std::cout << ".";
    }
    std::cout << std::endl;
  }
#endif

  // iterate over all problem sizes
  bool moreProblemSizes = true;
  unsigned int problemIdx = 0;
  do {

    // convert current sized and mapped indices to full sizes
    currentSizedIdx = 0;
    currentMappedIdx = 0;
    for (unsigned int i = 0; i < totalIndices; i++) {
      if (indexIsSized[i]) {
        fullSizes[i] = currentSizedIndexSizes[currentSizedIdx++];
      } else {
        fullSizes[i] = fullSizes[indicesMapped[currentMappedIdx++]];
      }
    }
#if 1
    // print size
    std::cout << "Problem[" << problemIdx << "/" << numProblems << "]: " << fullSizes[0];
    for (unsigned int i = 1; i < totalIndices; i++) {
      std::cout << ", " << fullSizes[i];
    }
    std::cout << std::endl;
#endif

    // benchmark all solutions for this problem size
    benchmarkAllSolutionsForSize( problemIdx, fullSizes );

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

  // cleanup
  destroyControls();
  destroyData();

  return 0;
} // main


/*******************************************************************************
 * benchmark all solutions for problem size
 ******************************************************************************/
void benchmarkAllSolutionsForSize(
    unsigned int problemIdx, unsigned int *sizes ) {

  size_t currentSizeC = 1;
  for (unsigned int i = 0; i < numIndicesC; i++) {
    currentSizeC *= sizes[i];
  }

  file << problemIdx << ", " << sizes[0];
  for (unsigned int i = 1; i < totalIndices; i++) {
    file << ", " << sizes[i];
  }
  size_t totalFlops = numFlopsPerMac;
  for (unsigned int i = 0; i < totalIndices; i++) { totalFlops *= sizes[i]; }
  file << ", " << totalFlops;

  // pre-compute referenceCPU if validating
  if (numElementsToValidate) {
    memcpy(referenceC, initialC, static_cast<size_t>(currentSizeC*sizeof(DataType)));
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
    generatedCallToReferenceCPU( sizes );
  }
  for (unsigned int solutionIdx = 0; solutionIdx < numSolutions; solutionIdx ++) {

    // copy data in language
#if Tensile_BACKEND_OCL
    status = clEnqueueWriteBuffer(stream, deviceC, CL_TRUE, 0,
        currentSizeC*sizeof(DataType), initialC, 0, NULL, NULL);
#else
    status = hipMemcpy(deviceC, initialC, currentSizeC*sizeof(DataType), hipMemcpyHostToDevice);
#endif
    tensileStatusCheck(status);

    // validate solution
    size_t numInvalids = 0;
    size_t numChecked = 0;
    if (numElementsToValidate) {

      // enqueue device solution
      generatedCallToSolution( solutionIdx , sizes );

      // copy data back to host
#if Tensile_BACKEND_OCL
      clEnqueueReadBuffer(stream, deviceC, CL_TRUE, 0,
          currentSizeC*sizeof(DataType), deviceOnHostC, 0, NULL, NULL);
#else
      hipMemcpy(deviceOnHostC, deviceC, currentSizeC*sizeof(DataType), hipMemcpyDeviceToHost);
#endif

      // compare
      //unsigned int maxPrint = 16;
      //bool printTrue = false;
      bool firstPrint = true;
      unsigned int printIdx = 0;
      for (size_t i = 0; i < currentSizeC; i+= validationStride) {
        bool equal = tensileAlmostEqual<DataType>(
            deviceOnHostC[i], referenceC[i]);
        numChecked++;
        if (!equal) numInvalids++;

        if (!equal || validationPrintValids) {
          if (printIdx < validationMaxToPrint) {
            if (firstPrint) {
              std::cout << "  Device | Reference" << std::endl;
              firstPrint = false;
            }
            std::cout << i << ": " << deviceOnHostC[i] <<
                (equal ? "==" : "!=") << referenceC[i] << std::endl;
            printIdx++;
          } else {
            break;
          }
        }
      } // compare loop
    } // if numElementsToValidate > 0

    // time solution
    timer.start();
    for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
      for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
        generatedCallToSolution( solutionIdx , sizes );
      }
      // sync
#if Tensile_BACKEND_OCL
      status = clFinish(stream); tensileStatusCheck(status);
#else
      status = hipStreamSynchronize(stream); tensileStatusCheck(status);
#endif
      tensileStatusCheck(status);
    } // sync loop
    double timeMs = timer.elapsed_ms()
      / numSyncsPerBenchmark / numEnqueuesPerSync;
    double gflops = totalFlops / timeMs / 1000000.0;

    if (numElementsToValidate) {
      std::cout << "  Solution[" << std::setw(2) << solutionIdx << "/" << numSolutions << "]:"
        << std::setw(10) << std::fixed << std::setprecision(3)
        << gflops << " GFlop/s |"
        << std::setw(9) << std::fixed << std::setprecision(3) << timeMs << " ms | v: " << (numInvalids ? "FAILED" : "PASSED")
        << " p: " << (numChecked-numInvalids) << "/" << numChecked << std::endl;
    } else {
      std::cout << "  Solution[" << solutionIdx << "/" << numSolutions << "]:"
        << std::setw(10) << std::fixed << std::setprecision(3)
        << gflops << " GFlop/s |"
        << std::setw(9) << std::fixed << std::setprecision(3) << timeMs << " ms" << std::endl;
    }
    file << ", " << gflops;
    solutionPerf[problemIdx][solutionIdx ] = static_cast<float>(gflops);
  } // solution loop
  file << std::endl;
} // benchmark solutions


/*******************************************************************************
 * init controls
 ******************************************************************************/
void initControls() {
#if Tensile_BACKEND_OCL
  // setup opencl objects
  cl_uint numPlatforms, numDevices;
  status = clGetPlatformIDs(0, nullptr, &numPlatforms);
  tensileStatusCheck(status);
  cl_platform_id *platforms = new cl_platform_id[numPlatforms];
  status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
  tensileStatusCheck(status);
  platform = platforms[platformIdx];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  tensileStatusCheck(status);
  cl_device_id *devices = new cl_device_id[numDevices];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
  tensileStatusCheck(status);
  device = devices[deviceIdx];
  size_t nameLength;
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, 0, nullptr, &nameLength );
  tensileStatusCheck(status);
  char *deviceName = new char[nameLength+1];
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, nameLength, deviceName, 0 );
  tensileStatusCheck(status);
  std::cout << "Device: \"" << deviceName << "\"" << std::endl;
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
  tensileStatusCheck(status);
  stream = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  tensileStatusCheck(status);
  delete[] devices;
  delete[] platforms;
#elif Tensile_BACKEND_HIP
  int numDevices;
  status = hipGetDeviceCount( &numDevices );
  tensileStatusCheck(status);
  status = hipSetDevice( deviceIdx );
  tensileStatusCheck(status);
  status = hipStreamCreate( &stream );
  tensileStatusCheck(status);
#endif

}


/*******************************************************************************
 * destroy controls
 ******************************************************************************/
void destroyControls() {
#if Tensile_BACKEND_OCL
  clReleaseCommandQueue(stream);
  clReleaseContext(context);
  clReleaseDevice(device);
#else
  hipStreamDestroy(stream);
#endif
}


/*******************************************************************************
 * initialize data
 ******************************************************************************/
void initData() {
  std::cout << "Initializing " << (sizeof(DataType)*(maxSizeC+maxSizeA+maxSizeB)/1000000) << " MBytes";
  std::cout << ".";

  // initial and reference buffers
  referenceC = new DataType[maxSizeC];
  deviceOnHostC = new DataType[maxSizeC];
  initialC = new DataType[maxSizeC];
  std::cout << ".";
  initialA = new DataType[maxSizeA];
  std::cout << ".";
  initialB = new DataType[maxSizeB];
  std::cout << ".";

  // initialize buffers
  if (dataInitType == 0) {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = static_cast<DataType>(rand() % 10);
    }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = static_cast<DataType>(rand() % 10);
    }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = static_cast<DataType>(rand() % 10);
    }
    std::cout << ".";
  } else if (dataInitType == 1) {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = tensileGetOne<DataType>();
    }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = tensileGetOne<DataType>();
    }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = tensileGetOne<DataType>();
    }
    std::cout << ".";
  } else {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = static_cast<DataType>(i);
    }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = static_cast<DataType>(i);
    }
    std::cout << ".";
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = static_cast<DataType>(i);
    }
    std::cout << ".";
  }

  // create device buffers and copy data
#if Tensile_BACKEND_OCL
  deviceC = clCreateBuffer(context, CL_MEM_READ_WRITE,
      maxSizeC*sizeof(DataType), NULL, &status);
  tensileStatusCheck(status);
    std::cout << ".";
  deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY,
      maxSizeA*sizeof(DataType), NULL, &status);
  tensileStatusCheck(status);
    std::cout << ".";
  deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY,
      maxSizeB*sizeof(DataType), NULL, &status);
  tensileStatusCheck(status);
    std::cout << ".";
  status = clEnqueueWriteBuffer(stream, deviceA, CL_TRUE, 0,
      maxSizeA*sizeof(DataType), initialA, 0, NULL, NULL);
  tensileStatusCheck(status);
    std::cout << ".";
  status = clEnqueueWriteBuffer(stream, deviceB, CL_TRUE, 0,
      maxSizeB*sizeof(DataType), initialB, 0, NULL, NULL);
  tensileStatusCheck(status);
    std::cout << ".";
#else
  status = hipMalloc( &deviceC, maxSizeC*sizeof(DataType) );
  tensileStatusCheck(status);
  std::cout << ".";
  status = hipMalloc( &deviceA, maxSizeA*sizeof(DataType) );
  tensileStatusCheck(status);
  std::cout << ".";
  status = hipMalloc( &deviceB, maxSizeB*sizeof(DataType) );
  tensileStatusCheck(status);
  std::cout << ".";
  status = hipMemcpy(deviceA, initialA, maxSizeC*sizeof(DataType), hipMemcpyHostToDevice);
  status = hipMemcpy(deviceB, initialB, maxSizeB*sizeof(DataType), hipMemcpyHostToDevice);
#endif
}


/*******************************************************************************
 * destroy data
 ******************************************************************************/
void destroyData() {
  delete[] initialC;
  delete[] initialA;
  delete[] initialB;
  delete[] deviceOnHostC;
  delete[] referenceC;

#if Tensile_BACKEND_OCL
  clReleaseMemObject(deviceC);
  clReleaseMemObject(deviceA);
  clReleaseMemObject(deviceB);
#else
  hipFree(deviceC);
#endif

}

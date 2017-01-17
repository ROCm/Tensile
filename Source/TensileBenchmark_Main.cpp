#include "TensileBenchmark_Main.h"
#include <string>
#include <cstring>
#include <cstdio>

// {rand, 1, index}
unsigned int dataInitType = 0;

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {

  initControls();

  initData();
  printf("\n");

  printf("ResultsFileName: %s\n", resultsFileName);
  file.open(resultsFileName);
  // write column headers
  file << "Idx ";
  char *indexChars = "IJKLMNOPQRSTUVWXYZ";
  for ( unsigned int i = 0; i < totalIndices; i++) {
    file << ", Size" << indexChars[i];
  }
  file << ", TotalSize";
  for ( unsigned int s = 0; s < numSolutions; s++) {
    file << ", " << solutionNames[s];
  }
  file << std::endl;

  // initialize index sizes
  unsigned int *fullSizes = new unsigned int[totalIndices];
  unsigned int *currentSizedIndexSizes = new unsigned int[numIndicesSized];
  unsigned int *currentSizedIndexIncrements = new unsigned int[numIndicesSized];
  for ( unsigned int i = 0; i < numIndicesSized; i++) {
    currentSizedIndexSizes[i] = indicesSized[i][0];
    currentSizedIndexIncrements[i] = indicesSized[i][1];
  }

  // iterate over all problem sizes
  bool moreProblemSizes = true;
  unsigned int problemIdx = 0;
  do {

    // convert current sized and mapped indices to full sizes
    unsigned int currentSizedIdx = 0;
    unsigned int currentMappedIdx = 0;
    for (unsigned int i = 0; i < totalIndices; i++) {
      if (indexIsSized[i]) {
        fullSizes[i] = currentSizedIndexSizes[currentSizedIdx++];
      } else {
        fullSizes[i] = fullSizes[indicesMapped[currentMappedIdx++]];
      }
    }
#if 1
    // print size
    printf("Problem[%2u]: %u", problemIdx, fullSizes[0]);
    for (unsigned int i = 1; i < totalIndices; i++) {
      printf(", %u", fullSizes[1]);
    }
    printf("\n");
#endif

    // benchmark all solutions for this problem size
    benchmarkAllSolutionsForSize( problemIdx, fullSizes );

    // increment sizes for next benchmark
    currentSizedIndexSizes[0] += currentSizedIndexIncrements[0];
    currentSizedIndexIncrements[0] += indicesSized[0][2];
    for (unsigned int i = 1; i < numIndicesSized+1; i++) {
      // if prior index past max, reset to min and increment next index
      if (currentSizedIndexSizes[i-1] >= indicesSized[i-1][3]) {
        // reset prior index
        currentSizedIndexSizes[i-1] = indicesSized[i-1][0];
        currentSizedIndexIncrements[i-1] += indicesSized[i-1][1];
        // increment next index
        if ( i >= numIndicesSized) {
          moreProblemSizes = false;
        } else {
          currentSizedIndexSizes[i] += currentSizedIndexIncrements[i];
          currentSizedIndexIncrements[i] += indicesSized[i][2];
        }
      }
      problemIdx++;
    }
  } while(moreProblemSizes);

  // close file
  file.close();

  // cleanup
  delete[] currentSizedIndexSizes;
  delete[] currentSizedIndexIncrements;
  delete[] fullSizes;
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
  size_t totalSize = numFlopsPerMac;
  for (unsigned int i = 0; i < totalIndices; i++) { totalSize *= sizes[i]; }
  file << ", " << totalSize;

  // pre-compute referenceCPU if validating
  if (doValidation) {
    memcpy(referenceC, initialC, currentSizeC*sizeof(DataType));
    generatedCallToReferenceCPU( sizes );
  }
  for (unsigned int s = 0; s < numSolutions; s++) {

    // validate solution
    if (doValidation) {
      // copy data in language
#if Tensile_BACKEND_OCL
      status = clEnqueueWriteBuffer(stream, deviceC, CL_TRUE, 0,
          currentSizeC*sizeof(DataType), initialC, 0, NULL, NULL);
#else
      status = hipMemcpy(deviceC, initialC, currentSizeC*sizeof(DataType));
#endif
      tensileStatusCheck(status);
      // enqueue device solution
      generatedCallToSolution( s, sizes );

      // copy data back to host
#if Tensile_BACKEND_OCL
      clEnqueueReadBuffer(stream, deviceC, CL_TRUE, 0,
          currentSizeC*sizeof(DataType), deviceOnHostC, 0, NULL, NULL);
#else
      hipMemcpy(deviceOnHostC, deviceC, currentSizeC*sizeof(DataType));
#endif

      // compare
      unsigned int maxPrint = 16;
      bool printTrue = false;
      bool firstPrint = true;
      unsigned int printIdx = 0;
      for (size_t i = 0; i < currentSizeC; i++) {
        bool equal = tensileAlmostEqual<DataType>(deviceOnHostC[i], referenceC[i]);
        if (!equal || printTrue) {
          if (printIdx < maxPrint) {
            if (firstPrint) {
              printf("Device | Reference\n");
              firstPrint = false;
            }
            printf("%f %s %f\n", deviceOnHostC[i],
                equal ? "==" : "!=", referenceC[i]);
            printIdx++;
          } // else don't print too many
        }
      } // compare loop
    } else {
      // dummy call to ensure kernels compiled
      generatedCallToSolution( s, sizes );
    }

    // re-initialize deviceC
#if Tensile_BACKEND_OCL
    status = clEnqueueWriteBuffer(stream, deviceC, CL_TRUE, 0,
        currentSizeC*sizeof(DataType), initialC, 0, NULL, NULL);
#else
    status = hipMemcpy(deviceC, initialC, currentSizeC*sizeof(DataType));
#endif
    tensileStatusCheck(status);
    
    // time solution
    timer.start();
    for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
      for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
        generatedCallToSolution( s, sizes );
      }
      // sync
#if Tensile_BACKEND_OCL
      status = clFinish(stream); tensileStatusCheck(status);
#else
      status = hipSync(stream); tensileStatusCheck(status);
#endif
      tensileStatusCheck(status);
    } // sync loop
    double timeMs = timer.elapsed_ms()
      / numSyncsPerBenchmark / numEnqueuesPerSync;
    printf("Time[%u]: %f ms\n", s, timeMs);
    file << ", " << timeMs;
    solutionTimes[problemIdx][s] = static_cast<float>(timeMs);
  } // solution loop
  file << std::endl;
} // benchmark solutions


/*******************************************************************************
 * init controls
 ******************************************************************************/
void initControls() {
#if Tensile_BACKEND_OCL
  // setup opencl objects
  unsigned int numPlatforms, numDevices;
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
  printf("Device: \"%s\"\n", deviceName);
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
  tensileStatusCheck(status);
  stream = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  tensileStatusCheck(status);
#elif Tensile_BACKEND_HIP
  status = hipGetDeviceCount( &numDevices );
  tensileStatusCheck(status);
  status = hipSetDevice( deviceIdx );
  tensileStatusCheck(status);
  status = hipStreamCreate( &stream );
  tensileStatusCheck(status);
#endif

  delete[] devices;
  delete[] platforms;
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
  printf("Initializing buffers (%u + %u + %u elements)",
      maxSizeC,
      maxSizeA,
      maxSizeB
      //maxSizeC*sizeof(DataType)/1024.0/1024.0,
      //maxSizeA*sizeof(DataType)/1024.0/1024.0,
      //maxSizeB*sizeof(DataType)/1024.0.1024.0
      );
    printf(".");
  // initial and reference buffers
  referenceC = new DataType[maxSizeC];
#if Tensile_BACKEND_OCL
  initialC = new DataType[maxSizeC];
  initialA = new DataType[maxSizeA];
  initialB = new DataType[maxSizeB];
#else
  status = hipMalloc( &initialC, sizeMaxC );
  tensileStatusCheck(status);
  status = hipMalloc( &initialA, sizeMaxA );
  tensileStatusCheck(status);
  status = hipMalloc( &initialB, sizeMaxB );
  tensileStatusCheck(status);
#endif
    printf(".");

  // initialize buffers
  if (dataInitType == 0) {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = static_cast<DataType>(rand() % 10);
    }
    printf(".");
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = static_cast<DataType>(rand() % 10);
    }
    printf(".");
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = static_cast<DataType>(rand() % 10);
    }
    printf(".");
  } else if (dataInitType == 0) {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = tensileGetOne<DataType>();
    }
    printf(".");
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = tensileGetOne<DataType>();
    }
    printf(".");
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = tensileGetOne<DataType>();
    }
    printf(".");
  } else {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = static_cast<DataType>(i);
    }
    printf(".");
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = static_cast<DataType>(i);
    }
    printf(".");
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = static_cast<DataType>(i);
    }
    printf(".");
  }
#if Tensile_BACKEND_OCL
  deviceC = clCreateBuffer(context, CL_MEM_READ_WRITE,
      maxSizeC*sizeof(DataType), NULL, &status);
  tensileStatusCheck(status);
  deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY,
      maxSizeA*sizeof(DataType), NULL, &status);
  tensileStatusCheck(status);
  deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY,
      maxSizeB*sizeof(DataType), NULL, &status);
  tensileStatusCheck(status);
  status = clEnqueueWriteBuffer(stream, deviceA, CL_TRUE, 0,
      maxSizeA*sizeof(DataType), initialA, 0, NULL, NULL);
  tensileStatusCheck(status);
  status = clEnqueueWriteBuffer(stream, deviceB, CL_TRUE, 0,
      maxSizeB*sizeof(DataType), initialB, 0, NULL, NULL);
  tensileStatusCheck(status);
#else
#endif
}


/*******************************************************************************
 * destroy data
 ******************************************************************************/
void destroyData() {
  delete[] initialC;
  delete[] initialA;
  delete[] initialB;
  delete[] referenceC;

#if Tensile_BACKEND_OCL
  clReleaseMemObject(deviceC);
  clReleaseMemObject(deviceA);
  clReleaseMemObject(deviceB);
#else
  hipFree(deviceC);
#endif

}

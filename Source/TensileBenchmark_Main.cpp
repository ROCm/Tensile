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

  printf("Writing results to %s\n", resultsFileName);
  file.open(resultsFileName);
  // write column headers
  file << "Idx, TotalSize";
  char *indexChars = "IJKLMNOPQRSTUVWXYZ";
  for ( unsigned int i = 0; i < totalIndices; i++) {
    file << ", Size" << indexChars[i];
  }
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
    printf("size={ %u", fullSizes[0]);
    for (unsigned int i = 1; i < totalIndices; i++) {
      printf(", %u", fullSizes[1]);
    }
    printf("}\n");
#endif

    // benchmark all solutions for this problem size
    for (unsigned int s = 0; s < numSolutions; s++) {
      benchmarkAllSolutionsForSize( problemIdx, fullSizes );
    }

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
        if ( i == numIndicesSized) {
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

  size_t totalSize = 1;
  for (unsigned int i = 0; i < totalIndices; i++) {
    totalSize *= sizes[i];
  }
  file << problemIdx << ", " << totalSize << ", " << sizes[0];
  for (unsigned int i = 1; i < totalIndices; i++) {
    file << ", " << sizes[i];
  }

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
      clEnqueueWriteBuffer(stream, deviceC, CL_TRUE, 0,
          currentSizeC*sizeof(DataType), initialC, 0, NULL, NULL);
#else
      hipMemcpy(deviceC, initialC, currentSizeC*sizeof(DataType));
#endif
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
    } // do validation
    
    // time solution
    timer.start();
    for (unsigned int syncIdx = 0; syncIdx < numSyncsPerBenchmark; syncIdx++) {
      for (unsigned int enqIdx = 0; enqIdx < numEnqueuesPerSync; enqIdx++) {
        generatedCallToSolution( s, sizes );
      }
      // sync
#if Tensile_BACKEND_OCL
      status = clFinish(stream);
#else
      status = hipSync(stream);
#endif
    } // sync loop
    double timeUs = timer.elapsed_us()
      / numSyncsPerBenchmark / numEnqueuesPerSync;
    //printf("Time[%u]: %f us\n", s, timeUs);
    file << ", " << timeUs;
    solutionTimes[problemIdx][s] = static_cast<float>(timeUs);
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
  cl_platform_id *platforms = new cl_platform_id[numPlatforms];
  status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
  platform = platforms[platformIdx];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  cl_device_id *devices = new cl_device_id[numDevices];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
  device = devices[deviceIdx];
  size_t nameLength;
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, 0, nullptr, &nameLength );
  char *deviceName = new char[nameLength+1];
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, nameLength, deviceName, 0 );
  printf("Device: \"%s\"\n", deviceName);
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
  stream = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
#elif Tensile_BACKEND_HIP
  status = hipGetDeviceCount( &numDevices );
  status = hipSetDevice( deviceIdx );
  status = hipStreamCreate( &stream );
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
  // initial and reference buffers
  referenceC = new DataType[maxSizeC];
#if Tensile_BACKEND_OCL
  initialC = new DataType[maxSizeC];
  initialA = new DataType[maxSizeA];
  initialB = new DataType[maxSizeB];
#else
  status = hipMalloc( &initialC, sizeMaxC );
  status = hipMalloc( &initialA, sizeMaxA );
  status = hipMalloc( &initialB, sizeMaxB );
#endif

  // initialize buffers
  if (dataInitType == 0) {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = static_cast<DataType>(rand() % 10);
    }
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = static_cast<DataType>(rand() % 10);
    }
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = static_cast<DataType>(rand() % 10);
    }
  } else if (dataInitType == 0) {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = tensileGetOne<DataType>();
    }
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = tensileGetOne<DataType>();
    }
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = tensileGetOne<DataType>();
    }
  } else {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = static_cast<DataType>(i);
    }
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = static_cast<DataType>(i);
    }
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = static_cast<DataType>(i);
    }
  }
#if Tensile_BACKEND_OCL
  deviceC = clCreateBuffer(context, CL_MEM_READ_WRITE, maxSizeC, NULL, &status);
  deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY, maxSizeA, NULL, &status);
  deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY, maxSizeB, NULL, &status);
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

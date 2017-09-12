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
#include "Client.h"
#include "DeviceStats.h"
#include <string>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <iomanip>

const unsigned int numLibraryCalls = 1024;

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {
  if (sizeof(size_t) != 8) {
    std::cout << "WARNING: Executable not 64-bit." << std::endl;
  }

  // parse command line parameters
  parseCommandLineParameters(argc, argv);

  // init runtime controls
  initControls();

  // init data
  unsigned int dataTypeIdx = 0;
  DataTypeEnum dataTypeEnum = dataTypeEnums[dataTypeIdx];
  bool invalids;
  std::cout << "Tensile Client Columns: GFlops, SolName, KernelMs, "
#if Tensile_CLIENT_LIBRARY
    << "ApiUs, "
#endif
    << "Valid/Total, CoreMhz, MemMhz, TempC, FanSpeed, Idx/Total" << std::endl;
  switch(dataTypeEnum) {
#ifdef Tensile_DATA_TYPE_FLOAT
  case enum_float: {
    float *initialC_float;
    float *initialA_float;
    float *initialB_float;
    float alpha_float;
    float beta_float;
    float *referenceC_float;
    float *deviceOnHostC_float;
    initData(&initialC_float, &initialA_float, &initialB_float, &alpha_float,
        &beta_float, &referenceC_float, &deviceOnHostC_float);
#if Tensile_CLIENT_BENCHMARK
    invalids = benchmarkProblemSizes(initialC_float, initialA_float,
        initialB_float, alpha_float, beta_float, referenceC_float,
        deviceOnHostC_float);
#else
    for (unsigned int i = 0; i < numLibraryCalls; i++) {
      invalids = callLibrary(initialC_float, initialA_float, initialB_float,
          alpha_float, beta_float, referenceC_float, deviceOnHostC_float);
    }
#endif
    destroyData(initialC_float, initialA_float, initialB_float,
        referenceC_float, deviceOnHostC_float);
    }
    break;
#endif
#ifdef Tensile_DATA_TYPE_DOUBLE
  case enum_double: {
    double *initialC_double;
    double *initialA_double;
    double *initialB_double;
    double alpha_double;
    double beta_double;
    double *referenceC_double;
    double *deviceOnHostC_double;
    initData(&initialC_double, &initialA_double, &initialB_double,
        &alpha_double, &beta_double, &referenceC_double, &deviceOnHostC_double);
#if Tensile_CLIENT_BENCHMARK
    invalids = benchmarkProblemSizes(initialC_double, initialA_double,
        initialB_double, alpha_double, beta_double, referenceC_double,
        deviceOnHostC_double);
#else
    for (unsigned int i = 0; i < numLibraryCalls; i++) {
      invalids = callLibrary(initialC_double, initialA_double, initialB_double,
          alpha_double, beta_double, referenceC_double, deviceOnHostC_double);
    }
#endif
    destroyData(initialC_double, initialA_double, initialB_double,
        referenceC_double, deviceOnHostC_double);
    }
    break;
#endif
#ifdef Tensile_DATA_TYPE_TENSILECOMPLEXFLOAT
  case enum_TensileComplexFloat: {
    TensileComplexFloat *initialC_TCF;
    TensileComplexFloat *initialA_TCF;
    TensileComplexFloat *initialB_TCF;
    TensileComplexFloat alpha_TCF;
    TensileComplexFloat beta_TCF;
    TensileComplexFloat *referenceC_TCF;
    TensileComplexFloat *deviceOnHostC_TCF;
    initData(&initialC_TCF, &initialA_TCF, &initialB_TCF, &alpha_TCF,
        &beta_TCF, &referenceC_TCF, &deviceOnHostC_TCF);
#if Tensile_CLIENT_BENCHMARK
    invalids = benchmarkProblemSizes(initialC_TCF, initialA_TCF, initialB_TCF,
        alpha_TCF, beta_TCF, referenceC_TCF, deviceOnHostC_TCF);
#else
    for (unsigned int i = 0; i < numLibraryCalls; i++) {
      invalids = callLibrary(initialC_TCF, initialA_TCF, initialB_TCF,
          alpha_TCF, beta_TCF, referenceC_TCF, deviceOnHostC_TCF);
    }
#endif
    destroyData(initialC_TCF, initialA_TCF, initialB_TCF, referenceC_TCF,
        deviceOnHostC_TCF);
    }
    break;
#endif
#ifdef Tensile_DATA_TYPE_TENSILECOMPLEXDOUBLE
  case enum_TensileComplexDouble: {
    TensileComplexDouble *initialC_TCD;
    TensileComplexDouble *initialA_TCD;
    TensileComplexDouble *initialB_TCD;
    TensileComplexDouble alpha_TCD;
    TensileComplexDouble beta_TCD;
    TensileComplexDouble *referenceC_TCD;
    TensileComplexDouble *deviceOnHostC_TCD;
    initData(&initialC_TCD, &initialA_TCD, &initialB_TCD, &alpha_TCD,
        &beta_TCD, &referenceC_TCD, &deviceOnHostC_TCD);
#if Tensile_CLIENT_BENCHMARK
    invalids = benchmarkProblemSizes(initialC_TCD, initialA_TCD, initialB_TCD,
        alpha_TCD, beta_TCD, referenceC_TCD, deviceOnHostC_TCD);
#else
    for (unsigned int i = 0; i < numLibraryCalls; i++) {
      invalids = callLibrary(initialC_TCD, initialA_TCD, initialB_TCD,
          alpha_TCD, beta_TCD, referenceC_TCD, deviceOnHostC_TCD);
    }
#endif
    destroyData(initialC_TCD, initialA_TCD, initialB_TCD, referenceC_TCD,
        deviceOnHostC_TCD);
    }
    break;
#endif
#ifdef Tensile_DATA_TYPE_TENSILEHALF
  case enum_TensileHalf: {
    TensileHalf *initialC_TH;
    TensileHalf *initialA_TH;
    TensileHalf *initialB_TH;
    TensileHalf alpha_TH;
    TensileHalf beta_TH;
    TensileHalf *referenceC_TH;
    TensileHalf *deviceOnHostC_TH;
    initData(&initialC_TH, &initialA_TH, &initialB_TH, &alpha_TH,
        &beta_TH, &referenceC_TH, &deviceOnHostC_TH);
#if Tensile_CLIENT_BENCHMARK
    invalids = benchmarkProblemSizes(initialC_TH, initialA_TH, initialB_TH,
        alpha_TH, beta_TH, referenceC_TH, deviceOnHostC_TH);
#else
    for (unsigned int i = 0; i < numLibraryCalls; i++) {
      invalids = callLibrary(initialC_TH, initialA_TH, initialB_TH,
          alpha_TH, beta_TH, referenceC_TH, deviceOnHostC_TH);
    }
#endif
    destroyData(initialC_TH, initialA_TH, initialB_TH, referenceC_TH,
        deviceOnHostC_TH);
    }
    break;
#endif
  default:
    break;
    // nothing

  }

  // cleanup
  destroyControls();
  std::cout << std::endl << "Fastest: " << globalFastestGFlops << " GFlop/s by ("
      << globalFastestIdx << ") ";
#if Tensile_CLIENT_BENCHMARK
  std::cout << solutionNames[fastestIdx];
#else
  std::cout << functionNames[fastestIdx];
#endif
  std::cout << std::endl;
  if (invalids) {
#if Tensile_CLIENT_BENCHMARK
    printf("\nInvalid Solutions: %u/%u\n", numInvalidSolutions, numSolutions);
    for (unsigned int i = 0; i < numInvalidSolutions; i++) {
      printf("[%2u] %s\n", invalidSolutions[i],
          solutionNames[invalidSolutions[i]]);
    }
#endif
    return EXIT_FAILURE;
  } else {
    return EXIT_SUCCESS;
  }
} // main



/*******************************************************************************
 * init controls
 ******************************************************************************/
void initControls() {
#if Tensile_RUNTIME_LANGUAGE_OCL
  // setup opencl objects
  cl_uint numPlatforms, numDevices;
  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (platformIdx >= numPlatforms) {
    std::cout << "Platform " << platformIdx << "/" << numPlatforms << " invalid"
        << std::endl;
    exit(1);
  }
  tensileStatusCheck(status);
  cl_platform_id *platforms = new cl_platform_id[numPlatforms];
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);
  tensileStatusCheck(status);
  platform = platforms[platformIdx];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  if (deviceIdx >= numDevices) {
    std::cout << "Device " << deviceIdx << "/" << numDevices << " invalid"
        << std::endl;
    exit(1);
  }
  tensileStatusCheck(status);
  cl_device_id *devices = new cl_device_id[numDevices];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices,
      NULL);
  tensileStatusCheck(status);
  device = devices[deviceIdx];
  size_t nameLength;
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, 0, NULL, &nameLength );
  tensileStatusCheck(status);
  char *deviceName = new char[nameLength+1];
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, nameLength, deviceName, 0 );
  tensileStatusCheck(status);
  std::cout << "Device: \"" << deviceName << "\"" << std::endl;
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  tensileStatusCheck(status);

  if (measureKernelTime) {
      stream = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE,
          &status);
  } else {
      stream = clCreateCommandQueue(context, device, 0x0, &status);
  }
  tensileStatusCheck(status);

  delete[] devices;
  delete[] platforms;
#elif Tensile_RUNTIME_LANGUAGE_HIP
  int numDevices;
  status = hipGetDeviceCount( &numDevices );
  if (deviceIdx >= static_cast<unsigned int>(numDevices)) {
    std::cout << "Device " << deviceIdx << "/" << numDevices << " invalid"
        << std::endl;
    exit(1);
  }
  tensileStatusCheck(status);
  status = hipSetDevice( deviceIdx );
  tensileStatusCheck(status);
  status = hipStreamCreate( &stream );
  tensileStatusCheck(status);

  // prepare to report device stats
  tensileInitDeviceStats();
#endif

}


/*******************************************************************************
 * destroy controls
 ******************************************************************************/
void destroyControls() {
#if Tensile_RUNTIME_LANGUAGE_OCL
  clReleaseCommandQueue(stream);
  clReleaseContext(context);
  clReleaseDevice(device);
#else
  hipStreamDestroy(stream);
#endif
}

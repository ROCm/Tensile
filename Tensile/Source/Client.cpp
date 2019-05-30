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
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

/*******************************************************************************
 * main
 ******************************************************************************/
int main(int argc, char* argv[])
{
    if(sizeof(size_t) != 8)
    {
        std::cout << "WARNING: Executable not 64-bit." << std::endl;
    }

#if Tensile_CLIENT_BENCHMARK
    solutionLocks = new SolutionLock[maxNumSolutions];
#else
    tensileInitialize();
#endif

    // parse command line parameters
    parseCommandLineParameters(argc, argv);

    // init runtime controls
    initControls();

    // init data
    unsigned int dataTypeIdx  = 0;
    DataTypeEnum dataTypeEnum = dataTypeEnums[dataTypeIdx];
    bool         invalids;
    if(dataTypeEnum == enum_TensileInt8x4)
    {
        std::cout
            << "Tensile Client Columns: GOps (clock-normalized), GOps (raw), SolName, KernelMs, ";
    }
    else
    {
        std::cout << "Tensile Client Columns: GFlops (clock-normalized), GFlops (raw), SolName, "
                     "KernelMs, ";
    }
#if Tensile_CLIENT_LIBRARY
    std::cout << "ApiUs, ";
#endif
    std::cout << "Valid/Total, CoreMhz, MemMhz, TempC, FanSpeed, Idx/Total, TimeStamp" << std::endl;

#if Tensile_CLIENT_BENCHMARK
#define TENSILE_CLIENT_CALL_PROBLEM                 \
    invalids = benchmarkProblemSizes(initialD,      \
                                     initialC,      \
                                     initialA,      \
                                     initialB,      \
                                     alpha,         \
                                     beta,          \
                                     referenceD,    \
                                     referenceC,    \
                                     deviceOnHostD, \
                                     deviceOnHostC);
#else
#define TENSILE_CLIENT_CALL_PROBLEM       \
    invalids = callLibrary(initialD,      \
                           initialC,      \
                           initialA,      \
                           initialB,      \
                           alpha,         \
                           beta,          \
                           lda,           \
                           ldb,           \
                           ldc,           \
                           ldd,           \
                           strideA,       \
                           strideB,       \
                           strideC,       \
                           strideD,       \
                           referenceD,    \
                           referenceC,    \
                           deviceOnHostD, \
                           deviceOnHostC);
#endif

#define TENSILE_CLIENT_CALL_SETUP(Ti, To, Tc)       \
    To* initialD;                                   \
    To* initialC;                                   \
    Ti* initialA;                                   \
    Ti* initialB;                                   \
    Tc  alpha;                                      \
    Tc  beta;                                       \
    To* referenceD;                                 \
    To* referenceC;                                 \
    To* deviceOnHostD;                              \
    To* deviceOnHostC;                              \
    initData(&initialD,                             \
             &initialC,                             \
             &initialA,                             \
             &initialB,                             \
             &alpha,                                \
             &beta,                                 \
             &referenceD,                           \
             &referenceC,                           \
             &deviceOnHostD,                        \
             &deviceOnHostC);                       \
    for(unsigned int i = 0; i < numBenchmarks; i++) \
    {                                               \
        TENSILE_CLIENT_CALL_PROBLEM                 \
    }                                               \
    destroyData(initialD,                           \
                initialC,                           \
                initialA,                           \
                initialB,                           \
                referenceD,                         \
                referenceC,                         \
                deviceOnHostD,                      \
                deviceOnHostC);

    switch(dataTypeEnum)
    {
#ifdef Tensile_DATA_TYPE_FLOAT
    case enum_float:
    {
        TENSILE_CLIENT_CALL_SETUP(float, float, float)
    }
    break;
#endif
#ifdef Tensile_DATA_TYPE_DOUBLE
    case enum_double:
    {
        TENSILE_CLIENT_CALL_SETUP(double, double, double)
    }
    break;
#endif
#ifdef Tensile_DATA_TYPE_TENSILECOMPLEXFLOAT
    case enum_TensileComplexFloat:
    {
        TENSILE_CLIENT_CALL_SETUP(TensileComplexFloat, TensileComplexFloat, TensileComplexFloat)
    }
    break;
#endif
#ifdef Tensile_DATA_TYPE_TENSILECOMPLEXDOUBLE
    case enum_TensileComplexDouble:
    {
        TENSILE_CLIENT_CALL_SETUP(TensileComplexDouble, TensileComplexDouble, TensileComplexDouble)
    }
    break;
#endif
#ifdef Tensile_DATA_TYPE_TENSILEHALF
    case enum_TensileHalf:
    {
        TENSILE_CLIENT_CALL_SETUP(TensileHalf, TensileHalf, TensileHalf)
    }
    break;
#endif

#ifdef Tensile_DATA_TYPE_TENSILEINT8X4
    case enum_TensileInt8x4:
    {
        TENSILE_CLIENT_CALL_SETUP(TensileInt8x4, TensileInt32, TensileInt32)
    }
    break;
#endif

#ifdef Tensile_DATA_TYPE_TENSILE_BFLOAT16
    case enum_tensile_bfloat16:
    {
        TENSILE_CLIENT_CALL_SETUP(tensile_bfloat16, tensile_bfloat16, float)
    }
    break;
#endif

    default:
        break;
        // nothing

#undef TENSILE_CLIENT_CALL_SETUP
#undef TENSILE_CLIENT_CALL_PROBLEM
    }

    // cleanup
    destroyControls();
    if(dataTypeEnum == enum_TensileInt8x4)
    {
        std::cout << std::endl
                  << "Fastest: " << globalFastestGFlops << " GOP/s " << globalFastestTime / 1000.0f
                  << " us by (" << globalFastestIdx << ") ";
    }
    else
    {
        std::cout << std::endl
                  << "Fastest: " << globalFastestGFlops << " GFlop/s "
                  << globalFastestTime / 1000.0f << " us by (" << globalFastestIdx << ") ";
    }
#if Tensile_CLIENT_BENCHMARK
    std::cout << solutions[fastestIdx]._name;
#else
    std::cout << functionNames[fastestIdx];
#endif
    std::cout << std::endl;
    if(invalids)
    {
#if Tensile_CLIENT_BENCHMARK
        printf("\nInvalid Solutions: %u/%u\n",
               static_cast<unsigned int>(invalidSolutions.size()),
               numSolutions);
        // for (unsigned int i = 0; i < numInvalidSolutions; i++) {
        for(std::set<unsigned int>::iterator i = invalidSolutions.begin();
            i != invalidSolutions.end();
            i++)
        {
            unsigned int invalidSolutionIdx = *i;
            printf("[%2u] %s\n", invalidSolutionIdx, solutions[invalidSolutionIdx]._name);
        }
#endif
        return EXIT_FAILURE;
    }
    else
    {
        return EXIT_SUCCESS;
    }
} // main

/*******************************************************************************
 * init controls
 ******************************************************************************/
void initControls()
{
#if Tensile_RUNTIME_LANGUAGE_OCL
    // setup opencl objects
    cl_uint numPlatforms, numDevices;
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    if(platformIdx >= numPlatforms)
    {
        std::cout << "Platform " << platformIdx << "/" << numPlatforms << " invalid" << std::endl;
        exit(1);
    }
    tensileStatusCheck(status);
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    status                    = clGetPlatformIDs(numPlatforms, platforms, NULL);
    tensileStatusCheck(status);
    platform = platforms[platformIdx];
    status   = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
    if(deviceIdx >= numDevices)
    {
        std::cout << "Device " << deviceIdx << "/" << numDevices << " invalid" << std::endl;
        exit(1);
    }
    tensileStatusCheck(status);
    cl_device_id* devices = new cl_device_id[numDevices];
    status                = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
    tensileStatusCheck(status);
    device = devices[deviceIdx];
    size_t nameLength;
    status = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &nameLength);
    tensileStatusCheck(status);
    char* deviceName = new char[nameLength + 1];
    status           = clGetDeviceInfo(device, CL_DEVICE_NAME, nameLength, deviceName, 0);
    tensileStatusCheck(status);
    std::cout << "Device: \"" << deviceName << "\"" << std::endl;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    tensileStatusCheck(status);

    if(useGPUTimer)
    {
        stream = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    }
    else
    {
        stream = clCreateCommandQueue(context, device, 0x0, &status);
    }
    tensileStatusCheck(status);

    delete[] devices;
    delete[] platforms;
#elif Tensile_RUNTIME_LANGUAGE_HIP
    int numDevices;
    status = hipGetDeviceCount(&numDevices);
    if(deviceIdx >= static_cast<unsigned int>(numDevices))
    {
        std::cout << "Device " << deviceIdx << "/" << numDevices << " invalid" << std::endl;
        exit(1);
    }
    tensileStatusCheck(status);
    status = hipSetDevice(deviceIdx);
    tensileStatusCheck(status);
    status = hipStreamCreate(&stream);
    tensileStatusCheck(status);

    hipDeviceProp_t deviceProperties;
    status = hipGetDeviceProperties(&deviceProperties, deviceIdx);
    tensileStatusCheck(status);
    expectedClockRate = deviceProperties.clockRate / 1000;
    size_t bandwidth  = (size_t)deviceProperties.memoryClockRate * deviceProperties.memoryBusWidth
                       * 2 / (8 * 1000 * 1000);
    size_t compute = (size_t)deviceProperties.clockRate * deviceProperties.multiProcessorCount * 2
                     * 64 / (1000 * 1000);
    std::cout << "################################################################################"
              << std::endl;
    std::cout << "# Device[" << deviceIdx << "]: " << deviceProperties.name << " (gfx"
              << deviceProperties.gcnArch << ")" << std::endl;
    std::cout << "# Compute:   " << compute << " GFlop/s (" << deviceProperties.multiProcessorCount
              << " CUs @ " << deviceProperties.clockRate / 1000 << " MHz)" << std::endl;
    std::cout << "# Bandwidth: " << bandwidth << " GB/s (" << deviceProperties.memoryBusWidth
              << "-bit @ " << deviceProperties.memoryClockRate / 1000 << " MHz)" << std::endl;
    std::cout << "################################################################################"
              << std::endl;
    /*
  std::cout << "# TotalGlobalMem: " << deviceProperties.totalGlobalMem << std::endl;
  std::cout << "# SharedMemPerBlock: " << deviceProperties.sharedMemPerBlock << std::endl;
  std::cout << "# RegsPerBlock: " << deviceProperties.regsPerBlock << std::endl;
  std::cout << "# WarpSize: " << deviceProperties.warpSize << std::endl;
  std::cout << "# MaxThreadsPerBlock: " << deviceProperties.maxThreadsPerBlock << std::endl;
  std::cout << "# TotalConstMem: " << deviceProperties.totalConstMem << std::endl;
  std::cout << "# Major: " << deviceProperties.major << std::endl;
  std::cout << "# Minor: " << deviceProperties.minor << std::endl;
  std::cout << "# L2CacheSize: " << deviceProperties.l2CacheSize << std::endl;
  std::cout << "# MaxThreadsPerMultiProcessor: " << deviceProperties.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "# ComputeMode: " << deviceProperties.computeMode << std::endl;
  std::cout << "# ClockInstructionRate: " << deviceProperties.clockInstructionRate << std::endl;
  std::cout << "# PCIDomainID: " << deviceProperties.pciDomainID << std::endl;
  std::cout << "# PCIBusID: " << deviceProperties.pciBusID << std::endl;
  std::cout << "# PCIDeviceID: " << deviceProperties.pciDeviceID << std::endl;
  std::cout << "# MaxSharedMemoryPerMultiProcessor: " << deviceProperties.maxSharedMemoryPerMultiProcessor << std::endl;
  std::cout << "# IsMultiGpuBoard: " << deviceProperties.isMultiGpuBoard << std::endl;
  std::cout << "# CanMapHostMemory: " << deviceProperties.canMapHostMemory << std::endl;
  std::cout << "# GCNArch: " << deviceProperties.gcnArch << std::endl;
  */

    // prepare to report device stats
    tensileInitDeviceStats();
#endif
}

/*******************************************************************************
 * destroy controls
 ******************************************************************************/
void destroyControls()
{
#if Tensile_RUNTIME_LANGUAGE_OCL
    clReleaseCommandQueue(stream);
    clReleaseContext(context);
    clReleaseDevice(device);
#else
    hipStreamDestroy(stream);
#endif
}

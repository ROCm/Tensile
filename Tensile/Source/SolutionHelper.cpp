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

#include "SolutionHelper.h"
#include "Tools.h"
#include <mutex>
#include <unistd.h>

#ifdef WIN32
__declspec(thread) KernelMap kernelMap;
#else
thread_local KernelMap kernelMap;
#endif

/*******************************************************************************
 * Compile OpenCL kernels
 ******************************************************************************/
#if Tensile_RUNTIME_LANGUAGE_OCL
void tensileGetCompiledOpenCLKernel(
  cl_kernel *kernel,
  const char *kernelSource,
  cl_command_queue queue,
  const char *sourceBuildOptions) {

  // is kernel already compiled?
  KernelMapKey key = std::make_tuple(queue, kernelSource);
  KernelMap::iterator idx = kernelMap.find(key); // < 1 microsecond
  if (idx != kernelMap.end()) {
    *kernel = idx->second;
    return;
  }

  // need to compile kernel
  cl_int status;
  cl_context clContext;
  cl_device_id clDevice;
  status = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
      sizeof(clContext), &clContext, NULL);
  tensileStatusCheck(status)
    status = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
        sizeof(clDevice), &clDevice, NULL);
  tensileStatusCheck(status)

  cl_program clProgram;
  clProgram = clCreateProgramWithSource(
    clContext,
    1, &kernelSource,
    NULL, &status );
  tensileStatusCheck(status)
  status = clBuildProgram(
    clProgram,
    1, &clDevice,
    sourceBuildOptions, NULL, NULL );
  tensileStatusCheck(status)

  // print build failure
  if (status != CL_SUCCESS) {
    printf("clBuildProgram Failed with status = %d\n", status);

    size_t len = 0;
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG,
        0, NULL, &len);
    char* buildLog = new char[len];
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG,
        len*sizeof(char), buildLog, 0);
    printf("\n\n\nBuild Log:\n\n");
    printf("%s\n", buildLog);
    printf("\n");
    printf("\nKernel Source:\n\n");
    printf("%s\n", kernelSource);
    delete[] buildLog;
  }
  status = clCreateKernelsInProgram(
    clProgram,
    1, kernel,
    NULL );
  tensileStatusCheck(status)
  status = clReleaseProgram(clProgram);
  tensileStatusCheck(status)

  // put kernel in map
  kernelMap[key] = *kernel;
}
#endif

/*******************************************************************************
 * Get Assembly Kernels for HIP
 ******************************************************************************/
#if Tensile_RUNTIME_LANGUAGE_HIP
hipError_t SolutionLock::getFunction(hipFunction_t *f, int deviceId,
                                     const std::string &kernelName,
                                     bool codeFromFiles)
{
  hipError_t e = hipSuccess;
  *f = nullptr;

  if (_hipFunctions ) {
    std::lock_guard<std::mutex> initFunctionsLock(_initFunctionsMutex);
    if ( !_hipFunctions ) {
      int numDevices = -1;
      e = hipGetDeviceCount( &numDevices );
      if (e) { return e; };

      _hipFunctions = new hipFunction_t[numDevices];
      for ( int i = 0; i < numDevices; i++) {
        _hipFunctions[i] = nullptr;
      }
    }
  }
  // TODO - handle CodeFromFiles=0
      //if not globalParameters["CodeFromFiles"]:
      //  s += "%shipModuleLoadData(&module, %s_coba);\n" % (t, kernelName)
  if ( !_hipFunctions[deviceId] ) {
    std::lock_guard<std::mutex> loadModuleLock(_loadModuleMutex);
    hipModule_t module = nullptr;
    if (codeFromFiles) {
      if (!_hipFunctions[deviceId]) {
        std::string pk1 = "assembly/" + kernelName + ".co";
        std::string pk2 = "../source/assembly/" + kernelName + ".co";
        if (access(pk2.c_str(), R_OK) != 0)
          e = hipModuleLoad(&module, pk1.c_str());
        else
          e = hipModuleLoad(&module, pk2.c_str());
      } else {
        std::string k = kernelName + "_coba";
        e = hipModuleLoadData(&module, k.c_str());
      }
      if (e) { return e; };
      e = hipModuleGetFunction(&_hipFunctions[deviceId], module, kernelName.c_str());
      if (e) { return e; };
    }
  }
  *f = _hipFunctions[deviceId];
  return e;
}

#endif

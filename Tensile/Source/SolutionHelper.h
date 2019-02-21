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

#ifndef SOLUTION_HELPER_H
#define SOLUTION_HELPER_H

#include "TensileTypes.h"
#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

/*******************************************************************************
 * Helper classes for locking, tracking, and getting solutions.
 * Works closely with SolutionMapper.h
 ******************************************************************************/

/*******************************************************************************
 * Kernel Cache
 ******************************************************************************/
#if Tensile_RUNTIME_LANGUAGE_OCL
typedef std::tuple<cl_command_queue, const char *> KernelMapKey;
typedef std::map<KernelMapKey, cl_kernel> KernelMap;
typedef void *DeviceFunctionType;
#else
typedef std::tuple<hipDevice_t, const char *> KernelMapKey;
typedef std::map<KernelMapKey, hipFunction_t> KernelMap;
typedef hipFunction_t DeviceFunctionType;
#endif

// Locks and tracker for kernel loading status
struct SolutionLock {
  SolutionLock() : _deviceFunctions(nullptr){};

  SolutionLock(const SolutionLock &other) {
    _deviceFunctions.store(other._deviceFunctions.load());
  };

  std::atomic<DeviceFunctionType *> _deviceFunctions;
  std::mutex _initFunctionsMutex;
  std::mutex _loadModuleMutex;

  // if codeFromExe==nullptr then load code from file using kernelName
  TensileStatus getFunction(DeviceFunctionType *f, int deviceId,
                            const std::string &kernelName,
                            const unsigned char *codeFromExe);
};

#ifdef WIN32
__declspec(thread) extern KernelMap kernelMap;
#else
extern thread_local KernelMap kernelMap;
#endif

/*******************************************************************************
 * Compile/Load Kernels
 ******************************************************************************/
#if Tensile_RUNTIME_LANGUAGE_OCL
void tensileGetCompiledOpenCLKernel(cl_kernel *kernel, const char *kernelSource,
                                    cl_command_queue queue,
                                    const char *sourceBuildOptions);
#else
// void tensileGetHipFunctionFromCodeObjectByteArray(
//  hipFunction_t *function,
//  const char *functionName,
//  const unsigned char *coba); // code object byte array
#endif

// solution info - constant compile or load-time information about the solution
struct SolutionInfo {
  // _functionPtr is a generic function pointer to a solution.
  // Different Problem types can have different solution function signatures ;
  // Use void* since so can use same type for all w/o flurry of auto-generated
  // template types
  void *_functionPtr;
  const char *_name;

  // These are requirements that the problem dims must meet in order to use this
  // solution
  // For example so kernels may be optimized with the assumption that the
  // summation is even
  // thus allowing faster code but the solution only works if the requirement is
  // met.
  // The structure here captures those requirements - they will be checked
  // before
  // launching the kernel
  ProblemProperties _assertionRequirements;
};

#endif

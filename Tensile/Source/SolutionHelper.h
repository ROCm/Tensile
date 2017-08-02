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
#include <map>
#include <tuple>

/*******************************************************************************
 * Kernel Cache
 ******************************************************************************/
#if Tensile_RUNTIME_LANGUAGE_OCL
typedef std::tuple<cl_command_queue, const char *> KernelMapKey;
typedef std::map<KernelMapKey, cl_kernel> KernelMap;
#else
typedef std::tuple<hipStream_t, const char *> KernelMapKey;
typedef std::map<KernelMapKey, hipFunction_t> KernelMap;
#endif

#ifdef WIN32
__declspec(thread) extern KernelMap kernelMap;
#else
extern thread_local KernelMap kernelMap;
#endif


/*******************************************************************************
 * Compile/Load Kernels
 ******************************************************************************/
#if Tensile_RUNTIME_LANGUAGE_OCL
void tensileGetCompiledOpenCLKernel(
  cl_kernel *kernel,
  const char *kernelSource,
  cl_command_queue queue,
  const char *sourceBuildOptions);
#else
void tensileGetHipFunctionFromCodeObjectByteArray(
  hipFunction_t *function,
  const char *functionName,
  const unsigned char *coba, // code object byte array
  hipStream_t stream );
#endif


/*******************************************************************************
 * Calculate sizes for multi kernel
 ******************************************************************************/
void tensileCalculateSizesForEdgeMultiKernel();
void tensileCalculateSizesForKernelMaxSizes();


#endif

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
#include <string>

/*******************************************************************************
 * OpenCL Kernel Cache
 ******************************************************************************/
#if Tensile_RUNTIME_LANGUAGE_OCL
typedef struct KernelMapKey_ {
  cl_command_queue queue;
  const char *kernelSource; // address of kernel source
} KernelMapKey;

typedef std::map<KernelMapKey, cl_kernel> KernelMap;
bool operator<(const KernelMapKey & l, const KernelMapKey & r);

#ifdef WIN32
__declspec(thread) extern KernelMap *kernelMap;
#else
extern __thread KernelMap *kernelMap;
#endif

#elif Tensile_RUNTIME_LANGUAGE_HIP
// HIP doesn't need kernel cache
#endif


/*******************************************************************************
 * Compile OpenCL kernels
 ******************************************************************************/
#if Tensile_RUNTIME_LANGUAGE_OCL
void tensileGetCompiledOpenCLKernel(
  cl_kernel *kernel,
  const char *kernelSource,
  cl_command_queue queue,
  const char *sourceBuildOptions);
#endif


/*******************************************************************************
 * Calculate sizes for multi kernel
 ******************************************************************************/
void tensileCalculateSizesForEdgeMultiKernel();
void tensileCalculateSizesForKernelMaxSizes();


#endif

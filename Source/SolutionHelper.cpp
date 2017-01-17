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

#if Tensile_BACKEND_OCL
#ifdef WIN32
__declspec(thread) KernelMap *kernelMap = 0;
#else
__thread KernelMap *kernelMap = 0;
#endif
bool operator<(const KernelMapKey & l, const KernelMapKey & r) {
  if (l.kernelSource < r.kernelSource) { return true; }
  else if (r.kernelSource < l.kernelSource) { return false; }
  if (l.queue < r.queue ) { return true; }
  else if (r.queue < l.queue) { return false; }
  return false;
}
#endif

/*******************************************************************************
 * Compile OpenCL kernels
 ******************************************************************************/
void tensileGetCompiledOpenCLKernel(
  cl_kernel *kernel,
  const char *kernelSource,
  cl_command_queue queue,
  const char *sourceBuildOptions) {
  // initialize kernel map
  if (!kernelMap) { kernelMap = new KernelMap(); }

  // is kernel already compiled?
  KernelMapKey key;
  key.queue = queue;
  key.kernelSource = kernelSource;
  KernelMap::iterator idx = kernelMap->find(key); // < 1 microsecond
  if (idx != kernelMap->end()) {
    *kernel = idx->second;
    //double t2 = timer.elapsed_us();
    //printf("kernel already compiled %p %p\n", kernel, *kernel);
    //printf("times %f, %f, %f\n", t0, t1, t2);
    return;
  }
  double t2 = timer.elapsed_us();
  //printf("times %f, %f, %f\n", t0, t1, t2);

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

  //printf("building kernel\n");
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
    printf("clBuildProgram Failed; statusor = %d\n", status);
    printf("\nKernel Source:\n\n");
    printf("%s\n", kernelSource);

    size_t len = 0;
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    char* buildLog = new char[len];
    clGetProgramBuildInfo(clProgram, clDevice, CL_PROGRAM_BUILD_LOG, len*sizeof(char), buildLog, 0);
    printf("\n\n\nBuild Log:\n\n");
    printf("%s\n", buildLog);
    printf("\n");
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
  (*kernelMap)[key] = *kernel;
}


/*******************************************************************************
 * Calculate sizes for multi kernel
 ******************************************************************************/
void tensileCalculateSizesForEdgeMultiKernel(){
}
void tensileCalculateSizesForKernelMaxSizes(){
}



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

/*******************************************************************************
 * Tensile Benchmark
 ******************************************************************************/

#include "Tensile.h"
#include "Tools.h"
#include "Solution.h"
#include "TensileSolutionCandidates.h"
#include "SolutionTensorContractionCPU.h"
#include "StructOperations.h"
#include "MathTemplates.h"

#include <tuple>

// commandline options

static bool doValidation;
static bool doValidationKernels;
static void parseCommandLineOptions(int argc, char *argv[]);

 // dummy/max sized just for filling initial data
static TensileTensor initialTensorFloatC;
static TensileTensor initialTensorFloatA;
static TensileTensor initialTensorFloatB;
static TensileTensor initialTensorDoubleC;
static TensileTensor initialTensorDoubleA;
static TensileTensor initialTensorDoubleB;

// buffers to hold initial data
static TensileTensorData initialTensorDataFloatC;
static TensileTensorData initialTensorDataFloatA;
static TensileTensorData initialTensorDataFloatB;
static TensileTensorData initialTensorDataDoubleC;
static TensileTensorData initialTensorDataDoubleA;
static TensileTensorData initialTensorDataDoubleB;

// scalar data
static float *alphaFloat;
static float *betaFloat;
static double *alphaDouble;
static double *betaDouble;

// device tensor data; max sized; initial data get clWriteBuffer each time
static TensileTensorData deviceTensorDataC; // input and result buffer
static TensileTensorData deviceTensorDataA;
static TensileTensorData deviceTensorDataB;
static TensileTensorData deviceTensorDataOnHostC; // result buffer coppied back to host for comparison
static TensileTensorData deviceTensorDataOnHostA; // result buffer coppied back to host for comparison
static TensileTensorData deviceTensorDataOnHostB; // result buffer coppied back to host for comparison

// reference tensor data
static TensileTensorData referenceTensorDataC; // input and result buffer on host

// setup opencl
static unsigned int numPlatforms;
#if Tensile_BACKEND_OPENCL12
static unsigned int numDevices;
static cl_int status;
static cl_platform_id *platforms;
static cl_platform_id platform;
static cl_device_id *devices;
static cl_device_id device;
static cl_context context;
#elif Tensile_BACKEND_HIP
static hipError_t status;
static int numDevices;
static int device;
#endif

// controls
static TensileDeviceProfile deviceProfileReference;
static TensileStatus tensileStatus;
static TensileControl ctrl;
static TensileControl ctrlValidation;

static void initTensorData();
static void destroyTensorData();
static void fillTensor(TensileTensor, TensileTensorData, Tensile::Tensor::FillType, void *src);
static void initControls();
static void destroyControls();

#ifdef Tensile_ENABLE_FP16
static bool tensileDataTypeIsHalf( TensileDataType dataType );
#endif
static bool tensileDataTypeIsFloat( TensileDataType dataType );
static bool tensileDataTypeIsDouble( TensileDataType dataType );

static unsigned int tensorSizeMaxC_0;
static unsigned int tensorSizeMaxC_1;
static unsigned int tensorSizeMaxA_0;
static unsigned int tensorSizeMaxA_1;
static unsigned int tensorSizeMaxB_0;
static unsigned int tensorSizeMaxB_1;

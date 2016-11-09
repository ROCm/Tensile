/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

/*******************************************************************************
 * Cobalt Benchmark
 ******************************************************************************/

#include "Cobalt.h"
#include "Tools.h"
#include "Solution.h"
#include "CobaltSolutionCandidates.h"
#include "SolutionTensorContractionCPU.h"
#include "StructOperations.h"
#include "MathTemplates.h"

#include <tuple>

// commandline options

static bool doValidation;
static bool doValidationKernels;
static void parseCommandLineOptions(int argc, char *argv[]);

 // dummy/max sized just for filling initial data
static CobaltTensor initialTensorFloatC;
static CobaltTensor initialTensorFloatA;
static CobaltTensor initialTensorFloatB;
static CobaltTensor initialTensorDoubleC;
static CobaltTensor initialTensorDoubleA;
static CobaltTensor initialTensorDoubleB;

// buffers to hold initial data
static CobaltTensorData initialTensorDataFloatC;
static CobaltTensorData initialTensorDataFloatA;
static CobaltTensorData initialTensorDataFloatB;
static CobaltTensorData initialTensorDataDoubleC;
static CobaltTensorData initialTensorDataDoubleA;
static CobaltTensorData initialTensorDataDoubleB;

// scalar data
static float *alphaFloat;
static float *betaFloat;
static double *alphaDouble;
static double *betaDouble;

// device tensor data; max sized; initial data get clWriteBuffer each time
static CobaltTensorData deviceTensorDataC; // input and result buffer
static CobaltTensorData deviceTensorDataA;
static CobaltTensorData deviceTensorDataB;
static CobaltTensorData deviceTensorDataOnHostC; // result buffer coppied back to host for comparison
static CobaltTensorData deviceTensorDataOnHostA; // result buffer coppied back to host for comparison
static CobaltTensorData deviceTensorDataOnHostB; // result buffer coppied back to host for comparison

// reference tensor data
static CobaltTensorData referenceTensorDataC; // input and result buffer on host

// setup opencl
static unsigned int numPlatforms;
#if Cobalt_BACKEND_OPENCL12
static unsigned int numDevices;
static cl_int status;
static cl_platform_id *platforms;
static cl_platform_id platform;
static cl_device_id *devices;
static cl_device_id device;
static cl_context context;
#elif Cobalt_BACKEND_HIP
static hipError_t status;
static int numDevices;
static int device;
#endif

// controls
static CobaltDeviceProfile deviceProfileReference;
static CobaltStatus cobaltStatus;
static CobaltControl ctrl;
static CobaltControl ctrlValidation;

static void initTensorData();
static void destroyTensorData();
static void fillTensor(CobaltTensor, CobaltTensorData, Cobalt::Tensor::FillType, void *src);
static void initControls();
static void destroyControls();

static bool cobaltDataTypeIsHalf( CobaltDataType dataType );
static bool cobaltDataTypeIsFloat( CobaltDataType dataType );
static bool cobaltDataTypeIsDouble( CobaltDataType dataType );

static unsigned int tensorSizeMaxC_0;
static unsigned int tensorSizeMaxC_1;
static unsigned int tensorSizeMaxA_0;
static unsigned int tensorSizeMaxA_1;
static unsigned int tensorSizeMaxB_0;
static unsigned int tensorSizeMaxB_1;

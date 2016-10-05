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

bool doValidation;
bool doValidationKernels;
void parseCommandLineOptions(int argc, char *argv[]);

 // dummy/max sized just for filling initial data
CobaltTensor initialTensorFloatC;
CobaltTensor initialTensorFloatA;
CobaltTensor initialTensorFloatB;
CobaltTensor initialTensorDoubleC;
CobaltTensor initialTensorDoubleA;
CobaltTensor initialTensorDoubleB;

// buffers to hold initial data
CobaltTensorData initialTensorDataFloatC;
CobaltTensorData initialTensorDataFloatA;
CobaltTensorData initialTensorDataFloatB;
CobaltTensorData initialTensorDataDoubleC;
CobaltTensorData initialTensorDataDoubleA;
CobaltTensorData initialTensorDataDoubleB;

// scalar data
float *alphaFloat;
float *betaFloat;
double *alphaDouble;
double *betaDouble;

// device tensor data; max sized; initial data get clWriteBuffer each time
CobaltTensorData deviceTensorDataC; // input and result buffer
CobaltTensorData deviceTensorDataA;
CobaltTensorData deviceTensorDataB;
CobaltTensorData deviceTensorDataOnHostC; // result buffer coppied back to host for comparison
CobaltTensorData deviceTensorDataOnHostA; // result buffer coppied back to host for comparison
CobaltTensorData deviceTensorDataOnHostB; // result buffer coppied back to host for comparison

// reference tensor data
CobaltTensorData referenceTensorDataC; // input and result buffer on host

// setup opencl
unsigned int numPlatforms;
#if Cobalt_BACKEND_OPENCL12
unsigned int numDevices;
cl_int status;
cl_platform_id *platforms;
cl_platform_id platform;
cl_device_id *devices;
cl_device_id device;
cl_context context;
#elif Cobalt_BACKEND_HIP
hipError_t status;
int numDevices;
int device;
#endif

// controls
CobaltDeviceProfile deviceProfileReference;
CobaltStatus cobaltStatus;
CobaltControl ctrl;
CobaltControl ctrlValidation;

void initTensorData();
void destroyTensorData();
void fillTensor(CobaltTensor, CobaltTensorData, Cobalt::Tensor::FillType, void *src);
void initControls();
void destroyControls();

bool cobaltDataTypeIsHalf( CobaltDataType dataType );
bool cobaltDataTypeIsFloat( CobaltDataType dataType );
bool cobaltDataTypeIsDouble( CobaltDataType dataType );


unsigned int tensorSizeMaxC_0;
unsigned int tensorSizeMaxC_1;
unsigned int tensorSizeMaxA_0;
unsigned int tensorSizeMaxA_1;
unsigned int tensorSizeMaxB_0;
unsigned int tensorSizeMaxB_1;

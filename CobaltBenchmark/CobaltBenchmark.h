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
CobaltScalarData alphaFloat;
CobaltScalarData betaFloat;
CobaltScalarData alphaDouble;
CobaltScalarData betaDouble;

// device tensor data; max sized; initial data get clWriteBuffer each time
CobaltTensorData deviceTensorDataC; // input and result buffer
CobaltTensorData deviceTensorDataA;
CobaltTensorData deviceTensorDataB;
CobaltTensorData deviceTensorDataOnHostC; // result buffer coppied back to host for comparison
CobaltTensorData deviceTensorDataOnHostA; // result buffer coppied back to host for comparison
CobaltTensorData deviceTensorDataOnHostB; // result buffer coppied back to host for comparison

// reference tensor data
CobaltTensorData referenceTensorDataC; // input and result buffer on host
CobaltTensorData referenceTensorDataA; // just points to initialTensorDataFloat or Double
CobaltTensorData referenceTensorDataB;

// setup opencl
cl_int status;
cl_uint numPlatforms;
cl_platform_id *platforms;
cl_platform_id platform;
cl_uint numDevices;
cl_device_id *devices;
cl_device_id device;
cl_context context;

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
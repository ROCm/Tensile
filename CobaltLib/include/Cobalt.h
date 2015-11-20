
/*******************************************************************************
 * Cobalt.h
 * - public API
 ******************************************************************************/
#ifndef COBALT_H
#define COBALT_H

#if Cobalt_BACKEND_OPENCL
#include "CL/cl.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * Status
 ******************************************************************************/
typedef enum CobaltStatus_ {

  // success
  cobaltStatusSuccess = 0,
  
  /* cobaltValidateProblem() */

  // tensor errors
  cobaltStatusTensorNumDimensionsInvalidA,
  cobaltStatusTensorNumDimensionsInvalidB,
  cobaltStatusTensorNumDimensionsInvalidC,
  cobaltStatusTensorDimensionSizeInvalidA,
  cobaltStatusTensorDimensionSizeInvalidB,
  cobaltStatusTensorDimensionSizeInvalidC,
  cobaltStatusTensorDimensionStrideInvalidA,
  cobaltStatusTensorDimensionStrideInvalidB,
  cobaltStatusTensorDimensionStrideInvalidC,
  
  // operation errors
  cobaltStatusOperandNumDimensionsMismatch,
  cobaltStatusOperationOperandNumIndicesMismatch,
  cobaltStatusOperationNumIndicesMismatch,
  cobaltStatusOperationIndexAssignmentInvalidA,
  cobaltStatusOperationIndexAssignmentInvalidB,
  cobaltStatusOperationIndexAssignmentDuplicateA,
  cobaltStatusOperationIndexAssignmentDuplicateB,
  cobaltStatusOperationNumIndicesInvalid,
  cobaltStatusOperationNumFreeIndicesInvalid,
  cobaltStatusOperationNumSummationIndicesInvalid,

  // device profile errors
  cobaltStatusDeviceProfileDeviceNameInvalid,

  /* cobaltGetSolution() */
  cobaltStatusOperationTypeNotFound,
  cobaltStatusDeviceProfileNumDevicesInvalid,
  cobaltStatusDeviceProfileNotFound,
  cobaltStatusProblemNotSupported, // purposefully not supported
  cobaltStatusProblemNotFound, // should be supported but wasn't found

  /* cobaltEnqueueSolution() */
  cobaltStatusPerformanceWarningProblemSizeTooSmall,

  /* control errors */
  cobaltStatusControlInvalid,
  cobaltStatusDependencyInvalid,

  /* misc */
  cobaltStatusParametersInvalid,


} CobaltStatus;


/*******************************************************************************
 * Tensor
 ******************************************************************************/
typedef struct CobaltDimension_ {
  size_t stride;
  size_t size;
} CobaltDimension;

typedef enum CobaltPrecision_ {
  cobaltPrecisionSingle,
  cobaltPrecisionDouble,
  cobaltPrecisionSingleComplex,
  cobaltPrecisionDoubleComplex
} CobaltPrecision;

typedef struct CobaltTensor_ {
  CobaltPrecision precision;
  enum { maxDimensions = 16 } maxDimensions_;
  size_t numDimensions;
  CobaltDimension dimensions[maxDimensions];
} CobaltTensor;


/*******************************************************************************
 * Tensor Data - OpenCL 1.2
 ******************************************************************************/
#if Cobalt_BACKEND_OPENCL12

typedef struct CobaltTensorData {
  cl_mem clMem;
  size_t offset;
} CobaltTensorData;

/*******************************************************************************
 * Tensor Data - OpenCL 2.0
 ******************************************************************************/
#elif Cobalt_BACKEND_OPENCL20
typedef enum CobaltOpenCLBufferType_ {
  cobaltOpenCLBufferTypeClMem,
  cobaltOpenClBufferTypeSVM
} CobaltOpenCLBufferType;

typedef struct CobaltTensorData_ {
  void *data;
  CobaltOpenCLBufferType bufferType;
  size_t offset;
} CobaltTensorData;

/*******************************************************************************
 * Tensor Data - HCC
 ******************************************************************************/
#elif Cobalt_BACKEND_HCC
typedef void* CobaltTensorData;

/*******************************************************************************
 * Tensor Data - HSA
 ******************************************************************************/
#elif Cobalt_BACKEND_HSA  
typedef void* CobaltTensorData;

#endif

/*******************************************************************************
 * Device
 ******************************************************************************/
typedef struct CobaltDevice_ {
  enum { maxNameLength = 256 } maxNameLength_;
  char name[maxNameLength];
  size_t numComputeUnits;
  size_t clockFrequency;
} CobaltDevice;

typedef struct CobaltDeviceProfile_ {
  enum { maxDevices = 1 } maxDevices_;
  size_t numDevices;
  CobaltDevice devices[maxDevices];
} CobaltDeviceProfile;


/*******************************************************************************
 * Operation
 ******************************************************************************/
typedef enum CobaltOperationType_ {
  cobaltOperationTypeTensorContraction,
  cobaltOperationTypeConvolution
} CobaltOperationType;


typedef struct CobaltOperation_ {
  // C[i,j,k] = Sum_l Sum_m Sum_n A[n,l,i,m,j] B[j,l,m,k,n]
  //   0,1,2        3     4     5   5 3 0 4 1    1 3 4 2 5
  // free indices: i, k
  // batch indices: j
  // summation indices: l m n
  // indexAssignmentsA: {5, 3, 0, 4, 1}
  // indexAssignmentsB: {1, 3, 4, 2, 5}
  // tensorA.numDim: numFreeIndices/2 + numBatchIndices + numSummationIndices
  // tensorB.numDim: numFreeIndices/2 + numBatchIndices + numSummationIndices
  // tensorC.numDim: numFreeIndices+numBatchIndices
  CobaltOperationType type;
  size_t numFreeIndices;
  size_t numBatchIndices;
  size_t numSummationIndices;
  size_t indexAssignmentsA[CobaltTensor::maxDimensions];
  size_t indexAssignmentsB[CobaltTensor::maxDimensions];
} CobaltOperation;


/*******************************************************************************
 * Problem
 ******************************************************************************/
typedef struct CobaltProblem_ {
  CobaltTensor tensorA;
  CobaltTensor tensorB;
  CobaltTensor tensorC;
  CobaltDeviceProfile deviceProfile;
  CobaltOperation operation;
} CobaltProblem;

CobaltStatus cobaltValidateProblem( CobaltProblem problem );

/*******************************************************************************
 * Control
 ******************************************************************************/
typedef struct CobaltControl_ {
  size_t numDependencies;
#if Cobalt_BACKEND_OPENCL
  size_t maxQueues = 16;
  size_t numQueues;
  cl_command_queue queues[maxQueues];
  size_t numInputEvents; // superfluous for AMD
  cl_event *inputEvents; // superfluous for AMD
  size_t numOutputEvents; // superfluous for AMD
  cl_event *outputEvents; // superfluous for AMD
#endif
} CobaltControl;


/*******************************************************************************
 * Solution
 ******************************************************************************/
struct CobaltSolution; // forward declaration

CobaltStatus cobaltGetSolution(
    const CobaltProblem problem,
    struct CobaltSolution **solution );

CobaltStatus cobaltEnqueueSolution(
    struct CobaltSolution *solution,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltTensorData tensorDataC,
    CobaltControl *control );


/*******************************************************************************
 * Setup & Teardown
 **********************************8********************************************/
CobaltStatus cobaltSetup();
CobaltStatus cobaltTeardown();


/*******************************************************************************
 * toStrings
 ******************************************************************************/
CobaltStatus cobaltStatusToString(
    CobaltStatus code, char *cstr, size_t *size );
CobaltStatus cobaltStatusToString(
    CobaltStatus status, char *cstr, size_t *size );
CobaltStatus cobaltPrecisionToString(
    CobaltPrecision precision, char *cstr, size_t *size );
CobaltStatus cobaltOperationToString(
    CobaltOperationType type, char *cstr, size_t *size );
CobaltStatus cobaltProblemToString(
    CobaltProblem problem, char *cstr, size_t *size );


#ifdef __cplusplus
} // extern "C"
#endif

#endif // COBALT_H
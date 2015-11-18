
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
typedef enum CobaltCode_ {

  // success
  cobaltCodeSuccess = 0,
  
  /* cobaltValidateProblem() */

  // tensor errors
  cobaltCodeTensorNumDimensionsInvalidA,
  cobaltCodeTensorNumDimensionsInvalidB,
  cobaltCodeTensorNumDimensionsInvalidC,
  cobaltCodeTensorDimensionSizeInvalidA,
  cobaltCodeTensorDimensionSizeInvalidB,
  cobaltCodeTensorDimensionSizeInvalidC,
  cobaltCodeTensorDimensionStrideInvalidA,
  cobaltCodeTensorDimensionStrideInvalidB,
  cobaltCodeTensorDimensionStrideInvalidC,
  
  // operation errors
  cobaltCodeOperationNumIndexAssignmentsMismatchNumDimensionsA,
  cobaltCodeOperationNumIndexAssignmentsMismatchNumDimensionsB,
  cobaltCodeOperationNumFreeIndicesMismatch,
  cobaltCodeOperationFreeIndexDuplicateA,
  cobaltCodeOperationFreeIndexDuplicateB,
  cobaltCodeOperationFreeIndexUnassigned,
  cobaltCodeOperationBoundIndexInvalidA,
  cobaltCodeOperationBoundIndexInvalidB,
  cobaltCodeOperationBoundIndexNotBoundTypeA,
  cobaltCodeOperationBoundIndexNotBoundTypeB,
  cobaltCodeOperationBoundIndexMismatchA,
  cobaltCodeOperationBoundIndexMismatchB,
  cobaltCodeOperationBoundIndexNumDimensionsMismatch,

  // device profile errors
  cobaltCodeDeviceProfileNumDevicesInvalid,
  cobaltCodeDeviceProfileDeviceNameInvalid,

  /* cobaltGetSolution() */
  cobaltCodeProblemNotSupported, // purposefully not supported
  cobaltCodeProblemNotFound, // should be supported but wasn't found

  /* cobaltEnqueueSolution() */
  cobaltCodePerformanceWarningProblemSizeTooSmall,

  /* control errors */
  cobaltCodeControlInvalid,
  cobaltCodeDependencyInvalid,

  /* misc */
  cobaltCodeParametersInvalid,


} CobaltCode;


typedef struct CobaltStatus_ {
  enum { maxCodes = 16 } maxCodes_;
  size_t numCodes;
  CobaltCode codes[maxCodes];
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

typedef enum CobaltOperationIndexAssignmentType_ {
  cobaltOperationIndexAssignmentTypeBound,
  cobaltOperationIndexAssignmentTypeFree
} CobaltOperationIndexAssignmentType;

typedef struct CobaltOperationIndexAssignment_ {
  CobaltOperationIndexAssignmentType type; // contract with A,B or free with C
  size_t index; // index of A,B if contracting or index of C if free
} CobaltOperationIndexAssignment;

typedef struct CobaltOperation_ {
  CobaltOperationType type;

  size_t numBoundIndices;
  size_t boundIndicesA[CobaltTensor::maxDimensions]; // boundIndicesA[0] = 1
  size_t boundIndicesB[CobaltTensor::maxDimensions]; // boundIndicesA[2] = 1 A0 and B2 bound
  size_t numFreeIndicesOperand;
  size_t freeIndicesA[CobaltTensor::maxDimensions]; // freeIndicesA[0] = 3 means 3rd dimension of A corresponds to 0th dimension of C
  size_t freeIndicesB[CobaltTensor::maxDimensions];


  //size_t numOperationIndexAssignmentsA;
  //CobaltOperationIndexAssignment
  //    operationIndexAssignmentsA[CobaltTensor::maxDimensions];
  //size_t numOperationIndexAssignmentsB;
  //CobaltOperationIndexAssignment
  //    operationIndexAssignmentsB[CobaltTensor::maxDimensions];

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
    struct CobaltSolution *solution );

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
CobaltStatus cobaltCodeToString(
    CobaltCode code, char *cstr, size_t *size );
CobaltStatus cobaltStatusToString(
    CobaltStatus status, char *cstr, size_t *size );
CobaltStatus cobaltPrecisionToString(
    CobaltPrecision precision, char *cstr, size_t *size );
CobaltStatus cobaltOperationToString(
    CobaltOperationType type, char *cstr, size_t *size );
CobaltStatus cobaltOperationIndexAssignmentTypeToString(
    CobaltOperationIndexAssignmentType type, char *cstr, size_t *size );
CobaltStatus cobaltProblemToString(
    CobaltProblem problem, char *cstr, size_t *size );


#ifdef __cplusplus
} // extern "C"
#endif

#endif // COBALT_H
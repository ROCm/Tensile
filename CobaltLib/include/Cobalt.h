
/*******************************************************************************
 * Cobalt.h
 * - public API
 ******************************************************************************/
#ifndef COBALT_H
#define COBALT_H

#if Cobalt_BACKEND_OPENCL12
#include "CL/cl.h"
typedef cl_float2 CobaltComplexFloat;
typedef cl_double2 CobaltComplexDouble;
#else

#if (defined( __GNUC__ ) || defined( __IBMC__ ))
    #define Cobalt_ALIGNED(_x) __attribute__ ((aligned(_x)))
#else
    #define Cobalt_ALIGNED(_x)
#endif

typedef union {
   float  Cobalt_ALIGNED(8) s[2];
   struct{ float  x, y; };
   struct{ float  s0, s1; };
} CobaltComplexFloat;

typedef union {
   double  Cobalt_ALIGNED(8) s[2];
   struct{ double  x, y; };
   struct{ double  s0, s1; };
} CobaltComplexDouble;

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

  /* VALIDATION ERRORS */
  cobaltStatusValidationErrorMin,
  
  /* cobaltValidateProblem() */
  cobaltStatusProblemIsNull,

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
  cobaltStatusOperationIndexUnassigned,
  cobaltStatusOperationFreeIndexAssignmentsInvalid,
  cobaltStatusOperationBatchIndexAssignmentsInvalid,
  cobaltStatusOperationSummationIndexAssignmentsInvalid,

  // device profile errors
  cobaltStatusDeviceProfileDeviceNameInvalid,

  /* cobaltGetSolution() */
  cobaltStatusOperationTypeNotFound,
  cobaltStatusDeviceProfileNumDevicesInvalid,
  cobaltStatusDeviceProfileNotFound,
  cobaltStatusProblemNotSupported, // purposefully not supported
  cobaltStatusProblemNotFound, // should be supported but wasn't found


  /* control errors */
  cobaltStatusControlInvalid,
  cobaltStatusDependencyInvalid,

  /* misc */
  cobaltStatusParametersInvalid,

  cobaltStatusValidationErrorMax,
  cobaltStatusPerformanceWarningMin,

  /* Performance Warnings */

  /* cobaltEnqueueSolution() */
  cobaltStatusPerformanceWarningProblemSizeTooSmall,

  cobaltStatusPerformanceWarningMax,


} CobaltStatus;

/*******************************************************************************
 * Status is Error (incorrect) vs Warning (correct but slow)
 ******************************************************************************/
bool cobaltStatusIsValidationError( CobaltStatus status );
bool cobaltStatusIsPerformanceWarning( CobaltStatus status );


/*******************************************************************************
 * Tensor
 ******************************************************************************/
typedef enum CobaltDataType_ {
  cobaltDataTypeHalf,
  cobaltDataTypeSingle,
  cobaltDataTypeDouble,
  cobaltdataTypeComplexHalf,
  cobaltDataTypeComplexSingle,
  cobaltDataTypeComplexDouble,
  cobaltNumDataTypes,
  cobaltDataTypeNone,
} CobaltDataType;

typedef struct CobaltDimension_ {
  unsigned int stride;
  unsigned int size;
} CobaltDimension;

typedef struct CobaltTensor_ {
  CobaltDataType dataType;
  enum { maxDimensions = 16 } maxDimensions_;
  unsigned int numDimensions;
  CobaltDimension dimensions[maxDimensions];
} CobaltTensor;


/*******************************************************************************
 * Tensor Data - OpenCL 1.2
 ******************************************************************************/
#if Cobalt_BACKEND_OPENCL12
#include "CL/cl.h"

typedef struct CobaltTensorData_ {
  void *data;
  unsigned int offset;
} CobaltTensorData;

typedef struct CobaltScalarData_ {
  void *data;
  CobaltDataType dataType;
} CobaltScalarData;

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
  unsigned int numComputeUnits;
  unsigned int clockFrequency;
} CobaltDevice;

typedef struct CobaltDeviceProfile_ {
  enum { maxDevices = 1 } maxDevices_;
  unsigned int numDevices;
  CobaltDevice devices[maxDevices];
} CobaltDeviceProfile;


/*******************************************************************************
 * Operation
 ******************************************************************************/
typedef enum CobaltOperationType_ {
  cobaltOperationTypeContraction,
  cobaltOperationTypeConvolution
  //cobaltOperationTypeCorrelation
} CobaltOperationType;

/*
typedef struct CobaltOperation_ {
  // C[i,j,k] = Sum_l Sum_m Sum_n A[n,l,i,m,j] B[j,l,m,k,n]
  //   0,1,2        3     4     5   5 3 0 4 1    1 3 4 2 5
  // free indices: i, k
  // batch indices: j
  // summation indices: l m n
  // indexAssignmentsA: {5, 3, 0, 4, 1}
  // indexAssignmentsB: {1, 3, 4, 2, 5}

  CobaltOperationType type;
  bool useAlpha; // alpha is specified
  CobaltDataType alphaType;
  //void *alpha;
  bool useBeta; // beta is specified
  CobaltDataType betaType;
  //void *beta;
  unsigned int numIndicesFree;
  unsigned int numIndicesBatch;
  unsigned int numIndicesSummation;
  unsigned int indexAssignmentsA[CobaltTensor::maxDimensions];
  unsigned int indexAssignmentsB[CobaltTensor::maxDimensions];

  // used for convolutions/correlations only
  // unsigned int pad[CobaltTensor::maxDimensions];
  // unsigned int stride[CobaltTensor::maxDimensions];
  // unsigned int upscale[CobaltOperation::maxSummationIndices]; // cuDNN requires 1

} CobaltOperation;
*/

/*******************************************************************************
 * Problem
 ******************************************************************************/
//typedef struct CobaltProblem_ {
//  CobaltTensor tensorC;
//  CobaltTensor tensorA;
//  CobaltTensor tensorB;
//  CobaltDeviceProfile deviceProfile;
//  CobaltOperation operation;
//} CobaltProblem;

CobaltStatus cobaltValidateProblem( CobaltProblem problem );

/*******************************************************************************
 * Control
 ******************************************************************************/
typedef struct CobaltControl_ {
  unsigned int numDependencies;
  unsigned int mode; // bitfield
#if Cobalt_BACKEND_OPENCL12
  enum { maxQueues = 16 } maxQueues_;
  unsigned int numQueues;
  cl_command_queue queues[maxQueues];
  cl_uint numInputEvents; // superfluous for AMD
  cl_event *inputEvents; // superfluous for AMD
  cl_uint numOutputEvents; // superfluous for AMD
  cl_event *outputEvents; // superfluous for AMD
#endif
} CobaltControl;

#define Cobalt_CONTROL_MODE_VALIDATE          (1<<0)
#define Cobalt_CONTROL_MODE_VALIDATE_KERNELS  (1<<1)
#define Cobalt_CONTROL_MODE_BENCHMARK         (1<<2)
#define Cobalt_CONTROL_MODE_BENCHMARK_KERNELS (1<<3)

/*******************************************************************************
 * Solution
 ******************************************************************************/
typedef struct _CobaltProblem * CobaltProblem; // forward declare pimpl
typedef struct _CobaltSolution * CobaltSolution; // forward declare pimpl

CobaltProblem cobaltCreateProblem(
    CobaltTensor tensorC,
    CobaltTensor tensorA,
    CobaltTensor tensorB,
    unsigned int *indexAssignmentsA,
    unsigned int *indexAssignmentsB,
    CobaltOperationType operationType,
    CobaltDataType alphaType,
    CobaltDataType betaType,
    CobaltDeviceProfile deviceProfile,
    CobaltStatus *status );

CobaltSolution cobaltGetSolutionForProblem(
    const CobaltProblem problem,
    CobaltStatus *status );

CobaltStatus cobaltEnqueueSolution(
    CobaltSolution solution,
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl *control );


/*******************************************************************************
 * Setup & Teardown
 * --enable-validation - validates and records results in log
 * --enable-validation-kernels - validates individual kernels and records results in log
 * --enable-benchmarking - times each solution and records in log
 * --enable-benchmarking-kernels - times each kernel and records in log
 * ? results queryable through
 ******************************************************************************/
CobaltStatus cobaltSetup( const char *logFileName );
CobaltStatus cobaltTeardown();


/*******************************************************************************
 * toStrings
 ******************************************************************************/
CobaltStatus cobaltStatusToString(
    CobaltStatus status, char *cstr, unsigned int *size );
CobaltStatus cobaltDataTypeToString(
    CobaltDataType dataType, char *cstr, unsigned int *size );
CobaltStatus cobaltOperationToString(
    CobaltOperationType type, char *cstr, unsigned int *size );
CobaltStatus cobaltProblemToString(
    CobaltProblem problem, char *cstr, unsigned int *size );


#ifdef __cplusplus
} // extern "C"
#endif

#endif // COBALT_H

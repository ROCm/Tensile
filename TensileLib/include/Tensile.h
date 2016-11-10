/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


/*******************************************************************************
 * Tensile.h
 * - public API
 ******************************************************************************/
#ifndef TENSILE_H
#define TENSILE_H

#ifdef WIN32
#define _CRTDBG_MAP_ALLOC
#endif
#ifdef _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>
#endif

#undef Tensile_ENABLE_FP16
#undef Tensile_ENABLE_FP16_HOST

#if Tensile_BACKEND_OPENCL12
#include "CL/cl.h"
#else
#include <hip/hip_runtime.h>
#endif

#if (defined( __GNUC__ ) || defined( __IBMC__ ))
    #define Tensile_ALIGNED(_x) __attribute__ ((aligned(_x)))
#else
    #define Tensile_ALIGNED(_x)
#endif

#ifdef Tensile_ENABLE_FP16
#if Tensile_BACKEND_OPENCL12
typedef half TensileHalf;
#else
typedef __fp16 TensileHalf;
#endif

typedef union {
   TensileHalf  Tensile_ALIGNED(8) s[2];
   struct{ TensileHalf x, y; };
   struct{ TensileHalf s0, s1; };
} TensileComplexHalf;
#endif

typedef union {
   float  Tensile_ALIGNED(8) s[2];
   struct { float x, y; };
   struct { float s0, s1; };
} TensileComplexFloat;

typedef union {
   double  Tensile_ALIGNED(8) s[2];
   struct { double x, y; };
   struct { double s0, s1; };
} TensileComplexDouble;


#ifdef __cplusplus
extern "C" {
#endif


/*******************************************************************************
 * TensileStatus
 ******************************************************************************/
typedef enum TensileStatus_ {

  /* success */
  tensileStatusSuccess = 0,                                // success

  /* tensor errors */
  tensileStatusTensorNumDimensionsInvalid,                 // num dimensions isn't between 1 and max
  tensileStatusTensorDimensionOrderInvalid,                // dimensions not in order smallest to largest stride
  tensileStatusTensorDimensionStrideInvalid,               // stride is 0
  tensileStatusTensorDimensionSizeInvalid,                 // size is 0

  /* operation errors */
  tensileStatusOperandNumDimensionsMismatch,               // tensor and indexAssignments num dimensions don't match
  tensileStatusOperationOperandNumIndicesMismatch,         // tensor A,B don't have correct number of
                                                          // free, summation and batch indices
  tensileStatusOperationIndexAssignmentInvalidA,           // indexAssignmentsA invalid
  tensileStatusOperationIndexAssignmentInvalidB,           // indexAssignmentsA invalid
  tensileStatusOperationIndexAssignmentDuplicateA,         // indexAssignmentsA contains duplicate assignments
  tensileStatusOperationIndexAssignmentDuplicateB,         // indexAssignmentsA contains duplicate assignments
  tensileStatusOperationNumFreeIndicesInvalid,             // tensorC doesn't have at least 2 free indices,
                                                          // or it has a odd number of free indices
                                                          // or num total - num batch != num free indices
  tensileStatusOperationNumSummationIndicesInvalid,        // indexAssignments don't contain at least 1 summation index
  tensileStatusOperationIndexUnassigned,                   // indexAssignments missing an assignment
  tensileStatusOperationSummationIndexAssignmentsInvalid,  // indexAssignment in C and either A or B but not both,
                                                          // so assignment isn't free, summation or batch

  /* tensileGetSolution() */
  tensileStatusDeviceProfileNumDevicesInvalid,             // num devices isn't between 1 and max
  tensileStatusDeviceProfileNotSupported,                  // TensileLib not configured for device profile
  tensileStatusProblemNotSupported,                        // TensileLib doesn't have solution for problem

  /* control errors */
  tensileStatusControlInvalid,                             // enqueueSolution given invalid control object

  /* misc */
  tensileStatusInvalidParameter,                           // function passed invalid parameter

} TensileStatus;


/*******************************************************************************
 * tensileStatusCheck
 * prints whether status is error, warning or success and status string
 ******************************************************************************/
#define tensileStatusCheck(status) \
  if (status != tensileStatusSuccess) { \
    unsigned int _tensileStatusStringSize; \
    tensileStatusToString( status, nullptr, &_tensileStatusStringSize); \
    char *_tensileStatusString = new char[_tensileStatusStringSize]; \
    tensileStatusToString(status, _tensileStatusString, &_tensileStatusStringSize); \
    printf("TensileStatus::%s on line %u of %s\n", \
      _tensileStatusString, \
      __LINE__, \
      __FILE__); \
    delete[] _tensileStatusString; \
  }


/*******************************************************************************
 * tensileSetup & tensileTeardown
 * logFileName is c-string of where to write log file
 ******************************************************************************/
TensileStatus tensileSetup( const char *logFilePath );
TensileStatus tensileTeardown();


/*******************************************************************************
 * TensileDataType
 ******************************************************************************/
typedef enum TensileDataType_ {
  tensileDataTypeSingle,                 // 0
  tensileDataTypeDouble,                 // 1
  tensileDataTypeComplexSingle,          // 2
  tensileDataTypeComplexDouble,          // 3
  tensileDataTypeComplexConjugateSingle, // 4
  tensileDataTypeComplexConjugateDouble, // 5
#ifdef Tensile_ENABLE_FP16
  tensileDataTypeHalf,                   // 6
  tensileDataTypeComplexHalf,            // 7
  tensileDataTypeComplexConjugateHalf,   // 8
#endif
  tensileNumDataTypes,                   // 9
  tensileDataTypeNone,                   // 10
} TensileDataType;


/*******************************************************************************
 * TensileDimension
 ******************************************************************************/
typedef struct TensileDimension_ {
  unsigned int stride;
  unsigned int size;
} TensileDimension;


/*******************************************************************************
 * TensileTensor
 ******************************************************************************/
typedef struct TensileTensor_ {
  TensileDataType dataType;
  enum { maxDimensions = 16 } maxDimensions_;
  unsigned int numDimensions;
  TensileDimension dimensions[maxDimensions];
} TensileTensor;

/*******************************************************************************
* tensileCreateEmptyTensor
* - returns TensileTensor initialized to zero
******************************************************************************/
TensileTensor tensileCreateEmptyTensor();

/*******************************************************************************
 * Tensor Data - OpenCL 1.2
 ******************************************************************************/
#if Tensile_BACKEND_OPENCL12
#include "CL/cl.h"

typedef struct TensileTensorData_ {
  void *data;
  unsigned int offset;
} TensileTensorData;
typedef struct TensileTensorDataConst_ {
  const void *data;
  unsigned int offset;
} TensileTensorDataConst;


/*******************************************************************************
 * Tensor Data - HIP
 ******************************************************************************/
#elif Tensile_BACKEND_HIP
typedef struct TensileTensorData_ {
  void *data;
  unsigned int offset;
} TensileTensorData;
typedef struct TensileTensorDataConst_ {
  const void *data;
  unsigned int offset;
} TensileTensorDataConst;

#endif

typedef struct TensileScalarData_ {
  const void *data;
} TensileScalarData;


/*******************************************************************************
 * Device
 * the device on which the problem is to be computed
 ******************************************************************************/
typedef struct TensileDevice_ {
  enum { maxNameLength = 256 } maxNameLength_;
  char name[maxNameLength];
  unsigned int numComputeUnits;
  unsigned int clockFrequency;
  static const unsigned int flopsPerClock = 2*64;
} TensileDevice;


/*******************************************************************************
 * TensileDeviceProfile
 * describes the device(s) on which the problem is to be computed
 ******************************************************************************/
typedef struct TensileDeviceProfile_ {
  enum { maxDevices = 1 } maxDevices_;
  unsigned int numDevices;
  TensileDevice devices[maxDevices];
} TensileDeviceProfile;


/*******************************************************************************
 * tensileEnumerateDeviceProfiles
 * list of available TensileDeviceProfiles
 * if size is non-null, it is set to size if string user needs to allocate
 * if cstr is non-null, string is written to cstr buffer
 ******************************************************************************/
TensileStatus tensileEnumerateDeviceProfiles( TensileDeviceProfile *profiles, unsigned int *size);


/*******************************************************************************
* tensileCreateEmptyDeviceProfile
* returns TensileDeviceProfile initialized to zero
******************************************************************************/
TensileDeviceProfile tensileCreateEmptyDeviceProfile();


/*******************************************************************************
 * TensileOperationType
 ******************************************************************************/
typedef enum TensileOperationType_ {
  tensileOperationTypeContraction,
  tensileOperationTypeConvolution
  //tensileOperationTypeCorrelation
} TensileOperationType;


/*******************************************************************************
 * TensileControl
 * controls the execution of the solution (queue, dependencies, dependents)
 ******************************************************************************/
typedef struct TensileControl_ {
  void *validate;
  unsigned int benchmark;
  enum { maxQueues = 16 } maxQueues_;
  unsigned int numQueues;       // supplied by user
  unsigned int numQueuesUsed;   // returned by library
  unsigned int numInputEvents;  // supplied by user
  unsigned int numOutputEvents; // returned by library
#if Tensile_BACKEND_OPENCL12
  cl_command_queue queues[maxQueues];
  cl_event *inputEvents;
  cl_event *outputEvents;
#elif Tensile_BACKEND_HIP
  hipStream_t queues[maxQueues];
  hipEvent_t *inputEvents;
  hipEvent_t *outputEvents;
#endif
} TensileControl;



/*******************************************************************************
* tensileCreateEmptyControl
* returns TensileControl initialized to zero
******************************************************************************/
TensileControl tensileCreateEmptyControl();


/*******************************************************************************
 * TensileProblem
 * problem describes the computation to be performed
 ******************************************************************************/
typedef struct _TensileProblem * TensileProblem;


/*******************************************************************************
* TensileSolution
* solution describes how problem will be computed
*   kernels to be enqueued
*   kernel parameters
*   kernel thread range
******************************************************************************/
typedef struct _TensileSolution * TensileSolution;


/*******************************************************************************
 * tensileCreateProblem
 * creates TensileProblem object
 * buffer pointers are not specified here
 ******************************************************************************/
TensileStatus tensileCreateProblem(
    TensileProblem *problem,
    TensileTensor tensorC,
    TensileTensor tensorA,
    TensileTensor tensorB,
    unsigned int *indexAssignmentsA,
    unsigned int *indexAssignmentsB,
    TensileOperationType operationType,
    TensileDataType alphaType,
    TensileDataType betaType,
    bool useOffsets,
    TensileDeviceProfile deviceProfile );
TensileStatus tensileDestroyProblem( TensileProblem problem );


/*******************************************************************************
 * tensileValidateProblem
 * checks that problem is self consistent
 *   number of tensor dimensions
 *   free indices
 *   batch indices
 *   summation indices
 *   indexAssignments
 ******************************************************************************/
TensileStatus tensileValidateProblem( TensileProblem problem );


/*******************************************************************************
 * tensileGetSolutionForProblem
 * returns optimal solution for input problem according to prior benchmarking
 ******************************************************************************/
TensileStatus tensileGetSolutionForProblem(
    TensileSolution *solution,
    const TensileProblem problem );
TensileStatus tensileDestroySolution( TensileSolution solution );


/*******************************************************************************
 * tensileEnqueueSolution
 *   enqueues solution
 *   buffer pointers are specified here
 ******************************************************************************/
TensileStatus tensileEnqueueSolution(
    TensileSolution solution,
    TensileTensorData tensorDataC,
    TensileTensorDataConst tensorDataA,
    TensileTensorDataConst tensorDataB,
    TensileScalarData alpha,
    TensileScalarData beta,
    TensileControl *control );


/*******************************************************************************
 * tensile*ToString
 * get c-string representation of objects
 * if size is non-null, it is set to size if string user needs to allocate
 * if cstr is non-null, string is written to cstr buffer
 ******************************************************************************/
TensileStatus tensileStatusToString(
    TensileStatus status, char *cstr, unsigned int *size );
TensileStatus tensileProblemToString(
    TensileProblem problem, char *cstr, unsigned int *size );
TensileStatus tensileSolutionToString(
    TensileSolution solution, char *cstr, unsigned int *size );



#ifdef __cplusplus
} // extern "C"
#endif

#endif // TENSILE_H

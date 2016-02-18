#include "Cobalt.h"
#include "Logger.h"
#include <assert.h>
#include <stdio.h>
#include "ReferenceTensorContraction.h"

#if Cobalt_SOLVER_ENABLED
#include "CobaltGetSolution.h"
#endif

Cobalt::Logger Cobalt::logger;

/*******************************************************************************
 * cobaltSetup()
 ******************************************************************************/
CobaltStatus cobaltSetup( const char *logFileName ) {
  
  Cobalt::logger.init(logFileName);
  return cobaltStatusSuccess;
}

/*******************************************************************************
 * cobaltTeardown
 ******************************************************************************/
CobaltStatus cobaltTeardown() {
  return cobaltStatusSuccess;
}


/*******************************************************************************
 * cobaltStatusIsValidationError
 ******************************************************************************/
bool cobaltStatusIsValidationError( CobaltStatus status ) {
  return status < cobaltStatusValidationErrorMax
      && status > cobaltStatusValidationErrorMin;
}


/*******************************************************************************
 * cobaltStatusIsPerformanceWarning
 ******************************************************************************/
bool cobaltStatusIsPerformanceWarning( CobaltStatus status ) {
  return status < cobaltStatusPerformanceWarningMax
      && status > cobaltStatusPerformanceWarningMin;
}

#if 0
/*******************************************************************************
 * cobaltGetSolution
 ******************************************************************************/
CobaltStatus cobaltGetSolutionCPU(
    CobaltProblem problem,
    CobaltSolution **solution ) {
  bool problemIsTensorContraction = true;

  if (problemIsTensorContraction) {
    switch(problem.tensorC.dataType) {
    case cobaltDataTypeSingle:
      (*solution)->pimpl = new CobaltSolutionTensorContractionCPU<float,float,float,float,float>( problem );
      break;
    case cobaltDataTypeDouble:
      (*solution)->pimpl = new CobaltSolutionTensorContractionCPU<double,double,double,double,double>( problem );
      break;
    case cobaltDataTypeComplexSingle:
      (*solution)->pimpl = new CobaltSolutionTensorContractionCPU<CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat,CobaltComplexFloat>( problem );
      break;
    case cobaltDataTypeComplexDouble:
      (*solution)->pimpl = new CobaltSolutionTensorContractionCPU<CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble,CobaltComplexDouble>( problem );
      break;
    default:
      (*solution)->pimpl = nullptr;
      return cobaltStatusProblemNotSupported;
    }

    return cobaltStatusSuccess;
  } else {
  // TODO - reorganize to include CPU convolution also
    return cobaltStatusProblemNotSupported;
  }
}
#endif

CobaltStatus cobaltGetSolution(
    CobaltProblem problem,
    CobaltSolution **solution ) {

  CobaltStatus status;

  // if devices
  if (problem.deviceProfile.numDevices) {
    
    // cpu device
    if ( strcmp(problem.deviceProfile.devices[0].name, "cpu")==0 ) {
      status = cobaltGetSolutionCPU( problem, solution );

    // gpu device
    } else {
#if Cobalt_SOLVER_ENABLED
      status = cobaltGetSolutionGenerated( problem, solution );
#else
      (*solution)->pimpl = new CobaltSolutionLogOnly<void,void,void,void,void>(problem);
      status = cobaltStatusSuccess;
#endif
    }

  // no devices
  } else {
    status = cobaltStatusDeviceProfileNumDevicesInvalid;
  }

#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.logGetSolution((*solution)->pimpl, status);
#endif

  return status;
}

/*******************************************************************************
 * cobaltEnqueueSolution
 ******************************************************************************/
CobaltStatus cobaltEnqueueSolution(
    CobaltSolution *solution,
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl *ctrl ) {
  CobaltStatus status = cobaltStatusSuccess;
  // cpu device
  if ( strcmp(solution->pimpl->getProblem().deviceProfile.devices[0].name, "cpu")==0 ) {
      status = solution->pimpl->enqueue( tensorDataC, tensorDataA, tensorDataB,
          alpha, beta, *ctrl );

  // gpu device
  } else {
#if Cobalt_SOLVER_ENABLED
    status = solution->enqueue( tensorDataC, tensorDataA, tensorDataB,
        alpha, beta, *ctrl );
#endif
  }
#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.logEnqueueSolution(solution->pimpl, status, ctrl);
#endif
  return status;
}


/*******************************************************************************
 * cobaltValidateProblem
 ******************************************************************************/
bool arrayContains( unsigned int *array, size_t arraySize, size_t value) {
  for (size_t i = 0; i < arraySize; i++) {
    if (array[i] == value) {
      return true;
    }
  }
  return false;
}

CobaltStatus cobaltValidateProblem( CobaltProblem problem ) {

  /* tensorA */
  if (problem.tensorA.numDimensions < 1
    || problem.tensorA.numDimensions > problem.tensorA.maxDimensions ) {
      return cobaltStatusTensorNumDimensionsInvalidA;
  }
  for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
    if (problem.tensorA.dimensions[i].size < 1) {
      return cobaltStatusTensorDimensionSizeInvalidA;
    }
    if (problem.tensorA.dimensions[i].stride < 1) {
      return cobaltStatusTensorDimensionStrideInvalidA;
    }
  }

  /* tensorB */
  if (problem.tensorB.numDimensions < 1
    || problem.tensorB.numDimensions > problem.tensorB.maxDimensions ) {
      return cobaltStatusTensorNumDimensionsInvalidB;
  }
  for (size_t i = 0; i < problem.tensorB.numDimensions; i++) {
    if (problem.tensorB.dimensions[i].size < 1) {
      return cobaltStatusTensorDimensionSizeInvalidB;
    }
    if (problem.tensorB.dimensions[i].stride < 1) {
      return cobaltStatusTensorDimensionStrideInvalidB;
    }
  }

  /* tensorA,B */
  if (problem.tensorA.numDimensions != problem.tensorB.numDimensions) {
    return cobaltStatusOperandNumDimensionsMismatch;
  }
  
  /* tensorC */
  if (problem.tensorC.numDimensions < 1
    || problem.tensorC.numDimensions > problem.tensorC.maxDimensions ) {
      return cobaltStatusTensorNumDimensionsInvalidC;
  }
  for (size_t i = 0; i < problem.tensorC.numDimensions; i++) {
    if (problem.tensorC.dimensions[i].size < 1) {
      return cobaltStatusTensorDimensionSizeInvalidC;
    }
    if (problem.tensorC.dimensions[i].stride < 1) {
      return cobaltStatusTensorDimensionStrideInvalidC;
    }
  }

  /* operation */
  // every element must correspond to a valid free idx or valid sum idx
  // no duplicates
  if (problem.operation.numIndicesFree%2 != 0
      || problem.operation.numIndicesFree < 2) {
    return cobaltStatusOperationNumFreeIndicesInvalid;
  }
  if (problem.operation.numIndicesFree/2
      + problem.operation.numIndicesBatch
      + problem.operation.numIndicesSummation
      != problem.tensorA.numDimensions ) {
    return cobaltStatusOperationOperandNumIndicesMismatch;
  }
  if (problem.operation.numIndicesFree + problem.operation.numIndicesBatch
      != problem.tensorC.numDimensions ) {
    return cobaltStatusOperationNumFreeIndicesInvalid;
  }
  if (problem.operation.numIndicesSummation < 1 ) {
    return cobaltStatusOperationNumSummationIndicesInvalid;
  }
  size_t maxAssignmentIndex = problem.operation.numIndicesFree + problem.operation.numIndicesBatch + problem.operation.numIndicesSummation - 1;
  for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
    if (problem.operation.indexAssignmentsA[i] > maxAssignmentIndex) {
      return cobaltStatusOperationIndexAssignmentInvalidA;
    }
    if (problem.operation.indexAssignmentsB[i] > maxAssignmentIndex) {
      return cobaltStatusOperationIndexAssignmentInvalidB;
    }
    for (size_t j = i+1; j < problem.tensorA.numDimensions; j++) {
      if ( problem.operation.indexAssignmentsA[i]
          == problem.operation.indexAssignmentsA[j] ) {
        return cobaltStatusOperationIndexAssignmentDuplicateA;
      }
          if ( problem.operation.indexAssignmentsB[i]
          == problem.operation.indexAssignmentsB[j] ) {
        return cobaltStatusOperationIndexAssignmentDuplicateB;
      }
    }
  }
  // TODO - verify that all summation indices show up as sums
  // TODO - verify that all free indices show up as free
  // TODO - verify that all batch indices show up as batch

  size_t freeIndexCount = 0;
  size_t batchIndexCount = 0;
  size_t summationIndexCount = 0;
  // verify free & batch indices
  for (size_t i = 0; i < problem.operation.numIndicesFree
      + problem.operation.numIndicesBatch; i++ ) {
    // if A&&B has this index, incr batch
    bool aHas = arrayContains( problem.operation.indexAssignmentsA,
        problem.tensorA.numDimensions, i );
    bool bHas = arrayContains( problem.operation.indexAssignmentsB,
        problem.tensorB.numDimensions, i );
    if (aHas && bHas) {
      batchIndexCount++;
    } else if (aHas || bHas) {
      freeIndexCount++;
    } else {
      return cobaltStatusOperationIndexUnassigned;
    }
  }
  // verify summation indices
  for (size_t i = problem.operation.numIndicesFree
      + problem.operation.numIndicesBatch;
      i < problem.operation.numIndicesFree + problem.operation.numIndicesBatch
      + problem.operation.numIndicesSummation; i++ ) {
    bool aHas = arrayContains( problem.operation.indexAssignmentsA,
        problem.tensorA.numDimensions, i );
    bool bHas = arrayContains( problem.operation.indexAssignmentsB,
        problem.tensorB.numDimensions, i );
    if (aHas && bHas) {
      summationIndexCount++;
    } else {
      return cobaltStatusOperationIndexUnassigned;
    }
  }
  if (problem.operation.numIndicesFree != freeIndexCount) {
    return cobaltStatusOperationFreeIndexAssignmentsInvalid;
  }
  if (problem.operation.numIndicesBatch != batchIndexCount) {
    return cobaltStatusOperationBatchIndexAssignmentsInvalid;
  }
  if (problem.operation.numIndicesSummation != summationIndexCount) {
    return cobaltStatusOperationSummationIndexAssignmentsInvalid;
  }


  /* device profile */
  if ( problem.deviceProfile.numDevices < 1
      || problem.deviceProfile.numDevices > problem.deviceProfile.maxDevices ) {
    return cobaltStatusDeviceProfileNumDevicesInvalid;
  }
  for (size_t i = 0; i < problem.deviceProfile.numDevices; i++) {
    size_t nameLength = strlen(problem.deviceProfile.devices[i].name);
    if (nameLength < 1 || nameLength
        >= problem.deviceProfile.devices[0].maxNameLength) {
      return cobaltStatusDeviceProfileDeviceNameInvalid;
    }
  }

  return cobaltStatusSuccess;
}


/*******************************************************************************
 * toStrings
 ******************************************************************************/

CobaltStatus cppStringToCString(
  std::string state, char *cstr, unsigned int *size ) {
  if (cstr) {
    // do copy
    if (size) {
      // copy up to size
      size_t lengthToCopy = std::min(*size-1 /* reserve space for null char*/,
          (unsigned int)state.size());
      memcpy(cstr, state.c_str(), lengthToCopy);
      cstr[lengthToCopy] = '\0';
    } else {
      // copy all
      memcpy(cstr, state.c_str(), state.size());
      cstr[state.size()+1] = '\0';
    }
  } else {
    // just return size
    if (size) {
      // return size
      *size = (unsigned int) (state.size()+1); // include space for null char
    } else {
      // can't do anything
      return cobaltStatusParametersInvalid;
    }
  }
  return cobaltStatusSuccess;
}

CobaltStatus cobaltStatusToString( CobaltStatus code, char *cstr, unsigned int *size ) {
  std::string state = toString(code);
  return cppStringToCString( state, cstr, size);
}

CobaltStatus cobaltDataTypeToString(
    CobaltDataType dataType, char *cstr, unsigned int *size ) {
  std::string state = toString(dataType);
  return cppStringToCString( state, cstr, size );
}

CobaltStatus cobaltOperationToString(
    CobaltOperationType type, char *cstr, unsigned int *size ) {
  std::string state = toString(type);
  return cppStringToCString( state, cstr, size );
}

CobaltStatus cobaltProblemToString(
    CobaltProblem problem, char *cstr, unsigned int *size ) {
  std::string state = toString(problem);
  return cppStringToCString( state, cstr, size );
}

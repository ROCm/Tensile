#include "Cobalt.h"
#include "Logger.h"
#include <assert.h>
#include <stdio.h>


/*******************************************************************************
 * cobaltSetup()
 ******************************************************************************/
CobaltStatus cobaltSetup() {
  CobaltStatus status;
  status.numCodes = 0;
  return status;
}

/*******************************************************************************
 * cobaltTeardown
 ******************************************************************************/
CobaltStatus cobaltTeardown() {
  CobaltStatus status;
  status.numCodes = 0;
  return status;
}

/*******************************************************************************
 * cobaltGetSolution
 ******************************************************************************/
CobaltStatus cobaltGetSolution(
    CobaltProblem problem,
    CobaltSolution *solution ) {
  INIT_STATUS

  // request solution
#if Cobalt_SOLUTIONS_ENABLED
  functionStatus = cobaltGetSolution(this, solution);
#else
  solution = new LogSolution(problem);
#endif

#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.logGetSolution(solution, status);
#endif

  RETURN_STATUS
}

/*******************************************************************************
 * cobaltEnqueueSolution
 ******************************************************************************/
CobaltStatus cobaltEnqueueSolution(
    CobaltSolution *solution,
    CobaltTensorData a,
    CobaltTensorData b,
    CobaltTensorData c,
    CobaltControl *ctrl ) {
  INIT_STATUS

#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.logEnqueueSolution(solution, status, ctrl);
#endif

  RETURN_STATUS
}


/*******************************************************************************
 * cobaltValidateProblem
 ******************************************************************************/
CobaltStatus cobaltValidateProblem( CobaltProblem problem ) {
  INIT_STATUS

  /* tensorA */
  if (problem.tensorA.numDimensions < 1
    || problem.tensorA.numDimensions > problem.tensorA.maxDimensions ) {
      ADD_CODE_TO_STATUS( cobaltCodeTensorNumDimensionsInvalidA )
  }
  for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
    if (problem.tensorA.dimensions[i].size < 1) {
      ADD_CODE_TO_STATUS( cobaltCodeTensorDimensionSizeInvalidA )
    }
    if (problem.tensorA.dimensions[i].stride < 1) {
      ADD_CODE_TO_STATUS( cobaltCodeTensorDimensionStrideInvalidA )
    }
  }

  /* tensorB */
  if (problem.tensorB.numDimensions < 1
    || problem.tensorB.numDimensions > problem.tensorB.maxDimensions ) {
      ADD_CODE_TO_STATUS( cobaltCodeTensorNumDimensionsInvalidB )
  }
  for (size_t i = 0; i < problem.tensorB.numDimensions; i++) {
    if (problem.tensorB.dimensions[i].size < 1) {
      ADD_CODE_TO_STATUS( cobaltCodeTensorDimensionSizeInvalidB )
    }
    if (problem.tensorB.dimensions[i].stride < 1) {
      ADD_CODE_TO_STATUS( cobaltCodeTensorDimensionStrideInvalidB )
    }
  }

  /* tensorA,B */
  if (problem.tensorA.numDimensions != problem.tensorB.numDimensions) {
    ADD_CODE_TO_STATUS( cobaltCodeTensorNumDimensionsMismatchAB )
  }
  
  /* tensorC */
  if (problem.tensorC.numDimensions < 1
    || problem.tensorC.numDimensions > problem.tensorC.maxDimensions ) {
      ADD_CODE_TO_STATUS( cobaltCodeTensorNumDimensionsInvalidC )
  }
  for (size_t i = 0; i < problem.tensorC.numDimensions; i++) {
    if (problem.tensorC.dimensions[i].size < 1) {
      ADD_CODE_TO_STATUS( cobaltCodeTensorDimensionSizeInvalidC )
    }
    if (problem.tensorC.dimensions[i].stride < 1) {
      ADD_CODE_TO_STATUS( cobaltCodeTensorDimensionStrideInvalidC )
    }
  }

  /* operation */
  // every element must correspond to a valid free idx or valid sum idx
  // no duplicates
  if (problem.operation.numFreeIndicesAB + problem.operation.numSummationIndices
      >= problem.tensorA.numDimensions ) {
    ADD_CODE_TO_STATUS( cobaltCodeOperationNumIndicesInvalid )
  }
  if (problem.operation.numFreeIndicesAB > problem.tensorC.numDimensions ) {
    ADD_CODE_TO_STATUS( cobaltCodeOperationNumFreeIndicesInvalid )
  }
  if (problem.operation.numSummationIndices < 1 ) {
    ADD_CODE_TO_STATUS( cobaltCodeOperationNumSummationIndicesInvalid )
  }
  size_t maxAssignmentIndex = problem.operation.numSummationIndices + problem.tensorC.numDimensions-1;
  for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
    if (problem.operation.indexAssignmentsA[i] > maxAssignmentIndex) {
      ADD_CODE_TO_STATUS( cobaltCodeOperationIndexAssignmentInvalidA )
    }
    if (problem.operation.indexAssignmentsB[i] > maxAssignmentIndex) {
      ADD_CODE_TO_STATUS( cobaltCodeOperationIndexAssignmentInvalidB )
    }
    for (size_t j = i+1; j < problem.tensorA.numDimensions; j++) {
      if ( problem.operation.indexAssignmentsA[i]
          == problem.operation.indexAssignmentsA[j] ) {
        ADD_CODE_TO_STATUS( cobaltCodeOperationIndexAssignmentDuplicateA )
      }
          if ( problem.operation.indexAssignmentsB[i]
          == problem.operation.indexAssignmentsB[j] ) {
        ADD_CODE_TO_STATUS( cobaltCodeOperationIndexAssignmentDuplicateB )
      }
    }
  }


  /* device profile */
  if ( problem.deviceProfile.numDevices < 1
      || problem.deviceProfile.numDevices > problem.deviceProfile.maxDevices ) {
    ADD_CODE_TO_STATUS( cobaltCodeDeviceProfileNumDevicesInvalid )
  }
  for (size_t i = 0; i < problem.deviceProfile.numDevices; i++) {
    size_t nameLength = strlen(problem.deviceProfile.devices[i].name);
    if (nameLength < 1 || nameLength
        >= problem.deviceProfile.devices[0].maxNameLength) {
      ADD_CODE_TO_STATUS( cobaltCodeDeviceProfileDeviceNameInvalid )
    }
  }

  RETURN_STATUS
}


/*******************************************************************************
 * toStrings
 ******************************************************************************/

void cppStringToCString(
  std::string state, char *cstr, size_t *size, CobaltStatus & status ) {
  if (cstr) {
    // do copy
    if (size) {
      // copy up to size
      size_t lengthToCopy = std::min(*size-1 /* reserve space for null char*/,
          state.size());
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
      *size = state.size()+1; // include space for null char
    } else {
      // can't do anything
      ADD_CODE_TO_STATUS( cobaltCodeParametersInvalid )
    }
  }
}

CobaltStatus cobaltCodeToString( CobaltCode code, char *cstr, size_t *size ) {
  INIT_STATUS
  std::string state = toString(code);
  cppStringToCString( state, cstr, size, status );
  RETURN_STATUS
}

CobaltStatus cobaltStatusToString(
    CobaltStatus inputStatus, char *cstr, size_t *size ) {
  INIT_STATUS
  char *div = ",\n  ";
  std::string state = "status(";
  state += std::to_string(inputStatus.numCodes);
  state += "){";
  state += " " + toString(inputStatus.codes[0]);
  for (size_t i = 1; i < inputStatus.numCodes; i++) {
    state += div;
    state += toString(inputStatus.codes[i]);
  }
  state += " }";
  cppStringToCString( state, cstr, size, inputStatus );
  RETURN_STATUS
}

CobaltStatus cobaltPrecisionToString(
    CobaltPrecision precision, char *cstr, size_t *size ) {
  INIT_STATUS
  std::string state = toString(precision);
  cppStringToCString( state, cstr, size, status );
  RETURN_STATUS
}

CobaltStatus cobaltOperationToString(
    CobaltOperationType type, char *cstr, size_t *size ) {
  INIT_STATUS
  std::string state = toString(type);
  cppStringToCString( state, cstr, size, status );
  RETURN_STATUS
}

CobaltStatus cobaltProblemToString(
    CobaltProblem problem, char *cstr, size_t *size ) {
  INIT_STATUS
  std::string state = toString(problem);
  cppStringToCString( state, cstr, size, status );
  RETURN_STATUS
}

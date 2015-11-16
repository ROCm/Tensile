#include "Cobalt.h"
#include "Logger.h"
#include <assert.h>
#include <stdio.h>

#define INIT_STATUS CobaltStatus status; status.numCodes = 0;
#define ADD_CODE_TO_STATUS(CODE) status.codes[status.numCodes++] = CODE;
#define RETURN_STATUS \
  if (status.numCodes == 0) { \
    ADD_CODE_TO_STATUS(cobaltCodeSuccess) \
  } \
  return status;

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
   const CobaltProblem & problem,
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

  // operation num indices match tensor num indices
  if ( problem.operation.numOperationIndexAssignmentsA
      != problem.tensorA.numDimensions ) {
    ADD_CODE_TO_STATUS(
        cobaltCodeOperationNumIndexAssignmentsMismatchNumDimensionsA );
  }
  if ( problem.operation.numOperationIndexAssignmentsB
      != problem.tensorB.numDimensions ) {
    ADD_CODE_TO_STATUS(
        cobaltCodeOperationNumIndexAssignmentsMismatchNumDimensionsB );
  }

  // tensorA,B have same num free indices
  size_t numFreeIndicesA = 0;
  for( size_t i = 0; i < problem.operation.numOperationIndexAssignmentsA; i++) {
    if (problem.operation.operationIndexAssignmentsA[i].type
      == cobaltOperationIndexAssignmentTypeFree ) {
        numFreeIndicesA++;
    }
  }
  size_t numFreeIndicesB = 0;
  for( size_t i = 0; i < problem.operation.numOperationIndexAssignmentsB; i++) {
    if (problem.operation.operationIndexAssignmentsB[i].type
      == cobaltOperationIndexAssignmentTypeFree ) {
        numFreeIndicesB++;
    }
  }
  if ( numFreeIndicesA != numFreeIndicesB ) {
    ADD_CODE_TO_STATUS( cobaltCodeOperationNumFreeIndicesMismatch )
  }

  // for each tensorC index; tensorA or B has it as a free assinment
  for (size_t i = 0; i < problem.tensorC.numDimensions; i++) {
    bool freeIndexOfA = false;
    for (size_t j = 0; j < problem.tensorA.numDimensions; j++) {
      if (problem.operation.operationIndexAssignmentsA[i].type
          ==cobaltOperationIndexAssignmentTypeFree
          && problem.operation.operationIndexAssignmentsA[i].index == j) {
        if (freeIndexOfA) {
          ADD_CODE_TO_STATUS( cobaltCodeOperationFreeIndexDuplicateA )
        }
        freeIndexOfA = true;
      }
    }
    bool freeIndexOfB = false;
    for (size_t j = 0; j < problem.tensorB.numDimensions; j++) {
      if (problem.operation.operationIndexAssignmentsB[i].type
          ==cobaltOperationIndexAssignmentTypeFree
          && problem.operation.operationIndexAssignmentsB[i].index == j) {
        if (freeIndexOfB) {
          ADD_CODE_TO_STATUS( cobaltCodeOperationFreeIndexDuplicateB )
        }
        freeIndexOfB = true;
      }
    }
    if ( !freeIndexOfA && !freeIndexOfB ) {
      ADD_CODE_TO_STATUS( cobaltCodeOperationFreeIndexUnassigned )
    }
  }

  // for each A index bound to B, B is bound back
  size_t numBoundIndices = 0;
  for( size_t i = 0; i < problem.operation.numOperationIndexAssignmentsA; i++) {
    if (problem.operation.operationIndexAssignmentsA[i].type
        == cobaltOperationIndexAssignmentTypeBound ) {
      size_t boundIndex = problem.operation.operationIndexAssignmentsA[i].index;
      if ( boundIndex >= problem.tensorB.numDimensions ) {
        ADD_CODE_TO_STATUS( cobaltCodeOperationBoundIndexInvalidA )
      }
      if ( problem.operation.operationIndexAssignmentsB[boundIndex].type
          != cobaltOperationIndexAssignmentTypeBound) {
        ADD_CODE_TO_STATUS( cobaltCodeOperationBoundIndexNotBoundTypeB )
      }
      if ( problem.operation.operationIndexAssignmentsB[boundIndex].index
          != i) {
        ADD_CODE_TO_STATUS( cobaltCodeOperationBoundIndexMismatchB )
      }
      if ( problem.tensorA.dimensions[i].size
          != problem.tensorB.dimensions[boundIndex].size) {
        ADD_CODE_TO_STATUS( cobaltCodeOperationBoundIndexNumDimensionsMismatch )
      }
      numBoundIndices++;
    }
  }
  // for each B index bound to A, A is bound back
  for( size_t i = 0; i < problem.operation.numOperationIndexAssignmentsB; i++) {
    if (problem.operation.operationIndexAssignmentsB[i].type
        == cobaltOperationIndexAssignmentTypeBound ) {
      size_t boundIndex = problem.operation.operationIndexAssignmentsB[i].index;
      if ( boundIndex >= problem.tensorA.numDimensions ) {
        ADD_CODE_TO_STATUS( cobaltCodeOperationBoundIndexInvalidB )
      }
      if ( problem.operation.operationIndexAssignmentsA[boundIndex].type
          != cobaltOperationIndexAssignmentTypeBound) {
        ADD_CODE_TO_STATUS( cobaltCodeOperationBoundIndexNotBoundTypeA )
      }
      if ( problem.operation.operationIndexAssignmentsA[boundIndex].index
          != i) {
        ADD_CODE_TO_STATUS( cobaltCodeOperationBoundIndexMismatchA )
      }
      if ( problem.tensorB.dimensions[i].size
          != problem.tensorA.dimensions[boundIndex].size) {
        ADD_CODE_TO_STATUS( cobaltCodeOperationBoundIndexNumDimensionsMismatch )
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

CobaltStatus cobaltOperationIndexAssignmentTypeToString(
    CobaltOperationIndexAssignmentType type, char *cstr, size_t *size ) {
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

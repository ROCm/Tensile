#include "Cobalt.h"
#include "Problem.h"
#include "Solution.h"
#include "Logger.h"
#include "SolutionTensorContractionCPU.h"

#include <assert.h>
#include <stdio.h>

#if Cobalt_SOLVER_ENABLED
#include "CobaltGetSolution.h"
#endif

// global logger object
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


/*******************************************************************************
 * cobaltCreateProblem
 ******************************************************************************/
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
    CobaltStatus *status ) {

  CobaltProblem problem;

  try{
    problem = new _CobaltProblem();
    problem->pimpl = new Cobalt::Problem(
        tensorC,
        tensorA,
        tensorB,
        indexAssignmentsA,
        indexAssignmentsB,
        operationType,
        alphaType,
        betaType,
        deviceProfile );
        *status = cobaltStatusSuccess;
  } catch( const std::exception& e) {
    *status = cobaltStatusProblemNotSupported;
    problem = nullptr;
  }
  return problem;
};





/*******************************************************************************
 * cobaltGetSolutionForProblem
 ******************************************************************************/
CobaltSolution cobaltGetSolutionForProblem(
    CobaltProblem problem,
    CobaltStatus *status ) {

  CobaltSolution solution;

  // cpu device
  if ( problem->pimpl->deviceIsReference() ) {
    solution = new _CobaltSolution();
    std::tie(solution->pimpl, *status) = Cobalt::getSolutionCPU( problem );

  // gpu device
  } else {
#if Cobalt_SOLVER_ENABLED
    status = cobaltGetSolutionGenerated( problem, solution );
#else
    CobaltSolution solution = new _CobaltSolution();
    solution->pimpl = new Cobalt::SolutionLogOnly<void,void,void,void,void>(problem);
    *status = cobaltStatusSuccess;
#endif
  }



#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.logGetSolution(solution->pimpl, *status);
#endif

  return solution;
}

/*******************************************************************************
 * cobaltEnqueueSolution
 ******************************************************************************/
CobaltStatus cobaltEnqueueSolution(
    CobaltSolution solution,
    CobaltTensorData tensorDataC,
    CobaltTensorData tensorDataA,
    CobaltTensorData tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl *ctrl ) {
  CobaltStatus status = cobaltStatusSuccess;
  // cpu device
  if (solution->pimpl->getProblem()->pimpl->deviceIsReference()) {
      status = solution->pimpl->enqueue( tensorDataC, tensorDataA, tensorDataB,
          alpha, beta, *ctrl );

  // gpu device
  } else {
#if Cobalt_SOLVER_ENABLED
    status = solution->pimpl->enqueue( tensorDataC, tensorDataA, tensorDataB,
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
  if (problem == nullptr) {
  return problem->pimpl->validate();
  } else {
    return cobaltStatusProblemIsNull;
  }
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
  std::string state = Cobalt::toString(code);
  return cppStringToCString( state, cstr, size);
}

CobaltStatus cobaltDataTypeToString(
    CobaltDataType dataType, char *cstr, unsigned int *size ) {
  std::string state = Cobalt::toString(dataType);
  return cppStringToCString( state, cstr, size );
}

CobaltStatus cobaltOperationToString(
    CobaltOperationType type, char *cstr, unsigned int *size ) {
  std::string state = Cobalt::toString(type);
  return cppStringToCString( state, cstr, size );
}

CobaltStatus cobaltProblemToString(
    CobaltProblem problem, char *cstr, unsigned int *size ) {
  std::string state = problem->pimpl->toString();
  return cppStringToCString( state, cstr, size );
}

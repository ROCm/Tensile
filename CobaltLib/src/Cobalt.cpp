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

  CobaltProblem problem = new _CobaltProblem();
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

  return problem;
};


CobaltStatus cobaltValidateProblem( CobaltProblem problem ) {
  if (problem == nullptr) {
  return problem->pimpl->validate();
  } else {
    return cobaltStatusProblemIsNull;
  }
}


/*******************************************************************************
 * cobaltGetSolutionForProblem
 ******************************************************************************/
CobaltSolution cobaltGetSolutionForProblem(
    CobaltProblem problem,
    CobaltStatus *status ) {

  CobaltSolution solution = new _CobaltSolution();

  // cpu device
  if ( problem->pimpl->deviceIsReference() ) {
    solution = new _CobaltSolution();
    std::tie(solution->pimpl, *status) = Cobalt::getSolutionCPU( *(problem->pimpl) );

  // gpu device
  } else {
#if Cobalt_SOLVER_ENABLED
    //status = cobaltGetSolutionGenerated( problem, solution );
#else
    solution->pimpl = new Cobalt::SolutionLogOnly<void,void,void,void,void>(*(problem->pimpl));
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
  if (solution->pimpl->getProblem().deviceIsReference()) {
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
 * toStrings
 ******************************************************************************/
CobaltStatus cppStringToCString(
  std::string state, char *cstr, unsigned int *size ) {
  if (cstr) {
    // do copy
    if (size) {
      // copy up to size
      size_t lengthToCopy = min(*size-1 /* reserve space for null char*/,
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

CobaltStatus cobaltProblemToString(
    CobaltProblem problem, char *cstr, unsigned int *size ) {
  std::string state = problem->pimpl->toString();
  return cppStringToCString( state, cstr, size );
}

CobaltStatus cobaltSolutionToString(
    CobaltSolution solution, char *cstr, unsigned int *size ) {
  std::string state = solution->pimpl->toString(0);
  return cppStringToCString( state, cstr, size );
}


#include "Cobalt.h"
#include "Logger.h"

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

CobaltStatus cobaltGetSolution(
   const CobaltProblem & problem,
    CobaltSolution *solution ) {
  CobaltStatus status;
  status.numCodes = 0;

  // request solution
#if Cobalt_SOLUTIONS_ENABLED
  functionStatus = cobaltGetSolution(this, solution);
#else
  solution = new LogSolution(problem);
  if (status.numCodes < status.maxCodes) {
    status.codes[status.numCodes] = cobaltCodeSolutionsDisabled;
    status.numCodes++;
  }
#endif

#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.logGetSolution(solution, status);
#endif

  return status;
}

CobaltStatus cobaltEnqueueSolution(
    CobaltSolution *solution,
    CobaltTensorData a,
    CobaltTensorData b,
    CobaltTensorData c,
    CobaltControl *ctrl ) {
  CobaltStatus status;
  status.numCodes = 0;

#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.logEnqueueSolution(solution, status, ctrl);
#endif

  return status;
}

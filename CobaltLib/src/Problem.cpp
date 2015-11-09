#include "Problem.h"
#include "Logger.h"
#include "Solution.h"
#include "Status.h"

namespace Cobalt {

  
/*******************************************************************************
 * constructor
 ******************************************************************************/
ProblemDescriptor::ProblemDescriptor(
    const TensorDescriptor & inputTensorDescA,
    const TensorDescriptor & inputTensorDescB,
    const TensorDescriptor & inputTensorDescC,
    const OperationDescriptor & inputOperation,
    const DeviceProfile & inputDeviceProfile )
    : tensorDescA(inputTensorDescA),
    tensorDescB(inputTensorDescB),
    tensorDescC(inputTensorDescC),
    operation(inputOperation),
    deviceProfile(inputDeviceProfile),
    solution(nullptr),
    assignSolutionRequested(false) {
}

/*******************************************************************************
 * assignsolution
 ******************************************************************************/
const Status ProblemDescriptor::assignSolution() {
  Status status;

  // request solution
  if (!assignSolutionRequested) {
#if COBALT_SOLUTIONS_ENABLED
  functionStatus = assignSolution(this, solution);
#else
  solution = nullptr;
  status.add(StatusCode::solutionsDisabled);
#endif

  // solution already requested
  } else {
    status.add(StatusCode::assignSolutionAlreadyRequested);
    assignSolutionRequested = true;
  }

#if COBALT_LOGGER_ENABLED
  logger.logAssignSolution(this, status);
#endif

  return status;
}


/*******************************************************************************
 * enqueue solution
 ******************************************************************************/
const Status ProblemDescriptor::enqueueSolution(
    const TensorData & tensorDataA,
    const TensorData & tensorDataB,
    const TensorData & tensorDataC,
    Control & ctrl ) const {
  Status status;

#if COBALT_SOLUTIONS_ENABLED
  functionStatus = solution->enqueue(
      tensorDataA,
      tensorDataB,
      tensorDataC,
      ctrl );
#else
  status.add(StatusCode::solutionsDisabled);
#endif

#if COBALT_LOGGER_ENABLED
  logger.logEnqueueSolution(this, status, ctrl);
#endif

  return status;
}


/*******************************************************************************
 * comparison operator for stl
 ******************************************************************************/
bool ProblemDescriptor::operator< (const ProblemDescriptor & other ) const {

  // tensor A
  if( tensorDescA < other.tensorDescA) {
    return true;
  } else if (other.tensorDescA < tensorDescA ) {
    return false;
  }

  // tensor B
  if( tensorDescB < other.tensorDescB) {
    return true;
  } else if ( other.tensorDescB < tensorDescB ) {
    return false;
  }

  // tensor C
  if( tensorDescC < other.tensorDescC) {
    return true;
  } else if ( other.tensorDescC < tensorDescC ) {
    return false;
  }

  // operation
  if( operation < other.operation) {
    return true;
  } else if ( other.operation < operation ) {
    return false;
  }

  // device
  if( deviceProfile < other.deviceProfile) {
    return true;
  } else if ( other.deviceProfile < deviceProfile ) {
    return false;
  }

  // identical
  return false;
}


/*******************************************************************************
 * toString
 ******************************************************************************/
std::string ProblemDescriptor::toString( size_t indentLevel ) const {
  std::string state = Logger::indent(indentLevel);
  state += "<" + Logger::problemTag + ">\n";
  state += tensorDescA.toString( indentLevel+1);
  state += tensorDescB.toString( indentLevel+1);
  state += tensorDescC.toString( indentLevel+1);
  state += operation.toString( indentLevel+1);
  state += deviceProfile.toString( indentLevel+1);
  state += Logger::indent(indentLevel) + "</" + Logger::problemTag + ">\n";
  return state;
}

} // namespace Cobalt
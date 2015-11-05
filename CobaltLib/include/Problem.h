
#pragma once

#include "Tensor.h"
#include "Operation.h"
#include "Device.h"
#include "Status.h"
#include "Control.h"

namespace Cobalt {

class Solution;
  
/*******************************************************************************
 * Problem
 * - comprised of 3 tensor descriptors, operation and DeviceProfile
 * - these values cannot change once the "problem" has been build
 * - at the end of its constructor, the problem looks up its own solution
 * - user can enqueue the solution over and over using different 
 ******************************************************************************/
class Problem {
public:

/*******************************************************************************
 * Problem
 * - comprised of 3 tensor descriptors, operation and DeviceProfile
 * - these values cannot change once the "problem" has been build
 * - at the end of its constructor, the problem looks up its own solution
 * - user can enqueue the solution over and over using different 
 ******************************************************************************/
  Problem(
      const TensorDescriptor & inputTensorDescA,
      const TensorDescriptor & inputTensorDescB,
      const TensorDescriptor & inputTensorDescC,
      const Operation & inputOperation,
      const DeviceProfile & inputDeviceProfile );

  /*******************************************************************************
 * assignSolution
 * - look up the solution object Cobalt created
 ******************************************************************************/
  const Status assignSolution();

/*******************************************************************************
 * enqueueSolution
 * - enqueue the solution to the problem (multiple kernels)
 *   using provided input/output data and control (queue) object
 ******************************************************************************/
  const Status enqueueSolution(
      const TensorData & tensorDataA,
      const TensorData & tensorDataB,
      const TensorData & tensorDataC,
      Control & ctrl ) const;
  
/*******************************************************************************
 * comparison operator for STL
 ******************************************************************************/
  bool operator< ( const Problem & other ) const;
  
/*******************************************************************************
 * toString for writing xml
 ******************************************************************************/
  std::string toString( size_t indentLevel ) const;

private:
/*******************************************************************************
 * Problem state
 ******************************************************************************/
  const TensorDescriptor tensorDescA;
  const TensorDescriptor tensorDescB;
  const TensorDescriptor tensorDescC;
  const Operation operation;
  const DeviceProfile deviceProfile;

  
/*******************************************************************************
 * user calling enqueueSolution(...) calls solution->enqueue(...)
 ******************************************************************************/
  Solution *solution;
  bool assignSolutionRequested;

}; // class Problem

} // namespace Cobalt
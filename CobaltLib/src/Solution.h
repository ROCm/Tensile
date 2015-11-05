
#pragma once

#include "Dependency.h"
#include "Problem.h"
#include "Control.h"
#include "Status.h"
#include "Tensor.h"

#include <string>

namespace Cobalt {

typedef struct SolutionDescriptor {

} SolutionDescriptor;

class Solution {
public:

  SolutionDescriptor getDescriptor();

  virtual Status enqueue(
      TensorData tensorDataA,
      TensorData tensorDataB,
      TensorData tensorDataC,
      Control & ctrl ) = 0;

  virtual std::string toString( size_t indentLevel ) = 0;

protected:

private:
  Problem * problem; // problem used to get this solution

}; // class Solution

} // namespace Cobalt
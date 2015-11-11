
#include "StructOperations.h"


const std::string tensorTag = "Tensor";
const std::string dimensionTag = "Dim";
const std::string dimPairTag = "DimPair";
const std::string operationTag = "Operation";
const std::string deviceTag = "Device";
const std::string deviceProfileTag = "DeviceProfile";
const std::string problemTag = "Problem";
const std::string solutionTag = "Solution";
const std::string statusTag = "Status";
const std::string traceEntryTag = "Entry";
const std::string traceTag = "Trace";
const std::string getSummaryTag = "SummaryOfGet";
const std::string enqueueSummaryTag = "SummaryOfEnqueue";
const std::string documentTag = "ApplicationProblemProfile";
const std::string numDimAttr = "numDim";
const std::string operationAttr = "operation";
const std::string dimNumberAttr = "number";
const std::string dimStrideAttr = "stride";
const std::string nameAttr = "name";
const std::string typeEnumAttr = "typeEnum";
const std::string typeStringAttr = "typeString";

/*******************************************************************************
 * enum toString
 ******************************************************************************/
std::string indent(size_t level) {
  std::string indentStr = "";
  for (size_t i = 0; i < level; i++) {
    indentStr += "  ";
  }
  return indentStr;
}

std::string toString( CobaltCode code ) {
  return "";
}

std::string toString( CobaltPrecision precision ) {
  return "";
}

std::string toString( CobaltOperationType type ) {
  return "";
}

std::string toString( CobaltOperationIndexAssignmentType type ) {
  return "";
}
  

/*******************************************************************************
 * struct toString
 ******************************************************************************/
std::string toString( const CobaltProblem problem, size_t indentLevel ) {
  return indent(indentLevel);
}

std::string toString( const CobaltSolution *solution, size_t indentLevel ) {
  return indent(indentLevel);
}

std::string toString( const CobaltOperation operation, size_t indentLevel ) {
  return indent(indentLevel);
}

std::string toString( const CobaltDeviceProfile deviceProfile, size_t indentLevel ) {
  return indent(indentLevel);
}

std::string toString( const CobaltDevice device, size_t indentLevel ) {
  return indent(indentLevel);
}

std::string toString( const CobaltTensor tensor, size_t indentLevel ) {
  return indent(indentLevel);
}

std::string toString( const CobaltStatus status, size_t indentLevel ) {
  return indent(indentLevel);
}


/*******************************************************************************
 * comparators
 ******************************************************************************/

// CobaltStatus
bool operator< ( const CobaltStatus & l, const CobaltStatus & r ) {
  // status codes
  if (l.numCodes < r.numCodes) {
    return true;
  } else if (r.numCodes < l.numCodes) {
    return false;
  }
  for (size_t i = 0; i < l.numCodes; i++) {
    if (l.codes[i] < r.codes[i]) {
      return true;
    } else if (r.codes[i] < l.codes[i]) {
      return false;
    }
  }
  // identical
  return false;
}

// CobaltDimension
#if 1
bool operator<(const CobaltDimension & l, const CobaltDimension & r) {
  if (l.size < r.size) {
    return true;
  } else if (r.size < l.size) {
    return false;
  }
  if (l.stride < r.stride) {
    return true;
  } else if (r.stride < l.stride) {
    return false;
  }
  return false;
}
#endif

#if 1
// CobaltTensor
bool operator<(const CobaltTensor & l, const CobaltTensor & r) {
#if 1
  // precision
  if (l.precision < r.precision) {
    return true;
  } else if (r.precision < l.precision) {
    return false;
  }
  // dimensions
  if (l.numDimensions < r.numDimensions) {
    return true;
  } else if (r.numDimensions < l.numDimensions) {
    return false;
  }
  for (size_t i = 0; i < l.numDimensions; i++) {
    if (l.dimensions[i] < r.dimensions[i]) {
      return true;
    } else if (r.dimensions[i] < l.dimensions[i]) {
      return false;
    }
  }
#endif
  // identical
  return false;
}
#endif
// CobaltDevice
bool operator< ( const CobaltDevice & l, const CobaltDevice & r ) {
  return l.name < r.name;
}

// CobaltDeviceProfile
bool operator< ( const CobaltDeviceProfile & l, const CobaltDeviceProfile & r ) {
  if (l.numDevices < r.numDevices) {
    return true;
  } else if (r.numDevices < l.numDevices) {
    return false;
  }
  for (size_t i = 0; i < l.numDevices; i++) {
    if (l.devices[i] < r.devices[i]) {
      return true;
    } else if (r.devices[i] < l.devices[i]) {
      return false;
    }
  }
  // identical
  return false;
}

// CobaltOperationIndexAssignment
bool operator<(const CobaltOperationIndexAssignment & l, const CobaltOperationIndexAssignment & r) {
  // type
  if (l.type < r.type) {
    return true;
  } else if (r.type < l.type) {
    return false;
  }
  // index
  if (l.index < r.index) {
    return true;
  } else if (r.index < l.index) {
    return false;
  }
  // identical
  return true;
}

// CobaltOperation
bool operator<(const CobaltOperation & l, const CobaltOperation & r) {
  // type
  if (l.type < r.type) {
    return true;
  } else if (r.type < l.type) {
    return false;
  }
  // operationIndexAssignmentsA
  if (l.numOperationIndexAssignmentsA < r.numOperationIndexAssignmentsA) {
    return true;
  } else if (r.numOperationIndexAssignmentsA < l.numOperationIndexAssignmentsA) {
    return false;
  }
  for (size_t i = 0; i < l.numOperationIndexAssignmentsA; i++) {
    if (l.operationIndexAssignmentsA[i] < r.operationIndexAssignmentsA[i]) {
      return true;
    } else if (r.operationIndexAssignmentsA[i] < l.operationIndexAssignmentsA[i]) {
      return false;
    }
  }
  // operationIndexAssignmentsB
  if (l.numOperationIndexAssignmentsB < r.numOperationIndexAssignmentsB) {
    return true;
  } else if (r.numOperationIndexAssignmentsB < l.numOperationIndexAssignmentsB) {
    return false;
  }
  for (size_t i = 0; i < l.numOperationIndexAssignmentsB; i++) {
    if (l.operationIndexAssignmentsB[i] < r.operationIndexAssignmentsB[i]) {
      return true;
    } else if (r.operationIndexAssignmentsB[i] < l.operationIndexAssignmentsB[i]) {
      return false;
    }
  }
  // identical
  return false;
}

// CobaltProblem
bool operator<(const CobaltProblem & l, const CobaltProblem & r) {
  // tensor A
  if( l.tensorA < r.tensorA) {
    return true;
  } else if (l.tensorA < r.tensorA ) {
    return false;
  }
  // tensor B
  if( l.tensorB < r.tensorB) {
    return true;
  } else if ( r.tensorB < l.tensorB ) {
    return false;
  }
  // tensor C
  if( l.tensorC < r.tensorC) {
    return true;
  } else if ( r.tensorC < l.tensorC ) {
    return false;
  }
  // operation
  if( l.operation < r.operation) {
    return true;
  } else if ( r.operation < l.operation ) {
    return false;
  }
  // device
  if( l.deviceProfile < r.deviceProfile) {
    return true;
  } else if ( r.deviceProfile < l.deviceProfile ) {
    return false;
  }
  // identical
  return false;
}

// CobaltControl
bool operator< ( const CobaltControl & l, const CobaltControl & r ) {
  return l.numDependencies < r.numDependencies;
}

// CobaltSolution
bool operator< ( const CobaltSolution & l, const CobaltSolution & r ) {
  // problem
  return l.problem < r.problem;
}

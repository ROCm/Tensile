
#include "StructOperations.h"
#include <assert.h>


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

#define COBALT_ENUM_TO_STRING_CASE(X) case X: return #X;
std::string toString( CobaltCode code ) {
  switch( code ) {

        // success
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeSuccess )
    
    // problem
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidProblem )
    
    // tensor errors
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidTensorDataA )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidTensorDataB )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidTensorDataC )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidTensorDescriptorA )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidTensorDescriptorB )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidTensorDescriptorC )
    
    // device errors
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidDeviceProfile )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidDevice )
    
    // operation errors
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidOperation )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidIndexOperationsA )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidIndexOperationsB )
    
    // solution errors
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeSolutionsDisabled )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidSolution )
    
    // control errors
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidControl )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeInvalidDependency )
    
    // performance warnings
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeGetSolutionAlreadyRequested )
    COBALT_ENUM_TO_STRING_CASE( cobaltCodeProblemSizeTooSmall )


  default:
    return "Error in toString(CobaltCode): no switch case for: " + std::to_string(code);
  };
}

std::string toString( CobaltPrecision precision ) {
  switch( precision ) {
    COBALT_ENUM_TO_STRING_CASE( cobaltPrecisionSingle )
    COBALT_ENUM_TO_STRING_CASE( cobaltPrecisionDouble )
    COBALT_ENUM_TO_STRING_CASE( cobaltPrecisionSingleComplex )
    COBALT_ENUM_TO_STRING_CASE( cobaltPrecisionDoubleComplex )
  default:
    return "Error in toString(CobaltPrecision): no switch case for: " + std::to_string(precision);
  };
}

std::string toString( CobaltOperationType type ) {
  switch( type ) {
    COBALT_ENUM_TO_STRING_CASE( cobaltOperationTypeTensorContraction )
    COBALT_ENUM_TO_STRING_CASE( cobaltOperationTypeConvolution )
  default:
    return "Error in toString(CobaltPrecision): no switch case for: " + std::to_string(type);
  };
}

std::string toString( CobaltOperationIndexAssignmentType type ) {
  switch( type ) {
    COBALT_ENUM_TO_STRING_CASE( cobaltOperationIndexAssignmentTypeBound )
    COBALT_ENUM_TO_STRING_CASE( cobaltOperationIndexAssignmentTypeFree )
  default:
    return "Error in toString(CobaltPrecision): no switch case for: " + std::to_string(type);
  };
}
  

/*******************************************************************************
 * struct toString
 ******************************************************************************/
std::string toString( const CobaltProblem problem, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<Problem>\n";
  state += toString( problem.tensorA, indentLevel+1);
  state += toString( problem.tensorB, indentLevel+1);
  state += toString( problem.tensorC, indentLevel+1);
  state += toString( problem.operation, indentLevel+1);
  state += toString( problem.deviceProfile, indentLevel+1);
  state += indent(indentLevel) + "</Problem>\n";
  return state;
}

std::string toString( const CobaltSolution *solution, size_t indentLevel ) {
  return solution->toString(indentLevel);
}

std::string toString( const CobaltOperation operation, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<Operation>\n";
  state += indent(indentLevel+1);
  // type
  state += "<OperationType enum=\"" + std::to_string(operation.type) + "\"";
  state += " string=\"" + toString(operation.type) + "\" />\n";
  // operationIndexAssignmentsA
  state += indent(indentLevel+1) + "<OperationIndexAssignments num=\"" + std::to_string( operation.numOperationIndexAssignmentsA) + "\">\n";
  for (size_t i = 0; i < operation.numOperationIndexAssignmentsA; i++) {
    state += indent(indentLevel+2);
    state += "<OperationIndexAssignment";
    state += " type=\"" + toString(operation.operationIndexAssignmentsA[i].type) + "\"";
    state += " index=\"" + std::to_string(operation.operationIndexAssignmentsA[i].index) + "\"";
    state += " />\n";
  }
  state += indent(indentLevel+1) + "</OperationIndexAssignments>\n";
  // operationIndexAssignmentsB
  state += indent(indentLevel+1) + "<OperationIndexAssignments num=\"" + std::to_string( operation.numOperationIndexAssignmentsA) + "\">\n";
  for (size_t i = 0; i < operation.numOperationIndexAssignmentsB; i++) {
    state += indent(indentLevel+2);
    state += "<OperationIndexAssignment";
    state += " type=\"" + toString(operation.operationIndexAssignmentsB[i].type) + "\"";
    state += " index=\"" + std::to_string(operation.operationIndexAssignmentsB[i].index) + "\"";
    state += " />\n";
  }
  state += indent(indentLevel+1) + "</OperationIndexAssignments>\n";
  state += indent(indentLevel) + "</Operation>\n";
  return state;
}


std::string toString( const CobaltDeviceProfile deviceProfile, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<DeviceProfile";
  state += " numDevices=\"" + std::to_string(deviceProfile.numDevices) + "\" >\n";
  for (size_t i = 0; i < deviceProfile.numDevices; i++) {
    state += toString( deviceProfile.devices[i], indentLevel+1);
  }
  state += indent(indentLevel) + "</DeviceProfile>\n";
  return state;
}

std::string toString( const CobaltDevice device, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<Device name=\"";
  state += device.name;
  state += "\"";
  state += " numCUs=\"" + std::to_string(device.numComputeUnits) + "\"";
  state += " clockFreq=\"" + std::to_string(device.clockFrequency) + "\"";
  state += " />\n";
  return state;
}

std::string toString( const CobaltTensor tensor, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<Tensor numDimensions=\"" + std::to_string(tensor.numDimensions) + "\"";
  state += " >\n";
  for (size_t i = 0; i < tensor.numDimensions; i++) {
    state += indent(indentLevel+1) + "<Dimension stride=\"" + std::to_string(tensor.dimensions[i].stride) + "\"";
    state += " size=\"" + std::to_string(tensor.dimensions[i].size) + "\" />\n";
  }
  state += indent(indentLevel) + "</Tensor>\n";
  return state;
}

std::string toString( const CobaltStatus status, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<Status numCodes=\"" + std::to_string(status.numCodes) + "\">\n";
  for (size_t i = 0; i < status.numCodes; i++) {
    state += indent(indentLevel+1) + "<Code enum=\"" + std::to_string(status.codes[i]) + "\"";
    state += " string=\"" + toString(status.codes[i]) + "\" />\n";
  }
  state += indent(indentLevel) + "</Status>\n";
  return state;
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
  // identical
  return false;
}

// CobaltTensor
bool operator<(const CobaltTensor & l, const CobaltTensor & r) {
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
  // identical
  return false;
}

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
  assert( l.type < l.type == false);
  assert( r.type < r.type == false);
  if (l.type < r.type) {
    return true;
  } else if (r.type < l.type) {
    return false;
  }
  // index
  assert( l.index < l.index == false);
  assert( r.index < r.index == false);
  if (l.index < r.index) {
    return true;
  } else if (r.index < l.index) {
    return false;
  }
  // identical
  return false;
}

// CobaltOperation
bool operator<(const CobaltOperation & l, const CobaltOperation & r) {
  // type
  assert( l.type < l.type == false);
  assert( r.type < r.type == false);
  if (l.type < r.type) {
    return true;
  } else if (r.type < l.type) {
    return false;
  }
  // operationIndexAssignmentsA
  assert( l.numOperationIndexAssignmentsA < l.numOperationIndexAssignmentsA == false);
  assert( r.numOperationIndexAssignmentsA < r.numOperationIndexAssignmentsA == false);
  if (l.numOperationIndexAssignmentsA < r.numOperationIndexAssignmentsA) {
    return true;
  } else if (r.numOperationIndexAssignmentsA < l.numOperationIndexAssignmentsA) {
    return false;
  }
  for (size_t i = 0; i < l.numOperationIndexAssignmentsA; i++) {
    assert( l.operationIndexAssignmentsA[i] < l.operationIndexAssignmentsA[i] == false);
    assert( r.operationIndexAssignmentsA[i] < r.operationIndexAssignmentsA[i] == false);
    if (l.operationIndexAssignmentsA[i] < r.operationIndexAssignmentsA[i]) {
      return true;
    } else if (r.operationIndexAssignmentsA[i] < l.operationIndexAssignmentsA[i]) {
      return false;
    }
  }
  // operationIndexAssignmentsB
  assert( l.numOperationIndexAssignmentsB < l.numOperationIndexAssignmentsB == false);
  assert( r.numOperationIndexAssignmentsB < r.numOperationIndexAssignmentsB == false);
  if (l.numOperationIndexAssignmentsB < r.numOperationIndexAssignmentsB) {
    return true;
  } else if (r.numOperationIndexAssignmentsB < l.numOperationIndexAssignmentsB) {
    return false;
  }
  for (size_t i = 0; i < l.numOperationIndexAssignmentsB; i++) {
    assert( l.operationIndexAssignmentsB[i] < l.operationIndexAssignmentsB[i] == false);
    assert( r.operationIndexAssignmentsB[i] < r.operationIndexAssignmentsB[i] == false);
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
  assert( l.tensorA < l.tensorA == false);
  assert( r.tensorA < r.tensorA == false);
  if( l.tensorA < r.tensorA) {
    return true;
  } else if (r.tensorA < l.tensorA ) {
    return false;
  }
  // tensor B
  assert( l.tensorB < l.tensorB == false);
  assert( r.tensorB < r.tensorB == false);
  if( l.tensorB < r.tensorB) {
    return true;
  } else if ( r.tensorB < l.tensorB ) {
    return false;
  }
  // tensor C
  assert( l.tensorC < l.tensorC == false);
  assert( r.tensorC < r.tensorC == false);
  if( l.tensorC < r.tensorC) {
    return true;
  } else if ( r.tensorC < l.tensorC ) {
    return false;
  }
  // operation
  assert( l.operation < l.operation == false);
  assert( r.operation < r.operation == false);
  if( l.operation < r.operation) {
    return true;
  } else if ( r.operation < l.operation ) {
    return false;
  }
  // device
  assert( l.deviceProfile < l.deviceProfile == false);
  assert( r.deviceProfile < r.deviceProfile == false);
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

#if 0
size_t TensorDescriptor::coordsToSerial( std::vector<size_t> coords ) const {
  size_t serial = 0;
  for (size_t i = 0; i < dimensions.size(); i++) {
    serial += coords[i] * dimensions[i].stride;
  }
  return serial;
}

std::vector<size_t> TensorDescriptor::serialToCoords( size_t serial ) const {
  std::vector<size_t> coords( dimensions.size() );
  size_t remainder = serial;
  for (size_t i = dimensions.size()-1; i >= 0; i--) {
    size_t coord = remainder / dimensions[i].stride;
    remainder = remainder % dimensions[i].stride;
  }
  return coords;
}

std::vector<DimensionDescriptor> TensorDescriptor::compactSizesToDimensions( std::vector<size_t> compactSizes ) {
  std::vector<DimensionDescriptor> dimensions( compactSizes.size() );
  dimensions[0].stride = 1;
  for (size_t i = 0; i < compactSizes.size()-1; i++) {
    dimensions[i].size = compactSizes[i];
    dimensions[i+1].stride = dimensions[i].size;
  }
  dimensions[compactSizes.size()-1].size = compactSizes[compactSizes.size()-1];
  return dimensions;
}
#endif
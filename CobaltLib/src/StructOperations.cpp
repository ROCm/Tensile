
#include "StructOperations.h"
#include <assert.h>
#include <string>
#include <stdio.h>
#include <iostream>


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
std::string toString( CobaltStatus status ) {
  switch( status ) {

    // success
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusSuccess )
  
  /* cobaltValidateProblem() */

  // tensor errors
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorNumDimensionsInvalidA )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorNumDimensionsInvalidB )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorNumDimensionsInvalidC )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionSizeInvalidA )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionSizeInvalidB )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionSizeInvalidC )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionStrideInvalidA )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionStrideInvalidB )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionStrideInvalidC )
  
  // operation errors
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperandNumDimensionsMismatch )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationOperandNumIndicesMismatch )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationNumIndicesMismatch )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationIndexAssignmentInvalidA )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationIndexAssignmentInvalidB )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationIndexAssignmentDuplicateA )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationIndexAssignmentDuplicateB )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationNumIndicesInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationNumFreeIndicesInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationNumSummationIndicesInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationIndexUnassigned )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationFreeIndexAssignmentsInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationBatchIndexAssignmentsInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationSummationIndexAssignmentsInvalid )

  // device profile errors
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusDeviceProfileNumDevicesInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusDeviceProfileDeviceNameInvalid )

  /* cobaltGetSolution() */
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusProblemNotSupported ) // purposefully not supported (real/complex mixed data types)
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusProblemNotFound ) // should be supported but wasn't found

  /* cobaltEnqueueSolution() */
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusPerformanceWarningProblemSizeTooSmall )

  /* control errors */
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusControlInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusDependencyInvalid )

  /* misc */
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusParametersInvalid )


  default:
    return "Error in toString(CobaltStatus): no switch case for: "
        + std::to_string(status);
  };
}

std::string toString( CobaltDataType dataType ) {
  switch( dataType ) {
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeSingle )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeDouble )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeSingleComplex )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeDoubleComplex )
  default:
    return "Error in toString(CobaltDataType): no switch case for: "
        + std::to_string(dataType);
  };
}

std::string toString( CobaltOperationType type ) {
  switch( type ) {
    COBALT_ENUM_TO_STRING_CASE( cobaltOperationTypeContraction )
    COBALT_ENUM_TO_STRING_CASE( cobaltOperationTypeConvolution )
  default:
    return "Error in toString(CobaltDataType): no switch case for: "
        + std::to_string(type);
  };
}

// pretty print
 std::string toString( CobaltProblem problem ) {
   // assumes problem has already been validated
  std::string state = "";
  static const char *indexChars = "ijklmnopqrstuvwxyz";

  state += "C[";
  state += indexChars[0];
  state += ":";
  state += std::to_string(problem.tensorC.dimensions[0].size);
  for (size_t i = 1; i < problem.tensorC.numDimensions; i++) {
    state += ",";
    state += indexChars[i];
    state += ":";
    state += std::to_string(problem.tensorC.dimensions[i].size);
  }
  state += "] = Sum(";
  state += indexChars[problem.tensorC.numDimensions];
  state += ":";
  //state += std::to_string(boundIndexSizes[0]);
  for (size_t i = 1; i < problem.operation.numIndicesSummation; i++) {
    state += ",";
    state += indexChars[problem.tensorA.numDimensions+i];
    state += ":";
    for (size_t j = 0; j < problem.tensorA.numDimensions; j++) {
      if (problem.operation.indexAssignmentsA[j] == i) {
        state += std::to_string(problem.tensorA.dimensions[j].size);
      }
    }
  }
  state += ") A[";
  for (size_t i = 0; i < problem.tensorA.numDimensions; i++) {
    state += indexChars[problem.operation.indexAssignmentsA[i]];
    if (i < problem.tensorA.numDimensions-1) {
      state += ",";
    }
  }
  state += "] * B[";
  for (size_t i = 0; i < problem.tensorB.numDimensions; i++) {
    state += indexChars[problem.operation.indexAssignmentsB[i]];
    if (i < problem.tensorB.numDimensions-1) {
      state += ",";
    }
  }
  state += "]";
  return state;
}
  

/*******************************************************************************
 * struct toString
 ******************************************************************************/
std::string toStringXML( const CobaltProblem problem, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<Problem string=\"" + toString(problem) + "\">\n";
  state += toStringXML( problem.tensorC, indentLevel+1);
  state += toStringXML( problem.tensorA, indentLevel+1);
  state += toStringXML( problem.tensorB, indentLevel+1);
  state += toStringXML( problem.operation, indentLevel+1);
  state += toStringXML( problem.deviceProfile, indentLevel+1);
  state += indent(indentLevel) + "</Problem>\n";
  return state;
}

std::string toStringXML( const CobaltSolution *solution, size_t indentLevel ) {
  return solution->toString(indentLevel);
}

std::string toStringXML( const CobaltOperation operation, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<Operation ";
  state += "alphaType=\""+std::to_string(operation.alphaType)+"\" ";
  state += "alpha=\""+std::to_string(operation.useAlpha)+"\" ";
  state += "betaType=\""+std::to_string(operation.alphaType)+"\" ";
  state += "beta=\""+std::to_string(operation.useBeta)+"\" ";
  state += "numIndicesFree=\""+std::to_string(operation.numIndicesFree)+"\" ";
  state += "numIndicesBatch=\""+std::to_string(operation.numIndicesBatch)+"\" ";
  state += "numIndicesSummation=\""+std::to_string(operation.numIndicesSummation)+"\" ";
  state += ">\n";
  state += indent(indentLevel+1);
  // type
  state += "<Type enum=\"" + std::to_string(operation.type) + "\"";
  state += " string=\"" + toString(operation.type) + "\" />\n";
  // operationIndexAssignmentsA
  state += indent(indentLevel+1) + "<IndexAssignments tensor=\"A\" >\n";
  for (size_t i = 0; i < operation.numIndicesFree/2 + operation.numIndicesBatch + operation.numIndicesSummation; i++) {
    state += indent(indentLevel+2);
    state += "<IndexAssignment";
    state += " index=\"" + std::to_string(i) + "\"";
    state += " indexAssignment=\""
        + std::to_string(operation.indexAssignmentsA[i]) + "\"";
    state += " />\n";
  }
  state += indent(indentLevel+1) + "</IndexAssignments>\n";
  // operationIndexAssignmentsB
  state += indent(indentLevel+1) + "<IndexAssignments tensor=\"B\" >\n";
  for (size_t i = 0; i < operation.numIndicesFree/2 + operation.numIndicesBatch + operation.numIndicesSummation; i++) {
    state += indent(indentLevel+2);
    state += "<IndexAssignment";
    state += " index=\"" + std::to_string(i) + "\"";
    state += " indexAssignment=\""
      + std::to_string(operation.indexAssignmentsB[i]) + "\"";
    state += " />\n";
  }
  state += indent(indentLevel+1) + "</IndexAssignments>\n";
  state += indent(indentLevel) + "</Operation>\n";
  return state;
}


std::string toStringXML(
    const CobaltDeviceProfile deviceProfile, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<DeviceProfile";
  state += " numDevices=\"" + std::to_string(deviceProfile.numDevices)
      + "\" >\n";
  for (size_t i = 0; i < deviceProfile.numDevices; i++) {
    state += toStringXML( deviceProfile.devices[i], indentLevel+1);
  }
  state += indent(indentLevel) + "</DeviceProfile>\n";
  return state;
}

std::string toStringXML( const CobaltDevice device, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<Device name=\"";
  state += device.name;
  state += "\"";
  state += " numComputeUnits=\"" + std::to_string(device.numComputeUnits) + "\"";
  state += " clockFrequency=\"" + std::to_string(device.clockFrequency) + "\"";
  state += " />\n";
  return state;
}

std::string toStringXML( const CobaltTensor tensor, size_t indentLevel ) {
  std::string state = indent(indentLevel);
  state += "<Tensor numDimensions=\"" + std::to_string(tensor.numDimensions)
      + "\"";
  state += " dataType=\"" + toString( tensor.dataType ) + "\"";
  state += " >\n";
  for (size_t i = 0; i < tensor.numDimensions; i++) {
    state += indent(indentLevel+1) + "<Dimension stride=\""
        + std::to_string(tensor.dimensions[i].stride) + "\"";
    state += " size=\"" + std::to_string(tensor.dimensions[i].size) + "\" />\n";
  }
  state += indent(indentLevel) + "</Tensor>\n";
  return state;
}


/*******************************************************************************
 * comparators
 ******************************************************************************/

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
  // dataType
  if (l.dataType < r.dataType) {
    return true;
  } else if (r.dataType < l.dataType) {
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
bool operator< (
    const CobaltDeviceProfile & l, const CobaltDeviceProfile & r ) {
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

// CobaltOperation
bool operator<(const CobaltOperation & l, const CobaltOperation & r) {
  // type
  if (l.type < r.type) {
    return true;
  } else if (r.type < l.type) {
    return false;
  }
  // numFree,Batch,SummationIndices
  if (l.numIndicesFree < r.numIndicesFree) {
    return true;
  } else if (r.numIndicesFree < l.numIndicesFree) {
    return false;
  }
  if (l.numIndicesBatch < r.numIndicesBatch) {
    return true;
  } else if (r.numIndicesBatch < l.numIndicesBatch) {
    return false;
  }
  if (l.numIndicesSummation < r.numIndicesSummation) {
    return true;
  } else if (r.numIndicesSummation < l.numIndicesSummation) {
    return false;
  }
  // indexAssignmentsA
  for (size_t i = 0; i < l.numIndicesFree/2+l.numIndicesBatch+l.numIndicesSummation; i++) {
    if (l.indexAssignmentsA[i] < r.indexAssignmentsA[i]) {
      return true;
    } else if (r.indexAssignmentsA[i] < l.indexAssignmentsA[i]) {
      return false;
    }
  }
  // indexAssignmentsB
  for (size_t i = 0; i < l.numIndicesFree/2+l.numIndicesBatch+l.numIndicesSummation; i++) {
    if (l.indexAssignmentsB[i] < r.indexAssignmentsB[i]) {
      return true;
    } else if (r.indexAssignmentsB[i] < l.indexAssignmentsB[i]) {
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
  } else if (r.tensorA < l.tensorA ) {
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

// get size of CobaltDataType
size_t getCobaltDataTypeSize( CobaltDataType type ) {
  switch( type ) {
  case cobaltDataTypeSingle:
    return sizeof(float);
  case cobaltDataTypeDouble:
    return sizeof(double);
  case cobaltDataTypeSingleComplex:
    return 2*sizeof(float);
  case cobaltDataTypeDoubleComplex:
    return 2*sizeof(double);
  default:
    return -1;
  }
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
numElements = 1000*100*10 = 1,000,000
idx0 x100000
idx1 x1000
idx2 x10

idx0 = (serial / 10 / 1000) % 100000
idx1 = (serial / 10) % 1000
idx2 = serial % 10
#endif

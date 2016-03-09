
#include "StructOperations.h"
#include "Solution.h"
#include <assert.h>
#include <string>
#include <stdio.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

namespace Cobalt {



#define COBALT_ENUM_TO_STRING_CASE(X) case X: return #X;
std::string toString( CobaltStatus status ) {
  switch( status ) {

    // success
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusSuccess )
  
  /* cobaltValidateProblem() */
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusProblemIsNull )

  // tensor errors
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorNumDimensionsInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionOrderInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionStrideInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionSizeInvalid )
  
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
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeHalf )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeSingle )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeDouble )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeComplexHalf )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeComplexSingle )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeComplexDouble )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeNone )
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


template<>
std::string tensorElementToString<float> ( float element ) {
  std::ostringstream state;
  state.precision(3);
  state << std::scientific << element;
  return state.str();
}
template<>
std::string tensorElementToString<double> ( double element ) {
  std::ostringstream state;
  state.precision(3);
  state << std::scientific << element;
  return state.str();
}
template<>
std::string tensorElementToString<CobaltComplexFloat> ( CobaltComplexFloat element ) {
  std::ostringstream state;
  state.precision(3);
  state << std::scientific << element.s[0] << ", " << element.s[1];
  return state.str();
}
template<>
std::string tensorElementToString<CobaltComplexDouble> ( CobaltComplexDouble element ) {
  std::ostringstream state;
  state.precision(3);
  state << std::scientific << element.s[0] << ", " << element.s[1];
  return state.str();
}


template<>
std::ostream& appendElement<float>(std::ostream& os, const float& element) {
  os << element;
  return os;
}
template<>
std::ostream& appendElement<double>(std::ostream& os, const double& element) {
  os << element;
  return os;
}
template<>
std::ostream& appendElement<CobaltComplexFloat>(std::ostream& os, const CobaltComplexFloat& element) {
  os << element.s[0] << "," << element.s[1];
  return os;
}
template<>
std::ostream& appendElement<CobaltComplexDouble>(std::ostream& os, const CobaltComplexDouble& element) {
  os << element.s[0] << "," << element.s[1];
  return os;
}


std::string toStringXML( const Cobalt::Solution *solution, size_t indentLevel ) {
  return solution->toString(indentLevel);
}


// get size of CobaltDataType
size_t sizeOf( CobaltDataType type ) {
  switch( type ) {
  case cobaltDataTypeSingle:
    return sizeof(float);
  case cobaltDataTypeDouble:
    return sizeof(double);
  case cobaltDataTypeComplexSingle:
    return sizeof(CobaltComplexFloat);
  case cobaltDataTypeComplexDouble:
    return sizeof(CobaltComplexDouble);
  case cobaltDataTypeNone:
    return 0;
  default:
    return -1;
  }
}





} // namespace

  // CobaltDimension
bool operator<(const CobaltDimension & l, const CobaltDimension & r) {

  if (l.stride > r.stride) {
    return true;
  } else if (r.stride > l.stride) {
    return false;
  }
  if (l.size > r.size) {
    return true;
  } else if (r.size > l.size) {
    return false;
  }
  // identical
  return false;
}

// CobaltControl
bool operator< (const CobaltControl & l, const CobaltControl & r) {
  return l.numDependencies < r.numDependencies;
}

bool operator==(const CobaltDimension & l, const CobaltDimension & r) {
  return l.size == r.size && l.stride == r.stride;
}

bool operator==(const CobaltComplexFloat & l, const CobaltComplexFloat & r) {
  return l.s[0] == r.s[0] && l.s[1] == r.s[1];
}
bool operator==(const CobaltComplexDouble & l, const CobaltComplexDouble & r) {
  return l.s[0] == r.s[0] && l.s[1] == r.s[1];
}

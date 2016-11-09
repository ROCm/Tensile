/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include "StructOperations.h"
#include "Solution.h"
#include <assert.h>
#include <string>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <limits>
#include <cmath>

namespace Cobalt {

#define COBALT_ENUM_TO_STRING_CASE(X) case X: return #X;
std::string toString( CobaltStatus status ) {
  switch( status ) {

    // success
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusSuccess )
  
  // tensor errors
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorNumDimensionsInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionOrderInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionStrideInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusTensorDimensionSizeInvalid )
  
  // operation errors
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperandNumDimensionsMismatch )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationOperandNumIndicesMismatch )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationIndexAssignmentInvalidA )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationIndexAssignmentInvalidB )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationIndexAssignmentDuplicateA )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationIndexAssignmentDuplicateB )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationNumFreeIndicesInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationNumSummationIndicesInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationIndexUnassigned )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusOperationSummationIndexAssignmentsInvalid )

  /* cobaltGetSolution() */
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusDeviceProfileNumDevicesInvalid )
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusDeviceProfileNotSupported)
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusProblemNotSupported )

  /* control errors */
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusControlInvalid )

  /* misc */
  COBALT_ENUM_TO_STRING_CASE( cobaltStatusInvalidParameter )


  // causes clang warning
  //default:
  //  return "Error in toString(CobaltStatus): no switch case for: "
  //      + std::to_string(status);
  };
}

std::string toString( CobaltDataType dataType ) {
  switch( dataType ) {
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeSingle )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeDouble )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeComplexSingle )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeComplexDouble )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeComplexConjugateSingle)
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeComplexConjugateDouble)
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeNone )
    COBALT_ENUM_TO_STRING_CASE( cobaltNumDataTypes )
#ifdef Cobalt_ENABLE_FP16
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeHalf )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeComplexHalf )
    COBALT_ENUM_TO_STRING_CASE( cobaltDataTypeComplexConjugateHalf)
#endif
  //default:
  //  return "Error in toString(CobaltDataType): no switch case for: "
  //      + std::to_string(dataType);
  };
}

std::string toString( CobaltOperationType type ) {
  switch( type ) {
    COBALT_ENUM_TO_STRING_CASE( cobaltOperationTypeContraction )
    COBALT_ENUM_TO_STRING_CASE( cobaltOperationTypeConvolution )
  //default:
  //  return "Error in toString(CobaltDataType): no switch case for: "
  //      + std::to_string(type);
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
  state << std::scientific << element.x << ", " << element.y;
  return state.str();
}
template<>
std::string tensorElementToString<CobaltComplexDouble> ( CobaltComplexDouble element ) {
  std::ostringstream state;
  state.precision(3);
  state << std::scientific << element.x << ", " << element.y;
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
  os << element.x << "," << element.y;
  return os;
}
template<>
std::ostream& appendElement<CobaltComplexDouble>(std::ostream& os, const CobaltComplexDouble& element) {
  os << element.x << "," << element.y;
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
  case cobaltDataTypeComplexConjugateSingle:
    return sizeof(CobaltComplexFloat);
  case cobaltDataTypeComplexConjugateDouble:
    return sizeof(CobaltComplexDouble);
#ifdef Cobalt_ENABLE_FP16
  case cobaltDataTypeHalf:
    return 2;
  case cobaltDataTypeComplexHalf:
    return 4;
  case cobaltDataTypeComplexConjugateHalf:
    return 4;
#endif
  case cobaltNumDataTypes:
    return 0;
  case cobaltDataTypeNone:
    return 0;
  //default:
  //  return -1;
  }
}

size_t flopsPerMadd( CobaltDataType type ) {
  switch( type ) {
  case cobaltDataTypeSingle:
  case cobaltDataTypeDouble:
    return 2;

  case cobaltDataTypeComplexSingle:
  case cobaltDataTypeComplexDouble:
  case cobaltDataTypeComplexConjugateSingle:
  case cobaltDataTypeComplexConjugateDouble:
    return 8;

  case cobaltDataTypeNone:
  case cobaltNumDataTypes:
    return 0;
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
  if (l.validate < r.validate) {
    return true;
  } else if (r.validate < l.validate) {
    return false;
  }
  if (l.benchmark < r.benchmark) {
    return true;
  } else if (r.benchmark < l.benchmark) {
    return false;
  }
#if Cobalt_BACKEND_OPENCL12
  if (l.numQueues < r.numQueues) {
    return true;
  } else if (r.numQueues < l.numQueues) {
    return false;
  }
  for (unsigned int i = 0; i < l.numQueues; i++) {
    if (l.queues[i] < r.queues[i]) {
      return true;
    } else if (r.queues[i] < l.queues[i]) {
      return false;
    }
  }
#endif
  // identical
  return false;
}

bool operator==(const CobaltDimension & l, const CobaltDimension & r) {
  return l.size == r.size && l.stride == r.stride;
}

bool operator==(const CobaltComplexFloat & l, const CobaltComplexFloat & r) {
  return std::fabs(l.x - r.x) < std::numeric_limits<float>::epsilon()
      && std::fabs(l.y - r.y) < std::numeric_limits<float>::epsilon();
}
bool operator==(const CobaltComplexDouble & l, const CobaltComplexDouble & r) {
  return std::fabs(l.x - r.x) < std::numeric_limits<double>::epsilon()
      && std::fabs(l.y - r.y) < std::numeric_limits<double>::epsilon();
}


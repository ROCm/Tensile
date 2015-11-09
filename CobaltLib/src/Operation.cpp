
#include "Operation.h"
#include "Logger.h"

namespace Cobalt {


/*******************************************************************************
 * constructor
 ******************************************************************************/
OperationDescriptor::OperationDescriptor(
    OperationType inputOperationType,
    size_t inputNumDims )
    : type(inputOperationType),
    dimensions(inputNumDims) {
}


/*******************************************************************************
 * comparison operator for stl
 ******************************************************************************/
bool OperationDescriptor::operator< ( const OperationDescriptor & other ) const {
  if (dimensions < other.dimensions) {
    return true;
  } else if (dimensions > other.dimensions) {
    return false;
  }

  if (type < other.type) {
    return true;
  } else if (type > other.type) {
    return false;
  }

  // identical
  return false;
}


bool DimensionPair::operator< ( const DimensionPair & other ) const {
  if (a < other.a) {
    return true;
  } else if (a > other.a) {
    return false;
  }

  if (b < other.b) {
    return true;
  } else if (b > other.b)  {
    return false;
  }

  return false;
}


/*******************************************************************************
 * toString
 ******************************************************************************/
std::string OperationDescriptor::toString( size_t indentLevel ) const {
  std::string state = Logger::indent(indentLevel);
  state += "<" + Logger::operationTag;
  state += " " + Logger::numDimAttr + "=\"" + std::to_string(dimensions.size()) + "\"";
  state += " >\n";
  for (size_t i = 0; i < dimensions.size(); i++) {
    state += dimensions[i].toString(indentLevel+1);
  }
  state += Logger::indent(indentLevel) + "</" + Logger::operationTag + ">\n";
  return state;
}

std::string DimensionPair::toString( size_t indentLevel ) const {
  std::string state = Logger::indent(indentLevel);
  state += "<" + Logger::dimPairTag;
  state += " a=\"" + std::to_string(a) + "\"";
  state += " b=\"" + std::to_string(b) + "\"";
  state += " />\n";
  return state;
}

} // namespace Cobalt
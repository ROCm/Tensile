
#include "Tensor.h"
#include "Logger.h"

namespace Cobalt {
  
/*******************************************************************************
 * constructor - numDims
 ******************************************************************************/
TensorDescriptor::TensorDescriptor( size_t inputNumDims )
    : dimensions(inputNumDims) {
}


/*******************************************************************************
 * constructor - list of sizes
 * - dimensions are auto-populated from compactSizes
 ******************************************************************************/
TensorDescriptor::TensorDescriptor( std::vector<size_t> compactSizes ) {
  std::vector<DimensionDescriptor> localDimensions = compactSizesToDimensions( compactSizes );
  dimensions = localDimensions;
}

/*******************************************************************************
 * comparison operator for stl
 ******************************************************************************/
bool TensorDescriptor::operator< ( const TensorDescriptor & other ) const {
  return dimensions < other.dimensions;
}

bool DimensionDescriptor::operator< ( const DimensionDescriptor & other ) const {
  if (size < other.size) {
    return true;
  } else if (other.size < size) {
    return false;
  }

  if (stride < other.stride) {
    return true;
  } else if (other.stride < stride) {
    return false;
  }

  return false;
}


/*******************************************************************************
 * toString
 ******************************************************************************/
std::string TensorDescriptor::toString( size_t indentLevel ) const {
  std::string state = Logger::indent(indentLevel);
  state += "<" + Logger::tensorTag;
  state += " " + Logger::numDimAttr + "=\"" + std::to_string(dimensions.size()) + "\"";
  state += " >\n";
  for (size_t i = 0; i < dimensions.size(); i++) {
    state += dimensions[i].toString( indentLevel+1);
  }
  state += Logger::indent(indentLevel) + "</" + Logger::tensorTag + ">\n";
  return state;
}

std::string DimensionDescriptor::toString( size_t indentLevel ) const {
  std::string state = Logger::indent(indentLevel);
  state += "<" + Logger::dimensionTag;
  state += " " + Logger::dimNumberAttr + "=\"" + std::to_string(size) + "\"";
  state += " " + Logger::dimStrideAttr + "=\"" + std::to_string(stride) + "\"";
  state += " />\n";
  return state;
}


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


} // namespace Cobalt
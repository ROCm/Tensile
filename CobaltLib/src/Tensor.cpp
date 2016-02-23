#include "Tensor.h"
#include "StructOperations.h"

namespace Cobalt {

/*******************************************************************************
 * Constructor
 ******************************************************************************/
Tensor::Tensor(CobaltTensor tensor)
  : dataType(tensor.dataType),
  dimensions(tensor.dimensions, tensor.dimensions+tensor.numDimensions) { }

/*******************************************************************************
 * toString
 ******************************************************************************/
std::string Tensor::toString() const {
  std::string state = "";
  return state;
}

std::string Tensor::toStringXML( size_t indent ) const {
  std::string state = Cobalt::indent(indent);
  state += "<Tensor numDimensions=\"" + std::to_string(dimensions.size())
      + "\"";
  state += " dataType=\"" + Cobalt::toString( dataType ) + "\"";
  state += " >\n";
  for (size_t i = 0; i < numDims(); i++) {
    state += Cobalt::indent(indent+1) + "<Dimension stride=\""
        + std::to_string(dimensions[i].stride) + "\"";
    state += " size=\"" + std::to_string(dimensions[i].size) + "\" />\n";
  }
  state += Cobalt::indent(indent) + "</Tensor>\n";
  return state;
}

std::string Tensor::toString( CobaltTensorData tensorData ) const {
  switch(dataType) {
  case cobaltDataTypeSingle:
    return toStringTemplate<float>(tensorData);
  case cobaltDataTypeDouble:
    return toStringTemplate<double>(tensorData);
  case cobaltDataTypeComplexSingle:
    return toStringTemplate<CobaltComplexFloat>(tensorData);
  case cobaltDataTypeComplexDouble:
    return toStringTemplate<CobaltComplexDouble>(tensorData);
  case cobaltDataTypeNone:
    return "";
  default:
    return "ERROR";
  }
}

template<typename T>
std::string Tensor::toStringTemplate( CobaltTensorData tensorData ) const {
  std::string state;
  std::vector<unsigned int> coords(numDims());

  for (unsigned int d0 = 0; d0 < dimensions[0].size; d0++) {
    size_t index = getIndex(coords);
    state += tensorElementToString( ((T *)tensorData.data)[index] );
  }
  return state;
}

/*******************************************************************************
 * comparator
 ******************************************************************************/
bool Tensor::operator<(const Tensor & other) const {
  // dataType
  if (dataType < other.dataType) {
    return true;
  } else if (other.dataType < dataType) {
    return false;
  }
  // dimensions
  if (numDims() < other.numDims()) {
    return true;
  } else if (other.numDims() < numDims()) {
    return false;
  }
  for (size_t i = 0; i < numDims(); i++) {
    if (dimensions[i] < other.dimensions[i]) {
      return true;
    } else if (other.dimensions[i] < dimensions[i]) {
      return false;
    }
  }
  // identical
  return false;
}

/*******************************************************************************
 * accessors
 ******************************************************************************/
size_t Tensor::getIndex( std::vector<unsigned int> coords ) const {
  size_t serial = 0;
  for (unsigned int i = 0; i < numDims(); i++) {
    serial += coords[i] * dimensions[i].stride;
  }
  return serial;
}

const CobaltDimension & Tensor::operator[]( size_t index ) const {
  return dimensions[index];
}

unsigned int Tensor::numDims() const {
  return static_cast<unsigned int>(dimensions.size());
}

size_t Tensor::size() const {
  size_t size = 0;
  for (unsigned int i = 0; i < numDims(); i++) {
    size_t dimSize = dimensions[i].size * dimensions[i].stride;
    size = dimSize > size ? dimSize : size;
  }
  return size;
}

CobaltDataType Tensor::getDataType() const { return dataType; }


} // namespace
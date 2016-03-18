#include "Tensor.h"
#include "StructOperations.h"
#include "MathTemplates.h"

#include <sstream>
#include <algorithm>

namespace Cobalt {

/*******************************************************************************
 * Constructor
 ******************************************************************************/
Tensor::Tensor(CobaltTensor tensor)
  : dataType(tensor.dataType),
  dimensions(tensor.dimensions, tensor.dimensions+tensor.numDimensions) {
  if (numDims() < 1 || numDims() > CobaltTensor::maxDimensions) {
    throw cobaltStatusTensorNumDimensionsInvalid;
  }
  for (size_t d = 0; d < numDims(); d++) {
    if (d < numDims() - 1) {
      if (dimensions[d].stride > dimensions[d + 1].stride) {
        throw cobaltStatusTensorDimensionOrderInvalid;
      }
    }
    if (dimensions[d].stride < 1) {
      throw cobaltStatusTensorDimensionStrideInvalid;
    }
    if (dimensions[d].size < 1) {
      throw cobaltStatusTensorDimensionSizeInvalid;
    }
  }
}

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
  std::ostringstream stream;
  std::vector<unsigned int> coords(numDims());
  for (unsigned int i = 0; i < numDims(); i++) {
    coords[i] = 0;
  }
  bool done = false;
  while (!done) { // exit criteria specified below

    // last is a screen row
    for (coords[0] = 0; coords[0] < dimensions[0].size; coords[0]++) {
      size_t index = getIndex(coords);
      stream.setf(std::ios::fixed);
      stream.precision(0);
      stream.width(4);
      appendElement(stream, ((T *)tensorData.data)[index] );
      stream << "; ";
    } // d0
    // append coords
    stream << "(";
    stream << "0:" << dimensions[0].size;
    for (unsigned int d = 1; d < numDims(); d++) {
      stream << ", " << coords[d];
    }
    stream << ")";

    // if 1-dimensional done
    if (coords.size() == 1) {
      done = true;
      break;
    } else { // 2+ dimensions
      bool dimIncremented = false; // for printing extra line
      // increment coord
      coords[1]++;
      for (unsigned int d = 1; d < numDims(); d++) {
        if (coords[d] >= dimensions[d].size) {
          if (d == numDims()-1) {
            // exit criteria - last dimension full
            done = true;
            break;
          }
          dimIncremented = true;
          coords[d] = 0;
          coords[d+1]++;
        }
      }
      // new lines
      stream << std::endl;
      if (dimIncremented && !done) {
        stream << std::endl;
      }

    }
  }

  return stream.str();
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

// descending stride; return change
//std::vector<unsigned int> Tensor::sortDimensions() {
//  auto dimensionsOld = dimensions;
//  std::sort(dimensions.begin(), dimensions.end());
//  std::vector<unsigned int> order;
//  for (unsigned int i = 0; i < numDims(); i++) {
//    auto value = dimensionsOld[i];
//    auto foundIdx = std::find(dimensions.begin(), dimensions.end(), value);
//    size_t idx = foundIdx - dimensions.begin();
//    order.push_back(static_cast<unsigned int>(idx));
//  }
//  return order;
//}

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

size_t Tensor::numElements() const {
  size_t size = 0;
  for (unsigned int i = 0; i < numDims(); i++) {
    size_t dimSize = dimensions[i].size * dimensions[i].stride;
    size = dimSize > size ? dimSize : size;
  }
  return size;
}
size_t Tensor::numBytes() const {
  return numElements() * sizeOf(dataType);
}

CobaltDataType Tensor::getDataType() const { return dataType; }


CobaltTensor Tensor::getTensorStruct() const {
  CobaltTensor simple;
  simple.dataType = dataType;
  simple.numDimensions = static_cast<unsigned int>(numDims());
  for (unsigned int d = 0; d < numDims(); d++) {
    simple.dimensions[d].stride = dimensions[d].stride;
    simple.dimensions[d].size = dimensions[d].size;
  }
  return simple;
}

/* Fill TensorData with values*/
void Tensor::fill(
  CobaltTensorData tensorData,
  FillType fillType,
  void *src) const {
  switch (dataType) {
  case cobaltDataTypeSingle:
    return fillTemplate<float>(tensorData, fillType, src);
  case cobaltDataTypeDouble:
    return fillTemplate<double>(tensorData, fillType, src);
  case cobaltDataTypeComplexSingle:
    return fillTemplate<CobaltComplexFloat>(tensorData, fillType, src);
  case cobaltDataTypeComplexDouble:
    return fillTemplate<CobaltComplexDouble>(tensorData, fillType, src);
  case cobaltDataTypeNone:
    return;
  default:
    return;
  }
}

template<typename T>
void Tensor::fillTemplate(
    CobaltTensorData tensorData,
    FillType fillType,
    void *srcVoid) const {

  T *src = static_cast<T*>(srcVoid);

  std::vector<unsigned int> coords(numDims());
  for (unsigned int i = 0; i < numDims(); i++) {
    coords[i] = 0;
  }
  bool done = false;
  size_t srcIdx = 0;

  while (!done) { // exit criteria specified below

    for (coords[0] = 0; coords[0] < dimensions[0].size; coords[0]++) {
      size_t index = getIndex(coords);
      switch(fillType) {
      case fillTypeZero:
        static_cast<T*>(tensorData.data)[index] = Cobalt::getZero<T>();
        break;
      case fillTypeOne:
        static_cast<T*>(tensorData.data)[index] = Cobalt::getOne<T>();
        break;
      case fillTypeRandom:
        static_cast<T*>(tensorData.data)[index] = Cobalt::getRandom<T>();
        break;
      case fillTypeCopy:
        static_cast<T*>(tensorData.data)[index] = src[srcIdx++];
        break;
      }
    } // d0

    // if 1-dimensional done
    if (coords.size() == 1) {
      done = true;
      break;
    } else { // 2+ dimensions
      bool dimIncremented = false; // for printing extra line
                                   // increment coord
      coords[1]++;
      for (unsigned int d = 1; d < numDims(); d++) {
        if (coords[d] >= dimensions[d].size) {
          if (d == numDims() - 1) {
            // exit criteria - last dimension full
            done = true;
            break;
          }
          dimIncremented = true;
          coords[d] = 0;
          coords[d + 1]++;
        }
      }
    }
  }
}

} // namespace
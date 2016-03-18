#ifndef TENSOR_H
#define TENSOR_H

#include "Cobalt.h"
#include <string>
#include <vector>

namespace Cobalt {

/*******************************************************************************
 * Tensor
 ******************************************************************************/
class Tensor {
  friend class Problem;
public:
  typedef enum { fillTypeZero, fillTypeOne, fillTypeRandom, fillTypeCopy } FillType;
  Tensor( CobaltTensor tensor );
  unsigned int numDims() const;
  std::string toString() const;
  std::string toStringXML(size_t indent) const;
  bool operator<(const Tensor & other) const;
  size_t getIndex( std::vector<unsigned int> coords ) const;
  size_t numElements() const;
  size_t numBytes() const;
  CobaltDataType getDataType() const;
  const CobaltDimension & operator[]( size_t index ) const;
  //std::vector<unsigned int> sortDimensions();
  CobaltTensor getTensorStruct() const;
  void fill(
    CobaltTensorData tensorData,
    FillType type,
    void *src) const;
  template<typename T>
  void fillTemplate(
      CobaltTensorData tensorData,
      FillType type,
      void *src) const;

  template<typename T>
  std::string toStringTemplate( CobaltTensorData tensorData ) const;

  std::string toString( CobaltTensorData tensorData ) const;

protected:
  CobaltDataType dataType;
  std::vector<CobaltDimension> dimensions;

};

} // namespace

#endif
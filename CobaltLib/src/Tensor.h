/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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
  typedef enum { fillTypeZero, fillTypeOne, fillTypeRandom, fillTypeIndex, fillTypeCopy } FillType;
  Tensor( CobaltTensor tensor );
  unsigned int numDims() const;
  std::string toString() const;
  std::string toStringXML(size_t indent, std::string which) const;
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
  std::string toStringTemplate( CobaltTensorDataConst tensorData ) const;

  std::string toString( CobaltTensorDataConst tensorData ) const;

protected:
  CobaltDataType dataType;
  std::vector<CobaltDimension> dimensions;

};


template<typename DataType>
bool compareTensorsTemplate(
  DataType *gpuData,
  DataType *cpuData,
  Cobalt::Tensor tensor);


bool compareTensors(
  CobaltTensorDataConst gpu,
  CobaltTensorDataConst cpu,
  Cobalt::Tensor tensor,
  CobaltControl ctrl);


template<typename DataType>
void printMismatch(size_t index, DataType gpuData, DataType cpuData);

template<typename DataType>
void printMatch(size_t index, DataType gpuData, DataType cpuData);

} // namespace

#endif

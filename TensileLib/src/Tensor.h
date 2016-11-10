/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef TENSOR_H
#define TENSOR_H

#include "Tensile.h"
#include <string>
#include <vector>

namespace Tensile {

/*******************************************************************************
 * Tensor
 ******************************************************************************/
class Tensor {
  friend class Problem;
public:
  typedef enum { fillTypeZero, fillTypeOne, fillTypeRandom, fillTypeIndex, fillTypeCopy } FillType;
  Tensor( TensileTensor tensor );
  unsigned int numDims() const;
  std::string toString() const;
  std::string toStringXML(size_t indent, std::string which) const;
  bool operator<(const Tensor & other) const;
  size_t getIndex( std::vector<unsigned int> coords ) const;
  size_t numElements() const;
  size_t numBytes() const;
  TensileDataType getDataType() const;
  const TensileDimension & operator[]( size_t index ) const;
  //std::vector<unsigned int> sortDimensions();
  TensileTensor getTensorStruct() const;
  void fill(
    TensileTensorData tensorData,
    FillType type,
    void *src) const;
  template<typename T>
  void fillTemplate(
      TensileTensorData tensorData,
      FillType type,
      void *src) const;

  template<typename T>
  std::string toStringTemplate( TensileTensorDataConst tensorData ) const;

  std::string toString( TensileTensorDataConst tensorData ) const;

protected:
  TensileDataType dataType;
  std::vector<TensileDimension> dimensions;

};


template<typename DataType>
bool compareTensorsTemplate(
  DataType *gpuData,
  DataType *cpuData,
  Tensile::Tensor tensor);


bool compareTensors(
  TensileTensorDataConst gpu,
  TensileTensorDataConst cpu,
  Tensile::Tensor tensor );


template<typename DataType>
void printMismatch(size_t index, DataType gpuData, DataType cpuData);

template<typename DataType>
void printMatch(size_t index, DataType gpuData, DataType cpuData);

} // namespace

#endif


/*******************************************************************************
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
* ies of the Software, and to permit persons to whom the Software is furnished
* to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
* PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
* FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
* COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
* IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
* CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*******************************************************************************/

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


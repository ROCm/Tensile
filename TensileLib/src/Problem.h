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

#ifndef PROBLEM_H
#define PROBLEM_H

#include "Tensile.h"
#include "Tensor.h"
#include "DeviceProfile.h"

#include <vector>

namespace Tensile {

/*******************************************************************************
 * Problem
 ******************************************************************************/
class Problem {
  friend class Solution;
public:
  Problem(
    TensileTensor tensorC,
    TensileTensor tensorA,
    TensileTensor tensorB,
    unsigned int *indexAssignmentsA,
    unsigned int *indexAssignmentsB,
    TensileOperationType operationType,
    TensileDataType alphaType,
    TensileDataType betaType,
    bool useOffsets,
    TensileDeviceProfile deviceProfile );
  bool useAlpha() const;
  bool useBeta() const;
  bool deviceIsReference() const;
  TensileStatus validate();
  std::string toString() const;
  std::string toStringXML( size_t indentLevel ) const;
  std::string toStringOperationXML( size_t indentLevel ) const;
  TensileDataType getDataTypeC() const;
  TensileDataType getDataTypeA() const;
  TensileDataType getDataTypeB() const;
  TensileDataType getDataTypeAlpha() const;
  TensileDataType getDataTypeBeta() const;
  size_t alphaSize() const;
  size_t betaSize() const;
  bool operator<( const Problem & other ) const;
  bool sortIndicesC( unsigned int i, unsigned int j) const; // may need when matching index orders
  bool sortSummationIndexDescending( std::pair<unsigned int, unsigned int> i, std::pair<unsigned int, unsigned int> j) const; // may need for matching index orders
  size_t getNumFlops();

//protected: // leave public since Solution classes need to access them and friendship isn't inherited
  Tensor tensorC;
  Tensor tensorA;
  Tensor tensorB;
  TensileOperationType operationType;
  TensileDataType alphaType;
  TensileDataType betaType;
  bool useOffsets;
  DeviceProfile deviceProfile;
  std::vector<unsigned int> indicesFree;
  std::vector<unsigned int> indicesBatch;
  std::vector<std::pair<unsigned int,unsigned int>> indicesSummation;
  std::vector<unsigned int> indicesA;
  std::vector<unsigned int> indicesB;
  size_t numFlops;
};

} // namespace


/*******************************************************************************
 * TensileProblem - public pimpl
 ******************************************************************************/
struct _TensileProblem {
  Tensile::Problem *pimpl;
};

#endif


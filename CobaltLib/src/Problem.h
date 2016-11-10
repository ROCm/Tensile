/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

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


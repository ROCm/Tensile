#ifndef PROBLEM_H
#define PROBLEM_H

#include "Cobalt.h"
#include "Tensor.h"
#include "DeviceProfile.h"

#include <vector>

namespace Cobalt {

/*******************************************************************************
 * Problem
 ******************************************************************************/
class Problem {
  friend class Solution;
public:
  Problem(
    CobaltTensor tensorC,
    CobaltTensor tensorA,
    CobaltTensor tensorB,
    unsigned int *indexAssignmentsA,
    unsigned int *indexAssignmentsB,
    CobaltOperationType operationType,
    CobaltDataType alphaType,
    CobaltDataType betaType,
    CobaltDeviceProfile deviceProfile );
  bool useAlpha() const;
  bool useBeta() const;
  bool deviceIsReference() const;
  CobaltStatus validate();
  std::string toString() const;
  std::string toStringXML( size_t indentLevel ) const;
  std::string toStringOperationXML( size_t indentLevel ) const;
  CobaltDataType getDataTypeC() const;
  CobaltDataType getDataTypeA() const;
  CobaltDataType getDataTypeB() const;
  CobaltDataType getDataTypeAlpha() const;
  CobaltDataType getDataTypeBeta() const;
  size_t alphaSize() const;
  size_t betaSize() const;
  bool operator<( const Problem & other ) const;
  bool sortIndicesC( unsigned int i, unsigned int j) const;
  bool sortSummationIndexDescending( std::pair<unsigned int, unsigned int> i, std::pair<unsigned int, unsigned int> j) const;

//protected: // leave public since Solution classes need to access them and friendship isn't inherited
  Tensor tensorC;
  Tensor tensorA;
  Tensor tensorB;
  CobaltOperationType operationType;
  CobaltDataType alphaType;
  CobaltDataType betaType;
  DeviceProfile deviceProfile;
  std::vector<unsigned int> indicesFree;
  std::vector<unsigned int> indicesBatch;
  std::vector<std::pair<unsigned int,unsigned int>> indicesSummation;
  std::vector<unsigned int> indicesA;
  std::vector<unsigned int> indicesB;
  CobaltStatus constructorStatus;


};

} // namespace


/*******************************************************************************
 * CobaltProblem - public pimpl
 ******************************************************************************/
struct _CobaltProblem {
  Cobalt::Problem *pimpl;
};

#endif
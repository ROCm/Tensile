#ifndef PROBLEM_H
#define PROBLEM_H

#include "Cobalt.h"
#include "CobaltCpp.h"

#include <vector>

namespace Cobalt {
  
/*******************************************************************************
 * Tensor
 ******************************************************************************/
class Tensor {
  friend class Problem;
public:
  Tensor( CobaltTensor tensor );
  unsigned int numDims() const;
  std::string toString() const;
  std::string toStringXML(size_t indent) const;

protected:
  CobaltDataType dataType;
  std::vector<CobaltDimension> dimensions;

};


/*******************************************************************************
 * DeviceProfile
 ******************************************************************************/
class Device {
  friend class DeviceProfile;
  friend class Problem;
public:
  Device( CobaltDevice device );
  void init( CobaltDevice device );
  std::string toStringXML(size_t indent) const;

protected:
  std::string name;
  unsigned int numComputeUnits;
  unsigned int clockFrequency;
};

class DeviceProfile {
  friend class Problem;
public:
  DeviceProfile( CobaltDeviceProfile profile );
  std::string toStringXML(size_t indent) const;
protected:
  std::vector<Device> devices;
};


/*******************************************************************************
 * Problem
 ******************************************************************************/
class Problem {
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

protected:
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
  bool invalidationCaughtInConstructor;


};

} // namespace

/*******************************************************************************
 * CobaltProblem - public pimpl
 ******************************************************************************/
struct _CobaltProblem {
  Cobalt::Problem *pimpl;
};

#endif
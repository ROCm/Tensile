

#pragma once

#include <string>
#include <vector>

namespace Cobalt {
/*******************************************************************************
 * DeviceDescriptor
 * - same proviles can use same compiled kernel(s) and solution
 * - describe device(s) to xml log so similar devices can use same profile
 ******************************************************************************/
class DeviceDescriptor {
public:
  
  std::string deviceName;
  size_t numComputeUnits;
  size_t clockFrequency; // MHz
  
 /*******************************************************************************
 * constructor
 * - initializes members to zero/null
 * - user must manually override as much as it can
 ******************************************************************************/
  DeviceDescriptor();

 /*******************************************************************************
 * comparison operator for STL
 ******************************************************************************/
  bool operator< ( const DeviceDescriptor & other ) const;
  
 /*******************************************************************************
 * toString for writing xml
 ******************************************************************************/
  std::string toString( size_t indentLevel ) const;

}; // class Device


 /*******************************************************************************
 * DeviceProfile
 * - same proviles can use same compiled kernel(s) and solution
 * - describe device(s) to xml log so similar devices can use same profile
 ******************************************************************************/
class DeviceProfile {
public:
  
  std::vector<DeviceDescriptor> devices;
  
 /*******************************************************************************
 * constructor
 * - initializes members to zero/null
 * - user must manually override as much as it can
 ******************************************************************************/
  DeviceProfile(size_t numDevices);

 /*******************************************************************************
 * comparison operator for STL
 ******************************************************************************/
  bool operator< ( const DeviceProfile & other ) const;
  
 /*******************************************************************************
 * toString for writing xml
 ******************************************************************************/
  std::string toString( size_t indentLevel ) const;

}; // class DeviceProfile

} // namespace Cobalt
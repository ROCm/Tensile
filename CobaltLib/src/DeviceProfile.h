#ifndef DEVICE_PROFILE_H
#define DEVICE_PROFILE_H

#include "Cobalt.h"
#include <string>
#include <vector>

namespace Cobalt {
  
/*******************************************************************************
 * Device
 ******************************************************************************/
class Device {
  friend class DeviceProfile;
  friend class Problem;
public:
  Device( CobaltDevice device );
  Device();
  void init( CobaltDevice device );
  std::string toStringXML(size_t indent) const;
  bool operator<( const Device & other ) const;

protected:
  std::string name;
  unsigned int numComputeUnits;
  unsigned int clockFrequency;
};


/*******************************************************************************
 * DeviceProfile
 ******************************************************************************/
class DeviceProfile {
  friend class Problem;
public:
  DeviceProfile( CobaltDeviceProfile profile );
  std::string toStringXML(size_t indent) const;
  bool operator<( const DeviceProfile & other ) const;
protected:
  std::vector<Device> devices;
};

} // namespace

#endif
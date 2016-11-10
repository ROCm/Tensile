/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef DEVICE_PROFILE_H
#define DEVICE_PROFILE_H

#include "Tensile.h"
#include <string>
#include <vector>

namespace Tensile {
  
/*******************************************************************************
 * Device
 ******************************************************************************/
class Device {
  friend class DeviceProfile;
  friend class Problem;
public:
  Device( TensileDevice device );
  Device();
  void init( TensileDevice device );
  std::string toStringXML(size_t indent) const;
  bool operator<( const Device & other ) const;
  bool matches( std::string name ) const;

protected:
  std::string name;
  unsigned int numComputeUnits;
  unsigned int clockFrequency;
  unsigned int flopsPerClock;
};


/*******************************************************************************
 * DeviceProfile
 ******************************************************************************/
class DeviceProfile {
  friend class Problem;
public:
  DeviceProfile( TensileDeviceProfile profile );
  std::string toStringXML(size_t indent) const;
  bool operator<( const DeviceProfile & other ) const;
  const Device & operator[]( size_t index ) const;
  size_t numDevices() const;
protected:
  std::vector<Device> devices;
};

} // namespace

#endif


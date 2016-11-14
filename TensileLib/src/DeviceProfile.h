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


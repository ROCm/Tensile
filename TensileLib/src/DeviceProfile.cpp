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

#include "DeviceProfile.h"
#include "StructOperations.h"
#include <sstream>
namespace Tensile {

/*******************************************************************************
 * Device - constructors
 ******************************************************************************/
Device::Device( TensileDevice device )
  : name(device.name) { }

Device::Device()
  : name("uninitialized"),
  numComputeUnits(0),
  clockFrequency(0),
  flopsPerClock(2*64) { }

void Device::init( TensileDevice device ) {
  name.assign(device.name);
  numComputeUnits = device.numComputeUnits;
  clockFrequency = device.clockFrequency;
  flopsPerClock = device.flopsPerClock;
}


bool Device::matches( std::string inputName ) const {
  return name == inputName;
}

/*******************************************************************************
 * Device - toString
 ******************************************************************************/
std::string Device::toStringXML( size_t indentLevel ) const {
  std::string state = indent(indentLevel);
  state += "<Device name=\"";
  state += name;
  state += "\"";
  state += " />\n";
  return state;
}

/*******************************************************************************
 * Device - comparator
 ******************************************************************************/
bool Device::operator< ( const Device & other ) const {
  return name < other.name;
}



/*******************************************************************************
 * DeviceProfile - constructor
 ******************************************************************************/
DeviceProfile::DeviceProfile( TensileDeviceProfile profile)
  : devices(profile.numDevices) {
  for (unsigned int i = 0; i < profile.numDevices; i++) {
    devices[i].init(profile.devices[i]);
  }
}

/*******************************************************************************
 * DeviceProfile - toString
 ******************************************************************************/
std::string DeviceProfile::toStringXML( size_t indent ) const {
  std::string state = Tensile::indent(indent);
  state += "<DP";
  state += " n=\"" + std::to_string(devices.size()) + "\"";
  for (size_t i = 0; i < devices.size(); i++) {
    state += " d" + std::to_string(i) + "=\"" + devices[i].name + "\"";
    state += " CU" + std::to_string(i) + "=\"" + std::to_string(devices[i].numComputeUnits) + "\"";
    state += " MHz" + std::to_string(i) + "=\"" + std::to_string(devices[i].clockFrequency) + "\"";
    state += " FPC" + std::to_string(i) + "=\"" + std::to_string(devices[i].flopsPerClock) + "\"";
  }
  state += " />\n";
  return state;
}

/*******************************************************************************
 * DeviceProfile - comparator
 ******************************************************************************/
bool DeviceProfile::operator<(const DeviceProfile & other) const {
  if (devices.size() < other.devices.size()) {
    return true;
  } else if (other.devices.size() < devices.size()) {
    return false;
  }
  for (size_t i = 0; i < devices.size(); i++) {
    if (devices[i] < other.devices[i]) {
      return true;
    } else if (other.devices[i] < devices[i]) {
      return false;
    }
  }
  // identical
  return false;
}


const Device & DeviceProfile::operator[]( size_t index ) const {
  return devices[index];
}

size_t DeviceProfile::numDevices() const {
  return devices.size();
}

} // namespace


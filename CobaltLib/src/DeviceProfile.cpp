#include "DeviceProfile.h"
#include "StructOperations.h"

namespace Cobalt {

/*******************************************************************************
 * Device - constructors
 ******************************************************************************/
Device::Device( CobaltDevice device )
  : name(device.name),
  numComputeUnits(device.numComputeUnits),
  clockFrequency(device.clockFrequency) { }

Device::Device()
  : name("uninitialized"),
  numComputeUnits(0),
  clockFrequency(0) { }

void Device::init( CobaltDevice device ) {
  name.assign(device.name);
  numComputeUnits = device.numComputeUnits;
  clockFrequency = device.clockFrequency;
}

/*******************************************************************************
 * Device - toString
 ******************************************************************************/
std::string Device::toStringXML( size_t indentLevel ) const {
  std::string state = indent(indentLevel);
  state += "<Device name=\"";
  state += name;
  state += "\"";
  state += " numComputeUnits=\"" + std::to_string(numComputeUnits) + "\"";
  state += " clockFrequency=\"" + std::to_string(clockFrequency) + "\"";
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
DeviceProfile::DeviceProfile( CobaltDeviceProfile profile)
  : devices(profile.numDevices) {
  for (unsigned int i = 0; i < profile.numDevices; i++) {
    devices[i].init(profile.devices[i]);
  }
}

/*******************************************************************************
 * DeviceProfile - toString
 ******************************************************************************/
std::string DeviceProfile::toStringXML( size_t indent ) const {
  std::string state = Cobalt::indent(indent);
  state += "<DeviceProfile";
  state += " numDevices=\"" + std::to_string(devices.size())
      + "\" >\n";
  for (size_t i = 0; i < devices.size(); i++) {
    state += devices[i].toStringXML(indent+1);
  }
  state += Cobalt::indent(indent) + "</DeviceProfile>\n";
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

} // namespace

#include "Device.h"
#include "Logger.h"

namespace Cobalt {

  
/*******************************************************************************
 * constructor - default
 ******************************************************************************/
Device::Device()
    : deviceName("unknown"),
    numComputeUnits(0),
    clockFrequency(0) {
  // nothing else
}


/*******************************************************************************
 * comparison operator for stl
 * - TODO
 ******************************************************************************/
bool Device::operator< ( const Device & other ) const {
  return deviceName < other.deviceName;
}


std::string Device::toString( size_t indentLevel ) const {
  std::string state = Logger::indent(indentLevel);
  state += "<" + Logger::deviceTag;
  state += " " + Logger::nameAttr + "=\"" + deviceName + "\"";
  state += " numCUs=\"" + std::to_string(numComputeUnits) + "\"";
  state += " clockFreq=\"" + std::to_string(clockFrequency) + "\"";
  state += " />\n";
  return state;
}


/*******************************************************************************
 * constructor - default
 ******************************************************************************/
DeviceProfile::DeviceProfile( size_t numDevices )
    : devices(numDevices) {
  // nothing else
}



/*******************************************************************************
 * comparison operator for stl
 ******************************************************************************/
bool DeviceProfile::operator< ( const DeviceProfile & other ) const {
  return devices < other.devices;
}


std::string DeviceProfile::toString( size_t indentLevel ) const {
  std::string state = Logger::indent(indentLevel);
  state += "<" + Logger::deviceProfileTag;
  state += " numDevices=\"" + std::to_string(devices.size()) + "\" >\n";
  for (size_t i = 0; i < devices.size(); i++) {
    state += devices[i].toString(indentLevel+1);
  }
  state += Logger::indent(indentLevel) + "</" + Logger::deviceProfileTag + ">\n";
  return state;
}

} // namespace Cobalt
#ifndef DEVICE_STATS
#define DEVICE_STATS

#if Tensile_RUNTIME_LANGUAGE_HIP

#include "TensileTypes.h"
#include <dirent.h>
#include <fstream>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#define WARNING(msg) std::cerr << "[!WARNING!] " << msg << std::endl;
#define FATAL(msg)                                                             \
  {                                                                            \
    std::cerr << "[!FATAL!] " << msg << std::endl;                             \
    exit(EXIT_FAILURE);                                                        \
  }                                                                            \
  while (0)
#define INFO(msg) std::cout << "[INFO] " << msg << std::endl;

namespace tensile {

std::vector<std::string> split(const std::string &str, char sep) {
  std::vector<std::string> strings;
  std::istringstream f(str);
  std::string s;
  while (std::getline(f, s, sep)) {
    strings.push_back(s);
  }
  return strings;
}

std::vector<std::string> lsDir(const std::string &dname) {
  std::vector<std::string> files;
  struct dirent *entry;
  DIR *dir = opendir(dname.c_str());
  if (dir == NULL) {
    return files;
  }

  while ((entry = readdir(dir)) != NULL) {
    std::string fname(entry->d_name);
    if (fname != "." && fname != "..")
      files.push_back(fname);
  }
  return files;
}

std::vector<std::string> lsDir(const std::string &dname,
                               const std::regex &match) {
  std::vector<std::string> files;
  struct dirent *entry;
  DIR *dir = opendir(dname.c_str());
  if (dir == NULL) {
    return files;
  }

  while ((entry = readdir(dir)) != NULL) {
    std::string fname(entry->d_name);
    if (fname != "." && fname != "..") {
      if (std::regex_match(fname, match)) {
        files.push_back(fname);
      }
    }
  }
  return files;
}

int readCurrentMhz(const std::string &fname) {
  std::ifstream f(fname);
  std::string line;
  while (std::getline(f, line)) {
    if (line.back() == '*') {
      std::string mhzstr = line.substr(3, line.size() - 3 - 5);
      std::istringstream iss(mhzstr);
      int mhz;
      iss >> mhz;
      return mhz;
    }
  }
  return -1;
}

struct Device {
  int hipId;
  hipDeviceProp_t hipProps;
  std::string drmPath;
  std::string hwmonPath;

  /// Find the sysfs paths given that the device is initalized by
  /// hipGetDeviceProperties
  /// The paths are found using hiDeviceProp_t::pciBusID
  void initSysPaths() {
    bool found = false;
    for (std::string cardname :
         lsDir("/sys/class/drm", std::regex("card\\d+"))) {
      std::string carddir = "/sys/class/drm/" + cardname;
      std::string fname = carddir + "/device/uevent";
      std::ifstream f(fname);
      if (f.good() && f.is_open()) {
        std::string line;
        while (std::getline(f, line)) {
          if (split(line, '=')[0] == "PCI_SLOT_NAME") {
            std::string pciids = split(line, '=')[1];
            std::vector<std::string> ids = split(pciids, ':');
            // std::string busid = "0x" + ids[1];
            int pciBusId = std::stoul("0x" + ids[1], nullptr, 16);
            if (pciBusId == hipProps.pciBusID) {
              drmPath = carddir;
              // find hwmon path
              std::vector<std::string> hwpaths =
                  lsDir(drmPath + "/device/hwmon", std::regex("hwmon\\d+"));
              if (hwpaths.size() != 1) {
                WARNING("No or multiple hwmon paths for " << drmPath);
              }
              if (hwpaths.size() > 0) {
                hwmonPath = drmPath + "/device/hwmon/" + hwpaths[0];
              }
              found = true;
            }
            break;
          }
        }
      }
      if (found)
        break;
    }
    if (!found) {
      WARNING("Can't find sysfs path for device " << hipId);
    }
  }

  void printInfo() {
    // print out device info
    INFO("Device " << hipId << ": " << hipProps.name);
    INFO("\tArch:\t" << hipProps.gcnArch)
    INFO("\tGMem:\t" << hipProps.totalGlobalMem / 1024 / 1024 << " MiB");
    INFO("\twarps:\t" << hipProps.warpSize);
    INFO("\tCUs:\t" << hipProps.multiProcessorCount);
    INFO("\tMaxClk:\t" << hipProps.clockRate);
    INFO("\tMemClk:\t" << hipProps.memoryClockRate);
    INFO("\tdrm:\t" << drmPath);
    INFO("\thwmon:\t" << hwmonPath);
    // INFO("\t\tpciDomainID:\t" << hipProps.pciDomainID);
    // INFO("\t\tpciBusID:\t" << hipProps.pciBusID);
    // INFO("\t\tpciDeviceID:\t" << hipProps.pciDeviceID);
  }

  float getTemp() {
    std::ifstream f(hwmonPath + "/temp1_input");
    // std::ifstream f("/sys/class/hwmon/hwmon0/temp1_input");
    int temp;
    f >> temp;
    return temp / 1000.f; // temp is in milli celsius
  }

  int getFanSpeed() {
    // std::ifstream f("/sys/class/hwmon/hwmon0/pwm1");
    std::ifstream f(hwmonPath + "/pwm1");
    int fan;
    f >> fan;
    return fan;
  }

  int getCoreClock() { return readCurrentMhz(drmPath + "/device/pp_dpm_sclk"); }

  int getMemClock() { return readCurrentMhz(drmPath + "/device/pp_dpm_mclk"); }
};

struct Devices {
  static std::vector<Device> &getDevices(bool fromInit = false) {
    static bool isInit = false;
    static std::vector<Device> d;
    if (!isInit) {
      isInit = true;
      if (!fromInit)
        initDevices();
    }
    return d;
  }

  static Device &getDefaultDevice() {
    if (getDevices().size() == 0) {
      FATAL("No HIP Devices available.");
    }
    return getDevices()[0];
  }

  static void initDevices() {
    int devcount;
    tensileStatusCheck(hipGetDeviceCount(&devcount));
    // INFO("Number of HIP devices found: " << devcount);

    if (devcount == 0) {
      FATAL("No HIP devices found.");
    }

    std::vector<Device> &devs = getDevices(true);
    devs.resize(devcount);

    // init and get devices
    for (int d = 0; d < devcount; ++d) {
      devs[d].hipId = d;
      tensileStatusCheck(
          hipGetDeviceProperties(&devs[d].hipProps, d /*deviceID*/));
      devs[d].initSysPaths();
      // devs[d].printInfo();
    }
  }
};
} // namespace tensile
#endif

void tensileInitDeviceStats() {
#if Tensile_RUNTIME_LANGUAGE_HIP
  tensile::Devices::initDevices();
#endif
}

float tensileGetDeviceTemp(unsigned int deviceId) {
#if Tensile_RUNTIME_LANGUAGE_HIP
  return tensile::Devices::getDevices()[deviceId].getTemp();
#else
  return -1;
#endif
}

int tensileGetDeviceFanSpeed(unsigned int deviceId) {
#if Tensile_RUNTIME_LANGUAGE_HIP
  return tensile::Devices::getDevices()[deviceId].getFanSpeed();
#else
  return -1;
#endif
}

int tensileGetDeviceCoreClock(unsigned int deviceId) {
#if Tensile_RUNTIME_LANGUAGE_HIP
  return tensile::Devices::getDevices()[deviceId].getCoreClock();
#else
  return -1;
#endif
}

int tensileGetDeviceMemClock(unsigned int deviceId) {
#if Tensile_RUNTIME_LANGUAGE_HIP
  return tensile::Devices::getDevices()[deviceId].getMemClock();
#else
  return -1;
#endif
}

#endif // DEVICE_STATS

#include "Client.h"
#include <string>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <iomanip>

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {
  if (sizeof(size_t) != 8) {
    std::cout << "WARNING: Executable not 64-bit." << std::endl;
  }

  // parse command line parameters
  parseCommandLineParameters(argc, argv);

  // init runtime controls
  initControls();

  // init data
  unsigned int dataTypeIdx = 0;
  DataTypeEnum dataTypeEnum = dataTypeEnums[dataTypeIdx];
  switch(dataTypeEnum) {
  case enum_float:
    float *initialC;
    float *initialA;
    float *initialB;
    float alpha;
    float beta;
    float *referenceC;
    float *deviceOnHostC;
    initData(&initialC, &initialA, &initialB, &alpha, &beta, &referenceC,
        &deviceOnHostC);
#if Tensile_CLIENT_BENCHMARK
    benchmarkProblemSizes(initialC, initialA, initialB, alpha, beta,
        referenceC, deviceOnHostC);
#else
    callLibrary(initialC, initialA, initialB, alpha, beta, referenceC,
        deviceOnHostC);
#endif
    destroyData(initialC, initialA, initialB, referenceC, deviceOnHostC);
    break;
  }

  // cleanup
  destroyControls();

} // main



/*******************************************************************************
 * init controls
 ******************************************************************************/
void initControls() {
#if Tensile_BACKEND_OCL
  // setup opencl objects
  cl_uint numPlatforms, numDevices;
  status = clGetPlatformIDs(0, nullptr, &numPlatforms);
  tensileStatusCheck(status);
  cl_platform_id *platforms = new cl_platform_id[numPlatforms];
  status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
  tensileStatusCheck(status);
  platform = platforms[platformIdx];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  tensileStatusCheck(status);
  cl_device_id *devices = new cl_device_id[numDevices];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
  tensileStatusCheck(status);
  device = devices[deviceIdx];
  size_t nameLength;
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, 0, nullptr, &nameLength );
  tensileStatusCheck(status);
  char *deviceName = new char[nameLength+1];
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, nameLength, deviceName, 0 );
  tensileStatusCheck(status);
  std::cout << "Device: \"" << deviceName << "\"" << std::endl;
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
  tensileStatusCheck(status);
  stream = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  tensileStatusCheck(status);
  delete[] devices;
  delete[] platforms;
#elif Tensile_BACKEND_HIP
  int numDevices;
  status = hipGetDeviceCount( &numDevices );
  tensileStatusCheck(status);
  status = hipSetDevice( deviceIdx );
  tensileStatusCheck(status);
  status = hipStreamCreate( &stream );
  tensileStatusCheck(status);
#endif

}


/*******************************************************************************
 * destroy controls
 ******************************************************************************/
void destroyControls() {
#if Tensile_BACKEND_OCL
  clReleaseCommandQueue(stream);
  clReleaseContext(context);
  clReleaseDevice(device);
#else
  hipStreamDestroy(stream);
#endif
}


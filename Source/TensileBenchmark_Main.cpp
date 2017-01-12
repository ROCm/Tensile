#include "TensileBenchmark_Main.h"
#include <string>
#include <cstring>
#include <cstdio>

// {rand, 1, index}
unsigned int dataInitType = 0;

/*******************************************************************************
 * main
 ******************************************************************************/
int main( int argc, char *argv[] ) {

  initControls();

  initData();



  return 0;
} // main


/*******************************************************************************
 * init controls
 ******************************************************************************/
void initControls() {
#if Tensile_BACKEND_OCL
  // setup opencl objects
  status = clGetPlatformIDs(0, nullptr, &numPlatforms);
  cl_platform_id *platforms = new cl_platform_id[numPlatforms];
  status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
  platform = platforms[platformIdx];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
  cl_device_id *devices = new cl_device_id[numDevices];
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
  device = devices[deviceIdx];
  size_t nameLength;
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, 0, nullptr, &nameLength );
  char *deviceName = new char[nameLength+1];
  status = clGetDeviceInfo( device, CL_DEVICE_NAME, nameLength, deviceName, 0 );
  printf("Using device \"%s\"\n", deviceName);
  context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
  stream = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
#elif Tensile_BACKEND_HIP
  status = hipGetDeviceCount( &numDevices );
  status = hipSetDevice( deviceIdx );
  status = hipStreamCreate( &stream );
#endif

  delete[] devices;
  delete[] platforms;
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


/*******************************************************************************
 * initialize data
 ******************************************************************************/
void initData() {
  // initial and reference buffers
  referenceC = new DataType[maxSizeC];
#if Tensile_BACKEND_OCL
  initialC = new DataType[maxSizeC];
  initialA = new DataType[maxSizeA];
  initialB = new DataType[maxSizeB];
#else
  status = hipMalloc( &initialC, sizeMaxC );
  status = hipMalloc( &initialA, sizeMaxA );
  status = hipMalloc( &initialB, sizeMaxB );
#endif

  // initialize buffers
  if (dataInitType == 0) {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = static_cast<DataType>(rand() % 10);
    }
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = static_cast<DataType>(rand() % 10);
    }
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = static_cast<DataType>(rand() % 10);
    }
  } else if (dataInitType == 0) {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = tensileGetOne<DataType>();
    }
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = tensileGetOne<DataType>();
    }
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = tensileGetOne<DataType>();
    }
  } else {
    for (size_t i = 0; i < maxSizeC; i++) {
      initialC[i] = static_cast<DataType>(i);
    }
    for (size_t i = 0; i < maxSizeA; i++) {
      initialA[i] = static_cast<DataType>(i);
    }
    for (size_t i = 0; i < maxSizeB; i++) {
      initialB[i] = static_cast<DataType>(i);
    }
  }
#if Tensile_BACKEND_OCL
  deviceC = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeMaxC, NULL, &status);
  deviceA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeMaxA, NULL, &status);
  deviceB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeMaxB, NULL, &status);
#else
#endif
}


/*******************************************************************************
 * destroy data
 ******************************************************************************/
void destroyData() {
  delete[] initialC;
  delete[] initialA;
  delete[] initialB;
  delete[] referenceC;

#if Tensile_BACKEND_OCL
  clReleaseMemObject(deviceC);
  clReleaseMemObject(deviceA);
  clReleaseMemObject(deviceB);
#else
  hipFree(deviceC);
#endif

}

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

#include "Tensile.h"
#include "Problem.h"
#include "Solution.h"
#include "Logger.h"
#include "SolutionTensorContractionCPU.h"

#include <assert.h>
#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>

#if Tensile_SOLVER_ENABLED
#include "TensileGetSolution.h"
#endif

// global logger object
Tensile::Logger Tensile::logger;


/*******************************************************************************
 * tensileSetup()
 ******************************************************************************/
TensileStatus tensileSetup( const char *logFilePath ) {
  Tensile::logger.init(logFilePath);
  return tensileStatusSuccess;
}


/*******************************************************************************
 * tensileTeardown
 ******************************************************************************/
TensileStatus tensileTeardown() {

#if Tensile_BACKEND_OPENCL12
  // delete kernels
  if (kernelMap) {
    unsigned int index = 0;
    for ( KernelMap::iterator i = kernelMap->begin(); i != kernelMap->end(); i++) {
      // printf("releasing kernel %u\n", index);
      clReleaseKernel(i->second);
      index++;
    }
    delete kernelMap;
    kernelMap = nullptr;
  }
#endif
  Tensile::logger.close();
  return tensileStatusSuccess;
}


/*******************************************************************************
* tensileCreateEmptyTensor
* - returns TensileTensor initialized to zero
******************************************************************************/
TensileTensor tensileCreateEmptyTensor() {
  TensileTensor tensor;
  tensor.dataType = tensileDataTypeNone;
  tensor.numDimensions = 0;
  for (int i = 0; i < TensileTensor::maxDimensions; i++) {
    tensor.dimensions[i].size = 0;
    tensor.dimensions[i].stride = 0;
  }
  return tensor;
}


/*******************************************************************************
* tensileCreateDeviceProfile
* returns TensileDeviceProfile initialized to zero
******************************************************************************/
TensileDeviceProfile tensileCreateEmptyDeviceProfile() {
  TensileDeviceProfile profile;
  profile.numDevices = 0;
  for (int i = 0; i < TensileDeviceProfile::maxDevices; i++) {
    profile.devices[i].name[0] = '\0';
  }
  return profile;
}


/*******************************************************************************
 * tensileCreateDeviceProfile
 * returns TensileDeviceProfile initialized to zero
 ******************************************************************************/
TensileStatus tensileEnumerateDeviceProfiles(
    TensileDeviceProfile *profiles,
    unsigned int *size) {

  static std::vector<TensileDeviceProfile> enumeratedProfiles;
  static bool profilesEnumerated = false;

  if (!profilesEnumerated) {
#if Tensile_SOLVER_ENABLED
    enumerateDeviceProfilesSupported(enumeratedProfiles);
#else

#if Tensile_BACKEND_OPENCL12
    //printf("tensileEnumerateDeviceProfiles(OpenCL)\n");
    cl_int status;
    cl_uint numPlatforms;
    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    cl_platform_id *platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    for (unsigned int p = 0; p < numPlatforms; p++) {
      cl_uint numDevices;
      status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
      cl_device_id *devices = new cl_device_id[numDevices];
      status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
      for (unsigned int d = 0; d < numDevices; d++) {
        TensileDeviceProfile profile = tensileCreateEmptyDeviceProfile();
        profile.numDevices = 1;
        status = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, profile.devices[0].maxNameLength, profile.devices[0].name, 0);
        Tensile::makeFileNameSafe( profile.devices[0].name );
        status = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(profile.devices[0].clockFrequency), &profile.devices[0].clockFrequency, 0);
        status = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(profile.devices[0].numComputeUnits), &profile.devices[0].numComputeUnits, 0);
        enumeratedProfiles.push_back(profile);
        //printf("  Device[%u/%u][%u/%u]: %s %u CUs @ %u MHz (%.0f GFlop/s)\n", p, numPlatforms, d, numDevices, profile.devices[0].name, profile.devices[0].numComputeUnits, profile.devices[0].clockFrequency, 1.0*profile.devices[0].numComputeUnits*profile.devices[0].clockFrequency*profile.devices[0].flopsPerClock/1000.f);
      }
      delete[] devices;
    }
    delete[] platforms;
#else
    hipError_t status;
    int numDevices;
    status = hipGetDeviceCount( &numDevices );
    for (int i = 0; i < numDevices; i++) {
      hipDeviceProp_t deviceProperties;
      hipGetDeviceProperties( &deviceProperties, i);
      TensileDeviceProfile profile = tensileCreateEmptyDeviceProfile();
      profile.numDevices = 1;
      strcpy( profile.devices[0].name, deviceProperties.name );
      Tensile::makeFileNameSafe( profile.devices[0].name );
      profile.devices[0].numComputeUnits = deviceProperties.multiProcessorCount;
      profile.devices[0].clockFrequency = deviceProperties.clockRate/1000; // kHz -> MHz
      enumeratedProfiles.push_back(profile);
    }
#endif

#endif
    profilesEnumerated = true;
  }

  if (profiles) {
    // do copy
    if (size) { // copy up to size
      size_t lengthToCopy = *size < enumeratedProfiles.size() ? *size : enumeratedProfiles.size();
      std::memcpy(profiles, &enumeratedProfiles[0], lengthToCopy*sizeof(TensileDeviceProfile));
    } else { // copy all
      std::memcpy(profiles, &enumeratedProfiles[0], enumeratedProfiles.size()*sizeof(TensileDeviceProfile));
    }
  } else {
    // just return size
    if (size) {
      *size = static_cast<unsigned int>(enumeratedProfiles.size());
    } else {
      // can't do anything
      return tensileStatusInvalidParameter;
    }
  }
  return tensileStatusSuccess;
}

/*******************************************************************************
* tensileCreateControl
* returns TensileControl initialized to zero
******************************************************************************/
TensileControl tensileCreateEmptyControl() {
  TensileControl control;
  control.validate = nullptr;
  control.benchmark = 0;
#if Tensile_BACKEND_OPENCL12
  control.numQueues = 0;
  control.numQueuesUsed = 0;
  control.numInputEvents = 0;
  control.numOutputEvents = 0;
  control.inputEvents = nullptr;
  control.outputEvents = nullptr;
  for (int i = 0; i < TensileControl::maxQueues; i++) {
    control.queues[i] = nullptr;
  }
#endif
  return control;
}


/*******************************************************************************
 * tensileCreateProblem
 ******************************************************************************/
TensileStatus tensileCreateProblem(
    TensileProblem *problem,
    TensileTensor tensorC,
    TensileTensor tensorA,
    TensileTensor tensorB,
    unsigned int *indexAssignmentsA,
    unsigned int *indexAssignmentsB,
    TensileOperationType operationType,
    TensileDataType alphaType,
    TensileDataType betaType,
    bool useOffsets,
    TensileDeviceProfile deviceProfile ) {

  try {
    Tensile::Problem *problemPtr = new Tensile::Problem(
        tensorC,
        tensorA,
        tensorB,
        indexAssignmentsA,
        indexAssignmentsB,
        operationType,
        alphaType,
        betaType,
        useOffsets,
        deviceProfile );
    (*problem) = new _TensileProblem();
    (*problem)->pimpl = problemPtr;
    return tensileStatusSuccess;
  } catch ( TensileStatus errorStatus ) {
    (*problem) = nullptr;
    return errorStatus;
  }

};

TensileStatus tensileDestroyProblem( TensileProblem problem ) {
  if (problem) {
    //printf("tensileDestroyProblem:: problem=%p\n", problem);
    if (problem->pimpl) {
      //printf("tensileDestroyProblem:: problem->pimpl=%p\n", problem->pimpl);
      delete problem->pimpl;
      delete problem;
      problem = nullptr;
      return tensileStatusSuccess;
    }
  }
  return tensileStatusInvalidParameter;
}


TensileStatus tensileValidateProblem( TensileProblem problem ) {
  if (problem == nullptr) {
    return tensileStatusInvalidParameter;
  } else {
    return problem->pimpl->validate();
  }
}


/*******************************************************************************
 * tensileGetSolutionForProblem
 ******************************************************************************/
TensileStatus tensileGetSolutionForProblem(
    TensileSolution *solution,
    TensileProblem problem ) {

  TensileStatus status;

  Tensile::Logger::TraceEntry entry;
  entry.type = Tensile::Logger::TraceEntryType::getSolution;
  (*solution) = new _TensileSolution();
  // cpu device
  if ( problem->pimpl->deviceIsReference() ) {
    std::tie((*solution)->pimpl, status) = Tensile::getSolutionCPU( *(problem->pimpl) );

  // gpu device
  } else {
#if Tensile_SOLVER_ENABLED
    (*solution)->pimpl = getSolutionTop( *(problem->pimpl), &status );
#else
    (*solution)->pimpl = new Tensile::SolutionLogOnly<void,void,void,void,void>(*(problem->pimpl));
    status = tensileStatusSuccess;
#endif
  }

  entry.solution = (*solution)->pimpl;
  entry.status = status;

#if Tensile_LOGGER_ENABLED
  Tensile::logger.log(entry);
#endif

  return status;
}

TensileStatus tensileDestroySolution(TensileSolution solution) {
  if (solution) {
    if (solution->pimpl) {
      delete solution->pimpl;
      delete solution;
      solution = nullptr;
      return tensileStatusSuccess;
    } else {
      return tensileStatusInvalidParameter;
    }
  }
  else {
    return tensileStatusInvalidParameter;
  }
}

/*******************************************************************************
 * tensileEnqueueSolution
 ******************************************************************************/
TensileStatus tensileEnqueueSolution(
    TensileSolution solution,
    TensileTensorData tensorDataC,
    TensileTensorDataConst tensorDataA,
    TensileTensorDataConst tensorDataB,
    TensileScalarData alpha,
    TensileScalarData beta,
    TensileControl *ctrl ) {
  TensileStatus status = tensileStatusSuccess;
  // if cpu device, enqueue even if solver turned off
  if (solution->pimpl->getProblem().deviceIsReference()) {
      status = solution->pimpl->enqueue( tensorDataC, tensorDataA, tensorDataB,
          alpha, beta, *ctrl );

  // gpu device, only enqueue if solver turned on
  } else {
#if Tensile_SOLVER_ENABLED
    status = solution->pimpl->enqueueEntry( tensorDataC, tensorDataA, tensorDataB,
        alpha, beta, *ctrl, false /*no print*/ );
#endif
  }
  return status;
}


/*******************************************************************************
 * toStrings
 ******************************************************************************/
TensileStatus cppStringToCString(
  std::string state, char *cstr, unsigned int *size ) {
  if (cstr) {
    // do copy
    if (size) {
      // copy up to size
      size_t lengthToCopy = *size-1 < state.size() ? *size-1 : state.size();
      std::memcpy(cstr, state.c_str(), lengthToCopy);
      cstr[lengthToCopy] = '\0';
    } else {
      // copy all
      std::memcpy(cstr, state.c_str(), state.size());
      cstr[state.size()] = '\0';
    }
  } else {
    // just return size
    if (size) {
      // return size
      *size = (unsigned int) (state.size()+1); // include space for null char
    } else {
      // can't do anything
      return tensileStatusInvalidParameter;
    }
  }
  return tensileStatusSuccess;
}

TensileStatus tensileStatusToString( TensileStatus code, char *cstr, unsigned int *size ) {
  std::string state = Tensile::toString(code);
  return cppStringToCString( state, cstr, size);
}

TensileStatus tensileProblemToString(
    TensileProblem problem, char *cstr, unsigned int *size ) {
  std::string state = problem->pimpl->toString();
  return cppStringToCString( state, cstr, size );
}

TensileStatus tensileSolutionToString(
    TensileSolution solution, char *cstr, unsigned int *size ) {
  std::string state = solution->pimpl->toString(0);
  return cppStringToCString( state, cstr, size );
}


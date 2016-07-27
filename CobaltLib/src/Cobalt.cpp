#include "Cobalt.h"
#include "Problem.h"
#include "Solution.h"
#include "Logger.h"
#include "SolutionTensorContractionCPU.h"

#include <assert.h>
#include <cstdio>
#include <string>
#include <cstring>
#include <algorithm>

#if Cobalt_SOLVER_ENABLED
#include "CobaltGetSolution.h"
#endif

// global logger object
Cobalt::Logger Cobalt::logger;


/*******************************************************************************
 * cobaltSetup()
 ******************************************************************************/
CobaltStatus cobaltSetup( const char *logFilePath ) {
  Cobalt::logger.init(logFilePath);
  return cobaltStatusSuccess;
}


/*******************************************************************************
 * cobaltTeardown
 ******************************************************************************/
CobaltStatus cobaltTeardown() {

#if Cobalt_BACKEND_OPENCL12
  // delete kernels
  if (kernelMap) {
    unsigned int index = 0;
    for ( KernelMap::iterator i = kernelMap->begin(); i != kernelMap->end(); i++) {
      printf("releasing kernel %u\n", index);
      clReleaseKernel(i->second);
      index++;
    }
    delete kernelMap;
    kernelMap = nullptr;
  }
#endif
  Cobalt::logger.close();
  return cobaltStatusSuccess;
}


/*******************************************************************************
* cobaltCreateEmptyTensor
* - returns CobaltTensor initialized to zero
******************************************************************************/
CobaltTensor cobaltCreateEmptyTensor() {
  CobaltTensor tensor;
  tensor.dataType = cobaltDataTypeNone;
  tensor.numDimensions = 0;
  for (int i = 0; i < CobaltTensor::maxDimensions; i++) {
    tensor.dimensions[i].size = 0;
    tensor.dimensions[i].stride = 0;
  }
  return tensor;
}


/*******************************************************************************
* cobaltCreateDeviceProfile
* returns CobaltDeviceProfile initialized to zero
******************************************************************************/
CobaltDeviceProfile cobaltCreateEmptyDeviceProfile() {
  CobaltDeviceProfile profile;
  profile.numDevices = 0;
  for (int i = 0; i < CobaltDeviceProfile::maxDevices; i++) {
    profile.devices[i].name[0] = '\0';
  }
  return profile;
}


/*******************************************************************************
 * cobaltCreateDeviceProfile
 * returns CobaltDeviceProfile initialized to zero
 ******************************************************************************/
CobaltStatus cobaltEnumerateDeviceProfiles(
    CobaltDeviceProfile *profiles,
    unsigned int *size) {

  // TODO - this will leak memory upon closing application
  static std::vector<CobaltDeviceProfile> enumeratedProfiles;
  static bool profilesEnumerated = false;

  if (!profilesEnumerated) {
#if Cobalt_SOLVER_ENABLED
    // TODO - enumerate devices supported by backend, rather than all devices on system
#else

#if Cobalt_BACKEND_OPENCL12
    //printf("cobaltEnumerateDeviceProfiles(OpenCL)\n");
    cl_int status;
    cl_uint numPlatforms;
    status = clGetPlatformIDs(0, nullptr, &numPlatforms);
    cl_platform_id *platforms = new cl_platform_id[numPlatforms];
    status = clGetPlatformIDs(numPlatforms, platforms, nullptr);
    unsigned int numProfiles = 0;
    for (unsigned int p = 0; p < numPlatforms; p++) {
      cl_uint numDevices;
      status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &numDevices);
      cl_device_id *devices = new cl_device_id[numDevices];
      status = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_GPU, numDevices, devices, nullptr);
      for (unsigned int d = 0; d < numDevices; d++) {
        CobaltDeviceProfile profile = cobaltCreateEmptyDeviceProfile();
        profile.numDevices = 1;
        status = clGetDeviceInfo(devices[d], CL_DEVICE_NAME, profile.devices[0].maxNameLength, profile.devices[0].name, 0);
        status = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(profile.devices[0].clockFrequency), &profile.devices[0].clockFrequency, 0);
        status = clGetDeviceInfo(devices[d], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(profile.devices[0].numComputeUnits), &profile.devices[0].numComputeUnits, 0);
        enumeratedProfiles.push_back(profile);
        //printf("  Device[%u/%u][%u/%u]: %s %u CUs @ %u MHz (%.0f GFlop/s)\n", p, numPlatforms, d, numDevices, profile.devices[0].name, profile.devices[0].numComputeUnits, profile.devices[0].clockFrequency, 1.0*profile.devices[0].numComputeUnits*profile.devices[0].clockFrequency*profile.devices[0].flopsPerClock/1000.f);
      }
      delete[] devices;
    }
    delete[] platforms;
#else
    // TODO
#endif

#endif
    profilesEnumerated = true;
  }

  if (profiles) {
    // do copy
    if (size) { // copy up to size
      size_t lengthToCopy = *size < enumeratedProfiles.size() ? *size : enumeratedProfiles.size();
      std::memcpy(profiles, &enumeratedProfiles[0], lengthToCopy*sizeof(CobaltDeviceProfile));
    } else { // copy all
      std::memcpy(profiles, &enumeratedProfiles[0], enumeratedProfiles.size()*sizeof(CobaltDeviceProfile));
    }
  } else {
    // just return size
    if (size) {
      *size = static_cast<unsigned int>(enumeratedProfiles.size());
    } else {
      // can't do anything
      return cobaltStatusInvalidParameter;
    }
  }
  return cobaltStatusSuccess;
}

/*******************************************************************************
* cobaltCreateControl
* returns CobaltControl initialized to zero
******************************************************************************/
CobaltControl cobaltCreateEmptyControl() {
  CobaltControl control;
  control.validate = nullptr;
  control.benchmark = 0;
#if Cobalt_BACKEND_OPENCL12
  control.numQueues = 0;
  control.numQueuesUsed = 0;
  control.numInputEvents = 0;
  control.numOutputEvents = 0;
  control.inputEvents = nullptr;
  control.outputEvents = nullptr;
  for (int i = 0; i < CobaltControl::maxQueues; i++) {
    control.queues[i] = nullptr;
  }
#endif
  return control;
}


/*******************************************************************************
 * cobaltCreateProblem
 ******************************************************************************/
CobaltStatus cobaltCreateProblem(
    CobaltProblem *problem,
    CobaltTensor tensorC,
    CobaltTensor tensorA,
    CobaltTensor tensorB,
    unsigned int *indexAssignmentsA,
    unsigned int *indexAssignmentsB,
    CobaltOperationType operationType,
    CobaltDataType alphaType,
    CobaltDataType betaType,
    bool useOffsets,
    CobaltDeviceProfile deviceProfile ) {

  try {
    Cobalt::Problem *problemPtr = new Cobalt::Problem(
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
    (*problem) = new _CobaltProblem();
    (*problem)->pimpl = problemPtr;
    return cobaltStatusSuccess;
  } catch ( CobaltStatus errorStatus ) {
    (*problem) = nullptr;
    return errorStatus;
  }

};

CobaltStatus cobaltDestroyProblem( CobaltProblem problem ) {
  if (problem) {
    //printf("cobaltDestroyProblem:: problem=%p\n", problem);
    if (problem->pimpl) {
      //printf("cobaltDestroyProblem:: problem->pimpl=%p\n", problem->pimpl);
      delete problem->pimpl;
      delete problem;
      problem = nullptr;
      return cobaltStatusSuccess;
    }
  }
  return cobaltStatusInvalidParameter;
}


CobaltStatus cobaltValidateProblem( CobaltProblem problem ) {
  if (problem == nullptr) {
    return cobaltStatusInvalidParameter;
  } else {
    return problem->pimpl->validate();
  }
}


/*******************************************************************************
 * cobaltGetSolutionForProblem
 ******************************************************************************/
CobaltStatus cobaltGetSolutionForProblem(
    CobaltSolution *solution,
    CobaltProblem problem ) {

  CobaltStatus status;

  Cobalt::Logger::TraceEntry entry;
  entry.type = Cobalt::Logger::TraceEntryType::getSolution;
  (*solution) = new _CobaltSolution();
  // cpu device
  if ( problem->pimpl->deviceIsReference() ) {
    std::tie((*solution)->pimpl, status) = Cobalt::getSolutionCPU( *(problem->pimpl) );

  // gpu device
  } else {
#if Cobalt_SOLVER_ENABLED
    (*solution)->pimpl = getSolutionTop( *(problem->pimpl), &status );
#else
    (*solution)->pimpl = new Cobalt::SolutionLogOnly<void,void,void,void,void>(*(problem->pimpl));
    status = cobaltStatusSuccess;
#endif
  }

  entry.solution = (*solution)->pimpl;
  entry.status = status;

#if Cobalt_LOGGER_ENABLED
  Cobalt::logger.log(entry);
#endif

  return status;
}

CobaltStatus cobaltDestroySolution(CobaltSolution solution) {
  if (solution) {
    if (solution->pimpl) {
      delete solution->pimpl;
      delete solution;
      solution = nullptr;
      return cobaltStatusSuccess;
    } else {
      return cobaltStatusInvalidParameter;
    }
  }
  else {
    return cobaltStatusInvalidParameter;
  }
}

/*******************************************************************************
 * cobaltEnqueueSolution
 ******************************************************************************/
CobaltStatus cobaltEnqueueSolution(
    CobaltSolution solution,
    CobaltTensorData tensorDataC,
    CobaltTensorDataConst tensorDataA,
    CobaltTensorDataConst tensorDataB,
    CobaltScalarData alpha,
    CobaltScalarData beta,
    CobaltControl *ctrl ) {
  CobaltStatus status = cobaltStatusSuccess;
  // if cpu device, enqueue even if solver turned off
  if (solution->pimpl->getProblem().deviceIsReference()) {
      status = solution->pimpl->enqueueEntry( tensorDataC, tensorDataA, tensorDataB,
          alpha, beta, *ctrl, false /*no print*/ );

  // gpu device, only enqueue if solver turned on
  } else {
#if Cobalt_SOLVER_ENABLED
    status = solution->pimpl->enqueueEntry( tensorDataC, tensorDataA, tensorDataB,
        alpha, beta, *ctrl, false /*no print*/ );
#endif
  }
  return status;
}


/*******************************************************************************
 * toStrings
 ******************************************************************************/
CobaltStatus cppStringToCString(
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
      return cobaltStatusInvalidParameter;
    }
  }
  return cobaltStatusSuccess;
}

CobaltStatus cobaltStatusToString( CobaltStatus code, char *cstr, unsigned int *size ) {
  std::string state = Cobalt::toString(code);
  return cppStringToCString( state, cstr, size);
}

CobaltStatus cobaltProblemToString(
    CobaltProblem problem, char *cstr, unsigned int *size ) {
  std::string state = problem->pimpl->toString();
  return cppStringToCString( state, cstr, size );
}

CobaltStatus cobaltSolutionToString(
    CobaltSolution solution, char *cstr, unsigned int *size ) {
  std::string state = solution->pimpl->toString(0);
  return cppStringToCString( state, cstr, size );
}


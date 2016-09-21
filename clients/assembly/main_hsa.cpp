////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2016, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include <sstream>
#include <cassert>
#include "hsa.h"
#include <string>
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <time.h>

#define FULL_VALIDATION 0

double total_time_ms;

void sgemm_cpu(
    bool transA,
    bool transB,
    float  *C,
    float  *A,
    float  *B,
    float const alpha,
    float const beta,
    unsigned int const ldc,
    unsigned int const lda,
    unsigned int const ldb,
    unsigned int const M,
    unsigned int const N,
    unsigned int const K ) {
  std::cout << "computing refC=A*B on host" << std::endl;

  for (unsigned int i = 0; i < M; i++) {
    for (unsigned int j = 0; j < N; j++) {
      float c = 0.f;
      for (unsigned int k = 0; k < K; k++) {
        float a = transA ? A[k+i*lda] : A[i+k*lda];
        float b = transB ? B[j+k*ldb] : B[k+j*ldb];
        c += a*b;
        //if (i==17 && j==17) {
        //  std::cout << c << " = " << a << " * " << b << " + " << (c-a*b) << std::endl;
        //}
      }
      size_t cIdx = i+j*ldc;
      C[cIdx] = alpha*c + beta*C[cIdx];
    }
  }
}


namespace amd {
namespace dispatch {

class Buffer {
private:
  size_t size;
  void *local_ptr, *system_ptr;

public:
  Buffer(size_t size_, void *local_ptr_, void *system_ptr_)
    : size(size_), local_ptr(local_ptr_), system_ptr(system_ptr_) { }
  Buffer(size_t size_, void *system_ptr_)
    : size(size_), local_ptr(system_ptr_), system_ptr(system_ptr_) { }
  void *LocalPtr() const { return local_ptr; }
  void *SystemPtr() { return system_ptr; }
  template <typename T>
  T* Ptr() { return (T*) system_ptr; }
  template <typename T>
  const T& Data(size_t i) const { return ((const T*) system_ptr)[i]; }
  template <typename T>
  T& Data(size_t i) { return ((T*) system_ptr)[i]; }
  bool IsLocal() const { return local_ptr != system_ptr; }
  size_t Size() const { return size; }
};

class Dispatch {
private:
  hsa_agent_t agent;
  hsa_agent_t cpu_agent;
  uint32_t queue_size;
  hsa_queue_t* queue;
  hsa_signal_t signal;
  hsa_region_t system_region;
  hsa_region_t kernarg_region;
  hsa_region_t local_region;
  hsa_kernel_dispatch_packet_t* aql;
  uint64_t packet_index;
  void *kernarg;
  size_t kernarg_offset;
  hsa_code_object_t code_object;
  hsa_executable_t executable;

  timespec startTime;

  bool Init();
  bool InitDispatch();
  bool RunDispatch();
  bool Wait();

protected:
  std::ostringstream output;
  bool Error(const char *msg);
  bool HsaError(const char *msg, hsa_status_t status = HSA_STATUS_SUCCESS);

public:
  Dispatch(int argc, const char** argv);

  void SetAgent(hsa_agent_t agent) { assert(!this->agent.handle); this->agent = agent; }
  bool HasAgent() const { return agent.handle != 0; }
  void SetCpuAgent(hsa_agent_t agent) { assert(!this->cpu_agent.handle); this->cpu_agent = agent; }
  bool HasCpuAgent() const { return cpu_agent.handle != 0; }
  void SetWorkgroupSize(uint16_t sizeX, uint16_t sizeY = 1, uint16_t sizeZ = 1);
  void SetGridSize(uint32_t sizeX, uint32_t sizeY = 1, uint32_t sizeZ = 1);
  void SetSystemRegion(hsa_region_t region);
  void SetKernargRegion(hsa_region_t region);
  void SetLocalRegion(hsa_region_t region);
  bool AllocateKernarg(uint32_t size);
  bool Run();
  int RunMain();
  virtual bool SetupExecutable();
  virtual bool SetupCodeObject();
  bool LoadCodeObjectFromFile(const std::string& filename);
  void* AllocateLocalMemory(size_t size);
  void* AllocateSystemMemory(size_t size);
  bool CopyToLocal(void* dest, void* src, size_t size);
  bool CopyFromLocal(void* dest, void* src, size_t size);
  Buffer* AllocateBuffer(size_t size);
  bool CopyTo(Buffer* buffer);
  bool CopyFrom(Buffer* buffer);
  virtual bool Setup() { return true; }
  virtual bool Verify() { return true; }
  void KernargRaw(const void* ptr, size_t size, size_t align);

  template <typename T>
  void Kernarg(const T* ptr, size_t size = sizeof(T), size_t align = sizeof(T)) {
    KernargRaw(ptr, size, align);
  }

  void Kernarg(Buffer* buffer);
  uint64_t GetTimestampFrequency();
};


Dispatch::Dispatch(int argc, const char** argv)
  : queue_size(0),
    queue(0)
{
  agent.handle = 0;
  cpu_agent.handle = 0;
  signal.handle = 0;
  kernarg_region.handle = 0;
  system_region.handle = 0;
  local_region.handle = 0;
}

hsa_status_t find_gpu_device(hsa_agent_t agent, void *data)
{
  if (data == NULL) { return HSA_STATUS_ERROR_INVALID_ARGUMENT; }

  hsa_device_type_t hsa_device_type;
  hsa_status_t hsa_error_code = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &hsa_device_type);
  if (hsa_error_code != HSA_STATUS_SUCCESS) { return hsa_error_code; }

  if (hsa_device_type == HSA_DEVICE_TYPE_GPU) {
    Dispatch* dispatch = static_cast<Dispatch*>(data);
    if (!dispatch->HasAgent()) {
      dispatch->SetAgent(agent);
    }
  }

  if (hsa_device_type == HSA_DEVICE_TYPE_CPU) {
    Dispatch* dispatch = static_cast<Dispatch*>(data);
    if (!dispatch->HasCpuAgent()) {
      dispatch->SetCpuAgent(agent);
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t FindRegions(hsa_region_t region, void* data)
{
  hsa_region_segment_t segment_id;
  hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, &segment_id);

  if (segment_id != HSA_REGION_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  hsa_region_global_flag_t flags;
  hsa_region_get_info(region, HSA_REGION_INFO_GLOBAL_FLAGS, &flags);
  //std::cout << "FindRegions: " << flags << std::endl;

  Dispatch* dispatch = static_cast<Dispatch*>(data);

  if (flags & HSA_REGION_GLOBAL_FLAG_FINE_GRAINED) {
    dispatch->SetSystemRegion(region);
  }

  if (flags & HSA_REGION_GLOBAL_FLAG_COARSE_GRAINED) {
    //std::cout << "  Coarse:" << flags << std::endl;
    dispatch->SetLocalRegion(region);
  }

  if (flags & HSA_REGION_GLOBAL_FLAG_KERNARG) {
    dispatch->SetKernargRegion(region);
  }

  return HSA_STATUS_SUCCESS;
}

bool Dispatch::HsaError(const char* msg, hsa_status_t status)
{
  const char* err = 0;
  if (status != HSA_STATUS_SUCCESS) {
    hsa_status_string(status, &err);
  }
  output << msg << ": " << (err ? err : "unknown error") << std::endl;
  return false;
}

bool Dispatch::Init()
{
  hsa_status_t status;
  status = hsa_init();
  if (status != HSA_STATUS_SUCCESS) { return HsaError("hsa_init failed", status); }

  // Find GPU
  status = hsa_iterate_agents(find_gpu_device, this);
  assert(status == HSA_STATUS_SUCCESS);

  char agent_name[64];
  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, agent_name);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("hsa_agent_get_info(HSA_AGENT_INFO_NAME) failed", status); }
  //output << "Using agent: " << agent_name << std::endl;

  status = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("hsa_agent_get_info(HSA_AGENT_INFO_QUEUE_MAX_SIZE) failed", status); }

  status = hsa_queue_create(agent, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("hsa_queue_create failed", status); }

  status = hsa_signal_create(1, 0, NULL, &signal);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("hsa_signal_create failed", status); }

  status = hsa_agent_iterate_regions(agent, FindRegions, this);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("Failed to iterate memory regions", status); }
  if (!kernarg_region.handle) { return HsaError("Failed to find kernarg memory region"); }

  return true;
}

bool Dispatch::InitDispatch()
{
  const uint32_t queue_mask = queue->size - 1;
  packet_index = hsa_queue_add_write_index_relaxed(queue, 1);
  aql = (hsa_kernel_dispatch_packet_t*) (hsa_kernel_dispatch_packet_t*)(queue->base_address) + (packet_index & queue_mask);
  memset((uint8_t*)aql + 4, 0, sizeof(*aql) - 4);
  aql->completion_signal = signal;
  aql->workgroup_size_x = 1;
  aql->workgroup_size_y = 1;
  aql->workgroup_size_z = 1;
  aql->grid_size_x = 1;
  aql->grid_size_y = 1;
  aql->grid_size_z = 1;
  aql->group_segment_size = 65536/2 - 512; // TODO
  aql->private_segment_size = 0;
  return true;
}

bool Dispatch::RunDispatch()
{
  //std::cout << "RunDispatch()" << std::endl;
  uint16_t header =
    (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
    (1 << HSA_PACKET_HEADER_BARRIER) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
    (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE);
  uint16_t setup = 2 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
  uint32_t header32 = header | (setup << 16);
    __atomic_store_n((uint32_t*)aql, header32, __ATOMIC_RELEASE);
  // Ring door bell
  hsa_signal_store_relaxed(queue->doorbell_signal, packet_index);

  clock_gettime( CLOCK_REALTIME, &startTime );

  return true;
}

void Dispatch::SetWorkgroupSize(uint16_t sizeX, uint16_t sizeY, uint16_t sizeZ)
{
  aql->workgroup_size_x = sizeX;
  aql->workgroup_size_y = sizeY;
  aql->workgroup_size_z = sizeZ;
}

void Dispatch::SetGridSize(uint32_t sizeX, uint32_t sizeY, uint32_t sizeZ)
{
  aql->grid_size_x = sizeX;
  aql->grid_size_y = sizeY;
  aql->grid_size_z = sizeZ;
}

void Dispatch::SetSystemRegion(hsa_region_t region)
{
  system_region = region;
}

void Dispatch::SetKernargRegion(hsa_region_t region)
{
  kernarg_region = region;
}

void Dispatch::SetLocalRegion(hsa_region_t region)
{
#if 1
  // choose 1st region
  if (local_region.handle == 0) { local_region = region; }
#else
  // choose 2nd region
  local_region = region;
#endif
}

bool Dispatch::AllocateKernarg(uint32_t size)
{
  hsa_status_t status;
  status = hsa_memory_allocate(kernarg_region, size, &kernarg);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("Failed to allocate kernarg", status); }
  aql->kernarg_address = kernarg;
  kernarg_offset = 0;
  return true;
}

bool Dispatch::LoadCodeObjectFromFile(const std::string& filename)
{
  std::ifstream in(filename.c_str(), std::ios::binary | std::ios::ate);
  if (!in) { output << "Error: failed to load " << filename << std::endl; return false; }
  size_t size = std::string::size_type(in.tellg());
  char *ptr = (char*) AllocateSystemMemory(size);
  if (!ptr) {
    output << "Error: failed to allocate memory for code object." << std::endl;
    return false;
  }
  in.seekg(0, std::ios::beg);
  std::copy(std::istreambuf_iterator<char>(in),
            std::istreambuf_iterator<char>(),
            ptr);
/*
  res.assign((std::istreambuf_iterator<char>(in)),
              std::istreambuf_iterator<char>());

*/
  hsa_status_t status = hsa_code_object_deserialize(ptr, size, NULL, &code_object);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("Failed to deserialize code object", status); }
  return true;
}

bool Dispatch::SetupCodeObject()
{
  return false;
}

bool Dispatch::SetupExecutable()
{
  hsa_status_t status;
  hsa_executable_symbol_t kernel_symbol;

  if (!SetupCodeObject()) { return false; }
  status = hsa_executable_create(HSA_PROFILE_FULL, HSA_EXECUTABLE_STATE_UNFROZEN,
                                 NULL, &executable);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("hsa_executable_create failed", status); }

  // Load code object
  status = hsa_executable_load_code_object(executable, agent, code_object, NULL);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("hsa_executable_load_code_object failed", status); }

  // Freeze executable
  status = hsa_executable_freeze(executable, NULL);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("hsa_executable_freeze failed", status); }

  // Get symbol handle
  //status = hsa_executable_get_symbol(executable, NULL, "mad2d", agent, 0, &kernel_symbol);
  status = hsa_executable_get_symbol(executable, NULL, "sgemm_NT", agent, 0, &kernel_symbol);

  if (status != HSA_STATUS_SUCCESS) { return HsaError("hsa_executable_get_symbol failed", status); }

  // Get code handle
  uint64_t code_handle;
  status = hsa_executable_symbol_get_info(kernel_symbol,
                                          HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                          &code_handle);
  if (status != HSA_STATUS_SUCCESS) { return HsaError("hsa_executable_symbol_get_info failed", status); }

  aql->kernel_object = code_handle;

  return true;
}

uint64_t TIMEOUT = 120;

bool Dispatch::Wait()
{
  //std::cout << "Wait()" << std::endl;
  hsa_signal_value_t result;
  do {
    result = hsa_signal_wait_acquire(signal,
      HSA_SIGNAL_CONDITION_EQ, 0, ~0ULL, HSA_WAIT_STATE_ACTIVE);
  } while (result != 0);
  timespec currentTime;
  clock_gettime( CLOCK_REALTIME, &currentTime );
  double elapsed_us = (currentTime.tv_sec-startTime.tv_sec)*1000000.0
      + (currentTime.tv_nsec - startTime.tv_nsec)/1000.0;
  double elapsed_ms = elapsed_us / 1000.0;
  total_time_ms += elapsed_ms;
  std::cout << "t = " << elapsed_ms << " ms" << std::endl;
  double tflops = (2.0*5760*5760*5760/elapsed_ms/1000000000);
  std::cout << tflops << " GFlop/s (" << (100*tflops/8.192) << " %-of-peak" << std::endl;
  
  return true;
}

void* Dispatch::AllocateLocalMemory(size_t size)
{
  assert(local_region.handle != 0);
  void *p = NULL;

  hsa_status_t status = hsa_memory_allocate(local_region, size, (void **)&p);
  if (status != HSA_STATUS_SUCCESS) { HsaError("hsa_memory_allocate(local_region) failed", status); return 0; }
  //status = hsa_memory_assign_agent(p, agent, HSA_ACCESS_PERMISSION_RW);
  //if (status != HSA_STATUS_SUCCESS) { HsaError("hsa_memory_assign_agent failed", status); return 0; }
  return p;
}

void* Dispatch::AllocateSystemMemory(size_t size)
{
  void *p = NULL;
  hsa_status_t status = hsa_memory_allocate(system_region, size, (void **)&p);
  if (status != HSA_STATUS_SUCCESS) { HsaError("hsa_memory_allocate(system_region) failed", status); return 0; }
  return p;
}

bool Dispatch::CopyToLocal(void* dest, void* src, size_t size)
{
  hsa_status_t status;
  status = hsa_memory_copy(dest, src, size);
  if (status != HSA_STATUS_SUCCESS) { HsaError("hsa_memory_copy failed", status); return false; }
  //status = hsa_memory_assign_agent(dest, agent, HSA_ACCESS_PERMISSION_RW);
  //if (status != HSA_STATUS_SUCCESS) { HsaError("hsa_memory_assign_agent failed", status); return false; }
  return true;
}

bool Dispatch::CopyFromLocal(void* dest, void* src, size_t size)
{
  hsa_status_t status;
  status = hsa_memory_assign_agent(src, cpu_agent, HSA_ACCESS_PERMISSION_RW);
  if (status != HSA_STATUS_SUCCESS) { HsaError("hsa_memory_assign_agent failed", status); return false; }
  status = hsa_memory_copy(dest, src, size);
  if (status != HSA_STATUS_SUCCESS) { HsaError("hsa_memory_copy failed", status); return false; }
  return true;
}

Buffer* Dispatch::AllocateBuffer(size_t size)
{
  if (local_region.handle != 0) {
    void* system_ptr = AllocateSystemMemory(size);
    if (!system_ptr) { return 0; }
    void* local_ptr = AllocateLocalMemory(size);
    if (!local_ptr) { free(system_ptr); return 0; }
    return new Buffer(size, local_ptr, system_ptr);
  } else {
    void* system_ptr = AllocateSystemMemory(size);
    if (!system_ptr) { return 0; }
    return new Buffer(size, system_ptr);
  }
}

bool Dispatch::CopyTo(Buffer* buffer)
{
  if (buffer->IsLocal()) {
    return CopyToLocal(buffer->LocalPtr(), buffer->SystemPtr(), buffer->Size());
  }
  return true;
}

bool Dispatch::CopyFrom(Buffer* buffer)
{
  if (buffer->IsLocal()) {
    return CopyFromLocal(buffer->SystemPtr(), buffer->LocalPtr(), buffer->Size());
  }
  return true;
}

void Dispatch::KernargRaw(const void* ptr, size_t size, size_t align)
{
  assert((align & (align - 1)) == 0);
  kernarg_offset = ((kernarg_offset + align - 1) / align) * align;
  memcpy((char*) kernarg + kernarg_offset, ptr, size);
  kernarg_offset += size;
}

void Dispatch::Kernarg(Buffer* buffer)
{
  void* localPtr = buffer->LocalPtr();
  Kernarg(&localPtr);
}

bool Dispatch::Run()
{
    bool res = true;
    res = res && Init();
    res = res && InitDispatch();
    res = res && SetupExecutable();
    res = res && Setup();
    res = res && RunDispatch();
    res = res && Wait();
    res = res && Verify();
  std::string out = output.str();
  if (!out.empty()) {
    std::cout << out;
  }
  //std::cout << (res ? "Success" : "Failed") << std::endl;
  return res;
}

int Dispatch::RunMain()
{
  return Run() ? 0 : 1;
}

uint64_t Dispatch::GetTimestampFrequency()
{
  uint64_t frequency;
  hsa_status_t status;
  status = hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency);
  if (status != HSA_STATUS_SUCCESS) {
    HsaError("hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY) failed", status);
    return 0;
  }

  return frequency;
}

} // namespace dispatch
} // namespace amd


using namespace amd::dispatch;

class AsmKernelDispatch : public Dispatch {
private:
  Buffer* c;
  Buffer* a;
  Buffer* b;
  Buffer* debug;
  const unsigned int workGroup[2] = {16, 16};
  const unsigned int microTile[2] = {8, 8};
  float vC, vA, vB;
  float alpha;
  float beta;
  unsigned int M, N, K;
  unsigned int strideCJ, strideAK, strideBK;
  unsigned int numElementsC, numElementsA, numElementsB;
  unsigned int sizeC, sizeA, sizeB;
  unsigned int offsetC, offsetA, offsetB;
  unsigned int size0I, size1J, sizeK;
  float *refC;

public:
  AsmKernelDispatch(int argc, const char **argv)
    : Dispatch(argc, argv),
    vC(1),
    vA(1),
    vB(1),
    alpha(1),
    beta(0),
#if FULL_VALIDATION
    M(512),
    N(256),
    K(16),
#else
    //M(5760),
    //N(5760),
    //K(5760),
    M(1024*4),
    N(1024/2),
    K(196608/2),
#endif
    strideCJ(M+1),
    strideAK(M+1),
    strideBK(N+1),
    numElementsC(strideCJ*N),
    numElementsA(strideAK*K),
    numElementsB(strideBK*K),
    sizeC(numElementsC*sizeof(float)),
    sizeA(numElementsA*sizeof(float)),
    sizeB(numElementsB*sizeof(float)),
    offsetC(0),
    offsetA(0),
    offsetB(0),
    size0I(M),
    size1J(N),
    sizeK(K),
    refC(new float[numElementsC])
  {
  }

  bool SetupCodeObject() override {
    return LoadCodeObjectFromFile("kernel.co");
  }

  bool Setup() override {
    if (!AllocateKernarg(1024)) { return false; }
    unsigned int numDebugElements = 16*(size0I*size1J)/(microTile[0]*microTile[1]); // 16 uints per thread


    debug = AllocateBuffer(numDebugElements*sizeof(int));
    c = AllocateBuffer(sizeC);
    a = AllocateBuffer(sizeA);
    b = AllocateBuffer(sizeB);
#if FULL_VALIDATION
    for (unsigned int i = 0; i < numElementsC; i++)           refC[i] = i;//(i%M)*vC;
    for (unsigned int i = 0; i < numElementsC; i++) c->Data<float>(i) = i;//(i%M)*vC;
    for (unsigned int i = 0; i < numElementsA; i++) a->Data<float>(i) = i;//(i%M)*vA;
    for (unsigned int i = 0; i < numElementsB; i++) b->Data<float>(i) = i;//(i%M)*vB;
#else
    for (unsigned int i = 0; i < numElementsC; i++)           refC[i] = 1;
    for (unsigned int i = 0; i < numElementsC; i++) c->Data<float>(i) = 1;
    for (unsigned int i = 0; i < numElementsA; i++) a->Data<float>(i) = 1;
    for (unsigned int i = 0; i < numElementsB; i++) b->Data<float>(i) = 1;
#endif
    for (unsigned int i = 0; i < numDebugElements; i++) debug->Data<unsigned int>(i) = 3;
    if (!CopyTo(c)) output << "Error: failed to copy to local" << std::endl;
    if (!CopyTo(a)) output << "Error: failed to copy to local" << std::endl;
    if (!CopyTo(b)) output << "Error: failed to copy to local" << std::endl;
    if (!CopyTo(debug)) output << "Error: failed to copy debug to local" << std::endl;

    Kernarg(c);
    Kernarg(a);
    Kernarg(b);
    Kernarg(&alpha);
    Kernarg(&beta);
    Kernarg(&offsetC);
    Kernarg(&offsetA);
    Kernarg(&offsetB);
    Kernarg(&strideCJ);
    Kernarg(&strideAK);
    Kernarg(&strideBK);
#if 0
    Kernarg(&size0I);
    Kernarg(&size1J);
#else
    Kernarg(debug);
#endif
    Kernarg(&sizeK);
    //std::cout << "grid=" << size0I/microTile[0] << "x" << size1J/microTile[1] << std::endl;
    //std::cout << "workgroup=" << workGroup[0] << "x" << workGroup[1] << std::endl;
    SetGridSize(size0I/microTile[0],size1J/microTile[1]);
    SetWorkgroupSize(workGroup[0], workGroup[1]);
    return true;
  }

  bool Verify() override {
    //std::cout << "Verify()" << std::endl;

#if 0 && FULL_VALIDATION
    if (!CopyFrom(debug)) {
      std::cout << "Error: failed to copy debug from local" << std::endl;
      return false;
    } else {
      //std::cout << "Coppied back debug buffer" << std::endl;
    }

    // debug
    unsigned int numThreadsD0 = size0I/microTile[0];
    unsigned int numThreadsD1 = size1J/microTile[1];
    for (unsigned int row = 0; row < numThreadsD0; row++) { // screen-row
      for (unsigned int col = 0; col < numThreadsD1; col++) { // screen-col
        unsigned int threadSerial = col*(strideCJ/microTile[1]) + row;
        unsigned int threadDebugStartIdx = threadSerial * 1;
#if 0
        std::cout << std::setw(5) << debug->Data<unsigned int>(threadDebugStartIdx) << ", ";
#else
        std::cout << std::setw(5) << debug->Data<float>(threadDebugStartIdx) << ", ";
#endif
        //std::cout << std::setw(2) << debug->Data<unsigned int>(threadDebugStartIdx+1) << ", ";
        if (col%16==15) {
          std::cout << "  ";
        }

      }
      std::cout << std::endl;
      if (row%16==15) {
        std::cout << std::endl;
      }
    }
    std::cout << std::endl;
    std::cout << std::endl;
#endif


    unsigned int numValid = 0;
    unsigned int numInvalid = 0;
#if 1
    // validation
    if (!CopyFrom(c)) {
      std::cout << "Error: failed to copy from local" << std::endl;
      return false;
    } else {
      //std::cout << "Copied back C buffer" << std::endl;
    }

#if FULL_VALIDATION
    sgemm_cpu(
        false, // N
        true,  // T
        refC,
        a->Ptr<float>(),
        b->Ptr<float>(),
        alpha,
        beta,
        strideCJ,
        strideAK,
        strideBK,
        M,
        N,
        K );
#endif

    for (unsigned int d1 = 0; d1 < size1J; d1++) { // row
      for (unsigned int d0 = 0; d0 < size0I; d0++) { // col
        unsigned int idxC = d1*strideCJ+d0;
        float correctC;
#if FULL_VALIDATION
        correctC = refC[idxC];
#else
        correctC = alpha*K*1+beta*1;
#endif
        bool equal = c->Data<float>(idxC) == correctC;
        if (equal) {
          numValid++;
        } else {
          numInvalid++;
          if (numInvalid < 4 ) {
            std::cout << "c[" << d1 << "," << d0 << "] = "
                << c->Data<float>(idxC) << (equal ? " == " : " != ") << refC[idxC] << std::endl;
          }
        }
        //output << (equal ? "#" : "~");
        //output << std::setw(6) << c->Data<float>(idxC) << "=" << std::setw(6) << refC[idxC] << ", ";
      }
      //output << std::endl;
    }
    output << numValid << " P + " << numInvalid << " F" << std::endl;
#endif
    return numInvalid==0;
  }
}; // end class

int main(int argc, const char** argv)
{
  total_time_ms = 0;
  unsigned int numSamples = 1;
  for (unsigned int i = 0; i < numSamples; i++) {
   AsmKernelDispatch(argc, argv).RunMain();
  }
  double avg_time = (total_time_ms/numSamples);
  double tflops = (2.0*5760*5760*5760/avg_time/1000000000);
  std::cout << "AvgTime: " << avg_time << " ms (" << numSamples << " samples)" << std::endl;
  std::cout << "AvgPerf: " << tflops << " TFlop/s (" << (100*tflops/8.192) << " %-of-peak" << std::endl;
  return 1;
}

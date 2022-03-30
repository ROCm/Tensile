/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#error \
    "This implementation relies on CL exceptions to be enabled. Please define CL_HPP_ENABLE_EXCEPTIONS"
#endif // CL_HPP_ENABLE_EXCEPTIONS

#include <CL/cl2.hpp>

#include <gtest/gtest.h>

#include <Tensile/ocl/OclHardware.hpp>
#include <Tensile/ocl/OclSolutionAdapter.hpp>
#include <Tensile/ocl/OclUtils.hpp>

#include <Tensile/Debug.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>

#include "TestData.hpp"

#include <string>
#include <valarray>
#include <vector>

#ifdef TENSILE_USE_HIP
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>
#endif // TENSILE_USE_HIP

using namespace Tensile;

KernelInvocation initKernelParams(Tensile::TensorDescriptor const& desc,
                                  float*                           device_d,
                                  float const*                     device_c,
                                  float                            beta)
{
    KernelInvocation k;
    k.kernelName      = "Cijk_S";
    k.workGroupSize.x = 256;
    k.workGroupSize.y = 1;
    k.workGroupSize.z = 1;

    k.numWorkGroups.x = CeilDivide(desc.totalLogicalElements(), k.workGroupSize.x);
    k.numWorkGroups.y = 1;
    k.numWorkGroups.z = 1;

    k.numWorkItems.x = k.workGroupSize.x * k.numWorkGroups.x;
    k.numWorkItems.y = k.workGroupSize.y * k.numWorkGroups.y;
    k.numWorkItems.z = k.workGroupSize.z * k.numWorkGroups.z;

    // For OpenCL can also do the following:
    // k.args.append<void*>("D", buffer_D());
    // k.args.append<void const*>("C", buffer_C());
    // OR
    // k.args.append<cl::Buffer>("D", buffer_D);
    // k.args.append<cl::Buffer>("C", buffer_C);
    k.args.append<float*>("D", device_d);
    k.args.append<float const*>("C", device_c);
    k.args.append<unsigned int>("strideD1", desc.strides()[1]);
    k.args.append<unsigned int>("strideD2", desc.strides()[2]);
    k.args.append<unsigned int>("strideC1", desc.strides()[1]);
    k.args.append<unsigned int>("strideC2", desc.strides()[2]);
    k.args.append<unsigned int>("size0", desc.sizes()[0]);
    k.args.append<unsigned int>("size1", desc.sizes()[1]);
    k.args.append<unsigned int>("size2", desc.sizes()[2]);
    k.args.append<unsigned int>("offsetD", desc.offset());
    k.args.append<unsigned int>("offsetC", desc.offset());
    k.args.append<float>("beta", beta);

    return k;
}

void initRef(std::vector<float>& ref, float refVal, Tensile::TensorDescriptor& desc)
{
    for(int k = 0; k < desc.sizes()[2]; k++)
    {
        for(int j = 0; j < desc.sizes()[1]; j++)
        {
            for(int i = 0; i < desc.sizes()[0]; i++)
            {
                ref[desc.index(i, j, k)] = refVal;
            }
        }
    }
}

void validate(std::vector<float> const& ref, std::vector<float> const& result)
{
    ASSERT_EQ(ref.size(), result.size());
    for(int i = 0; i < ref.size(); i++)
    {
        ASSERT_FLOAT_EQ(ref[i], result[i]) << i;
    }
}

auto profileEvent(cl::Event const& event) -> std::tuple<size_t, size_t, size_t, size_t>
{
    auto queueTick  = event.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    auto submitTick = event.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    auto startTick  = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto endTick    = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    if(Debug::Instance().printPropertyEvaluation())
    {
        auto queueTime_ms   = static_cast<float>(submitTick - queueTick) * 1.0e-6;
        auto submitTime_ms  = static_cast<float>(startTick - submitTick) * 1.0e-6;
        auto executeTime_ms = static_cast<float>(endTick - startTick) * 1.0e-6;
        auto totalTime_ms   = static_cast<float>(endTick - queueTick) * 1.0e-6;
        std::cout << "PROFILING(ms): <queue | submit | exec | total>  = <" << queueTime_ms << " | "
                  << submitTime_ms << " | " << executeTime_ms << " | " << totalTime_ms << ">\n";
    }

    return std::make_tuple(queueTick, submitTick, startTick, endTick);
}

float eventExecTime_ms(cl::Event const& event)
{
    auto profile = profileEvent(event);
    return static_cast<float>(std::get<3>(profile) - std::get<2>(profile)) * 1.0e-6;
}

float eventTotalTime_ms(cl::Event const& event)
{
    auto profile = profileEvent(event);
    return static_cast<float>(std::get<3>(profile) - std::get<0>(profile)) * 1.0e-6;
}

// Manual initialization of CL context and devices
// Use embedded library
TEST(OclSolutionAdapterTest, TestInitKernel)
{
    // Initialize adapter
    ocl::SolutionAdapter adapter(false);
    adapter.loadEmbeddedCodeObjects("ocl_kernels_lite_mixed");

    EXPECT_THROW(adapter.initKernel("NoSuchKernel"), cl::Error);
    EXPECT_NO_THROW(adapter.initKernel("Cijk_S"));
}

TEST(OclSolutionAdapterTest, BetaOnlyKernel_Zero_Manual_Embedded)
{
    // Manually discover platform, context and devices
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    std::vector<cl::Platform>::iterator iter;
    for(iter = platforms.begin(); iter != platforms.end(); ++iter)
    {
        std::vector<cl::string> platformInfos = {(*iter).getInfo<CL_PLATFORM_PROFILE>(),
                                                 (*iter).getInfo<CL_PLATFORM_VERSION>(),
                                                 (*iter).getInfo<CL_PLATFORM_NAME>(),
                                                 (*iter).getInfo<CL_PLATFORM_VENDOR>(),
                                                 (*iter).getInfo<CL_PLATFORM_EXTENSIONS>()};

        if(platformInfos[3] == cl::string("Advanced Micro Devices, Inc."))
        {
            if(Debug::Instance().printPropertyEvaluation())
            {
                for(auto str : platformInfos)
                {
                    std::cout << str << std::endl;
                }
            }
            break;
        }
    }

    cl_context_properties cps[3] = {CL_CONTEXT_PLATFORM, (cl_context_properties)(*iter)(), 0};

    auto context = cl::Context(CL_DEVICE_TYPE_GPU, cps);
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();

    // Initialize adapter for first device
    ocl::SolutionAdapter adapter(false, "OclSolutionAdapter", context, devices[0]);
    adapter.loadEmbeddedCodeObjects("ocl_kernels_lite_mixed");

    // Initialize command queue for first device
    auto queue = cl::CommandQueue(context, devices[0]);

    // Initialize problem
    Tensile::TensorDescriptor desc(Tensile::DataType::Float, {43, 13, 65}, {1, 50, 50 * 16});
    cl::Buffer                buffer_C(context, CL_MEM_READ_WRITE, desc.totalAllocatedBytes());
    cl::Buffer                buffer_D(context, CL_MEM_READ_WRITE, desc.totalAllocatedBytes());

    // Initialize arrays C and D on the device
    const char fillC = 0x33;
    const char fillD = 0x22;
    queue.enqueueFillBuffer(buffer_C, fillC, 0, desc.totalAllocatedBytes());
    queue.enqueueFillBuffer(buffer_D, fillD, 0, desc.totalAllocatedBytes());

    // Prepare kernel and args
    auto k = initKernelParams(desc,
                              static_cast<float*>((void*)buffer_D()),
                              static_cast<float const*>((void*)buffer_C()),
                              0.0f);

    // Launch kernel on device queue
    adapter.launchKernel(k, queue);

    // Read result back from device
    std::vector<float> result(desc.totalAllocatedElements());
    queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, desc.totalAllocatedBytes(), result.data());

    // Close off queue
    queue.finish();

    // Create reference values and validate
    std::vector<float> reference(desc.totalAllocatedElements());
    memset(reference.data(), 0x22, desc.totalAllocatedBytes());
    initRef(reference, 0.0f, desc);
    validate(reference, result);
}

TEST(OclSolutionAdapterTest, BetaOnlyKernel_Zero_Default_Embedded)
{
    // Initialize adapter for default device
    ocl::SolutionAdapter adapter;
    adapter.loadEmbeddedCodeObjects("ocl_kernels_lite_mixed");

    // Initialize command queue for default device
    auto queue = cl::CommandQueue::getDefault();

    // Initialize problem
    Tensile::TensorDescriptor desc(Tensile::DataType::Float, {43, 13, 65}, {1, 50, 50 * 16});
    cl::Buffer buffer_C(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());
    cl::Buffer buffer_D(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());

    // Initialize arrays C and D on the device
    const char fillC = 0x33;
    const char fillD = 0x22;
    queue.enqueueFillBuffer(buffer_C, fillC, 0, desc.totalAllocatedBytes());
    queue.enqueueFillBuffer(buffer_D, fillD, 0, desc.totalAllocatedBytes());

    // Prepare kernel and args
    auto k = initKernelParams(desc,
                              static_cast<float*>((void*)buffer_D()),
                              static_cast<float const*>((void*)buffer_C()),
                              0.0f);

    // Launch kernel on device queue
    adapter.launchKernel(k, queue);

    // Read result back from device
    std::vector<float> result(desc.totalAllocatedElements());
    queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, desc.totalAllocatedBytes(), result.data());

    // Close off queue
    queue.finish();

    // Create reference values and validate
    std::vector<float> reference(desc.totalAllocatedElements());
    memset(reference.data(), 0x22, desc.totalAllocatedBytes());
    initRef(reference, 0.0f, desc);
    validate(reference, result);
}

TEST(OclSolutionAdapterTest, BetaOnlyKernel_Zero_Default_NonEmbedded)
{
    // Initialize adapter for default device
    ocl::SolutionAdapter adapter(false);
    for(auto file : TestData::Instance().glob("ocl_kernels_lite_mixed/Cijk_S.so*.hsaco"))
    {
        adapter.loadCodeObjectFile(file.native());
    }

    // Initialize command queue for default device
    auto queue = cl::CommandQueue::getDefault();

    // Initialize problem
    Tensile::TensorDescriptor desc(Tensile::DataType::Float, {43, 13, 65}, {1, 50, 50 * 16});
    cl::Buffer buffer_C(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());
    cl::Buffer buffer_D(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());

    // Initialize arrays C and D on the device
    const char fillC = 0x33;
    const char fillD = 0x22;
    queue.enqueueFillBuffer(buffer_C, fillC, 0, desc.totalAllocatedBytes());
    queue.enqueueFillBuffer(buffer_D, fillD, 0, desc.totalAllocatedBytes());

    // Prepare kernel and args
    auto k = initKernelParams(desc,
                              static_cast<float*>((void*)buffer_D()),
                              static_cast<float const*>((void*)buffer_C()),
                              0.0f);

    // Launch kernel on device queue
    adapter.launchKernel(k, queue);

    // Read result back from device
    std::vector<float> result(desc.totalAllocatedElements());
    queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, desc.totalAllocatedBytes(), result.data());

    // Close off queue
    queue.finish();

    // Create reference values and validate
    std::vector<float> reference(desc.totalAllocatedElements());
    memset(reference.data(), 0x22, desc.totalAllocatedBytes());
    initRef(reference, 0.0f, desc);
    validate(reference, result);
}

TEST(OclSolutionAdapterTest, BetaOnlyKernel_Nonzero)
{
    // Initialize adapter for default device
    ocl::SolutionAdapter adapter(false);
    adapter.loadEmbeddedCodeObjects("ocl_kernels_lite_mixed");

    // Initialize command queue for default device
    auto queue = cl::CommandQueue::getDefault();

    // Initialize problem
    Tensile::TensorDescriptor desc(Tensile::DataType::Float, {43, 13, 65}, {1, 50, 50 * 16});
    cl::Buffer buffer_C(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());
    cl::Buffer buffer_D(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());

    // Initialize arrays C and D on the device
    const char fillC = 0x22;
    const char fillD = 0x33;
    queue.enqueueFillBuffer(buffer_C, fillC, 0, desc.totalAllocatedBytes());
    queue.enqueueFillBuffer(buffer_D, fillD, 0, desc.totalAllocatedBytes());

    // Initialize beta and expected value
    const float beta     = 1.9f;
    float       initialC = 0.0f;
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float), &initialC);

    const float expectedD = initialC * beta;

    // Prepare kernel and args
    auto k = initKernelParams(desc,
                              static_cast<float*>((void*)buffer_D()),
                              static_cast<float const*>((void*)buffer_C()),
                              beta);

    // Launch kernel on device queue
    adapter.launchKernel(k, queue);

    // Read result back from device
    std::vector<float> result(desc.totalAllocatedElements());
    queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, desc.totalAllocatedBytes(), result.data());

    // Close off queue
    queue.finish();

    // Create reference values and validate
    std::vector<float> reference(desc.totalAllocatedElements());
    memset(reference.data(), 0x33, desc.totalAllocatedBytes());
    initRef(reference, expectedD, desc);
    validate(reference, result);
}

#ifdef TENSILE_USE_HIP

// Simultaneously launch the same kernel using both OpenCL and HIP
TEST(OclSolutionAdapterTest, PlayNiceWithHip)
{
    /////////////////////////////////////////////////////////////
    // First, do the OCL version
    ////////////////////////////////////////////////////////////

    // Initialize adapter for default device
    ocl::SolutionAdapter adapter(false);
    adapter.loadEmbeddedCodeObjects("ocl_kernels_lite_mixed");

    // Initialize command queue for default device
    auto queue = cl::CommandQueue::getDefault();

    // Initialize problem
    Tensile::TensorDescriptor desc(Tensile::DataType::Float, {43, 13, 65}, {1, 50, 50 * 16});
    cl::Buffer buffer_C(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());
    cl::Buffer buffer_D(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());

    // Initialize arrays C and D on the device
    const char fillC = 0x22;
    const char fillD = 0x33;
    queue.enqueueFillBuffer(buffer_C, fillC, 0, desc.totalAllocatedBytes());
    queue.enqueueFillBuffer(buffer_D, fillD, 0, desc.totalAllocatedBytes());

    // Initialize beta and expected value
    const float beta     = 1.9f;
    float       initialC = 0.0f;
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float), &initialC);

    const float expectedD = initialC * beta;

    // Prepare kernel and args
    auto k = initKernelParams(desc,
                              static_cast<float*>((void*)buffer_D()),
                              static_cast<float const*>((void*)buffer_C()),
                              beta);

    // Launch kernel on device queue
    adapter.launchKernel(k, queue);

    // Read result back from device
    std::vector<float> result(desc.totalAllocatedElements());
    queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, desc.totalAllocatedBytes(), result.data());

    // Close off queue
    queue.finish();

    // Create reference values and validate
    std::vector<float> reference(desc.totalAllocatedElements());
    memset(reference.data(), 0x33, desc.totalAllocatedBytes());
    initRef(reference, expectedD, desc);
    validate(reference, result);

    //////////////////////////////////////////////////////////////
    // Second, do the HIP version of the same kernel
    /////////////////////////////////////////////////////////////

    float* c_d = nullptr;
    float* d_d = nullptr;

    HIP_CHECK_EXC(hipMalloc(&c_d, desc.totalAllocatedBytes()));
    HIP_CHECK_EXC(hipMalloc(&d_d, desc.totalAllocatedBytes()));

    HIP_CHECK_EXC(hipMemset(c_d, 0x22, desc.totalAllocatedBytes()));
    HIP_CHECK_EXC(hipMemset(d_d, 0x33, desc.totalAllocatedBytes()));

    float c_initial_value;
    HIP_CHECK_EXC(hipMemcpy(&c_initial_value, c_d, sizeof(float), hipMemcpyDeviceToHost));
    float d_final_value = c_initial_value * beta;

    auto k_hip = initKernelParams(desc, d_d, c_d, beta);

    hip::SolutionAdapter adapter_hip(false);
    adapter_hip.loadEmbeddedCodeObjects("ocl_kernels_lite_mixed");

    adapter_hip.launchKernel(k_hip);

    // Copy back result
    std::vector<float> d_h(desc.totalAllocatedElements());
    HIP_CHECK_EXC(hipMemcpy(d_h.data(), d_d, desc.totalAllocatedBytes(), hipMemcpyDeviceToHost));

    // Validate against reference
    validate(reference, d_h);
}

#endif // TENSILE_USE_HIP

// Test the timings of single kernel launches
TEST(OclSolutionAdapterTest, TimingSingle)
{
    // Initialize adapter for default device
    ocl::SolutionAdapter adapter(false);
    adapter.loadEmbeddedCodeObjects("ocl_kernels_lite_mixed");

    // Initialize command queue for default device
    auto queue = cl::CommandQueue(
        cl::Context::getDefault(), cl::Device::getDefault(), CL_QUEUE_PROFILING_ENABLE);

    // Initialize problem
    Tensile::TensorDescriptor desc(Tensile::DataType::Float, {43, 13, 65}, {1, 50, 50 * 16});
    cl::Buffer buffer_C(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());
    cl::Buffer buffer_D(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());

    // Initialize arrays C and D on the device
    const char fillC = 0x22;
    const char fillD = 0x33;
    queue.enqueueFillBuffer(buffer_C, fillC, 0, desc.totalAllocatedBytes());
    queue.enqueueFillBuffer(buffer_D, fillD, 0, desc.totalAllocatedBytes());

    // Initialize beta and expected value
    const float beta     = 1.9f;
    float       initialC = 0.0f;
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float), &initialC);

    const float expectedD = initialC * beta;

    // Prepare kernel and args
    auto k = initKernelParams(desc,
                              static_cast<float*>((void*)buffer_D()),
                              static_cast<float const*>((void*)buffer_C()),
                              beta);

    // Launch kernel on device queue
    cl::Event event;
    adapter.launchKernel(k, queue, &event);

    // Shouldn't have issues, and should
    // hold a valid exec time.
    EXPECT_NO_THROW(event.wait());
    ASSERT_GT(eventExecTime_ms(event), 0.0f);

    // Read result back from device
    std::vector<float> result(desc.totalAllocatedElements());
    queue.enqueueReadBuffer(buffer_D, CL_TRUE, 0, desc.totalAllocatedBytes(), result.data());

    // Close off queue
    queue.finish();

    // Create reference values and validate
    std::vector<float> reference(desc.totalAllocatedElements());
    memset(reference.data(), 0x33, desc.totalAllocatedBytes());
    initRef(reference, expectedD, desc);
    validate(reference, result);
}

// Test the functionality profiling multiple kernel launches,
// or group kernel launch.
TEST(OclSolutionAdapterTest, TimingMulti)
{
    // Initialize adapter for default device
    ocl::SolutionAdapter adapter(false);
    adapter.loadEmbeddedCodeObjects("ocl_kernels_lite_mixed");

    // Initialize command queue for default device
    auto queue = cl::CommandQueue(
        cl::Context::getDefault(), cl::Device::getDefault(), CL_QUEUE_PROFILING_ENABLE);

    // Initialize problem
    Tensile::TensorDescriptor desc(Tensile::DataType::Float, {43, 13, 65}, {1, 50, 50 * 16});
    cl::Buffer buffer_C(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());
    cl::Buffer buffer_D(cl::Context::getDefault(), CL_MEM_READ_WRITE, desc.totalAllocatedBytes());

    // Initialize arrays C and D on the device
    const char fillC = 0x22;
    const char fillD = 0x33;
    queue.enqueueFillBuffer(buffer_C, fillC, 0, desc.totalAllocatedBytes());
    queue.enqueueFillBuffer(buffer_D, fillD, 0, desc.totalAllocatedBytes());

    // Initialize beta and expected value
    const float beta     = 0.0f;
    float       initialC = 0.0f;
    queue.enqueueReadBuffer(buffer_C, CL_TRUE, 0, sizeof(float), &initialC);

    // Prepare kernel and args
    auto k = initKernelParams(desc,
                              static_cast<float*>((void*)buffer_D()),
                              static_cast<float const*>((void*)buffer_C()),
                              beta);

    std::vector<decltype(k)> kernels(10, k);
    std::vector<cl::Event>   kernelEvents(10, cl::Event());

    // First, test individual events
    adapter.launchKernels(kernels, queue, kernelEvents);
    cl::Event::waitForEvents(kernelEvents);

    std::valarray<float> timings_ms(kernelEvents.size());
    int                  i = 0;
    for(auto ev : kernelEvents)
    {
        timings_ms[i++] = eventTotalTime_ms(ev);
    }

    //Second, test one aggregate event
    cl::Event groupEvent;
    adapter.launchKernels(kernels, queue, &groupEvent);
    groupEvent.wait();

    auto groupTime_ms = eventTotalTime_ms(groupEvent);
    auto avgInd_ms    = timings_ms.sum() / timings_ms.size();
    auto diff         = fabs(groupTime_ms - timings_ms.sum()) / timings_ms.size();

    if(Debug::Instance().printPropertyEvaluation())
    {
        std::cout << "Summed Time (ms): " << timings_ms.sum() << std::endl;
        std::cout << "Group Time (ms): " << groupTime_ms << std::endl;
        std::cout << "Diff: " << diff << std::endl;
    }

    // Loosly make sure that measured group time
    // correlates with the sum of the individual
    // group timings (within avg 1 kernel deviation)
    ASSERT_LT(diff, avgInd_ms) << diff;
}

// Test the functionality profiling multiple kernel launches,
// or group kernel launch.
TEST(OclSolutionAdapterTest, HardwareTest)
{
    int  deviceId = 0;
    auto oclProps = ocl::clGetDevicePropertiesAMD(deviceId);

    hipDeviceProp_t hipProps;
    hipGetDeviceProperties(&hipProps, deviceId);

    // Match the hip properties interface as much as possible
    ASSERT_EQ(oclProps.name, hipProps.name);
    ASSERT_EQ(oclProps.totalGlobalMem, hipProps.totalGlobalMem);
    ASSERT_EQ(oclProps.sharedMemPerBlock, hipProps.sharedMemPerBlock);
    ASSERT_EQ(oclProps.warpSize, hipProps.warpSize);
    ASSERT_EQ(oclProps.maxThreadsPerBlock, hipProps.maxThreadsPerBlock);
    ASSERT_EQ(oclProps.maxThreadsDim[0], hipProps.maxThreadsDim[0]);
    ASSERT_EQ(oclProps.maxThreadsDim[1], hipProps.maxThreadsDim[1]);
    ASSERT_EQ(oclProps.maxThreadsDim[2], hipProps.maxThreadsDim[2]);
    ASSERT_EQ(oclProps.maxGridSize[0], hipProps.maxGridSize[0]);
    ASSERT_EQ(oclProps.maxGridSize[1], hipProps.maxGridSize[1]);
    ASSERT_EQ(oclProps.maxGridSize[2], hipProps.maxGridSize[2]);
    ASSERT_EQ(oclProps.clockRate, hipProps.clockRate);
    // ASSERT_EQ(oclProps.multiProcessorCount, hipProps.multiProcessorCount);
    ASSERT_EQ(oclProps.pciBusID, hipProps.pciBusID);
    ASSERT_EQ(oclProps.pciDeviceID, hipProps.pciDeviceID);
    ASSERT_EQ(oclProps.maxSharedMemoryPerMultiProcessor, hipProps.maxSharedMemoryPerMultiProcessor);
    ASSERT_EQ(oclProps.gcnArch, hipProps.gcnArch);

    // Check that AMDGPU objects match
    auto oclGPU = std::dynamic_pointer_cast<AMDGPU>(ocl::GetDevice(oclProps));
    auto hipGPU = std::dynamic_pointer_cast<AMDGPU>(hip::GetDevice(hipProps));

    ASSERT_EQ(*oclGPU, *hipGPU);
}

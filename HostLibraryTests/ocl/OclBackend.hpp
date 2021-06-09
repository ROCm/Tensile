/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2021 Advanced Micro Devices, Inc.
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

#ifndef TESTING_OCL_BACKEND_HPP
#define TESTING_OCL_BACKEND_HPP

#ifndef CL_HPP_ENABLE_EXCEPTIONS
#error \
    "This implementation relies on CL exceptions to be enabled. Please define CL_HPP_ENABLE_EXCEPTIONS"
#endif // CL_HPP_ENABLE_EXCEPTIONS

#include <CL/cl2.hpp>

#include <Tensile/ocl/OclHardware.hpp>
#include <Tensile/ocl/OclSolutionAdapter.hpp>
#include <Tensile/ocl/OclUtils.hpp>

using namespace Tensile;

// NOTE: Best not to include this file in other HPP files.
// Try not to pollute include chains with OpenCL as it will
// then create additional dependencies on the OpenCL library.
// Include this file instead in isolated CPP compilation units
// where needed. E.g. RunGEMMKernel_test.cpp

struct OclBackend
{
    // Encapsulate type for
    // argument deduction
    template <typename T>
    struct BufferWrapper
    {
        BufferWrapper()                        = default;
        BufferWrapper(BufferWrapper<T> const&) = default;
        BufferWrapper<T>& operator=(BufferWrapper<T> const&) = default;

        BufferWrapper(std::nullptr_t) {}
        BufferWrapper(cl::Buffer const& buf)
            : mBuf(buf)
        {
        }

        using Type = T;
        cl::Buffer mBuf; // Full buffer allocation
        cl::Buffer mSubBuf; // Used to index buffer offsets
    };

    // Types
    using SolutionAdapter = ocl::SolutionAdapter;
    using Stream          = cl::CommandQueue;
    using Event           = cl::Event;
    template <typename T>
    using BufferObj = BufferWrapper<T>;

    // Device mgmt
    static inline void setDefaultDevice(int deviceId)
    {
        auto ctx     = cl::Context::getDefault();
        auto devices = ctx.template getInfo<CL_CONTEXT_DEVICES>();
        for(int i = 0; i < devices.size(); i++)
        {
            if(i == deviceId)
            {
                cl::Device::setDefault(devices[i]);
                break;
            }
        }
    }

    static inline std::shared_ptr<Hardware> getCurrentDevice()
    {
        return ocl::GetCurrentDevice();
    }

    static inline void deviceReset() {}

    static inline cl_uint offsetAlignment()
    {
        auto dev = cl::Device::getDefault();
        return dev.template getInfo<CL_DEVICE_MEM_BASE_ADDR_ALIGN>();
    }

    // Buffer mgmt
    template <typename T>
    static inline void malloc(BufferObj<T>& bufferObj, size_t bytes)
    {
        bufferObj = BufferObj<T>({cl::Context::getDefault(), CL_MEM_READ_WRITE, bytes});
    }

    template <typename T>
    static inline void free(BufferObj<T>& bufferObj)
    {
        bufferObj = BufferObj<T>();
    }

    template <typename T>
    static inline T* dataPtr(BufferObj<T>& bufferObj, size_t offsetBytes = 0)
    {
        if(offsetBytes == 0)
        {
            return reinterpret_cast<T*>(bufferObj.mBuf());
        }
        else
        {
            // Create a sub buffer to obtain device memory offset.
            // OpenCL has specific requirements about the offset alignment.
            assert((0 == ((offsetAlignment() - 1) & offsetBytes))
                   && "Offset alignment requirement not met");

            auto bufSize  = bufferObj.mBuf.template getInfo<CL_MEM_SIZE>();
            auto bufFlags = bufferObj.mBuf.template getInfo<CL_MEM_FLAGS>();

            auto             subBufSize = bufSize - offsetBytes;
            cl_buffer_region subRegion  = {offsetBytes, subBufSize};
            bufferObj.mSubBuf           = bufferObj.mBuf.createSubBuffer(
                bufFlags, CL_BUFFER_CREATE_TYPE_REGION, &subRegion);
            return reinterpret_cast<T*>(bufferObj.mSubBuf());
        }
    }

    template <typename T>
    static inline void copyHostToDevice(BufferObj<T>& bufferObj,
                                        size_t        byteOffset,
                                        T const*      hostData,
                                        size_t        byteCount)
    {
        auto queue = cl::CommandQueue::getDefault();
        queue.enqueueWriteBuffer(bufferObj.mBuf, CL_TRUE, byteOffset, byteCount, hostData);
    }

    template <typename T>
    static inline void copyDeviceToHost(T*                  hostData,
                                        BufferObj<T> const& bufferObj,
                                        size_t              byteOffset,
                                        size_t              byteCount)
    {
        auto queue = cl::CommandQueue::getDefault();
        queue.enqueueReadBuffer(bufferObj.mBuf, CL_TRUE, byteOffset, byteCount, hostData);
    }

    // Flag for the KernelArguments debug mode
    static inline bool kernelArgsLog()
    {
        return true;
    }
};

#endif // TESTING_OCL_BACKEND_HPP

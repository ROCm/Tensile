/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <CL/cl2.hpp>

#include <Tensile/AMDGPU.hpp>

#include <Tensile/ocl/OclHardware.hpp>
#include <Tensile/ocl/OclUtils.hpp>

namespace Tensile
{
    namespace ocl
    {
        OclAMDGPU::OclAMDGPU(oclDeviceProp_t const& prop)
            : AMDGPU(
                static_cast<AMDGPU::Processor>(prop.gcnArch),
                prop.multiProcessorCount,
                0, // hipDeviceAttributeDirectManagedMemAccessFromHost not accessible through OCL interface
                std::string(prop.name))
            , properties(prop)
        {
        }

        std::shared_ptr<Hardware> GetCurrentDevice()
        {
            return GetDevice(clGetDevicePropertiesAMD());
        }

        std::shared_ptr<Hardware> GetDevice(int deviceId)
        {
            return GetDevice(clGetDevicePropertiesAMD(deviceId));
        }

        std::shared_ptr<Hardware> GetDevice(int deviceId, cl::Context context)
        {
            return GetDevice(clGetDevicePropertiesAMD(deviceId, context));
        }

        std::shared_ptr<Hardware> GetDevice(oclDeviceProp_t const& prop)
        {
            return std::make_shared<OclAMDGPU>(prop);
        }
    } // namespace hip
} // namespace Tensile

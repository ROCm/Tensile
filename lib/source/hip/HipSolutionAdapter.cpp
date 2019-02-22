/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <hip/hip_runtime.h>

#if 0
hipError_t hipHccModuleLaunchKernel(hipFunction_t f, uint32_t globalWorkSizeX,
                                    uint32_t globalWorkSizeY, uint32_t globalWorkSizeZ,
                                    uint32_t localWorkSizeX, uint32_t localWorkSizeY,
                                    uint32_t localWorkSizeZ, size_t sharedMemBytes,
                                    hipStream_t hStream, void** kernelParams, void** extra,
                                    hipEvent_t startEvent = nullptr,
                                    hipEvent_t stopEvent = nullptr);
#endif

#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

namespace Tensile
{
    namespace hip
    {
        SolutionAdapter::SolutionAdapter(bool debug)
            : m_debug(debug)
        {
        }

        SolutionAdapter::~SolutionAdapter()
        {
            if(m_module)
                hipModuleUnload(m_module);
        }

        void SolutionAdapter::loadCodeObjectFile(std::string const& path)
        {
            HIP_CHECK_EXC(hipModuleLoad(&m_module, path.c_str()));
        }

        hipFunction_t SolutionAdapter::getKernel(std::string const& name)
        {
            auto it = m_kernels.find(name);
            if(it != m_kernels.end())
                return it->second;

            hipFunction_t rv;
            HIP_CHECK_EXC(hipModuleGetFunction(&rv, m_module, name.c_str()));
            m_kernels[name] = rv;
            return rv;
        }

        void SolutionAdapter::launchKernel(KernelInvocation const& kernel)
        {
            if(m_debug)
            {
                std::cout << "Kernel " << kernel.solution->name() << std::endl;
                std::cout << " l" << kernel.workGroupSize << " x g" << kernel.numWorkGroups << " = " << kernel.numWorkItems << std::endl;
                std::cout << kernel.args;
            }

            hipFunction_t function = getKernel(kernel.solution->name());

            void * kernelArgs = const_cast<void *>(kernel.args.data());
            size_t argsSize = kernel.args.size();

            void * hipLaunchParams[] = 
            {
                HIP_LAUNCH_PARAM_BUFFER_POINTER, kernelArgs,
                HIP_LAUNCH_PARAM_BUFFER_SIZE, &argsSize,
                HIP_LAUNCH_PARAM_END
            };

            HIP_CHECK_EXC(hipHccModuleLaunchKernel(
                        function,
                        kernel.numWorkItems.x, kernel.numWorkItems.y, kernel.numWorkItems.z,
                        kernel.workGroupSize.x, kernel.workGroupSize.y, kernel.workGroupSize.z,
                        kernel.sharedMemBytes, // sharedMem
                        0, // stream
                        nullptr,
                        (void **)&hipLaunchParams,
                        nullptr, // event
                        nullptr  // event
                        ));
        }

        void SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels)
        {
            for(auto const& k: kernels) launchKernel(k);
        }
    }
}



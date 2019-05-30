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

#include <Tensile/Debug.hpp>
#include <Tensile/EmbeddedData.hpp>
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
            for(auto module : m_modules)
                hipModuleUnload(module);
        }

        void SolutionAdapter::loadCodeObjectFile(std::string const& path)
        {
            hipModule_t module;
            auto        error = hipModuleLoad(&module, path.c_str());

            if(error == hipErrorFileNotFound)
                throw std::runtime_error(concatenate("Code object file '", path, "' not found."));
            else
                HIP_CHECK_EXC(error);

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.push_back(module);
            }
        }

        void SolutionAdapter::loadCodeObjectBytes(std::vector<uint8_t> const& bytes)
        {
            loadCodeObject(bytes.data());
        }

        void SolutionAdapter::loadCodeObject(const void* image)
        {
            hipModule_t module;

            HIP_CHECK_EXC(hipModuleLoadData(&module, image));

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.push_back(module);
            }
        }

        void SolutionAdapter::loadEmbeddedCodeObjects()
        {
            loadEmbeddedCodeObjects("");
        }

        void SolutionAdapter::loadEmbeddedCodeObjects(std::string const& key)
        {
            auto const& embeddedData = EmbeddedData<Tensile::SolutionAdapter>::Get(key);

            if(embeddedData.size() == 0)
            {
                if(Debug::Instance().printCodeObjectInfo())
                {
                    std::cerr << "Found no embedded code objects";
                    if(key != "")
                        std::cerr << " with the key " << key;

                    std::cerr << "." << std::endl;
                }
                return;
            }

            std::vector<hipModule_t> newModules(embeddedData.size());

            for(size_t i = 0; i < embeddedData.size(); i++)
                HIP_CHECK_EXC(hipModuleLoadData(&newModules[i], embeddedData[i].data()));

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.insert(m_modules.end(), newModules.begin(), newModules.end());
            }
        }

        hipFunction_t SolutionAdapter::getKernel(std::string const& name)
        {
            std::unique_lock<std::mutex> guard(m_access);

            auto it = m_kernels.find(name);
            if(it != m_kernels.end())
                return it->second;

            for(auto module : m_modules)
            {
                hipFunction_t rv;
                auto          err = hipModuleGetFunction(&rv, module, name.c_str());

                if(err == hipSuccess)
                {
                    m_kernels[name] = rv;
                    return rv;
                }
                else if(err != hipErrorNotFound)
                {
                    HIP_CHECK_EXC(err);
                }
            }

            throw std::runtime_error(
                concatenate("Kernel ", name, " not found in any loaded module."));
        }

        void SolutionAdapter::launchKernel(KernelInvocation const& kernel)
        {
            if(m_debug)
            {
                std::cout << "Kernel " << kernel.kernelName << std::endl;
                std::cout << " l" << kernel.workGroupSize << " x g" << kernel.numWorkGroups << " = "
                          << kernel.numWorkItems << std::endl;
                std::cout << kernel.args;
            }

            hipFunction_t function = getKernel(kernel.kernelName);

            void*  kernelArgs = const_cast<void*>(kernel.args.data());
            size_t argsSize   = kernel.args.size();

            void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                       kernelArgs,
                                       HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                       &argsSize,
                                       HIP_LAUNCH_PARAM_END};

            HIP_CHECK_EXC(hipExtModuleLaunchKernel(function,
                                                   kernel.numWorkItems.x,
                                                   kernel.numWorkItems.y,
                                                   kernel.numWorkItems.z,
                                                   kernel.workGroupSize.x,
                                                   kernel.workGroupSize.y,
                                                   kernel.workGroupSize.z,
                                                   kernel.sharedMemBytes, // sharedMem
                                                   0, // stream
                                                   nullptr,
                                                   (void**)&hipLaunchParams,
                                                   nullptr, // event
                                                   nullptr // event
                                                   ));
        }

        void SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels)
        {
            for(auto const& k : kernels)
                launchKernel(k);
        }
    }
}

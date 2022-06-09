/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>

#include <cstddef>

#include <Tensile/Debug.hpp>
#include <Tensile/EmbeddedData.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

namespace Tensile
{
    namespace hip
    {
        SolutionAdapter::SolutionAdapter()
            : m_debug(Debug::Instance().printKernelArguments())
            , m_debugSkipLaunch(Debug::Instance().skipKernelLaunch())
        {
        }

        SolutionAdapter::SolutionAdapter(bool debug)
            : m_debug(debug)
        {
            m_debug = debug || Debug::Instance().printKernelArguments();
        }

        SolutionAdapter::SolutionAdapter(bool debug, std::string const& name)
            : m_debug(debug)
            , m_name(name)
        {
            m_debug = debug || Debug::Instance().printKernelArguments();
        }

        SolutionAdapter::~SolutionAdapter()
        {
            for(auto module : m_modules)
                hipModuleUnload(module);
        }

        hipError_t SolutionAdapter::loadCodeObjectFile(std::string const& path)
        {
            hipModule_t module;

            HIP_CHECK_RETURN(hipModuleLoad(&module, path.c_str()));

            if(m_debug)
                std::cout << "loaded code object" << path << std::endl;

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.push_back(module);
                m_loadedModuleNames.push_back(concatenate("File ", path));
            }
            return hipSuccess;
        }

        hipError_t SolutionAdapter::loadCodeObjectBytes(std::vector<uint8_t> const& bytes)
        {
            return loadCodeObject(bytes.data());
        }

        hipError_t SolutionAdapter::loadCodeObject(const void* image)
        {
            hipModule_t module;

            HIP_CHECK_RETURN(hipModuleLoadData(&module, image));

            if(m_debug)
                std::cout << "loaded code object data." << std::endl;

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.push_back(module);
                m_loadedModuleNames.push_back("Module from bytes");
            }
            return hipSuccess;
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
                if(m_debug || Debug::Instance().printCodeObjectInfo())
                {
                    std::cerr << "Found no embedded code objects";
                    if(key != "")
                        std::cerr << " with the key " << key;

                    std::cerr << "." << std::endl;
                }
                return;
            }

            std::vector<hipModule_t> newModules;
            newModules.reserve(embeddedData.size());

            for(size_t i = 0; i < embeddedData.size(); i++)
            {
                hipModule_t nextModule;
                try
                {
                    auto error = hipModuleLoadData(&nextModule, embeddedData[i].data());

                    if(error == hipErrorUnknown || error == hipErrorSharedObjectInitFailed)
                        continue;
                    newModules.push_back(nextModule);
                    HIP_CHECK_EXC(error);

                    if(m_debug)
                        std::cout << "Loaded code object for key " << key << std::endl;
                }
                catch(std::runtime_error const& exc)
                {
                    std::cout << exc.what() << std::endl;
                }
            }

            {
                std::lock_guard<std::mutex> guard(m_access);
                m_modules.insert(m_modules.end(), newModules.begin(), newModules.end());
                m_loadedModuleNames.push_back(
                    concatenate("Embedded code object ", key, " (", newModules.size(), ")"));
            }
        }

        hipError_t SolutionAdapter::initKernel(std::string const& name)
        {
            hipFunction_t function;
            return getKernel(function, name);
        }

        hipError_t SolutionAdapter::getKernel(hipFunction_t& rv, std::string const& name)
        {
            std::unique_lock<std::mutex> guard(m_access);
            hipError_t                   err = hipErrorNotFound;

            auto it = m_kernels.find(name);
            if(it != m_kernels.end())
            {
                rv = it->second;
                return hipSuccess;
            }

            for(auto module : m_modules)
            {
                err = hipModuleGetFunction(&rv, module, name.c_str());

                if(err == hipSuccess)
                {
                    m_kernels[name] = rv;
                    return err;
                }
                else if(err != hipErrorNotFound)
                {
                    return err;
                }
            }

            return err;
        }

        hipError_t SolutionAdapter::launchKernel(KernelInvocation const& kernel)
        {
            return launchKernel(kernel, nullptr, nullptr, nullptr);
        }

        hipError_t SolutionAdapter::launchKernel(KernelInvocation const& kernel,
                                                 hipStream_t             stream,
                                                 hipEvent_t              startEvent,
                                                 hipEvent_t              stopEvent)
        {
            if(m_debug)
            {
                std::cout << "Kernel " << kernel.kernelName << std::endl;
                std::cout << " l" << kernel.workGroupSize << " x g" << kernel.numWorkGroups << " = "
                          << kernel.numWorkItems << std::endl;
                std::cout << kernel.args;
            }
            if(m_debugSkipLaunch)
            {
                std::cout << "DEBUG: Skip kernel execution" << std::endl;
                if(startEvent != nullptr)
                    HIP_CHECK_RETURN(hipEventRecord(startEvent, stream));
                if(stopEvent != nullptr)
                    HIP_CHECK_RETURN(hipEventRecord(stopEvent, stream));
                return hipSuccess;
            }

            hipFunction_t function;
            HIP_CHECK_RETURN(getKernel(function, kernel.kernelName));

            void*  kernelArgs = const_cast<void*>(kernel.args.data());
            size_t argsSize   = kernel.args.size();

            void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                       kernelArgs,
                                       HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                       &argsSize,
                                       HIP_LAUNCH_PARAM_END};

            if(startEvent != nullptr)
                HIP_CHECK_RETURN(hipEventRecord(startEvent, stream));
            HIP_CHECK_RETURN(hipExtModuleLaunchKernel(function,
                                                      kernel.numWorkItems.x,
                                                      kernel.numWorkItems.y,
                                                      kernel.numWorkItems.z,
                                                      kernel.workGroupSize.x,
                                                      kernel.workGroupSize.y,
                                                      kernel.workGroupSize.z,
                                                      kernel.sharedMemBytes, // sharedMem
                                                      stream, // stream
                                                      nullptr,
                                                      (void**)&hipLaunchParams,
                                                      nullptr, // event
                                                      nullptr // event
                                                      ));
            if(stopEvent != nullptr)
                HIP_CHECK_RETURN(hipEventRecord(stopEvent, stream));
            return hipSuccess;
        }

        hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels)
        {
            for(auto const& k : kernels)
            {
                HIP_CHECK_RETURN(launchKernel(k));
            }
            return hipSuccess;
        }

        hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels,
                                                  hipStream_t                          stream,
                                                  hipEvent_t                           startEvent,
                                                  hipEvent_t                           stopEvent)
        {
            auto first = kernels.begin();
            auto last  = kernels.end() - 1;

            for(auto iter = kernels.begin(); iter != kernels.end(); iter++)
            {
                hipEvent_t kStart = nullptr;
                hipEvent_t kStop  = nullptr;

                if(iter == first)
                    kStart = startEvent;
                if(iter == last)
                    kStop = stopEvent;

                HIP_CHECK_RETURN(launchKernel(*iter, stream, kStart, kStop));
            }
            return hipSuccess;
        }

        hipError_t SolutionAdapter::launchKernels(std::vector<KernelInvocation> const& kernels,
                                                  hipStream_t                          stream,
                                                  std::vector<hipEvent_t> const&       startEvents,
                                                  std::vector<hipEvent_t> const&       stopEvents)
        {
            if(kernels.size() != startEvents.size() || kernels.size() != stopEvents.size())
                throw std::runtime_error(concatenate("Must have an equal number of kernels (",
                                                     kernels.size(),
                                                     "), start events (",
                                                     startEvents.size(),
                                                     "), and stop events. (",
                                                     stopEvents.size(),
                                                     ")"));

            for(size_t i = 0; i < kernels.size(); i++)
            {
                HIP_CHECK_RETURN(launchKernel(kernels[i], stream, startEvents[i], stopEvents[i]));
            }
            return hipSuccess;
        }

        std::ostream& operator<<(std::ostream& stream, SolutionAdapter const& adapter)
        {
            stream << "hip::SolutionAdapter";

            if(adapter.m_debug)
            {
                stream << "[" << std::endl;
                for(auto const& name : adapter.m_loadedModuleNames)
                    stream << name << std::endl;

                stream << "]";
            }

            stream << " (" << adapter.name() << ", " << adapter.m_modules.size()
                   << " total modules)" << std::endl;

            return stream;
        }

        std::ostream& operator<<(std::ostream& stream, std::shared_ptr<SolutionAdapter> const& ptr)
        {
            if(ptr)
            {
                return stream << "*" << *ptr;
            }
            else
            {
                return stream << "(nullptr)";
            }
        }
    } // namespace hip
} // namespace Tensile

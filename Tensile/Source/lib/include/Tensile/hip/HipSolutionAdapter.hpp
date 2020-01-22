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

#pragma once

#include <Tensile/Tensile.hpp>
#include <hip/hip_runtime.h>

#include <mutex>

namespace Tensile
{
    namespace hip
    {
        class SolutionAdapter: public Tensile::SolutionAdapter
        {
        public:
            SolutionAdapter();
            SolutionAdapter(bool debug);
            SolutionAdapter(bool debug, std::string const& name);
            ~SolutionAdapter();

            virtual std::string name() const { return m_name; }

            void loadCodeObjectFile(std::string const& path);

            void loadCodeObject(const void * image);

            void loadCodeObjectBytes(std::vector<uint8_t> const& bytes);

            void loadEmbeddedCodeObjects();
            void loadEmbeddedCodeObjects(std::string const& key);

            void launchKernel(KernelInvocation const& kernel);
            void launchKernel(KernelInvocation const& kernel,
                              hipStream_t stream,
                              hipEvent_t startEvent,
                              hipEvent_t stopEvent);

            void launchKernels(std::vector<KernelInvocation> const& kernels);

            void launchKernels(std::vector<KernelInvocation> const& kernels,
                               hipStream_t stream,
                               hipEvent_t startEvent,
                               hipEvent_t stopEvent);

            void launchKernels(std::vector<KernelInvocation> const& kernels,
                               hipStream_t stream,
                               std::vector<hipEvent_t> const& startEvents,
                               std::vector<hipEvent_t> const& stopEvents);

        private:

            hipFunction_t getKernel(std::string const& name);

            std::mutex m_access;

            std::vector<hipModule_t> m_modules;
            std::unordered_map<std::string, hipFunction_t> m_kernels;
            bool m_debug = false;
            std::string m_name = "HipSolutionAdapter";

            std::vector<std::string> m_loadedModuleNames;

            friend std::ostream & operator<<(std::ostream & stream, SolutionAdapter const& adapter);
        };

        std::ostream & operator<<(std::ostream & stream, SolutionAdapter const& adapter);
    }
}

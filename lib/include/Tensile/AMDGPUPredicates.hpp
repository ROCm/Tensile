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

#include <Tensile/Predicates.hpp>
#include <Tensile/AMDGPU.hpp>

#include <vector>

namespace Tensile
{
    namespace AMDGPUPredicates
    {
        using namespace Tensile::Predicates;

        struct ProcessorEqual: public Predicate<AMDGPU>
        {
            AMDGPU::Processor value;

            ProcessorEqual() = default;
            ProcessorEqual(AMDGPU::Processor p) : value(p) {}

            static std::string Key() { return "ProcessorEqual"; }
            virtual std::string key() const { return Key(); }

            virtual bool operator()(AMDGPU const& gpu) const
            {
                return gpu.processor == value;
            }
        };

        struct RunsKernelTargeting: public Predicate<AMDGPU>
        {
            AMDGPU::Processor value;

            RunsKernelTargeting() = default;
            RunsKernelTargeting(AMDGPU::Processor p) : value(p) {}

            static std::string Key() { return "ProcessorEqual"; }
            virtual std::string key() const { return Key(); }

            virtual bool operator()(AMDGPU const& gpu) const
            {
                return gpu.runsKernelTargeting(value);
            }
        };
    }
}


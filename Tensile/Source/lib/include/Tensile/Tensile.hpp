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

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Tensile/Macros.hpp>
#include <Tensile/Tensile_fwd.hpp>
#include <Tensile/SolutionLibrary_fwd.hpp>

#include <Tensile/KernelArguments.hpp>
#include <Tensile/geom.hpp>

namespace Tensile
{
    class TENSILE_API Problem
    {
    public:
        __host__ virtual ~Problem();

        virtual std::string description() const = 0;
    };

    class TENSILE_API ProblemInputs
    {
    public:
        __host__ virtual ~ProblemInputs();

    };

    struct TENSILE_API KernelInvocation
    {
    public:
        std::string kernelName;

        dim3 workGroupSize;
        dim3 numWorkGroups;
        dim3 numWorkItems;
        size_t sharedMemBytes = 0;

        KernelArguments args;
    };

    class TENSILE_API Hardware
    {
    public:
        Hardware();
        virtual ~Hardware();

        virtual std::string description() const = 0;
    };

    /// Generally encapsulates a single kernel.
    class TENSILE_API Solution
    {
    public:
        __host__ virtual ~Solution();

        virtual std::string name() const = 0;
        virtual std::string description() const = 0;

    };

    class TENSILE_API SolutionAdapter
    {
    public:
        __host__ virtual ~SolutionAdapter();

        virtual std::string name() const = 0;
    };

#ifdef TENSILE_DEFAULT_SERIALIZATION
    template <typename MyProblem, typename MySolution = typename MyProblem::Solution>
    TENSILE_API std::shared_ptr<SolutionLibrary<MyProblem, MySolution>> LoadLibraryFile(std::string const& filename);
#endif
}


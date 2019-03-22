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

namespace Tensile
{
    struct TENSILE_API AMDGPU: public Hardware
    {
        static std::string Type() { return "AMDGPU"; }
        virtual std::string type() const;

        enum class Processor: int
        {
            gfx803 = 803,
            gfx900 = 900,
            gfx906 = 906
        };

        AMDGPU();
        AMDGPU(Processor p, int computeUnitCount, std::string const& deviceName);
        ~AMDGPU();

        Processor   processor = Processor::gfx900;
        int         computeUnitCount = 0;
        std::string deviceName;

        virtual bool runsKernelTargeting(Processor p) const;
        virtual std::string description() const;
    };

    inline bool operator<(AMDGPU::Processor l, AMDGPU::Processor r)
    {
        return static_cast<int>(l) < static_cast<int>(r);
    }

    inline bool operator>(AMDGPU::Processor l, AMDGPU::Processor r)
    {
        return static_cast<int>(l) > static_cast<int>(r);
    }

    inline bool operator<=(AMDGPU::Processor l, AMDGPU::Processor r)
    {
        return static_cast<int>(l) <= static_cast<int>(r);
    }

    inline bool operator>=(AMDGPU::Processor l, AMDGPU::Processor r)
    {
        return static_cast<int>(l) >= static_cast<int>(r);
    }

    std::ostream & operator<<(std::ostream & stream, AMDGPU::Processor p);
}



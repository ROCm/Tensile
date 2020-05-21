/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
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

#pragma once

#include <Tensile/Tensile.hpp>

namespace Tensile
{
    /**
 * \ingroup Hardware
 * Represents a particular AMD GPU in terms of processor model and number of
 * compute units.
 *
 * See subclass in `hip` directory which can create an instance
 * automatically.
 */
    struct TENSILE_API AMDGPU : public Hardware
    {
        static std::string Type()
        {
            return "AMDGPU";
        }
        virtual std::string type() const;

        enum class Processor : int
        {
            gfx803  = 803,
            gfx900  = 900,
            gfx906  = 906,
            gfx908  = 908,
            gfx1010 = 1010
        };

        AMDGPU();
        AMDGPU(Processor p, int computeUnitCount, std::string const& deviceName);
        ~AMDGPU();

        Processor   processor        = Processor::gfx900;
        int         wavefrontSize    = 64;
        int         simdPerCu        = 4;
        int         computeUnitCount = 0;
        std::string deviceName;

        virtual bool   runsKernelTargeting(Processor p) const;
        virtual size_t id() const
        {
            return (size_t)processor;
        }
        virtual std::string description() const;

        bool operator==(AMDGPU const& rhs) const
        {
            return processor == rhs.processor && computeUnitCount == rhs.computeUnitCount;
        }
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

    TENSILE_API std::ostream& operator<<(std::ostream& stream, AMDGPU::Processor p);
    TENSILE_API std::ostream& operator<<(std::ostream& stream, AMDGPU g);
} // namespace Tensile

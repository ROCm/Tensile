/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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
            // matching enum used in hipGcnArch
            // only including supported types
            //gfx000  =  0,
            //gfx701  =  1,
            //gfx801  =  2,
            //gfx802  =  3,
            gfx803  = 803,
            gfx900  = 900,
            gfx906  = 906,
            gfx908  = 908,
            gfx90a  = 910,
            gfx940  = 940,
            gfx941  = 941,
            gfx942  = 942,
            gfx1010 = 1010,
            gfx1011 = 1011,
            gfx1012 = 1012,
            gfx1030 = 1030,
            gfx1031 = 1031,
            gfx1032 = 1032,
            gfx1034 = 1034,
            gfx1035 = 1035,
            gfx1100 = 1100,
            gfx1101 = 1101,
            gfx1102 = 1102
        };

        static std::string toString(Processor p)
        {
            switch(p)
            {
            case AMDGPU::Processor::gfx803:
                return "gfx803";
            case AMDGPU::Processor::gfx900:
                return "gfx900";
            case AMDGPU::Processor::gfx906:
                return "gfx906";
            case AMDGPU::Processor::gfx908:
                return "gfx908";
            case AMDGPU::Processor::gfx90a:
                return "gfx90a";
            case AMDGPU::Processor::gfx940:
                return "gfx940";
            case AMDGPU::Processor::gfx941:
                return "gfx941";
            case AMDGPU::Processor::gfx942:
                return "gfx942";
            case AMDGPU::Processor::gfx1010:
                return "gfx1010";
            case AMDGPU::Processor::gfx1011:
                return "gfx1011";
            case AMDGPU::Processor::gfx1012:
                return "gfx1012";
            case AMDGPU::Processor::gfx1030:
                return "gfx1030";
            case AMDGPU::Processor::gfx1031:
                return "gfx1031";
            case AMDGPU::Processor::gfx1032:
                return "gfx1032";
            case AMDGPU::Processor::gfx1034:
                return "gfx1034";
            case AMDGPU::Processor::gfx1035:
                return "gfx1035";
            case AMDGPU::Processor::gfx1100:
                return "gfx1100";
            case AMDGPU::Processor::gfx1101:
                return "gfx1101";
            case AMDGPU::Processor::gfx1102:
                return "gfx1102";
            }
            return "";
        }

        AMDGPU::Processor toProcessorId(std::string const& deviceString)
        {
            if(deviceString.find("gfx803") != std::string::npos)
            {
                return AMDGPU::Processor::gfx803;
            }
            else if(deviceString.find("gfx900") != std::string::npos)
            {
                return AMDGPU::Processor::gfx900;
            }
            else if(deviceString.find("gfx906") != std::string::npos)
            {
                return AMDGPU::Processor::gfx906;
            }
            else if(deviceString.find("gfx908") != std::string::npos)
            {
                return AMDGPU::Processor::gfx908;
            }
            else if(deviceString.find("gfx90a") != std::string::npos)
            {
                return AMDGPU::Processor::gfx90a;
            }
            else if(deviceString.find("gfx940") != std::string::npos)
            {
                return AMDGPU::Processor::gfx940;
            }
            else if(deviceString.find("gfx941") != std::string::npos)
            {
                return AMDGPU::Processor::gfx941;
            }
            else if(deviceString.find("gfx942") != std::string::npos)
            {
                return AMDGPU::Processor::gfx942;
            }
            else if(deviceString.find("gfx1010") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1010;
            }
            else if(deviceString.find("gfx1011") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1011;
            }
            else if(deviceString.find("gfx1012") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1012;
            }
            else if(deviceString.find("gfx1030") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1030;
            }
            else if(deviceString.find("gfx1100") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1100;
            }
            else if(deviceString.find("gfx1101") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1101;
            }
            else if(deviceString.find("gfx1102") != std::string::npos)
            {
                return AMDGPU::Processor::gfx1102;
            }
            else
            {
                return static_cast<AMDGPU::Processor>(0);
            }
        }

        AMDGPU();
        AMDGPU(Processor p, int computeUnitCount, int isAPU, std::string const& deviceName);
        AMDGPU(std::string const& archName,
               int                computeUnitCount,
               int                isAPU,
               std::string const& deviceName);

        ~AMDGPU();

        Processor   processor        = Processor::gfx900;
        int         wavefrontSize    = 64;
        int         simdPerCu        = 4;
        int         computeUnitCount = 0;
        int         isAPU            = 0;
        std::string deviceName;

        virtual bool   runsKernelTargeting(Processor p) const;
        virtual size_t id() const
        {
            return (size_t)processor;
        }

        virtual std::string archName() const
        {
            return toString(processor);
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

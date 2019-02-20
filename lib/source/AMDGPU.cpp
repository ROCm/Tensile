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

#include <Tensile/AMDGPU.hpp>

namespace Tensile
{
    AMDGPU::AMDGPU(AMDGPU::Processor p, int cus, std::string const& name)
        : processor(p),
          computeUnitCount(cus),
          deviceName(name)
    {
    }

    bool AMDGPU::runsKernelTargeting(AMDGPU::Processor other) const
    {
        if(other > this->processor)
            return false;
        if(other == this->processor)
            return true;

        if(other == Processor::gfx803)
            return false;

        if(other == Processor::gfx900)
            return true;

        return false;
    }

    std::ostream & operator<<(std::ostream & stream, AMDGPU::Processor p)
    {
        switch(p)
        {
            case AMDGPU::Processor::gfx803: return stream << "gfx803";
            case AMDGPU::Processor::gfx900: return stream << "gfx900";
            case AMDGPU::Processor::gfx906: return stream << "gfx906";
        }
        return stream;
    }

    std::string AMDGPU::description() const
    {
        std::ostringstream rv;

        rv << deviceName << "(" << computeUnitCount << "-CU " << processor << ")";

        return rv.str();
    }
}

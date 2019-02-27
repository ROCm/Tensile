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
 * Copies of the Software, and to permit persons to whom the Software is
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

#include <functional>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Serialization/Base.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename IO>
        struct MappingTraits<ContractionSolution, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO & io, ContractionSolution & s)
            {
                iot::mapRequired(io, "name",       s.kernelName);
                iot::mapRequired(io, "index",      s.index);

                iot::mapRequired(io, "workGroup",  s.workGroupSize);
                iot::mapRequired(io, "threadTile", s.threadTile);
                iot::mapRequired(io, "macroTile",  s.macroTile);

                iot::mapRequired(io, "AType", s.aType);
                iot::mapRequired(io, "BType", s.bType);
                iot::mapRequired(io, "CType", s.cType);
                iot::mapRequired(io, "DType", s.dType);
            }
        };
    }
}


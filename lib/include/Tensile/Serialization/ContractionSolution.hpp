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
        struct MappingTraits<std::shared_ptr<ContractionSolution>, IO>
        {
            static void mapping(IO & io, std::shared_ptr<ContractionSolution> & p)
            {
                PointerMappingTraits<ContractionSolution, IO>::mapping(io, p);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<ContractionSolution, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO & io, ContractionSolution & s)
            {
                iot::mapRequired(io, "name",  s.kernelName);
                iot::mapRequired(io, "index", s.index);

                iot::mapRequired(io, "hardwarePredicate", s.hardwarePredicate);
                iot::mapRequired(io, "problemPredicate", s.problemPredicate);

                iot::mapRequired(io, "debugKernel", s.debugKernel);
                iot::mapRequired(io, "info", s.info);

                iot::mapRequired(io, "sizeMapping", s.sizeMapping);
                iot::mapRequired(io, "problemType", s.problemType);

            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<ContractionSolution::SizeMapping, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO & io, ContractionSolution::SizeMapping & s)
            {
                iot::mapRequired(io, "workGroup",  s.workGroupSize);
                iot::mapRequired(io, "threadTile", s.threadTile);
                iot::mapRequired(io, "macroTile",  s.macroTile);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<ContractionSolution::ProblemType, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO & io, ContractionSolution::ProblemType & s)
            {
                iot::mapRequired(io, "operationIdentifier", s.operationIdentifier);

                iot::mapRequired(io, "aType", s.aType);
                iot::mapRequired(io, "bType", s.bType);
                iot::mapRequired(io, "cType", s.cType);
                iot::mapRequired(io, "dType", s.dType);
            }

            const static bool flow = false;
        };
    }
}


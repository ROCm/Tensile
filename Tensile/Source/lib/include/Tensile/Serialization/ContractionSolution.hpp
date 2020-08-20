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
 * Copies of the Software, and to permit persons to whom the Software is
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
            static void mapping(IO& io, std::shared_ptr<ContractionSolution>& p)
            {
                PointerMappingTraits<ContractionSolution, IO>::mapping(io, p);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<ContractionSolution, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, ContractionSolution& s)
            {
                iot::mapRequired(io, "name", s.kernelName);
                iot::mapRequired(io, "index", s.index);

                iot::mapRequired(io, "hardwarePredicate", s.hardwarePredicate);
                iot::mapRequired(io, "problemPredicate", s.problemPredicate);

                iot::mapRequired(io, "debugKernel", s.debugKernel);
                iot::mapRequired(io, "info", s.info);
                iot::mapOptional(io, "ideals", s.ideals);

                iot::mapRequired(io, "sizeMapping", s.sizeMapping);
                iot::mapRequired(io, "problemType", s.problemType);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<ContractionSolution::SizeMapping, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, ContractionSolution::SizeMapping& s)
            {
                iot::mapRequired(io, "workGroup", s.workGroupSize);
                iot::mapRequired(io, "threadTile", s.threadTile);
                iot::mapRequired(io, "macroTile", s.macroTile);

                iot::mapRequired(io, "staggerU", s.staggerU);
                iot::mapRequired(io, "depthU", s.depthU);
                iot::mapRequired(io, "globalSplitU", s.globalSplitU);
                iot::mapRequired(io, "staggerStrideShift", s.staggerStrideShift);
                iot::mapRequired(io, "workGroupMapping", s.workGroupMapping);

                iot::mapOptional(io, "packBatchDims", s.packBatchDims);
                iot::mapOptional(io, "packSummationDims", s.packSummationDims);
                iot::mapOptional(io, "magicDivAlg", s.magicDivAlg);
                iot::mapRequired(io, "persistentKernel", s.persistentKernel);
                iot::mapRequired(io, "sourceKernel", s.sourceKernel);

                iot::mapRequired(io, "globalAccumulation", s.globalAccumulation);
                iot::mapRequired(io, "workspaceSizePerElemC", s.workspaceSizePerElemC);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<ContractionSolution::ProblemType, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, ContractionSolution::ProblemType& s)
            {
                iot::mapRequired(io, "operationIdentifier", s.operationIdentifier);

                iot::mapRequired(io, "aType", s.aType);
                iot::mapRequired(io, "bType", s.bType);
                iot::mapRequired(io, "cType", s.cType);
                iot::mapRequired(io, "dType", s.dType);
                iot::mapRequired(io, "useBeta", s.useBeta);
                iot::mapRequired(io, "highPrecisionAccumulate", s.highPrecisionAccumulate);
                iot::mapOptional(io, "useInitialStridesAB", s.useInitialStridesAB);
                iot::mapOptional(io, "useInitialStridesCD", s.useInitialStridesCD);
            }

            const static bool flow = false;
        };

        template <typename IO>
        struct MappingTraits<BufferLoadCheckPacket, IO>
        {
            using iot = IOTraits<IO>;
            static void mapping(IO& io, BufferLoadCheckPacket& s)
            {
                iot::mapRequired(io, "ShiftPtrElemA", s.shiftPtrElemA);
                iot::mapRequired(io, "ShiftPtrElemB", s.shiftPtrElemB);
                iot::mapRequired(io, "DUorMT0", s.depthUorMT0);
                iot::mapRequired(io, "DUorMT1", s.depthUorMT1);
            }

            const static bool flow = false;
        };
    } // namespace Serialization
} // namespace Tensile

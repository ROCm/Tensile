/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <Tensile/ContractionSolution.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/Utils.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>

namespace Tensile
{
    PerfModel perf;

    int32_t ContractionSolution::staggerUIter(ContractionSolution::Problem const& problem,
                                              ContractionSolution::Inputs const&  inputs,
                                              Hardware const&                     hardware) const
    {
        uint32_t sizeL = problem.boundSize(0);

        // how many stride-sized clicks to stagger start offset
        unsigned int staggerUIter = sizeMapping.staggerU;

        // /DepthU/GSU
        int unrollLoopIters = sizeL / sizeMapping.depthU / sizeMapping.globalSplitU;

        unsigned int shifted = 1 << sizeMapping.staggerStrideShift;

        while(staggerUIter > 1)
        {
            if(unrollLoopIters >= (staggerUIter * shifted))
                break;

            staggerUIter /= 2; // step down to smaller stagger
        }

        if(staggerUIter >= 1)
            staggerUIter -= 1;

        return staggerUIter;
    }

    // Return magic number.  If magicShift is 0, compute and return it.
    uint32_t ContractionSolution::magicNumberAlg1(uint32_t x, uint32_t* magicShift) const
    {
        uint64_t magicNum;
        *magicShift = 33;
        magicNum    = (1L << *magicShift) / x + 1;
        if((magicNum >> 32) != 0)
        {
            *magicShift = 31;
            magicNum    = (1L << *magicShift) / x + 1;
        }

        assert(magicNum >> 32 == 0); // ensure magic number fits

        return static_cast<uint32_t>(magicNum);
    }

    uint32_t ContractionSolution::magicNumberAlg2(uint32_t d, uint32_t* magicShift) const
    {
        struct mu
        {
            unsigned M; // Magic number,
            int      a; // "add" indicator,
            int      s;
        }; // and shift amount.

        struct mu magu;
        if(d == 0)
        {
            // Make dividend of 0 return 0
            magu.M = 0;
            magu.a = 0;
            magu.s = 0;
        }
        else
        {
            // Must have 1 <= d <= 2**32-1.
            int      p;
            unsigned nc, delta, q1, r1, q2, r2;
            magu.a = 0; // Initialize "add" indicator.
            nc     = -1 - (-d) % d; // Unsigned arithmetic here.
            p      = 31; // Init. p.
            q1     = 0x80000000 / nc; // Init. q1 = 2**p/nc.
            r1     = 0x80000000 - q1 * nc; // Init. r1 = rem(2**p, nc).
            q2     = 0x7FFFFFFF / d; // Init. q2 = (2**p - 1)/d.
            r2     = 0x7FFFFFFF - q2 * d; // Init. r2 = rem(2**p - 1, d).
            do
            {
                p = p + 1;
                if(r1 >= nc - r1)
                {
                    q1 = 2 * q1 + 1; // Update q1.
                    r1 = 2 * r1 - nc;
                } // Update r1.
                else
                {
                    q1 = 2 * q1;
                    r1 = 2 * r1;
                }
                if(r2 + 1 >= d - r2)
                {
                    if(q2 >= 0x7FFFFFFF)
                        magu.a = 1;
                    q2 = 2 * q2 + 1; // Update q2.
                    r2 = 2 * r2 + 1 - d;
                } // Update r2.
                else
                {
                    if(q2 >= 0x80000000)
                        magu.a = 1;
                    q2 = 2 * q2;
                    r2 = 2 * r2 + 1;
                }
                delta = d - 1 - r2;
            } while(p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));

            magu.M = q2 + 1; // Magic number
            magu.s = p - 32; // and shift amount to return
        }

        *magicShift         = magu.s;
        const uint32_t abit = 0x80000000;
        if(magu.a)
            *magicShift |= abit;

        // std::cout << " d=" << d << " M=" << magu.M << " a=" << magu.a << " s=" <<
        // magu.s << "\n";

        return magu.M;
    }

    uint32_t
        ContractionSolution::magicNumber(int magicDivAlg, uint32_t x, uint32_t* magicShift) const
    {
        if(magicDivAlg == 1)
            return magicNumberAlg1(x, magicShift);
        else if(magicDivAlg == 2)
            return magicNumberAlg2(x, magicShift);
        else
            throw std::runtime_error("bad magicDivAlg");
    }

    uint32_t ContractionSolution::smallMagicNumber(uint32_t x) const
    {
        uint64_t  magicNum;
        const int smallMagicShift = 31;
        magicNum                  = (1L << smallMagicShift) / x + 1;
        assert(magicNum >> 32 == 0); // ensure magic number fits
        return static_cast<uint32_t>(magicNum);
    }

    std::vector<size_t> generatePackedIndicesA(ContractionSolution::Problem const& problem,
                                               size_t                              packBatchDims)
    {
        std::vector<size_t> packedIndices;

        // TODO -move packedIndices calc to problem decode.
        for(auto idx = 0; idx < problem.a().dimensions(); idx++)
        {
            bool isSum = problem.boundIndices().end()
                         != std::find_if(problem.boundIndices().begin(),
                                         problem.boundIndices().end(),
                                         [idx](const ContractionProblem::BoundIndex& bi) {
                                             return bi.a == idx;
                                         });

            bool nonPackableBatch = false;
            // TODO - base this check on if the batch is SetConstStrideA=0 - if so,
            // don't pack
            if(!(packBatchDims & 0x1))
            {
                nonPackableBatch = problem.batchIndices().end()
                                   != std::find_if(problem.batchIndices().begin(),
                                                   problem.batchIndices().end(),
                                                   [idx](const ContractionProblem::BatchIndex& bi) {
                                                       return bi.a == idx;
                                                   });
            }

            if(!isSum && !nonPackableBatch)
                packedIndices.push_back(idx);
        }

        return packedIndices;
    }

    std::vector<size_t> generatePackedIndicesB(ContractionSolution::Problem const& problem,
                                               size_t                              packBatchDims)
    {
        std::vector<size_t> packedIndices;

        // Pack in all non-summation indices, except don't need magic number for the
        // last one
        for(auto idx = 0; idx < problem.b().dimensions(); idx++)
        {
            bool isSum = problem.boundIndices().end()
                         != std::find_if(problem.boundIndices().begin(),
                                         problem.boundIndices().end(),
                                         [idx](const ContractionProblem::BoundIndex& bi) {
                                             return bi.b == idx;
                                         });

            bool nonPackableBatch = false;
            // TODO - base this check on if the batch is SetConstStrideB=0 - if so,
            // don't pack
            if(!(packBatchDims & 0x2))
            {
                nonPackableBatch = problem.batchIndices().end()
                                   != std::find_if(problem.batchIndices().begin(),
                                                   problem.batchIndices().end(),
                                                   [idx](const ContractionProblem::BatchIndex& bi) {
                                                       return bi.b == idx;
                                                   });
            }

            if(!isSum && !nonPackableBatch)
                packedIndices.push_back(idx);
        }

        return packedIndices;
    }

    template <typename TypedInputs, bool T_Debug>
    KernelInvocation
        ContractionSolution::generateSingleCall(ContractionSolution::Problem const& problem,
                                                TypedInputs const&                  inputs,
                                                Hardware const&                     hardware) const
    {
        TENSILE_ASSERT_EXC(sizeMapping.workGroupMapping >= 0);

        TensorDescriptor const& a = problem.a();
        TensorDescriptor const& b = problem.b();
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();

        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(1024, 128);

        rv.kernelName = kernelName;

        rv.workGroupSize.x = sizeMapping.workGroupSize.x * sizeMapping.workGroupSize.y
                             * sizeMapping.workGroupSize.z;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        rv.numWorkGroups.x = 1;
        rv.numWorkGroups.y = 1;

        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
        {
            rv.numWorkGroups.x *= problem.freeSizeA(i);
        }
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
        {
            rv.numWorkGroups.y *= problem.freeSizeB(i);
        }

        rv.numWorkGroups.z = 1;
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
        {
            if(sizeMapping.packBatchDims & 0x1)
                rv.numWorkGroups.x *= problem.batchSize(i);
            if(sizeMapping.packBatchDims & 0x2)
                rv.numWorkGroups.y *= problem.batchSize(i);
            if(!sizeMapping.packBatchDims)
                rv.numWorkGroups.z *= problem.batchSize(i);
        }

        if(problem.transposeC01())
            std::swap(rv.numWorkGroups.x, rv.numWorkGroups.y);

        rv.numWorkGroups.x = CeilDivide(rv.numWorkGroups.x, sizeMapping.macroTile.x);
        rv.numWorkGroups.y = CeilDivide(rv.numWorkGroups.y, sizeMapping.macroTile.y);

        uint32_t problemNumGroupTiles0 = rv.numWorkGroups.x;
        uint32_t problemNumGroupTiles1 = rv.numWorkGroups.y;
        // used only when persistent kernel along batch
        uint32_t problemNumGroupTiles2 = rv.numWorkGroups.z;

        rv.numWorkGroups.y *= sizeMapping.globalSplitU;

        if(sizeMapping.persistentKernel != 0)
        {
            AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
            assert(pAMDGPU != nullptr && pAMDGPU->computeUnitCount != 0);

            size_t cuCount       = pAMDGPU->computeUnitCount;
            size_t finalPKValue  = sizeMapping.persistentKernel;
            size_t problemGroups = rv.numWorkGroups.x * rv.numWorkGroups.y;
            if(sizeMapping.persistentKernelAlongBatch)
            {
                problemGroups *= rv.numWorkGroups.z;
                rv.numWorkGroups.z = 1;
            }

            if(finalPKValue == -1)
            {
                // 1. Get the largest pk value (ex.3)
                //    which can make the PK.G (ex.3*120=360) <= problemGroups (ex.433)
                // 2. Scale by 5/8 (can try 0.5~1, to control the tiles-per-workgroup = 1~2)
                finalPKValue = 5 * (problemGroups / cuCount) / 8;
                finalPKValue = std::max(finalPKValue, (size_t)1);
                //std::cout << "final persistent kernel value: " << finalPKValue << std::endl;
            }

            size_t persistentGroups = cuCount * finalPKValue;
            rv.numWorkGroups.x      = std::min(persistentGroups, problemGroups);
            rv.numWorkGroups.y      = 1;
        }

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        if(debugKernel)
        {
            rv.args.appendUnbound<unsigned int*>("debugBuffer");
        }

        rv.sharedMemBytes = 0;

        if(!isSourceKernel())
        {
            uint64_t tensor2dSizeC = c.totalAllocatedElements();
            uint64_t tensor2dSizeA = (sizeMapping.packBatchDims & 0x1)
                                         ? a.totalAllocatedElements()
                                         : problem.allocatedElementsNonBatchA();
            uint64_t tensor2dSizeB = (sizeMapping.packBatchDims & 0x2)
                                         ? b.totalAllocatedElements()
                                         : problem.allocatedElementsNonBatchB();

            rv.args.append<uint64_t>("tensor2dSizeC", tensor2dSizeC);
            rv.args.append<uint64_t>("tensor2dSizeA", tensor2dSizeA);
            rv.args.append<uint64_t>("tensor2dSizeB", tensor2dSizeB);
        }

        if(sizeMapping.globalAccumulation)
        {
            rv.args.append<void const*>("ws_d", inputs.ws);
            rv.args.append<void const*>("ws_c", inputs.ws);
        }
        else if(problemType.stridedBatched)
        {
            rv.args.append<typename TypedInputs::DType const*>("d", inputs.d);
            rv.args.append<typename TypedInputs::CType const*>("c", inputs.c);
        }
        else
        {
            rv.args.append<typename TypedInputs::DType const* const*>("batchD", inputs.batchD);
            rv.args.append<typename TypedInputs::CType const* const*>("batchC", inputs.batchC);
        }

        if(problemType.stridedBatched)
        {
            rv.args.append<typename TypedInputs::AType const*>("a", inputs.a);
            rv.args.append<typename TypedInputs::BType const*>("b", inputs.b);
        }
        else
        {
            rv.args.append<typename TypedInputs::AType const* const*>("batchA", inputs.batchA);
            rv.args.append<typename TypedInputs::BType const* const*>("batchB", inputs.batchB);
        }

        rv.args.append<typename TypedInputs::AlphaType>("alpha", inputs.alpha);
        if(std::is_same<typename TypedInputs::AlphaType, Half>::value && !isSourceKernel())
            rv.args.append<typename TypedInputs::AlphaType>("alpha_2", inputs.alpha);

        if(problemType.useBeta)
        {
            rv.args.append<typename TypedInputs::BetaType>("beta", inputs.beta);
            if(std::is_same<typename TypedInputs::BetaType, Half>::value && !isSourceKernel())
                rv.args.append<typename TypedInputs::BetaType>("beta_2", inputs.beta);
        }

        size_t startStrideCD = problemType.useInitialStridesCD ? 0 : 1;
        size_t startStrideAB = problemType.useInitialStridesAB ? 0 : 1;

        if(sizeMapping.globalAccumulation)
        {
            size_t wsStride = startStrideCD ? d.sizes()[0] : 1;
            for(size_t i = startStrideCD; i < d.dimensions(); i++)
            {
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideW_D", i), wsStride);
                wsStride *= d.sizes()[i];
            }

            wsStride = startStrideCD ? d.sizes()[0] : 1;
            for(size_t i = startStrideCD; i < c.dimensions(); i++)
            {
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideW_C", i), wsStride);
                wsStride *= d.sizes()[i];
            }
        }
        else
        {
            for(size_t i = startStrideCD; i < d.dimensions(); i++)
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideD", i), d.strides()[i]);

            for(size_t i = startStrideCD; i < c.dimensions(); i++)
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideC", i), c.strides()[i]);
        }

        for(size_t i = startStrideAB; i < a.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideA", i), a.strides()[i]);

        for(size_t i = startStrideAB; i < b.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideB", i), b.strides()[i]);

        {
            int idx = 0;
            for(auto size : problem.problemSizes())
            {
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("size_", idx), size);
                idx++;
            }
        }

        if(sizeMapping.packSummationDims)
            // boundIndices are ordered with unroll last.
            // Magic numbers for all but first are needed to unpack other dims.
            for(auto si = 1; si < problem.boundIndices().size(); si++)
            {
                auto numIter  = problem.boundSize(si);
                bool isUnroll = si == problem.boundIndices().size() - 1;
                if(isUnroll)
                {
                    numIter = numIter / sizeMapping.depthU / sizeMapping.globalSplitU
                              * sizeMapping.depthU;
                }
                uint32_t magicShift;
                rv.args.append<uint32_t>(
                    concatenate_if<T_Debug>("magicNumberNumIter_", si),
                    magicNumber(sizeMapping.magicDivAlg, numIter, &magicShift));
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("magicShiftNumIter_", si),
                                         magicShift);

                if(isUnroll and sizeMapping.globalSplitU > 1)
                {
                    // compute magic number for gsu remainder iterations:
                    // Kernel will select whether to use above or remainder portion based on work-group assignment
                    rv.args.append<uint32_t>(
                        concatenate_if<T_Debug>("magicNumberNumIter_GsuRemainder"),
                        magicNumber(
                            sizeMapping.magicDivAlg, numIter + sizeMapping.depthU, &magicShift));
                    rv.args.append<uint32_t>(
                        concatenate_if<T_Debug>("magicShiftNumIter_GsuRemainder"), magicShift);
                }
            }

        if(problem.freeIndicesA().size() > 1 || sizeMapping.packBatchDims & 0x1)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesA(problem, sizeMapping.packBatchDims);

            // Pack in all non-summation indices, except don't need magic number for the
            // last one
            for(auto pi = packedIndices.begin(); pi != packedIndices.end() - 1; pi++)
            {
                auto     idx  = *pi;
                auto     size = a.sizes()[idx];
                uint32_t magicShift;
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("magicNumberSizeA_", idx),
                                         magicNumber(sizeMapping.magicDivAlg, size, &magicShift));
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("magicShiftSizeA_", idx),
                                         magicShift);
            }
        }
        if(problem.freeIndicesB().size() > 1 || sizeMapping.packBatchDims & 0x2)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesB(problem, sizeMapping.packBatchDims);

            // Pack in all non-summation indices, except don't need magic number for the
            // last one
            for(auto pi = packedIndices.begin(); pi != packedIndices.end() - 1; pi++)
            {
                auto     idx  = *pi;
                auto     size = b.sizes()[idx];
                uint32_t magicShift;
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("magicNumberSizeB_", idx),
                                         magicNumber(sizeMapping.magicDivAlg, size, &magicShift));
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("magicShiftSizeB_", idx),
                                         magicShift);
            }
        }

        for(auto si : problem.boundIndices())
        {
            if(si.aZeroPad.valid())
            {
                rv.args.append<int32_t>(concatenate_if<T_Debug>("padStartA_", si.a),
                                        si.aZeroPad.padStart);
                rv.args.append<int32_t>(concatenate_if<T_Debug>("padEndA_", si.a),
                                        si.aZeroPad.padEnd);
            }
            if(si.bZeroPad.valid())
            {
                rv.args.append<int32_t>(concatenate_if<T_Debug>("padStartB_", si.b),
                                        si.bZeroPad.padStart);
                rv.args.append<int32_t>(concatenate_if<T_Debug>("padEndB_", si.b),
                                        si.bZeroPad.padEnd);
            }
        }

        rv.args.append<int32_t>("staggerUIter", staggerUIter(problem, inputs, hardware));

        rv.args.append<uint32_t>("problemNumGroupTiles0", problemNumGroupTiles0);
        rv.args.append<uint32_t>("problemNumGroupTiles1", problemNumGroupTiles1);

        if(!isSourceKernel())
        {
            uint32_t numFullBlocks            = problemNumGroupTiles1;
            uint32_t wgmRemainder1            = 0;
            uint32_t magicNumberWgmRemainder1 = 0;

            // conditional args, aligned with KernelWriterAssembly.py
            if(sizeMapping.persistentKernel != 0)
            {
                uint32_t magicShift;
                rv.args.append<uint32_t>("magicNumberProblemNumGroupTiles0",
                                         magicNumber(2, problemNumGroupTiles0, &magicShift));
                rv.args.append<uint32_t>("magicShiftProblemNumGroupTiles0", magicShift);
                rv.args.append<uint32_t>("gridNumWorkGroups0", rv.numWorkGroups.x);
            }

            if(sizeMapping.persistentKernelAlongBatch)
            {
                uint32_t numGroupTiles0x1 = problemNumGroupTiles0 * problemNumGroupTiles1;
                uint32_t magicShift;

                rv.args.append<uint32_t>("problemNumGroupTiles2", problemNumGroupTiles2);
                rv.args.append<uint32_t>("magicNumberProblemNumGroupTiles0By1",
                                         magicNumber(2, numGroupTiles0x1, &magicShift));
                rv.args.append<uint32_t>("magicShiftProblemNumGroupTiles0By1", magicShift);
            }

            if(sizeMapping.workGroupMapping != 0)
            {
                numFullBlocks = problemNumGroupTiles1 / sizeMapping.workGroupMapping;
                wgmRemainder1 = problemNumGroupTiles1 % sizeMapping.workGroupMapping;
                if(wgmRemainder1 == 0)
                    wgmRemainder1 = sizeMapping.workGroupMapping;
                magicNumberWgmRemainder1 = smallMagicNumber(wgmRemainder1);
            }

            rv.args.append<uint32_t>("numFullBlocks", numFullBlocks);
            rv.args.append<uint32_t>("wgmRemainder1", wgmRemainder1);
            rv.args.append<uint32_t>("magicNumberWgmRemainder1", magicNumberWgmRemainder1);
        }

        rv.args.append<uint32_t>("offsetD", d.offset());
        rv.args.append<uint32_t>("offsetC", c.offset());
        rv.args.append<uint32_t>("offsetA", a.offset());
        rv.args.append<uint32_t>("offsetB", b.offset());

        if(!isSourceKernel())
        {
            rv.args.append<uint32_t>("pad", 0);
        }

        return rv;
    }

    bool ContractionSolution::isSourceKernel() const
    {
        return sizeMapping.sourceKernel;
    }

    template <typename TypedInputs, bool T_Debug>
    KernelInvocation ContractionSolution::generateBetaOnlyCall(Problem const&     problem,
                                                               TypedInputs const& inputs,
                                                               Hardware const&    hardware) const
    {
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();

        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(512, 64);

        rv.kernelName = betaOnlyKernelName(problem, inputs, hardware);

        rv.workGroupSize.x = 256;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        size_t wiX = 1;
        size_t wiY = 1;
        size_t wiZ = 1;
        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            wiX *= problem.freeSizeA(i);
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            wiY *= problem.freeSizeB(i);
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
            wiZ *= problem.batchSize(i);

        rv.numWorkGroups.x = CeilDivide(wiX * wiY * wiZ, rv.workGroupSize.x);
        rv.numWorkGroups.y = 1;
        rv.numWorkGroups.z = 1;

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        if(sizeMapping.globalAccumulation)
            rv.args.append<void*>("WS", inputs.ws);
        else if(problemType.stridedBatched)
            rv.args.append<typename TypedInputs::DType*>("D", inputs.d);
        else
            rv.args.append<typename TypedInputs::DType const* const*>("batchD", inputs.batchD);

        if(problemType.stridedBatched)
            rv.args.append<typename TypedInputs::CType const*>("C", inputs.c);
        else
            rv.args.append<typename TypedInputs::CType const* const*>("batchC", inputs.batchC);

        if(sizeMapping.globalAccumulation)
        {
            size_t stride = d.sizes()[0];
            for(size_t i = 1; i < d.dimensions(); i++)
            {
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideW", i),
                                         d.sizes()[i] == 1 ? 0 : stride);
                stride *= d.sizes()[i];
            }
        }
        else
        {
            for(size_t i = 1; i < d.dimensions(); i++)
                rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideD", i),
                                         d.sizes()[i] == 1 ? 0 : d.strides()[i]);
        }

        for(size_t i = 1; i < c.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideC", i),
                                     c.sizes()[i] == 1 ? 0 : c.strides()[i]);

        int idx = 0;
        for(auto size : problem.d().sizes())
        {
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("size_", idx), size);
            idx++;
        }

        rv.args.append<uint32_t>("offsetD", d.offset());
        rv.args.append<uint32_t>("offsetC", c.offset());

        rv.args.append<typename TypedInputs::BetaType>("beta", inputs.beta);

        return rv;
    }

    template <typename TypedInputs>
    std::string ContractionSolution::betaOnlyKernelName(Problem const&     problem,
                                                        TypedInputs const& inputs,
                                                        Hardware const&    hardware) const
    {
        std::string name = concatenate(
            "C", problem.cNames(), "_", TypeInfo<typename TypedInputs::DType>::Abbrev());

        if(!problemType.stridedBatched)
        {
            name += "_GB";
        }

        if(sizeMapping.globalAccumulation)
        {
            name += "_GA";
        }

        return name;
    }

    template <typename TypedInputs, bool T_Debug>
    KernelInvocation ContractionSolution::generateOutputConversionCall(
        Problem const& problem, TypedInputs const& inputs, Hardware const& hardware) const
    {
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();

        KernelInvocation rv;

        rv.args = KernelArguments(T_Debug);

        rv.args.reserve(512, 64);

        rv.kernelName = outputConversionKernelName(problem, inputs, hardware);

        rv.workGroupSize.x = 256;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        size_t wiX = 1;
        size_t wiY = 1;
        size_t wiZ = 1;
        for(size_t i = 0; i < problem.freeIndicesA().size(); i++)
            wiX *= problem.freeSizeA(i);
        for(size_t i = 0; i < problem.freeIndicesB().size(); i++)
            wiY *= problem.freeSizeB(i);
        for(size_t i = 0; i < problem.batchIndices().size(); i++)
            wiZ *= problem.batchSize(i);

        rv.numWorkGroups.x = CeilDivide(wiX * wiY * wiZ, rv.workGroupSize.x);
        rv.numWorkGroups.y = 1;
        rv.numWorkGroups.z = 1;

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        if(problemType.stridedBatched)
            rv.args.append<typename TypedInputs::DType*>("D", inputs.d);
        else
            rv.args.append<typename TypedInputs::DType const* const*>("batchD", inputs.batchD);

        rv.args.append<void*>("WS", inputs.ws);

        if(problemType.stridedBatched)
            rv.args.append<typename TypedInputs::CType const*>("C", inputs.c);
        else
            rv.args.append<typename TypedInputs::CType const* const*>("batchC", inputs.batchC);

        if(sizeMapping.globalAccumulation == 2)
            rv.args.append<typename TypedInputs::AlphaType>("alpha", inputs.alpha);
        else
            rv.args.append<typename TypedInputs::AlphaType>("alpha", 1.0f);

        if(sizeMapping.globalAccumulation == 2 and problemType.useBeta)
            rv.args.append<typename TypedInputs::BetaType>("beta", inputs.beta);
        else
            rv.args.append<typename TypedInputs::BetaType>("beta", 0.0f);

        for(size_t i = 1; i < d.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideD", i), d.strides()[i]);

        uint32_t wsStride = d.sizes()[0];
        for(size_t i = 1; i < d.dimensions(); i++)
        {
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideW", i), wsStride);
            wsStride *= d.sizes()[i];
        }

        for(size_t i = 1; i < c.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("strideC", i), c.strides()[i]);

        int idx = 0;
        for(auto size : problem.d().sizes())
        {
            rv.args.append<uint32_t>(concatenate_if<T_Debug>("size_", idx), size);
            idx++;
        }

        rv.args.append<uint32_t>("offsetD", d.offset());
        rv.args.append<uint32_t>("offsetC", c.offset());

        if(sizeMapping.globalAccumulation == 1)
            rv.args.append<uint32_t>("gsu", 1);
        else
            rv.args.append<uint32_t>("gsu", sizeMapping.globalSplitU);

        return rv;
    }

    template <typename TypedInputs>
    std::string ContractionSolution::outputConversionKernelName(Problem const&     problem,
                                                                TypedInputs const& inputs,
                                                                Hardware const&    hardware) const
    {
        std::string name = concatenate(
            "C", problem.cNames(), "_", TypeInfo<typename TypedInputs::DType>::Abbrev());

        if(!problemType.stridedBatched)
        {
            name += "_GB";
        }

        name += "_PostGSU";

        return name;
    }

    template <typename TypedInputs>
    std::vector<KernelInvocation> ContractionSolution::solveTyped(Problem const&     problem,
                                                                  TypedInputs const& inputs,
                                                                  Hardware const&    hardware) const
    {
        bool debug = Debug::Instance().printKernelArguments() || this->kernelArgsLog;

        int boundSize = 1;
        for(size_t i = 0; i < problem.boundIndices().size(); i++)
            boundSize *= problem.boundSize(i);

        // Check for nullptrs if alpha is non-zero.
        if(((inputs.alpha != static_cast<typename TypedInputs::AlphaType>(0)) && (boundSize != 0))
           && ((problem.stridedBatched() && (inputs.a == nullptr || inputs.b == nullptr))
               || (!problem.stridedBatched()
                   && (inputs.batchA == nullptr || inputs.batchB == nullptr))))
        {
            std::string matrixID = inputs.a == nullptr ? "A" : "B";
            std::string msg      = std::string("Unsupported nullptr for ") + matrixID
                              + std::string(" when (Alpha !=0) && (K != 0)\n");
            throw std::runtime_error(msg.c_str());
        }

        // Check if alpha matches problem definition
        if(problem.alphaRestriction() != ScalarValue::Any
           && problem.alphaRestriction() != toScalarValueEnum(inputs.alpha))
        {
            std::stringstream inputValue;
            inputValue << inputs.alpha;
            std::string msg = std::string("Alpha value ") + inputValue.str()
                              + std::string(" doesn't match that set in problem: ")
                              + ToString(problem.alphaRestriction());
            throw std::runtime_error(msg.c_str());
        }

        // Check if beta matches problem definition
        if(problem.betaRestriction() != ScalarValue::Any
           && problem.betaRestriction() != toScalarValueEnum(inputs.beta))
        {
            std::stringstream inputValue;
            inputValue << inputs.beta;
            std::string msg = std::string("Beta value ") + inputValue.str()
                              + std::string(" doesn't match that set in problem: ")
                              + ToString(problem.betaRestriction());
            throw std::runtime_error(msg.c_str());
        }

        if(problem.cEqualsD() && inputs.c != inputs.d)
            throw std::runtime_error(
                "ContractionProblem has cEqualsD set, but pointers for c and d are not equal");

        std::vector<KernelInvocation> rv;

        if(sizeMapping.globalSplitU > 1 && sizeMapping.globalAccumulation != 2)
        {
            if(debug)
                rv.push_back(generateBetaOnlyCall<TypedInputs, true>(problem, inputs, hardware));
            else
                rv.push_back(generateBetaOnlyCall<TypedInputs, false>(problem, inputs, hardware));
        }

        if(debug)
            rv.push_back(generateSingleCall<TypedInputs, true>(problem, inputs, hardware));
        else
            rv.push_back(generateSingleCall<TypedInputs, false>(problem, inputs, hardware));

        if(sizeMapping.globalAccumulation)
        {
            if(debug)
                rv.push_back(
                    generateOutputConversionCall<TypedInputs, true>(problem, inputs, hardware));
            else
                rv.push_back(
                    generateOutputConversionCall<TypedInputs, false>(problem, inputs, hardware));
        }

        return rv;
    }

    std::vector<KernelInvocation>
        ContractionSolution::solve(ContractionSolution::Problem const& problem,
                                   ContractionSolution::Inputs const&  inputs,
                                   Hardware const&                     hardware) const
    {
        if(Debug::Instance().printWinningKernelName())
            std::cout << "Running kernel: " << this->KernelName() << std::endl;

        // retreive alpha/beta type set via setAlpha/BetaType()
        auto alphaType = problem.alphaType();
        auto betaType  = problem.betaType();

        // TODO: Some gtests are passing the "problem" without actually defining the
        // alpha/beta type (alphaType and betaType remain None).
        // Until we fix those gtests, we need to keep this condition to adjust the missing
        // alpha/beta data types.
        if(alphaType == DataType::None)
        {
            alphaType
                = problemType.aType == DataType::BFloat16 ? DataType::Float : problemType.dType;
        }
        if(betaType == DataType::None)
        {
            betaType = alphaType;
        }

        auto contractionInputsTypeId = ContractionInputs::TypeId(problemType.aType,
                                                                 problemType.bType,
                                                                 problemType.cType,
                                                                 problemType.dType,
                                                                 alphaType,
                                                                 betaType);

        switch(contractionInputsTypeId)
        {
        case ContractionInputs_S_S_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_S_S_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_D_D_D::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_D_D_D const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_C_C_C::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_C_C_C const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_Z_Z_Z::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_Z_Z_Z const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#ifdef TENSILE_USE_HALF
        case ContractionInputs_H_H_H::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_H_H_H const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_H_H_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_H_H_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_H_S_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_H_S_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#endif // TENSILE_USE_HALF
        case ContractionInputs_I8x4_I32_I32::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_I8x4_I32_I32 const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_I32_I32_I32::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_I32_I32_I32 const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_I8_I32_I32::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_I8_I32_I32 const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#ifdef TENSILE_USE_BF16
        case ContractionInputs_B_B_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_B_B_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        case ContractionInputs_B_S_S::TypeId():
        {
            auto const& typedInputs = dynamic_cast<ContractionInputs_B_S_S const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#endif // TENSILE_USE_BF16

        default:;
        }
        throw std::runtime_error("Data type not implemented.");
    }

    ContractionSolution::StaticPerformanceModel
        ContractionSolution::staticPerformanceModel(double M,
                                                    double N,
                                                    double K,
                                                    double NumBatches,
                                                    double MT0,
                                                    double MT1,
                                                    double NumCUs,
                                                    double TotalGranularity,
                                                    int    GlobalSplitU) const
    {
        StaticPerformanceModel spm;

        int beta      = (int)problemType.useBeta;
        int betaReads = 0, betaWrites = 0;
        if(GlobalSplitU == 1)
        {
            if(beta != 0.0)
                betaReads = 1.0;
        }
        else
        {
            if(beta == 0)
                betaWrites = 1; // zero output
            else if(beta != 1.0) // if 1.0, just atomic update output
            {
                // if not 1.0, read, scale, write, then atomic update in kernel
                betaReads  = 1; // initial read for scale
                betaWrites = 1; // writeback after scale
            }
        }

        auto aInfo = DataTypeInfo::Get(problemType.aType);
        auto bInfo = DataTypeInfo::Get(problemType.bType);
        auto cInfo = DataTypeInfo::Get(problemType.cType);
        auto dInfo = DataTypeInfo::Get(problemType.dType);

        spm.memReadBytesA = (NumBatches * M * N * K) / MT1 * aInfo.elementSize;
        spm.memReadBytesB = (NumBatches * M * N * K) / MT0 * bInfo.elementSize;
        spm.memReadBytesC = (NumBatches * M * N) * betaReads * cInfo.elementSize;

        if(GlobalSplitU == 1)
            spm.memWriteBytesD = (NumBatches * M * N) * (1 + betaWrites) * dInfo.elementSize;
        else
        {
            bool   hardwareAtomic   = false; // TODO-model
            double atomicOperations = hardwareAtomic ? 2 : 3; // read-mod-write or cas  //TODO-model
            double atomicCollisions = 1.0; // TODO-could be based on K, GSU
            spm.memWriteBytesD      = (NumBatches * M * N)
                                 * (betaWrites + atomicOperations * atomicCollisions)
                                 * dInfo.elementSize;
        }
        spm.memReadBytes   = spm.memReadBytesA + spm.memReadBytesB + spm.memReadBytesC;
        spm.memGlobalReads = spm.memReadBytesA / aInfo.elementSize
                             + spm.memReadBytesB / bInfo.elementSize
                             + spm.memReadBytesC / cInfo.elementSize;
        spm.memGlobalWrites = spm.memWriteBytesD / dInfo.elementSize;

        return spm;
    }

    size_t ContractionSolution::requiredWorkspaceSize(Problem const& problem) const
    {
        size_t size = 0;

        size += problem.d().totalLogicalElements() * sizeMapping.workspaceSizePerElemC;

        return size;
    }

    ContractionSolution::Granularities ContractionSolution::computeGranularities(
        Hardware const& hardware, double M, double N, double K, double NumBatches) const
    {
        ContractionSolution::Granularities granularities;

        double MT0 = sizeMapping.macroTile.x;
        double MT1 = sizeMapping.macroTile.y;

        AMDGPU const* pAMDGPU = dynamic_cast<AMDGPU const*>(&hardware);
        assert(pAMDGPU);

        double NumCUs        = pAMDGPU->computeUnitCount;
        double wavefrontSize = pAMDGPU->wavefrontSize;
        double simdPerCu     = pAMDGPU->simdPerCu;

        double GlobalSplitU = sizeMapping.globalSplitU;
        double LocalSplitU  = sizeMapping.workGroupSize.z;

        granularities.MT0 = MT0;
        granularities.MT1 = MT1;
        granularities.GSU = GlobalSplitU;
        granularities.LSU = LocalSplitU;
        granularities.CUs = NumCUs;

        granularities.numTiles0 = M / MT0;
        granularities.numTiles1 = N / MT1;

        granularities.tile0Granularity = granularities.numTiles0 / ceil(granularities.numTiles0);
        granularities.tile1Granularity = granularities.numTiles1 / ceil(granularities.numTiles1);

        granularities.tilesPerCu
            = (NumBatches * ceil(granularities.numTiles0) * ceil(granularities.numTiles1))
              / (NumCUs / GlobalSplitU / LocalSplitU);

        granularities.totalTiles    = ceil(granularities.numTiles0) * ceil(granularities.numTiles1);
        granularities.natTilesPerCu = NumBatches * granularities.totalTiles / NumCUs;
        granularities.suTilesPerCu  = (granularities.totalTiles * GlobalSplitU) / NumCUs;
        granularities.suCuGranularity
            = granularities.suTilesPerCu / ceil(granularities.suTilesPerCu);

        granularities.waveGranularity = std::min(
            1.00,
            static_cast<double>(floor(granularities.tilesPerCu + 1.0) * sizeMapping.workGroupSize.x
                                * sizeMapping.workGroupSize.y * sizeMapping.workGroupSize.z)
                / pAMDGPU->wavefrontSize / pAMDGPU->simdPerCu);

        granularities.waves
            = ceil((sizeMapping.workGroupSize.x * sizeMapping.workGroupSize.y) / wavefrontSize);

        granularities.suWavesPerSimdx2
            = (granularities.suTilesPerCu * granularities.waves) / (2 * simdPerCu);
        granularities.suWaveGranularity
            = granularities.suWavesPerSimdx2 * ceil(granularities.suWavesPerSimdx2);

        double nat_tiles_per_cu
            = NumBatches * ceil(granularities.numTiles0) * ceil(granularities.numTiles1) / NumCUs;
        granularities.natCuGranularity = ceil(nat_tiles_per_cu) * ceil(nat_tiles_per_cu) / NumCUs;

        granularities.cuGranularity = granularities.tilesPerCu / ceil(granularities.tilesPerCu);

        granularities.totalGranularity
            = granularities.tile0Granularity * granularities.tile1Granularity
              * granularities.cuGranularity * granularities.waveGranularity;

        granularities.totalTileAwareGranularity
            = granularities.tile0Granularity * granularities.tile1Granularity
              * granularities.suCuGranularity * granularities.suWaveGranularity;

        return granularities;
    }

    ContractionSolution::ProjectedPerformance
        ContractionSolution::projectedPerformance(Problem const&  problem,
                                                  Hardware const& hardware) const
    {
        ProjectedPerformance pp;

        double M = 1.0, N = 1.0;
        if(problem.freeIndicesA().size() > 1 || sizeMapping.packBatchDims & 0x1)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesA(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                M *= problem.a().sizes()[*pi];
        }
        else
            M = problem.freeSizeA(0);

        if(problem.freeIndicesB().size() > 1 || sizeMapping.packBatchDims & 0x2)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesB(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                N *= problem.b().sizes()[*pi];
        }
        else
            N = problem.freeSizeB(0);

        double NumBatches = 1;
        if(sizeMapping.packBatchDims == 0)
        {
            for(size_t i = 0; i < problem.batchIndices().size(); i++)
                NumBatches *= problem.batchSize(i);
        }
        double K = problem.boundSize(0); // TODO - fix for multiple summations

        pp.granularities = ContractionSolution::computeGranularities(hardware, M, N, K, NumBatches);

        auto it = ideals.begin();

        int    closestKMeasure     = std::numeric_limits<int>::max();
        double closestKPerformance = 0.0;

        while(it != ideals.end())
        {
            int myK       = it->first;
            int myMeasure = std::abs(myK - K);
            if(myMeasure < closestKMeasure)
            {
                closestKMeasure     = myMeasure;
                closestKPerformance = it->second;
            }
            it++;
        }

        double MT0    = pp.granularities.MT0;
        double MT1    = pp.granularities.MT1;
        double NumCUs = pp.granularities.CUs;

        double GlobalSplitU         = pp.granularities.GSU;
        double IdealGranularityPerf = closestKPerformance;

        pp.staticModel = staticPerformanceModel(
            M, N, K, NumBatches, MT0, MT1, NumCUs, pp.granularities.totalGranularity, GlobalSplitU);

        pp.speedGFlops = IdealGranularityPerf * pp.granularities.totalGranularity;
        pp.CUs         = NumCUs;

        return pp;
    }

    ContractionSolution::TAMetricProblemScore ContractionSolution::computeProblemScore(
        Hardware const& hardware, double M, double N, double K, double NumBatches) const
    {
        ContractionSolution::TAMetricProblemScore pp;
        pp.granularites = ContractionSolution::computeGranularities(hardware, M, N, K, NumBatches);

        pp.M = M;
        pp.N = N;
        pp.K = K;

        double slope     = linearModel.slope;
        double intercept = linearModel.intercept;
        double perf_max  = linearModel.max;

        double sum_value        = K;
        double sum_perf0        = sum_value / (intercept + (slope * sum_value));
        pp.summationPerformance = 1000.0 * sum_perf0 / perf_max;

        return pp;
    }

    double ContractionSolution::computeTileAwareMetric(
        ContractionSolution::TAMetricProblemScore pp,
        ContractionSolution::TAMetricProblemScore ppReference) const
    {
        double tile0GranularityDim = abs(log(ppReference.granularites.tile0Granularity)
                                         - log(pp.granularites.tile0Granularity));
        double metric              = tile0GranularityDim;

        double tile1GranularityDim = abs(log(ppReference.granularites.tile1Granularity)
                                         - log(pp.granularites.tile1Granularity));
        metric += tile1GranularityDim;

        double natCuGranularityDim = abs(log(ppReference.granularites.natCuGranularity)
                                         - log(pp.granularites.natCuGranularity));
        metric += natCuGranularityDim;

        double suCuGranularityDim = abs(log(ppReference.granularites.suCuGranularity)
                                        - log(pp.granularites.suCuGranularity));
        metric += suCuGranularityDim;

        double suWaveGranularityDim = abs(log(ppReference.granularites.suWaveGranularity)
                                          - log(pp.granularites.suWaveGranularity));
        metric += suWaveGranularityDim;

        double natTilesPerCuDim
            = abs(log(ppReference.granularites.natTilesPerCu) - log(pp.granularites.natTilesPerCu));
        metric += natTilesPerCuDim;

        double suTilesPerCuDim
            = abs(log(ppReference.granularites.suTilesPerCu) - log(pp.granularites.suTilesPerCu));
        metric += suTilesPerCuDim;

        double summationPerformanceDim
            = abs(ppReference.summationPerformance - pp.summationPerformance);
        metric += summationPerformanceDim;

        return metric;
    }

    double ContractionSolution::computeTAMScore(Problem const&  problem,
                                                Hardware const& hardware,
                                                double          model_M,
                                                double          model_N,
                                                double          model_K,
                                                double          model_NumBatches) const
    {
        double M = 1.0, N = 1.0;
        if(problem.freeIndicesA().size() > 1 || sizeMapping.packBatchDims & 0x1)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesA(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                M *= problem.a().sizes()[*pi];
        }
        else
            M = problem.freeSizeA(0);

        if(problem.freeIndicesB().size() > 1 || sizeMapping.packBatchDims & 0x2)
        {
            std::vector<size_t> packedIndices
                = generatePackedIndicesB(problem, sizeMapping.packBatchDims);
            for(auto pi = packedIndices.begin(); pi != packedIndices.end(); pi++)
                N *= problem.b().sizes()[*pi];
        }
        else
            N = problem.freeSizeB(0);

        double NumBatches = 1;
        if(sizeMapping.packBatchDims == 0)
        {
            for(size_t i = 0; i < problem.batchIndices().size(); i++)
                NumBatches *= problem.batchSize(i);
        }
        double K = problem.boundSize(0); // TODO - fix for multiple summations

        ContractionSolution::TAMetricProblemScore pp
            = computeProblemScore(hardware, M, N, K, NumBatches);

        ContractionSolution::TAMetricProblemScore ppReference
            = computeProblemScore(hardware, model_M, model_N, model_K, model_NumBatches);

        double distance = computeTileAwareMetric(pp, ppReference);

        return distance;
    }

    std::ostream& operator<<(std::ostream&                                      stream,
                             ContractionSolution::StaticPerformanceModel const& spm)
    {
        return stream << " memReadBytesA=" << spm.memReadBytesA
                      << " memReadBytesB=" << spm.memReadBytesB
                      << " memReadBytesC=" << spm.memReadBytesC
                      << " memWriteBytesD=" << spm.memWriteBytesD;
    }

    std::ostream& operator<<(std::ostream&                                    stream,
                             ContractionSolution::ProjectedPerformance const& pp)
    {
        return stream << " numTiles0=" << pp.granularities.numTiles0
                      << " numTiles1=" << pp.granularities.numTiles1
                      << " tilesPerCu=" << pp.granularities.tilesPerCu

                      << " totalGranularity=" << pp.granularities.totalGranularity
                      << " tile0Granularity=" << pp.granularities.tile0Granularity
                      << " tile1Granularity=" << pp.granularities.tile1Granularity
                      << " cuGranularity=" << pp.granularities.cuGranularity
                      << " waveGranularity=" << pp.granularities.waveGranularity

                      << " speedGFlops=" << pp.speedGFlops

                      << " staticModel=[ " << pp.staticModel << " ]";
    }

    std::ostream& operator<<(std::ostream& stream, BufferLoadCheckPacket const& st)
    {
        return stream << " shiftPtrElemA=" << st.shiftPtrElemA
                      << " shiftPtrElemB=" << st.shiftPtrElemB << " depthUorMT0=" << st.depthUorMT0
                      << " depthUorMT1=" << st.depthUorMT1;
    }

} // namespace Tensile

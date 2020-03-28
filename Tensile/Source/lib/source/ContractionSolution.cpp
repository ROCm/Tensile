/**
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <Tensile/ContractionSolution.hpp>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblem.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>

namespace Tensile
{
    int32_t ContractionSolution::staggerUIter(ContractionSolution::Problem const& problem,
                                              ContractionSolution::Inputs  const& inputs,
                                              Hardware    const& hardware) const
    {
        uint32_t sizeL = problem.boundSize(0);

        // how many stride-sized clicks to stagger start offset
        unsigned int staggerUIter = sizeMapping.staggerU;

        // /DepthU/GSU
        int unrollLoopIters = sizeL/sizeMapping.depthU/sizeMapping.globalSplitU;

        unsigned int shifted = 1 << sizeMapping.staggerStrideShift;

        while (staggerUIter>1)
        {
            if (unrollLoopIters >= (staggerUIter * shifted))
                break;

            staggerUIter /= 2; // step down to smaller stagger
        }

        if (staggerUIter>=1) staggerUIter -= 1;

        return staggerUIter;
    }



    // Return magic number.  If magicShift is 0, compute and return it.
    uint32_t ContractionSolution::magicNumberAlg1(uint32_t x, uint32_t *magicShift) const
    {
        uint64_t magicNum;
        *magicShift = 33;
        magicNum = (1L<<*magicShift) / x + 1;
        if ((magicNum >> 32) != 0) {
            *magicShift= 31;
            magicNum = (1L<<*magicShift) / x + 1;
        }

        assert(magicNum >> 32 == 0);  // ensure magic number fits

        return static_cast<uint32_t>(magicNum);
    }

    uint32_t ContractionSolution::magicNumberAlg2(uint32_t d, uint32_t *magicShift) const
    {
        struct mu {
          unsigned M; // Magic number,
          int a; // "add" indicator,
          int s;}; // and shift amount.

        struct mu magu;
        if (d==0) {
            // Make dividend of 0 return 0
            magu.M = 0;
            magu.a = 0;
            magu.s = 0;
        } else {
            // Must have 1 <= d <= 2**32-1.
            int p;
            unsigned nc, delta, q1, r1, q2, r2;
            magu.a = 0; // Initialize "add" indicator.
            nc = -1 - (-d)%d; // Unsigned arithmetic here.
            p = 31; // Init. p.
            q1 = 0x80000000/nc; // Init. q1 = 2**p/nc.
            r1 = 0x80000000 - q1*nc;// Init. r1 = rem(2**p, nc).
            q2 = 0x7FFFFFFF/d; // Init. q2 = (2**p - 1)/d.
            r2 = 0x7FFFFFFF - q2*d; // Init. r2 = rem(2**p - 1, d).
            do {
              p = p + 1;
              if (r1 >= nc - r1) {
                q1 = 2*q1 + 1; // Update q1.
                r1 = 2*r1 - nc;} // Update r1.
              else {
                q1 = 2*q1;
                r1 = 2*r1;}
              if (r2 + 1 >= d - r2) {
                if (q2 >= 0x7FFFFFFF) magu.a = 1;
                q2 = 2*q2 + 1; // Update q2.
                r2 = 2*r2 + 1 - d;} // Update r2.
              else {
                if (q2 >= 0x80000000) magu.a = 1;
                q2 = 2*q2;
                r2 = 2*r2 + 1;}
              delta = d - 1 - r2;
            } while (p < 64 && (q1 < delta || (q1 == delta && r1 == 0)));

            magu.M = q2 + 1; // Magic number
            magu.s = p - 32; // and shift amount to return
        }

        *magicShift = magu.s;
        const uint32_t abit = 0x80000000;
        if (magu.a)
          *magicShift |= abit;

        //std::cout << " d=" << d << " M=" << magu.M << " a=" << magu.a << " s=" << magu.s << "\n";

        return magu.M;
    }

    uint32_t ContractionSolution::magicNumber(int magicDivAlg, uint32_t x, uint32_t *magicShift) const
    {
        if (magicDivAlg==1)
            return magicNumberAlg1(x, magicShift);
        else if (magicDivAlg==2)
            return magicNumberAlg2(x, magicShift);
        else
            throw std::runtime_error("bad magicDivAlg");
    }

    uint32_t ContractionSolution::smallMagicNumber(uint32_t x) const
    {
        uint64_t magicNum;
        const int smallMagicShift=31;
        magicNum = (1L<<smallMagicShift) / x + 1;
        assert(magicNum >> 32 == 0);  // ensure magic number fits
        return static_cast<uint32_t>(magicNum);
    }


    std::vector<size_t> generatePackedIndicesA(ContractionSolution::Problem const &problem,
                                               size_t packBatchDims)
    {
        std::vector<size_t> packedIndices;

        // TODO -move packedIndices calc to problem decode.
        for (auto idx=0; idx<problem.a().dimensions(); idx++)
        {
            bool isSum = problem.boundIndices().end() !=
                          std::find_if(problem.boundIndices().begin(), problem.boundIndices().end(),
                            [idx](const ContractionProblem::BoundIndex &bi)
                            {return bi.a == idx;});

            bool nonPackableBatch = false;
            // TODO - base this check on if the batch is SetConstStrideA=0 - if so, don't pack
            if (!(packBatchDims & 0x1))
            {
                nonPackableBatch = problem.batchIndices().end() !=
                             std::find_if(problem.batchIndices().begin(), problem.batchIndices().end(),
                                [idx](const ContractionProblem::BatchIndex &bi)
                                {return bi.a == idx;});
            }

            if (!isSum && !nonPackableBatch)
                packedIndices.push_back(idx);
        }

        return packedIndices;
    }


    std::vector<size_t> generatePackedIndicesB(ContractionSolution::Problem const &problem,
                                               size_t packBatchDims)
    {
        std::vector<size_t> packedIndices;

        // Pack in all non-summation indices, except don't need magic number for the last one
        for (auto idx=0; idx<problem.b().dimensions(); idx++)
        {
            bool isSum = problem.boundIndices().end() !=
                         std::find_if(problem.boundIndices().begin(), problem.boundIndices().end(),
                            [idx](const ContractionProblem::BoundIndex &bi)
                            {return bi.b == idx;});

            bool nonPackableBatch = false;
            // TODO - base this check on if the batch is SetConstStrideB=0 - if so, don't pack
            if (!(packBatchDims & 0x2))
            {
                nonPackableBatch = problem.batchIndices().end() !=
                             std::find_if(problem.batchIndices().begin(), problem.batchIndices().end(),
                                [idx](const ContractionProblem::BatchIndex &bi)
                                {return bi.b == idx;});
            }

            if (!isSum && !nonPackableBatch)
                packedIndices.push_back(idx);
        }

        return packedIndices;
    }


    template <typename TypedInputs>
    KernelInvocation ContractionSolution::generateSingleCall(ContractionSolution::Problem const& problem,
                                                             TypedInputs                  const& inputs,
                                                             Hardware                     const& hardware) const
    {
        TENSILE_ASSERT_EXC(sizeMapping.workGroupMapping >= 0);

        TensorDescriptor const& a = problem.a();
        TensorDescriptor const& b = problem.b();
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();

        KernelInvocation rv;

        rv.args = KernelArguments(true);

        rv.kernelName = kernelName;

        rv.workGroupSize.x = sizeMapping.workGroupSize.x
                           * sizeMapping.workGroupSize.y
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
            if (sizeMapping.packBatchDims & 0x1)
                rv.numWorkGroups.x *= problem.batchSize(i);
            if (sizeMapping.packBatchDims & 0x2)
                rv.numWorkGroups.y *= problem.batchSize(i);
            if (!sizeMapping.packBatchDims)
                rv.numWorkGroups.z *= problem.batchSize(i);
        }

        rv.numWorkGroups.x = CeilDivide(rv.numWorkGroups.x, sizeMapping.macroTile.x);
        rv.numWorkGroups.y = CeilDivide(rv.numWorkGroups.y, sizeMapping.macroTile.y);

        if (problem.transposeC01())
            std::swap(rv.numWorkGroups.x, rv.numWorkGroups.y);

        uint32_t problemNumGroupTiles0 = rv.numWorkGroups.x;
        uint32_t problemNumGroupTiles1 = rv.numWorkGroups.y;

        rv.numWorkGroups.y *= sizeMapping.globalSplitU;

        if(sizeMapping.persistentKernel != 0)
        {
            size_t persistentGroups = dynamic_cast<AMDGPU const&>(hardware).computeUnitCount * sizeMapping.persistentKernel;
            size_t problemGroups = rv.numWorkGroups.x * rv.numWorkGroups.y;

            rv.numWorkGroups.x = std::min(persistentGroups, problemGroups);
            rv.numWorkGroups.y = 1;
        }

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        if(debugKernel)
        {
            rv.args.appendUnbound<unsigned int *>("debugBuffer");
        }

        rv.sharedMemBytes = 0;

        if(!isSourceKernel())
        {
            uint64_t tensor2dSizeC = 0;
            uint64_t tensor2dSizeA = (sizeMapping.packBatchDims & 0x1) ? a.totalAllocatedElements() : problem.allocatedElementsNonBatchA() ;
            uint64_t tensor2dSizeB = (sizeMapping.packBatchDims & 0x2) ? b.totalAllocatedElements() : problem.allocatedElementsNonBatchB() ;

            rv.args.append<uint64_t>("tensor2dSizeC", tensor2dSizeC);
            rv.args.append<uint64_t>("tensor2dSizeA", tensor2dSizeA);
            rv.args.append<uint64_t>("tensor2dSizeB", tensor2dSizeB);
        }

        rv.args.append<typename TypedInputs::DType const *>("d", inputs.d);
        rv.args.append<typename TypedInputs::CType const *>("c", inputs.c);
        rv.args.append<typename TypedInputs::AType const *>("a", inputs.a);
        rv.args.append<typename TypedInputs::BType const *>("b", inputs.b);

        rv.args.append<typename TypedInputs::AlphaType>("alpha", inputs.alpha);
        if(std::is_same<typename TypedInputs::AlphaType, Half>::value && !isSourceKernel())
            rv.args.append<typename TypedInputs::AlphaType>("alpha_2", inputs.alpha);

        if(problemType.useBeta)
        {
            rv.args.append<typename TypedInputs::BetaType>("beta", inputs.beta);
            if(std::is_same<typename TypedInputs::BetaType, Half>::value && !isSourceKernel())
                rv.args.append<typename TypedInputs::BetaType>("beta_2", inputs.beta);
        }

        size_t startStrideCD = problemType.useInitialStridesCD ? 0:1;
        size_t startStrideAB = problemType.useInitialStridesAB ? 0:1;

        for(size_t i = startStrideCD; i < d.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideD", i), d.strides()[i]);

        for(size_t i = startStrideCD; i < c.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideC", i), c.strides()[i]);

        for(size_t i = startStrideAB; i < a.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideA", i), a.strides()[i]);

        for(size_t i = startStrideAB; i < b.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideB", i), b.strides()[i]);

        {
            int idx=0;
            for(auto size: problem.problemSizes())
            {
                rv.args.append<uint32_t>(concatenate("size_",idx), size);
                idx++;
            }
        }

        if (sizeMapping.packSummationDims)
            // boundIndices are ordered with unroll last.
            // Magic numbers for all but first are needed to unpack other dims.
            for (auto si=1; si<problem.boundIndices().size(); si++)
            {
                auto numIter = problem.boundSize(si);
                bool isUnroll = si==problem.boundIndices().size()-1;
                if (isUnroll) {
                    numIter = numIter / sizeMapping.depthU / sizeMapping.globalSplitU * sizeMapping.depthU;
                }
                uint32_t magicShift;
                rv.args.append<uint32_t>(concatenate("magicNumberNumIter_",si),
                                          magicNumber(sizeMapping.magicDivAlg, numIter, &magicShift));
                rv.args.append<uint32_t>(concatenate("magicShiftNumIter_",si), magicShift);

                if (isUnroll and sizeMapping.globalSplitU>1)
                {
                    // compute magic number for gsu remainder iterations:
                    // Kernel will select whether to use above or remainder portion based on work-group assignment
                    rv.args.append<uint32_t>(concatenate("magicNumberNumIter_GsuRemainder"),
                                              magicNumber(sizeMapping.magicDivAlg, 
                                                numIter+sizeMapping.depthU, &magicShift));
                    rv.args.append<uint32_t>(concatenate("magicShiftNumIter_GsuRemainder"), magicShift);
                }
            }

        if (problem.freeIndicesA().size() > 1 || sizeMapping.packBatchDims & 0x1)
        {
            std::vector<size_t> packedIndices = generatePackedIndicesA(problem, sizeMapping.packBatchDims);

            // Pack in all non-summation indices, except don't need magic number for the last one
            for (auto pi=packedIndices.begin(); pi!=packedIndices.end()-1; pi++)
            {
                auto idx = *pi;
                auto size = a.sizes()[idx];
                uint32_t magicShift;
                rv.args.append<uint32_t>(concatenate("magicNumberSizeA_",idx),
                                          magicNumber(sizeMapping.magicDivAlg, size, &magicShift));
                rv.args.append<uint32_t>(concatenate("magicShiftSizeA_",idx), magicShift);
            }
        }
        if (problem.freeIndicesB().size() > 1 || sizeMapping.packBatchDims & 0x2)
        {
            std::vector<size_t> packedIndices = generatePackedIndicesB(problem, sizeMapping.packBatchDims);

            // Pack in all non-summation indices, except don't need magic number for the last one
            for (auto pi=packedIndices.begin(); pi!=packedIndices.end()-1; pi++)
            {
                auto idx = *pi;
                auto size = b.sizes()[idx];
                uint32_t magicShift;
                rv.args.append<uint32_t>(concatenate("magicNumberSizeB_",idx),
                                          magicNumber(sizeMapping.magicDivAlg, size, &magicShift));
                rv.args.append<uint32_t>(concatenate("magicShiftSizeB_",idx), magicShift);
            }
        }

        for (auto si : problem.boundIndices())
        {
            if (si.aZeroPad.valid())
            {
                rv.args.append<int32_t>(concatenate("padStartA_",si.a),
                            si.aZeroPad.padStart);
                rv.args.append<int32_t>(concatenate("padEndA_",si.a),
                            si.aZeroPad.padEnd );
            }
            if (si.bZeroPad.valid())
            {
                rv.args.append<int32_t>(concatenate("padStartB_",si.b),
                            si.bZeroPad.padStart);
                rv.args.append<int32_t>(concatenate("padEndB_",si.b),
                            si.bZeroPad.padEnd);
            }
        }

        rv.args.append< int32_t>("staggerUIter", staggerUIter(problem, inputs, hardware));

        rv.args.append<uint32_t>("problemNumGroupTiles0", problemNumGroupTiles0);
        rv.args.append<uint32_t>("problemNumGroupTiles1", problemNumGroupTiles1);
        rv.args.append<uint32_t>("magicNumberProblemNumGroupTiles0", smallMagicNumber(problemNumGroupTiles0));

        if(!isSourceKernel())
        {
            rv.args.append<uint32_t>("gridNumWorkGroups0", rv.numWorkGroups.x);

            uint32_t numFullBlocks = problemNumGroupTiles1;
            uint32_t wgmRemainder1 = 0;
            uint32_t magicNumberWgmRemainder1 = 0;

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

            rv.args.append<uint32_t>("pad", 0);
        }

        return rv;
    }

    bool ContractionSolution::isSourceKernel() const
    {
        return sizeMapping.sourceKernel;
    }

    template <typename TypedInputs>
    KernelInvocation ContractionSolution::generateBetaOnlyCall(Problem     const& problem,
                                                               TypedInputs const& inputs,
                                                               Hardware    const& hardware) const
    {
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();

        KernelInvocation rv;

        rv.args = KernelArguments(true);

        rv.kernelName = betaOnlyKernelName(problem, inputs, hardware);

        rv.workGroupSize.x = 8;
        rv.workGroupSize.y = 8;
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

        rv.numWorkGroups.x = CeilDivide(wiX, rv.workGroupSize.x);
        rv.numWorkGroups.y = CeilDivide(wiY, rv.workGroupSize.y);
        rv.numWorkGroups.z = CeilDivide(wiZ, rv.workGroupSize.z);

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        rv.args.append<typename TypedInputs::DType      *>("D", inputs.d);
        rv.args.append<typename TypedInputs::CType const*>("C", inputs.c);

        for(size_t i = 1; i < d.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideD", i), d.sizes()[i] == 1 ? 0 : d.strides()[i]);

        for(size_t i = 1; i < c.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideC", i), c.sizes()[i] == 1 ? 0 : c.strides()[i]);

        int idx=0;
        for(auto size: problem.d().sizes())
        {
            rv.args.append<uint32_t>(concatenate("size_",idx), size);
            idx++;
        }

        if(inputs.beta != static_cast<typename TypedInputs::BetaType>(0))
        {
            rv.args.append<typename TypedInputs::BetaType>("beta", inputs.beta);
        }

        return rv;
    }

    template <typename TypedInputs>
    std::string ContractionSolution::betaOnlyKernelName(Problem     const& problem,
                                                        TypedInputs const& inputs,
                                                        Hardware    const& hardware) const
    {
        std::string name = concatenate("C", problem.cNames(),
                                        "_",
                                        TypeInfo<typename TypedInputs::DType>::Abbrev());

        if(inputs.beta != static_cast<typename TypedInputs::BetaType>(0))
        {
            name += "B";
        }
        return name;
    }

    template <typename TypedInputs>
    std::vector<KernelInvocation>
    ContractionSolution::solveTyped(Problem     const& problem,
                                    TypedInputs const& inputs,
                                    Hardware    const& hardware) const
    {
        std::vector<KernelInvocation> rv;

        if(sizeMapping.globalSplitU > 1)
            rv.reserve(2);
        else
            rv.reserve(1);

        if(sizeMapping.globalSplitU > 1)
            rv.push_back(generateBetaOnlyCall(problem, inputs, hardware));

        rv.push_back(generateSingleCall(problem, inputs, hardware));

        return rv;
    }


    std::vector<KernelInvocation>
    ContractionSolution::solve(ContractionSolution::Problem const& problem,
                               ContractionSolution::Inputs  const& inputs,
                               Hardware                     const& hardware) const
    {
        if(problemType.aType == DataType::Float
        && problemType.bType == DataType::Float
        && problemType.cType == DataType::Float
        && problemType.dType == DataType::Float)
        {
            auto const& typedInputs = dynamic_cast<TypedContractionInputs<float> const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        else if(problemType.aType == DataType::Double
             && problemType.bType == DataType::Double
             && problemType.cType == DataType::Double
             && problemType.dType == DataType::Double)
        {
            auto const& typedInputs = dynamic_cast<TypedContractionInputs<double> const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        else if(problemType.aType == DataType::ComplexFloat
             && problemType.bType == DataType::ComplexFloat
             && problemType.cType == DataType::ComplexFloat
             && problemType.dType == DataType::ComplexFloat)
        {
            auto const& typedInputs = dynamic_cast<TypedContractionInputs<std::complex<float>> const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        else if(problemType.aType == DataType::ComplexDouble
             && problemType.bType == DataType::ComplexDouble
             && problemType.cType == DataType::ComplexDouble
             && problemType.dType == DataType::ComplexDouble)
        {
            auto const& typedInputs = dynamic_cast<TypedContractionInputs<std::complex<double>> const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#ifdef TENSILE_USE_HALF
        else if(problemType.aType == DataType::Half
             && problemType.bType == DataType::Half
             && problemType.cType == DataType::Half
             && problemType.dType == DataType::Half)
        {
            auto const& typedInputs = dynamic_cast<TypedContractionInputs<Half> const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#endif
        else if(problemType.aType == DataType::Int8x4
             && problemType.bType == DataType::Int8x4
             && problemType.cType == DataType::Int32
             && problemType.dType == DataType::Int32)
        {
            auto const& typedInputs =
                dynamic_cast<TypedContractionInputs<Int8x4, Int8x4, int32_t, int32_t> const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        else if(problemType.aType == DataType::Int32
             && problemType.bType == DataType::Int32
             && problemType.cType == DataType::Int32
             && problemType.dType == DataType::Int32)
        {
            auto const& typedInputs = dynamic_cast<TypedContractionInputs<int32_t> const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#ifdef TENSILE_USE_BF16
        else if(problemType.aType == DataType::BFloat16
             && problemType.bType == DataType::BFloat16
             && problemType.cType == DataType::BFloat16
             && problemType.dType == DataType::BFloat16)
        {
            auto const& typedInputs = dynamic_cast<BFloat16ContractionInputs const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
#endif
        else
        {
            throw std::runtime_error("Data type not implemented.");
        }
    }

    ContractionSolution::StaticPerformanceModel ContractionSolution::staticPerformanceModel
        (double M, double N, double K, double NumBatches, double MT0, double MT1,
         double NumCUs, double TotalGranularity, int GlobalSplitU) const
    {
        StaticPerformanceModel spm;
        
        int beta = (int)problemType.useBeta;
        int betaReads=0, betaWrites=0;
        if (GlobalSplitU==1)
        {
            if (beta != 0.0)
                betaReads = 1.0;
        }
        else
        {
            if (beta == 0)
                betaWrites = 1; // zero output
            else if (beta != 1.0) // if 1.0, just atomic update output
            {
                // if not 1.0, read, scale, write, then atomic update in kernel
                betaReads = 1; // initial read for scale
                betaWrites = 1; // writeback after scale
            }
        }

        auto aInfo = DataTypeInfo::Get(problemType.aType);
        auto bInfo = DataTypeInfo::Get(problemType.bType);
        auto cInfo = DataTypeInfo::Get(problemType.cType);
        auto dInfo = DataTypeInfo::Get(problemType.dType);
        
        double l2ReadBwMultiplier = perf.l2ReadBwMul;
        spm.memReadBytesA = (NumBatches*M*N*K)/MT1 * aInfo.elementSize;
        spm.memReadBytesB = (NumBatches*M*N*K)/MT0 * bInfo.elementSize;
        spm.memReadBytesC = (NumBatches*M*N) * betaReads * cInfo.elementSize;
        
        if (GlobalSplitU == 1)
            spm.memWriteBytesD   = (NumBatches*M*N)*(1+betaWrites)*dInfo.elementSize;
        else
        {
            bool hardwareAtomic = false;  //TODO-model
            double atomicOperations = hardwareAtomic ? 2:3; //read-mod-write or cas  //TODO-model
            double atomicCollisions = 1.0;  //TODO-could be based on K, GSU
            spm.memWriteBytesD   = (NumBatches*M*N) * (betaWrites + atomicOperations * atomicCollisions) * dInfo.elementSize;
        }
        spm.memReadBytes = spm.memReadBytesA + spm.memReadBytesB + spm.memReadBytesC;
        spm.memGlobalReads = spm.memReadBytesA/aInfo.elementSize + spm.memReadBytesB/bInfo.elementSize + spm.memReadBytesC/cInfo.elementSize;
        spm.memGlobalWrites = spm.memWriteBytesD/dInfo.elementSize;
       
        double readEfficiency = perf.readEff;         
        double l2ReadHit = perf.l2ReadHitRate;
        double l2WriteHit = perf.l2WriteHitRate;
        double frequency = perf.clock;
        double memFrequency = perf.memClock;
        double memBandwidthMBps = perf.memBandwidthMBps;
        double l2BandwidthMBps = perf.memBandwidthMBps*perf.l2ReadBwMul;        
        double peakMFlops = perf.peakGFlops*1000.0;
 
        spm.memReadUs  = (spm.memReadBytes*l2ReadHit/l2BandwidthMBps + spm.memReadBytes*(1.0-l2ReadHit))/memBandwidthMBps;
        spm.memWriteUs = (spm.memWriteBytesD * l2WriteHit/l2BandwidthMBps + spm.memWriteBytesD * (1.0 - l2WriteHit)) / l2BandwidthMBps;

        double flops = 2.0*l2ReadBwMultiplier*NumBatches*M*N*K;
        spm.aluUs = flops/(peakMFlops*TotalGranularity);

        return spm;
    }

    ContractionSolution::ProjectedPerformance ContractionSolution::projectedPerformance(
        Problem const& problem, Hardware const& hardware) const
    {
        ProjectedPerformance pp;

        double M=1.0, N=1.0;
        if (problem.freeIndicesA().size() > 1 || sizeMapping.packBatchDims & 0x1)
        {
            std::vector<size_t> packedIndices = generatePackedIndicesA(problem, sizeMapping.packBatchDims);
            for (auto pi=packedIndices.begin(); pi!=packedIndices.end(); pi++)
                M *= problem.a().sizes()[*pi];
        } else
            M = problem.freeSizeA(0);

        if (problem.freeIndicesB().size() > 1 || sizeMapping.packBatchDims & 0x2)
        {
            std::vector<size_t> packedIndices = generatePackedIndicesB(problem, sizeMapping.packBatchDims);
            for (auto pi=packedIndices.begin(); pi!=packedIndices.end(); pi++)
                N *= problem.b().sizes()[*pi];
        }
        else
            N = problem.freeSizeB(0);


        double NumBatches = 1;
        if (sizeMapping.packBatchDims == 0)
        {
            for(size_t i = 0; i < problem.batchIndices().size(); i++)
                NumBatches *= problem.batchSize(i);
        }
        double K = problem.boundSize(0); // TODO - fix for multiple summations

        auto it = ideals.begin();

        int closestK = -1;
        int closestKMeasure = std::numeric_limits<int>::max();
        double closestKPerformance = 0.0;

        while(it != ideals.end())
        {
            int myK = it->first;
            int myMeasure = std::abs(myK - K);
            if (myMeasure < closestKMeasure)
            {
                closestKMeasure = myMeasure;
                closestK = myK;
                closestKPerformance = it->second;
            }
            it++;
        }

        double MT0 = sizeMapping.macroTile.x;
        double MT1 = sizeMapping.macroTile.y;

        double NumCUs = perf.CUs;

        AMDGPU const *pAMDGPU = dynamic_cast<AMDGPU const *>(&hardware);
        if (pAMDGPU != nullptr)
        {
            NumCUs = pAMDGPU->computeUnitCount;
        }

        double GlobalSplitU = sizeMapping.globalSplitU;
        double LocalSplitU = sizeMapping.workGroupSize.z;
        double IdealGranularityPerf = closestKPerformance;

        pp.numTiles0 = M / MT0;
        pp.numTiles1 = N / MT1;

        pp.tilesPerCu = (NumBatches * ceil(pp.numTiles0) * ceil(pp.numTiles1)) /
                          (NumCUs / GlobalSplitU / LocalSplitU);
        pp.tile0Granularity = pp.numTiles0/ceil(pp.numTiles0);
        pp.tile1Granularity = pp.numTiles1/ceil(pp.numTiles1);

        pp.waveGranularity = std::min(1.00,
                                static_cast<double>(
                                floor(pp.tilesPerCu + 1.0) *
                                sizeMapping.workGroupSize.x*
                                sizeMapping.workGroupSize.y*
                                sizeMapping.workGroupSize.z)
                                / pAMDGPU->wavefrontSize / pAMDGPU->simdPerCu);

        pp.cuGranularity = pp.tilesPerCu / ceil(pp.tilesPerCu);
        pp.totalGranularity = pp.tile0Granularity * pp.tile1Granularity * pp.cuGranularity * pp.waveGranularity;

        pp.speedGFlops = IdealGranularityPerf * pp.totalGranularity;

        pp.staticModel = staticPerformanceModel(M, N, K, NumBatches, MT0, MT1, NumCUs, pp.totalGranularity, GlobalSplitU);

        return pp;
    }

    std::ostream & operator<<(std::ostream & stream, ContractionSolution::StaticPerformanceModel const& spm)
    {
        return stream
            << " memReadBytesA=" << spm.memReadBytesA
            << " memReadBytesB=" << spm.memReadBytesB
            << " memReadBytesC=" << spm.memReadBytesC
            << " memWriteBytesD=" << spm.memWriteBytesD

            << " aluUs=" << spm.aluUs
            << " memReadUs=" << spm.memReadUs
            << " memWriteUs=" << spm.memWriteUs
            ;
    }

    std::ostream & operator<<(std::ostream & stream, ContractionSolution::ProjectedPerformance const& pp)
    {
        return stream
            << " numTiles0=" << pp.numTiles0
            << " numTiles1=" << pp.numTiles1
            << " tilesPerCu=" << pp.tilesPerCu

            << " totalGranularity=" << pp.totalGranularity
            << " tile0Granularity=" << pp.tile0Granularity
            << " tile1Granularity=" << pp.tile1Granularity
            << " cuGranularity=" << pp.cuGranularity
            << " waveGranularity=" << pp.waveGranularity

            << " speedGFlops=" << pp.speedGFlops

            << " staticModel=[ " << pp.staticModel << " ]"
            ;
    }

}


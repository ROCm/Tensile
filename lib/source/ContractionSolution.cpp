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

#include <Tensile/ContractionProblem.hpp>

namespace Tensile
{
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

    uint32_t ContractionSolution::magicNumber(uint32_t x) const
    {
        // TODO: bozo, review
        uint32_t magicShift = 31;
        return (1L << magicShift) / x + 1;
    }

    template <typename TypedInputs>
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

        rv.args = KernelArguments(true);

        rv.kernelName = kernelName;

        rv.workGroupSize.x = sizeMapping.workGroupSize.x * sizeMapping.workGroupSize.y
                             * sizeMapping.workGroupSize.z;
        rv.workGroupSize.y = 1;
        rv.workGroupSize.z = 1;

        rv.numWorkGroups.x = CeilDivide(c.sizes()[0], sizeMapping.macroTile.x);
        rv.numWorkGroups.y = CeilDivide(c.sizes()[1], sizeMapping.macroTile.y);
        rv.numWorkGroups.z = c.sizes()[2];

        uint32_t problemNumGroupTiles0 = rv.numWorkGroups.x;
        uint32_t problemNumGroupTiles1 = rv.numWorkGroups.y;

        rv.numWorkGroups.y *= sizeMapping.globalSplitU;

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        if(debugKernel)
        {
            rv.args.appendUnbound<unsigned int*>("debugBuffer");
        }

        rv.sharedMemBytes = 0;

        rv.args.append<uint64_t>("tensor2dSizeC", c.strides()[2]);
        rv.args.append<uint64_t>("tensor2dSizeA", a.strides()[2]);
        rv.args.append<uint64_t>("tensor2dSizeB", b.strides()[2]);

        rv.args.append<float*>("d", inputs.d);
        rv.args.append<float const*>("c", inputs.c);
        rv.args.append<float const*>("a", inputs.a);
        rv.args.append<float const*>("b", inputs.b);

        rv.args.append<float>("alpha", inputs.alpha);
        rv.args.append<float>("beta", inputs.beta);

        for(size_t i = 1; i < d.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideD", i),
                                     d.sizes()[i] == 1 ? 0 : d.strides()[i]);

        for(size_t i = 1; i < c.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideC", i),
                                     c.sizes()[i] == 1 ? 0 : c.strides()[i]);

        for(size_t i = 1; i < a.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideA", i),
                                     a.sizes()[i] == 1 ? 0 : a.strides()[i]);

        for(size_t i = 1; i < b.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideB", i),
                                     b.sizes()[i] == 1 ? 0 : b.strides()[i]);

        rv.args.append<uint32_t>("sizeI", problem.freeSizeA(0));
        rv.args.append<uint32_t>("sizeJ", problem.freeSizeB(0));
        rv.args.append<uint32_t>("sizeK", problem.batchSize(0));
        rv.args.append<uint32_t>("sizeL", problem.boundSize(0));

        rv.args.append<int32_t>("staggerUIter", staggerUIter(problem, inputs, hardware));

        rv.args.append<uint32_t>("problemNumGroupTiles0", problemNumGroupTiles0);
        rv.args.append<uint32_t>("problemNumGroupTiles1", problemNumGroupTiles1);
        rv.args.append<uint32_t>("magicNumberProblemNumGroupTiles0",
                                 magicNumber(problemNumGroupTiles0));
        rv.args.append<uint32_t>("gridNumWorkGroups0", rv.numWorkGroups.x);

        uint32_t numFullBlocks            = problemNumGroupTiles1;
        uint32_t wgmRemainder1            = 0;
        uint32_t magicNumberWgmRemainder1 = 0;

        if(sizeMapping.workGroupMapping != 0)
        {
            numFullBlocks = problemNumGroupTiles1 / sizeMapping.workGroupMapping;
            wgmRemainder1 = problemNumGroupTiles1 % sizeMapping.workGroupMapping;
            if(wgmRemainder1 == 0)
                wgmRemainder1 = sizeMapping.workGroupMapping;
            magicNumberWgmRemainder1 = magicNumber(wgmRemainder1);
        }

        rv.args.append<uint32_t>("numFullBlocks", numFullBlocks);
        rv.args.append<uint32_t>("wgmRemainder1", wgmRemainder1);
        rv.args.append<uint32_t>("magicNumberWgmRemainder1", magicNumberWgmRemainder1);

        rv.args.append<uint32_t>("pad", 0);

        return rv;
    }

    template <typename TypedInputs>
    KernelInvocation ContractionSolution::generateBetaOnlyCall(Problem const&     problem,
                                                               TypedInputs const& inputs,
                                                               Hardware const&    hardware) const
    {
        TensorDescriptor const& c = problem.c();
        TensorDescriptor const& d = problem.d();

        KernelInvocation rv;

        rv.args = KernelArguments(true);

        rv.kernelName = betaOnlyKernelName(problem, inputs, hardware);

        rv.workGroupSize.x = 8;
        rv.workGroupSize.y = 8;
        rv.workGroupSize.z = 1;

        rv.numWorkGroups.x = CeilDivide(c.sizes()[0], rv.workGroupSize.x);
        rv.numWorkGroups.y = CeilDivide(c.sizes()[1], rv.workGroupSize.y);
        rv.numWorkGroups.z = c.sizes()[2];

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        rv.args.append<typename TypedInputs::DType*>("D", inputs.d);
        rv.args.append<typename TypedInputs::CType const*>("C", inputs.c);

        for(size_t i = 1; i < d.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideD", i),
                                     d.sizes()[i] == 1 ? 0 : d.strides()[i]);

        for(size_t i = 1; i < c.dimensions(); i++)
            rv.args.append<uint32_t>(concatenate("strideC", i),
                                     c.sizes()[i] == 1 ? 0 : c.strides()[i]);

        rv.args.append<uint32_t>("sizeI", problem.freeSizeA(0));
        rv.args.append<uint32_t>("sizeJ", problem.freeSizeB(0));
        rv.args.append<uint32_t>("sizeK", problem.batchSize(0));

        if(inputs.beta != static_cast<typename TypedInputs::BetaType>(0))
        {
            rv.args.append<typename TypedInputs::BetaType>("beta", inputs.beta);
            //rv.args.append<typename TypedInputs::BetaType>("beta", 0);
        }

        return rv;
    }

    template <typename TypedInputs>
    std::string ContractionSolution::betaOnlyKernelName(Problem const&     problem,
                                                        TypedInputs const& inputs,
                                                        Hardware const&    hardware) const
    {
        if(inputs.beta == static_cast<typename TypedInputs::BetaType>(0))
        {
            return "Cijk_S";
        }
        else
        {
            return "Cijk_SB";
        }
    }

    template <typename TypedInputs>
    std::vector<KernelInvocation> ContractionSolution::solveTyped(Problem const&     problem,
                                                                  TypedInputs const& inputs,
                                                                  Hardware const&    hardware) const
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
                                   ContractionSolution::Inputs const&  inputs,
                                   Hardware const&                     hardware) const
    {
        if(problemType.aType == DataType::Float && problemType.bType == DataType::Float
           && problemType.cType == DataType::Float && problemType.dType == DataType::Float)
        {
            auto const& typedInputs = dynamic_cast<TypedContractionInputs<float> const&>(inputs);
            return solveTyped(problem, typedInputs, hardware);
        }
        else
        {
            throw std::runtime_error("Data type not implemented.");
        }
    }

}

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

#include <Tensile/GEMMSolution.hpp>

namespace Tensile
{
    int32_t GEMMSolution::staggerUIter(GEMMProblem const& problem,
                                       GEMMInputs  const& inputs,
                                       Hardware    const& hardware) const
    {
        uint32_t sizeL = problem.tensile_L();

        unsigned int staggerUIter = 32; // how many stride-sized clicks to stagger start offset
        int unrollLoopIters = sizeL/8/1; // /DepthU/GSU
        while (staggerUIter>1) {
          if (unrollLoopIters >= (staggerUIter*8)) {
            break;}
          staggerUIter /= 2; // step down to smaller stagger
        }
        if (staggerUIter>=1) staggerUIter -= 1;

        return staggerUIter;
    }

    uint32_t GEMMSolution::magicNumber(uint32_t x) const
    {
        // TODO: bozo, review
        uint32_t magicShift = 31;
        return (1L<<magicShift) / x + 1;
    }

    KernelInvocation GEMMSolution::generateSingleCall(GEMMProblem const& problem,
                                                      GEMMInputs  const& inputs,
                                                      Hardware    const& hardware) const
    {
        KernelInvocation rv;

        rv.args = KernelArguments(true);

        rv.solution = this;

        rv.workGroupSize = workGroupSize;

        std::vector<size_t> cSize = problem.c.sizes();

        rv.numWorkGroups.x = CeilDivide(cSize[0], macroTile.x);
        rv.numWorkGroups.y = CeilDivide(cSize[1], macroTile.y);
        rv.numWorkGroups.z = cSize[2];

        rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x;
        rv.numWorkItems.y = rv.workGroupSize.y * rv.numWorkGroups.y;
        rv.numWorkItems.z = rv.workGroupSize.z * rv.numWorkGroups.z;

        if(debugKernel)
        {
            rv.args.appendUnbound<unsigned int *>("debugBuffer");
        }

        auto aStrides = problem.a.strides();
        auto bStrides = problem.b.strides();
        auto cStrides = problem.c.strides();

        rv.sharedMemBytes = 0;

        rv.args.append<uint64_t>("tensor2dSizeC", problem.c.totalAllocatedElements());
        rv.args.append<uint64_t>("tensor2dSizeA", problem.a.totalAllocatedElements());
        rv.args.append<uint64_t>("tensor2dSizeB", problem.b.totalAllocatedElements());

        rv.args.append<float       *>("d", inputs.d);
        rv.args.append<float const *>("c", inputs.c);
        rv.args.append<float const *>("a", inputs.a);
        rv.args.append<float const *>("b", inputs.b);

        rv.args.append<float>("alpha", inputs.alpha);
        rv.args.append<float>("beta",  inputs.beta);

        //rv.args.append<uint32_t>("offsetC", 0);
        //rv.args.append<uint32_t>("offsetA", 0);
        //rv.args.append<uint32_t>("offsetB", 0);

        rv.args.append<uint32_t>("strideC1", problem.tensile_strideC1());
        rv.args.append<uint32_t>("strideC2", problem.tensile_strideC2());

        rv.args.append<uint32_t>("strideA1", problem.tensile_strideA1());
        rv.args.append<uint32_t>("strideA2", problem.tensile_strideA2());

        rv.args.append<uint32_t>("strideB1", problem.tensile_strideB1());
        rv.args.append<uint32_t>("strideB2", problem.tensile_strideB2());

        rv.args.append<uint32_t>("sizeI", problem.tensile_I());
        rv.args.append<uint32_t>("sizeJ", problem.tensile_J());
        rv.args.append<uint32_t>("sizeK", problem.tensile_K());
        rv.args.append<uint32_t>("sizeL", problem.tensile_L());

        rv.args.append<uint32_t>("staggerUIter", staggerUIter(problem, inputs, hardware));

        rv.args.append<uint32_t>("problemNumGroupTiles0", rv.numWorkGroups.x);
        rv.args.append<uint32_t>("problemNumGroupTiles1", rv.numWorkGroups.y);
        rv.args.append<uint32_t>("magicNumberProblemNumGroupTiles0", magicNumber(rv.numWorkGroups.x));
        rv.args.append<uint32_t>("gridNumWorkGroups0", rv.numWorkGroups.x);

        rv.args.append<uint32_t>("pad", 0);

        return rv;
    }

    std::vector<KernelInvocation>
    GEMMSolution::solve(GEMMProblem const& problem,
                        GEMMInputs  const& inputs,
                        Hardware    const& hardware) const
    {
        std::vector<KernelInvocation> rv;
        
        rv.push_back(generateSingleCall(problem, inputs, hardware));

        return rv;
    }

}

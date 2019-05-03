/**
 * MIT License
 *
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

#include "Reference.hpp"
//#include <ReferenceCPU.h>

namespace Tensile
{
    namespace Client
    {
        template <typename Inputs>
        void ReferenceSolution<Inputs>::SolveCPU(ContractionProblem const& problem, Inputs const& inputs)
        {
            auto const& freeIndices = problem.freeIndices();
            auto const& batchIndices = problem.batchIndices();
            auto const& boundIndices = problem.boundIndices();

            auto const& a = problem.a();
            auto const& b = problem.b();
            auto const& c = problem.c();
            auto const& d = problem.d();

            std::vector<size_t> freeASize(problem.freeIndices().size());
            std::vector<size_t> freeBSize(problem.freeIndices().size());
            std::vector<size_t> batchSize(problem.batchIndices().size());
            std::vector<size_t> boundSize(problem.boundIndices().size());

            for(int i = 0; i < freeASize.size(); i++) freeASize[i] = problem.freeSizeA(i);
            for(int i = 0; i < freeBSize.size(); i++) freeBSize[i] = problem.freeSizeB(i);
            for(int i = 0; i < batchSize.size(); i++) batchSize[i] = problem.batchSize(i);
            for(int i = 0; i < boundSize.size(); i++) boundSize[i] = problem.boundSize(i);


            auto batchCount = CoordCount(batchSize.begin(), batchSize.end());
            auto freeACount = CoordCount(freeASize.begin(), freeASize.end());
            auto freeBCount = CoordCount(freeBSize.begin(), freeBSize.end());
            auto boundCount = CoordCount(boundSize.begin()+1, boundSize.end());

#pragma omp parallel for collapse(3)
            for(size_t batchNum = 0; batchNum < batchCount; batchNum++)
            for(size_t freeANum = 0; freeANum < freeACount; freeANum++)
            for(size_t freeBNum = 0; freeBNum < freeBCount; freeBNum++)
            {
                std::vector<size_t> aCoord(problem.a().dimensions());
                std::vector<size_t> bCoord(problem.b().dimensions());
                std::vector<size_t> cCoord(problem.c().dimensions());
                std::vector<size_t> dCoord(problem.d().dimensions());

                std::vector<size_t> batch(problem.batchIndices().size());
                CoordNumbered(batchNum, batch.begin(), batch.end(), batchSize.begin(), batchSize.end());

                std::vector<size_t> freeA(problem.freeIndices().size());
                CoordNumbered(freeANum, freeA.begin(), freeA.end(), freeASize.begin(), freeASize.end());
                std::vector<size_t> freeB(problem.freeIndices().size());
                CoordNumbered(freeBNum, freeB.begin(), freeB.end(), freeBSize.begin(), freeBSize.end());

                for(int i = 0; i < batch.size(); i++)
                {
                    aCoord[batchIndices[i].a] = batch[i];
                    bCoord[batchIndices[i].b] = batch[i];
                    cCoord[batchIndices[i].c] = batch[i];
                    dCoord[batchIndices[i].d] = batch[i];
                }

                for(int i = 0; i < freeA.size(); i++)
                {
                    aCoord[freeIndices[i].a ] = freeA[i];
                    cCoord[freeIndices[i].ca] = freeA[i];
                    dCoord[freeIndices[i].da] = freeA[i];
                }

                for(int i = 0; i < freeB.size(); i++)
                {
                    bCoord[freeIndices[i].b ] = freeB[i];
                    cCoord[freeIndices[i].cb] = freeB[i];
                    dCoord[freeIndices[i].db] = freeB[i];
                }


                typename Inputs::DType value = 0;

                for(size_t boundNum = 0; boundNum < boundCount; boundNum++)
                {
                    std::vector<size_t> bound(problem.boundIndices().size());
                    CoordNumbered(boundNum, bound.begin()+1, bound.end(), boundSize.begin()+1, boundSize.end());

                    for(int i = 1; i < bound.size(); i++)
                    {
                        aCoord[boundIndices[i].a] = bound[i];
                        bCoord[boundIndices[i].b] = bound[i];
                    }

                    auto aIndex = a.index(aCoord);
                    auto bIndex = b.index(bCoord);

                    auto aStride = problem.a().strides()[boundIndices[0].a];
                    auto bStride = problem.b().strides()[boundIndices[0].b];

                    for(size_t i = 0; i < boundSize[0]; i++)
                        value += inputs.a[aIndex + (i * aStride)] * inputs.b[bIndex + (i * bStride)];
                }

                auto cIndex = c.index(cCoord);
                auto dIndex = d.index(cCoord);

                inputs.d[dIndex] = inputs.alpha * value
                                 + inputs.beta * inputs.c[cIndex];

            }
        }

        void SolveCPU(ContractionProblem const& problem, ContractionInputs const& inputs)
        {
            if(problem.a().dataType() == DataType::Float
            && problem.b().dataType() == DataType::Float
            && problem.c().dataType() == DataType::Float
            && problem.d().dataType() == DataType::Float)
            {
                auto const& typedInputs = dynamic_cast<TypedContractionInputs<float> const&>(inputs);
                return ReferenceSolution<TypedContractionInputs<float>>::SolveCPU(problem, typedInputs);
            }
            else
            {
                throw std::runtime_error("Data type not implemented.");
            }
        }
    }
}



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

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        template <typename T>
        struct Transform
        {
            inline static T Input(T const& val, bool conj)
            {
                return val;
            }
        };

        template <typename T>
        struct Transform<std::complex<T>>
        {
            inline static std::complex<T> Input(std::complex<T> const& val, bool conj)
            {
                if(conj)
                    return std::conj(val);

                return val;
            }
        };

        template <typename Inputs, typename Accumulator>
        void ReferenceSolution<Inputs, Accumulator>::SolveCPU(ContractionProblem const& problem, Inputs const& inputs,
                                                              size_t validationStride)
        {
            auto const& freeIndicesA = problem.freeIndicesA();
            auto const& freeIndicesB = problem.freeIndicesB();
            auto const& batchIndices = problem.batchIndices();
            auto const& boundIndices = problem.boundIndices();

            auto const& a = problem.a();
            auto const& b = problem.b();
            auto const& c = problem.c();
            auto const& d = problem.d();

            bool aConjugate = false;
            bool bConjugate = false;

            for(auto const& op: problem.aOps())
                if(op.type == TensorOp::Type::ComplexConjugate)
                    aConjugate = true;

            for(auto const& op: problem.bOps())
                if(op.type == TensorOp::Type::ComplexConjugate)
                    bConjugate = true;

            std::vector<size_t> freeASize(problem.freeSizesA().size());
            std::vector<size_t> freeBSize(problem.freeSizesB().size());
            std::vector<size_t> batchSize(problem.batchIndices().size());
            std::vector<size_t> boundSize(problem.boundIndices().size());

            for(int i = 0; i < freeASize.size(); i++) freeASize[i] = problem.freeSizeA(i);
            for(int i = 0; i < freeBSize.size(); i++) freeBSize[i] = problem.freeSizeB(i);
            for(int i = 0; i < batchSize.size(); i++) batchSize[i] = problem.batchSize(i);
            for(int i = 0; i < boundSize.size(); i++) boundSize[i] = problem.boundSize(i);

            auto boundCount = CoordCount(boundSize.begin()+1, boundSize.end());

#pragma omp parallel for
            for(size_t dNum = 0; dNum < d.totalLogicalElements(); dNum += validationStride)
            {
                std::vector<size_t> aCoord(a.dimensions());
                std::vector<size_t> bCoord(b.dimensions());
                std::vector<size_t> cCoord(c.dimensions());
                std::vector<size_t> dCoord(d.dimensions());

                CoordNumbered(dNum, dCoord.begin(), dCoord.end(), d.sizes().begin(), d.sizes().end());

                for(size_t i = 0; i < problem.batchIndices().size(); i++)
                {
                    auto const& idx = problem.batchIndices()[i];
                    size_t coord = dCoord[idx.d];

                    aCoord[idx.a] = coord;
                    bCoord[idx.b] = coord;
                    cCoord[idx.c] = coord;
                }

                for(size_t i = 0; i < problem.freeIndices().size(); i++)
                {
                    auto const& idx = problem.freeIndices()[i];
                    size_t coord = dCoord[idx.d];

                    cCoord[idx.c] = coord;

                    if(idx.isA)
                        aCoord[idx.i] = coord;
                    else
                        bCoord[idx.i] = coord;
                }

                Accumulator value(0);

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
                    {
                        auto aVal = Transform<typename Inputs::AType>::Input(inputs.a[aIndex + (i * aStride)], aConjugate);
                        auto bVal = Transform<typename Inputs::BType>::Input(inputs.b[bIndex + (i * bStride)], bConjugate);

                        value += static_cast<Accumulator>(aVal * bVal);
                    }
                }

                auto cIndex = c.index(cCoord);
                auto dIndex = d.index(cCoord);

                inputs.d[dIndex] = static_cast<typename Inputs::DType>(inputs.alpha) * static_cast<typename Inputs::DType>(value)
                                 + static_cast<typename Inputs::DType>(inputs.beta) * inputs.c[cIndex];

            }
        }

        void SolveCPU(ContractionProblem const& problem, ContractionInputs const& inputs,
                      size_t validationStride)
        {
            if(problem.a().dataType() == DataType::Float
            && problem.b().dataType() == DataType::Float
            && problem.c().dataType() == DataType::Float
            && problem.d().dataType() == DataType::Float)
            {
                auto const& typedInputs = dynamic_cast<TypedContractionInputs<float> const&>(inputs);
                return ReferenceSolution<TypedContractionInputs<float>>::SolveCPU(problem, typedInputs, validationStride);
            }
            else if(problem.a().dataType() == DataType::Double
                 && problem.b().dataType() == DataType::Double
                 && problem.c().dataType() == DataType::Double
                 && problem.d().dataType() == DataType::Double)
            {
                auto const& typedInputs = dynamic_cast<TypedContractionInputs<double> const&>(inputs);
                return ReferenceSolution<TypedContractionInputs<double>>::SolveCPU(problem, typedInputs, validationStride);
            }
            else if(problem.a().dataType() == DataType::ComplexFloat
                 && problem.b().dataType() == DataType::ComplexFloat
                 && problem.c().dataType() == DataType::ComplexFloat
                 && problem.d().dataType() == DataType::ComplexFloat)
            {
                auto const& typedInputs = dynamic_cast<TypedContractionInputs<std::complex<float>> const&>(inputs);
                return ReferenceSolution<TypedContractionInputs<std::complex<float>>>::SolveCPU(problem, typedInputs, validationStride);
            }
            else if(problem.a().dataType() == DataType::ComplexDouble
                 && problem.b().dataType() == DataType::ComplexDouble
                 && problem.c().dataType() == DataType::ComplexDouble
                 && problem.d().dataType() == DataType::ComplexDouble)
            {
                auto const& typedInputs = dynamic_cast<TypedContractionInputs<std::complex<double>> const&>(inputs);
                return ReferenceSolution<TypedContractionInputs<std::complex<double>>>::SolveCPU(problem, typedInputs, validationStride);
            }
            else if(problem.a().dataType() == DataType::Half
                 && problem.b().dataType() == DataType::Half
                 && problem.c().dataType() == DataType::Half
                 && problem.d().dataType() == DataType::Half)
            {
                auto const& typedInputs = dynamic_cast<TypedContractionInputs<Half> const&>(inputs);

                if(problem.highPrecisionAccumulate())
                    return ReferenceSolution<TypedContractionInputs<Half>, float>::SolveCPU(problem, typedInputs, validationStride);
                else
                    return ReferenceSolution<TypedContractionInputs<Half>>::SolveCPU(problem, typedInputs, validationStride);
            }
            else if(problem.a().dataType() == DataType::Int8x4
                 && problem.b().dataType() == DataType::Int8x4
                 && problem.c().dataType() == DataType::Int32
                 && problem.d().dataType() == DataType::Int32)
            {
                auto const& typedInputs = dynamic_cast<TypedContractionInputs<Int8x4, Int8x4, int32_t, int32_t> const&>(inputs);
                return ReferenceSolution<TypedContractionInputs<Int8x4, Int8x4, int32_t, int32_t>>::SolveCPU(problem, typedInputs, validationStride);
            }
            else if(problem.a().dataType() == DataType::Int32
                 && problem.b().dataType() == DataType::Int32
                 && problem.c().dataType() == DataType::Int32
                 && problem.d().dataType() == DataType::Int32)
            {
                auto const& typedInputs = dynamic_cast<TypedContractionInputs<int32_t> const&>(inputs);
                return ReferenceSolution<TypedContractionInputs<int32_t>>::SolveCPU(problem, typedInputs, validationStride);
            }
            else if(problem.a().dataType() == DataType::BFloat16
                 && problem.b().dataType() == DataType::BFloat16
                 && problem.c().dataType() == DataType::BFloat16
                 && problem.d().dataType() == DataType::BFloat16)
            {
                auto const& typedInputs = dynamic_cast<BFloat16ContractionInputs const&>(inputs);

                if(problem.highPrecisionAccumulate())
                    return ReferenceSolution<BFloat16ContractionInputs, float>::SolveCPU(problem, typedInputs, validationStride);
                else
                    return ReferenceSolution<BFloat16ContractionInputs>::SolveCPU(problem, typedInputs, validationStride);
            }
            else
            {
                throw std::runtime_error("Data type not implemented.");
            }
        }
    }
}



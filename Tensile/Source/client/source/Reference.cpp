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

            std::vector<size_t> freeASize(problem.freeIndicesA().size());
            std::vector<size_t> freeBSize(problem.freeIndicesB().size());
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

        // A is activation, B is weights
        // Assume packed.
        template <typename Inputs, typename Accumulator>
        void ReferenceSolution<Inputs, Accumulator>::SolveCPUConvolution(ConvolutionProblem const &convProblem,
                ContractionProblem const& problem, Inputs const& inputs)
        {
            const unsigned db = 0x0;
            size_t batchCount = problem.a().sizes()[convProblem.tensorA().batchPosition()];
            size_t cinCount = problem.a().sizes()[convProblem.tensorA().channelPosition()];
            size_t spatial0 = problem.a().sizes()[convProblem.tensorA().spatialPosition(0)];
            size_t coutCount = problem.b().sizes()[convProblem.tensorB().weights().coutPosition()];

            std::vector<size_t> activationDims;
            switch (convProblem.tensorA().format()) {
                case ConvolutionProblem::TensorFormat::NCHW:
                    for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
                      if (convProblem.tensorA().filterPosition(fi) != ConvolutionProblem::InvalidPos)
                        activationDims.push_back(convProblem.filter()[fi]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().spatialPosition(0)]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().channelPosition()]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().batchPosition()]);
                    break;
                case ConvolutionProblem::TensorFormat::NHWC:
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().channelPosition()]);
                    for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
                      if (convProblem.tensorA().filterPosition(fi) != ConvolutionProblem::InvalidPos)
                        activationDims.push_back(convProblem.filter()[fi]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().spatialPosition(0)]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().batchPosition()]);
                case ConvolutionProblem::TensorFormat::CNHW:
                    for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
                      if (convProblem.tensorA().filterPosition(fi) != ConvolutionProblem::InvalidPos)
                        activationDims.push_back(convProblem.filter()[fi]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().spatialPosition(0)]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().batchPosition()]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().channelPosition()]);
                    break;
                default:
                    throw std::runtime_error ("unknown tensorA format");
            };
            TensorDescriptor activationTensor(problem.a().dataType(), activationDims.begin(), activationDims.end());

            std::vector<size_t> outputDims;
            switch (convProblem.tensorD().activation().format()) {
                case ConvolutionProblem::TensorFormat::NCHW:
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().spatialPosition(0)]);
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().channelPosition()]);
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().batchPosition()]);
                    break;
                case ConvolutionProblem::TensorFormat::NHWC:
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().channelPosition()]);
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().spatialPosition(0)]);
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().batchPosition()]);
                    break;
                case ConvolutionProblem::TensorFormat::CNHW:
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().spatialPosition(0)]);
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().batchPosition()]);
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().channelPosition()]);
                    break;
                default:
                    throw std::runtime_error ("unknown tensorD format");
            };
            TensorDescriptor outputTensor(problem.d().dataType(), outputDims.begin(), outputDims.end());

            std::vector<size_t> filterDims;
            switch (convProblem.tensorB().weights().format()) {
                case ConvolutionProblem::TensorFormat::KCYX:
                    for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
                      if (convProblem.tensorB().weights().filterPosition(fi) != ConvolutionProblem::InvalidPos)
                        filterDims.push_back(convProblem.filter()[fi]);
                    filterDims.push_back(problem.b().sizes()[convProblem.tensorB().weights().cinPosition()]);
                    filterDims.push_back(problem.b().sizes()[convProblem.tensorB().weights().coutPosition()]);
                    break;
                default:
                    throw std::runtime_error ("unknown tensorB format");
            };
            TensorDescriptor filterTensor(problem.b().dataType(), filterDims.begin(), filterDims.end());

            if (db & 0x1) {
                std::cout  << "SolveCPUConvolution:\n";
                std::cout  << "  tensorA=" << convProblem.tensorA().description() << "\n";
                std::cout  << "  tensorB=" << convProblem.tensorB().weights().description() << "\n";
                std::cout
                    << "  cout=" <<  coutCount
                    << " spatial0=" << spatial0
                    << " filter_zyx="
                    << convProblem.filter()[0] << "x"
                    << convProblem.filter()[1] << "x"
                    << convProblem.filter()[2]
                    << " batchCount=" << batchCount
                    << " cinCount=" << cinCount
                    << "\n";
            }

            // Loops always traverse in same order but addressing in memory can be flexible to support different activation
            // and filter formats
            std::vector<size_t> f(ConvolutionProblem::MaxNumSpatialDims,0);
            for (size_t cout = 0; cout<coutCount; cout++)
            for (size_t s0 = 0; s0<spatial0; s0++)
            {
                for (size_t n = 0; n<batchCount; n++)
                {
                    Accumulator value(0);
                    for (size_t cin = 0; cin<cinCount; cin++)
                    for (f[0] = 0; f[0]<convProblem.filter()[0]; f[0]++)
                    for (f[1] = 0; f[1]<convProblem.filter()[1]; f[1]++)
                    for (f[2] = 0; f[2]<convProblem.filter()[2]; f[2]++)
                    {
                        // Save coordinates from the looop and compute memeory index
                        // Each component stores in appropriate memory order
                        std::vector<size_t> aCoord(activationTensor.dimensions(),0);
                        std::vector<size_t> bCoord(filterTensor.dimensions(),0);

                        aCoord.at(convProblem.tensorA().batchPosition()) = n;
                        aCoord.at(convProblem.tensorA().channelPosition()) = cin;
                        aCoord.at(convProblem.tensorA().spatialPosition(0)) = s0;

                        // add filters to address calc, if they have non-unit strides:
                        for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
                        {
                            auto fp = convProblem.tensorA().filterPosition(fi);
                            if (fp != ConvolutionProblem::InvalidPos)
                              aCoord.at(fp) = f[fi];
                            else
                              assert(f[fi] == 0);
                        }

                        bCoord.at(convProblem.tensorB().weights().coutPosition()) = cout;
                        bCoord.at(convProblem.tensorB().weights().cinPosition())  = cin;
                        for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
                        {
                            auto fp = convProblem.tensorB().weights().filterPosition(fi);
                            if (fp != ConvolutionProblem::InvalidPos)
                              bCoord.at(fp) = f[fi];
                            else
                              assert(f[fi] == 0);
                        }

                        auto aIndex = activationTensor.index(aCoord);
                        auto aVal = Transform<typename Inputs::AType>::Input(inputs.a[aIndex], false);

                        auto bIndex = filterTensor.index(bCoord);
                        auto bVal = Transform<typename Inputs::BType>::Input(inputs.b[bIndex], false);

                        if (db & 0x2) {
                            std::cout   << "  n,cin,s,cout,f[0],f[1],f[2]="
                                        << n << "," << cin << "," << s0 << "," << cout << ","
                                        << f[0] << "," << f[1] << "," << f[2]
                                        << " aIndex=" << aIndex
                                        << " bIndex=" << bIndex
                                        << " aVal=" << aVal
                                        << " bVal=" << bVal
                                        << "\n";
                        }

                        value += static_cast<Accumulator>(aVal * bVal);
                    }
                    std::vector<size_t> dCoord(outputTensor.dimensions(),0);
                    dCoord.at(convProblem.tensorD().activation().batchPosition()) = n;
                    dCoord.at(convProblem.tensorD().activation().channelPosition()) = cout;
                    dCoord.at(convProblem.tensorD().activation().spatialPosition(0)) = s0;

                    auto dIndex = outputTensor.index(dCoord);
                    if (db & 0x1) {
                        std::cout   << "output: [n,s,cout=" << n << "," << s0 << "," << cout << "]"
                                    << "dIndex=" << dIndex
                                    << " value=" << value
                                    << "\n";
                    }
                    inputs.d[dIndex] = static_cast<typename Inputs::DType>(value);
                }
            }
        }

        void SolveCPUConvolution(ConvolutionProblem const &convProblem, ContractionProblem const& problem, ContractionInputs & inputs)
        {
            //std::cout << "SolveCPUConvolution:" << convProblem << " (vs " << problem << ")\n";

            if(problem.a().dataType() == DataType::Float
            && problem.b().dataType() == DataType::Float
            && problem.c().dataType() == DataType::Float
            && problem.d().dataType() == DataType::Float)
            {
                auto & typedInputs = dynamic_cast<TypedContractionInputs<float> &>(inputs);
                return ReferenceSolution<TypedContractionInputs<float>>::SolveCPUConvolution(convProblem, problem, typedInputs);
            } else {
                throw std::runtime_error("Data type not implemented for conv-vs-contract.");
            }
        }
    }
}



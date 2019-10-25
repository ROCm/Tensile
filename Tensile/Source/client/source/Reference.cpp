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
#include "Tensile/Debug.hpp"

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

        bool inZeroPad(ContractionProblem const &problem, ContractionProblem::ZeroPad const &zp,
                       const std::vector<int64_t> &dCoord, int64_t sumOffset)
        {
            if (zp.valid()) {
                int64_t anchorRelCoord = dCoord[problem.toDPos(zp.anchorIndex)] + sumOffset;
                int64_t elementEdge    = problem.d().sizes().at(problem.toDPos(zp.anchorIndex)) +
                                         problem.boundSize(problem.toBoundsPos(zp.boundIndex)) -
                                         zp.trailingPad - 1;
                //std::cout << "i=" << i << " anchorRelCoord="<< anchorRelCoord<< " leadingPad="<< zp.leadingPad<< " edge="<< elementEdge<< " trailingPad="<< zp.trailingPad << "\n";
                return (anchorRelCoord < zp.leadingPad || anchorRelCoord >= elementEdge);
            } else {
                return false;
            }
        }

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
                std::vector<int64_t> aCoord(a.dimensions());
                std::vector<int64_t> bCoord(b.dimensions());
                std::vector<int64_t> cCoord(c.dimensions());
                std::vector<int64_t> dCoord(d.dimensions());

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
                    std::vector<int64_t> bound(problem.boundIndices().size());
                    CoordNumbered(boundNum, bound.begin()+1, bound.end(), boundSize.begin()+1, boundSize.end());
                    bool aInZeroPad = false;
                    bool bInZeroPad = false;

                    for(int i = 1; i < bound.size(); i++)
                    {
                        auto const &zpA = problem.boundIndices()[i].aZeroPad;
                        auto const &zpB = problem.boundIndices()[i].bZeroPad;
                        aCoord[boundIndices[i].a] = bound[i] - zpA.leadingPad;
                        bCoord[boundIndices[i].b] = bound[i] - zpB.leadingPad;

                        if (zpA.valid() && inZeroPad(problem, zpA, dCoord, bound.at(problem.toBoundsPos(zpA.boundIndex))))
                            aInZeroPad = true;
                        if (zpB.valid() && inZeroPad(problem, zpB, dCoord, bound.at(problem.toBoundsPos(zpB.boundIndex))))
                            bInZeroPad = true;
                    }

                    auto aIndex = a.index(aCoord);
                    auto bIndex = b.index(bCoord);

                    auto aStride = problem.a().strides()[boundIndices[0].a];
                    auto bStride = problem.b().strides()[boundIndices[0].b];

                    for(size_t i = 0; i < boundSize[0]; i++)
                    {
                        auto const &zpA = problem.boundIndices()[0].aZeroPad;
                        auto const &zpB = problem.boundIndices()[0].bZeroPad;

                        typename Inputs::AType aVal(0);
                        typename Inputs::BType bVal(0);
                        if (!aInZeroPad && !inZeroPad(problem, zpA, dCoord, i))
                            aVal = Transform<typename Inputs::AType>::Input(
                                    inputs.a[aIndex + (i * aStride) - zpA.leadingPad], aConjugate);
                        if (!bInZeroPad && !inZeroPad(problem, zpB, dCoord, i))
                            bVal = Transform<typename Inputs::BType>::Input(
                                    inputs.b[bIndex + (i * bStride) - zpB.leadingPad], bConjugate);

                        value += static_cast<Accumulator>(aVal * bVal);

                        bool innerZpa = inZeroPad(problem, zpA, dCoord, i);
                        if (0) {
                            std::cout << "dNum=" << dNum << " value=" << value << " aInZeroPad=" << aInZeroPad
                                    << " innerZpa=" << innerZpa << " aindex=" << aIndex << " +offset="
                                    << (i * aStride) - zpA.leadingPad << "\n";
                        }
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
            const bool db1 = Debug::Instance().printConvolutionReference1();
            const bool db2 = Debug::Instance().printConvolutionReference2();

            if (static_cast<typename Inputs::DType>(inputs.beta) != static_cast<typename Inputs::DType>(0.0))
              throw std::runtime_error ("convolution requires beta==0");

            // Counts are the loop counters max values:
            size_t batchCount = problem.a().sizes()[convProblem.tensorA().batchPosition()];
            size_t cinCount = problem.a().sizes()[convProblem.tensorA().channelPosition()];
            size_t coutCount = problem.b().sizes()[convProblem.tensorB().weights().coutPosition()];

            std::vector<size_t> scount(ConvolutionProblem::MaxNumSpatialDims,1);
            for (int si=0; si<convProblem.tensorA().spatialPositions().size(); si++)
            {
                auto spatialPositionA = convProblem.tensorA().spatialPositions()[si];
                auto const problemSpatialSize = problem.a().sizes()[spatialPositionA];
                scount[si] = problemSpatialSize;
            }

            // Setup filter counts, translate -1 to the filter dim from problem size
            // fcount[0] is X
            std::vector<size_t> fcount(ConvolutionProblem::MaxNumSpatialDims,1);
            for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
            {
                auto const filterPositionA = convProblem.tensorA().filterPositions()[fi];
                if (filterPositionA != ConvolutionProblem::InvalidPos)
                {
                    auto const convFilterSize = convProblem.filter()[fi]; // filter from convolution-identifier
                    auto const problemFilterSize = problem.a().sizes()[filterPositionA];
                    if (convFilterSize != -1)
                      assert(convFilterSize == problemFilterSize);
                    fcount[fi] = problemFilterSize;
                }
            }

            // Mimic the expected dimension order in tensorA:
            std::vector<size_t>  activationDims;
            std::vector<int64_t> activationStri;
            switch (convProblem.tensorA().format()) {
                case ConvolutionProblem::TensorFormat::NCHW:
                    for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
                        if (convProblem.tensorA().filterPositions()[fi] != ConvolutionProblem::InvalidPos)
                        {
                            activationDims.push_back(fcount[fi]);
                            activationStri.push_back(fi==0 ?
                                convProblem.dilation().at(fi) :
                                convProblem.dilation().at(fi) *
                                  convProblem.spatials().at(fi-1));
                        }
                    for (int si=0; si<convProblem.tensorA().spatialPositions().size(); si++)
                    {
                        activationDims.push_back(scount[si]);
                        activationStri.push_back(si==0 ?
                                convProblem.stride().at(si) :
                                convProblem.stride().at(si) * convProblem.spatials().at(si-1));
                    }
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().channelPosition()]);
                    activationStri.push_back(-1);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().batchPosition()]);
                    activationStri.push_back(-1);
                    break;
                case ConvolutionProblem::TensorFormat::NHWC:
                    assert(0); // need strides
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().channelPosition()]);
                    for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
                        if (convProblem.tensorA().filterPositions()[fi] != ConvolutionProblem::InvalidPos)
                            activationDims.push_back(fcount[fi]);
                    for (int si=0; si<convProblem.tensorA().spatialPositions().size(); si++)
                        activationDims.push_back(scount[si]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().batchPosition()]);
                case ConvolutionProblem::TensorFormat::CNHW:
                    assert(0); // need strides
                    for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
                        if (convProblem.tensorA().filterPositions()[fi] != ConvolutionProblem::InvalidPos)
                            activationDims.push_back(fcount[fi]);
                    for (int si=0; si<convProblem.tensorA().spatialPositions().size(); si++)
                        activationDims.push_back(scount[si]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().batchPosition()]);
                    activationDims.push_back(problem.a().sizes()[convProblem.tensorA().channelPosition()]);
                    break;
                default:
                    throw std::runtime_error ("unknown tensorA format");
            };
            TensorDescriptor activationTensor(problem.a().dataType(),
                                    activationDims.begin(), activationDims.end(),
                                    activationStri.begin(), activationStri.end());

            std::vector<size_t> outputDims;
            switch (convProblem.tensorD().activation().format()) {
                case ConvolutionProblem::TensorFormat::NCHW:
                    for (int si=0; si<convProblem.tensorA().spatialPositions().size(); si++)
                        outputDims.push_back(scount[si]);
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().channelPosition()]);
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().batchPosition()]);
                    break;
                case ConvolutionProblem::TensorFormat::NHWC:
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().channelPosition()]);
                    for (int si=0; si<convProblem.tensorA().spatialPositions().size(); si++)
                        outputDims.push_back(scount[si]);
                    outputDims.push_back(problem.d().sizes()[convProblem.tensorD().activation().batchPosition()]);
                    break;
                case ConvolutionProblem::TensorFormat::CNHW:
                    for (int si=0; si<convProblem.tensorA().spatialPositions().size(); si++)
                        outputDims.push_back(scount[si]);
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
                        if (convProblem.tensorB().weights().filterPositions()[fi] != ConvolutionProblem::InvalidPos)
                            filterDims.push_back(fcount[fi]);
                    filterDims.push_back(problem.b().sizes()[convProblem.tensorB().weights().cinPosition()]);
                    filterDims.push_back(problem.b().sizes()[convProblem.tensorB().weights().coutPosition()]);
                    break;
                case ConvolutionProblem::TensorFormat::CKYX:
                    for (int fi=0; fi<ConvolutionProblem::MaxNumSpatialDims; fi++)
                        if (convProblem.tensorB().weights().filterPositions()[fi] != ConvolutionProblem::InvalidPos)
                            filterDims.push_back(fcount[fi]);
                    filterDims.push_back(problem.b().sizes()[convProblem.tensorB().weights().coutPosition()]);
                    filterDims.push_back(problem.b().sizes()[convProblem.tensorB().weights().cinPosition()]);
                    break;
                default:
                    throw std::runtime_error ("unknown tensorB format");
            };
            TensorDescriptor filterTensor(problem.b().dataType(), filterDims.begin(), filterDims.end());

            if (db1) {
                std::cout  << "SolveCPUConvolution:\n";
                std::cout  << "  tensorA=" << convProblem.tensorA().description() << "\n";
                std::cout  << "  tensorB=" << convProblem.tensorB().weights().description() << "\n";
                std::cout  << "  activationTensor=" << activationTensor << "\n";
                std::cout
                    << " batchCount=" << batchCount
                    << " coutCount=" <<  coutCount
                    << " scalarCount_dhw="
                    << scount[2] << "x"
                    << scount[1] << "x"
                    << scount[0]
                    << " filterCount_zyx="
                    << fcount[2] << "x"
                    << fcount[1] << "x"
                    << fcount[0]
                    << " cinCount=" << cinCount
                    << "\n";
            }

            // Loops always traverse in same order but addressing in memory can be flexible to support different activation
            // and filter formats
            std::vector<size_t> s(ConvolutionProblem::MaxNumSpatialDims,0);
            std::vector<size_t> f(ConvolutionProblem::MaxNumSpatialDims,0);
            for (size_t cout = 0; cout<coutCount; cout++)
            for (s[2]=0; s[2]<scount[2]; s[2]++)
            for (s[1]=0; s[1]<scount[1]; s[1]++)
            for (s[0]=0; s[0]<scount[0]; s[0]++)
            {
                for (size_t n = 0; n<batchCount; n++)
                {
                    Accumulator value(0);
                    for (size_t cin = 0; cin<cinCount; cin++)
                    for (f[2] = 0; f[2]<fcount[2]; f[2]++)
                    for (f[1] = 0; f[1]<fcount[1]; f[1]++)
                    for (f[0] = 0; f[0]<fcount[0]; f[0]++)
                    {
                        // Save coordinates from the looop and compute memeory index
                        // Each component stores in appropriate memory order
                        std::vector<size_t> aCoord(activationTensor.dimensions(),0);
                        std::vector<size_t> bCoord(filterTensor.dimensions(),0);

                        aCoord.at(convProblem.tensorA().batchPosition()) = n;
                        aCoord.at(convProblem.tensorA().channelPosition()) = cin;
                        for (auto i=0; i<convProblem.tensorA().spatialPositions().size(); i++)
                            aCoord.at(convProblem.tensorA().spatialPositions()[i]) = s[i];

                        // add filters to address calc, if they have non-unit strides:
                        for (int fi=ConvolutionProblem::MaxNumSpatialDims-1; fi>=0; fi--)
                        {
                            auto fp = convProblem.tensorA().filterPositions()[fi];
                            if (fp != ConvolutionProblem::InvalidPos)
                              aCoord.at(fp) = f[fi];
                            else
                              assert(f[fi] == 0);
                        }

                        bCoord.at(convProblem.tensorB().weights().coutPosition()) = cout;
                        bCoord.at(convProblem.tensorB().weights().cinPosition())  = cin;
                        for (int fi=ConvolutionProblem::MaxNumSpatialDims-1; fi>=0; fi--)
                        {
                            auto fp = convProblem.tensorB().weights().filterPositions()[fi];
                            if (fp != ConvolutionProblem::InvalidPos)
                              bCoord.at(fp) = f[fi];
                            else
                              assert(f[fi] == 0);
                        }

                        auto aIndex = activationTensor.index(aCoord);
                        auto aVal = Transform<typename Inputs::AType>::Input(inputs.a[aIndex], false);

                        auto bIndex = filterTensor.index(bCoord);
                        auto bVal = Transform<typename Inputs::BType>::Input(inputs.b[bIndex], false);

                        if (db2) {
                            std::cout   << "  n,cin,s,cout="
                                        << n << "," << cin << "," << "," << cout << ","
                                        << " s[2,1,0]=" << s[2] << ","<< s[1] << "," << s[0]
                                        << " f[2,1,0]=" << f[2] << "," << f[1] << "," << f[0]
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
                    for (auto i=0; i<convProblem.tensorD().activation().spatialPositions().size(); i++)
                        dCoord.at(convProblem.tensorD().activation().spatialPositions()[i]) = s[i];

                    auto dIndex = outputTensor.index(dCoord);
                    if (db1) {
                        std::cout   << "output: [n,s,cout=" << n << "," << "," << cout << "]"
                                    << " s[2,1,0]=" << s[2] << ","<< s[1] << "," << s[0]
                                    << " dIndex=" << dIndex
                                    << " value=" << value
                                    << "\n";
                    }
                    inputs.d[dIndex] = static_cast<typename Inputs::DType>(inputs.alpha) * static_cast<typename Inputs::DType>(value);
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



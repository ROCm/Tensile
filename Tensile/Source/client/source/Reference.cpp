/**
 * MIT License
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "Reference.hpp"
#include "Tensile/Debug.hpp"
#include "Tensile/Utils.hpp"

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

        // coord is vector with coordinates for dimensions in the anchor tensor
        // tensor is tensor descriptor for a or b
        // sumCoord is the coordinate in the sum dimension associated with the zero-pad
        bool inZeroPad(ContractionProblem const&          problem,
                       ContractionProblem::ZeroPad const& zp,
                       const TensorDescriptor&            tensor,
                       const std::vector<int64_t>&        anchorCoord,
                       int64_t                            sumCoord)
        {
            if(zp.valid())
            {
                // Check to see if the element coordinate is below or above the zero-pad
                // range The comparison is done in the element domain.
                assert(zp.anchorPos != -1); // ensure initialized.
                const auto sumPos      = problem.toBoundsPos(zp.boundIndex);
                int64_t anchorRelCoord = anchorCoord[zp.anchorPos] * tensor.strides()[zp.anchorPos]
                                         + sumCoord * tensor.strides()[zp.boundPos];
                // elementEdge calculation:
                // size of anchor dim is in the output space, so add filter size-1 to get
                // input spatial dim, then subtract padEnd anchorStride is typically spatial
                // stride (W,H) * convolution stride boundPos stride is typically spatial
                // stride (W,H) * dilation padStart, padEnd are pre-scaled by spatial stride
                int64_t elementEdge
                    = tensor.sizes().at(zp.anchorPos) * tensor.strides()[zp.anchorPos]
                      + (tensor.sizes().at(zp.boundPos) - 1) * tensor.strides()[zp.boundPos]
                      - zp.padEnd;

                bool rv = anchorRelCoord < zp.padStart || anchorRelCoord >= elementEdge;

                bool checkUnsignedRangeOpt = false;
                if(checkUnsignedRangeOpt)
                {
                    unsigned anchorRelCoord2
                        = anchorCoord[zp.anchorPos] * tensor.strides()[zp.anchorPos]
                          + sumCoord * tensor.strides()[zp.boundPos] - zp.padStart;

                    unsigned elementEdge2
                        = tensor.sizes().at(zp.anchorPos) * tensor.strides()[zp.anchorPos]
                          + (tensor.sizes().at(zp.boundPos) - 1) * tensor.strides()[zp.boundPos]
                          - zp.padEnd - zp.padStart;
                    bool rv2 = anchorRelCoord >= elementEdge2;
                    assert(rv == rv2);
                }

                if(0)
                {
                    std::cout << "  rv=" << rv << " anchorCoord=" << anchorCoord[zp.anchorPos]
                              << " boundIndex=" << zp.boundIndex << " sumCoord=" << sumCoord
                              << " anchorRelCoord=" << anchorRelCoord << " padStart=" << zp.padStart
                              << " stride=" << tensor.strides()[zp.anchorPos]
                              << " edge=" << elementEdge << " padEnd=" << zp.padEnd << "\n";
                }
                return rv;
            }
            else
            {
                return false;
            }
        }

        void throwException(const std::string& msg)
        {
            throw std::runtime_error(msg.c_str());
        }

        template <typename Inputs, typename Accumulator>
        void ReferenceSolution<Inputs, Accumulator>::SolveCPU(ContractionProblem const& problem,
                                                              Inputs const&             inputs,
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

            for(auto const& op : problem.aOps())
                if(op.type == TensorOp::Type::ComplexConjugate)
                    aConjugate = true;

            for(auto const& op : problem.bOps())
                if(op.type == TensorOp::Type::ComplexConjugate)
                    bConjugate = true;

            std::vector<size_t> freeASize(problem.freeIndicesA().size());
            std::vector<size_t> freeBSize(problem.freeIndicesB().size());
            std::vector<size_t> batchSize(problem.batchIndices().size());
            std::vector<size_t> boundSize(problem.boundIndices().size());

            for(int i = 0; i < freeASize.size(); i++)
                freeASize[i] = problem.freeSizeA(i);
            for(int i = 0; i < freeBSize.size(); i++)
                freeBSize[i] = problem.freeSizeB(i);
            for(int i = 0; i < batchSize.size(); i++)
                batchSize[i] = problem.batchSize(i);
            for(int i = 0; i < boundSize.size(); i++)
                boundSize[i] = problem.boundSize(i);

            auto boundCount = CoordCount(boundSize.begin() + 1, boundSize.end());

#pragma omp parallel for
            for(size_t dNum = 0; dNum < d.totalLogicalElements(); dNum += validationStride)
            {
                std::vector<int64_t> aCoord(a.dimensions());
                std::vector<int64_t> bCoord(b.dimensions());
                std::vector<int64_t> cCoord(c.dimensions());
                std::vector<int64_t> dCoord(d.dimensions());

                CoordNumbered(
                    dNum, dCoord.begin(), dCoord.end(), d.sizes().begin(), d.sizes().end());

                for(size_t i = 0; i < problem.batchIndices().size(); i++)
                {
                    auto const& idx   = problem.batchIndices()[i];
                    size_t      coord = dCoord[idx.d];

                    aCoord[idx.a] = coord;
                    bCoord[idx.b] = coord;
                    cCoord[idx.c] = coord;
                }

                for(size_t i = 0; i < problem.freeIndices().size(); i++)
                {
                    auto const& idx   = problem.freeIndices()[i];
                    size_t      coord = dCoord[idx.d];

                    cCoord[idx.c] = coord;

                    if(idx.isA)
                        aCoord[idx.i] = coord;
                    else
                        bCoord[idx.i] = coord;
                }

                Accumulator value(0);

                // Check short-circuit for alpha = 0
                if(inputs.alpha != static_cast<typename Inputs::AlphaType>(0))
                {
                    if(inputs.a == nullptr || inputs.b == nullptr)
                    {
                        std::string matrixID = inputs.a == nullptr ? "A" : "B";
                        std::string msg      = std::string("Unsupported nullptr for ") + matrixID
                                          + std::string(" when Alpha !=0\n");
                        // HACK moving throw temporarily to test targetid update
                        //      throw is currently causing a compiler error in this function
                        throwException(msg);
                        // throw std::runtime_error(msg.c_str());
                    }

                    for(size_t boundNum = 0; boundNum < boundCount; boundNum++)
                    {
                        std::vector<int64_t> bound(problem.boundIndices().size());
                        CoordNumbered(boundNum,
                                      bound.begin() + 1,
                                      bound.end(),
                                      boundSize.begin() + 1,
                                      boundSize.end());
                        bool aInZeroPad = false;
                        bool bInZeroPad = false;

                        for(int i = 1; i < bound.size(); i++)
                        {
                            auto const& zpA           = problem.boundIndices()[i].aZeroPad;
                            auto const& zpB           = problem.boundIndices()[i].bZeroPad;
                            aCoord[boundIndices[i].a] = bound[i];
                            bCoord[boundIndices[i].b] = bound[i];

                            if(problem.boundIndices()[i].aMirror)
                                aCoord[boundIndices[i].a]
                                    = boundSize[i] - aCoord[boundIndices[i].a] - 1;
                            if(problem.boundIndices()[i].bMirror)
                                bCoord[boundIndices[i].b]
                                    = boundSize[i] - bCoord[boundIndices[i].b] - 1;

                            if(zpA.valid())
                            {
                                auto sumCoord = bound.at(problem.toBoundsPos(zpA.boundIndex));
                                if(problem.boundIndices()[i].aMirror)
                                    sumCoord = boundSize[i] - sumCoord - 1;

                                if(inZeroPad(problem, zpA, a, aCoord, sumCoord))
                                    aInZeroPad = true;
                            }
                            if(zpB.valid())
                            {
                                auto sumCoord = bound.at(problem.toBoundsPos(zpB.boundIndex));
                                if(problem.boundIndices()[i].bMirror)
                                    sumCoord = boundSize[i] - sumCoord - 1;
                                if(inZeroPad(problem, zpB, b, bCoord, sumCoord))
                                    bInZeroPad = true;
                            }
                        }

                        size_t aIndex = a.index(aCoord);
                        size_t bIndex = b.index(bCoord);
                        for(int i = 1; i < bound.size(); i++)
                        {
                            auto const& zpA = problem.boundIndices()[i].aZeroPad;
                            auto const& zpB = problem.boundIndices()[i].bZeroPad;

                            aIndex -= zpA.padStart;
                            bIndex -= zpB.padStart;
                        }

                        auto aStride = problem.a().strides()[boundIndices[0].a];
                        auto bStride = problem.b().strides()[boundIndices[0].b];

                        // innermost bound calculation:
                        for(size_t i = 0; i < boundSize[0]; i++)
                        {
                            auto const& zpA = problem.boundIndices()[0].aZeroPad;
                            auto const& zpB = problem.boundIndices()[0].bZeroPad;
                            size_t      aI
                                = problem.boundIndices()[0].aMirror ? (boundSize[0] - i - 1) : i;
                            size_t bI
                                = problem.boundIndices()[0].bMirror ? (boundSize[0] - i - 1) : i;

                            typename Inputs::AType aVal(0);
                            typename Inputs::BType bVal(0);
                            if(!aInZeroPad && !inZeroPad(problem, zpA, a, aCoord, aI))
                                aVal = Transform<typename Inputs::AType>::Input(
                                    inputs.a[aIndex + (aI * aStride) - zpA.padStart], aConjugate);
                            if(!bInZeroPad && !inZeroPad(problem, zpB, b, bCoord, bI))
                                bVal = Transform<typename Inputs::BType>::Input(
                                    inputs.b[bIndex + (bI * bStride) - zpB.padStart], bConjugate);

                            value += static_cast<Accumulator>(aVal * bVal);

                            if(0)
                            {
                                std::cout << " bound=" << bound[0] << "," << bound[1]
                                          << " dNum=" << dNum << " value=" << value
                                          << " aInZeroPad=" << aInZeroPad << " aindex=" << aIndex
                                          << " +offset="
                                          << (int64_t)(i * aStride) - zpA.padStart
                                          //<< " aVal=" << aVal // disable int8
                                          << "\n";
                            }
                        }
                    }
                }

                auto cIndex = c.index(cCoord);
                auto dIndex = d.index(cCoord);

                // Ensure zero*nan returns zero
                auto beta = static_cast<typename Inputs::DType>(inputs.beta);
                auto zero = static_cast<typename Inputs::DType>(0);

                inputs.d[dIndex] = static_cast<typename Inputs::DType>(inputs.alpha)
                                       * static_cast<typename Inputs::DType>(value)
                                   + ((beta == zero) ? zero : beta * inputs.c[cIndex]);
            }
        }

        void SolveCPU(ContractionProblem const& problem,
                      ContractionInputs const&  inputs,
                      size_t                    validationStride)
        {
            // retreive alpha/beta type set via setAlpha/BetaType()
            auto alphaType = problem.alphaType();
            auto betaType  = problem.betaType();

            // Backward-compatible: when setAlpha/BetaType() wasn't called, use the old way
            // Could remove after rocBLAS is updated
            if(alphaType == DataType::None)
            {
                alphaType = problem.a().dataType() == DataType::BFloat16 ? DataType::Float
                                                                         : problem.d().dataType();
            }
            if(betaType == DataType::None)
            {
                betaType = alphaType;
            }

            auto contractionInputsTypeId = ContractionInputs::TypeId(problem.a().dataType(),
                                                                     problem.b().dataType(),
                                                                     problem.c().dataType(),
                                                                     problem.d().dataType(),
                                                                     alphaType,
                                                                     betaType);

            switch(contractionInputsTypeId)
            {
            case ContractionInputs_S_S_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_S_S_S const&>(inputs);
                return ReferenceSolution<ContractionInputs_S_S_S>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_D_D_D::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_D_D_D const&>(inputs);
                return ReferenceSolution<ContractionInputs_D_D_D>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_C_C_C::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_C_C_C const&>(inputs);
                return ReferenceSolution<ContractionInputs_C_C_C>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_Z_Z_Z::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_Z_Z_Z const&>(inputs);
                return ReferenceSolution<ContractionInputs_Z_Z_Z>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
#ifdef TENSILE_USE_HALF
            case ContractionInputs_H_H_H::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_H_H_H const&>(inputs);

                if(problem.highPrecisionAccumulate())
                {
                    return ReferenceSolution<ContractionInputs_H_H_H, float>::SolveCPU(
                        problem, typedInputs, validationStride);
                }
                else
                {
                    return ReferenceSolution<ContractionInputs_H_H_H>::SolveCPU(
                        problem, typedInputs, validationStride);
                }
            }
            case ContractionInputs_H_S_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_H_S_S const&>(inputs);
                return ReferenceSolution<ContractionInputs_H_S_S>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_H_H_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_H_H_S const&>(inputs);
                return ReferenceSolution<ContractionInputs_H_H_S, float>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
#endif // TENSILE_USE_HALF
            case ContractionInputs_I8x4_I32_I32::TypeId():
            {
                auto const& typedInputs
                    = dynamic_cast<ContractionInputs_I8x4_I32_I32 const&>(inputs);
                return ReferenceSolution<ContractionInputs_I8x4_I32_I32>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
            case ContractionInputs_I32_I32_I32::TypeId():
            {
                auto const& typedInputs
                    = dynamic_cast<ContractionInputs_I32_I32_I32 const&>(inputs);
                return ReferenceSolution<ContractionInputs_I32_I32_I32>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
#ifdef TENSILE_USE_BF16
            case ContractionInputs_B_B_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_B_B_S const&>(inputs);

                if(problem.highPrecisionAccumulate())
                {
                    return ReferenceSolution<ContractionInputs_B_B_S, float>::SolveCPU(
                        problem, typedInputs, validationStride);
                }
                else
                {
                    return ReferenceSolution<ContractionInputs_B_B_S>::SolveCPU(
                        problem, typedInputs, validationStride);
                }
            }
            case ContractionInputs_B_S_S::TypeId():
            {
                auto const& typedInputs = dynamic_cast<ContractionInputs_B_S_S const&>(inputs);
                return ReferenceSolution<ContractionInputs_B_S_S>::SolveCPU(
                    problem, typedInputs, validationStride);
            }
#endif // TENSILE_USE_BF16

            default:;
            }

            throw std::runtime_error("Data type not implemented.");
        }

        // A is activation, B is weights
        // Assume packed.
        template <typename Inputs, typename Accumulator>
        void ReferenceSolution<Inputs, Accumulator>::SolveCPUConvolution(
            ConvolutionProblem const& convProblem,
            ContractionProblem const& problem,
            Inputs const&             inputs)
        {
            const bool db1 = Debug::Instance().printConvolutionReference1();
            const bool db2 = Debug::Instance().printConvolutionReference2();

            if(static_cast<typename Inputs::DType>(inputs.beta)
               != static_cast<typename Inputs::DType>(0.0))
                throw std::runtime_error("convolution requires beta==0");

            ConvolutionProblem::LoopCounts counts;
            counts.setupForData(convProblem, problem);

            TensorDescriptor activationTensor = convProblem.setupDataActivation(counts, problem);
            TensorDescriptor weightTensor     = convProblem.setupForwardWeights(counts, problem);
            TensorDescriptor outputTensor     = convProblem.setupDataOutput(counts, problem);

            if(db1)
            {
                std::cout << "SolveCPUConvolution:\n";
                std::cout << "  formatA=" << convProblem.formatA().description() << "\n";
                std::cout << "  formatB=" << convProblem.formatB().weights().description() << "\n";
                std::cout << "  activationTensor=" << activationTensor << "\n";
                std::cout << " " << counts.description() << "\n";
            }

            // Loops always traverse in same order but addressing in memory can be flexible to support different activation
            // and filter formats
            size_t spatialCoordCount = CoordCount(counts.scount.begin(), counts.scount.end());
#pragma omp parallel for collapse(3)
            for(size_t cout = 0; cout < counts.coutCount; cout++)
                for(size_t spatialIndex = 0; spatialIndex < spatialCoordCount; spatialIndex++)
                    for(size_t n = 0; n < counts.batchCount; n++)
                    {
                        std::vector<size_t> spatialCoord(ConvolutionProblem::MaxNumSpatialDims, 0);
                        std::vector<size_t> filterCoord(ConvolutionProblem::MaxNumSpatialDims, 0);

                        CoordNumbered(spatialIndex,
                                      spatialCoord.begin(),
                                      spatialCoord.end(),
                                      counts.scount.begin(),
                                      counts.scount.end());

                        Accumulator value(0);
                        for(size_t cin = 0; cin < counts.cinCount; cin++)
                            for(filterCoord[2] = 0; filterCoord[2] < counts.fcount[2];
                                filterCoord[2]++)
                                for(filterCoord[1] = 0; filterCoord[1] < counts.fcount[1];
                                    filterCoord[1]++)
                                    for(filterCoord[0] = 0; filterCoord[0] < counts.fcount[0];
                                        filterCoord[0]++)
                                    {
                                        // Save coordinates from the looop and compute memeory index
                                        // Each component stores in appropriate memory order
                                        std::vector<size_t> aCoord(activationTensor.dimensions(),
                                                                   0);
                                        std::vector<size_t> bCoord(weightTensor.dimensions(), 0);

                                        aCoord[convProblem.formatA().batchPosition()]   = n;
                                        aCoord[convProblem.formatA().channelPosition()] = cin;
                                        for(auto i = 0;
                                            i < convProblem.formatA().spatialPositions().size();
                                            i++)
                                            aCoord[convProblem.formatA().spatialPositions()[i]]
                                                = spatialCoord[i];

                                        // add filters to address calc, if they have non-unit strides:
                                        for(int fi = ConvolutionProblem::MaxNumSpatialDims - 1;
                                            fi >= 0;
                                            fi--)
                                        {
                                            auto fp = convProblem.formatA().filterPositions()[fi];
                                            if(fp != ConvolutionProblem::InvalidPos)
                                                aCoord[fp] = filterCoord[fi];
                                            else
                                                assert(filterCoord[fi] == 0);
                                        }

                                        bCoord[convProblem.formatB().weights().coutPosition()]
                                            = cout;
                                        bCoord[convProblem.formatB().weights().cinPosition()] = cin;
                                        for(int fi = ConvolutionProblem::MaxNumSpatialDims - 1;
                                            fi >= 0;
                                            fi--)
                                        {
                                            auto fp = convProblem.formatB()
                                                          .weights()
                                                          .filterPositions()[fi];
                                            if(fp != ConvolutionProblem::InvalidPos)
                                                bCoord[fp] = filterCoord[fi];
                                            else
                                                assert(filterCoord[fi] == 0);
                                        }

                                        auto aIndex = activationTensor.index(aCoord);
                                        auto aVal   = Transform<typename Inputs::AType>::Input(
                                            inputs.a[aIndex], false);

                                        auto bIndex = weightTensor.index(bCoord);
                                        auto bVal   = Transform<typename Inputs::BType>::Input(
                                            inputs.b[bIndex], false);

                                        if(db2)
                                        {
                                            std::cout
                                                << "  n,cin,spatialCoord,cout=" << n << "," << cin
                                                << ","
                                                << "," << cout << ","
                                                << " spatialCoord[2,1,0]=" << spatialCoord[2] << ","
                                                << spatialCoord[1] << "," << spatialCoord[0]
                                                << " filterCoord[2,1,0]=" << filterCoord[2] << ","
                                                << filterCoord[1] << "," << filterCoord[0]
                                                << " aIndex=" << aIndex << " bIndex=" << bIndex
                                                << " aVal=" << aVal << " bVal=" << bVal << "\n";
                                        }
                                        value += static_cast<Accumulator>(aVal * bVal);
                                    }
                        std::vector<size_t> dCoord(outputTensor.dimensions(), 0);
                        dCoord[convProblem.formatD().activation().batchPosition()]   = n;
                        dCoord[convProblem.formatD().activation().channelPosition()] = cout;
                        for(auto i = 0;
                            i < convProblem.formatD().activation().spatialPositions().size();
                            i++)
                            dCoord[convProblem.formatD().activation().spatialPositions()[i]]
                                = spatialCoord[i];

                        auto dIndex = outputTensor.index(dCoord);
                        if(db1)
                        {
                            std::cout << "output: [n,spatialCoord,cout=" << n << ","
                                      << "," << cout << "]"
                                      << " spatialCoord[2,1,0]=" << spatialCoord[2] << ","
                                      << spatialCoord[1] << "," << spatialCoord[0]
                                      << " dIndex=" << dIndex << " value=" << value << "\n";
                        }
                        inputs.d[dIndex] = static_cast<typename Inputs::DType>(inputs.alpha)
                                           * static_cast<typename Inputs::DType>(value);
                    }
        }

        void SolveCPUConvolution(ConvolutionProblem const& convProblem,
                                 ContractionProblem const& problem,
                                 ContractionInputs&        inputs)
        {
            // std::cout << "SolveCPUConvolution:" << convProblem << " (vs " << problem <<
            // ")\n";

            if(problem.a().dataType() == DataType::Float
               && problem.b().dataType() == DataType::Float
               && problem.c().dataType() == DataType::Float
               && problem.d().dataType() == DataType::Float)
            {
                auto& typedInputs = dynamic_cast<TypedContractionInputs<float>&>(inputs);
                return ReferenceSolution<TypedContractionInputs<float>>::SolveCPUConvolution(
                    convProblem, problem, typedInputs);
            }
            else
            {
                throw std::runtime_error("Data type not implemented for conv-vs-contract.");
            }
        }
    } // namespace Client
} // namespace Tensile

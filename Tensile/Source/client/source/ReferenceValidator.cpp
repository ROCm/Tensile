/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include "ReferenceValidator.hpp"
#include "ResultReporter.hpp"

#include "Reference.hpp"

#include <Tensile/DataTypes.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <cstddef>

namespace Tensile
{
    namespace Client
    {
        ReferenceValidator::ReferenceValidator(po::variables_map const& args, std::shared_ptr<DataInitialization> dataInit)
                : m_dataInit(dataInit)
        {
            m_elementsToValidate = args["num-elements-to-validate"].as<int>();
            m_printValids = args["print-valids"].as<bool>();
            m_printMax = args["print-max"].as<int>();

            m_printTensorA = args["print-tensor-a"].as<bool>();
            m_printTensorB = args["print-tensor-b"].as<bool>();
            m_printTensorC = args["print-tensor-c"].as<bool>();
            m_printTensorD = args["print-tensor-d"].as<bool>();
            m_printTensorRef = args["print-tensor-ref"].as<bool>();

            m_enabled = m_elementsToValidate != 0
                     || m_printTensorA
                     || m_printTensorB
                     || m_printTensorC
                     || m_printTensorD
                     || m_printTensorRef;
        }

        bool ReferenceValidator::needMoreBenchmarkRuns() const
        {
            if(m_enabled && m_numBenchmarkRuns == 0)
                return true;

            return false;
        }

        void ReferenceValidator::preBenchmarkRun()
        {
        }

        void ReferenceValidator::postBenchmarkRun()
        {
            m_numBenchmarkRuns++;
        }

        void ReferenceValidator::preProblem(ContractionProblem const& problem)
        {
            if(m_enabled)
            {
                m_problem = problem;
                m_referenceInputs = m_dataInit->prepareCPUInputs();
                m_validationStride = 1;
                if(m_elementsToValidate > 0 && m_elementsToValidate < problem.d().totalLogicalElements())
                    m_validationStride = NextPrime(problem.d().totalAllocatedElements() / m_elementsToValidate);

                SolveCPU(problem, *m_referenceInputs, m_validationStride);
            }
        }

        void ReferenceValidator::preSolution(ContractionSolution const& solution)
        {
            m_validatedSolution = false;
            m_errorInSolution = false;
        }

        bool ReferenceValidator::needMoreRunsInSolution() const
        {
            if(m_enabled && !m_validatedSolution)
                return true;

            return false;
        }

        size_t ReferenceValidator::numWarmupRuns()
        {
            if(m_enabled && !m_validatedSolution)
                return 1;

            return 0;
        }

        void ReferenceValidator::setNumWarmupRuns(size_t count)
        {
        }

        void ReferenceValidator::preWarmup()
        {
        }

        void ReferenceValidator::postWarmup()
        {
        }

        void ReferenceValidator::validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                                 TimingEvents const& startEvents,
                                                 TimingEvents const&  stopEvents)
        {
            if(m_enabled && !m_validatedSolution)
            {
                if(m_problem.a().dataType() == DataType::Float
                && m_problem.b().dataType() == DataType::Float
                && m_problem.c().dataType() == DataType::Float
                && m_problem.d().dataType() == DataType::Float)
                {
                    auto const& typedReference = dynamic_cast<TypedContractionInputs<float> const&>(*m_referenceInputs);
                    auto const& typedResult = dynamic_cast<TypedContractionInputs<float> const&>(*inputs);
                    validateTyped(typedReference, typedResult);
                }
                else if(m_problem.a().dataType() == DataType::Double
                     && m_problem.b().dataType() == DataType::Double
                     && m_problem.c().dataType() == DataType::Double
                     && m_problem.d().dataType() == DataType::Double)
                {
                    auto const& typedReference = dynamic_cast<TypedContractionInputs<double> const&>(*m_referenceInputs);
                    auto const& typedResult = dynamic_cast<TypedContractionInputs<double> const&>(*inputs);
                    validateTyped(typedReference, typedResult);
                }
                else if(m_problem.a().dataType() == DataType::ComplexFloat
                     && m_problem.b().dataType() == DataType::ComplexFloat
                     && m_problem.c().dataType() == DataType::ComplexFloat
                     && m_problem.d().dataType() == DataType::ComplexFloat)
                {
                    auto const& typedReference = dynamic_cast<TypedContractionInputs<std::complex<float>> const&>(*m_referenceInputs);
                    auto const& typedResult = dynamic_cast<TypedContractionInputs<std::complex<float>> const&>(*inputs);
                    validateTyped(typedReference, typedResult);
                }
                else if(m_problem.a().dataType() == DataType::ComplexDouble
                     && m_problem.b().dataType() == DataType::ComplexDouble
                     && m_problem.c().dataType() == DataType::ComplexDouble
                     && m_problem.d().dataType() == DataType::ComplexDouble)
                {
                    auto const& typedReference = dynamic_cast<TypedContractionInputs<std::complex<double>> const&>(*m_referenceInputs);
                    auto const& typedResult = dynamic_cast<TypedContractionInputs<std::complex<double>> const&>(*inputs);
                    validateTyped(typedReference, typedResult);
                }
                else if(m_problem.a().dataType() == DataType::Half
                     && m_problem.b().dataType() == DataType::Half
                     && m_problem.c().dataType() == DataType::Half
                     && m_problem.d().dataType() == DataType::Half)
                {
                    auto const& typedReference = dynamic_cast<TypedContractionInputs<Half> const&>(*m_referenceInputs);
                    auto const& typedResult = dynamic_cast<TypedContractionInputs<Half> const&>(*inputs);
                    validateTyped(typedReference, typedResult);
                }
                else if(m_problem.a().dataType() == DataType::Int8x4
                     && m_problem.b().dataType() == DataType::Int8x4
                     && m_problem.c().dataType() == DataType::Int32
                     && m_problem.d().dataType() == DataType::Int32)
                {
                    auto const& typedReference = dynamic_cast<TypedContractionInputs<Int8x4, Int8x4, int32_t, int32_t> const&>(*m_referenceInputs);
                    auto const& typedResult = dynamic_cast<TypedContractionInputs<Int8x4, Int8x4, int32_t, int32_t> const&>(*inputs);
                    validateTyped(typedReference, typedResult);
                }
                else if(m_problem.a().dataType() == DataType::Int32
                     && m_problem.b().dataType() == DataType::Int32
                     && m_problem.c().dataType() == DataType::Int32
                     && m_problem.d().dataType() == DataType::Int32)
                {
                    auto const& typedReference = dynamic_cast<TypedContractionInputs<int32_t> const&>(*m_referenceInputs);
                    auto const& typedResult = dynamic_cast<TypedContractionInputs<int32_t> const&>(*inputs);
                    validateTyped(typedReference, typedResult);
                }
                else if(m_problem.a().dataType() == DataType::BFloat16
                     && m_problem.b().dataType() == DataType::BFloat16
                     && m_problem.c().dataType() == DataType::BFloat16
                     && m_problem.d().dataType() == DataType::BFloat16)
                {
                    auto const& typedReference = dynamic_cast<BFloat16ContractionInputs const&>(*m_referenceInputs);
                    auto const& typedResult = dynamic_cast<BFloat16ContractionInputs const&>(*inputs);
                    validateTyped(typedReference, typedResult);
                }
                else
                {
                    throw std::runtime_error("Data type not implemented.");
                }
            }
        }

        template <typename TypedInputs>
        void ReferenceValidator::validateTyped(TypedInputs const& reference, TypedInputs const& result)
        {
            if(!m_enabled || m_validatedSolution)
                return;

            if(m_printTensorA || m_printTensorB
            || m_printTensorC || m_printTensorD
            || m_printTensorRef)
                printTensorsTyped(reference, result);

            if(m_elementsToValidate != 0)
                checkResultsTyped(reference, result);

            m_validatedSolution = true;
        }

        template <typename TypedInputs>
        void ReferenceValidator::printTensorsTyped(TypedInputs const& reference, TypedInputs const& result)
        {
            size_t requiredBufferSize = 0;

            std::cout << "reference alpha: " << reference.alpha << ", beta: " << reference.beta << std::endl;
            std::cout << "result    alpha: " << result.alpha << ", beta: " << result.beta << std::endl;

            if(m_printTensorA) requiredBufferSize = std::max(requiredBufferSize, m_problem.a().totalAllocatedBytes());
            if(m_printTensorB) requiredBufferSize = std::max(requiredBufferSize, m_problem.b().totalAllocatedBytes());
            if(m_printTensorC) requiredBufferSize = std::max(requiredBufferSize, m_problem.c().totalAllocatedBytes());
            if(m_printTensorD) requiredBufferSize = std::max(requiredBufferSize, m_problem.d().totalAllocatedBytes());
            if(m_printTensorRef) requiredBufferSize = std::max(requiredBufferSize, m_problem.d().totalAllocatedBytes());

            if(m_cpuResultBuffer.size() < requiredBufferSize)
                m_cpuResultBuffer.resize(requiredBufferSize);

            if(m_printTensorA)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(), result.a, m_problem.a().totalAllocatedBytes(), hipMemcpyDeviceToHost));
                auto const* buffer = reinterpret_cast<typename TypedInputs::AType const*>(m_cpuResultBuffer.data());

                m_reporter->logTensor(LogLevel::Verbose, "A", buffer, m_problem.a(), result.a);
            }

            if(m_printTensorB)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(), result.b, m_problem.b().totalAllocatedBytes(), hipMemcpyDeviceToHost));
                auto const* buffer = reinterpret_cast<typename TypedInputs::BType const*>(m_cpuResultBuffer.data());

                m_reporter->logTensor(LogLevel::Verbose, "B", buffer, m_problem.b(), result.b);
            }

            if(result.c == result.d && (m_printTensorC || m_printTensorD))
            {
                // If the pointers are the same, only print the buffer once.
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(), result.c, m_problem.c().totalAllocatedBytes(), hipMemcpyDeviceToHost));
                auto const* buffer = reinterpret_cast<typename TypedInputs::CType const*>(m_cpuResultBuffer.data());

                m_reporter->logTensor(LogLevel::Verbose, "C/D", buffer, m_problem.c(), result.c);
            }
            else
            {
                if(m_printTensorC)
                {
                    HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(), result.c, m_problem.c().totalAllocatedBytes(), hipMemcpyDeviceToHost));
                    auto const* buffer = reinterpret_cast<typename TypedInputs::CType const*>(m_cpuResultBuffer.data());

                    m_reporter->logTensor(LogLevel::Verbose, "C", buffer, m_problem.c(), result.c);
                }

                if(m_printTensorD)
                {
                    HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(), result.d, m_problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));
                    auto const* buffer = reinterpret_cast<typename TypedInputs::DType const*>(m_cpuResultBuffer.data());

                    m_reporter->logTensor(LogLevel::Verbose, "D", buffer, m_problem.d(), result.d);
                }
            }

            if(m_printTensorRef)
            {
                m_reporter->logTensor(LogLevel::Verbose, "Ref", reference.d, m_problem.d(), reference.d);
            }
        }

        template <typename TypedInputs>
        void ReferenceValidator::checkResultsTyped(TypedInputs const& reference, TypedInputs const& result)
        {
            auto const& tensor = m_problem.d();
            if(m_cpuResultBuffer.size() < tensor.totalAllocatedBytes())
                m_cpuResultBuffer.resize(tensor.totalAllocatedBytes());

            HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(), result.d, tensor.totalAllocatedBytes(), hipMemcpyDeviceToHost));

            using Type = typename TypedInputs::DType;
            Type const* resultBuffer = reinterpret_cast<Type const*>(m_cpuResultBuffer.data());

            int printed = 0;

            bool doPrint = m_printMax < 0 || printed < m_printMax;

            size_t errors = 0;

            auto compareValues =
            [&](Type referenceValue, Type resultValue, size_t elemIndex, size_t elemNumber)
            {
                bool match = AlmostEqual(referenceValue, resultValue);
                if(!match)
                    errors++;

                if(!match || m_printValids)
                {
                    if(doPrint)
                    {
                        if(printed == 0)
                        {
                            std::cout << "Index:  Device | Reference" << std::endl;
                        }

                        std::cout << "[" << (printed) << "] " 
                                  << " elem=" << elemNumber
                                  << " idx=" << elemIndex << ": "
                                  << resultValue
                                  << (match ? "==" : "!=") << referenceValue
                                  << std::endl;

                        printed++;

                        if(m_printMax >= 0 && printed >= m_printMax)
                            doPrint = false;
                    }
                }
            };

            if(m_validationStride == 1)
            {
                std::vector<size_t> coord(tensor.dimensions());
                size_t outerCount = CoordCount(tensor.sizes().begin()+1, tensor.sizes().end());

                for(size_t i = 0; i < outerCount; i++)
                {
                    CoordNumbered(i, coord.begin()+1, coord.end(), tensor.sizes().begin()+1, tensor.sizes().end());
                    size_t baseElemIndex = tensor.index(coord);

                    for(size_t j = 0; j < tensor.sizes()[0]; j++)
                    {
                        size_t elemIndex = baseElemIndex + j;

                        Type referenceValue = reference.d[elemIndex];
                        Type resultValue = resultBuffer[elemIndex];

                        compareValues(referenceValue, resultValue, elemIndex, (i*tensor.sizes()[0]) + j);
                    }
                }
            }
            else
            {
                std::vector<size_t> coord(tensor.dimensions());
                for(size_t elemNumber = 0; elemNumber < tensor.totalLogicalElements(); elemNumber += m_validationStride)
                {
                    CoordNumbered(elemNumber, coord.begin(), coord.end(), tensor.sizes().begin(), tensor.sizes().end());
                    size_t elemIndex = tensor.index(coord);

                    Type referenceValue = reference.d[elemIndex];
                    Type resultValue = resultBuffer[elemIndex];

                    compareValues(referenceValue, resultValue, elemIndex, elemNumber);
                }
            }

            if(errors > 0)
            {
                m_errorInSolution = true;
                m_error = true;
            }
        }

        void ReferenceValidator::postSolution()
        {
            if(m_enabled && !m_validatedSolution)
                return;

            if(m_elementsToValidate != 0)
            {
                if(m_errorInSolution)
                    m_reporter->report(ResultKey::Validation, "FAILED");
                else
                    m_reporter->report(ResultKey::Validation, "PASSED");
            }
            else
            {
                m_reporter->report(ResultKey::Validation, "NO_CHECK");
            }

            m_errorInSolution = false;
        }

        void ReferenceValidator::postProblem()
        {
        }

        void ReferenceValidator::finalizeReport()
        {
        }

        int  ReferenceValidator::error() const
        {
            return 0;
        }
    }
}



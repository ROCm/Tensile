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
#include "DataInitializationTyped.hpp"

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

            m_boundsCheck = args["bounds-check"].as<bool>();

            m_printTensorA = args["print-tensor-a"].as<bool>();
            m_printTensorB = args["print-tensor-b"].as<bool>();
            m_printTensorC = args["print-tensor-c"].as<bool>();
            m_printTensorD = args["print-tensor-d"].as<bool>();
            m_printTensorRef = args["print-tensor-ref"].as<bool>();

            m_convolutionVsContraction = args["convolution-vs-contraction"].as<bool>();
            if(args.count("convolution-identifier"))
                m_convolutionProblem.FromIdentifier(args["convolution-identifier"].as<std::string>());

            m_enabled = m_elementsToValidate != 0
                     || m_printTensorA
                     || m_printTensorB
                     || m_printTensorC
                     || m_printTensorD
                     || m_printTensorRef
                     || m_convolutionVsContraction;
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
                m_referenceInputs = m_dataInit->prepareCPUInputs(problem);
                m_validationStride = 1;
                if(m_elementsToValidate > 0 && m_elementsToValidate < problem.d().totalLogicalElements())
                    m_validationStride = NextPrime(problem.d().totalAllocatedElements() / m_elementsToValidate);

                SolveCPU(problem, *m_referenceInputs, m_validationStride);

                if (m_convolutionVsContraction) {

                  m_convolutionProblem.validate(problem);

                  SolveCPUConvolution(m_convolutionProblem, problem, *(m_dataInit->cpuConvInputs()));
                  //std::cout << "ValidateConv--Start\n";
                  m_errorInConvolutionVsContraction = validateSolution(m_dataInit->cpuConvInputs());  // validate conv against reference
                // TODO - print problem dimensions??
                  std::cout << m_convolutionProblem << " vs " << problem.operationIdentifier() << " :  ";
                  if (m_errorInConvolutionVsContraction) {
                      std::cout << "FAILED_CONV";
                  } else {
                      std::cout << "PASSED_CONV";
                  }
                  std::cout << "\n";
                }
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

        template <typename ManagedInputs>
        bool ReferenceValidator::validateSolutionCast(std::shared_ptr<ContractionInputs> inputs)
        {
            auto const& typedReference = dynamic_cast<ManagedInputs const&>(*m_referenceInputs);
            auto const& typedResult    = dynamic_cast<ManagedInputs const&>(*inputs);

            auto rv =  validateTyped(typedReference, typedResult);

            if (0 and inputs == m_dataInit->cpuConvInputs()) {
                m_reporter->logTensor(LogLevel::Verbose, "Aval-conv", typedResult.a, m_problem.a());
                m_reporter->logTensor(LogLevel::Verbose, "Bval-conv", typedResult.b, m_problem.b());
                m_reporter->logTensor(LogLevel::Verbose, "Dval-conv", typedResult.d, m_problem.d());
                m_reporter->logTensor(LogLevel::Verbose, "Bval-contraction", typedReference.b, m_problem.b());
                m_reporter->logTensor(LogLevel::Verbose, "Dval-contraction", typedReference.d, m_problem.d());
            }

            return rv;
        }

        bool ReferenceValidator::validateSolution(std::shared_ptr<ContractionInputs> inputs)
        {
            if(m_problem.a().dataType() == DataType::Float
            && m_problem.b().dataType() == DataType::Float
            && m_problem.c().dataType() == DataType::Float
            && m_problem.d().dataType() == DataType::Float)
            {
                return validateSolutionCast<ManagedContractionInputs<float>>(inputs);
            }
            else if(m_problem.a().dataType() == DataType::Double
                 && m_problem.b().dataType() == DataType::Double
                 && m_problem.c().dataType() == DataType::Double
                 && m_problem.d().dataType() == DataType::Double)
            {
                return validateSolutionCast<ManagedContractionInputs<double>>(inputs);
            }
            else if(m_problem.a().dataType() == DataType::ComplexFloat
                 && m_problem.b().dataType() == DataType::ComplexFloat
                 && m_problem.c().dataType() == DataType::ComplexFloat
                 && m_problem.d().dataType() == DataType::ComplexFloat)
            {
                return validateSolutionCast<ManagedContractionInputs<std::complex<float>>>(inputs);
            }
            else if(m_problem.a().dataType() == DataType::ComplexDouble
                 && m_problem.b().dataType() == DataType::ComplexDouble
                 && m_problem.c().dataType() == DataType::ComplexDouble
                 && m_problem.d().dataType() == DataType::ComplexDouble)
            {
                return validateSolutionCast<ManagedContractionInputs<std::complex<double>>>(inputs);
            }
            else if(m_problem.a().dataType() == DataType::Half
                 && m_problem.b().dataType() == DataType::Half
                 && m_problem.c().dataType() == DataType::Half
                 && m_problem.d().dataType() == DataType::Half)
            {
                return validateSolutionCast<ManagedContractionInputs<Half>>(inputs);
            }
            else if(m_problem.a().dataType() == DataType::Int8x4
                 && m_problem.b().dataType() == DataType::Int8x4
                 && m_problem.c().dataType() == DataType::Int32
                 && m_problem.d().dataType() == DataType::Int32)
            {
                return validateSolutionCast<ManagedContractionInputs<Int8x4, Int8x4, int32_t, int32_t>>
                                           (inputs);
            }
            else if(m_problem.a().dataType() == DataType::Int32
                 && m_problem.b().dataType() == DataType::Int32
                 && m_problem.c().dataType() == DataType::Int32
                 && m_problem.d().dataType() == DataType::Int32)
            {
                return validateSolutionCast<ManagedContractionInputs<int32_t>>(inputs);
            }
            else if(m_problem.a().dataType() == DataType::BFloat16
                 && m_problem.b().dataType() == DataType::BFloat16
                 && m_problem.c().dataType() == DataType::BFloat16
                 && m_problem.d().dataType() == DataType::BFloat16)
            {
                return validateSolutionCast<ManagedBFloat16ContractionInputs>(inputs);
            }
            else
            {
                throw std::runtime_error("Data type not implemented.");
            }
        }

        void ReferenceValidator::validateWarmups(std::shared_ptr<ContractionInputs> inputs,
                                                 TimingEvents const& startEvents,
                                                 TimingEvents const&  stopEvents)
        {
            if(m_enabled && !m_validatedSolution)
            {
              validateSolution(inputs);
              m_validatedSolution = true;
            }
        }

        template <typename ManagedInputs>
        bool ReferenceValidator::validateTyped(ManagedInputs const& reference, ManagedInputs const& result)
        {
            bool rv = false;
            if(!m_enabled)
                return rv;

            if(m_printTensorA || m_printTensorB || m_printTensorC || m_printTensorD)
                printTensorsTyped(reference, result);

            if(m_elementsToValidate != 0)
                rv = checkResultsTyped(reference, result);

            return rv;
        }

        template <typename ManagedInputs>
        void ReferenceValidator::printTensorsTyped(ManagedInputs const& reference, ManagedInputs const& result)
        {
            size_t requiredBufferSize = 0;

            std::cout << "alpha: " << result.alpha << ", beta: " << result.beta << std::endl;

            if(m_printTensorA) requiredBufferSize = std::max(requiredBufferSize, m_problem.a().totalAllocatedBytes());
            if(m_printTensorB) requiredBufferSize = std::max(requiredBufferSize, m_problem.b().totalAllocatedBytes());
            if(m_printTensorC) requiredBufferSize = std::max(requiredBufferSize, m_problem.c().totalAllocatedBytes());
            if(m_printTensorD) requiredBufferSize = std::max(requiredBufferSize, m_problem.d().totalAllocatedBytes());

            if(m_cpuResultBuffer.size() < requiredBufferSize)
                m_cpuResultBuffer.resize(requiredBufferSize);

            if(m_printTensorA)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(), result.a,
                                        m_problem.a().totalAllocatedBytes(), hipMemcpyDeviceToHost));
                auto const* buffer = reinterpret_cast<typename ManagedInputs::AType const*>(m_cpuResultBuffer.data());

                m_reporter->logTensor(LogLevel::Verbose, "A", buffer, m_problem.a());
            }

            if(m_printTensorB)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(), result.b,
                                        m_problem.b().totalAllocatedBytes(), hipMemcpyDeviceToHost));
                auto const* buffer = reinterpret_cast<typename ManagedInputs::BType const*>(m_cpuResultBuffer.data());

                m_reporter->logTensor(LogLevel::Verbose, "B", buffer, m_problem.b());
            }

            if(m_printTensorC)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(), result.c,
                                        m_problem.c().totalAllocatedBytes(), hipMemcpyDeviceToHost));
                auto const* buffer = reinterpret_cast<typename ManagedInputs::CType const*>(m_cpuResultBuffer.data());

                m_reporter->logTensor(LogLevel::Verbose, "C", buffer, m_problem.c());
            }

            if(m_printTensorD)
            {
                HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(), result.d,
                                        m_problem.d().totalAllocatedBytes(), hipMemcpyDeviceToHost));
                auto const* buffer = reinterpret_cast<typename ManagedInputs::DType const*>(m_cpuResultBuffer.data());

                m_reporter->logTensor(LogLevel::Verbose, "D", buffer, m_problem.d());
            }
            if(m_printTensorRef)
            {
                m_reporter->logTensor(LogLevel::Verbose, "Reference-D", reference.d, m_problem.d());
            }
        }

        template <typename ManagedInputs>
        bool ReferenceValidator::checkResultsTyped(ManagedInputs const& reference, ManagedInputs const& result)
        {
            using Type = typename ManagedInputs::DType;
            auto const& tensor = m_problem.d();

            size_t elementsToCopy = tensor.totalAllocatedElements();
            if(m_boundsCheck)
                elementsToCopy = result.dElements;
            size_t bytesToCopy = elementsToCopy * sizeof(Type);

            if(m_cpuResultBuffer.size() < bytesToCopy)
                m_cpuResultBuffer.resize(bytesToCopy);

            HIP_CHECK_EXC(hipMemcpy(m_cpuResultBuffer.data(),
                                    result.managedD.get(),
                                    bytesToCopy, hipMemcpyDeviceToHost));

            auto elementsBeforeData = result.d - result.managedD.get();
            auto elementsAfterData = elementsToCopy - (tensor.totalAllocatedElements() + elementsBeforeData);

            // If there was extra data allocated before the tensor to do bounds
            // checking, resultBuffer is the whole allocation, while resultData
            // points directly to the result.
            Type const* resultBuffer = reinterpret_cast<Type const*>(m_cpuResultBuffer.data());
            Type const* resultData = resultBuffer + elementsBeforeData;
            Type const* resultAfterData = resultData + tensor.totalAllocatedElements();
            
            int printed = 0;

            bool doPrint = m_printMax < 0 || printed < m_printMax;

            size_t errors = 0;

            size_t boundsCheckElements = 0;

            bool printedPreBuffer = false;
            bool printedInsideBuffer = false;
            bool printedPostBuffer = false;

            for(ptrdiff_t i = 0; i < elementsBeforeData; i++)
            {
                boundsCheckElements++;
                if(!DataInitialization::isBadOutput(resultBuffer[i]))
                {
                    errors++;
                    if(doPrint)
                    {
                        if(!printedPreBuffer)
                        {
                            std::cout << "Value written before output buffer:" << std::endl;
                            printedPreBuffer = true;
                        }
                         
                        std::cout << "Index " << i << " / " << elementsBeforeData
                                  << ": found " << resultBuffer[i]
                                  << " instead of "
                                  << DataInitialization::getValue<Type, InitMode::BadOutput>()
                                  << std::endl;
                    }
                }
            }

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

                size_t prevBaseIndex = 0;
                const size_t innerDimSize = tensor.sizes()[0];

                for(size_t i = 0; i < outerCount; i++)
                {
                    CoordNumbered(i, coord.begin()+1, coord.end(), tensor.sizes().begin()+1, tensor.sizes().end());
                    size_t baseElemIndex = tensor.index(coord);

                    if(m_boundsCheck
                    && baseElemIndex != 0
                    && baseElemIndex != prevBaseIndex + innerDimSize)
                    {
                        for(auto innerIndex = prevBaseIndex + innerDimSize; innerIndex < baseElemIndex; innerIndex++)
                        {
                            boundsCheckElements++;
                            if(!DataInitialization::isBadOutput(resultData[innerIndex]))
                            {
                                errors++;
                                if(doPrint)
                                {
                                    if(!printedInsideBuffer)
                                    {
                                        std::cout << "Value written outside tensor, inside output buffer:" << std::endl;
                                        printedInsideBuffer = true;
                                    }
                                    
                                    std::cout << "Index " << innerIndex << " / " << baseElemIndex
                                            << ": found " << resultData[innerIndex]
                                            << " instead of "
                                            << DataInitialization::getValue<Type, InitMode::BadOutput>()
                                            << std::endl;
                                }
                            }
                        }
                    }

                    prevBaseIndex = baseElemIndex;

                    for(size_t j = 0; j < innerDimSize; j++)
                    {
                        size_t elemIndex = baseElemIndex + j;

                        Type referenceValue = reference.d[elemIndex];
                        Type resultValue = resultData[elemIndex];

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
                    Type resultValue = resultData[elemIndex];

                    compareValues(referenceValue, resultValue, elemIndex, elemNumber);
                }
            }

            for(ptrdiff_t i = 0; i < elementsAfterData; i++)
            {
                boundsCheckElements++;
                if(!DataInitialization::isBadOutput(resultAfterData[i]))
                {
                    errors++;
                    if(doPrint)
                    {
                        if(!printedPostBuffer)
                        {
                            std::cout << "Value written after output buffer:" << std::endl;
                            printedPreBuffer = true;
                        }
                         
                        std::cout << "Index " << i << " / " << elementsAfterData
                                  << ": found " << resultAfterData[i]
                                  << " instead of "
                                  << DataInitialization::getValue<Type, InitMode::BadOutput>()
                                  << std::endl;
                    }
                }
            }

            if(boundsCheckElements > 0)
                std::cout << "Performed bounds check on " << boundsCheckElements
                          << " elements" << std::endl;

            if(errors > 0)
            {
                m_errorInSolution = true;
                m_error = true;
            }

            return (errors > 0);
        }

        void ReferenceValidator::postSolution()
        {
            if(m_enabled && !m_validatedSolution)
                return;

            if(m_elementsToValidate != 0)
            {
                if(m_errorInConvolutionVsContraction)
                {
                    m_errorsReported++;
                    m_reporter->report(ResultKey::Validation, "FAILED_CONV");
                }
                else if(m_errorInSolution)
                {
                    m_errorsReported++;
                    m_reporter->report(ResultKey::Validation, "FAILED");
                }
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
            return m_errorsReported;
        }
    }
}



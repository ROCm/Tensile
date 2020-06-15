/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <Tensile/ArithmeticUnitTypes.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/Predicates.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace Tensile
{
    namespace Predicates
    {
        /**
 * \addtogroup Predicates
 * @{
 */
        /**
 * @brief ContractionProblem predicates
 */
        namespace Contraction
        {
            struct FreeSizeAMultiple : public Predicate_CRTP<FreeSizeAMultiple, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                FreeSizeAMultiple() = default;
                FreeSizeAMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "FreeSizeAMultiple";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.freeSizeA(index) % value == 0;
                }
            };

            struct FreeSizeBMultiple : public Predicate_CRTP<FreeSizeBMultiple, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                FreeSizeBMultiple() = default;
                FreeSizeBMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "FreeSizeBMultiple";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.freeSizeB(index) % value == 0;
                }
            };

            struct BatchSizeMultiple : public Predicate_CRTP<BatchSizeMultiple, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                BatchSizeMultiple() = default;
                BatchSizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "BatchSizeMultiple";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.batchSize(index) % value == 0;
                }
            };

            struct BatchSizeEqual : public Predicate_CRTP<BatchSizeEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                BatchSizeEqual() = default;
                BatchSizeEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "BatchSizeEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.batchSize(index) == value;
                }
            };

            struct BoundSizeMultiple : public Predicate_CRTP<BoundSizeMultiple, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                int64_t index;
                size_t  value;

                BoundSizeMultiple() = default;
                BoundSizeMultiple(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "BoundSizeMultiple";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    if(index < 0)
                        return problem.boundSize(problem.boundIndices().size() + index) % value
                               == 0;
                    else
                        return problem.boundSize(index) % value == 0;
                }
            };

            struct ProblemSizeEqual : public Predicate_CRTP<ProblemSizeEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                ProblemSizeEqual() = default;
                ProblemSizeEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "ProblemSizeEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.problemSizes()[index] == value;
                }
            };

            struct MaxProblemSizeGreaterThan
                : public Predicate_CRTP<MaxProblemSizeGreaterThan, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                MaxProblemSizeGreaterThan() = default;
                MaxProblemSizeGreaterThan(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "MaxProblemSizeGreaterThan";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.maxProblemSize() > value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.maxProblemSize() << " > " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            // If the tensor contains no free dimensions, then the first batch dimension
            // serves as the leading free size
            struct LeadingFreeSizesGreaterOrEqual
                : public Predicate_CRTP<LeadingFreeSizesGreaterOrEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                size_t value;

                LeadingFreeSizesGreaterOrEqual() = default;
                LeadingFreeSizesGreaterOrEqual(size_t value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "LeadingFreeSizesGreaterOrEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    assert(problem.batchIndices().size() <= 1);
                    // TODO: this is not quite right since it assumes batchSize(0) is lowest
                    // order in index assignments
                    //   If tensor contains multiple batch dims this may not be true.
                    //   Really should modify Contractions.py to select SizeN >= value, based on
                    //   desired index requirement
                    return (problem.freeIndicesA().size() ? problem.freeSizeA(0) >= value
                                                          : problem.batchSize(0) >= value)
                           && (problem.freeIndicesB().size() ? problem.freeSizeB(0) >= value
                                                             : problem.batchSize(0) >= value);
                }
                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": ("
                           << (problem.freeIndicesA().size() ? "freeA0:" : "batchA0:")
                           << (problem.freeIndicesA().size() ? problem.freeSizeA(0)
                                                             : problem.batchSize(0))
                           << " >= " << value << " && "
                           << (problem.freeIndicesB().size() ? "freeB0:" : "batchB0:")
                           << (problem.freeIndicesB().size() ? problem.freeSizeB(0)
                                                             : problem.batchSize(0))
                           << " >= " << value << ") == " << rv;

                    return rv;
                }
            };

            struct SizeEqual : public Predicate_CRTP<SizeEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                SizeEqual() = default;
                SizeEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "SizeEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.size(index) == value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.size(index) << " == " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct StrideAEqual : public Predicate_CRTP<StrideAEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideAEqual() = default;
                StrideAEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideAEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.a().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.a().strides()[index] << " == " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct StrideBEqual : public Predicate_CRTP<StrideBEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideBEqual() = default;
                StrideBEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideBEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.b().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.b().strides()[index] << " == " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct StrideCEqual : public Predicate_CRTP<StrideCEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideCEqual() = default;
                StrideCEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideCEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.c().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.c().strides()[index] << " == " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct StrideDEqual : public Predicate_CRTP<StrideDEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = true,
                    HasValue = true
                };
                size_t index;
                size_t value;

                StrideDEqual() = default;
                StrideDEqual(size_t index, size_t value)
                    : index(index)
                    , value(value)
                {
                }

                static std::string Type()
                {
                    return "StrideDEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.d().strides()[index] == value;
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << *this << ": (" << problem.d().strides()[index] << " == " << value
                           << ") == " << rv;

                    return rv;
                }
            };

            struct CDStridesEqual : public Predicate_CRTP<CDStridesEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                bool value;

                CDStridesEqual() = default;
                CDStridesEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "CDStridesEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return value == (problem.c().strides() == problem.d().strides());
                }
            };

            struct LDCEqualsLDD : public Predicate_CRTP<LDCEqualsLDD, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                static std::string Type()
                {
                    return "LDCEqualsLDD";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.c().strides()[1] == problem.d().strides()[1];
                }
            };

            struct BetaZero : public Predicate_CRTP<BetaZero, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                BetaZero() = default;

                static std::string Type()
                {
                    return "BetaZero";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.beta() == 0.0;
                }
            };

            struct BetaOne : public Predicate_CRTP<BetaOne, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = false
                };
                BetaOne() = default;

                static std::string Type()
                {
                    return "BetaOne";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.beta() == 1.0;
                }
            };

            struct HighPrecisionAccumulateEqual
                : public Predicate_CRTP<HighPrecisionAccumulateEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                HighPrecisionAccumulateEqual() = default;
                HighPrecisionAccumulateEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "HighPrecisionAccumulate";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.highPrecisionAccumulate() == value;
                }
            };

            struct KernelLanguageEqual
                : public Predicate_CRTP<KernelLanguageEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                KernelLanguage value;

                KernelLanguageEqual() = default;

                static std::string Type()
                {
                    return "KernelLanguage";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.kernelLanguage() == value;
                }
            };

            struct DeterministicModeEqual
                : public Predicate_CRTP<DeterministicModeEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                bool value;

                DeterministicModeEqual() = default;
                DeterministicModeEqual(bool value)
                    : value(value)
                {
                }

                static std::string Type()
                {
                    return "DeterministicMode";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.deterministicMode() == value;
                }
            };

            struct ArithmeticUnitEqual
                : public Predicate_CRTP<ArithmeticUnitEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };

                ArithmeticUnit value;

                ArithmeticUnitEqual() = default;

                static std::string Type()
                {
                    return "ArithmeticUnit";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.arithmeticUnit() == value;
                }
            };

            struct TypesEqual : public Predicate_CRTP<TypesEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                TypesEqual() = default;

                std::array<DataType, 4> value;

                static std::string Type()
                {
                    return "TypesEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.a().dataType() == value[0] && problem.b().dataType() == value[1]
                           && problem.c().dataType() == value[2]
                           && problem.d().dataType() == value[3];
                }

                virtual std::string toString() const override
                {
                    return concatenate(this->type(),
                                       "(a:",
                                       value[0],
                                       ", b:",
                                       value[1],
                                       ", c:",
                                       value[2],
                                       ", d:",
                                       value[3],
                                       ")");
                }

                virtual bool debugEval(ContractionProblem const& problem,
                                       std::ostream&             stream) const override
                {
                    bool rv = (*this)(problem);

                    stream << this->type() << "(a:" << problem.a().dataType() << " == " << value[0]
                           << "&& b:" << problem.b().dataType() << " == " << value[1]
                           << "&& c:" << problem.c().dataType() << " == " << value[2]
                           << "&& d:" << problem.d().dataType() << " == " << value[3]
                           << "): " << rv;

                    return rv;
                }
            };

            struct OperationIdentifierEqual
                : public Predicate_CRTP<OperationIdentifierEqual, ContractionProblem>
            {
                enum
                {
                    HasIndex = false,
                    HasValue = true
                };
                OperationIdentifierEqual() = default;

                std::string value;

                static std::string Type()
                {
                    return "OperationIdentifierEqual";
                }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.operationIdentifier() == value;
                }
            };
        } // namespace Contraction

        /**
 * @}
 */
    } // namespace Predicates
} // namespace Tensile

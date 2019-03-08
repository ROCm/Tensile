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

#pragma once

#include <Tensile/Predicates.hpp>
#include <Tensile/ContractionProblem.hpp>

#include <array>
#include <vector>

namespace Tensile
{
    namespace Predicates
    {
        namespace Contraction
        {
            struct FreeSizeAMultiple: public Predicate<ContractionProblem>
            {
                enum { HasIndex = true, HasValue = true };
                size_t index;
                size_t value;

                FreeSizeAMultiple() = default;
                FreeSizeAMultiple(size_t index, size_t value): index(index), value(value) {}

                static std::string Type() { return "FreeSizeAMultiple"; }
                virtual std::string type() const override { return Type(); }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.freeSizeA(index) % value == 0;
                }
            };

            struct FreeSizeBMultiple: public Predicate<ContractionProblem>
            {
                enum { HasIndex = true, HasValue = true };
                size_t index;
                size_t value;

                FreeSizeBMultiple() = default;
                FreeSizeBMultiple(size_t index, size_t value): index(index), value(value) {}

                static std::string Type() { return "FreeSizeBMultiple"; }
                virtual std::string type() const override { return Type(); }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.freeSizeA(index) % value == 0;
                }
            };

            struct BatchSizeMultiple: public Predicate<ContractionProblem>
            {
                enum { HasIndex = true, HasValue = true };
                size_t index;
                size_t value;

                BatchSizeMultiple() = default;
                BatchSizeMultiple(size_t index, size_t value): index(index), value(value) {}

                static std::string Type() { return "BatchSizeMultiple"; }
                virtual std::string type() const override { return Type(); }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.batchSize(index) % value == 0;
                }
            };

            struct BoundSizeMultiple: public Predicate<ContractionProblem>
            {
                enum { HasIndex = true, HasValue = true };
                size_t index;
                size_t value;

                BoundSizeMultiple() = default;
                BoundSizeMultiple(size_t index, size_t value): index(index), value(value) {}

                static std::string Type() { return "BoundSizeMultiple"; }
                virtual std::string type() const override { return Type(); }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.boundSize(index) % value == 0;
                }
            };

            struct MaxProblemSizeGreaterThan: public Predicate<ContractionProblem>
            {
                enum { HasIndex = false, HasValue = true };
                size_t value;

                MaxProblemSizeGreaterThan() = default;
                MaxProblemSizeGreaterThan(size_t value): value(value) {}

                static std::string Type() { return "MaxProblemSizeGreaterThan"; }
                virtual std::string type() const override { return Type(); }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.maxProblemSize() > value;
                }
            };

            struct CDStridesEqual: public Predicate<ContractionProblem>
            {
                enum { HasIndex = false, HasValue = false };
                static std::string Type() { return "CDStridesEqual"; }
                virtual std::string type() const override { return Type(); }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.c().strides() == problem.d().strides();
                }
            };

            struct LDCEqualsLDD: public Predicate<ContractionProblem>
            {
                enum { HasIndex = false, HasValue = false };
                static std::string Type() { return "LDCEqualsLDD"; }
                virtual std::string type() const override { return Type(); }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.c().strides()[1] == problem.d().strides()[1];
                }
            };

            struct BetaZero: public Predicate<ContractionProblem>
            {
                enum { HasIndex = false, HasValue = false };
                BetaZero() = default;

                static std::string Type() { return "BetaZero"; }
                virtual std::string type() const override { return Type(); }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.beta() == 0.0;
                }
            };

            struct BetaOne: public Predicate<ContractionProblem>
            {
                enum { HasIndex = false, HasValue = false };
                BetaOne() = default;

                static std::string Type() { return "BetaOne"; }
                virtual std::string type() const override { return Type(); }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.beta() == 1.0;
                }
            };

            struct TypesEqual: public Predicate<ContractionProblem>
            {
                enum { HasIndex = false, HasValue = true };
                TypesEqual() = default;

                std::array<DataType, 4> value;

                static std::string Type() { return "TypesEqual"; }
                virtual std::string type() const override { return Type(); }

                virtual bool operator()(ContractionProblem const& problem) const override
                {
                    return problem.a().dataType() == value[0]
                        && problem.b().dataType() == value[1]
                        && problem.c().dataType() == value[2]
                        && problem.d().dataType() == value[3];
                }
            };
        }
    }
}


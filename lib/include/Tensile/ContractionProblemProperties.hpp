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

#pragma once

#include <Tensile/PropertyMatching.hpp>
#include <Tensile/ContractionProblem.hpp>

namespace Tensile
{
    namespace Contraction
    {
        struct FreeSizeA: public Property<ContractionProblem>
        {
            enum { HasIndex = true, HasValue = false };
            size_t index;

            static std::string Type() { return "FreeSizeA"; }
            virtual std::string type() const { return Type(); }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.freeSizeA(index);
            }
        };

        struct FreeSizeB: public Property<ContractionProblem>
        {
            enum { HasIndex = true, HasValue = false };
            size_t index;

            static std::string Type() { return "FreeSizeB"; }
            virtual std::string type() const { return Type(); }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.freeSizeB(index);
            }
        };

        struct BatchSize: public Property<ContractionProblem>
        {
            enum { HasIndex = true, HasValue = false };
            size_t index;

            static std::string Type() { return "BatchSize"; }
            virtual std::string type() const { return Type(); }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.batchSize(index);
            }
        };

        struct BoundSize: public Property<ContractionProblem>
        {
            enum { HasIndex = true, HasValue = false };
            size_t index;

            static std::string Type() { return "BoundSize"; }
            virtual std::string type() const { return Type(); }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.boundSize(index);
            }
        };

        struct AStride: public Property<ContractionProblem>
        {
            enum { HasIndex = true, HasValue = false };
            size_t index;

            static std::string Type() { return "AStride"; }
            virtual std::string type() const { return Type(); }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.a().strides()[index];
            }
        };

        struct BStride: public Property<ContractionProblem>
        {
            enum { HasIndex = true, HasValue = false };
            size_t index;

            static std::string Type() { return "BStride"; }
            virtual std::string type() const { return Type(); }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.b().strides()[index];
            }
        };

        struct CStride: public Property<ContractionProblem>
        {
            enum { HasIndex = true, HasValue = false };
            size_t index;

            static std::string Type() { return "CStride"; }
            virtual std::string type() const { return Type(); }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.c().strides()[index];
            }
        };

        struct DStride: public Property<ContractionProblem>
        {
            enum { HasIndex = true, HasValue = false };
            size_t index;

            static std::string Type() { return "DStride"; }
            virtual std::string type() const { return Type(); }

            virtual size_t operator()(ContractionProblem const& problem) const
            {
                return problem.d().strides()[index];
            }
        };

        struct OperationIdentifier: public Property<ContractionProblem, std::string>
        {
            enum { HasIndex = false, HasValue = false };

            static std::string Type() { return "OperationIdentifier"; }
            virtual std::string type() const { return Type(); }

            virtual std::string operator()(ContractionProblem const& problem) const
            {
                return problem.operationIdentifier();
            }
        };
    }
}


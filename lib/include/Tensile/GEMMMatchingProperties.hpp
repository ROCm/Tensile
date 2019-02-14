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

namespace Tensile
{
    namespace Matching
    {
        namespace GEMM
        {
            struct I: public Property<GEMMProblem>
            {
                static std::string Key() { return "I"; }
                virtual std::string key() const { return Key(); }

                virtual size_t operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_I();
                }
            };

            struct J: public Property<GEMMProblem>
            {
                static std::string Key() { return "J"; }
                virtual std::string key() const { return Key(); }

                virtual size_t operator()(GEMMProblem const& problem) const
                {
                    return problem.blas_n();
                }
            };

            struct K: public Property<GEMMProblem>
            {
                static std::string Key() { return "K"; }
                virtual std::string key() const { return Key(); }

                virtual size_t operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_K();
                }
            };

            struct L: public Property<GEMMProblem>
            {
                static std::string Key() { return "L"; }
                virtual std::string key() const { return Key(); }

                virtual size_t operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_L();
                }
            };

            struct LDA: public Property<GEMMProblem>
            {
                static std::string Key() { return "LDA"; }
                virtual std::string key() const { return Key(); }

                virtual size_t operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_strideA1();
                }
            };

            struct LDB: public Property<GEMMProblem>
            {
                static std::string Key() { return "LDB"; }
                virtual std::string key() const { return Key(); }

                virtual size_t operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_strideB1();
                }
            };

            struct LDC: public Property<GEMMProblem>
            {
                static std::string Key() { return "LDC"; }
                virtual std::string key() const { return Key(); }

                virtual size_t operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_strideC1();
                }
            };

            struct LDD: public Property<GEMMProblem>
            {
                static std::string Key() { return "LDD"; }
                virtual std::string key() const { return Key(); }

                virtual size_t operator()(GEMMProblem const& problem) const
                {
                    return problem.tensile_strideD1();
                }
            };
        }
    }
}


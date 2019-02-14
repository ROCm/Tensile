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

#pragma once

#include <Tensile/Tensile.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/TensorOps.hpp>

namespace Tensile
{

    class GEMMProblem: public Problem
    {
    public:
        GEMMProblem() = default;

        virtual std::string description() const { return "asdf"; }

        static GEMMProblem FromBLAS(bool transA, bool transB,
                                    size_t m, size_t n, size_t k,
                                    size_t lda, size_t ldb, size_t ldc,
                                    bool useBeta, bool colMajor, size_t batchCount);

        static GEMMProblem FromTensile(/* TODO */);

        GEMMProblem(TensorDescriptor const& a, TensorOps const& aOps,
                    TensorDescriptor const& b, TensorOps const& bOps,
                    TensorDescriptor const& c, TensorOps const& cOps,
                    TensorDescriptor const& d, TensorOps const& dOps,
                    bool useBeta);

        void normalize();
        void consistencyCheck() const;

        size_t blas_m()          const { return a.logicalCounts()[0]; }
        size_t blas_n()          const { return b.logicalCounts()[1]; }
        size_t blas_k()          const { return a.logicalCounts()[1]; }
        size_t blas_batchCount() const { return a.logicalCounts()[2]; }

        bool blas_transA() const;
        bool blas_transB() const;

        size_t tensile_I() const { return d.logicalCounts()[0]; }
        size_t tensile_J() const { return d.logicalCounts()[1]; }
        size_t tensile_K() const { return d.logicalCounts()[2]; }
        size_t tensile_L() const { return a.logicalCounts()[1]; }

        size_t tensile_strideA1() const;
        size_t tensile_strideA2() const;

        size_t tensile_strideB1() const;
        size_t tensile_strideB2() const;

        size_t tensile_strideC1() const;
        size_t tensile_strideC2() const;

        size_t tensile_strideD1() const;
        size_t tensile_strideD2() const;


        bool useBeta;
        TensorDescriptor a, b, c, d;
        TensorOps aOps, bOps, cOps, dOps;
    };

    struct GEMMInputs: public ProblemInputs
    {
        GEMMInputs() = default;
        
        float const* a;
        float const* b;
        float const* c;
        float      * d;

        float alpha;
        float beta;
    };
}


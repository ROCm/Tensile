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

    class ContractionProblem: public Problem
    {
    public:
        ContractionProblem() = default;

        struct FreeIndex
        {
            size_t a, b, ca, cb, da, db;
        };
        using FreeIndices = std::vector<FreeIndex>;

        struct BatchIndex
        {
            size_t a, b, c, d;
        };
        using BatchIndices = std::vector<BatchIndex>;

        struct BoundIndex
        {
            size_t a, b;
        };
        using BoundIndices = std::vector<BoundIndex>;

        virtual std::string description() const { return "asdf"; }

        static ContractionProblem GEMM(bool transA, bool transB,
                                       size_t m, size_t n, size_t k,
                                       size_t lda, size_t ldb, size_t ldc,
                                       bool useBeta, bool colMajor, size_t batchCount);

        static ContractionProblem FromTensile(/* TODO */);

        ContractionProblem(TensorDescriptor const& a, TensorOps const& aOps,
                           TensorDescriptor const& b, TensorOps const& bOps,
                           TensorDescriptor const& c, TensorOps const& cOps,
                           TensorDescriptor const& d, TensorOps const& dOps,
                           FreeIndices  const& freeIndices,
                           BatchIndices const& batchIndices,
                           BoundIndices const& boundIndices);

        void normalize();
        void consistencyCheck() const;

        size_t freeSizeA(size_t idx);
        size_t freeSizeB(size_t idx);
        size_t batchSize(size_t idx);
        size_t boundSize(size_t idx);

        /*
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
        */

        TensorDescriptor a;
        TensorDescriptor b;
        TensorDescriptor c;
        TensorDescriptor d;
        TensorOps aOps;
        TensorOps bOps;
        TensorOps cOps;
        TensorOps dOps;

        FreeIndices freeIndices;
        BatchIndices batchIndices;
        BoundIndices boundIndices;
    };

    template <typename A = float, typename B = A, typename C = A, typename D = A, typename Alpha = D, typename Beta = D>
    struct ContractionInputs: public ProblemInputs
    {
        ContractionInputs() = default;
        
        A const* a = nullptr;
        B const* b = nullptr;
        C const* c = nullptr;
        D      * d = nullptr;

        Alpha alpha = 0;
        Beta beta   = 0;
    };
}


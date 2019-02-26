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
#include <Tensile/Utils.hpp>

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
                                       double beta, bool colMajor, size_t batchCount);

        static ContractionProblem FromTensile(/* TODO */);

        ContractionProblem(TensorDescriptor const& a, TensorOps const& aOps,
                           TensorDescriptor const& b, TensorOps const& bOps,
                           TensorDescriptor const& c, TensorOps const& cOps,
                           TensorDescriptor const& d, TensorOps const& dOps,
                           FreeIndices  const& freeIndices,
                           BatchIndices const& batchIndices,
                           BoundIndices const& boundIndices,
                           double beta);


        size_t freeSizeA(size_t idx);
        size_t freeSizeB(size_t idx);
        size_t batchSize(size_t idx);
        size_t boundSize(size_t idx);

        TensorDescriptor const& a() const { return m_a; }
        TensorDescriptor const& b() const { return m_b; }
        TensorDescriptor const& c() const { return m_c; }
        TensorDescriptor const& d() const { return m_d; }

        TensorOps const& aOps() const { return m_aOps; }
        TensorOps const& bOps() const { return m_bOps; }
        TensorOps const& cOps() const { return m_cOps; }
        TensorOps const& dOps() const { return m_dOps; }

        FreeIndices  const&  freeIndices() const { return m_freeIndices; }
        BatchIndices const& batchIndices() const { return m_batchIndices; }
        BoundIndices const& boundIndices() const { return m_boundIndices; }

        double beta() const { return m_beta; }

        std::string const& aNames()   const { return m_aNames; }
        std::string const& bNames()   const { return m_bNames; }
        std::string const& cNames()   const { return m_cNames; }
        std::string const& dNames()   const { return m_dNames; }
        std::string const& sumNames() const { return m_sumNames; }

        std::string operationName() const;
        std::string const& operationIdentifer()   const { return m_operationIdentifier; }
        std::string        operationDescription() const { return getOperationDescription(); }

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

    private:
        TensorDescriptor m_a;
        TensorDescriptor m_b;
        TensorDescriptor m_c;
        TensorDescriptor m_d;
        TensorOps m_aOps;
        TensorOps m_bOps;
        TensorOps m_cOps;
        TensorOps m_dOps;

        std::string m_aNames;
        std::string m_bNames;
        std::string m_cNames;
        std::string m_dNames;
        std::string m_sumNames;
        std::string m_operationIdentifier;

        FreeIndices m_freeIndices;
        BatchIndices m_batchIndices;
        BoundIndices m_boundIndices;

        double m_beta;

        void normalize();
        void consistencyCheck() const;

        void getIndexNames(std::string & aNames,
                           std::string & bNames,
                           std::string & cNames,
                           std::string & dNames,
                           std::string & sumNames) const;

        std::string getOperationIdentifier() const;
        std::string getOperationDescription() const;
    };

    template <>
    struct Comparison<ContractionProblem::FreeIndex>
    {
        enum { implemented = true };

        static int compare(ContractionProblem::FreeIndex const& lhs, ContractionProblem::FreeIndex const& rhs)
        {
            return LexicographicCompare(lhs.da, rhs.da,
                                        lhs.db, rhs.db,
                                        lhs.ca, rhs.ca,
                                        lhs.cb, rhs.cb,
                                        lhs.a,  rhs.a,
                                        lhs.b,  rhs.b);
        }
    };

    template <>
    struct Comparison<ContractionProblem::BatchIndex>
    {
        enum { implemented = true };

        static int compare(ContractionProblem::BatchIndex const& lhs, ContractionProblem::BatchIndex const& rhs)
        {
            return LexicographicCompare(lhs.d, rhs.d,
                                        lhs.c, rhs.c,
                                        lhs.a, rhs.a,
                                        lhs.b, rhs.b);
        }
    };

    template <>
    struct Comparison<ContractionProblem::BoundIndex>
    {
        enum { implemented = true };

        static int compare(ContractionProblem::BoundIndex const& lhs, ContractionProblem::BoundIndex const& rhs)
        {
            return LexicographicCompare(lhs.a, rhs.a,
                                        lhs.b, rhs.b);
        }
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

    std::ostream & operator<<(std::ostream & stream, ContractionProblem::FreeIndex  const& free);
    std::ostream & operator<<(std::ostream & stream, ContractionProblem::BatchIndex const& batch);
    std::ostream & operator<<(std::ostream & stream, ContractionProblem::BoundIndex const& bound);
}


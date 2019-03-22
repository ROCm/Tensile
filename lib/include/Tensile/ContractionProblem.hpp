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

#include <Tensile/ContractionSolution_fwd.hpp>
#include <Tensile/ContractionProblem_fwd.hpp>

#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/TensorOps.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{

    class TENSILE_API ContractionProblem: public Problem
    {
    public:
        using Solution = ContractionSolution;
        using Inputs   = ContractionInputs;

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


        size_t freeSizeA(size_t idx) const;
        size_t freeSizeB(size_t idx) const;
        size_t batchSize(size_t idx) const;
        size_t boundSize(size_t idx) const;

        /// Largest of the free and bound indices.  Does not include batch size.
        size_t maxProblemSize() const { return m_maxProblemSize; }

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

        bool transA() const { return m_transA; }
        bool transB() const { return m_transB; }

        std::string operationName() const;
        std::string const& operationIdentifier()   const { return m_operationIdentifier; }
        std::string        operationDescription() const { return getOperationDescription(); }

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

        bool m_transA;
        bool m_transB;

        FreeIndices m_freeIndices;
        BatchIndices m_batchIndices;
        BoundIndices m_boundIndices;

        double m_beta;

        size_t m_maxProblemSize = 1;

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

    struct TENSILE_API ContractionInputs: public ProblemInputs
    {
        ContractionInputs();
        ~ContractionInputs();
    };

    template <typename A, typename B, typename C, typename D, typename Alpha, typename Beta>
    struct TENSILE_API TypedContractionInputs: public ContractionInputs
    {
        using AType = A;
        using BType = B;
        using CType = C;
        using DType = D;
        using AlphaType = Alpha;
        using BetaType = Beta;

        TypedContractionInputs();
        ~TypedContractionInputs();
        
        A const* a = nullptr;
        B const* b = nullptr;
        C const* c = nullptr;
        D      * d = nullptr;

        Alpha alpha = 0;
        Beta beta   = 0;
    };

    TENSILE_API std::ostream & operator<<(std::ostream & stream, ContractionProblem::FreeIndex  const& free);
    TENSILE_API std::ostream & operator<<(std::ostream & stream, ContractionProblem::BatchIndex const& batch);
    TENSILE_API std::ostream & operator<<(std::ostream & stream, ContractionProblem::BoundIndex const& bound);
}


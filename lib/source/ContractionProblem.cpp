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

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblem_Detail.hpp>
#include <Tensile/Utils.hpp>

namespace Tensile
{
    ContractionProblem ContractionProblem::GEMM_Strides(bool transA, bool transB,
                                                        DataType aType, DataType bType, DataType cType, DataType dType,
                                                        size_t m, size_t n, size_t k, size_t batchSize,
                                                        size_t lda, size_t aStride,
                                                        size_t ldb, size_t bStride,
                                                        size_t ldc, size_t cStride,
                                                        size_t ldd, size_t dStride,
                                                        double beta)
    {
        Tensile::ContractionProblem::FreeIndices  free(1);
        Tensile::ContractionProblem::BoundIndices bound(1);
        Tensile::ContractionProblem::BatchIndices batch(1);

        free[0].ca = free[0].da = 0;
        free[0].cb = free[0].db = 1;

        batch[0].a = batch[0].b = batch[0].c = batch[0].d = 2;

        TensorDescriptor a, b, c, d;

        if(transA)
        {
            a = TensorDescriptor(aType, {k, m, batchSize}, {1, lda, aStride});
            free[0].a = 1;
            bound[0].a = 0;
        }
        else
        {
            a = TensorDescriptor(aType, {m, k, batchSize}, {1, lda, aStride});
            free[0].a = 0;
            bound[0].a = 1;
        }

        if(transB)
        {
            b = TensorDescriptor(bType, {n, k, batchSize}, {1, ldb, bStride});
            free[0].b = 0;
            bound[0].b = 1;
        }
        else
        {
            b = TensorDescriptor(bType, {k, n, batchSize}, {1, ldb, bStride});
            free[0].b = 1;
            bound[0].b = 0;
        }

        c = TensorDescriptor(cType, {m, n, batchSize}, {1, ldc, cStride});
        d = TensorDescriptor(dType, {m, n, batchSize}, {1, ldd, dStride});

        TensorOps nop;

        ContractionProblem problem(a, nop, b, nop, c, nop, d, nop,
                                   free, batch, bound, beta);

        return problem;
    }

    ContractionProblem ContractionProblem::GEMM(bool transA, bool transB,
                                                size_t m, size_t n, size_t k,
                                                size_t lda, size_t ldb, size_t ldc,
                                                double beta, bool colMajor, size_t batchCount)
    {
        if(colMajor) throw std::runtime_error("Column major not yet implemented.");

        FreeIndex free;
        BoundIndex bound;

        free.ca = free.da = 0;
        free.cb = free.db = 1;

        TensorDescriptor a, b, c, d;
        if(transA)
        {
            a = TensorDescriptor(DataType::Float, {k, m}, {1, lda});
            free.a = 1;
            bound.a = 0;
        }
        else
        {
            a = TensorDescriptor(DataType::Float, {m, k}, {1, lda});
            free.a = 0;
            bound.a = 1;
        }

        if(transB)
        {
            b = TensorDescriptor(DataType::Float, {n, k}, {1, ldb});
            free.b = 0;
            bound.b = 1;
        }
        else
        {
            b = TensorDescriptor(DataType::Float, {k, n}, {1, ldb});
            free.b = 1;
            bound.b = 0;
        }

        FreeIndices freeIndices{free};
        BatchIndices batchIndices;
        BoundIndices boundIndices{bound};

        d = TensorDescriptor(DataType::Float, {m, n}, {1, ldc});

        a.appendDim(batchCount);
        b.appendDim(batchCount);
        d.appendDim(batchCount);

        batchIndices.push_back({2,2,2,2});

        if(beta != 0.0)
            c = d;

        TensorOps nop;

        return ContractionProblem(a, nop, b, nop, c, nop, d, nop, freeIndices, batchIndices, boundIndices, beta);
    }

    ContractionProblem::ContractionProblem(TensorDescriptor const& a, TensorOps const& aOps,
                                           TensorDescriptor const& b, TensorOps const& bOps,
                                           TensorDescriptor const& c, TensorOps const& cOps,
                                           TensorDescriptor const& d, TensorOps const& dOps,
                                           FreeIndices  const& freeIndices,
                                           BatchIndices const& batchIndices,
                                           BoundIndices const& boundIndices,
                                           double beta)
        : m_a(a),
          m_b(b),
          m_c(c),
          m_d(d),
          m_aOps(aOps),
          m_bOps(bOps),
          m_cOps(cOps),
          m_dOps(dOps),
          m_freeIndices(freeIndices),
          m_batchIndices(batchIndices),
          m_boundIndices(boundIndices),
          m_beta(beta)
    {
        consistencyCheck();
        normalize();
    }

    void ContractionProblem::normalize()
    {
        std::sort(m_freeIndices.begin(),  m_freeIndices.end());
        std::sort(m_batchIndices.begin(), m_batchIndices.end());
        std::sort(m_boundIndices.begin(), m_boundIndices.end());

        getIndexNames(m_aNames, m_bNames, m_cNames, m_dNames, m_sumNames);

        m_operationIdentifier = getOperationIdentifier();

        m_maxProblemSize = 0;

        for(int i = 0; i < m_freeIndices.size(); i++)
            m_maxProblemSize = std::max({m_maxProblemSize, freeSizeA(i), freeSizeB(i)});

        for(int i = 0; i < m_boundIndices.size(); i++)
            m_maxProblemSize = std::max(m_maxProblemSize, boundSize(i));

        m_transA = m_aNames == "lik";
        m_transB = m_bNames == "jlk";
    }

    void ContractionProblem::consistencyCheck() const
    {
        std::vector<int> aUseCount(m_a.dimensions(), 0);
        std::vector<int> bUseCount(m_b.dimensions(), 0);
        std::vector<int> cUseCount(m_c.dimensions(), 0);
        std::vector<int> dUseCount(m_d.dimensions(), 0);

        for(FreeIndex const& free: m_freeIndices)
        {
            TENSILE_ASSERT_EXC(free.a  < m_a.dimensions());
            TENSILE_ASSERT_EXC(free.b  < m_b.dimensions());
            TENSILE_ASSERT_EXC(free.da < m_d.dimensions());
            TENSILE_ASSERT_EXC(free.db < m_d.dimensions());

            aUseCount[free.a]++;
            bUseCount[free.b]++;

            dUseCount[free.da]++;
            dUseCount[free.db]++;

            TENSILE_ASSERT_EXC(m_a.sizes()[free.a] == m_d.sizes()[free.da]);
            TENSILE_ASSERT_EXC(m_b.sizes()[free.b] == m_d.sizes()[free.db]);

            if(!m_c.empty())
            {
                TENSILE_ASSERT_EXC(free.ca < m_c.dimensions());
                TENSILE_ASSERT_EXC(free.cb < m_c.dimensions());

                cUseCount[free.ca]++;
                cUseCount[free.cb]++;

                TENSILE_ASSERT_EXC(m_a.sizes()[free.a] == m_c.sizes()[free.ca]);
                TENSILE_ASSERT_EXC(m_b.sizes()[free.b] == m_c.sizes()[free.cb]);
            }
        }

        for(BatchIndex const& batch: m_batchIndices)
        {
            TENSILE_ASSERT_EXC(batch.a < m_a.dimensions());
            TENSILE_ASSERT_EXC(batch.b < m_b.dimensions());
            TENSILE_ASSERT_EXC(batch.d < m_d.dimensions());

            aUseCount[batch.a]++;
            bUseCount[batch.b]++;
            dUseCount[batch.d]++;

            TENSILE_ASSERT_EXC(m_a.sizes()[batch.a] == m_b.sizes()[batch.b]);
            TENSILE_ASSERT_EXC(m_a.sizes()[batch.a] == m_d.sizes()[batch.b]);

            if(!m_c.empty())
            {
                TENSILE_ASSERT_EXC(batch.c < m_c.dimensions());
                cUseCount[batch.c]++;

                TENSILE_ASSERT_EXC(m_a.sizes()[batch.a] == m_c.sizes()[batch.b]);
            }
        }

        for(BoundIndex const& bound: m_boundIndices)
        {
            TENSILE_ASSERT_EXC(bound.a < m_a.dimensions());
            TENSILE_ASSERT_EXC(bound.b < m_b.dimensions());

            aUseCount[bound.a]++;
            bUseCount[bound.b]++;

            TENSILE_ASSERT_EXC(m_a.sizes()[bound.a] == m_b.sizes()[bound.b]);
        }

        for(int aUse: aUseCount) TENSILE_ASSERT_EXC(aUse == 1);
        for(int bUse: bUseCount) TENSILE_ASSERT_EXC(bUse == 1);
        for(int cUse: cUseCount) TENSILE_ASSERT_EXC(cUse == 1);
        for(int dUse: dUseCount) TENSILE_ASSERT_EXC(dUse == 1);
    }

    size_t ContractionProblem::freeSizeA(size_t idx) const
    {
        return m_a.sizes()[m_freeIndices[idx].a];
    }

    size_t ContractionProblem::freeSizeB(size_t idx) const
    {
        return m_b.sizes()[m_freeIndices[idx].b];
    }

    size_t ContractionProblem::batchSize(size_t idx) const
    {
        return m_a.sizes()[m_batchIndices[idx].a];
    }

    size_t ContractionProblem::boundSize(size_t idx) const
    {
        return m_a.sizes()[m_boundIndices[idx].a];
    }

    void ContractionProblem::getIndexNames(std::string & aNames,
                                           std::string & bNames,
                                           std::string & cNames,
                                           std::string & dNames,
                                           std::string & sumNames) const
    {
        aNames.resize(m_a.dimensions(), '_');
        bNames.resize(m_b.dimensions(), '_');
        cNames.resize(m_c.dimensions(), '_');
        dNames.resize(m_d.dimensions(), '_');
        sumNames.resize(m_boundIndices.size(), '_');

        char name = 'i';

        for(char & ch: dNames)
        {
            ch = name;
            name++;
        }

        for(char & ch: sumNames)
        {
            ch = name;
            name++;
        }

        for(auto const& free: m_freeIndices)
        {
            aNames[free.a] = dNames[free.da];
            bNames[free.b] = dNames[free.db];
            if(!m_c.empty())
            {
                cNames[free.ca] = dNames[free.da];
                cNames[free.cb] = dNames[free.db];
            }
        }

        for(auto const& batch: m_batchIndices)
        {
            aNames[batch.a] = dNames[batch.d];
            bNames[batch.b] = dNames[batch.d];
            if(!m_c.empty())
                cNames[batch.c] = dNames[batch.d];
        }

        for(size_t i = 0; i < sumNames.size(); i++)
        {
            aNames[m_boundIndices[i].a] = sumNames[i];
            bNames[m_boundIndices[i].b] = sumNames[i];
        }

        if(m_c.empty() || m_beta == 0.0)
            cNames = dNames;
    }

    std::string ContractionProblem::getOperationDescription() const
    {
        std::ostringstream rv;

        rv << "D[" << m_dNames << "] = alpha * (";

        if(!m_sumNames.empty())
            rv << "Sum[" << m_sumNames << "] ";

        rv << "A[" << m_aNames << "] * B[" << m_bNames << "])";

        if(!m_c.empty() && m_beta != 0)
        {
            rv << " + ";
            if(m_beta != 1.0)
                rv << "beta * ";
            rv << "C[" << m_cNames << "]";
        }

        return rv.str();
    }

    std::string ContractionProblem::getOperationIdentifier() const
    {
        std::ostringstream rv;

        rv << "Contraction";

        rv << "_" << m_sumNames;
        rv << "_A" << m_aNames;
        rv << "_B" << m_bNames;
        rv << "_C" << m_cNames;
        rv << "_D" << m_dNames;

        return rv.str();
    }

    std::ostream & operator<<(std::ostream & stream, ContractionProblem::FreeIndex  const& free)
    {
        return stream << "{a=" << free.a << " b=" << free.b
                      << " ca=" << free.ca << " cb=" << free.cb
                      << " da=" << free.da << " db=" << free.db << "}";
    }
    std::ostream & operator<<(std::ostream & stream, ContractionProblem::BatchIndex const& batch)
    {
        if(batch.a == batch.b && batch.a == batch.c && batch.a == batch.d)
            return stream << "{" << batch.a << "}";

        return stream << "{a=" << batch.a
                      << " b=" << batch.b
                      << " c=" << batch.c
                      << " d=" << batch.d << "}";
    }

    std::ostream & operator<<(std::ostream & stream, ContractionProblem::BoundIndex const& bound)
    {
        return stream << "{a=" << bound.a << " b=" << bound.b << "}";
    }

    ContractionInputs::ContractionInputs() = default;
    ContractionInputs::~ContractionInputs() = default;
    template <> TypedContractionInputs<float>::TypedContractionInputs() = default;
    template <> TypedContractionInputs<float>::~TypedContractionInputs() = default;
}


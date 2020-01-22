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

#include <cstddef>
#include <set>

namespace Tensile
{
    std::string ContractionProblem::ZeroPad::description() const
    {
        std::ostringstream rv;

        rv << "anchorIndex: " << anchorIndex
           << " boundIndex: "  << boundIndex
           << " leadingPad: "  << leadingPad
           << " trailingPad: " << trailingPad;

        return rv.str();
    }


    ContractionProblem ContractionProblem::GEMM_Strides(bool transA, bool transB,
                                                        DataType aType, DataType bType, DataType cType, DataType dType,
                                                        size_t m, size_t n, size_t k, size_t batchSize,
                                                        size_t lda, size_t aStride,
                                                        size_t ldb, size_t bStride,
                                                        size_t ldc, size_t cStride,
                                                        size_t ldd, size_t dStride,
                                                        double beta)
    {
        Tensile::ContractionProblem::FreeIndices  free(2);
        Tensile::ContractionProblem::BoundIndices bound(1);
        Tensile::ContractionProblem::BatchIndices batch(1);

        free[0].isA=true;
        free[0].i = free[0].c = free[0].d = 0;
        free[1].isA=false;
        free[1].i = free[1].c = free[1].d = 1;

        batch[0].a = batch[0].b = batch[0].c = batch[0].d = 2;

        TensorDescriptor a, b, c, d;

        if(transA)
        {
            a = TensorDescriptor(aType, {k, m, batchSize}, {1, lda, aStride});
            free[0].i = 1;
            bound[0].a = 0;
        }
        else
        {
            a = TensorDescriptor(aType, {m, k, batchSize}, {1, lda, aStride});
            free[0].i = 0;
            bound[0].a = 1;
        }

        if(transB)
        {
            b = TensorDescriptor(bType, {n, k, batchSize}, {1, ldb, bStride});
            free[1].i = 0;
            bound[0].b = 1;
        }
        else
        {
            b = TensorDescriptor(bType, {k, n, batchSize}, {1, ldb, bStride});
            free[1].i = 1;
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

        Tensile::ContractionProblem::FreeIndices  free(2);
        BoundIndex bound;

        free[0].isA=true;
        free[0].i = free[0].c = free[0].d = 0;
        free[1].isA=false;
        free[1].i = free[1].c = free[1].d = 1;

        TensorDescriptor a, b, c, d;
        if(transA)
        {
            a = TensorDescriptor(DataType::Float, {k, m}, {1, lda});
            free[0].i = 1;
            bound.a = 0;
        }
        else
        {
            a = TensorDescriptor(DataType::Float, {m, k}, {1, lda});
            free[0].i = 0;
            bound.a = 1;
        }

        if(transB)
        {
            b = TensorDescriptor(DataType::Float, {n, k}, {1, ldb});
            free[1].i = 0;
            bound.b = 1;
        }
        else
        {
            b = TensorDescriptor(DataType::Float, {k, n}, {1, ldb});
            free[1].i = 1;
            bound.b = 0;
        }

        FreeIndices  freeIndices{free};
        BatchIndices batchIndices;
        BoundIndices boundIndices{bound};

        d = TensorDescriptor(DataType::Float, {m, n}, {1, ldc});

        a.appendDim(batchCount);
        b.appendDim(batchCount);
        d.appendDim(batchCount);

        batchIndices.push_back({2,2,2,2});

        c = d;

        TensorOps nop;

        return ContractionProblem(a, nop, b, nop, c, nop, d, nop, freeIndices, batchIndices, boundIndices, beta);
    }

    ContractionProblem ContractionProblem::GEMM(bool transA, bool transB,
                                                TensorDescriptor const& a, TensorOps const& aOps,
                                                TensorDescriptor const& b, TensorOps const& bOps,
                                                TensorDescriptor const& c, TensorOps const& cOps,
                                                TensorDescriptor const& d, TensorOps const& dOps,
                                                double beta)
    {
        Tensile::ContractionProblem::FreeIndices  free(2);
        BoundIndex bound;

        free[0].isA=true;
        free[0].i = free[0].c = free[0].d = 0;
        free[1].isA=false;
        free[1].i = free[1].c = free[1].d = 1;

        if(transA)
        {
            free[0].i = 1;
            bound.a = 0;
        }
        else
        {
            free[0].i = 0;
            bound.a = 1;
        }

        if(transB)
        {
            free[1].i = 0;
            bound.b = 1;
        }
        else
        {
            free[1].i = 1;
            bound.b = 0;
        }

        FreeIndices  freeIndices{free};
        BatchIndices batchIndices;
        BoundIndices boundIndices{bound};

        batchIndices.push_back({2,2,2,2});

        return ContractionProblem(a, aOps, b, bOps, c, cOps, d, cOps, freeIndices, batchIndices, boundIndices, beta);
    }

    void ContractionProblem::IdentifierToIndices(std::string  const& identifier,
                                                 FreeIndices       & freeIndices,
                                                 BatchIndices      & batchIndices,
                                                 BoundIndices      & boundIndices,
                                                 TensorOps         & aOps,
                                                 TensorOps         & bOps,
                                                 TensorOps         & cOps,
                                                 TensorOps         & dOps)
    {
        FreeIndices  free;
        BatchIndices batch;
        BoundIndices bound;

        std::string prefix = "Contraction_";
        if(identifier.find(prefix) != 0)
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ") must start with '", prefix, "'."));

        size_t begin = prefix.size();
        size_t end = identifier.find("_", begin);
        size_t nextBegin = end+1;

        std::string boundStr = identifier.substr(begin, end-begin);

        begin = nextBegin;
        end = identifier.find("_", begin);
        nextBegin = end+1;

        if(identifier.at(begin) != 'A')
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        if(identifier.at(end-1) == 'C')
        {
            aOps.push_back(TensorOp::ComplexConjugate());
            end--;
        }

        begin++;
        std::string a = identifier.substr(begin, end-begin);

        begin = nextBegin;
        end = identifier.find("_", begin);
        nextBegin = end+1;

        if(identifier.at(begin) != 'B')
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        if(identifier.at(end-1) == 'C')
        {
            bOps.push_back(TensorOp::ComplexConjugate());
            end--;
        }

        begin++;
        std::string b = identifier.substr(begin, end-begin);

        begin = nextBegin;
        end = identifier.find("_", begin);
        nextBegin = end+1;

        if(identifier.at(begin) != 'C')
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        if(identifier.at(end-1) == 'C')
        {
            cOps.push_back(TensorOp::ComplexConjugate());
            end--;
        }

        begin++;
        std::string c = identifier.substr(begin, end-begin);

        begin = nextBegin;
        end = identifier.find("_", begin);

        if(identifier.at(begin) != 'D')
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        if(end != std::string::npos)
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        end = identifier.size();

        if(identifier.at(end-1) == 'C')
        {
            dOps.push_back(TensorOp::ComplexConjugate());
            end--;
        }

        begin++;
        std::string d = identifier.substr(begin, end-begin);

        std::set<char> allIndices(a.begin(), a.end());
        allIndices.insert(b.begin(), b.end());
        allIndices.insert(c.begin(), c.end());
        allIndices.insert(d.begin(), d.end());

        for(char index: allIndices)
        {
            size_t aIndex = a.find(index);
            size_t bIndex = b.find(index);
            size_t cIndex = c.find(index);
            size_t dIndex = d.find(index);

            if(aIndex != std::string::npos && bIndex != std::string::npos
            && cIndex != std::string::npos && dIndex != std::string::npos)
            {
                batch.push_back(BatchIndex{aIndex, bIndex, cIndex, dIndex});
            }
            else if(aIndex != std::string::npos && bIndex != std::string::npos
                 && cIndex == std::string::npos && dIndex == std::string::npos)
            {
                bound.push_back(BoundIndex{aIndex, bIndex});
            }
            else if(aIndex != std::string::npos && bIndex == std::string::npos
                 && cIndex != std::string::npos && dIndex != std::string::npos)
            {
                free.resize(free.size()+1);

                free.back().isA = true;
                free.back().i = aIndex;
                free.back().c = cIndex;
                free.back().d = dIndex;
            }
            else if(aIndex == std::string::npos && bIndex != std::string::npos
                 && cIndex != std::string::npos && dIndex != std::string::npos)
            {
                free.resize(free.size()+1);

                free.back().isA = false;
                free.back().i = bIndex;
                free.back().c = cIndex;
                free.back().d = dIndex;
            }
        }
        freeIndices  = std::move(free);
        batchIndices = std::move(batch);
        boundIndices = std::move(bound);
    }

    ContractionProblem ContractionProblem::FromIndexSizes(
            std::string const& operationIdentifier,
            std::vector<size_t> const& indexSizes,
            DataType aType, std::vector<size_t> const& aStrides,
            DataType bType, std::vector<size_t> const& bStrides,
            DataType cType, std::vector<size_t> const& cStrides,
            DataType dType, std::vector<size_t> const& dStrides,
            double beta)
    {
        FreeIndices freeIndices;
        BatchIndices batchIndices;
        BoundIndices boundIndices;

        TensorOps aOps, bOps, cOps, dOps;

        IdentifierToIndices(operationIdentifier,
                            freeIndices,
                            batchIndices,
                            boundIndices,
                            aOps, bOps, cOps, dOps);

        return FromIndexSizes(freeIndices, batchIndices, boundIndices,
                              indexSizes,
                              aType, aStrides, aOps,
                              bType, bStrides, bOps,
                              cType, cStrides, cOps,
                              dType, dStrides, dOps,
                              beta);
    }

    ContractionProblem ContractionProblem::FromIndexSizes(
            FreeIndices const& freeIndices,
            BatchIndices const& batchIndices,
            BoundIndices const& boundIndices,
            std::vector<size_t> const& indexSizes,
            DataType aType, std::vector<size_t> const& aStrides, TensorOps const& aOps,
            DataType bType, std::vector<size_t> const& bStrides, TensorOps const& bOps,
            DataType cType, std::vector<size_t> const& cStrides, TensorOps const& cOps,
            DataType dType, std::vector<size_t> const& dStrides, TensorOps const& dOps,
            double beta)
    {
        size_t maxA = 0;
        size_t maxB = 0;
        size_t maxC = 0;
        size_t maxD = 0;

        // Determine number of dimension for each tensor.

        for(auto const& free: freeIndices)
        {
            if (free.isA)
                maxA = std::max(maxA, free.i);
            else
                maxB = std::max(maxB, free.i);
            maxC = std::max(maxC, free.c);
            maxD = std::max(maxD, free.d);
        }

        for(auto const& batch: batchIndices)
        {
            maxA = std::max(maxA, batch.a);
            maxB = std::max(maxB, batch.b);
            maxC = std::max(maxC, batch.c);
            maxD = std::max(maxD, batch.d);
        }

        for(auto const& bound: boundIndices)
        {
            maxA = std::max(maxA, bound.a);
            maxB = std::max(maxB, bound.b);
        }

        std::vector<size_t> aSizes(maxA+1), bSizes(maxB+1), cSizes(maxC+1), dSizes(maxD+1);

        for(auto const& free: freeIndices)
        {
            size_t indexSize = indexSizes.at(free.d);
            if (free.isA)
                aSizes[free.i] = indexSize;
            else
                bSizes[free.i] = indexSize;

            cSizes[free.c] = indexSize;
            dSizes[free.d] = indexSize;
        }

        for(auto const& batch: batchIndices)
        {
            size_t indexSize = indexSizes.at(batch.d);

            aSizes[batch.a] = indexSize;
            bSizes[batch.b] = indexSize;
            cSizes[batch.c] = indexSize;
            dSizes[batch.d] = indexSize;
        }

        size_t indexIdx = dSizes.size();
        for(auto const& bound: boundIndices)
        {
            size_t indexSize = indexSizes.at(indexIdx);

            aSizes[bound.a] = indexSize;
            bSizes[bound.b] = indexSize;

            indexIdx++;
        }

        TensorDescriptor a(aType, aSizes.begin(), aSizes.end(), aStrides.begin(), aStrides.end());
        TensorDescriptor b(bType, bSizes.begin(), bSizes.end(), bStrides.begin(), bStrides.end());
        TensorDescriptor c(cType, cSizes.begin(), cSizes.end(), cStrides.begin(), cStrides.end());
        TensorDescriptor d(dType, dSizes.begin(), dSizes.end(), dStrides.begin(), dStrides.end());

        return ContractionProblem(a, aOps, b, bOps, c, cOps, d, dOps,
                                  freeIndices, batchIndices, boundIndices,
                                  beta);
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

    void ContractionProblem::addAZeroPad(const ZeroPad &zp)
    {
        m_aZeroPads.push_back(zp);
        m_boundIndices[toBoundsPos(zp.boundIndex)].aZeroPad = zp;
    }

    void ContractionProblem::addBZeroPad(const ZeroPad &zp)
    {
        m_bZeroPads.push_back(zp);
        m_boundIndices[toBoundsPos(zp.boundIndex)].bZeroPad = zp;
    }

    void ContractionProblem::normalize()
    {
        m_maxProblemSize = 0;

        m_batchSizes.resize(m_batchIndices.size());
        m_boundSizes.resize(m_boundIndices.size());

        for(int i = 0; i < m_freeIndices.size(); i++)
        {
            size_t maxSize=0; // TODO - aren't these all the same?
            if (m_freeIndices[i].isA)
            {
                m_freeIndicesA.push_back(m_freeIndices[i]);
                maxSize = std::max({m_a.sizes()[m_freeIndices[i].i],
                             m_c.empty() ? 0 : m_c.sizes()[m_freeIndices[i].c],
                             m_d.sizes()[m_freeIndices[i].d]});
                m_freeSizesA.push_back(maxSize);
                assert(m_a.sizes()[m_freeIndices[i].i] == m_d.sizes()[m_freeIndices[i].d]);
                assert(maxSize == m_a.sizes()[m_freeIndices[i].i]);
            }
            else
            {
                m_freeIndicesB.push_back(m_freeIndices[i]);
                maxSize = std::max({m_b.sizes()[m_freeIndices[i].i],
                                    m_c.empty() ? 0 : m_c.sizes()[m_freeIndices[i].c],
                                    m_d.sizes()[m_freeIndices[i].d]});
                m_freeSizesB.push_back(maxSize);
                assert(m_b.sizes()[m_freeIndices[i].i] == m_d.sizes()[m_freeIndices[i].d]);
                assert(maxSize == m_b.sizes()[m_freeIndices[i].i]);
            }

            m_maxProblemSize = std::max(m_maxProblemSize, maxSize);
        }

        for(int i = 0; i < m_batchIndices.size(); i++)
        {
            m_batchSizes[i] = std::max({m_a.sizes()[m_batchIndices[i].a],
                                        m_b.sizes()[m_batchIndices[i].b],
                                        m_c.empty() ? 0 : m_c.sizes()[m_batchIndices[i].c],
                                        m_d.sizes()[m_batchIndices[i].d]});
        }

        for(int i = 0; i < m_boundIndices.size(); i++)
        {
            m_boundSizes[i] = std::max(m_a.sizes()[m_boundIndices[i].a],
                                       m_b.sizes()[m_boundIndices[i].b]);

            m_maxProblemSize = std::max(m_maxProblemSize, m_boundSizes[i]);
        }

        for (auto zp : m_aZeroPads)
            m_boundIndices[toBoundsPos(zp.boundIndex)].aZeroPad = zp;
        for (auto zp : m_bZeroPads)
            m_boundIndices[toBoundsPos(zp.boundIndex)].bZeroPad = zp;

        getIndexNames(m_aNames, m_bNames, m_cNames, m_dNames, m_sumNames);

        m_operationIdentifier = getOperationIdentifier();

        m_transA = m_aNames == "lik";
        m_transB = m_bNames == "jlk";

        m_problemSizes.resize(0);
        m_problemSizes.reserve(m_c.dimensions() + m_boundSizes.size());

        m_problemSizes.insert(m_problemSizes.end(), m_c.sizes().begin(), m_c.sizes().end());
        m_problemSizes.insert(m_problemSizes.end(), m_boundSizes.begin(), m_boundSizes.end());

        m_problemStrides.insert(m_problemStrides.end(), m_a.strides().begin(), m_a.strides().end());
        m_problemStrides.insert(m_problemStrides.end(), m_b.strides().begin(), m_b.strides().end());
        m_problemStrides.insert(m_problemStrides.end(), m_c.strides().begin(), m_c.strides().end());
        m_problemStrides.insert(m_problemStrides.end(), m_d.strides().begin(), m_d.strides().end());

        m_allocatedElementsNonBatchA = 1;
        for(int idx = 0; idx < a().dimensions(); idx++)
        {
            bool isBatch =  batchIndices().end() !=
                            std::find_if(batchIndices().begin(), batchIndices().end(),
                            [idx](const ContractionProblem::BatchIndex &bi)
                            {return bi.a == idx;});
            if (!isBatch)
                m_allocatedElementsNonBatchA += a().strides()[idx] * (a().sizes()[idx]-1);
        }

        m_allocatedElementsNonBatchB = 1;
        for(int idx = 0; idx < b().dimensions(); idx++)
        {
            bool isBatch =  batchIndices().end() !=
                            std::find_if(batchIndices().begin(), batchIndices().end(),
                            [idx](const ContractionProblem::BatchIndex &bi)
                            {return bi.b == idx;});
            if (!isBatch)
                m_allocatedElementsNonBatchB += b().strides()[idx] * (b().sizes()[idx]-1);
        }
    }

    void ContractionProblem::consistencyCheck() const
    {
        std::vector<int> aUseCount(m_a.dimensions(), 0);
        std::vector<int> bUseCount(m_b.dimensions(), 0);
        std::vector<int> cUseCount(m_c.dimensions(), 0);
        std::vector<int> dUseCount(m_d.dimensions(), 0);

        for(FreeIndex const& free: m_freeIndices)
        {
            if (free.isA)
            {
                aUseCount[free.i]++;
                TENSILE_ASSERT_EXC(free.i  < m_a.dimensions());
                TENSILE_ASSERT_EXC(m_a.sizes()[free.i] == m_d.sizes()[free.d]);
            }
            else
            {
                bUseCount[free.i]++;
                TENSILE_ASSERT_EXC(free.i  < m_b.dimensions());
                TENSILE_ASSERT_EXC(m_b.sizes()[free.i] == m_d.sizes()[free.d]);
            }

            TENSILE_ASSERT_EXC(free.d < m_d.dimensions());
            dUseCount[free.d]++;

            if(!m_c.empty())
            {
                TENSILE_ASSERT_EXC(free.c < m_c.dimensions());

                cUseCount[free.c]++;

                if (free.isA)
                    TENSILE_ASSERT_EXC(m_a.sizes()[free.i] == m_c.sizes()[free.c]);
                else
                    TENSILE_ASSERT_EXC(m_b.sizes()[free.i] == m_c.sizes()[free.c]);
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

            size_t aSize = m_a.sizes()[batch.a];
            size_t bSize = m_b.sizes()[batch.b];
            size_t cSize = 1;
            size_t dSize = m_d.sizes()[batch.d];

            if(!m_c.empty())
            {
                TENSILE_ASSERT_EXC(batch.c < m_c.dimensions());
                cUseCount[batch.c]++;

                cSize = m_c.sizes()[batch.c];
            }

            size_t indexSize = std::max({aSize, bSize, cSize, dSize});

            TENSILE_ASSERT_EXC(aSize == 1 || aSize == indexSize);
            TENSILE_ASSERT_EXC(bSize == 1 || bSize == indexSize);
            TENSILE_ASSERT_EXC(cSize == 1 || cSize == indexSize);
            TENSILE_ASSERT_EXC(dSize == 1 || dSize == indexSize);
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

        for(auto const& op: m_aOps)
            if(op.type == TensorOp::Type::ComplexConjugate)
                TENSILE_ASSERT_EXC(DataTypeInfo::Get(m_a.dataType()).isComplex);

        for(auto const& op: m_bOps)
            if(op.type == TensorOp::Type::ComplexConjugate)
                TENSILE_ASSERT_EXC(DataTypeInfo::Get(m_b.dataType()).isComplex);

        for(auto const& op: m_cOps)
            if(op.type == TensorOp::Type::ComplexConjugate)
                TENSILE_ASSERT_EXC(DataTypeInfo::Get(m_c.dataType()).isComplex);

        for(auto const& op: m_dOps)
            if(op.type == TensorOp::Type::ComplexConjugate)
                TENSILE_ASSERT_EXC(DataTypeInfo::Get(m_d.dataType()).isComplex);
    }

    size_t ContractionProblem::freeSizeA(size_t idx) const
    {
        return m_freeSizesA.at(idx);
    }

    size_t ContractionProblem::freeSizeB(size_t idx) const
    {
        return m_freeSizesB.at(idx);
    }

    size_t ContractionProblem::batchSize(size_t idx) const
    {
        return m_batchSizes[idx];
    }

    size_t ContractionProblem::boundSize(size_t idx) const
    {
        return m_boundSizes[idx];
    }

    size_t ContractionProblem::size(size_t idx) const
    {
        if (idx < c().sizes().size())
            return c().sizes()[idx];
        else
            return m_boundSizes.at(idx - c().sizes().size());
    }


    size_t ContractionProblem::flopsPerMac() const
    {
        return 2;
    }

    size_t ContractionProblem::flopCount() const
    {
        size_t rv = flopsPerMac();

        for(auto size: m_freeSizesA)
            rv *= size;

        for(auto size: m_freeSizesB)
            rv *= size;

        for(auto size: m_batchSizes)
            rv *= size;

        for(auto size: m_boundSizes)
            rv *= size;

        return rv;
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
            if (free.isA)
                aNames[free.i] = dNames[free.d];
            else
                bNames[free.i] = dNames[free.d];
            if(!m_c.empty())
            {
                cNames[free.c] = dNames[free.d];
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
        for(auto const& op: m_aOps)
            rv << op.suffix();

        rv << "_B" << m_bNames;
        for(auto const& op: m_bOps)
            rv << op.suffix();

        rv << "_C" << m_cNames;
        for(auto const& op: m_cOps)
            rv << op.suffix();

        rv << "_D" << m_dNames;
        for(auto const& op: m_dOps)
            rv << op.suffix();

        return rv.str();
    }

    std::string ContractionProblem::description() const
    {
        std::ostringstream rv;

        rv << operationIdentifier() << ",\n"
           << "A: " << m_a << ",\n"
           << "B: " << m_b << ",\n"
           << "C: " << m_c << ",\n"
           << "D: " << m_d << "\n";

        return rv.str();
    }

    TENSILE_API std::ostream & operator<<(std::ostream & stream, ContractionProblem const& contraction)
    {
        return stream << contraction.description();
    }

    std::ostream & operator<<(std::ostream & stream, ContractionProblem::FreeIndex  const& free)
    {
        return stream << "{isA=" << free.isA << " i=" << free.i
                      << " c=" << free.c << " d=" << free.d << "}";
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

    std::istream & operator>>(std::istream & stream, ContractionProblem::FreeIndex       & free)
    {
        StreamRead comma(",");
        return stream   >> free.isA
               >> comma >> free.i
               >> comma >> free.c
               >> comma >> free.d;
    }

    std::istream & operator>>(std::istream & stream, ContractionProblem::BatchIndex      & batch)
    {
        StreamRead comma(",");
        return stream   >> batch.a
               >> comma >> batch.b
               >> comma >> batch.c
               >> comma >> batch.d;
    }

    std::istream & operator>>(std::istream & stream, ContractionProblem::BoundIndex      & bound)
    {
        StreamRead comma(",");
        return stream >> bound.a >> comma >> bound.b;
    }

    TENSILE_API ProblemInputs::~ProblemInputs() = default;
    ContractionInputs::ContractionInputs() = default;
    ContractionInputs::~ContractionInputs() = default;

    template <typename A, typename B, typename C, typename D, typename Alpha, typename Beta>
    TypedContractionInputs<A, B, C, D, Alpha, Beta>::TypedContractionInputs() = default;

    template <typename A, typename B, typename C, typename D, typename Alpha, typename Beta>
    TypedContractionInputs<A, B, C, D, Alpha, Beta>::~TypedContractionInputs() = default;

    template <typename A, typename B, typename C, typename D, typename Alpha, typename Beta>
    TypedContractionInputs<A, B, C, D, Alpha, Beta>::TypedContractionInputs(
            A const* _a, B const* _b, C const* _c, D * _d,
            Alpha _alpha, Beta _beta)
        : a(_a), b(_b), c(_c), d(_d), alpha(_alpha), beta(_beta)
    {
    }

    template struct TypedContractionInputs<float>;
    template struct TypedContractionInputs<double>;
    template struct TypedContractionInputs<std::complex<float>>;
    template struct TypedContractionInputs<std::complex<double>>;
    template struct TypedContractionInputs<Int8x4, Int8x4, int32_t, int32_t>;
    template struct TypedContractionInputs<int32_t>;

#ifdef TENSILE_USE_HALF
    template struct TypedContractionInputs<Half>;
    template struct TypedContractionInputs<Half, Half, Half, Half, float, float>;
#endif
#ifdef TENSILE_USE_BF16
    template struct TypedContractionInputs<BFloat16, BFloat16, BFloat16, BFloat16, float, float>;
#endif
}

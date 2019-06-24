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

    ContractionProblem ContractionProblem::GEMM(bool transA, bool transB,
                                                TensorDescriptor const& a, TensorOps const& aOps,
                                                TensorDescriptor const& b, TensorOps const& bOps,
                                                TensorDescriptor const& c, TensorOps const& cOps,
                                                TensorDescriptor const& d, TensorOps const& dOps,
                                                double beta)
    {
        FreeIndex free;
        BoundIndex bound;

        free.ca = free.da = 0;
        free.cb = free.db = 1;

        if(transA)
        {
            free.a = 1;
            bound.a = 0;
        }
        else
        {
            free.a = 0;
            bound.a = 1;
        }

        if(transB)
        {
            free.b = 0;
            bound.b = 1;
        }
        else
        {
            free.b = 1;
            bound.b = 0;
        }

        FreeIndices freeIndices{free};
        BatchIndices batchIndices;
        BoundIndices boundIndices{bound};

        batchIndices.push_back({2,2,2,2});

        return ContractionProblem(a, aOps, b, bOps, c, cOps, d, cOps, freeIndices, batchIndices, boundIndices, beta);
    }

    void ContractionProblem::IdentifierToIndices(std::string  const& identifier, 
                                                 FreeIndices       & freeIndices,
                                                 BatchIndices      & batchIndices,
                                                 BoundIndices      & boundIndices)
    {
        FreeIndices  free;
        BatchIndices batch;
        BoundIndices bound;

        std::string prefix = "Contraction_";
        if(identifier.find(prefix) != 0)
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ") must start with '", prefix, "'."));

        size_t begin = prefix.size();
        size_t end = identifier.find("_", begin);

        std::string boundStr = identifier.substr(begin, end-begin);

        begin = end+1;
        end = identifier.find("_", begin);

        if(identifier.at(begin) != 'A')
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        begin++;
        std::string a = identifier.substr(begin, end-begin);

        begin = end+1;
        end = identifier.find("_", begin);

        if(identifier.at(begin) != 'B')
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        begin++;
        std::string b = identifier.substr(begin, end-begin);

        begin = end+1;
        end = identifier.find("_", begin);

        if(identifier.at(begin) != 'C')
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        begin++;
        std::string c = identifier.substr(begin, end-begin);

        begin = end+1;
        end = identifier.find("_", begin);

        if(identifier.at(begin) != 'D')
            throw std::runtime_error(concatenate("Contraction identifier (", identifier, ")must match 'Contraction_s_A_B_C_D'"));

        begin++;
        std::string d = identifier.substr(begin, end-begin);

        std::set<char> allIndices(a.begin(), a.end());
        allIndices.insert(b.begin(), b.end());
        allIndices.insert(c.begin(), c.end());
        allIndices.insert(d.begin(), d.end());

        bool freeHasA = true;
        bool freeHasB = true;

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
                if(freeHasA && freeHasB)
                {
                    free.resize(free.size()+1);
                    freeHasB = false;
                }
                else if(freeHasA)
                {
                    throw std::runtime_error(concatenate("Inconsistent free index pairing: ", identifier));
                }

                free.back().a = aIndex;
                free.back().ca = cIndex;
                free.back().da = dIndex;
                freeHasA = true;
            }
            else if(aIndex == std::string::npos && bIndex != std::string::npos
                 && cIndex != std::string::npos && dIndex != std::string::npos)
            {
                if(freeHasA && freeHasB)
                {
                    free.resize(free.size()+1);
                    freeHasA = false;
                }
                else if(freeHasB)
                {
                    throw std::runtime_error(concatenate("Inconsistent free index pairing: ", identifier));
                }

                free.back().b = bIndex;
                free.back().cb = cIndex;
                free.back().db = dIndex;
                freeHasB = true;
            }
        }
        freeIndices  = std::move(free);
        batchIndices = std::move(batch);
        boundIndices = std::move(bound);
    }

    ContractionProblem ContractionProblem::FromIndexSizes(
            std::string const& operationIdentifier,
            std::vector<size_t> const& indexSizes,
            DataType aType, std::vector<size_t> const& aStrides, TensorOps const& aOps,
            DataType bType, std::vector<size_t> const& bStrides, TensorOps const& bOps,
            DataType cType, std::vector<size_t> const& cStrides, TensorOps const& cOps,
            DataType dType, std::vector<size_t> const& dStrides, TensorOps const& dOps,
            double beta)
    {
        FreeIndices freeIndices;
        BatchIndices batchIndices;
        BoundIndices boundIndices;

        IdentifierToIndices(operationIdentifier, 
                            freeIndices,
                            batchIndices,
                            boundIndices);

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
            maxA = std::max(maxA, free.a);
            maxB = std::max(maxB, free.b);
            maxC = std::max({maxC, free.ca, free.cb});
            maxD = std::max({maxD, free.da, free.db});
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
            size_t indexSizeA = indexSizes.at(free.da);
            size_t indexSizeB = indexSizes.at(free.db);

            aSizes[free.a] = indexSizeA;
            bSizes[free.b] = indexSizeB;

            cSizes[free.ca] = indexSizeA;
            cSizes[free.cb] = indexSizeB;

            dSizes[free.da] = indexSizeA;
            dSizes[free.db] = indexSizeB;
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

    void ContractionProblem::normalize()
    {
        std::sort(m_freeIndices.begin(),  m_freeIndices.end());
        std::sort(m_batchIndices.begin(), m_batchIndices.end());
        std::sort(m_boundIndices.begin(), m_boundIndices.end());

        m_maxProblemSize = 0;

        m_freeSizeA.resize(m_freeIndices.size());
        m_freeSizeB.resize(m_freeIndices.size());
        m_batchSizes.resize(m_batchIndices.size());
        m_boundSizes.resize(m_boundIndices.size());

        for(int i = 0; i < m_freeIndices.size(); i++)
        {
            m_freeSizeA[i] = std::max({m_a.sizes()[m_freeIndices[i].a],
                                       m_c.empty() ? 0 : m_c.sizes()[m_freeIndices[i].ca],
                                       m_d.sizes()[m_freeIndices[i].da]});

            m_freeSizeB[i] = std::max({m_b.sizes()[m_freeIndices[i].b],
                                       m_c.empty() ? 0 : m_c.sizes()[m_freeIndices[i].cb],
                                       m_d.sizes()[m_freeIndices[i].db]});

            m_maxProblemSize = std::max({m_maxProblemSize, m_freeSizeA[i], m_freeSizeB[i]});
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

        getIndexNames(m_aNames, m_bNames, m_cNames, m_dNames, m_sumNames);

        m_operationIdentifier = getOperationIdentifier();

        m_transA = m_aNames == "lik";
        m_transB = m_bNames == "jlk";

        m_problemSizes.resize(0);
        m_problemSizes.reserve(m_c.dimensions() + m_boundSizes.size());

        m_problemSizes.insert(m_problemSizes.end(), m_c.sizes().begin(), m_c.sizes().end());
        m_problemSizes.insert(m_problemSizes.end(), m_boundSizes.begin(), m_boundSizes.end());
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
    }

    size_t ContractionProblem::freeSizeA(size_t idx) const
    {
        return m_freeSizeA[idx];
    }

    size_t ContractionProblem::freeSizeB(size_t idx) const
    {
        return m_freeSizeB[idx];
    }

    size_t ContractionProblem::batchSize(size_t idx) const
    {
        return m_batchSizes[idx];
    }

    size_t ContractionProblem::boundSize(size_t idx) const
    {
        return m_boundSizes[idx];
    }

    size_t ContractionProblem::flopsPerMac() const
    {
        return 2;
    }

    size_t ContractionProblem::flopCount() const
    {
        size_t rv = flopsPerMac();

        for(auto size: m_freeSizeA)
            rv *= size;

        for(auto size: m_freeSizeB)
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

    std::istream & operator>>(std::istream & stream, ContractionProblem::FreeIndex       & free)
    {
        StreamRead comma(",");
        return stream   >> free.a
               >> comma >> free.b
               >> comma >> free.ca
               >> comma >> free.cb
               >> comma >> free.da
               >> comma >> free.db;
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
    template struct TypedContractionInputs<Half>;
    template struct TypedContractionInputs<BFloat16>;
}


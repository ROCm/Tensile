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
#include <Tensile/Utils.hpp>

namespace Tensile
{
    ContractionProblem ContractionProblem::GEMM(bool transA, bool transB,
                                                size_t m, size_t n, size_t k,
                                                size_t lda, size_t ldb, size_t ldc,
                                                bool useBeta, bool colMajor, size_t batchCount)
    {
        if(colMajor) throw std::runtime_error("Column major not yet implemented.");

        FreeIndex free;
        BoundIndex bound;

        free.ca = free.da = 0;
        free.cb = free.db = 1;

        TensorDescriptor a, b, c, d;
        if(transA)
        {
            a = TensorDescriptor(DataType::Float, {k, m}, {lda, m});
            free.a = 1;
            bound.a = 0;
        }
        else
        {
            a = TensorDescriptor(DataType::Float, {m, k}, {lda, k});
            free.a = 0;
            bound.a = 1;
        }

        if(transB)
        {
            b = TensorDescriptor(DataType::Float, {n, k}, {ldb, k});
            free.b = 0;
            bound.b = 1;
        }
        else
        {
            b = TensorDescriptor(DataType::Float, {k, n}, {ldb, n});
            free.b = 1;
            bound.b = 0;
        }

        FreeIndices freeIndices{free};
        BatchIndices batchIndices;
        BoundIndices boundIndices{bound};

        d = TensorDescriptor(DataType::Float, {m, n});
        if(batchCount > 1)
        {
            a.appendDim(batchCount);
            b.appendDim(batchCount);
            d.appendDim(batchCount);

            batchIndices.push_back({2,2,2,2});
        }

        if(useBeta)
            c = d;

        TensorOps nop;

        return ContractionProblem(a, nop, b, nop, c, nop, d, nop, freeIndices, batchIndices, boundIndices);
    }

    ContractionProblem::ContractionProblem(TensorDescriptor const& a, TensorOps const& aOps,
                                           TensorDescriptor const& b, TensorOps const& bOps,
                                           TensorDescriptor const& c, TensorOps const& cOps,
                                           TensorDescriptor const& d, TensorOps const& dOps,
                                           FreeIndices  const& freeIndices,
                                           BatchIndices const& batchIndices,
                                           BoundIndices const& boundIndices)
        : a(a), aOps(aOps),
          b(b), bOps(bOps),
          c(c), cOps(cOps),
          d(d), dOps(dOps),
          freeIndices(freeIndices),
          batchIndices(batchIndices),
          boundIndices(boundIndices)
    {
        normalize();
        consistencyCheck();
    }

    void ContractionProblem::normalize()
    {
    }

    void ContractionProblem::consistencyCheck() const
    {
        std::vector<int> aUseCount(a.dimensions(), 0);
        std::vector<int> bUseCount(b.dimensions(), 0);
        std::vector<int> cUseCount(c.dimensions(), 0);
        std::vector<int> dUseCount(d.dimensions(), 0);

        for(FreeIndex const& free: freeIndices)
        {
            TENSILE_ASSERT_EXC(free.a  < a.dimensions());
            TENSILE_ASSERT_EXC(free.b  < b.dimensions());
            TENSILE_ASSERT_EXC(free.da < d.dimensions());
            TENSILE_ASSERT_EXC(free.db < d.dimensions());

            aUseCount[free.a]++;
            bUseCount[free.b]++;

            dUseCount[free.da]++;
            dUseCount[free.db]++;

            TENSILE_ASSERT_EXC(a.logicalCounts()[free.a] == d.logicalCounts()[free.da]);
            TENSILE_ASSERT_EXC(b.logicalCounts()[free.b] == d.logicalCounts()[free.db]);

            if(!c.empty())
            {
                TENSILE_ASSERT_EXC(free.ca < c.dimensions());
                TENSILE_ASSERT_EXC(free.cb < c.dimensions());

                cUseCount[free.ca]++;
                cUseCount[free.cb]++;

                TENSILE_ASSERT_EXC(a.logicalCounts()[free.a] == c.logicalCounts()[free.ca]);
                TENSILE_ASSERT_EXC(b.logicalCounts()[free.b] == c.logicalCounts()[free.cb]);
            }
        }

        for(BatchIndex const& batch: batchIndices)
        {
            TENSILE_ASSERT_EXC(batch.a < a.dimensions());
            TENSILE_ASSERT_EXC(batch.b < b.dimensions());
            TENSILE_ASSERT_EXC(batch.d < d.dimensions());

            aUseCount[batch.a]++;
            bUseCount[batch.b]++;
            dUseCount[batch.d]++;

            TENSILE_ASSERT_EXC(a.logicalCounts()[batch.a] == b.logicalCounts()[batch.b]);
            TENSILE_ASSERT_EXC(a.logicalCounts()[batch.a] == d.logicalCounts()[batch.b]);

            if(!c.empty())
            {
                TENSILE_ASSERT_EXC(batch.c < c.dimensions());
                cUseCount[batch.c]++;

                TENSILE_ASSERT_EXC(a.logicalCounts()[batch.a] == c.logicalCounts()[batch.b]);
            }
        }

        for(BoundIndex const& bound: boundIndices)
        {
            TENSILE_ASSERT_EXC(bound.a < a.dimensions());
            TENSILE_ASSERT_EXC(bound.b < b.dimensions());

            aUseCount[bound.a]++;
            bUseCount[bound.b]++;

            TENSILE_ASSERT_EXC(a.logicalCounts()[bound.a] == b.logicalCounts()[bound.b]);
        }

        for(int aUse: aUseCount) TENSILE_ASSERT_EXC(aUse == 1);
        for(int bUse: bUseCount) TENSILE_ASSERT_EXC(bUse == 1);
        for(int cUse: cUseCount) TENSILE_ASSERT_EXC(cUse == 1);
        for(int dUse: dUseCount) TENSILE_ASSERT_EXC(dUse == 1);
    }

    size_t ContractionProblem::freeSizeA(size_t idx)
    {
        return a.logicalCounts()[freeIndices[idx].a];
    }

    size_t ContractionProblem::freeSizeB(size_t idx)
    {
        return b.logicalCounts()[freeIndices[idx].b];
    }

    size_t ContractionProblem::batchSize(size_t idx)
    {
        return a.logicalCounts()[batchIndices[idx].a];
    }

    size_t ContractionProblem::boundSize(size_t idx)
    {
        return a.logicalCounts()[boundIndices[idx].a];
    }

    std::string ContractionProblem::operationDescription() const
    {
        std::string aNames(a.dimensions(), '_');
        std::string bNames(b.dimensions(), '_');
        std::string cNames(c.dimensions(), '_');
        std::string dNames(d.dimensions(), '_');
        std::string sumNames(boundIndices.size(), '_');

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

        for(auto const& free: freeIndices)
        {
            aNames[free.a] = dNames[free.da];
            bNames[free.b] = dNames[free.db];
            if(!c.empty())
            {
                cNames[free.ca] = dNames[free.da];
                cNames[free.cb] = dNames[free.db];
            }
        }

        for(auto const& batch: batchIndices)
        {
            aNames[batch.a] = dNames[batch.d];
            bNames[batch.b] = dNames[batch.d];
            if(!c.empty())
                cNames[batch.c] = dNames[batch.d];
        }

        for(size_t i = 0; i < sumNames.size(); i++)
        {
            aNames[boundIndices[i].a] = sumNames[i];
            bNames[boundIndices[i].b] = sumNames[i];
        }

        std::ostringstream rv;

        rv << "D[" << dNames << "] = alpha * (";

        if(!sumNames.empty())
            rv << "Sum[" << sumNames << "] ";

        rv << "A[" << aNames << "] * B[" << bNames << "])";

        if(!c.empty())
            rv << " + beta * C[" << cNames << "]";

        return rv.str();
    }

#if 0
    {
        if(a.dimensions() != b.dimensions() || a.dimensions() != c.dimensions() || a.dimensions() != d.dimensions())
            throw std::runtime_error("Tensors must all have the same number of dimensions.");

        if(a.dimensions() != 3)
            throw std::runtime_error("Only 3- dimensional tensors are accepted.");

        if(c.logicalCounts() != d.logicalCounts())
            throw std::runtime_error("C and D must have the same logical dimensions.");

        // "M"
        if(a.logicalCounts()[0] != d.logicalCounts()[0])
            throw std::runtime_error("A size 0 and C/D size 0 must be equal.");

        // "N"
        if(b.logicalCounts()[1] != d.logicalCounts()[1])
            throw std::runtime_error("B size 1 and C/D size 1 must be equal.");

        // "K"
        if(a.logicalCounts()[1] != b.logicalCounts()[0])
            throw std::runtime_error("A size 1 and B size 0 must be equal.");

        if(a.logicalCounts()[2] != b.logicalCounts()[2])
            throw std::runtime_error("Batch dimensions must be equal. A and B mismatched.");

        if(a.logicalCounts()[2] != d.logicalCounts()[2])
            throw std::runtime_error("Batch dimensions must be equal. A and C/D mismatched.");
    }

    size_t ContractionProblem::tensile_strideA1() const
    {
        return a.storedStride(1);
    }

    size_t ContractionProblem::tensile_strideA2() const
    {
        return a.storedStride(2);
    }

    size_t ContractionProblem::tensile_strideB1() const
    {
        return b.storedStride(1);
    }

    size_t ContractionProblem::tensile_strideB2() const
    {
        return b.storedStride(2);
    }

    size_t ContractionProblem::tensile_strideC1() const
    {
        return c.storedStride(1);
    }

    size_t ContractionProblem::tensile_strideC2() const
    {
        return c.storedStride(2);
    }

    size_t ContractionProblem::tensile_strideD1() const
    {
        return d.storedStride(1);
    }

    size_t ContractionProblem::tensile_strideD2() const
    {
        return d.storedStride(2);
    }

    bool ContractionProblem::blas_transA() const
    {
        return a.dimensionOrder() == std::vector<size_t>{1,0,2};
    }

    bool ContractionProblem::blas_transB() const
    {
        return b.dimensionOrder() == std::vector<size_t>{1,0,2};
    }
#endif

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

}


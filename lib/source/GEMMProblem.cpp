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

#include <Tensile/GEMMProblem.hpp>

namespace Tensile
{
    GEMMProblem GEMMProblem::FromBLAS(bool transA, bool transB,
                                      size_t m, size_t n, size_t k,
                                      size_t lda, size_t ldb, size_t ldc,
                                      bool useBeta, bool colMajor, size_t batchCount)
    {
        if(colMajor) throw std::runtime_error("Column major not yet implemented.");

        TensorDescriptor a, b, c, d;
        if(transA)
        {
            throw std::runtime_error("Nope");
            //a = TensorDescriptor(DataType::Float, {k, m, batchCount}, {lda, m, batchCount});
            //a.transpose(0,1);
        }
        else
        {
            a = TensorDescriptor(DataType::Float, {m, k, batchCount}, {1, lda, k*lda});
        }

        if(transB)
        {
            throw std::runtime_error("Nope");
            //b = TensorDescriptor(DataType::Float, {n, k, batchCount}, {ldb, k, batchCount});
            //b.transpose(0,1);
        }
        else
        {
            b = TensorDescriptor(DataType::Float, {k, n, batchCount}, {1, ldb, n*ldb});
        }

        c = TensorDescriptor(DataType::Float, {m, n, batchCount});
        d = c;

        TensorOps nop;

        return GEMMProblem(a, nop, b, nop, c, nop, d, nop, useBeta);
    }

    GEMMProblem::GEMMProblem(TensorDescriptor const& a, TensorOps const& aOps,
                             TensorDescriptor const& b, TensorOps const& bOps,
                             TensorDescriptor const& c, TensorOps const& cOps,
                             TensorDescriptor const& d, TensorOps const& dOps,
                             bool useBeta)
        : a(a), aOps(aOps),
          b(b), bOps(bOps),
          c(c), cOps(cOps),
          d(d), dOps(dOps),
          useBeta(useBeta)
    {
        normalize();
        consistencyCheck();
    }

    void GEMMProblem::normalize()
    {
    }

    void GEMMProblem::consistencyCheck() const
    {
        if(a.dimensions() != b.dimensions() || a.dimensions() != c.dimensions() || a.dimensions() != d.dimensions())
            throw std::runtime_error("Tensors must all have the same number of dimensions.");

        if(a.dimensions() != 3)
            throw std::runtime_error("Only 3- dimensional tensors are accepted.");

        if(c.sizes() != d.sizes())
            throw std::runtime_error("C and D must have the same logical dimensions.");

        // "M"
        if(a.sizes()[0] != d.sizes()[0])
            throw std::runtime_error("A size 0 and C/D size 0 must be equal.");

        // "N"
        if(b.sizes()[1] != d.sizes()[1])
            throw std::runtime_error("B size 1 and C/D size 1 must be equal.");

        // "K"
        if(a.sizes()[1] != b.sizes()[0])
            throw std::runtime_error("A size 1 and B size 0 must be equal.");

        if(a.sizes()[2] != b.sizes()[2])
            throw std::runtime_error("Batch dimensions must be equal. A and B mismatched.");

        if(a.sizes()[2] != d.sizes()[2])
            throw std::runtime_error("Batch dimensions must be equal. A and C/D mismatched.");
    }

    size_t GEMMProblem::tensile_strideA1() const
    {
        return a.strides()[1];
    }

    size_t GEMMProblem::tensile_strideA2() const
    {
        return a.strides()[2];
    }

    size_t GEMMProblem::tensile_strideB1() const
    {
        return b.strides()[1];
    }

    size_t GEMMProblem::tensile_strideB2() const
    {
        return b.strides()[2];
    }

    size_t GEMMProblem::tensile_strideC1() const
    {
        return c.strides()[1];
    }

    size_t GEMMProblem::tensile_strideC2() const
    {
        return c.strides()[2];
    }

    size_t GEMMProblem::tensile_strideD1() const
    {
        return d.strides()[1];
    }

    size_t GEMMProblem::tensile_strideD2() const
    {
        return d.strides()[2];
    }

    bool GEMMProblem::blas_transA() const
    {
        return false;
        //return a.dimensionOrder() == std::vector<size_t>{1,0,2};
    }

    bool GEMMProblem::blas_transB() const
    {
        return false;
        //return b.dimensionOrder() == std::vector<size_t>{1,0,2};
    }

}


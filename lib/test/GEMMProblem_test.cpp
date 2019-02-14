/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2019 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>

#include <Tensile/GEMMProblem.hpp>

using namespace Tensile;

TEST(GEMMProblem, Simple)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {1534, 2147, 28});
    TensorDescriptor b(DataType::Float, {2147, 3481, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});

    GEMMProblem p(a, noOps, b, noOps, c, noOps, c, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), false);
    EXPECT_EQ(p.blas_transB(), false);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 1534);
    EXPECT_EQ(p.tensile_strideA2(), 1534*2147);

    EXPECT_EQ(p.tensile_strideB1(), 2147);
    EXPECT_EQ(p.tensile_strideB2(), 2147*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1534);
    EXPECT_EQ(p.tensile_strideC2(), 1534*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1534);
    EXPECT_EQ(p.tensile_strideD2(), 1534*3481);
}

TEST(GEMMProblem, TransposeA)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {2147, 1534, 28});
    TensorDescriptor b(DataType::Float, {2147, 3481, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});

    a.transpose(0,1);

    GEMMProblem p(a, noOps, b, noOps, c, noOps, c, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), true);
    EXPECT_EQ(p.blas_transB(), false);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 2147);
    EXPECT_EQ(p.tensile_strideA2(), 1534*2147);

    EXPECT_EQ(p.tensile_strideB1(), 2147);
    EXPECT_EQ(p.tensile_strideB2(), 2147*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1534);
    EXPECT_EQ(p.tensile_strideC2(), 1534*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1534);
    EXPECT_EQ(p.tensile_strideD2(), 1534*3481);
}

TEST(GEMMProblem, TransposeB)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {1534, 2147, 28});
    TensorDescriptor b(DataType::Float, {3481, 2147, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});

    b.transpose(0,1);

    GEMMProblem p(a, noOps, b, noOps, c, noOps, c, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), false);
    EXPECT_EQ(p.blas_transB(), true);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 1534);
    EXPECT_EQ(p.tensile_strideA2(), 1534*2147);

    EXPECT_EQ(p.tensile_strideB1(), 3481);
    EXPECT_EQ(p.tensile_strideB2(), 2147*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1534);
    EXPECT_EQ(p.tensile_strideC2(), 1534*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1534);
    EXPECT_EQ(p.tensile_strideD2(), 1534*3481);
}

TEST(GEMMProblem, TransposeAB)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {2147, 1534, 28});
    TensorDescriptor b(DataType::Float, {3481, 2147, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});

    a.transpose(0,1);
    b.transpose(0,1);

    GEMMProblem p(a, noOps, b, noOps, c, noOps, c, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), true);
    EXPECT_EQ(p.blas_transB(), true);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 2147);
    EXPECT_EQ(p.tensile_strideA2(), 1534*2147);

    EXPECT_EQ(p.tensile_strideB1(), 3481);
    EXPECT_EQ(p.tensile_strideB2(), 2147*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1534);
    EXPECT_EQ(p.tensile_strideC2(), 1534*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1534);
    EXPECT_EQ(p.tensile_strideD2(), 1534*3481);
}

TEST(GEMMProblem, Padding)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {1534, 2147, 28}, {1536, 2147, 28});
    TensorDescriptor b(DataType::Float, {2147, 3481, 28}, {2176, 3481, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28}, {1536, 3481, 28});
    TensorDescriptor d(DataType::Float, {1534, 3481, 28}, {1568, 3481, 28});

    GEMMProblem p(a, noOps, b, noOps, c, noOps, d, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), false);
    EXPECT_EQ(p.blas_transB(), false);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 1536);
    EXPECT_EQ(p.tensile_strideA2(), 1536*2147);

    EXPECT_EQ(p.tensile_strideB1(), 2176);
    EXPECT_EQ(p.tensile_strideB2(), 2176*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1536);
    EXPECT_EQ(p.tensile_strideC2(), 1536*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1568);
    EXPECT_EQ(p.tensile_strideD2(), 1568*3481);
}

TEST(GEMMProblem, Bad)
{
    TensorOps noOps;

    TensorDescriptor a(DataType::Float, {2147, 1534, 28});
    TensorDescriptor b(DataType::Float, {3481, 2147, 28});
    TensorDescriptor c(DataType::Float, {1534, 3481, 28});
    
    TensorDescriptor b_batch(DataType::Float, {3481, 2147, 12});

    EXPECT_THROW(GEMMProblem(a, noOps, a, noOps, c, noOps, c, noOps, false), std::runtime_error);
    EXPECT_THROW(GEMMProblem(a, noOps, b, noOps, c, noOps, a, noOps, false), std::runtime_error);
    EXPECT_THROW(GEMMProblem(a, noOps, b, noOps, a, noOps, c, noOps, false), std::runtime_error);

    EXPECT_THROW(GEMMProblem(a, noOps, b_batch, noOps, c, noOps, c, noOps, false), std::runtime_error);

    a.transpose(0,1);
    b.transpose(0,1);

    GEMMProblem p(a, noOps, b, noOps, c, noOps, c, noOps, false);

    EXPECT_EQ(p.blas_m(),          1534);
    EXPECT_EQ(p.blas_n(),          3481);
    EXPECT_EQ(p.blas_k(),          2147);
    EXPECT_EQ(p.blas_batchCount(),   28);

    EXPECT_EQ(p.blas_transA(), true);
    EXPECT_EQ(p.blas_transB(), true);

    EXPECT_EQ(p.tensile_I(), 1534);
    EXPECT_EQ(p.tensile_J(), 3481);
    EXPECT_EQ(p.tensile_K(),   28);
    EXPECT_EQ(p.tensile_L(), 2147);

    EXPECT_EQ(p.tensile_strideA1(), 2147);
    EXPECT_EQ(p.tensile_strideA2(), 1534*2147);

    EXPECT_EQ(p.tensile_strideB1(), 3481);
    EXPECT_EQ(p.tensile_strideB2(), 2147*3481);

    EXPECT_EQ(p.tensile_strideC1(), 1534);
    EXPECT_EQ(p.tensile_strideC2(), 1534*3481);

    EXPECT_EQ(p.tensile_strideD1(), 1534);
    EXPECT_EQ(p.tensile_strideD2(), 1534*3481);
}


/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <Tensile/TensorDescriptor.hpp>

using namespace Tensile;

TEST(TensorDescriptor, Simple)
{
    TensorDescriptor t(DataType::Float, {11,13,17});

    EXPECT_EQ(t.dimensions(), 3);
    EXPECT_EQ(t.sizes(),   std::vector<size_t>({11,13,17}));
    EXPECT_EQ(t.strides(), std::vector<size_t>({1,11,11*13}));

    EXPECT_EQ(t.totalLogicalElements(),   11*13*17);
    EXPECT_EQ(t.totalAllocatedElements(), 11*13*17);
    EXPECT_EQ(t.totalAllocatedBytes(),    11*13*17*4);

    EXPECT_EQ(t.index(3,4,1), 3 + 4*11 + 11*13);
}

TEST(TensorDescriptor, Padded)
{
    TensorDescriptor t(DataType::Float, {11,13,17,4}, {1, 16, 16*13, 16*13*17});

    EXPECT_EQ(t.dimensions(), 4);
    EXPECT_EQ(t.sizes(),   std::vector<size_t>({11,13,17,4}));
    EXPECT_EQ(t.strides(), std::vector<size_t>({1,16,16*13,16*13*17}));

    EXPECT_EQ(t.totalLogicalElements(),   11*13*17*4);
    EXPECT_EQ(t.totalAllocatedElements(), 16*13*17*4);
    EXPECT_EQ(t.totalAllocatedBytes(),    16*13*17*4*4);

    EXPECT_EQ(t.index(3,4,1,2), 3 + 4*16 + 16*13 + 16*13*17*2);

}


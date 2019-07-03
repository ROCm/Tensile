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

#pragma once

#include <random>

namespace Tensile
{
    template <typename T, typename RNG=std::mt19937>
    struct RandomInt
    {
        RandomInt(RNG & _rng)
            : rng(_rng)
        {
        }

        RNG & rng;
        std::uniform_int_distribution<int> dist = std::uniform_int_distribution<int>(1,10);
        template <typename... Args>
        T operator()(Args &&...)
        {
            return dist(rng);
        }
    };

    template <typename T, typename RNG=std::mt19937>
    struct RandomAlternatingInt
    {
        RandomInt<T, RNG> parent;

        RandomAlternatingInt(RNG & _rng)
            : parent(_rng)
        {
        }

        T operator()(std::vector<size_t> const& index3)
        {
            T sign = ((index3[0] % 2) ^ (index3[1] % 2)) ? 1 : -1;
            return sign * parent();
        }
    };

    template <typename T>
    struct Iota
    {
        int value = 0;
        int inc = 1;

        Iota() = default;
        Iota(int initial) : value(initial) {}
        Iota(int initial, int increment) : value(initial), inc(increment) {}

        template <typename... Args>
        T operator()(Args &&...)
        {
            T rv = value;
            value += inc;
            return rv;
        }
    };

    template <typename T, typename Generator>
    void InitTensor(T * data, TensorDescriptor const& desc, Generator g)
    {
        if(desc.dimensions() != 3)
            throw std::runtime_error("Fix this function to work with dimensions != 3");

        std::vector<size_t> index3{0,0,0};

        for(index3[2] = 0; index3[2] < desc.sizes()[2]; index3[2]++)
        {
            for(index3[1] = 0; index3[1] < desc.sizes()[1]; index3[1]++)
            {
                index3[0] = 0;
                size_t baseIdx = desc.index(index3);

                for(; index3[0] < desc.sizes()[0]; index3[0]++)
                    data[baseIdx + index3[0]] = g(index3);
            }
        }
    }

    template <typename T>
    void CopyTensor(T * dst, T const* src, TensorDescriptor const& dstDesc, TensorDescriptor const& srcDesc)
    {
        if(dstDesc.dimensions() != 3 || srcDesc.dimensions() != 3)
            throw std::runtime_error("Fix this function to work with dimensions != 3");

        if(dstDesc.sizes() != srcDesc.sizes())
            throw std::runtime_error("Sizes must be equal!");

        size_t bytes = dstDesc.sizes()[0] * sizeof(T);

        for(int k = 0; k < dstDesc.sizes()[2]; k++)
        for(int j = 0; j < dstDesc.sizes()[1]; j++)
        {
            T      * dst_col = dst + dstDesc.index(0, j, k);
            T const* src_col = src + srcDesc.index(0, j, k);

            memcpy(dst_col, src_col, bytes);
        }
    }

    inline ContractionProblem RandomGEMM()
    {
        static std::mt19937 rng;

        std::uniform_int_distribution<int> random_bool(0,1);
        //std::uniform_int_distribution<int> random_size(2,8192);
        std::uniform_int_distribution<int> random_padding(0,32);
        std::uniform_int_distribution<int> random_batch(1,10);
        std::uniform_int_distribution<int> random_beta(0,2);

        std::uniform_real_distribution<double> random_size(1.0, std::log(8192.0));

        bool transA = random_bool(rng);
        bool transB = random_bool(rng);

        size_t m = std::exp(random_size(rng)) + 1;
        size_t n = std::exp(random_size(rng)) + 1;
        size_t k = std::exp(random_size(rng)) + 1;

        int beta_category = random_beta(rng);
        double beta;
        if(beta_category == 0)
            beta = 0.0;
        else if(beta_category == 1)
            beta = 1.0;
        else
            beta = 1.2;

        auto random_pad = [&](size_t cols, size_t rows, size_t &ld, size_t & stride)
        {
            ld = cols;

            bool pad_ld = random_bool(rng);

            if(pad_ld)
            {
                size_t padding = random_padding(rng);
                if(padding == 0)
                    ld = RoundUpToMultiple<size_t>(ld, 128);
                else
                    ld += padding;
            }

            stride = ld * rows;

            bool pad_stride = random_bool(rng);

            if(pad_stride)
            {
                size_t padding = random_padding(rng);

                if(padding == 0)
                    stride = RoundUpToMultiple<size_t>(stride, 256);
                else
                    stride += padding;
            }
        };

        size_t lda, ldb, ldc, ldd;
        size_t strideA, strideB, strideC, strideD;

        if(transA)
            random_pad(k, m, lda, strideA);
        else
            random_pad(m, k, lda, strideA);

        if(transB)
            random_pad(n, k, ldb, strideB);
        else
            random_pad(k, n, ldb, strideB);

        random_pad(m, n, ldc, strideC);

        // ldd support not yet merged in.
        ldd = ldc;
        strideD = strideC;
        //random_pad(m, n, ldd, strideD);

        size_t batchCount = random_batch(rng);

        return ContractionProblem::GEMM_Strides(transA, transB,
                                                DataType::Float, DataType::Float, DataType::Float, DataType::Float,
                                                m, n, k, batchCount,
                                                lda, strideA,
                                                ldb, strideB,
                                                ldc, strideC,
                                                ldd, strideD,
                                                beta);
    }
}

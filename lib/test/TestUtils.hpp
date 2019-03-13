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
    template <typename T>
    struct RandomInt
    {
        std::mt19937 rng;
        std::uniform_int_distribution<int> dist = std::uniform_int_distribution<int>(1,10);
        template <typename... Args>
        T operator()(Args &&...)
        {
            return dist(rng);
        }
    };

    template <typename T>
    struct RandomAlternatingInt
    {
        RandomInt<T> parent;

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

    inline ContractionProblem RandomGEMM()
    {
        static std::mt19937 rng;

        std::uniform_int_distribution<int> random_bool(0,1);
        //std::uniform_int_distribution<int> random_size(2,8192);
        std::uniform_int_distribution<int> random_padding(0,32);
        std::uniform_int_distribution<int> random_batch(1,10);

        std::uniform_real_distribution<double> random_size(1.0, std::log(8192.0));

        bool transA = random_bool(rng);
        bool transB = random_bool(rng);

        size_t m = std::exp(random_size(rng)) + 1;
        size_t n = std::exp(random_size(rng)) + 1;
        size_t k = std::exp(random_size(rng)) + 1;

        bool padA = random_bool(rng);
        bool padB = random_bool(rng);
        bool padC = random_bool(rng);

        size_t lda = transA ? k : m;
        size_t ldb = transB ? n : k;
        size_t ldc = m;

        if(padA)
        {
            size_t aPadding = random_padding(rng);
            if(aPadding == 0)
                lda = RoundUpToMultiple<size_t>(lda, 128);
            else
                lda += aPadding;
        }

        if(padB)
        {
            size_t bPadding = random_padding(rng);
            if(bPadding == 0)
                ldb = RoundUpToMultiple<size_t>(ldb, 128);
            else
                ldb += bPadding;
        }

        if(padC)
        {
            size_t cPadding = random_padding(rng);
            if(cPadding == 0)
                ldc = RoundUpToMultiple<size_t>(ldc, 128);
            else
                ldc += cPadding;
        }

        size_t batchCount = random_batch(rng);

        //std::cout << "ContractionProblem::GEMM(" << transA << ", " << transB << ", "
        //                                         << m << ", " << n << ", " << k << ", "
        //                                         << lda << ", " << ldb << ", " << ldc << ", "
        //                                         << 1.2 << ", " << false << ", " << batchCount << ");" << std::endl;

        return ContractionProblem::GEMM(transA, transB,
                                        m, n, k,
                                        lda, ldb, ldc,
                                        1.2, false, batchCount);
    }
}

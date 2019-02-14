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

    template <typename T, typename Generator>
    void InitTensor(T * data, TensorDescriptor const& desc, Generator g)
    {
        if(desc.dimensions() != 3)
            throw std::runtime_error("Fix this function to work with dimensions != 3");

        if(desc.dimensionOrder() != std::vector<size_t>{0,1,2})
            throw std::runtime_error("Fix this function to work with transposed tensors.");

        std::vector<size_t> index3{0,0,0};

        for(index3[2] = 0; index3[2] < desc.logicalCounts()[2]; index3[2]++)
        {
            for(index3[1] = 0; index3[1] < desc.logicalCounts()[1]; index3[1]++)
            {
                index3[0] = 0;
                size_t baseIdx = desc.index(index3);

                for(; index3[0] < desc.logicalCounts()[0]; index3[0]++)
                    data[baseIdx + index3[0]] = g(index3);
            }
        }
    }

}

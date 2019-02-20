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

#include <iostream>

#include <Tensile/TensorDescriptor.hpp>

namespace Tensile
{

    template <typename T>
    void WriteTensor(std::ostream & stream, T * data, TensorDescriptor const& desc)
    {
        if(desc.dimensions() != 3)
            throw std::runtime_error("Fix this function to work with dimensions != 3");

        bool transpose = desc.dimensionOrder() != std::vector<size_t>{0,1,2};

        std::vector<size_t> index3{0,0,0};

        const size_t d0 = desc.dimensionOrder()[0];
        const size_t d1 = desc.dimensionOrder()[1];
        const size_t d2 = desc.dimensionOrder()[2];

        stream << "Tensor("
            << desc.allocatedCounts()[0] << ", "
            << desc.allocatedCounts()[1] << ", "
            << desc.allocatedCounts()[2] << ")";
        if(transpose)
            stream << " dimensionOrder(" << d0 << ", " << d1 << ", " << d2 << ")";

       stream << std::endl;

        for(index3[d2] = 0; index3[d2] < desc.logicalCounts()[d2]; index3[d2]++)
        {
            stream << "[" << std::endl;
            for(index3[d0] = 0; index3[d0] < desc.logicalCounts()[d0]; index3[d0]++)
            {
                for(index3[d1] = 0; index3[d1] < desc.logicalCounts()[d1]; index3[d1]++)
                {
                    size_t idx = desc.index(index3);
                    stream << data[idx] << " ";
                }
                stream << std::endl;
            }
            stream << "]" << std::endl;
        }
    }

}

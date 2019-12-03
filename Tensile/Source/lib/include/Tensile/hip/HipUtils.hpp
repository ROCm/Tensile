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

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <stdexcept>
#include <sstream>

#include <Tensile/Utils.hpp>
#include <Tensile/TensorDescriptor.hpp>

#define HIP_CHECK_EXC(expr) \
    do \
    { \
        hipError_t e = (expr); \
        if(e) \
        { \
            const char * errName = hipGetErrorName(e); \
            const char * errMsg = hipGetErrorString(e); \
            std::ostringstream msg; \
            msg << "Error " << e << "(" << errName << ") " \
                          << __FILE__ << ":" << __LINE__ << ": " << std::endl \
                      << #expr << std::endl \
                      << errMsg << std::endl; \
            throw std::runtime_error(msg.str()); \
        } \
    } while(0)

#define HIP_CHECK_EXC_MESSAGE(expr, message) \
    do \
    { \
        hipError_t e = (expr); \
        if(e) \
        { \
            const char * errName = hipGetErrorName(e); \
            const char * errMsg = hipGetErrorString(e); \
            std::ostringstream msg; \
            msg << "Error " << e << "(" << errName << ") " \
                          << __FILE__ << ":" << __LINE__ << ": " << std::endl \
                      << #expr << std::endl \
                      << errMsg << std::endl  \
                      << (message) << std::endl; \
            throw std::runtime_error(msg.str()); \
        } \
    } while(0)

namespace Tensile
{
    namespace hip
    {
        template <typename T>
        void CopyTensor(T * dst, T const* src, TensorDescriptor const& desc, hipMemcpyKind direction, hipStream_t stream = 0)
        {
            if(desc.dimensions() == 0 || desc.totalLogicalElements() == 0)
                return;

            auto const& sizes = desc.sizes();
            auto const& strides = desc.strides();
            std::vector<size_t> coord(desc.dimensions(), 0);

            size_t contiguousDimensions = 0;
            size_t expectedStride = 1;

            // Optimize the number of copy operations by coalescing all the
            // dimensions that are contiguous in memory.
            for(size_t i = 0; i < desc.dimensions(); i++)
            {
                contiguousDimensions = i+1;

                if(strides[i] > expectedStride)
                    break;

                if(i < desc.dimensions()-1)
                    expectedStride = strides[i] * sizes[i+1];
            }

            auto copyCount = CoordCount(sizes.begin() + contiguousDimensions, sizes.end());

            size_t maxStride = *std::max_element(strides.begin(), strides.begin() + contiguousDimensions);
            size_t copyBytes = maxStride * sizes.at(contiguousDimensions-1) * sizeof(T);

            for(size_t idx = 0; idx < copyCount; idx++)
            {
                CoordNumbered(idx, coord.begin() + contiguousDimensions, coord.end(),
                              sizes.begin() + contiguousDimensions, sizes.end());

                auto beginOffset = desc.index(coord);

                HIP_CHECK_EXC(hipMemcpyAsync(dst + beginOffset,
                                             src + beginOffset,
                                             copyBytes,
                                             direction,
                                             stream));

            }
        }
    }
}


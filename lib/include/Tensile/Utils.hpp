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

        std::vector<size_t> index3{0,0,0};

        stream << "Tensor("
            << desc.sizes()[0] << ", "
            << desc.sizes()[1] << ", "
            << desc.sizes()[2] << ")";

       stream << std::endl;

        for(index3[2] = 0; index3[2] < desc.sizes()[2]; index3[2]++)
        {
            stream << "[" << std::endl;
            for(index3[0] = 0; index3[0] < desc.sizes()[0]; index3[0]++)
            {
                for(index3[1] = 0; index3[1] < desc.sizes()[1]; index3[1]++)
                {
                    size_t idx = desc.index(index3);
                    stream << data[idx] << " ";
                }
                stream << std::endl;
            }
            stream << "]" << std::endl;
        }
    }

    inline int LexicographicCompare()
    {
        return 0;
    }

    template<typename A>
    inline int LexicographicCompare(A const& lhs, A const& rhs)
    {
        if(lhs < rhs) return -1;
        if(lhs > rhs) return  1;
        return 0;
    }

    template<typename A, typename... Args>
    inline int LexicographicCompare(A const& lhs, A const& rhs, Args const&... rest)
    {
        if(lhs < rhs) return -1;
        if(lhs > rhs) return  1;
        return LexicographicCompare(rest...);
    }

    template <typename T>
    struct Comparison
    {
        enum { implemented = false };
    };

    template <typename T, typename = typename std::enable_if<Comparison<T>::implemented>::type>
    inline bool operator==(T const& lhs, T const& rhs)
    {
        return Comparison<T>::compare(lhs, rhs) == 0;
    }

    template <typename T, typename = typename std::enable_if<Comparison<T>::implemented>::type>
    inline bool operator!=(T const& lhs, T const& rhs)
    {
        return Comparison<T>::compare(lhs, rhs) != 0;
    }

    template <typename T, typename = typename std::enable_if<Comparison<T>::implemented>::type>
    inline bool operator<(T const& lhs, T const& rhs)
    {
        return Comparison<T>::compare(lhs, rhs) < 0; 
    }

    template <typename T, typename = typename std::enable_if<Comparison<T>::implemented>::type>
    inline bool operator<=(T const& lhs, T const& rhs)
    {
        return Comparison<T>::compare(lhs, rhs) <= 0; 
    }

    template <typename T, typename = typename std::enable_if<Comparison<T>::implemented>::type>
    inline bool operator>(T const& lhs, T const& rhs)
    {
        return Comparison<T>::compare(lhs, rhs) > 0; 
    }

    template <typename T, typename = typename std::enable_if<Comparison<T>::implemented>::type>
    inline bool operator>=(T const& lhs, T const& rhs)
    {
        return Comparison<T>::compare(lhs, rhs) >= 0; 
    }

}

#define TENSILE_STR_(x) #x
#define TENSILE_STR(x) TENSILE_STR_(x)
#define TENSILE_LINENO TENSILE_STR(__LINE__)
#define TENSILE_LINEINFO __FILE__ ":" TENSILE_LINENO

#define TENSILE_ASSERT_EXC(exp)                                             \
    do                                                                      \
    {                                                                       \
        if(!(exp))                                                          \
        {                                                                   \
            throw std::runtime_error("Error in " TENSILE_LINEINFO ": " #exp);  \
        }                                                                   \
    } while(false)


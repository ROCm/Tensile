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

    template <typename T, typename U = T>
    struct Comparison
    {
        enum { implemented = false };
    };

    template <typename T, typename U, typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator==(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) == 0;
    }

    template <typename T, typename U, typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator!=(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) != 0;
    }

    template <typename T, typename U, typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator<(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) < 0; 
    }

    template <typename T, typename U, typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator<=(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) <= 0; 
    }

    template <typename T, typename U, typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator>(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) > 0; 
    }

    template <typename T, typename U, typename = typename std::enable_if<Comparison<T, U>::implemented>::type>
    inline bool operator>=(T const& lhs, U const& rhs)
    {
        return Comparison<T, U>::compare(lhs, rhs) >= 0; 
    }
}


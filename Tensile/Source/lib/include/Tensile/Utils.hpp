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

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <type_traits>

namespace Tensile
{

    /**
     * \ingroup Tensile
     * \addtogroup Utilities
     * @{
     */

    template <typename T>
    T CeilDivide(T num, T den)
    {
        return (num + (den-1))/den;
    }

    template <typename T>
    T RoundUpToMultiple(T val, T den)
    {
        return CeilDivide(val, den) * den;
    }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>>
    T IsPrime(T val)
    {
        if(val < 2)
            return false;
        if(val < 4)
            return true;

        T end = sqrt(val);

        for(T i = 2; i <= end; i++)
            if(val % i == 0)
                return false;
        return true;
    }

    template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>>
    T NextPrime(T val)
    {
        if(val < 2)
            return 2;
        while(!IsPrime(val)) val++;
        return val;
    }

    template <typename Container, typename Joiner>
    void streamJoin(std::ostream & stream, Container const& c, Joiner const& j)
    {
        bool first = true;
        for(auto const& item: c)
        {
            if(!first) stream << j;
            stream << item;
            first = false;
        }
    }

    template <typename T>
    inline std::ostream & stream_write(std::ostream & stream, T const& val)
    {
        return stream << val;
    }

    template <typename T, typename... Ts>
    inline std::ostream & stream_write(std::ostream & stream, T const& val, Ts const&... vals)
    {
        return stream_write(stream << val, vals...);
    }

    template <typename... Ts>
    inline std::string concatenate(Ts const&... vals)
    {
        std::ostringstream msg;
        stream_write(msg, vals...);

        return msg.str();
    }

    template <typename T, size_t N>
    inline std::ostream & operator<<(std::ostream & stream, std::array<T, N> const& array)
    {
        streamJoin(stream, array, ", ");
        return stream;
    }

    class StreamRead
    {
    public:
        StreamRead(std::string const& value, bool except=true);
        ~StreamRead();

        bool read(std::istream & stream);

    private:
        std::string const& m_value;
        bool m_except;
        bool m_success = false;
    };

    //inline std::istream & operator>>(std::istream & stream, StreamRead & value);
    inline std::istream & operator>>(std::istream & stream, StreamRead & value)
    {
        value.read(stream);
        return stream;
    }

    /**
     * @}
     */
}

/*
 * \addtogroup Utilities
 * @{
 */
#define TENSILE_STR_(x) #x
#define TENSILE_STR(x) TENSILE_STR_(x)
#define TENSILE_LINENO TENSILE_STR(__LINE__)
#define TENSILE_LINEINFO __FILE__ ":" TENSILE_LINENO

#define TENSILE_ASSERT_EXC(exp)                                               \
    do                                                                        \
    {                                                                         \
        if(!(exp))                                                            \
        {                                                                     \
            throw std::runtime_error("Error in " TENSILE_LINEINFO ": " #exp); \
        }                                                                     \
    } while(false)

/**
 * @}
 */
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

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Tensile/Comparison.hpp>

namespace Tensile
{

    enum class DataType: int
    {
        Half,
        Float,
        Int32,
        Int8,
        Count
    };

    inline size_t TypeSize(DataType d)
    {
        switch(d)
        {
            case DataType::Int32:
            case DataType::Float: return 4;
            case DataType::Half: return 2;
            case DataType::Int8: return 1;

            case DataType::Count:
                throw std::runtime_error("Unknown data type");
        }
        throw std::runtime_error("Unknown data type");
    }

    template <typename T>
    DataType GetDataType();

    template <>
    inline DataType GetDataType<float>() { return DataType::Float; }

    std::string ToString(DataType d);
    std::ostream& operator<<(std::ostream& stream, DataType const& t);
    std::istream& operator>>(std::istream& stream, DataType      & t);

    template <typename T, DataType D>
    struct DistinctType
    {
        using Value = T;

        DistinctType() = default;
        DistinctType(DistinctType const& other) = default;

        DistinctType(T const& v) : value(v) { }

        DistinctType & operator=(DistinctType const& other) = default;
        DistinctType & operator=(T const& other)
        {
            value = other;
            return *this;
        }

        operator const T &() const { return value; }

        T value;
    };

    template <typename T, DataType D>
    struct Comparison<DistinctType<T, D>>
    {
        enum { implemented = true };

        static int compare(DistinctType<T, D> const& lhs, DistinctType<T, D> const& rhs)
        {
            return LexicographicCompare(lhs.value, rhs.value);
        }
    };

    template <typename T, DataType D>
    struct Comparison<DistinctType<T, D>, T>
    {
        enum { implemented = true };

        static int compare(DistinctType<T, D> const& lhs, T const& rhs)
        {
            return LexicographicCompare(lhs.value, rhs);
        }
    };

    using Int8 = DistinctType<uint32_t, DataType::Int8>;

    template <typename T>
    struct TypeInfo
    { };

    template <>
    struct TypeInfo<float>
    {
        const static DataType Enum = DataType::Float;

        const static size_t ElementSize = 4;

        static inline size_t dataBytes(size_t elements)
        {
            return elements * ElementSize;
        }
    };
}


/**
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <iostream>

namespace Tensile
{
    template <typename T>
    struct vector2
    {
        enum { count = 2 };

        T x;
        T y;
    };

    template <typename T>
    inline bool operator==(vector2<T> const& l, vector2<T> const& r)
    {
        return (l.x == r.x) && (l.y == r.y);
    }

    template <typename T>
    inline std::ostream & operator<<(std::ostream & stream, vector2<T> const& v)
    {
        return stream << "(" << v.x << ", " << v.y << ")";
    }

    template <typename T>
    struct vector3
    {
        enum { count = 3 };

        T x;
        T y;
        T z;
    };

    template <typename T>
    inline bool operator==(vector3<T> const& l, vector3<T> const& r)
    {
        return (l.x == r.x) && (l.y == r.y) && (l.z == r.z);
    }

    template <typename T>
    inline std::ostream & operator<<(std::ostream & stream, vector3<T> const& v)
    {
        return stream << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    }

    template <typename T>
    struct vector4
    {
        enum { count = 4 };

        T x;
        T y;
        T z;
        T w;
    };

    template <typename T>
    inline bool operator==(vector4<T> const& l, vector4<T> const& r)
    {
        return (l.x == r.x) && (l.y == r.y) && (l.z == r.z) && (l.w == r.w);
    }

    template <typename T>
    inline std::ostream & operator<<(std::ostream & stream, vector4<T> const& v)
    {
        return stream << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
    }

    using dim3 = vector3<size_t>;
    using int3 = vector3<int>;

    template <typename T>
    T CeilDivide(T num, T den)
    {
        return (num + (den-1))/den;
    }
}


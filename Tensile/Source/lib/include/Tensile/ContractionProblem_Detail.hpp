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

#include <Tensile/Comparison.hpp>

#include <Tensile/ContractionProblem.hpp>

namespace Tensile
{
    template <>
    struct Comparison<ContractionProblem::FreeIndex>
    {
        enum { implemented = true };

        static int compare(ContractionProblem::FreeIndex const& lhs, ContractionProblem::FreeIndex const& rhs)
        {
            return LexicographicCompare(lhs.da, rhs.da,
                                        lhs.db, rhs.db,
                                        lhs.ca, rhs.ca,
                                        lhs.cb, rhs.cb,
                                        lhs.a,  rhs.a,
                                        lhs.b,  rhs.b);
        }
    };

    template <>
    struct Comparison<ContractionProblem::BatchIndex>
    {
        enum { implemented = true };

        static int compare(ContractionProblem::BatchIndex const& lhs, ContractionProblem::BatchIndex const& rhs)
        {
            return LexicographicCompare(lhs.d, rhs.d,
                                        lhs.c, rhs.c,
                                        lhs.a, rhs.a,
                                        lhs.b, rhs.b);
        }
    };

    template <>
    struct Comparison<ContractionProblem::BoundIndex>
    {
        enum { implemented = true };

        static int compare(ContractionProblem::BoundIndex const& lhs, ContractionProblem::BoundIndex const& rhs)
        {
            return LexicographicCompare(lhs.a, rhs.a,
                                        lhs.b, rhs.b);
        }
    };
}

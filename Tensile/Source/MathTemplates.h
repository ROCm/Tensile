/*******************************************************************************
 * Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *******************************************************************************/

#ifndef MATH_TEMPLATES_H
#define MATH_TEMPLATES_H
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#if Tensile_RUNTIME_LANGUAGE_OCL
#include "CL/cl.h"
#define TENSILEREAL(C) C.s[0]
#define TENSILECOMP(C) C.s[1]
#else
#include <hip/hip_runtime.h>
#define TENSILEREAL(C) C.x
#define TENSILECOMP(C) C.y
#endif

/*******************************************************************************
 * Zero Templates
 ******************************************************************************/
template <typename T>
T tensileGetZero();

/*******************************************************************************
 * One Templates
 ******************************************************************************/
template <typename T>
T tensileGetOne();

/*******************************************************************************
 * Random Templates
 ******************************************************************************/
template <typename T>
T tensileGetRandom();

/*******************************************************************************
 * Trig Templates
 ******************************************************************************/
template <typename T>
T tensileGetTrig(int i);

/*******************************************************************************
 * NaN Templates
 ******************************************************************************/
template <typename T>
T tensileGetNaN();

/*******************************************************************************
 * Integer Templates
 ******************************************************************************/
template <typename T>
T tensileGetTypeForInt(size_t s);

/*******************************************************************************
 * Multiply Templates
 ******************************************************************************/
template <typename Type_return, typename Type_a, typename Type_b>
Type_return tensileMultiply(Type_a a, Type_b b);

/*******************************************************************************
 * Add Templates
 ******************************************************************************/
template <typename Type_return, typename Type_a, typename Type_b>
Type_return tensileAdd(Type_a a, Type_b b);

/*******************************************************************************
 * Floating-Point Equals
 ******************************************************************************/
template <typename T>
bool tensileAlmostEqual(T a, T b);

/*******************************************************************************
 * Floating-Point Equals
 ******************************************************************************/
template <typename T>
bool tensileEqual(T a, T b);

/*******************************************************************************
 * Complex Conjugate
 ******************************************************************************/
template <typename T>
void tensileComplexConjugate(T&);

/*******************************************************************************
 * sizeOf
 ******************************************************************************/
template <typename Type>
size_t tensileSizeOfType();

/*******************************************************************************
 * ToString
 ******************************************************************************/
template <typename Type>
std::string tensileToString(Type);

/*******************************************************************************
 * Floating-Point Equals Zero
 ******************************************************************************/
template <typename T>
bool tensileIsZero(T a)
{
    return tensileAlmostEqual(a, tensileGetZero<T>());
}

#endif

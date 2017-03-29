/*******************************************************************************
* Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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
*******************************************************************************/

#ifndef MATH_TEMPLATES_H
#define MATH_TEMPLATES_H
#include <cmath>
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
template< typename T> T tensileGetZero();


/*******************************************************************************
 * One Templates
 ******************************************************************************/
template< typename T> T tensileGetOne();


/*******************************************************************************
 * Random Templates
 ******************************************************************************/
template< typename T> T tensileGetRandom();


/*******************************************************************************
 * Integer Templates
 ******************************************************************************/
template< typename T> T tensileGetTypeForInt( size_t s );


/*******************************************************************************
 * Multiply Templates
 ******************************************************************************/
template< typename Type >
Type tensileMultiply( Type a, Type b );


/*******************************************************************************
 * Add Templates
 ******************************************************************************/
template< typename Type >
Type tensileAdd( Type a, Type b );


/*******************************************************************************
* Floating-Point Equals
******************************************************************************/
template<typename T>
bool tensileAlmostEqual( T a, T b);


/*******************************************************************************
* Complex Conjugate
******************************************************************************/
template<typename T>
void tensileComplexConjugate(T&);


/*******************************************************************************
* sizeOf
******************************************************************************/
template<typename Type>
size_t tensileSizeOfType();


/*******************************************************************************
* ToString
******************************************************************************/
template<typename Type>
std::string tensileToString(Type);

#endif


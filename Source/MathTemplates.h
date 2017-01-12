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

#if   Tensile_BACKEND_OCL
#include "CL/cl.h"
typedef cl_float2 TensileComplexFloat;
typedef cl_double2 TensileComplexDouble;
#define TENSILEREAL(X) X.s[0]
#define TENSILECOMP(X) X.s[1]
#else
#include <hip/hip_runtime.h>
#endif

namespace Tensile {

/*******************************************************************************
 * Zero Templates
 ******************************************************************************/
template< typename T> T getZero(); // { return static_cast<T>(0); };


/*******************************************************************************
 * One Templates
 ******************************************************************************/
template< typename T> T getOne(); // { return static_cast<T>(1); };

/*******************************************************************************
 * Random Templates
 ******************************************************************************/
template< typename T> T getRandom(); // { return static_cast<T>(1); };

/*******************************************************************************
 * Integer Templates
 ******************************************************************************/
template< typename T> T getTypeForInt( size_t s );


/*******************************************************************************
 * Multiply Templates
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB >
TypeC multiply( TypeA a, TypeB b ); /* {
  return static_cast<TypeC>( a * b );
};*/



/*******************************************************************************
 * Add Templates
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB >
TypeC add( TypeA a, TypeB b ); /* {
  return static_cast<TypeC>( a + b );
};*/


/*******************************************************************************
* Floating-Point Equals
******************************************************************************/
template<typename T>
bool almostEqual( T a, T b); /* {
  return std::fabs(a - b) < std::numeric_limits<T>::epsilon();
} */

/*******************************************************************************
* Complex Conjugate
******************************************************************************/
template<typename T>
void complexConjugate(T&);


/*******************************************************************************
* sizeOf Type
******************************************************************************/
template<typename Type>
size_t sizeOfType();


} // end Tensile namespace

#endif


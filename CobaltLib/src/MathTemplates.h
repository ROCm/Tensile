/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#ifndef MATH_TEMPLATES_H
#define MATH_TEMPLATES_H
#include <cmath>
#include <limits>

namespace Cobalt {

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


} // end Cobalt namespace

#endif


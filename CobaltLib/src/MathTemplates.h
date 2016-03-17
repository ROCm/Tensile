#ifndef MATH_TEMPLATES_H
#define MATH_TEMPLATES_H
#include <cmath>
#include <limits>

/*******************************************************************************
 * Zero Templates
 ******************************************************************************/
template< typename T> T getZero(); // { return static_cast<T>(0); };


/*******************************************************************************
 * One Templates
 ******************************************************************************/
template< typename T> T getOne(); // { return static_cast<T>(1); };



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

#endif
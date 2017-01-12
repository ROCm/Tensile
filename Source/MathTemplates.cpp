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

#include "Tensile.h"
#include "MathTemplates.h"
#include <cstdio>
#include <cstdlib>

namespace Tensile {

/*******************************************************************************
 * Zero Templates
 ******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template<> TensileHalf getZero<TensileHalf>() { return 0.; }
#endif
template<> float getZero<float>() { return 0.f; }
template<> double getZero<double>() { return 0.0; }
template<> TensileComplexFloat getZero<TensileComplexFloat>() {
  TensileComplexFloat zero = {{0.f, 0.f}};
  return zero;
}
template<> TensileComplexDouble getZero<TensileComplexDouble>() {
  TensileComplexDouble zero = {{0.0, 0.0}};
  return zero;
}


/*******************************************************************************
 * One Templates
 ******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template<> TensileHalf getOne<TensileHalf>() { return 1.; }
#endif
template<> float getOne<float>() { return 1.f; }
template<> double getOne<double>() { return 1.0; }
template<> TensileComplexFloat getOne<TensileComplexFloat>() {
  TensileComplexFloat zero = {{1.f, 0.f}};
  return zero;
}
template<> TensileComplexDouble getOne<TensileComplexDouble>() {
  TensileComplexDouble zero = {{1.0, 0.0}};
  return zero;
}


/*******************************************************************************
* Random Templates
******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template<> TensileHalf getRandom<TensileHalf>() { return static_cast<TensileHalf>(rand()%100) /*/static_cast<float>(RAND_MAX)*/ ; }
#endif
template<> float getRandom<float>() { return static_cast<float>(rand()%100) /*/static_cast<float>(RAND_MAX)*/ ; }
template<> double getRandom<double>() { return static_cast<double>(rand()) / static_cast<double>(RAND_MAX); }
template<> TensileComplexFloat getRandom<TensileComplexFloat>() {
  //TensileComplexFloat r = { 1.f, 0.f };
  return {{ getRandom<float>(), getRandom<float>() }};
}
template<> TensileComplexDouble getRandom<TensileComplexDouble>() {
  //TensileComplexDouble r = { 1.0, 0.0 };
  return {{ getRandom<double>(), getRandom<double>() }};
}

template<> float getTypeForInt<float>( size_t s ) { return static_cast<float>(s); }
template<> double getTypeForInt<double>( size_t s ) { return static_cast<double>(s); }
template<> TensileComplexFloat getTypeForInt<TensileComplexFloat>( size_t s ) { return {{ static_cast<float>(s), static_cast<float>(s) }}; }
template<> TensileComplexDouble getTypeForInt<TensileComplexDouble>( size_t s ) { return {{ static_cast<double>(s), static_cast<double>(s) }}; }

/*******************************************************************************
 * Multiply Templates
 ******************************************************************************/

// half
#ifdef Tensile_Enable_FP16_HOST
template< >
TensileHalf multiply( TensileHalf a, TensileHalf b ) {
  return a*b;
}
#endif
// single
template< >
float multiply( float a, float b ) {
  return a*b;
}
// double
template< >
double multiply( double a, double b ) {
  return a*b;
}
// complex single
template< >
TensileComplexFloat multiply( TensileComplexFloat a, TensileComplexFloat b ) {
  TensileComplexFloat c;
  TENSILEREAL(c) = TENSILEREAL(a)*TENSILEREAL(b) - TENSILECOMP(a)*TENSILECOMP(b);
  TENSILECOMP(c) = TENSILEREAL(a)*TENSILECOMP(b) + TENSILECOMP(a)*TENSILEREAL(b);
  return c;
}
// complex double
template< >
TensileComplexDouble multiply( TensileComplexDouble a, TensileComplexDouble b ) {
  TensileComplexDouble c;
  TENSILEREAL(c) = TENSILEREAL(a)*TENSILEREAL(b) - TENSILECOMP(a)*TENSILECOMP(b);
  TENSILECOMP(c) = TENSILEREAL(a)*TENSILECOMP(b) + TENSILECOMP(a)*TENSILEREAL(b);
  return c;
}


/*******************************************************************************
 * Add Templates
 ******************************************************************************/

// half
#ifdef Tensile_Enable_FP16_HOST
template< >
TensileHalf add( TensileHalf a, TensileHalf b ) {
  return a+b;
}
#endif
// single
template< >
float add( float a, float b ) {
  return a+b;
}
// double
template< >
double add( double a, double b ) {
  return a+b;
}
// complex single
template< >
TensileComplexFloat add( TensileComplexFloat a, TensileComplexFloat b ) {
  TensileComplexFloat c;
  TENSILEREAL(c) = TENSILEREAL(a)+TENSILEREAL(b);
  TENSILECOMP(c) = TENSILECOMP(a)+TENSILECOMP(b);
  return c;
}
// complex double
template< >
TensileComplexDouble add( TensileComplexDouble a, TensileComplexDouble b ) {
  TensileComplexDouble c;
  TENSILEREAL(c) = TENSILEREAL(a)+TENSILEREAL(b);
  TENSILECOMP(c) = TENSILECOMP(a)+TENSILECOMP(b);
  return c;
}

/*******************************************************************************
* Floating-Point Equals
******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template< >
bool almostEqual(TensileHalf a, TensileHalf b) {
  return std::fabs(a - b)/(std::fabs(a)+std::fabs(b)+1) < 0.01; // ?
}
#endif
template< >
bool almostEqual(float a, float b) {
  return std::fabs(a - b)/(std::fabs(a)+std::fabs(b)+1) < 0.0001; // 7 digits of precision - 2
}
template< >
bool almostEqual(double a, double b) {
  return std::fabs(a - b) / ( std::fabs(a) + std::fabs(b)+1 ) < 0.000000000001; // 15 digits of precision - 2
}
template< >
bool almostEqual( TensileComplexFloat a, TensileComplexFloat b) {
  return almostEqual(TENSILEREAL(a), TENSILEREAL(b)) && almostEqual(TENSILECOMP(a), TENSILECOMP(b));
}
template< >
bool almostEqual(TensileComplexDouble a, TensileComplexDouble b) {
  return almostEqual(TENSILEREAL(a), TENSILEREAL(b)) && almostEqual(TENSILECOMP(a), TENSILECOMP(b));
}

/*******************************************************************************
* Complex Conjugate
******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template< >
void complexConjugate(TensileHalf &) {}
#endif
template< >
void complexConjugate(float &) {}
template< >
void complexConjugate(double &) {}
template< >
void complexConjugate( TensileComplexFloat & v) {
  TENSILECOMP(v) = -TENSILECOMP(v);
}
template< >
void complexConjugate(TensileComplexDouble & v) {
  TENSILECOMP(v) = -TENSILECOMP(v);
}


/*******************************************************************************
 * sizeOf Type
 ******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template<> size_t sizeOfType<TensileHalf>(){ return sizeof(TensileHalf); }
#endif
template<> size_t sizeOfType<float>(){ return sizeof(float); }
template<> size_t sizeOfType<double>(){ return sizeof(double); }
template<> size_t sizeOfType<TensileComplexFloat>(){ return sizeof(TensileComplexFloat); }
template<> size_t sizeOfType<TensileComplexDouble>(){ return sizeof(TensileComplexDouble); }
template<> size_t sizeOfType<void>() { return 0; }

} // end namespace Tensile


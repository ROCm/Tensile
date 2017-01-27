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


/*******************************************************************************
 * Zero Templates
 ******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template<> TensileHalf tensileGetZero<TensileHalf>() { return 0.; }
#endif
template<> float tensileGetZero<float>() { return 0.f; }
template<> double tensileGetZero<double>() { return 0.0; }
template<> TensileComplexFloat tensileGetZero<TensileComplexFloat>() {
  TensileComplexFloat zero;
  TENSILEREAL(zero) = 0.f;
  TENSILECOMP(zero) = 0.f;
  return zero;
}
template<> TensileComplexDouble tensileGetZero<TensileComplexDouble>() {
  TensileComplexDouble zero;
  TENSILEREAL(zero) = 0.0;
  TENSILECOMP(zero) = 0.0;
  return zero;
}


/*******************************************************************************
 * One Templates
 ******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template<> TensileHalf tensileGetOne<TensileHalf>() { return 1.; }
#endif
template<> float tensileGetOne<float>() { return 1.f; }
template<> double tensileGetOne<double>() { return 1.0; }
template<> TensileComplexFloat tensileGetOne<TensileComplexFloat>() {
  TensileComplexFloat one;
  TENSILEREAL(one) = 1.f;
  TENSILECOMP(one) = 0.f;
  return one;
}
template<> TensileComplexDouble tensileGetOne<TensileComplexDouble>() {
  TensileComplexDouble one;
  TENSILEREAL(one) = 1.0;
  TENSILECOMP(one) = 0.0;
  return one;
}


/*******************************************************************************
* Random Templates
******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template<> TensileHalf tensileGetRandom<TensileHalf>() { return static_cast<TensileHalf>(rand()%100) /*/static_cast<float>(RAND_MAX)*/ ; }
#endif
template<> float tensileGetRandom<float>() { return static_cast<float>(rand()%100) /*/static_cast<float>(RAND_MAX)*/ ; }
template<> double tensileGetRandom<double>() { return static_cast<double>(rand()) / static_cast<double>(RAND_MAX); }
template<> TensileComplexFloat tensileGetRandom<TensileComplexFloat>() {
  TensileComplexFloat r;
  TENSILEREAL(r) = tensileGetRandom<float>();
  TENSILECOMP(r) = tensileGetRandom<float>();
  return r;
}
template<> TensileComplexDouble tensileGetRandom<TensileComplexDouble>() {
  TensileComplexDouble r;
  TENSILEREAL(r) = tensileGetRandom<double>();
  TENSILECOMP(r) = tensileGetRandom<double>();
  return r;
}


template<> float tensileGetTypeForInt<float>( size_t s ) { return static_cast<float>(s); }
template<> double tensileGetTypeForInt<double>( size_t s ) { return static_cast<double>(s); }
template<> TensileComplexFloat tensileGetTypeForInt<TensileComplexFloat>( size_t s ) {
  TensileComplexFloat f;
  TENSILEREAL(f) = static_cast<float>(s);
  TENSILECOMP(f) = static_cast<float>(s);
  return f;
}
template<> TensileComplexDouble tensileGetTypeForInt<TensileComplexDouble>( size_t s ) {
  TensileComplexDouble d;
  TENSILEREAL(d) = static_cast<float>(s);
  TENSILECOMP(d) = static_cast<float>(s);
  return d;
}

/*******************************************************************************
 * tensileMultiply Templates
 ******************************************************************************/

// half
#ifdef Tensile_Enable_FP16_HOST
template< >
TensileHalf tensileMultiply( TensileHalf a, TensileHalf b ) {
  return a*b;
}
#endif
// single
template< >
float tensileMultiply( float a, float b ) {
  return a*b;
}
// double
template< >
double tensileMultiply( double a, double b ) {
  return a*b;
}
// complex single
template< >
TensileComplexFloat tensileMultiply( TensileComplexFloat a, TensileComplexFloat b ) {
  TensileComplexFloat c;
  TENSILEREAL(c) = TENSILEREAL(a)*TENSILEREAL(b) - TENSILECOMP(a)*TENSILECOMP(b);
  TENSILECOMP(c) = TENSILEREAL(a)*TENSILECOMP(b) + TENSILECOMP(a)*TENSILEREAL(b);
  return c;
}
// complex double
template< >
TensileComplexDouble tensileMultiply( TensileComplexDouble a, TensileComplexDouble b ) {
  TensileComplexDouble c;
  TENSILEREAL(c) = TENSILEREAL(a)*TENSILEREAL(b) - TENSILECOMP(a)*TENSILECOMP(b);
  TENSILECOMP(c) = TENSILEREAL(a)*TENSILECOMP(b) + TENSILECOMP(a)*TENSILEREAL(b);
  return c;
}


/*******************************************************************************
 * tensileAdd Templates
 ******************************************************************************/

// half
#ifdef Tensile_Enable_FP16_HOST
template< >
TensileHalf tensileAdd( TensileHalf a, TensileHalf b ) {
  return a+b;
}
#endif
// single
template< >
float tensileAdd( float a, float b ) {
  return a+b;
}
// double
template< >
double tensileAdd( double a, double b ) {
  return a+b;
}
// complex single
template< >
TensileComplexFloat tensileAdd( TensileComplexFloat a, TensileComplexFloat b ) {
  TensileComplexFloat c;
  TENSILEREAL(c) = TENSILEREAL(a)+TENSILEREAL(b);
  TENSILECOMP(c) = TENSILECOMP(a)+TENSILECOMP(b);
  return c;
}
// complex double
template< >
TensileComplexDouble tensileAdd( TensileComplexDouble a, TensileComplexDouble b ) {
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
bool tensileAlmostEqual(TensileHalf a, TensileHalf b) {
  return std::fabs(a - b)/(std::fabs(a)+std::fabs(b)+1) < 0.01; // ?
}
#endif
template< >
bool tensileAlmostEqual(float a, float b) {
  return std::fabs(a - b)/(std::fabs(a)+std::fabs(b)+1) < 0.0001; // 7 digits of precision - 2
}
template< >
bool tensileAlmostEqual(double a, double b) {
  return std::fabs(a - b) / ( std::fabs(a) + std::fabs(b)+1 ) < 0.000000000001; // 15 digits of precision - 2
}
template< >
bool tensileAlmostEqual( TensileComplexFloat a, TensileComplexFloat b) {
  return tensileAlmostEqual(TENSILEREAL(a), TENSILEREAL(b)) && tensileAlmostEqual(TENSILECOMP(a), TENSILECOMP(b));
}
template< >
bool tensileAlmostEqual(TensileComplexDouble a, TensileComplexDouble b) {
  return tensileAlmostEqual(TENSILEREAL(a), TENSILEREAL(b)) && tensileAlmostEqual(TENSILECOMP(a), TENSILECOMP(b));
}

/*******************************************************************************
* Complex Conjugate
******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template< >
void tensileComplexConjugate(TensileHalf &) {}
#endif
template< >
void tensileComplexConjugate(float &) {}
template< >
void tensileComplexConjugate(double &) {}
template< >
void tensileComplexConjugate( TensileComplexFloat & v) {
  TENSILECOMP(v) = -TENSILECOMP(v);
}
template< >
void tensileComplexConjugate(TensileComplexDouble & v) {
  TENSILECOMP(v) = -TENSILECOMP(v);
}


/*******************************************************************************
 * sizeOf Type
 ******************************************************************************/
#ifdef Tensile_Enable_FP16_HOST
template<> size_t tensileSizeOfType<TensileHalf>(){ return sizeof(TensileHalf); }
#endif
template<> size_t tensileSizeOfType<float>(){ return sizeof(float); }
template<> size_t tensileSizeOfType<double>(){ return sizeof(double); }
template<> size_t tensileSizeOfType<TensileComplexFloat>(){ return sizeof(TensileComplexFloat); }
template<> size_t tensileSizeOfType<TensileComplexDouble>(){ return sizeof(TensileComplexDouble); }
template<> size_t tensileSizeOfType<void>() { return 0; }



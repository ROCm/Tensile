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

#include "TensileTypes.h"
#include "MathTemplates.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string.h>
#include <sstream>
#include <stdexcept>


/*******************************************************************************
 * Zero Templates
 ******************************************************************************/
#ifdef Tensile_ENABLE_HALF
template<> TensileHalf tensileGetZero<TensileHalf>() { return 0.; }
#endif
template<> uint32_t tensileGetZero<uint32_t>() { return 0; }
template<> int32_t tensileGetZero<int32_t>() { return 0; }
template<> tensile_bfloat16 tensileGetZero<tensile_bfloat16>() { return static_cast<tensile_bfloat16>(0.f); }
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
#ifdef Tensile_ENABLE_HALF
template<> TensileHalf tensileGetOne<TensileHalf>() { return 1.; }
#endif
template<> uint32_t tensileGetOne<uint32_t>() { return 0x01010101; }
template<> int32_t tensileGetOne<int32_t>() { return 1; }
template<> tensile_bfloat16 tensileGetOne<tensile_bfloat16>() { return static_cast<tensile_bfloat16>(1.f); }
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
#ifdef Tensile_ENABLE_HALF
template<> TensileHalf tensileGetRandom<TensileHalf>() { return static_cast<TensileHalf>((rand()%7) - 3); }
#endif
template<> uint32_t tensileGetRandom<uint32_t>() { 
   int8_t t0 = static_cast<int8_t>((rand()%7) - 3); 
   int8_t t1 = static_cast<int8_t>((rand()%7) - 3); 
   int8_t t2 = static_cast<int8_t>((rand()%7) - 3); 
   int8_t t3 = static_cast<int8_t>((rand()%7) - 3); 
   int8_t t1x4[4] = {t0, t1, t2, t3};
   uint32_t tmp; 
   memcpy(&tmp, t1x4, sizeof(uint32_t));
   return tmp; 
}
template<> int32_t tensileGetRandom<int32_t>() { return static_cast<int32_t>((rand()%7) - 3); }
template<> float tensileGetRandom<float>() { return static_cast<float>((rand()%201) - 100); }
template<> tensile_bfloat16 tensileGetRandom<tensile_bfloat16>() { return static_cast<tensile_bfloat16>(static_cast<float>((rand()%7) - 3)); }
template<> double tensileGetRandom<double>() { return static_cast<double>((rand()%2001) - 1000); }
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


#ifdef Tensile_ENABLE_HALF
template<> TensileHalf tensileGetTypeForInt<TensileHalf>( size_t s ) { return static_cast<TensileHalf>(s); }
#endif
template<> tensile_bfloat16 tensileGetTypeForInt<tensile_bfloat16>( size_t s ) { return static_cast<tensile_bfloat16>(static_cast<float>(s)); }
template<> float tensileGetTypeForInt<float>( size_t s ) { return static_cast<float>(s); }
template<> double tensileGetTypeForInt<double>( size_t s ) { return static_cast<double>(s); }
template<> int tensileGetTypeForInt<int>( size_t s ) { return static_cast<int>(s); }
template<> unsigned int tensileGetTypeForInt<unsigned int>( size_t s ) { return static_cast<unsigned int>(s); }
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
* Trig Templates
******************************************************************************/
#ifdef Tensile_ENABLE_HALF
template<> TensileHalf tensileGetTrig<TensileHalf>(int i) { return static_cast<TensileHalf>(sin(i)); }
#endif
template<> uint32_t tensileGetTrig<uint32_t>(int i) { 
   int8_t t0 = static_cast<int8_t>((rand()%7) - 3); 
   int8_t t1 = static_cast<int8_t>((rand()%7) - 3); 
   int8_t t2 = static_cast<int8_t>((rand()%7) - 3); 
   int8_t t3 = static_cast<int8_t>((rand()%7) - 3); 
   int8_t t1x4[4] = {t0, t1, t2, t3};
   uint32_t tmp; 
   memcpy(&tmp, t1x4, sizeof(uint32_t));
   return tmp; 
}
template<> int32_t tensileGetTrig<int32_t>(int i) { return static_cast<int32_t>((rand()%7) - 3); }
template<> float tensileGetTrig<float>(int i) { return static_cast<float>(sin(i)); }
template<> tensile_bfloat16 tensileGetTrig<tensile_bfloat16>(int i) { return sin(static_cast<tensile_bfloat16>(i)); }
template<> double tensileGetTrig<double>(int i) { return static_cast<double>(sin(i)); }
template<> TensileComplexFloat tensileGetTrig<TensileComplexFloat>(int i) {
  TensileComplexFloat r;
  TENSILEREAL(r) = tensileGetTrig<float>(i);
  TENSILECOMP(r) = tensileGetTrig<float>(static_cast<float>(i) + 0.5);
  return r;
}
template<> TensileComplexDouble tensileGetTrig<TensileComplexDouble>(int i) {
  TensileComplexDouble r;
  TENSILEREAL(r) = tensileGetTrig<double>(i);
  TENSILECOMP(r) = tensileGetTrig<double>(static_cast<double>(i) + 0.5);
  return r;
}


/*******************************************************************************
 * NaN Templates
 ******************************************************************************/
#ifdef Tensile_ENABLE_HALF
template<> TensileHalf tensileGetNaN<TensileHalf>() { return std::numeric_limits<TensileHalf>::quiet_NaN(); }
#endif
template<> tensile_bfloat16 tensileGetNaN<tensile_bfloat16>() { return static_cast<tensile_bfloat16>(std::numeric_limits<float>::quiet_NaN()); }
template<> float tensileGetNaN<float>() { return std::numeric_limits<float>::quiet_NaN(); }
template<> double tensileGetNaN<double>() { return std::numeric_limits<double>::quiet_NaN(); }
template<> int tensileGetNaN<int>() { return std::numeric_limits<int>::max(); }
template<> unsigned int tensileGetNaN<unsigned int>() { return std::numeric_limits<unsigned int>::max(); }
template<> TensileComplexFloat tensileGetNaN<TensileComplexFloat>() {
  TensileComplexFloat nan_value;
  TENSILEREAL(nan_value) = std::numeric_limits<float>::quiet_NaN();
  TENSILECOMP(nan_value) = std::numeric_limits<float>::quiet_NaN();
  return nan_value;
}
template<> TensileComplexDouble tensileGetNaN<TensileComplexDouble>() {
  TensileComplexDouble nan_value;
  TENSILEREAL(nan_value) = std::numeric_limits<double>::quiet_NaN();
  TENSILECOMP(nan_value) = std::numeric_limits<double>::quiet_NaN();
  return nan_value;
}


/*******************************************************************************
 * tensileMultiply Templates
 ******************************************************************************/

// half
#ifdef Tensile_ENABLE_HALF
template< >
TensileHalf tensileMultiply( TensileHalf a, TensileHalf b ) {
  return a*b;
}
template< >
float tensileMultiply( TensileHalf a, TensileHalf b ) {
  return (float)a * (float)b;
}
template< >
float tensileMultiply( TensileHalf a, float b ) {
  return ((float)a) * b;
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
// int
template< >
int tensileMultiply( int a, int b ) {
  return a*b;
}

template< >
unsigned int tensileMultiply( int a, int b ) {
  return a*b;
}
// unsigned int
template< >
unsigned int tensileMultiply( unsigned int a, unsigned int b ) {
  return a*b;
}

template< >
unsigned int tensileMultiply( int a, unsigned int b ) {
  return a*b;
}
// mixed tensile_bfloat16 float
template< >
tensile_bfloat16 tensileMultiply( tensile_bfloat16 a, tensile_bfloat16 b ) {
  return a * b;
}
template< >
tensile_bfloat16 tensileMultiply( float a, tensile_bfloat16 b ) {
  return static_cast<tensile_bfloat16>(a) * b;
}
template< >
tensile_bfloat16 tensileMultiply( tensile_bfloat16 a, float b) {
  return a * static_cast<tensile_bfloat16>(b);
}
template< >
float tensileMultiply( tensile_bfloat16 a, tensile_bfloat16 b ) {
  return static_cast<float>(a) * static_cast<float>(b);
}
template< >
float tensileMultiply( float a, tensile_bfloat16 b ) {
  return a * static_cast<float>(b);
}
template< >
float tensileMultiply( tensile_bfloat16 a, float b ) {
  return static_cast<float>(a) * b;
}




// complex single
template< >
float tensileMultiply( TensileComplexFloat a, TensileComplexFloat b ) {
  throw std::logic_error( "Reached a supposed unreachable point" );
  return 0.0f;
}
template< >
float tensileMultiply( TensileComplexFloat a, float b ) {
  throw std::logic_error( "Reached a supposed unreachable point" );
  return 0.0f;
}
template< >
TensileComplexFloat tensileMultiply( TensileComplexFloat a, TensileComplexFloat b ) {
  TensileComplexFloat c;
  TENSILEREAL(c) = TENSILEREAL(a)*TENSILEREAL(b) - TENSILECOMP(a)*TENSILECOMP(b);
  TENSILECOMP(c) = TENSILEREAL(a)*TENSILECOMP(b) + TENSILECOMP(a)*TENSILEREAL(b);
  return c;
}
// complex double
template< >
float tensileMultiply( TensileComplexDouble a, TensileComplexDouble b ) {
  throw std::logic_error( "Reached a supposed unreachable point" );
  return 0.0f;
}
template< >
float tensileMultiply( TensileComplexDouble a, float b ) {
  throw std::logic_error( "Reached a supposed unreachable point" );
  return 0.0f;
}
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
#ifdef Tensile_ENABLE_HALF
template< >
TensileHalf tensileAdd( TensileHalf a, TensileHalf b ) {
  return a+b;
}
template< >
float tensileAdd( TensileHalf a, float b ) {
  return (float)a+b;
}
#endif
template< >
tensile_bfloat16 tensileAdd( tensile_bfloat16 a, tensile_bfloat16 b ) {
  return a+b;
}
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
// unsigned int
template< >
unsigned int tensileAdd( unsigned int a, unsigned int b ) {
  return a+b;
}
// int
template< >
int tensileAdd( int a, int b ) {
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
template< >
float tensileAdd( TensileComplexFloat a, float b ) {
  throw std::logic_error( "Reached a supposed unreachable point" );
  return 0.0f;
}

// complex double
template< >
TensileComplexDouble tensileAdd( TensileComplexDouble a, TensileComplexDouble b ) {
  TensileComplexDouble c;
  TENSILEREAL(c) = TENSILEREAL(a)+TENSILEREAL(b);
  TENSILECOMP(c) = TENSILECOMP(a)+TENSILECOMP(b);
  return c;
}
template< >
float tensileAdd( TensileComplexDouble a, float b ) {
  throw std::logic_error( "Reached a supposed unreachable point" );
  return 0.0f;
}

/*******************************************************************************
* Floating-Point Almost Equals
******************************************************************************/
#ifdef Tensile_ENABLE_HALF
template< >
bool tensileAlmostEqual(TensileHalf a, TensileHalf b) {
  TensileHalf absA = (a > 0) ? a : -a;
  TensileHalf absB = (b > 0) ? b : -b;
  TensileHalf absDiff = (a-b > 0) ? a-b : b-a;
  return absDiff/(absA+absB+1) < 0.01;
}
#endif
template< >
bool tensileAlmostEqual(tensile_bfloat16 a, tensile_bfloat16 b) {
  tensile_bfloat16 absA = (a > static_cast<tensile_bfloat16>(0.0f)) ? a : static_cast<tensile_bfloat16>(0.0f) - a;
  tensile_bfloat16 absB = (b > static_cast<tensile_bfloat16>(0.0f)) ? b : static_cast<tensile_bfloat16>(0.0f) - b;
  tensile_bfloat16 absDiff = (a-b > static_cast<tensile_bfloat16>(0.0f)) ? a-b : b-a;
  return absDiff/(absA+absB+static_cast<tensile_bfloat16>(1.0f)) < static_cast<tensile_bfloat16>(0.1f);
}
template< >
bool tensileAlmostEqual(float a, float b) {
  return std::fabs(a - b)/(std::fabs(a)+std::fabs(b)+1) < 0.0001; // 7 digits of precision - 2
}
template< >
bool tensileAlmostEqual(double a, double b) {
  return std::fabs(a - b) / ( std::fabs(a) + std::fabs(b)+1 ) < 0.000000000001; // 15 digits of precision - 2
}
template< >
bool tensileAlmostEqual(int a, int b) {
  return a == b;
}
template< >
bool tensileAlmostEqual(unsigned int a, unsigned int b) {
  return a == b;
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
* Floating-Point Equals
******************************************************************************/
#ifdef Tensile_ENABLE_HALF
template< >
bool tensileEqual(TensileHalf a, TensileHalf b) {
  return a == b;
}
#endif
template< >
bool tensileEqual(float a, float b) {
  return a == b;
}
template< >
bool tensileEqual(double a, double b) {
  return a == b;
}
template< >
bool tensileEqual(int a, int b) {
  return a == b;
}
template< >
bool tensileEqual(unsigned int a, unsigned int b) {
  return a == b;
}
template< >
bool tensileEqual( TensileComplexFloat a, TensileComplexFloat b) {
  return tensileEqual(TENSILEREAL(a), TENSILEREAL(b)) && tensileEqual(TENSILECOMP(a), TENSILECOMP(b));
}
template< >
bool tensileEqual(TensileComplexDouble a, TensileComplexDouble b) {
  return tensileEqual(TENSILEREAL(a), TENSILEREAL(b)) && tensileEqual(TENSILECOMP(a), TENSILECOMP(b));
}


/*******************************************************************************
* Complex Conjugate
******************************************************************************/
#ifdef Tensile_ENABLE_HALF
template< >
void tensileComplexConjugate(TensileHalf &) {}
#endif
template< >
void tensileComplexConjugate(unsigned int &) {}
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
#ifdef Tensile_ENABLE_HALF
template<> size_t tensileSizeOfType<TensileHalf>(){ return sizeof(TensileHalf); }
#endif
template<> size_t tensileSizeOfType<float>(){ return sizeof(float); }
template<> size_t tensileSizeOfType<double>(){ return sizeof(double); }
template<> size_t tensileSizeOfType<TensileComplexFloat>(){ return sizeof(TensileComplexFloat); }
template<> size_t tensileSizeOfType<TensileComplexDouble>(){ return sizeof(TensileComplexDouble); }
template<> size_t tensileSizeOfType<void>() { return 0; }

/*******************************************************************************
 * ToString
 ******************************************************************************/
template<> std::string tensileToString(float v){
  std::ostringstream s;
  s << v;
  return s.str();
  //return std::to_string(v);
  }
template<> std::string tensileToString(double v){
  std::ostringstream s;
  s << v;
  return s.str();
  //return std::to_string(v);
  }
template<> std::string tensileToString(int v){
  std::ostringstream s;
  s << v;
  return s.str();
  //return std::to_string(v);
  }
template<> std::string tensileToString(unsigned int v){
  std::ostringstream s;
  s << v;
  return s.str();
  //return std::to_string(v);
  }
template<> std::string tensileToString(TensileComplexFloat v){
  std::string s;
  s += tensileToString(TENSILEREAL(v));
  s += ",";
  s += tensileToString(TENSILECOMP(v));
  return s;
  //return tensileToString(TENSILEREAL(v))+","+tensileToString(TENSILECOMP(v));
}
template<> std::string tensileToString(TensileComplexDouble v){
  std::string s;
  s += tensileToString(TENSILEREAL(v));
  s += ",";
  s += tensileToString(TENSILECOMP(v));
  return s;
  //return tensileToString(TENSILEREAL(v))+","+tensileToString(TENSILECOMP(v));
}
#ifdef Tensile_ENABLE_HALF
template<> std::string tensileToString(TensileHalf v){
  return tensileToString(static_cast<float>(v)); }
#endif
template<> std::string tensileToString(tensile_bfloat16 v){
  return tensileToString(static_cast<float>(v)); }




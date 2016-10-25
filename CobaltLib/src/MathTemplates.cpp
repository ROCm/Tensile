/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

#include "Cobalt.h"
#include "MathTemplates.h"
#include <cstdio>
#include <cstdlib>

namespace Cobalt {

/*******************************************************************************
 * Zero Templates
 ******************************************************************************/
#ifdef Cobalt_Enable_FP16_HOST
template<> CobaltHalf getZero<CobaltHalf>() { return 0.; }
#endif
template<> float getZero<float>() { return 0.f; }
template<> double getZero<double>() { return 0.0; }
template<> CobaltComplexFloat getZero<CobaltComplexFloat>() {
  CobaltComplexFloat zero = {{0.f, 0.f}};
  return zero;
}
template<> CobaltComplexDouble getZero<CobaltComplexDouble>() {
  CobaltComplexDouble zero = {{0.0, 0.0}};
  return zero;
}


/*******************************************************************************
 * One Templates
 ******************************************************************************/
#ifdef Cobalt_Enable_FP16_HOST
template<> CobaltHalf getOne<CobaltHalf>() { return 1.; }
#endif
template<> float getOne<float>() { return 1.f; }
template<> double getOne<double>() { return 1.0; }
template<> CobaltComplexFloat getOne<CobaltComplexFloat>() {
  CobaltComplexFloat zero = {{1.f, 0.f}};
  return zero;
}
template<> CobaltComplexDouble getOne<CobaltComplexDouble>() {
  CobaltComplexDouble zero = {{1.0, 0.0}};
  return zero;
}


/*******************************************************************************
* Random Templates
******************************************************************************/
#ifdef Cobalt_Enable_FP16_HOST
template<> CobaltHalf getRandom<CobaltHalf>() { return static_cast<CobaltHalf>(rand()%100) /*/static_cast<float>(RAND_MAX)*/ ; }
#endif
template<> float getRandom<float>() { return static_cast<float>(rand()%100) /*/static_cast<float>(RAND_MAX)*/ ; }
template<> double getRandom<double>() { return static_cast<double>(rand()) / static_cast<double>(RAND_MAX); }
template<> CobaltComplexFloat getRandom<CobaltComplexFloat>() {
  //CobaltComplexFloat r = { 1.f, 0.f };
  return {{ getRandom<float>(), getRandom<float>() }};
}
template<> CobaltComplexDouble getRandom<CobaltComplexDouble>() {
  //CobaltComplexDouble r = { 1.0, 0.0 };
  return {{ getRandom<double>(), getRandom<double>() }};
}

template<> float getTypeForInt<float>( size_t s ) { return static_cast<float>(s); }
template<> double getTypeForInt<double>( size_t s ) { return static_cast<double>(s); }
template<> CobaltComplexFloat getTypeForInt<CobaltComplexFloat>( size_t s ) { return {{ static_cast<float>(s), static_cast<float>(s) }}; }
template<> CobaltComplexDouble getTypeForInt<CobaltComplexDouble>( size_t s ) { return {{ static_cast<double>(s), static_cast<double>(s) }}; }

/*******************************************************************************
 * Multiply Templates
 ******************************************************************************/

// half
#ifdef Cobalt_Enable_FP16_HOST
template< >
CobaltHalf multiply( CobaltHalf a, CobaltHalf b ) {
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
CobaltComplexFloat multiply( CobaltComplexFloat a, CobaltComplexFloat b ) {
  CobaltComplexFloat c;
  c.x = a.x*b.x - a.y*b.y;
  c.y = a.x*b.y + a.y*b.x;
  return c;
}
// complex double
template< >
CobaltComplexDouble multiply( CobaltComplexDouble a, CobaltComplexDouble b ) {
  CobaltComplexDouble c;
  c.x = a.x*b.x - a.y*b.y;
  c.y = a.x*b.y + a.y*b.x;
  return c;
}


/*******************************************************************************
 * Add Templates
 ******************************************************************************/

// half
#ifdef Cobalt_Enable_FP16_HOST
template< >
CobaltHalf add( CobaltHalf a, CobaltHalf b ) {
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
CobaltComplexFloat add( CobaltComplexFloat a, CobaltComplexFloat b ) {
  CobaltComplexFloat c;
  c.x = a.x+b.x;
  c.y = a.y+b.y;
  return c;
}
// complex double
template< >
CobaltComplexDouble add( CobaltComplexDouble a, CobaltComplexDouble b ) {
  CobaltComplexDouble c;
  c.x = a.x+b.x;
  c.y = a.y+b.y;
  return c;
}

/*******************************************************************************
* Floating-Point Equals
******************************************************************************/
#ifdef Cobalt_Enable_FP16_HOST
template< >
bool almostEqual(CobaltHalf a, CobaltHalf b) {
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
bool almostEqual( CobaltComplexFloat a, CobaltComplexFloat b) {
  return almostEqual(a.x, b.x) && almostEqual(a.y, b.y);
}
template< >
bool almostEqual(CobaltComplexDouble a, CobaltComplexDouble b) {
  return almostEqual(a.x, b.x) && almostEqual(a.y, b.y);
}

/*******************************************************************************
* Complex Conjugate
******************************************************************************/
#ifdef Cobalt_Enable_FP16_HOST
template< >
void complexConjugate(CobaltHalf & v) {}
#endif
template< >
void complexConjugate(float & v) {}
template< >
void complexConjugate(double & v) {}
template< >
void complexConjugate( CobaltComplexFloat & v) {
  v.y = -v.y;
}
template< >
void complexConjugate(CobaltComplexDouble & v) {
  v.y = -v.y;
}


/*******************************************************************************
 * sizeOf Type
 ******************************************************************************/
#ifdef Cobalt_Enable_FP16_HOST
template<> size_t sizeOfType<CobaltHalf>(){ return sizeof(CobaltHalf); }
#endif
template<> size_t sizeOfType<float>(){ return sizeof(float); }
template<> size_t sizeOfType<double>(){ return sizeof(double); }
template<> size_t sizeOfType<CobaltComplexFloat>(){ return sizeof(CobaltComplexFloat); }
template<> size_t sizeOfType<CobaltComplexDouble>(){ return sizeof(CobaltComplexDouble); }
template<> size_t sizeOfType<void>() { return 0; }

} // end namespace Cobalt


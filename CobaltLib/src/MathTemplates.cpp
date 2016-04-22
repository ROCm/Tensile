#include "Cobalt.h"
#include "MathTemplates.h"
#include <cstdio>

namespace Cobalt {

/*******************************************************************************
 * Zero Templates
 ******************************************************************************/
template<> float getZero<float>() { return 0.f; }
template<> double getZero<double>() { return 0.0; }
template<> CobaltComplexFloat getZero<CobaltComplexFloat>() {
  CobaltComplexFloat zero = {0.f, 0.f};
  return zero;
}
template<> CobaltComplexDouble getZero<CobaltComplexDouble>() {
  CobaltComplexDouble zero = {0.0, 0.0};
  return zero;
}


/*******************************************************************************
 * One Templates
 ******************************************************************************/
template<> float getOne<float>() { return 1.f; }
template<> double getOne<double>() { return 1.0; }
template<> CobaltComplexFloat getOne<CobaltComplexFloat>() {
  CobaltComplexFloat zero = {1.f, 0.f};
  return zero;
}
template<> CobaltComplexDouble getOne<CobaltComplexDouble>() {
  CobaltComplexDouble zero = {1.0, 0.0};
  return zero;
}


/*******************************************************************************
* Random Templates
******************************************************************************/
template<> float getRandom<float>() { return static_cast<float>(rand())/static_cast<float>(RAND_MAX); }
template<> double getRandom<double>() { return static_cast<double>(rand()) / static_cast<double>(RAND_MAX); }
template<> CobaltComplexFloat getRandom<CobaltComplexFloat>() {
  //CobaltComplexFloat r = { 1.f, 0.f };
  return { getRandom<float>(), getRandom<float>() };
}
template<> CobaltComplexDouble getRandom<CobaltComplexDouble>() {
  //CobaltComplexDouble r = { 1.0, 0.0 };
  return { getRandom<double>(), getRandom<double>()};
}

/*******************************************************************************
 * Multiply Templates
 ******************************************************************************/

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
  c.s[0] = a.s[0]*b.s[0] - a.s[1]*b.s[1];
  c.s[1] = a.s[0]*b.s[1] + a.s[1]*b.s[0];
  return c;
}
// complex double
template< >
CobaltComplexDouble multiply( CobaltComplexDouble a, CobaltComplexDouble b ) {
  CobaltComplexDouble c;
  c.s[0] = a.s[0]*b.s[0] - a.s[1]*b.s[1];
  c.s[1] = a.s[0]*b.s[1] + a.s[1]*b.s[0];
  return c;
}


/*******************************************************************************
 * Add Templates
 ******************************************************************************/

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
  c.s[0] = a.s[0]+b.s[0];
  c.s[1] = a.s[1]+b.s[1];
  return c;
}
// complex double
template< >
CobaltComplexDouble add( CobaltComplexDouble a, CobaltComplexDouble b ) {
  CobaltComplexDouble c;
  c.s[0] = a.s[0]+b.s[0];
  c.s[1] = a.s[1]+b.s[1];
  return c;
}

/*******************************************************************************
* Floating-Point Equals
******************************************************************************/
template< >
bool almostEqual(float a, float b) {
  bool equal = std::fabs(a - b)/(std::fabs(a)+std::fabs(b)+1) < 0.0001; // 7 digits of precision - 2
#if 0
  if (!equal) {
    printf("a=%.7e, b=%.7e, a-b=%.7e, denom=%.7e, frac=%.7e\n",
      a,
      b,
      std::fabs(a - b),
      (std::fabs(a) + std::fabs(b) + 1),
      std::fabs(a - b) / (std::fabs(a) + std::fabs(b) + 1)
      );
  }
#endif
  return equal;
}
template< >
bool almostEqual(double a, double b) {
  return std::fabs(a - b) / ( std::fabs(a) + std::fabs(b)+1 ) < 0.000000000001; // 15 digits of precision - 2
}
template< >
bool almostEqual( CobaltComplexFloat a, CobaltComplexFloat b) {
  return almostEqual(a.s[0], b.s[0]) && almostEqual(a.s[1], b.s[1]);
}
template< >
bool almostEqual(CobaltComplexDouble a, CobaltComplexDouble b) {
  return almostEqual(a.s[0], b.s[0]) && almostEqual(a.s[1], b.s[1]);
}

/*******************************************************************************
* Complex Conjugate
******************************************************************************/
template< >
void complexConjugate(float & v) {}
template< >
void complexConjugate(double & v) {}
template< >
void complexConjugate( CobaltComplexFloat & v) {
  v.s[1] = -v.s[1];
}
template< >
void complexConjugate(CobaltComplexDouble & v) {
  v.s[1] = -v.s[1];
}

} // end namespace Cobalt
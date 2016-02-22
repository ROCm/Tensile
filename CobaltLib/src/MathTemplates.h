#ifndef MATH_TEMPLATES_H
#define MATH_TEMPLATES_H


/*******************************************************************************
 * Zero Templates
 ******************************************************************************/
template< typename T> T getZero() { return static_cast<T>(0); };
template<> float getZero<float>() { return 0.f; };
template<> double getZero<double>() { return 0.0; };
template<> CobaltComplexFloat getZero<CobaltComplexFloat>() {
  CobaltComplexFloat zero = {0.f, 0.f};
  return zero;
};
template<> CobaltComplexDouble getZero<CobaltComplexDouble>() {
  CobaltComplexDouble zero = {0.0, 0.0};
  return zero;
};


/*******************************************************************************
 * One Templates
 ******************************************************************************/
template< typename T> T getOne() { return static_cast<T>(1); };
template<> float getOne<float>() { return 1.f; };
template<> double getOne<double>() { return 1.0; };
template<> CobaltComplexFloat getOne<CobaltComplexFloat>() {
  CobaltComplexFloat zero = {1.f, 0.f};
  return zero;
};
template<> CobaltComplexDouble getOne<CobaltComplexDouble>() {
  CobaltComplexDouble zero = {1.0, 0.0};
  return zero;
};


/*******************************************************************************
 * Multiply Templates
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB >
TypeC multiply( TypeA a, TypeB b ) {
  return static_cast<TypeC>( a * b );
};
// single
template< >
float multiply( float a, float b ) {
  return a*b;
};
// double
template< >
double multiply( double a, double b ) {
  return a*b;
};
// complex single
template< >
CobaltComplexFloat multiply( CobaltComplexFloat a, CobaltComplexFloat b ) {
  CobaltComplexFloat c;
  c.x = a.x*b.x - a.y*b.y;
  c.y = a.x*b.y + a.y*b.x;
  return c;
};
// complex double
template< >
CobaltComplexDouble multiply( CobaltComplexDouble a, CobaltComplexDouble b ) {
  CobaltComplexDouble c;
  c.x = a.x*b.x - a.y*b.y;
  c.y = a.x*b.y + a.y*b.x;
  return c;
};


/*******************************************************************************
 * Add Templates
 ******************************************************************************/
template< typename TypeC, typename TypeA, typename TypeB >
TypeC add( TypeA a, TypeB b ) {
  return static_cast<TypeC>( a + b );
};
// single
template< >
float add( float a, float b ) {
  return a+b;
};
// double
template< >
double add( double a, double b ) {
  return a+b;
};
// complex single
template< >
CobaltComplexFloat add( CobaltComplexFloat a, CobaltComplexFloat b ) {
  CobaltComplexFloat c;
  c.x = a.x+b.x;
  c.y = a.y+b.y;
  return c;
};
// complex double
template< >
CobaltComplexDouble add( CobaltComplexDouble a, CobaltComplexDouble b ) {
  CobaltComplexDouble c;
  c.x = a.x+b.x;
  c.y = a.y+b.y;
  return c;
};

#endif
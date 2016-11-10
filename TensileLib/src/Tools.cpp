/*******************************************************************************
 * Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/


#include "Tools.h"
#include <ctype.h>
#include <cmath>

namespace Tensile {

Timer::Timer() {
#ifdef WIN32
  QueryPerformanceFrequency( &frequency );
#else
  // nothing
#endif
}

void Timer::start() {
#ifdef WIN32
  QueryPerformanceCounter( &startTime );
#else
  clock_gettime( CLOCK_REALTIME, &startTime );
#endif
}

// returns elapsed time in seconds
double Timer::elapsed_sec() {
  return elapsed_us() / 1000000.0;
}
// returns elapsed time in seconds
double Timer::elapsed_ms() {
  return elapsed_us() / 1000.0;
}
double Timer::elapsed_us() {
  double return_elapsed_us;
#ifdef WIN32
  LARGE_INTEGER currentTime;
  QueryPerformanceCounter( &currentTime );
  return_elapsed_us = double(currentTime.QuadPart-startTime.QuadPart)/(frequency.QuadPart/1000000.0);
#else
  timespec currentTime;
  clock_gettime( CLOCK_REALTIME, &currentTime);
  return_elapsed_us = (currentTime.tv_sec - startTime.tv_sec)*1000000.0
      + (currentTime.tv_nsec - startTime.tv_nsec)/1000.0;
#endif
  return return_elapsed_us;
}


/*******************************************************************************
 * indent
 ******************************************************************************/
std::string indent(size_t level) {
  std::string indentStr = "";
  for (size_t i = 0; i < level; i++) {
    indentStr += " ";
  }
  return indentStr;
}



/*******************************************************************************
* factor 64-bit uint into 2 32-bit uints
******************************************************************************/
bool factor(size_t input, unsigned int & a, unsigned int & b) {
  double sqrt = std::sqrt(input);
  a = static_cast<unsigned int>(sqrt+0.5);
  for ( /*a*/; a >= 2; a--) {
    b = static_cast<unsigned int>(input / a);
    if (a*b == input) {
      return true;
    }
  }
  // plan B: return two numbers just larger than input
  a = static_cast<unsigned int>(sqrt + 1.5);
  b = a;
  return false;
}


void makeFileNameSafe( char *str ) {
  for ( ; *str != '\0'; str++ ) {
    if ( !isalnum( *str ) ) {
      *str = '_';
    }
  }
}

} // namespace


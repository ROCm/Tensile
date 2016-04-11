#include "Tools.h"

namespace Cobalt {

Timer::Timer() {
  QueryPerformanceFrequency( &frequency );
}

void Timer::start() {
  QueryPerformanceCounter( &startTime );
}

// returns elapsed time in seconds
double Timer::elapsed_sec() {
  LARGE_INTEGER currentTime;
  QueryPerformanceCounter( &currentTime );
  return double(currentTime.QuadPart-startTime.QuadPart)/frequency.QuadPart;
}
// returns elapsed time in seconds
double Timer::elapsed_ms() {
  LARGE_INTEGER currentTime;
  QueryPerformanceCounter( &currentTime );
  return double(currentTime.QuadPart-startTime.QuadPart)/(frequency.QuadPart/1000.0);
}
double Timer::elapsed_us() {
  LARGE_INTEGER currentTime;
  QueryPerformanceCounter( &currentTime );
  return double(currentTime.QuadPart-startTime.QuadPart)/(frequency.QuadPart/1000000.0);
}


/*******************************************************************************
 * indent
 ******************************************************************************/
std::string indent(size_t level) {
  std::string indentStr = "";
  for (size_t i = 0; i < level; i++) {
    indentStr += "  ";
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


} // namespace
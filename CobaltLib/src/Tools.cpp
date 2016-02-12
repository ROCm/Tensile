#include "Tools.h"

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
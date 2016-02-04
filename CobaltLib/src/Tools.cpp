#include "Tools.h"

Timer::Timer() {
  QueryPerformanceFrequency( &frequency );
}

void Timer::start() {
  QueryPerformanceCounter( &startTime );
}

// returns elapsed time in seconds
double Timer::elapsed() {
  LARGE_INTEGER currentTime;
  QueryPerformanceCounter( &currentTime );
  return double(currentTime.QuadPart-startTime.QuadPart)/frequency.QuadPart;
}
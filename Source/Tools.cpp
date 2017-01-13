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


#include "Tools.h"
#include <ctype.h>
#include <cmath>

TensileTimer::TensileTimer() {
#ifdef WIN32
  QueryPerformanceFrequency( &frequency );
#else
  // nothing
#endif
}

void TensileTimer::start() {
#ifdef WIN32
  QueryPerformanceCounter( &startTime );
#else
  clock_gettime( CLOCK_REALTIME, &startTime );
#endif
}

// returns elapsed time in seconds
double TensileTimer::elapsed_sec() {
  return elapsed_us() / 1000000.0;
}
// returns elapsed time in seconds
double TensileTimer::elapsed_ms() {
  return elapsed_us() / 1000.0;
}
double TensileTimer::elapsed_us() {
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




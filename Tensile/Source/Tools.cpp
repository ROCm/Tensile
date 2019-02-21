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

// QueryPerformanceFrequency is defined to return in units of counts per second
// QueryPerformanceCounter is defined to return in units of counts
// counts / (counts / seconds) == seconds, as the native windows time unit

// However, in practice the frequency of QueryPerformanceFrequency() has a resolution
// measured in nano-seconds on modern hardware, so it possible to return meaningful
// information on the nano-second time scale

const double TensileTimer::billion = 1E9;
const double TensileTimer::million = 1E6;
const double TensileTimer::thousand = 1E3;
const double TensileTimer::reciprical_billion = 1E-9;
const double TensileTimer::reciprical_million = 1E-6;
const double TensileTimer::reciprical_thousand = 1E-3;

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
  clock_gettime(clock_type, &startTime );
#endif
}

// elapsed time in seconds
double TensileTimer::elapsed_sec() {
  return elapsed_ns() * reciprical_billion;
}

// elapsed time in milliseconds
double TensileTimer::elapsed_ms() {
  return elapsed_ns() * reciprical_million;
}

// elapsed time in microseconds
double TensileTimer::elapsed_us() {
  return elapsed_ns() * reciprical_thousand;
}

// elapsed time in microseconds
double TensileTimer::elapsed_ns() {
    double return_elapsed_ns = 0;
#ifdef WIN32
    LARGE_INTEGER currentTime;
    QueryPerformanceCounter(&currentTime);

    double delta_time = static_cast<double>(currentTime.QuadPart - startTime.QuadPart);
    delta_time /= frequency.QuadPart;
    return_elapsed_ns = delta_time * billion;
#else
    timespec currentTime;
    clock_gettime(clock_type, &currentTime);

    // Commented out for subtle subtraction bug; explained below
    //return_elapsed_ns = (currentTime.tv_sec - startTime.tv_sec)*billion
    //    + (currentTime.tv_nsec - startTime.tv_nsec);

    // (currentTime.tv_nsec - startTime.tv_nsec) might be negative, if a 'second' boundary crossed and tv_nsec reset to 0
    // Convert to double type before subtracting to properly borrow from seconds when nano-seconds would be negative

    double d_startTime = static_cast<double>(startTime.tv_sec)*billion + static_cast<double>(startTime.tv_nsec);
    double d_currentTime = static_cast<double>(currentTime.tv_sec)*billion + static_cast<double>(currentTime.tv_nsec);
    return_elapsed_ns = d_currentTime - d_startTime;
#endif
    return return_elapsed_ns;
}

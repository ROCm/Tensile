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

#ifndef TOOLS_H
#define TOOLS_H

#include <string>
#ifdef WIN32
#include "Windows.h"
#else
#include <time.h>
#endif

namespace Tensile {

/*******************************************************************************
 * Timer
 ******************************************************************************/
class Timer {
public:
  Timer();
  void start();
  double elapsed_sec();
  double elapsed_ms();
  double elapsed_us();

private:
#ifdef WIN32
  LARGE_INTEGER startTime;
  LARGE_INTEGER frequency; 
#else
  timespec startTime;
#endif
};


/*******************************************************************************
 * xml tags for toString
 ******************************************************************************/
std::string indent(size_t level);

bool factor( size_t input, unsigned int & a, unsigned int & b);

void makeFileNameSafe( char *str );

} // namespace


#define tensileMin(a,b) (((a) < (b)) ? (a) : (b))
#define tensileMax(a,b) (((a) > (b)) ? (a) : (b))

#endif


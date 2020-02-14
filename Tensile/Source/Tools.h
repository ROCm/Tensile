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
#ifdef Tensile_RESUME_BENCHMARK
#include <fstream>
#endif
#ifdef WIN32
#include "Windows.h"
#else
#include <time.h>
#endif

/*******************************************************************************
 * Timer
 ******************************************************************************/
class TensileTimer {
public:
  TensileTimer();
  void start();
  double elapsed_sec();
  double elapsed_ms();
  double elapsed_us();
  double elapsed_ns();

  static const double billion;
  static const double million;
  static const double thousand;
  static const double reciprical_billion;
  static const double reciprical_million;
  static const double reciprical_thousand;

private:
#ifdef WIN32
  LARGE_INTEGER startTime;
  LARGE_INTEGER frequency; 
#else
    const clockid_t clock_type = CLOCK_MONOTONIC;
    //const clockid_t clock_type = CLOCK_MONOTONIC_RAW;
  timespec startTime;
#endif
};


#define tensileMin(a,b) (((a) < (b)) ? (a) : (b))
#define tensileMax(a,b) (((a) > (b)) ? (a) : (b))

#ifdef Tensile_RESUME_BENCHMARK
inline unsigned int countFileLines(const std::string& file_path){
    std::ifstream read_file;
    
    read_file.open(file_path, std::ios::in);
    if(read_file.fail()) return 0;

    unsigned int line_counts = 0;
    std::string tmp;
    while(getline(read_file, tmp)){
      ++line_counts;
    }

    read_file.close();
    return line_counts;
}
#endif

#endif


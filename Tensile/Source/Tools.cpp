/*******************************************************************************
 * Copyright 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *******************************************************************************/
#ifdef _WIN32
#include <winbase.h>
#endif

#include <cmath>
#include <ctype.h>
#include <stdlib.h>

#include "Tools.h"


/*******************************************************************************
 * Cross platform helpers
 ******************************************************************************/

const char* read_env(const char* env_var)
{
#ifdef _WIN32
    const DWORD nSize = _MAX_PATH;
    static thread_local char lpBuffer[nSize];
    DWORD len = GetEnvironmentVariableA(env_var, lpBuffer, nSize);
    if (len && len < nSize)
        return lpBuffer;
    else 
        return nullptr; // variable not found or longer than nSize
#else
    return std::getenv(env_var);
#endif
}


/*******************************************************************************
 * Timer
 ******************************************************************************/

const double TensileTimer::billion             = 1E9;
const double TensileTimer::million             = 1E6;
const double TensileTimer::thousand            = 1E3;
const double TensileTimer::reciprical_billion  = 1E-9;
const double TensileTimer::reciprical_million  = 1E-6;
const double TensileTimer::reciprical_thousand = 1E-3;

TensileTimer::TensileTimer()
{
}

void TensileTimer::start()
{
    auto now = std::chrono::steady_clock::now();
    m_startTime = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
}

// elapsed time in nanoseconds
double TensileTimer::elapsed_ns()
{
    auto now = std::chrono::steady_clock::now();
    auto currentTime = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

    double return_elapsed_ns = static_cast<double>(currentTime) - static_cast<double>(m_startTime);

    return return_elapsed_ns;
}

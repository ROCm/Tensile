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

#ifndef TENSILE_H
#define TENSILE_H
#include <stdio.h>

// OpenCL
#if Tensile_RUNTIME_LANGUAGE_OCL
#include "CL/cl.h"
#define TensileStatus cl_int
#define tensileStatusSuccess CL_SUCCESS
#define tensileStatusFailure -1
#define TensileComplexFloat cl_float2
#define TensileComplexDouble cl_double2
#define TensileHalf cl_half

// HIP
#else
#include <hip/hip_runtime.h>
#define TensileStatus hipError_t
#define tensileStatusSuccess hipSuccess
#define tensileStatusFailure hipErrorUnknown
#define TensileComplexFloat float2
#define TensileComplexDouble double2
#define TensileHalf __fp16

inline std::ostream& operator<<(std::ostream& os, const __fp16& dt)  
{  
   os << (float)(dt);
   return os;  
}

#endif

/*******************************************************************************
 * tensileSetup
 ******************************************************************************/
TensileStatus tensileSetup();

/*******************************************************************************
 * tensileTeardown
 ******************************************************************************/
TensileStatus tensileTeardown();

/*******************************************************************************
 * tensileCheckStatus
 ******************************************************************************/
#define tensileStatusCheck(RET) { \
  TensileStatus tensileCheckStatusTmp = RET; \
  if(tensileCheckStatusTmp != tensileStatusSuccess) { \
    printf("TensileStatusFailure %i on line %u of %s\n", \
        tensileCheckStatusTmp, __LINE__, __FILE__); \
  } }


#endif

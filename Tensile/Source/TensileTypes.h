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
#define tensileStatusAssertFailure -2
#define TensileComplexFloat cl_float2
#define TensileComplexDouble cl_double2
#define TensileHalf cl_half

// HIP
#else
#include <hip/hip_runtime.h>
#define TensileStatus hipError_t
#define tensileStatusSuccess hipSuccess
// FIXME - steal hip error encodings since rocBLAS expects hip error codes to be returned...
#define tensileStatusFailure hipErrorUnknown
#define tensileStatusAssertFailure hipErrorRuntimeOther
#define TensileComplexFloat float2
#define TensileComplexDouble double2
#define TensileHalf _Float16
#define TensileInt8x4 uint32_t
#define TensileInt32 int32_t

inline std::ostream& operator<<(std::ostream& os, const _Float16& dt)
{
   os << (float)(dt);
   return os;
}

#endif // HIP

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
    fprintf(stderr, "ERROR:  TensileStatusFailure %i on line %u of %s\n", \
        tensileCheckStatusTmp, __LINE__, __FILE__); \
    abort();\
  } }

template <int NumSizes, int LastSummationIdx, int Free0Idx>
class ProblemSizes {
public:
  using SizeType = unsigned int;

  // Constructor accepts variable number of sizes:
  template<typename... Ts>
  ProblemSizes(Ts... args) {
    init<NumSizes>(args...);
  }

  bool operator< (const ProblemSizes<NumSizes, LastSummationIdx, Free0Idx> & p) const
  {
    for (int i=0; i<NumSizes; i++) {
      if (p.sizes[i] < this->sizes[i])
        return true;
      else if (p.sizes[i] > this->sizes[i])
        return false;
      // if equal, continue to next index
    }
    return false; // get here if all indices are equal
  };

  bool operator== (const ProblemSizes<NumSizes, LastSummationIdx, Free0Idx> & p) const
  {
    for (int i=0; i<NumSizes; i++) {
      if (p.sizes[i] != this->sizes[i])
        return false;
    }
    return true;
  };

  SizeType lastSummationSize() const { return sizes[LastSummationIdx]; };
  SizeType free0Size() const { return sizes[Free0Idx]; };
  SizeType free1Size() const { return sizes[1]; };

  size_t hash() const {
    size_t h=0;
    for (int i=0; i<NumSizes; i++) {
      h ^= sizes[i] + 0x9b9773e99e3779b9 + (h<<6) + (h>>2);
    }
    return h;
  }

private:
  template<int I, typename T>
  void init (T v) {
    sizes[NumSizes-I] = v;
  }

  template<int I, typename T, typename... Ts>
  void init (T v, Ts... args )
  {
    sizes[NumSizes-I] = v;
    init<I-1> (args...);
  }


private:
  // Data members:
  SizeType sizes[NumSizes];
};

template <class Object>
struct ObjectHasher {
    size_t operator()(const Object &o ) const
    {
      return o.hash();
    }
};


// These are assertions used to generate the solution
// Must be checked by the runtime before launchin the solution
struct AssertionProperties {
  AssertionProperties(unsigned x_summationElementMultiple,
                      unsigned x_free0ElementMultiple,
                      unsigned x_free1ElementMultiple
                      )
    : summationElementMultiple(x_summationElementMultiple),
      free0ElementMultiple(x_free0ElementMultiple),
      free1ElementMultiple(x_free1ElementMultiple)
     {}

  template<class ProblemSizes>
  AssertionProperties(const ProblemSizes &p) {
    summationElementMultiple = 1; // problem summation element multiple
    auto sumSize = p.lastSummationSize();
    if ((sumSize & 0x7) == 0) summationElementMultiple=8;
    else if ((sumSize & 0x3) == 0) summationElementMultiple=4;
    else if ((sumSize & 0x1) == 0) summationElementMultiple=2;

    auto free0Size = p.free0Size();
    free0ElementMultiple = 1; // problem free0 element multiple
    if ((free0Size & 0x7) == 0) free0ElementMultiple=8;
    else if ((free0Size & 0x3) == 0) free0ElementMultiple=4;
    else if ((free0Size & 0x1) == 0) free0ElementMultiple=2;

    auto free1Size = p.free1Size();
    free1ElementMultiple = 1; // problem free1 element multiple
    if ((free1Size & 0x7) == 0) free1ElementMultiple=8;
    else if ((free1Size & 0x3) == 0) free1ElementMultiple=4;
    else if ((free1Size & 0x1) == 0) free1ElementMultiple=2;
  };

  // Returns True if this AsssertionProperties meet the requirements for the specified soluition
  // (this object represents the 'Problem')
  bool validForSolution(const AssertionProperties &solutionAssertions) {
    return (this->summationElementMultiple >= solutionAssertions.summationElementMultiple) &&
           (this->free0ElementMultiple >= solutionAssertions.free0ElementMultiple) &&
           (this->free1ElementMultiple >= solutionAssertions.free1ElementMultiple);
  }

  unsigned summationElementMultiple;
  unsigned free0ElementMultiple;
  unsigned free1ElementMultiple;
};


// solution info
template <typename F>
struct SolutionInfo {
  F                       functionPtr;
  const char *            name;
  AssertionProperties     assertions;
};


#endif

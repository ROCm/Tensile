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


// Base template for ProblemSizes
// -  stores the sizes
// -  supports hash generation and comparison for lookup
// TensileCreateLibrary.cpp will create a typedef for each specific problem, ie
// ProblemSizes_Cijk_Ailk_Bljk_SB.
// Some templates below use a parm called ProblemSizesType which can be any of these
// generated types.
template <int NumSizes>
class ProblemSizes {
public:
  using SizeType = unsigned int;

  // Constructor accepts variable number of sizes:
  template<typename... Ts>
  ProblemSizes(Ts... args) {
    init<NumSizes>(args...);
  }

  bool operator< (const ProblemSizes<NumSizes> & p) const
  {
    for (int i=0; i<NumSizes; i++) {
      if (p._sizes[i] < this->_sizes[i])
        return true;
      else if (p._sizes[i] > this->_sizes[i])
        return false;
      // if equal, continue to next index
    }
    return false; // get here if all indices are equal
  };

  bool operator== (const ProblemSizes<NumSizes> & p) const
  {
    for (int i=0; i<NumSizes; i++) {
      if (p._sizes[i] != this->_sizes[i])
        return false;
    }
    return true;
  };

  size_t hash() const {
    size_t h=0;
    for (int i=0; i<NumSizes; i++) {
      h ^= _sizes[i] + 0x9b9773e99e3779b9 + (h<<6) + (h>>2);
    }
    return h;
  }

  const SizeType *sizes() const { return _sizes;};
  int numSizes() const { return NumSizes;};

  std::ostream &print(std::ostream &os) const {
    for (int i=0; i<NumSizes; i++) {
      if (i != 0) {
        os << ", ";
      };
      os << _sizes[i];
    };
    return os;
  };


private:
  template<int I, typename T>
  void init (T v) {
    _sizes[NumSizes-I] = v;
  }

  template<int I, typename T, typename... Ts>
  void init (T v, Ts... args )
  {
    _sizes[NumSizes-I] = v;
    init<I-1> (args...);
  }

private:
  // Data members:
  SizeType _sizes[NumSizes];
};

//-------------
// Distance Functions between two problem sizes - used to find nearest neighbor or closest solution
// Assumes p1 and p2 have same number of sizes
//-------------

template <class ProblemSizesType>
struct RatioDistance {
  double operator() (const ProblemSizesType &p1, const ProblemSizesType &p2) const
  {
    double distance = 1.0;
    for (int i=0; i<p1.numSizes(); i++) {
      distance += fabs(log(double(p1.sizes()[i])/double(p2.sizes()[i])));
    }
    return distance;
  }
};

template <class ProblemSizesType>
struct ManhattanDistance {
  double operator() (const ProblemSizesType &p1, const ProblemSizesType &p2) const
  {
    double distance = 0;
    for (int i=0; i<p1.numSizes(); i++) {
      distance += fabs(double(p1.sizes()[i]) - double(p2.sizes()[i]));
    }
    return distance;
  }
};


template <class ProblemSizesType>
struct EuclideanDistance {
  double operator() (const ProblemSizesType &p1, const ProblemSizesType &p2) const
  {
    double distance = 0;
    for (int i=0; i<p1.numSizes(); i++) {
      if (p1.sizes()[i] > p2.sizes()[i])
        distance += pow((p1.sizes()[i] - p2.sizes()[i]), 2);
      else
        distance += pow((p2.sizes()[i] - p1.sizes()[i]), 2);
    }
// distance = sqrt(distance);
    return distance;
  }
};

template <class ProblemSizesType>
struct RandomDistance {
  double operator() (const ProblemSizesType &p1, const ProblemSizesType &p2) const
  {
    return double(rand()) / RAND_MAX;
  }
};

template <class Object>
struct ObjectHasher {
    size_t operator()(const Object &o ) const
    {
      return o.hash();
    }
};

class ProblemProperties
{
public:
  ProblemProperties(const std::vector<int> &indicesFree,
                    const std::vector<int> &indicesSummation,
                    const std::vector<int> &indicesBatch)
    : _indicesFree(indicesFree),
      _indicesSummation(indicesSummation),
      _indicesBatch(indicesBatch)
  {
  }

  int lastSummationIdx() const { return _indicesSummation.back(); };
  int free0Idx() const { return _indicesFree[0]; };
  bool isBatchIdx(int idx) const {
    return std::find(_indicesBatch.begin(), _indicesBatch.end(), idx) != _indicesBatch.end();
  };

private:
  const std::vector<int> _indicesFree;
  const std::vector<int> _indicesSummation;
  const std::vector<int> _indicesBatch;
};


// These are assertions used to generate the solution
// Must be checked by the runtime before launchin the solution
struct AssertionProperties {

  // Constructor used in solution tables
  AssertionProperties(unsigned summationElementMultiple,
                      unsigned free0ElementMultiple,
                      int approxSize)
    : _summationElementMultiple(summationElementMultiple),
      _free0ElementMultiple(free0ElementMultiple),
      _approxSize(approxSize)
     {}

  // Constructor used to compute assertions for a specified problem size
  template<class ProblemSizes>
  AssertionProperties(const ProblemSizes &p, const ProblemProperties *props) {
    _summationElementMultiple = 1; // problem summation element multiple
    auto sumSize = p.sizes()[props->lastSummationIdx()];
    if ((sumSize & 0x7) == 0) _summationElementMultiple=8;
    else if ((sumSize & 0x3) == 0) _summationElementMultiple=4;
    else if ((sumSize & 0x1) == 0) _summationElementMultiple=2;

    auto free0Size = p.sizes()[props->free0Idx()];
    _free0ElementMultiple = 1; // problem free0 element multiple
    if ((free0Size & 0x7) == 0) _free0ElementMultiple=8;
    else if ((free0Size & 0x3) == 0) _free0ElementMultiple=4;
    else if ((free0Size & 0x1) == 0) _free0ElementMultiple=2;

    bool allBelow1 = true;
    bool allBelow32 = true;
    for (int si=0; si!=p.numSizes(); si++) {
      if (!props->isBatchIdx(si)) {
        auto size = p.sizes()[si];
        if (size > 32)
          allBelow32 = false;
        if (size > 1)
          allBelow1 = false;
      }
    }
    if (allBelow1)
      _approxSize = 1; // really small
    else if (allBelow32)
      _approxSize = 2; // still small
    else
      _approxSize = 99; // big enough
  };

  // Returns True if this AsssertionProperties meet the requirements for the specified soluition
  // (this object represents the 'Problem')
  bool validForSolution(const AssertionProperties &solutionAssertions) {
    return (this->_summationElementMultiple >= solutionAssertions._summationElementMultiple) &&
           (this->_free0ElementMultiple >= solutionAssertions._free0ElementMultiple) &&
           ((this->_approxSize) >= solutionAssertions._approxSize);
  }

  unsigned _summationElementMultiple;
  unsigned _free0ElementMultiple;;
  int      _approxSize;
};


// solution info
template <typename F>
struct SolutionInfo {
  F                       functionPtr;
  const char *            name;
  AssertionProperties     assertions;
};


#endif

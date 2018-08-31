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


// SolutionMapper:
// Efficiently map problems to exact or best solution
// Supports efficient searching and various algorithms to find
// a 'best match' from the available solutions
template <class ProblemParmsType, typename SolutionInfoType>
class SolutionMapper {
  // Problem to Solution mapping:
  typedef std::pair<const ProblemParmsType, int>  PtoS;

public:
  SolutionMapper(const SolutionInfoType *solutionTable, size_t numSolutions,
                 const PtoS *embeddedExactTable, size_t numExacts)
     : _solutionTable(solutionTable), _numSolutions(numSolutions)
  {
    for (size_t i=0; i<numExacts; i++) {
      auto &p = embeddedExactTable[i].first;  //problem
      auto si = embeddedExactTable[i].second; //solutionIndex
      auto const &solution = solutionTable[si];

      if (AssertionProperties(p).validForSolution(solution.assertions)) {
        _exactVector.push_back(embeddedExactTable[i]);
        _map.insert({p, si});
      } else {
        // TODO - ideally these should never make it into the exact table in the first place,
        // need to check in python land
        //printf ("warning: removing bogus exact problem (does not meet assertion requirements for solution)\n");
      }
    }
  }

  // Returns integer solutionIdx if exact match is found else -1
  int findExactMatch(const ProblemParmsType &p) const
  {
    auto fiter = _map.find(p);
    if (fiter != _map.end()) {
      return fiter->second;
    } else {
      return -1;
    }
  }

  // Iterates through all known exact matching and finds the 'closest' match.
  int findNearestMatch(const ProblemParmsType &p) const
  {
    AssertionProperties pa(p);
  };

private:
  const SolutionInfoType  *_solutionTable;
  size_t                   _numSolutions;

  // Two different structures supporting mapping from problems to solutions:
  // Map for fast exact lookups and a vector for fast walking
  std::map<const ProblemParmsType, int> _map;
  std::vector<PtoS>                     _exactVector;
};


// For the specified matrix dimensions, find a best-fit GEMM kernel
// This routine does perform any auto-tuning or benchmarking
template <class ProblemParmsType,typename SolutionInfoType>
int find_algorithm_static(
    const ProblemParmsType &p,
    const SolutionMapper<ProblemParmsType, SolutionInfoType> &smapper)
{

  int solutionIdx = smapper.findExactMatch(p);
  //printf ("find_algorithm_static, solutionIdx=%d\n", solutionIdx);
  return solutionIdx;

  // eventually can try a nearest neighbor search here or something fancier
  // to find a match
}

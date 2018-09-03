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

#include <limits>

#define DEBUG_SM 0

// SolutionMapper:
// Efficiently map problems to exact or best solution
// Supports efficient searching and various algorithms to find
// a 'best match' from the available solutions
template <class ProblemParmsType, typename SolutionInfoType>
class SolutionMapper {
  // Problem to Solution mapping:
  typedef std::pair<const ProblemParmsType, int>  PtoS;

  enum Algo {PickNoneAlgo= -1, RandomAlgo= -2, RatioDistanceAlgo= -3, EuclideanDistanceAlgo= -4, ManhattanDistanceAlgo= -5};

public:
  SolutionMapper(const SolutionInfoType *solutionTable, size_t numSolutions,
                 const PtoS *embeddedExactTable, size_t numExacts,
                 const ProblemProperties *props)
     : _solutionTable(solutionTable), _numSolutions(numSolutions), _props(props), _findAlg(EuclideanDistanceAlgo)
  {
    for (size_t i=0; i<numExacts; i++) {
      auto &p = embeddedExactTable[i].first;  //problem
      auto si = embeddedExactTable[i].second; //solutionIndex
      auto const &solution = solutionTable[si];

      if (AssertionProperties(p,props).validForSolution(solution.assertions)) {
        _exactVector.push_back(embeddedExactTable[i]);
        _map.insert({p, si});
      } else {
        // TODO - ideally these should never make it into the exact table in the first place,
        // need to check in python land
        if (DEBUG_SM)
          std::cout << "warning: removing bogus exact problem (does not meet assertion requirements for solution)\n";
      }
    }

    const char *alg = std::getenv("TENSILE_FIND_ALGO"); //See Algo or >=0 specified specific solution
    if (alg) {
      _findAlg = strtol(alg,nullptr,0);
    }
    if (DEBUG_SM & 0x1)
      printf ("TENSILE_FIND_ALGO= %d (%s)\n", _findAlg, algoString(_findAlg));
  }

#define CASE_STRING(X)  case X: return(#X)
  const char *algoString(int algo) o
  {
    if (algo >= 0) {
      return "Explicitly-Selected";
    }
    switch (algo) {
      CASE_STRING(PickNoneAlgo);
      CASE_STRING(RandomAlgo);
      CASE_STRING(RatioDistanceAlgo);
      CASE_STRING(EuclideanDistanceAlgo);
      CASE_STRING(ManhattanDistanceAlgo);
      default: return ("Unknown Algo");
    };
  };

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
  template <class DistanceFunction>
  int findNearestMatch(const ProblemParmsType &p, DistanceFunction distanceF) const
  {
    AssertionProperties pa(p,_props);

    auto bestIter = _exactVector.end();
    double bestDistance = std::numeric_limits<double>::max();

    for (auto iter = _exactVector.begin(); iter != _exactVector.end(); iter++) {
      auto tableP = iter->first;
      auto solution = getSolution(iter->second);
      if (pa.validForSolution(solution.assertions)) {
        double distance = distanceF(p, tableP);
        if (DEBUG_SM & 0x2)
          iter->first.print(std::cout);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIter = iter;
          if (DEBUG_SM & 0x2)
            std::cout << " distance=" << distance << " **newBest**" << "\n";
        } else {
          //std::cout << " distance=" << distance << "\n";
        }
      }
    }

    if (bestIter != _exactVector.end())
      return bestIter->second;
    else
      return -1; // if no solutions in the table
  };

  int findNearestMatchWithAlg(const ProblemParmsType &p) const
  {
    if (_findAlg >= 0) {
      if (_findAlg < _numSolutions) {
        return _findAlg; // user specified a specific algorithm
      }
    }
    switch (_findAlg) {
      case PickNoneAlgo: // Fall through to range logic
        return -1;
      case RandomAlgo:
        return findNearestMatch (p, RandomDistance<decltype(p)>());
      case EuclideanDistanceAlgo:
        return findNearestMatch (p, EuclideanDistance<decltype(p)>());
      case ManhattanDistanceAlgo:
        return findNearestMatch (p, ManhattanDistance<decltype(p)>());
      case RatioDistanceAlgo:
      default:
        return findNearestMatch (p, RatioDistance<decltype(p)>());
        break;
    }

    return -1;
  }

private:
  const SolutionInfoType getSolution(int solutionIdx) const { return _solutionTable[solutionIdx]; };

private:
  const SolutionInfoType  *_solutionTable;
  size_t                   _numSolutions;

  const ProblemProperties *_props;

  // Two different structures supporting mapping from problems to solutions:
  // Map for fast exact lookups and a vector for fast walking
  std::map<const ProblemParmsType, int> _map;
  std::vector<PtoS>                     _exactVector;

  int                    _findAlg;
};


// For the specified matrix dimensions, find a best-fit GEMM kernel
// This routine does perform any auto-tuning or benchmarking
template <class ProblemParmsType,typename SolutionInfoType>
int find_algorithm_static(
    const ProblemParmsType &p,
    const SolutionMapper<ProblemParmsType, SolutionInfoType> &smapper)
{
  int solutionIdx = smapper.findExactMatch(p);
  //
  if (solutionIdx == -1) {
    //solutionIdx = smapper.findNearestMatch (p, RatioDistance<decltype(p)>());
    solutionIdx = smapper.findNearestMatchWithAlg (p);
    if (DEBUG_SM)
      std::cout << "find_algorithm_static picked best-fit solutionIdx=" << solutionIdx << "\n";
  } else {
    if (DEBUG_SM)
      std::cout << "find_algorithm_static picked exact solutionIdx=" << solutionIdx << "\n";
  }


  return solutionIdx;
}

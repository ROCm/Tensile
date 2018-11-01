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

#pragma once

#include <limits>

// Set to non-zero to debug the solution mapper
#define DEBUG_SM 0

class SolutionMapperBase;

//--------------------
// Maps from deviceId to appropriate solution
//
// One of these per problem type.
// TODO - move to SolutionMapper.cpp
class DeviceSolutionMapper
{
public:
  DeviceSolutionMapper()  {
    int numDevices;
    hipGetDeviceCount(&numDevices);
    _mapper.resize(numDevices);
    for (int i=0; i<numDevices; i++) {
      _mapper[i] = nullptr;
    }
  };

  int addMapper(const std::string &mapperName, SolutionMapperBase *mapper)
  {
    int matches = 0;

    // walk through each device and if name matches then
    for (int i=0; i<_mapper.size(); i++) {
      hipDeviceProp_t deviceProperties;
      hipGetDeviceProperties(&deviceProperties, i);
      std::string deviceName(deviceProperties.name);
      if (DEBUG_SM) {
        printf("compare #%d:%s == %s\n", i, deviceName.c_str(), mapperName.c_str());
      }
      if ((deviceName == mapperName) ||
          ((mapperName == "fallback" || mapperName == "Device 0000") && _mapper[i] == nullptr)) {
        matches++;
        _mapper[i] = mapper;
          if (DEBUG_SM) {
            printf ("  match->%d\n", matches);
          }
      }
    }
    return matches;
  }

  SolutionMapperBase *mapper()
  {
    int deviceId;
    hipGetDevice(&deviceId);
    return _mapper[deviceId];
  }

private:
  // Index is deviceId, points at the mapper to use for that device.
  std::vector <SolutionMapperBase*> _mapper;;
};


class SolutionMapperBase {
public:
  // Runtime information for the solution:
  //   - const pointer to the info including function pointers, name, and assertion requirements
  //   - runtime information including status of the necessary code object(s) in memory
  struct SolutionRuntime {
    SolutionRuntime() : _info(nullptr) {};

    const SolutionInfo *_info;
    SolutionLock _lock;
    bool isValid() const { return _info != nullptr; };
  };
protected:
  enum Algo {PickNoneAlgo= -1, RandomAlgo= -2, RatioDistanceAlgo= -3, EuclideanDistanceAlgo= -4, ManhattanDistanceAlgo= -5};
};


// SolutionMapper:
// Efficiently map problems to exact or best solution
// Supports efficient searching and various algorithms to find
// a 'best match' from the available solutions
// This provides mappings for a single device type
template <class ProblemDimsType, class ProblemKeyType>
class SolutionMapper : public SolutionMapperBase {
  // Problem to Solution mapping:
  typedef std::pair<const ProblemKeyType, int>  PtoS;

public:
  SolutionMapper(const std::string &name, const std::vector<std::string> &deviceNames,
                 DeviceSolutionMapper *deviceSolutionMapper,
                 const SolutionInfo *solutionTable, size_t numSolutions,
                 const PtoS *embeddedExactTable, size_t numExacts,
                 const ProblemProperties *props)
     : _name(name), _props(props), _numSolutions(numSolutions), _findAlg(EuclideanDistanceAlgo)
  {
    int used=0; // how many devices are using this solution mapper:
    for (auto iter=deviceNames.begin(); iter!=deviceNames.end(); iter++) {
      used += deviceSolutionMapper->addMapper(*iter, this);
    }

    if (DEBUG_SM) {
      printf ("info: mapper init - %s was used in %d devices\n", name.c_str(), used);
    }
    if (used==0) {
      if (DEBUG_SM) {
        printf ("info: **skipping mapper init - no devices of type: %s found\n", name.c_str());
      }
      return;
    }

    _solutionTable = new SolutionRuntime[numSolutions];

    for (size_t i=0; i<numSolutions; i++) {
      _solutionTable[i]._info = &solutionTable[i];
    }

    for (size_t i=0; i<numExacts; i++) {
      auto &pkey = embeddedExactTable[i].first;
      auto solutionIdx = embeddedExactTable[i].second;
      auto const &solution = solutionTable[solutionIdx];

      _exactVector.push_back(embeddedExactTable[i]);
      _exactMap.insert({pkey, solutionIdx});
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
  int findExactMatch(const AssertionProperties  &pa,
                     const ProblemKeyType &pkey) const
  {
    auto fiter = _exactMap.find(pkey);
    //if (fiter != _exactMap.end() &&
    //    pa.validForSolution(getSolution(fiter->second)._info->_assertions)) {
    if (fiter != _exactMap.end()) {
      if (pa.validForSolution(getSolution(fiter->second)._info->_assertions)) {
        return fiter->second;
      } else {
        //printf ("Possible exact match %d failed assertion requirements\n", fiter->second);
        return -1;
      }
    } else {
      return -1;
    }
  }

  // Iterates through all known exact matching and finds the 'closest' match.
  template <class DistanceFunction>
  int findNearestMatch(const AssertionProperties &pa,
                       const ProblemKeyType &pkey,
                       DistanceFunction distanceF) const
  {

    auto bestIter = _exactVector.end();
    double bestDistance = std::numeric_limits<double>::max();

    for (auto iter = _exactVector.begin(); iter != _exactVector.end(); iter++) {
      auto tableP = iter->first;
      auto solutionInfo= getSolution(iter->second)._info;
      if (pa.validForSolution(solutionInfo->_assertions)) {
        double distance = distanceF(pkey, tableP);
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

  int findNearestMatchWithAlg(const AssertionProperties &pa, const ProblemKeyType &pkey) const
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
        return findNearestMatch (pa, pkey, RandomDistance<decltype(pkey)>());
      case EuclideanDistanceAlgo:
        return findNearestMatch (pa, pkey, EuclideanDistance<decltype(pkey)>());
      case ManhattanDistanceAlgo:
        return findNearestMatch (pa, pkey, ManhattanDistance<decltype(pkey)>());
      case RatioDistanceAlgo:
      default:
        return findNearestMatch (pa, pkey, RatioDistance<decltype(pkey)>());
        break;
    }

    return -1;
  }

  // For the specified matrix dimensions, find a best-fit GEMM kernel
  // This routine does perform any auto-tuning or benchmarking
  int findAlgorithmStatic(const ProblemDimsType &pdims, bool exactOnly)
  {
    ProblemKeyType pkey(pdims);

    // Assertions that we can make based on the problem dims,
    // for example summation is some int multiple or macro-tile bounds are <32bits
    AssertionProperties pa(pdims,_props);

    std::lock_guard<std::mutex> lockGuard(_cachedMutex);
    auto fiter = _cachedLookups.find(pkey);
    if (fiter != _cachedLookups.end()) {
      if (DEBUG_SM)
        std::cout << "findAlgorithmStatic hit in cache, " << fiter->second << "\n";
      return fiter->second;

    } else {
      // Less frequently come here, this is only first time problem size is seen.
      int solutionIdx = findExactMatch(pa, pkey);
      if (solutionIdx == -1 and !exactOnly) {
        solutionIdx = findNearestMatchWithAlg (pa, pkey);
        if (DEBUG_SM)
          std::cout << "findAlgorithmStatic picked nearest-match solutionIdx=" << solutionIdx << "\n";
      } else {
        if (DEBUG_SM)
          std::cout << "findAlgorithmStatic picked exact solutionIdx=" << solutionIdx << "\n";
      }

      // Save problem->solutionIdx mapping so future lookups are fast:
      if (solutionIdx != -1) {
        _cachedLookups.insert({pkey, solutionIdx});
      }

      return solutionIdx;
    }
  }

  const SolutionRuntime &getSolution(int solutionIdx) const {
    //printf ("getSolution for solutionIdx=%d\n", solutionIdx);
    return _solutionTable[solutionIdx];
  };

  const SolutionRuntime &cacheSolution(const ProblemDimsType &pdims, int solutionIdx) {
    {
      if (solutionIdx != -1) {
        ProblemKeyType pkey(pdims);
        std::lock_guard<std::mutex> lockGuard(_cachedMutex);
        auto fiter = _cachedLookups.find(pkey);
        if (fiter != _cachedLookups.end()) {
          _cachedLookups.insert({pkey, solutionIdx});
        }
      }
    }
    return _solutionTable[solutionIdx];
  };
  const std::string name() const { return _name; };

private:
  const std::string        _name;
  const ProblemProperties *_props;

  SolutionRuntime *   _solutionTable;
  size_t              _numSolutions;

  // Two different structures supporting mapping from problems to solutions:
  // Map for fast exact lookups and a vector for fast walking
  std::map<const ProblemKeyType, int> _exactMap;
  std::vector<PtoS>                   _exactVector;

  std::mutex                          _cachedMutex;
  std::map<const ProblemKeyType, int> _cachedLookups;

  // Algorithm that should be used to find nearest match - See Algo enum
  int                                 _findAlg;
};







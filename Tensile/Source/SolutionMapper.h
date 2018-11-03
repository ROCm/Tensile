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

#include <limits:
/*******************************************************************************
 * Functions to map from ProblemDims to the best available solution
 *   - Provides efficient hash tables for lookup with thread-safe access
 *   - Defines several different static lookup functions, and can be extended
 *     for more sophisticated algorithms
 ******************************************************************************/

// Set to non-zero to debug the solution mapper
// 0x1 = informational debug
// 0x2 = deeper debug including distance winners
// 0x4 = print all distance calcs
// DEBUG_SM sets compile-time default - also can use TENSILE_DB env var with same encoding
#define DEBUG_SM 0

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

//--------------------
// Maps from deviceId to appropriate solution
//
// One of these per problem type.
// TODO - move to SolutionMapper.cpp
class MasterSolutionMapper
{
public:
  MasterSolutionMapper()  {
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
      //  printf("compare #%d:%s == %s\n", i, deviceName.c_str(), mapperName.c_str());
      if ((deviceName == mapperName) ||
          ((mapperName == "fallback" || mapperName == "Device 0000") && _mapper[i] == nullptr)) {
        matches++;
        _mapper[i] = mapper;
        //printf ("  match->%d\n", matches);
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
                 MasterSolutionMapper *masterSolutionMapper,
                 const SolutionInfo *solutionTable, size_t numSolutions,
                 const PtoS *embeddedExactTable, size_t numExacts,
                 const ProblemType *problemType)
     :  _name(name), _problemType(problemType), _numSolutions(numSolutions),
        _findAlg(EuclideanDistanceAlgo), _db(DEBUG_SM)
  {
    int used=0; // how many devices are using this solution mapper:
    for (auto iter=deviceNames.begin(); iter!=deviceNames.end(); iter++) {
      used += masterSolutionMapper->addMapper(*iter, this);
    }

    if (_db & 0x8) {
      printf ("info: mapper init - %s was used in %d devices\n", name.c_str(), used);
    }
    if (used==0) {
      if (_db & 0x8) {
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

    const char *db = std::getenv("TENSILE_DB");
    if (db) {
      _db = strtol(db,nullptr,0);
    }

    const char *alg = std::getenv("TENSILE_FIND_ALG"); // If <0 see Algo enumeration, or >=0 specified specific solution index
    if (alg) {
      _findAlg = strtol(alg,nullptr,0);
    }
    if (_db & 0x1)
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
  int findExactMatch(const ProblemProperties  &pa,
                     const ProblemKeyType &pkey) const
  {
    auto fiter = _exactMap.find(pkey);
    if (fiter != _exactMap.end()) {
      if (pa.validForSolution(getSolution(fiter->second)->_info->_assertionRequirements)) {
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
  int findNearestMatch(const ProblemProperties &pa,
                       const ProblemKeyType &pkey,
                       DistanceFunction distanceF) const
  {

    auto bestIter = _exactVector.end();
    double bestDistance = std::numeric_limits<double>::max();

    for (auto iter = _exactVector.begin(); iter != _exactVector.end(); iter++) {
      auto tableP = iter->first;
      auto solutionInfo= getSolution(iter->second)->_info;
      if (pa.validForSolution(solutionInfo->_assertionRequirements)) {
        double distance = distanceF(pkey, tableP);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestIter = iter;
          if (_db & 0x2) {
            std::cerr << " solutionIdx=" << iter->second << " pdims={";
            iter->first.print(std::cerr);
            std::cerr << "}";
            std::cerr << " distance=" << distance << "        <------------- newBest" << "\n";
          }
        } else {
          if (_db & 0x4) {
            std::cerr << " solutionIdx=" << iter->second << " pdims={";
            iter->first.print(std::cerr);
            std::cerr << "}";
            std::cerr << " distance=" << distance << "\n";
          }
        }
      }
    }

    if (bestIter != _exactVector.end())
      return bestIter->second;
    else
      return -1; // if no solutions in the table
  };

  int findNearestMatchWithAlg(const ProblemProperties &pa, const ProblemKeyType &pkey) const
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
  int findAlgorithmStatic(const ProblemDimsType &pdims)
  {
    ProblemKeyType pkey(pdims);

    // Assertions that we can make based on the problem dims,
    // for example summation is some int multiple or macro-tile bounds are <32bits
    ProblemProperties pa(pdims,_problemType);

    std::lock_guard<std::mutex> lockGuard(_cachedMutex);
    auto fiter = _cachedLookups.find(pkey);
    if (fiter != _cachedLookups.end()) {
      if (_db & 0x1)
        std::cerr << "findAlgorithmStatic hit in cache solutionIdx=" << fiter->second << "\n";
      return fiter->second;

    } else {
      // Less frequently come here, this is only first time problem size is seen.
      int solutionIdx = findExactMatch(pa, pkey);
      if (solutionIdx == -1) {
        solutionIdx = findNearestMatchWithAlg (pa, pkey);
        if (_db & 0x1)
          std::cerr << "findAlgorithmStatic picked nearest-match solutionIdx=" << solutionIdx << "\n";
      } else {
        if (_db & 0x1)
          std::cerr << "findAlgorithmStatic picked exact solutionIdx=" << solutionIdx << "\n";
      }

      // Save problem->solutionIdx mapping so future lookups are fast:
      if (solutionIdx != -1) {
        _cachedLookups.insert({pkey, solutionIdx});
      }

      return solutionIdx;
    }
  }

  SolutionRuntime *getSolution(int solutionIdx) const {
    //printf ("getSolution for solutionIdx=%d\n", solutionIdx);
    return &_solutionTable[solutionIdx];
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
  const ProblemType        *_problemType;

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

  // Debug print control:
  int                                 _db;
};

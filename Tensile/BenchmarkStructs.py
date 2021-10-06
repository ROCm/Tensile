################################################################################
# Copyright 2016-2021 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from copy import copy, deepcopy
from .Common import print1, print2, printWarning, defaultSolution, \
    defaultBenchmarkFinalProblemSizes, defaultBatchedBenchmarkFinalProblemSizes, \
    defaultBenchmarkCommonParameters, hasParam, printExit, \
    validParameters, globalParameters
from .SolutionStructs import Solution, ProblemType, ProblemSizes


def getDefaultsForMissingParameters(parameterConfigurationList, defaultParameters):
  """Temp doc"""
  benchmarkParameters = []
  for paramDict in defaultParameters:
    for paramName in paramDict:
      if not hasParam(paramName, parameterConfigurationList) \
          or paramName == "ProblemSizes":
        benchmarkParameters.append(paramDict)
  return benchmarkParameters

def checkForValidParameters(params, validParameterNames):
  """Temp doc"""
  for paramName in params:
    if paramName in ["ProblemSizes"]:
      continue
    else:
      if paramName not in validParameterNames:
        raise RuntimeError("Invalid parameter name: %s\nValid parameters are %s." \
            % (paramName, sorted(validParameterNames)))
      paramValues = params[paramName]
      for paramValue in paramValues:
        if validParameters[paramName] != -1 and paramValue not in validParameters[paramName]:
          raise RuntimeError("Invalid parameter value: %s = %s\nValid values for %s are %s%s." \
                    % (paramName, paramValue, paramName, validParameters[paramName][:32],
                        " (only first 32 combos printed)\nRefer to Common.py for more info" if len(validParameters[paramName])>32 else ""))

def constructForkPermutations(forkParametersConfig):
  """Temp doc"""
  totalPermutations = 1
  for k, v in forkParametersConfig.items():
    totalPermutations *= len(v)
  forkPermutations = []
  for i in range(0, totalPermutations):
    forkPermutations.append({})
    pIdx = i
    for k, v in forkParametersConfig.items():
      values = deepcopy(v)
      valueIdx = pIdx % len(v)
      forkPermutations[i][k] = values[valueIdx]
      pIdx //= len(values)
  return forkPermutations

def getSingleValues(parameterSetList):
  """Temp doc"""
  singleVaules = {}
  for stepList in parameterSetList:
    for paramDict in copy(stepList):
      for paramName in copy(paramDict):
        paramValues = paramDict[paramName]
        if paramValues == None:
          printExit("You must specify value for parameters \"%s\"" % paramName )
        if len(paramValues) < 2 and paramName != "ProblemSizes":
          paramDict.pop(paramName)
          singleVaules[paramName] = paramValues[0]
          if len(paramDict) == 0:
            stepList.remove(paramDict)

  return singleVaules

def checkCDBufferAndStrides(problemType, problemSizes, isCEqualD):
  """Temp doc"""
  if isCEqualD and problemType["OperationType"] == "GEMM":
    for problem in problemSizes.problems:
      ldd = problem.sizes[problemType["IndexAssignmentsLD"][0]]
      ldc = problem.sizes[problemType["IndexAssignmentsLD"][1]]
      if ldd != ldc:
        printExit("LDD(%d) != LDC(%d) causes unpredictable result when CEqualD(True)" % (ldd, ldc))


class BenchmarkProcess:
  """
  Steps in config need to be expanded and missing elements need to be assigned a default.
  TODO better docs
  """

  def __init__(self, problemTypeConfig, problemSizeGroupConfig):
    """Temp doc"""
    self.problemType = ProblemType(problemTypeConfig)
    self.isBatched = "Batched" in problemTypeConfig and problemTypeConfig["Batched"]
    print2("# BenchmarkProcess beginning %s" % str(self.problemType))

    # create initial solution parameters
    self.initialSolutionParameters = { "ProblemType": problemTypeConfig }
    self.initialSolutionParameters.update(defaultSolution)

    # fill parameter values from config
    self.singleValueParameters = {}
    self.multiValueParameters = {}
    self.getConfigParameter(self.isBatched, problemSizeGroupConfig)

    # convert parameter lists to steps
    # currently only 1 benchmark step is possible, more may be added back later
    self.currentProblemSizes = []
    self.benchmarkStepIdx = 0
    self.convertParametersToSteps()

  def getConfigParameter(self, isbatched, config):
    """Temp doc"""
    print2("")
    print2("####################################################################")
    print1("# Filling in Parameters With Defaults")
    print2("####################################################################")
    print2("")

    # check for no longer supported legacy benchmark steps
    badParams = ["InitialSolutionParameters", "BenchmarkForkParameters", \
                 "JoinParameters", "BenchmarkJoinParameters"]
    badsInConfig = []

    for p in badParams:
      if config.get(p) is not None:
        badsInConfig.append(p)

    if len(badsInConfig) == 1:
      printExit("Benchmark step {} is no longer supported".format("'" + badsInConfig[0] + "'"))
    elif len(badsInConfig) > 1:
      printExit("Benchmark steps {} are no longer supported".format(badsInConfig))

    # get supported legacy benchmark steps
    defaultSizes = [{"ProblemSizes": defaultBatchedBenchmarkFinalProblemSizes}] if isbatched \
        else [{"ProblemSizes": defaultBenchmarkFinalProblemSizes}]

    benchmarkCommonParameters      = config.get("BenchmarkCommonParameters", [])
    forkParameters                 = config.get("ForkParameters", [])
    self.benchmarkFinalParameters  = config.get("BenchmarkFinalParameters", defaultSizes)

    configParameters = benchmarkCommonParameters + forkParameters

    # ensure only valid solution parameters were requested
    validParameterNames = set(validParameters.keys())
    for paramDict in configParameters:
      try:
        checkForValidParameters(paramDict, validParameterNames)
      except RuntimeError as e:
        printExit(str(e))

    # get defaults for parameters not specified in config file
    missingParameters = getDefaultsForMissingParameters( \
        configParameters, deepcopy(defaultBenchmarkCommonParameters))

    # split parameters into single value and multi-value
    self.singleValueParameters = getSingleValues([missingParameters, configParameters])

    # above function call removes singles
    self.multiValueParameters = {}
    for paramDict in configParameters:
      for param, values in paramDict.items():
        self.multiValueParameters[param] = values

    # print summary of parameter values
    print2("Single Value Parameters:")
    for k, v in self.singleValueParameters.items():
      print2("    {}: {}".format(k, v))
    print2("Multi-Value Parameters:")
    for k, v in self.multiValueParameters.items():
      print2("    {}: {}".format(k, v))

  def convertParametersToSteps(self):
    """Temp doc"""
    print2("")
    print2("####################################################################")
    print1("# Convert Parameters to Benchmark Step(s)")
    print2("####################################################################")
    print2("")

    # currently only a single step is supported
    print2("")
    print2("####################################################################")
    print1("# Benchmark Final")
    for problemSizesDict in self.benchmarkFinalParameters:
        problemSizes = problemSizesDict["ProblemSizes"]
        self.currentProblemSizes = ProblemSizes(self.problemType, problemSizes)
        checkCDBufferAndStrides(self.problemType, \
            self.currentProblemSizes, globalParameters["CEqualD"])
        benchmarkStep = BenchmarkStep( \
            self.multiValueParameters, \
            self.singleValueParameters, \
            self.currentProblemSizes, \
            self.benchmarkStepIdx )
        self.benchmarkSteps.append(benchmarkStep)
        self.benchmarkStepIdx+=1

  def __len__(self):
    return len(self.benchmarkSteps)

  def __getitem__(self, key):
    return self.benchmarkSteps[key]

  def __str__(self):
    string = "BenchmarkProcess:\n"
    for step in self.benchmarkSteps:
      string += str(step)
    return string

  def __repr__(self):
    return self.__str__()


class BenchmarkStep:
  """Temp doc"""

  def __init__(self, forkParams, constantParams, problemSizes, idx):
    """Temp doc"""
    #TODO see if deepcopy really needed
    self.forkParams = deepcopy(forkParams)
    self.constantParams = deepcopy(constantParams)
    self.problemSizes = deepcopy(problemSizes)
    self.stepIdx = idx

    print2("# Creating BenchmarkStep: {} fork params and {} sizes" \
        .format( len(forkParams), problemSizes.totalProblemSizes))

  def isFinal(self):
    """Temp doc"""
    # currently only one benchmark step is possible
    return True

  def abbreviation(self):
    """Temp doc"""
    string = "{:02d}".format(self.stepIdx)
    if self.isFinal():
      string += "_Final"
    else:
      for param in self.benchmarkParameters:
        string += "_" + Solution.getParameterNameAbbreviation(param)
    return string

  def __str__(self):
    string = "{:02d}".format(self.stepIdx)
    if self.isFinal():
      string += "_Final"
    else:
      for param in self.benchmarkParameters:
        string += "_" + str(param)
    return string

  def __repr__(self):
    return self.__str__()

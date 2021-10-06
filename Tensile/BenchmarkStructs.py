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
    defaultProblemSizes, defaultBenchmarkFinalProblemSizes, \
    defaultBatchedProblemSizes, defaultBatchedBenchmarkFinalProblemSizes, \
    defaultBenchmarkCommonParameters, hasParam, getParamValues, printExit, \
    validParameters, defaultSolutionSummationSizes, globalParameters
from .SolutionStructs import Solution, ProblemType, ProblemSizes


def forkHardcodedParameters(basePermutations, update):
  """Temp doc"""
  updatedHardcodedParameters = []
  for oldPermutation in basePermutations:
    for newPermutation in update:
      permutation = {}
      permutation.update(oldPermutation)
      permutation.update(newPermutation)
      updatedHardcodedParameters.append(permutation)
  return updatedHardcodedParameters

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
  for param in forkParametersConfig:
    for name in param: # only 1
      values = param[name]
      totalPermutations *= len(values)
  forkPermutations = []
  for i in range(0, totalPermutations):
    forkPermutations.append({})
    pIdx = i
    for param in forkParametersConfig:
      for name in param:
        values = deepcopy(param[name])
        valueIdx = pIdx % len(values)
        forkPermutations[i][name] = values[valueIdx]
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
  """

  def __init__(self, problemTypeConfig, problemSizeGroupConfig):
    """Temp doc"""
    self.problemType = ProblemType(problemTypeConfig)
    self.isBatched = "Batched" in problemTypeConfig and problemTypeConfig["Batched"]
    print2("# BenchmarkProcess beginning %s" % str(self.problemType))

    # create initial solution parameters
    self.initialSolutionParameters = { "ProblemType": problemTypeConfig }
    self.initialSolutionParameters.update(defaultSolution)

    # fill in missing steps using defaults
    self.benchmarkCommonParameters = []
    self.forkParameters = []
    self.benchmarkFinalParameters = []
    self.benchmarkSteps = []
    self.hardcodedParameters = [{}]
    self.singleValueParameters = {} # keep
    self.solutionSummationSizes = []

    self.multiValueParameters = {}

    # (I)
    self.fillInMissingStepsWithDefaults(self.isBatched, problemSizeGroupConfig)

    # convert list of parameters to list of steps
    self.currentProblemSizes = []
    self.benchmarkStepIdx = 0

    # (II)
    self.convertParametersToSteps()


  ##############################################################################
  # (I) Create lists of param, filling in missing params from defaults
  ##############################################################################
  def fillInMissingStepsWithDefaults(self, isbatched, config):
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

    self.solutionSummationSizes    = defaultSolutionSummationSizes
    self.benchmarkCommonParameters = config.get("BenchmarkCommonParameters", [])
    self.forkParameters            = config.get("ForkParameters", [])
    self.benchmarkFinalParameters  = config.get("BenchmarkFinalParameters", defaultSizes)

    configParameters = self.benchmarkCommonParameters + self.forkParameters

    # ensure only valid solution parameters were requested
    validParameterNames = set(validParameters.keys())
    for paramDict in configParameters:
      try:
        checkForValidParameters(paramDict, validParameterNames)
      except RuntimeError as e:
        printExit(str(e))

    missingParameters = getDefaultsForMissingParameters( \
        configParameters, deepcopy(defaultBenchmarkCommonParameters))

    ############################################################################
    # (I-7) any default param with 1 value will be hardcoded; move to beginning
    singleValues = getSingleValues([missingParameters, self.benchmarkCommonParameters, \
        self.forkParameters])
    for paramName in singleValues:
      paramValue = singleValues[paramName]
      self.hardcodedParameters[0][paramName] = paramValue
      self.singleValueParameters[paramName] = [ paramValue ]
      self.initialSolutionParameters[paramName] = paramValue

    ############################################################################
    # (I-10) Parameter Lists
    # benchmarkCommonParameters
    print2("HardcodedParameters:")
    for paramName in self.hardcodedParameters[0]:
      paramValues = self.hardcodedParameters[0][paramName]
      print2("    %s: %s" % (paramName, paramValues))
    print2("BenchmarkCommonParameters:")
    for step in self.benchmarkCommonParameters:
      print2("    %s" % step)
    # forkParameters
    print2("ForkParameters:")
    for param in self.forkParameters:
      print2("    %s" % param)

  ##############################################################################
  # (II) convert lists of parameters to benchmark steps
  ##############################################################################
  def convertParametersToSteps(self):
    print2("")
    print2("####################################################################")
    print1("# Convert Parameters to Benchmark Step")
    print2("####################################################################")
    print2("")

    ############################################################################
    # (II-2) fork parameters
    # calculate permutations of
    print2("")
    print2("####################################################################")
    print1("# Fork Parameters")
    print2(self.forkParameters)
    forkPermutations = constructForkPermutations(self.forkParameters)
    if len(forkPermutations) > 0:
      self.forkHardcodedParameters(forkPermutations)

    ############################################################################
    # (II-6) benchmark final
    print2("")
    print2("####################################################################")
    print1("# Benchmark Final")
    for problemSizesDict in self.benchmarkFinalParameters:
      if "SolutionSummationSizes" in problemSizesDict:
        self.solutionSummationSizes = problemSizesDict["SolutionSummationSizes"]
      else:
        problemSizes = problemSizesDict["ProblemSizes"]
        self.currentProblemSizes = ProblemSizes(self.problemType, problemSizes)
        currentBenchmarkParameters = {}
        checkCDBufferAndStrides(self.problemType, \
            self.currentProblemSizes, globalParameters["CEqualD"])
        benchmarkStep = BenchmarkStep(
            self.hardcodedParameters,
            currentBenchmarkParameters,
            self.initialSolutionParameters,
            self.currentProblemSizes,
            self.benchmarkStepIdx )
        self.benchmarkSteps.append(benchmarkStep)
        self.benchmarkStepIdx+=1

  ##############################################################################
  # Add new permutations of hardcoded parameters to old permutations of params
  ##############################################################################
  def forkHardcodedParameters( self, update ):
    #updatedHardcodedParameters = []
    #for oldPermutation in self.hardcodedParameters:
      #for newPermutation in update:
      #  permutation = {}
      #  permutation.update(oldPermutation)
      #  permutation.update(newPermutation)
      #  updatedHardcodedParameters.append(permutation)
    updatedHardcodedParameters = forkHardcodedParameters( self.hardcodedParameters, update )
      #updatedHardcodedParameters.append(permutation)
    self.hardcodedParameters = updatedHardcodedParameters

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

################################################################################
# Benchmark Step
################################################################################
class BenchmarkStep:

  def __init__(self, hardcodedParameters, \
      benchmarkParameters, initialSolutionParameters, problemSizes, idx):
    # what is my step Idx
    self.stepIdx = idx

    # what parameters don't need to be benchmarked because hard-coded or forked
    # it's a list of dictionaries, each element a permutation
    self.hardcodedParameters = deepcopy(hardcodedParameters)
    #if len(self.hardcodedParameters) == 0:
    #  printExit("hardcodedParameters is empty")

    # what parameters will I benchmark
    self.benchmarkParameters = deepcopy(benchmarkParameters)
    #if len(self.benchmarkParameters) == 0:
    #  printExit("benchmarkParameters is empty")

    # what solution parameters do I use for what hasn't been benchmarked
    self.initialSolutionParameters = initialSolutionParameters

    # what problem sizes do I benchmark
    self.problemSizes = deepcopy(problemSizes)

    print2("# Creating BenchmarkStep [BP]=%u [HCP]=%u [P]=%u" \
        % ( len(benchmarkParameters), len(hardcodedParameters), \
        problemSizes.totalProblemSizes))

  def isFinal(self):
    return len(self.benchmarkParameters) == 0

  def abbreviation(self):
    string = "%02u" % self.stepIdx
    if self.isFinal():
      string += "_Final"
    else:
      for param in self.benchmarkParameters:
        string += "_%s" % Solution.getParameterNameAbbreviation(param)
    return string

  def __str__(self):
    string = "%02u" % self.stepIdx
    if self.isFinal():
      string += "_Final"
    else:
      for param in self.benchmarkParameters:
        string += "_%s" % str(param)
    return string

  def __repr__(self):
    return self.__str__()

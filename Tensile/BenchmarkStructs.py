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
    defaultBenchmarkCommonParameters, hasParam, \
    defaultBenchmarkJoinParameters, getParamValues, defaultForkParameters, \
    defaultBenchmarkForkParameters, defaultJoinParameters, printExit, \
    validParameters, defaultSolutionSummationSizes, globalParameters
from .SolutionStructs import Solution, ProblemType, ProblemSizes


### modularize benchmark steps construction

##############################################################################
# forkHardcodedParameters
##############################################################################
def forkHardcodedParameters( basePermutations, update ):
  updatedHardcodedParameters = []
  for oldPermutation in basePermutations:
    for newPermutation in update:
      permutation = {}
      permutation.update(oldPermutation)
      permutation.update(newPermutation)
      updatedHardcodedParameters.append(permutation)
  return updatedHardcodedParameters

def fillMissingParametersWithDefaults(parameterConfigurationList, defaultParameters):

  benchmarkParameters = []
  for paramDict in defaultParameters:
    for paramName in paramDict:
      if not hasParam( paramName, parameterConfigurationList) \
          or paramName == "ProblemSizes":
        benchmarkParameters.append(paramDict)
  return benchmarkParameters

def checkForValidParameters(params, validParameterNames):
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
  ############################################################################
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

##############################################################################
# assignParameters
##############################################################################
def assignParameters(problemTypeConfig, configBenchmarkCommonParameters, configForkParameters):

  problemTypeObj = ProblemType(problemTypeConfig)
  initialSolutionParameters = { "ProblemType": problemTypeConfig }
  initialSolutionParameters.update(defaultSolution)

  hardcodedParameters = []
  benchmarkCommonParameters = fillMissingParametersWithDefaults([configBenchmarkCommonParameters, configForkParameters], defaultBenchmarkCommonParameters)
  if configBenchmarkCommonParameters != None:
    for paramDict in configBenchmarkCommonParameters:
      benchmarkCommonParameters.append(paramDict)

  singleValues = getSingleValues([benchmarkCommonParameters, configForkParameters])
  for paramName in singleValues:
    paramValue = singleValues[paramName]
    initialSolutionParameters[paramName] = paramValue

  forkPermutations = constructForkPermutations(configForkParameters)
  if len(forkPermutations) > 0:
    hardcodedParameters = forkHardcodedParameters([initialSolutionParameters], forkPermutations)

  return (problemTypeObj, hardcodedParameters, initialSolutionParameters)


##############################################################################
# check LDD == LDC if CEqualD"
##############################################################################
def checkCDBufferAndStrides(problemType, problemSizes, isCEqualD):
  if isCEqualD and problemType["OperationType"] == "GEMM":
    for problem in problemSizes.problems:
      ldd = problem.sizes[problemType["IndexAssignmentsLD"][0]]
      ldc = problem.sizes[problemType["IndexAssignmentsLD"][1]]
      if ldd != ldc:
        printExit("LDD(%d) != LDC(%d) causes unpredictable result when CEqualD(True)" % (ldd, ldc))


class BenchmarkProcess:
  """
  Benchmark Process
  Steps in config need to be expanded and missing elements need to be assigned a default.
  """

  def __init__(self, problemTypeConfig, problemSizeGroupConfig):

    self.problemType = ProblemType(problemTypeConfig)
    self.isBatched = "Batched" in problemTypeConfig and problemTypeConfig["Batched"]
    print2("# BenchmarkProcess beginning %s" % str(self.problemType))

    # read initial solution parameters
    self.initialSolutionParameters = { "ProblemType": problemTypeConfig }
    self.initialSolutionParameters.update(defaultSolution)
    if "InitialSolutionParameters" not in problemSizeGroupConfig:
      print2("No InitialSolutionParameters; using defaults.")
    else:
      if problemSizeGroupConfig["InitialSolutionParameters"] != None:
        for paramDict in problemSizeGroupConfig["InitialSolutionParameters"]:
          for paramName in paramDict:
            paramValueList = paramDict[paramName]
            if isinstance(paramValueList, list):
              if len(paramValueList) != 1:
                printWarning("InitialSolutionParameters must have length=1: %s:%s" % (paramName, paramValueList))
              self.initialSolutionParameters[paramName] = paramValueList[0]
            else:
              self.initialSolutionParameters[paramName] = paramValueList
    print2("# InitialSolutionParameters: %s" % str(self.initialSolutionParameters))

    # fill in missing steps using defaults
    self.benchmarkCommonParameters = []
    self.forkParameters = []
    self.benchmarkForkParameters = []
    self.joinParameters = []
    self.benchmarkJoinParameters = []
    self.benchmarkFinalParameters = []
    self.benchmarkSteps = []
    self.hardcodedParameters = [{}]
    self.singleValueParameters = {}
    self.solutionSummationSizes = []

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

    self.solutionSummationSizes = defaultSolutionSummationSizes
    ############################################################################
    # (I-0) get 6 phases from config
    configBenchmarkCommonParameters = config["BenchmarkCommonParameters"] \
        if "BenchmarkCommonParameters" in config else []
    configForkParameters = config["ForkParameters"] \
        if "ForkParameters" in config else []

    # TODO cleanup error messages
    if config.get("BenchmarkForkParameters") is not None:
      print("no longer supported")
    if config.get("JoinParameters") is not None:
      print("no longer supported")
    if config.get("BenchmarkJoinParameters") is not None:
      print("no longer supported")

    configBenchmarkFinalParameters = config["BenchmarkFinalParameters"] \
        if "BenchmarkFinalParameters" in config and config["BenchmarkFinalParameters"] != None \
        and len(config["BenchmarkFinalParameters"]) > 0 \
        else [{"ProblemSizes": defaultBatchedBenchmarkFinalProblemSizes}] \
        if isbatched \
        else [{"ProblemSizes": defaultBenchmarkFinalProblemSizes}]

    ############################################################################
    # Ensure only valid solution parameters were requested
    validParameterNames = set(validParameters.keys())
    for paramDictList in [configBenchmarkCommonParameters, \
        configForkParameters]:
      if paramDictList != None:
        for paramDict in paramDictList:
          try:
            checkForValidParameters(paramDict, validParameterNames)
          except RuntimeError as e:
            printExit(str(e))

    # TODO warn/guard against BenchmarkCommons that have mor than one value
    benchmarkCommonParameters = fillMissingParametersWithDefaults( \
        [configBenchmarkCommonParameters, configForkParameters], \
        deepcopy(defaultBenchmarkCommonParameters))
    self.benchmarkCommonParameters.extend(benchmarkCommonParameters)

    if configBenchmarkCommonParameters != None:
      for paramDict in configBenchmarkCommonParameters:
        self.benchmarkCommonParameters.append(paramDict)

    ############################################################################
    # (I-2) into fork we put in all Dfork that
    # don't show up in Bcommon/Cfork/CBfork/Cjoin/CBjoin
    # followed by Cfork
    self.forkParameters = []
    # need to use deepcopy to prevent default parameters from being washed-out later
    for paramDict in deepcopy(defaultForkParameters):
      for paramName in paramDict:
        if not hasParam( paramName, [ self.benchmarkCommonParameters, \
            configForkParameters]) \
            or paramName == "ProblemSizes":
          self.forkParameters.append(paramDict)
    if configForkParameters != None:
      for paramDict in configForkParameters:
        self.forkParameters.append(paramDict)
    else: # make empty
      self.forkParameters = []

    ############################################################################
    # (I-6) benchmark final sizes
    self.benchmarkFinalParameters = configBenchmarkFinalParameters
    # no other parameters besides problem sizes

    ############################################################################
    # (I-7) any default param with 1 value will be hardcoded; move to beginning
    singleValues = getSingleValues([self.benchmarkCommonParameters, \
        self.forkParameters, self.benchmarkForkParameters, \
        self.benchmarkJoinParameters])
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
    print1("# Convert Parameters to Steps")
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
  # For list of config parameters convert to steps and append to steps list
  ##############################################################################
  def addStepsForParameters(self, configParameterList):
    print2("# AddStepsForParameters: %s" % configParameterList)
    for paramConfig in configParameterList:
      if isinstance(paramConfig, dict):
        if "ProblemSizes" in paramConfig:
          self.currentProblemSizes = ProblemSizes(self.problemType, paramConfig["ProblemSizes"])
          continue
      currentBenchmarkParameters = {}
      for paramName in paramConfig:
        paramValues = paramConfig[paramName]
        if isinstance(paramValues, list):
          currentBenchmarkParameters[paramName] = paramValues
        else:
          printExit("Parameter \"%s\" for ProblemType %s must be formatted as a list but isn't" \
              % ( paramName, str(self.problemType) ) )
      if len(currentBenchmarkParameters) > 0:
        print2("Adding BenchmarkStep for %s" % str(currentBenchmarkParameters))
        checkCDBufferAndStrides(self.problemType, self.currentProblemSizes, \
            globalParameters["CEqualD"])
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

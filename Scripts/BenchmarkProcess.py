import sys
from copy import *

from Common import *
from Structs import *

################################################################################
# Benchmark Process
# steps in config need to be expanded and
# missing elements need to be assigned a default
################################################################################
class BenchmarkProcess:

  ##############################################################################
  def __init__(self, config):
    # read problem type
    if "ProblemType" in config:
      problemTypeConfig = config["ProblemType"]
    else:
      problemTypeConfig = {}
      printWarning("No ProblemType in config: %s; using defaults." % str(config) )
    self.problemType = ProblemType(problemTypeConfig)
    printStatus("BenchmarkProcess beginning %s" % str(self.problemType))

    # read initial solution parameters
    solutionConfig = { "ProblemType": problemTypeConfig }
    if "InitialSolutionParameters" not in config:
      printWarning("No InitialSolutionParameters; using defaults.")
    else:
      solutionConfig.update(config["InitialSolutionParameters"])
    self.initialSolutionParameters = Solution(solutionConfig)
    printExtra("InitialSolutionParameters: %s" % str(self.initialSolutionParameters))

    # fill in missing steps using defaults
    self.benchmarkCommonParameters = []
    self.forkParameters = []
    self.benchmarkForkParameters = []
    self.joinParameters = []
    self.benchmarkJoinParameters = []
    self.benchmarkFinalParameters = []
    self.benchmarkSteps = []
    self.fillInMissingStepsWithDefaults(config)

    # convert list of parameters to list of steps
    self.prevParameters = []
    self.currentProblemSizes = []
    self.hardcodedParameters = [{}]
    self.benchmarkStepIdx = 0
    self.convertParametersToSteps()


  ##############################################################################
  # convert lists of parameters to benchmark steps
  def convertParametersToSteps(self):
    printExtra("beginning")

    # (1) benchmark common parameters
    printExtra("1")
    self.addStepsForParameters( self.benchmarkCommonParameters  )

    # (2) fork parameters
    # calculate permutations of
    printExtra("2")
    totalPermutations = 1
    for param in self.forkParameters:
      for name in param: # only 1
        values = param[name]
        totalPermutations *= len(values)
    forkPermutations = []
    for i in range(0, totalPermutations):
      forkPermutations.append({})
      pIdx = i
      for param in self.forkParameters:
        for name in param:
          values = param[name]
          valueIdx = pIdx % len(values)
          forkPermutations[i][name] = values[valueIdx]
          pIdx /= len(values)
      #print forkPermutations[i]
    #self.hardcodedParameters.append(forkPermutations)
    self.forkHardcodedParameters(forkPermutations)

    # (3) benchmark common parameters
    printExtra("3")
    self.addStepsForParameters( self.benchmarkForkParameters  )

    # (4) join parameters
    # answer should go in hard-coded parameters
    # does it remove the prior forks? Yes.
    printExtra("4")
    macroTileJoinSet = set()
    depthUJoinSet = set()
    totalPermutations = 1
    for joinName in self.joinParameters:
      # find in hardcoded; that's where forked will be
      if hasParam(joinName, self.forkParameters):
        for param in self.forkParameters:
          for name in param: # only 1
            values = param[name]
            localPermutations = len(values)
            print "JoinParameter %s has %u possibilities" % (joinName, localPermutations)
            totalPermutations *= localPermutations
      elif joinName == "MacroTile":
        print "JoinParam: MacroTile"
        # get possible WorkGroupEdges from forked
        print self.forkParameters
        workGroupEdgeValues = []
        workGroupShapeValues = []
        threadTileEdgeValues = []
        threadTileShapeValues = []
        for paramList in [self.benchmarkCommonParameters, self.forkParameters, self.benchmarkForkParameters, self.benchmarkJoinParameters]:
          if hasParam("WorkGroupEdge", paramList):
            workGroupEdgeValues = getParamValues("WorkGroupEdge", paramList)
          if hasParam("WorkGroupShape", paramList):
            workGroupShapeValues = getParamValues("WorkGroupShape", paramList)
          if hasParam("ThreadTileEdge", paramList):
            threadTileEdgeValues = getParamValues("ThreadTileEdge", paramList)
          if hasParam("ThreadTileShape", paramList):
            threadTileShapeValues = getParamValues("ThreadTileShape", paramList)
        macroTilePermutations = len(workGroupEdgeValues)*len(workGroupShapeValues)*len(threadTileEdgeValues)*len(threadTileShapeValues)
        printStatus("Total JoinMacroTile Permutations: %u" % macroTilePermutations)

        for i in range(0, macroTilePermutations):
          pIdx = i
          workGroupEdgeIdx = pIdx % len(workGroupEdgeValues)
          pIdx /= len(workGroupEdgeValues)
          workGroupShapeIdx = pIdx % len(workGroupShapeValues)
          pIdx /= len(workGroupShapeValues)
          threadTileEdgeIdx = pIdx % len(threadTileEdgeValues)
          pIdx /= len(threadTileEdgeValues)
          threadTileShapeIdx = pIdx % len(threadTileShapeValues)
          macroTileDim0 = workGroupEdgeValues[workGroupEdgeIdx]*threadTileEdgeValues[threadTileEdgeIdx]
          macroTileDim1 = macroTileDim0
          if workGroupShapeValues[workGroupShapeIdx] < 0:
            macroTileDim1 /= 2
          elif workGroupShapeValues[workGroupShapeIdx] > 0:
            macroTileDim1 *= 2
          if threadTileShapeValues[threadTileShapeIdx] < 0:
            macroTileDim1 /= 2
          elif threadTileShapeValues[threadTileShapeIdx] > 0:
            macroTileDim1 *= 2
          if macroTileDim0/macroTileDim1 <= self.initialSolutionParameters["MacroTileMaxRatio"] and macroTileDim1/macroTileDim0 <= self.initialSolutionParameters["MacroTileMaxRatio"]:
            macroTileJoinSet.add((macroTileDim0, macroTileDim1))
        totalPermutations *=len(macroTileJoinSet)
        printStatus("JoinMacroTileSet(%u): %s" % (len(macroTileJoinSet), macroTileJoinSet) )

      # Join DepthU
      elif joinName == "DepthU":
        unrollValues = []
        splitUValues = []
        for paramList in [self.benchmarkCommonParameters, self.forkParameters, self.benchmarkForkParameters, self.benchmarkJoinParameters]:
          if hasParam("Unroll", paramList):
            unrollValues = getParamValues("Unroll", paramList)
          if hasParam("SplitU", paramList):
            splitUValues = getParamValues("SplitU", paramList)
        depthUPermutations = len(unrollValues)*len(splitUValues)
        printStatus("Total JoinDepthU Permutations: %u" % depthUPermutations)
        for i in range(0, depthUPermutations):
          pIdx = i
          unrollIdx = pIdx % len(unrollValues)
          pIdx /= len(unrollValues)
          splitUIdx = pIdx % len(splitUValues)
          depthU = unrollValues[unrollIdx]*splitUValues[splitUIdx]
          depthUJoinSet.add(depthU)
        totalPermutations *= len(depthUJoinSet)
        printStatus("JoinSplitUSet(%u): %s" % (len(depthUJoinSet), depthUJoinSet) )

      # invalid join parameter
      else:
        validJoinNames = ["MacroTile", "DepthU"]
        for validParam in self.forkParameters:
          for validName in validParam: # only 1
            validJoinNames.append(validName)
        printExit("JoinParameter \"%s\" not in %s" % (joinName, validJoinNames) )

    # now we need to create permutations of MacroTile*DepthU*
    macroTiles = list(macroTileJoinSet)
    depthUs = list(depthUJoinSet)
    printStatus("TotalJoinPermutations = %u" % ( totalPermutations) )
    joinPermutations = []
    for i in range(0, totalPermutations):
      joinPermutations.append({})
      pIdx = i
      for joinName in self.joinParameters:
        if hasParam(joinName, self.forkParameters):
          for paramDict in self.forkParameters:
            if joinName in paramDict:
              paramValues = paramDict[joinName]
              valueIdx = pIdx % len(paramValues)
              joinPermutations[i][joinName] = paramValues[valueIdx]
              pIdx /= len(paramValues)
              break
        elif joinName == "MacroTile":
          valueIdx = pIdx % len(macroTiles)
          joinPermutations[i][joinName] = macroTiles[valueIdx]
        elif joinName == "DepthU":
          valueIdx = pIdx % len(depthUs)
          joinPermutations[i][joinName] = depthUs[valueIdx]
      #print joinPermutations[i]
    #self.hardcodedParameters.append(joinPermutations)
    self.joinHardcodedParameters(joinPermutations)


    # (5) benchmark join parameters
    printExtra("5")
    self.addStepsForParameters( self.benchmarkJoinParameters  )

    # (6) benchmark final
    printExtra("6")
    self.currentProblemSizes = ProblemSizes(self.problemType, \
        self.benchmarkFinalParameters["ProblemSizes"])
    currentBenchmarkParameters = {}
    benchmarkStep = BenchmarkStep(
        self.hardcodedParameters,
        self.prevParameters,
        currentBenchmarkParameters,
        self.initialSolutionParameters.state,
        self.currentProblemSizes,
        self.benchmarkStepIdx )

  ##############################################################################
  # for list of config parameters convert to steps and append to steps list
  def addStepsForParameters(self, configParameterList):
    for paramConfig in configParameterList:
      if isinstance(paramConfig, dict):
        if "ProblemSizes" in paramConfig:
          self.currentProblemSizes = ProblemSizes(self.problemType, paramConfig["ProblemSizes"])
          continue
      currentBenchmarkParameters = {}
      for paramName in paramConfig:
        paramValues = paramConfig[paramName]
        if len(paramValues) == 1:
          #self.hardcodedParameters.append([{paramName: paramValues[0]}])
          self.forkHardcodedParameters([{paramName: paramValues[0]}])
        else:
          currentBenchmarkParameters[paramName] = paramValues
      if len(currentBenchmarkParameters) > 0:
        benchmarkStep = BenchmarkStep(
            self.hardcodedParameters,
            self.prevParameters,
            currentBenchmarkParameters,
            self.initialSolutionParameters.state,
            self.currentProblemSizes,
            self.benchmarkStepIdx )
        self.benchmarkSteps.append(benchmarkStep)
        self.prevParameters.append(currentBenchmarkParameters)
        self.benchmarkStepIdx+=1


  ##############################################################################
  # create thorough lists of parameters, filling in missing info from defaults
  def fillInMissingStepsWithDefaults(self, config):

    # TODO - print warning when config contains a parameter
    # that doesn't have a default; that means they probably spelled it wrong

    # get benchmark steps from config
    configBenchmarkCommonParameters = config["BenchmarkCommonParameters"] \
        if "BenchmarkCommonParameters" in config \
        else [{"ProblemSizes": defaultProblemSizes}]
    configForkParameters = config["ForkParameters"] \
        if "ForkParameters" in config else []
    configBenchmarkForkParameters = config["BenchmarkForkParameters"] \
        if "BenchmarkForkParameters" in config \
        else [{"ProblemSizes": defaultProblemSizes}]
    configJoinParameters = config["JoinParameters"] \
        if "JoinParameters" in config else []
    configBenchmarkJoinParameters = config["BenchmarkJoinParameters"] \
        if "BenchmarkJoinParameters" in config \
        else [{"ProblemSizes": defaultProblemSizes}]
    configBenchmarkFinalParameters = config["BenchmarkFinalParameters"] \
        if "BenchmarkFinalParameters" in config \
        else {"ProblemSizes": defaultBenchmarkFinalProblemSizes}

    ########################################
    # (1) get current problem sizes
    if "ProblemSizes" in configBenchmarkCommonParameters[0]:
      # user specified one, so use it, remove it from config and insert later
      currentProblemSizes = configBenchmarkCommonParameters[0]["ProblemSizes"]
      del configBenchmarkCommonParameters[0]
    else:
      currentProblemSizes = defaultProblemSizes
    # (1) into common we put in all Dcommon that
    # don't show up in Ccommon/Cfork/CBfork/Cjoin/CBjoin
    # followed by Ccommon
    self.benchmarkCommonParameters = [{"ProblemSizes": currentProblemSizes}]
    for paramDict in defaultBenchmarkCommonParameters:
      for paramName in paramDict:
        if not hasParam( paramName, [ configBenchmarkCommonParameters, \
            configForkParameters, configBenchmarkForkParameters, \
            configJoinParameters, configBenchmarkJoinParameters]) \
            or paramName == "ProblemSizes":
          self.benchmarkCommonParameters.append(paramDict)
    for paramDict in configBenchmarkCommonParameters:
      self.benchmarkCommonParameters.append(paramDict)

    ########################################
    # (2) into fork we put in all Dfork that
    # don't show up in Bcommon/Cfork/CBfork/Cjoin/CBjoin
    # followed by Cfork
    self.forkParameters = []
    for paramDict in defaultForkParameters:
      for paramName in paramDict:
        if not hasParam( paramName, [ self.benchmarkCommonParameters, \
            configForkParameters, configBenchmarkForkParameters, \
            configJoinParameters, configBenchmarkJoinParameters]) \
            or paramName == "ProblemSizes":
          self.forkParameters.append(paramDict)
    for paramDict in configForkParameters:
      self.forkParameters.append(paramDict)

    ########################################
    # (3) get current problem sizes
    if "ProblemSizes" in configBenchmarkForkParameters[0]:
      # user specified one, so use it, remove it from config and insert later
      currentProblemSizes = configBenchmarkForkParameters[0]["ProblemSizes"]
      del configBenchmarkForkParameters[0]
    # (3) into Bfork we put in all DBfork that
    # don't show up in Bcommon/Bfork/CBfork/Cjoin/CBjoin
    # followed by CBforked
    self.benchmarkForkParameters = [{"ProblemSizes": currentProblemSizes}]
    for paramDict in defaultBenchmarkForkParameters:
      for paramName in paramDict:
        if not hasParam( paramName, [ self.benchmarkCommonParameters, \
            self.forkParameters, configBenchmarkForkParameters, \
            configJoinParameters, configBenchmarkJoinParameters]) \
            or paramName == "ProblemSizes":
          self.benchmarkForkParameters.append(paramDict)
    for paramDict in configBenchmarkForkParameters:
      self.benchmarkForkParameters.append(paramDict)

    ########################################
    # (4) into join we put in all Djoin that
    # don't show up in Bcommon/Bfork/CBfork/Cjoin/CBjoin
    # followed by CBforked
    self.joinParameters = []
    for paramName in defaultJoinParameters:
      if not hasParam( paramName, [ self.benchmarkCommonParameters, \
          self.forkParameters, self.benchmarkForkParameters, \
          configJoinParameters, configBenchmarkJoinParameters]) \
          or paramName == "ProblemSizes":
        self.joinParameters.append(paramName)
    for paramName in configJoinParameters:
      self.joinParameters.append(paramName)

    ########################################
    # (5) get current problem sizes
    if "ProblemSizes" in configBenchmarkJoinParameters[0]:
      # user specified one, so use it, remove it from config and insert later
      currentProblemSizes = configBenchmarkJoinParameters[0]["ProblemSizes"]
      del configBenchmarkJoinParameters[0]
    # (5) into Bjoin we put in all DBjoin that
    # don't show up in Bcommon/Bfork/BBfork/Bjoin/CBjoin
    # followed by CBjoin
    self.benchmarkJoinParameters = [{"ProblemSizes": currentProblemSizes}]
    for paramDict in defaultBenchmarkJoinParameters:
      for paramName in paramDict:
        if not hasParam( paramName, [ self.benchmarkCommonParameters, \
            self.forkParameters, self.benchmarkForkParameters, \
            self.joinParameters, configBenchmarkJoinParameters]) \
            or paramName == "ProblemSizes":
          self.benchmarkJoinParameters.append(paramDict)
    for paramDict in configBenchmarkJoinParameters:
      self.benchmarkJoinParameters.append(paramDict)

    ########################################
    # (6) get current problem sizes
    self.benchmarkFinalParameters = configBenchmarkFinalParameters
    # no other parameters besides problem sizes


    # benchmarkCommonParameters
    printExtra("benchmarkCommonParameters")
    for step in self.benchmarkCommonParameters:
      print "    %s" % step
    # forkParameters
    printExtra("forkParameters")
    for param in self.forkParameters:
      print "    %s" % param
    # benchmarkForkParameters
    printExtra("benchmarkForkParameters")
    for step in self.benchmarkForkParameters:
      print "    %s" % step
    # joinParameters
    printExtra("joinParameters")
    for param in self.joinParameters:
      print "    %s" % param
    # benchmarkJoinParameters
    printExtra("benchmarkJoinParameters")
    for step in self.benchmarkJoinParameters:
      print "    %s" % step

    # Final Benchmark
    # if "BenchmarkFinal" in config:
    printStatus("DONE.")

  ##############################################################################
  # add new permutations of hardcoded parameters to old permutations of params
  def forkHardcodedParameters( self, update ):
    #printStatus("\nold = %u:%s\nnew = %u:%s" % ( len(self.hardcodedParameters), self.hardcodedParameters, len(update), update))
    #printStatus("update = %s" % str(update))
    updatedHardcodedParameters = []
      #print "oldPermutation = %s" % str(oldPermutation)
    for oldPermutation in self.hardcodedParameters:
      for newPermutation in update:
        permutation = {}
        permutation.update(oldPermutation)
        permutation.update(newPermutation)
        #print "  joinedPermutation = %s" % str(permutation)
        updatedHardcodedParameters.append(permutation)
    self.hardcodedParameters = updatedHardcodedParameters
    #print("  updated = %u" % ( len(self.hardcodedParameters)))

  ##############################################################################
  # contract old permutations of hardcoded parameters based on new
  def joinHardcodedParameters( self, update ):
    #printStatus("\n  old = %u:%s\n  new = %u:%s" % ( len(self.hardcodedParameters), self.hardcodedParameters, len(update), update))
    #printStatus("update = %s" % str(update))
      #print "oldPermutation = %s" % str(oldPermutation)
    newHasMacroTile = False
    for newPermutation in update:
      if "MacroTile" in newPermutation:
        newHasMacroTile = True
        break
    newHasDepthU = False
    for newPermutation in update:
      if "DepthU" in newPermutation:
        newHasDepthU = True
        break

    if newHasMacroTile:
      for oldPermutation in self.hardcodedParameters:
        oldPermutation.pop("WorkGroupEdge", None )
        oldPermutation.pop("WorkGroupShape", None )
        oldPermutation.pop("ThreadTileEdge", None )
        oldPermutation.pop("ThreadTileShape", None )
    if newHasDepthU:
      for oldPermutation in self.hardcodedParameters:
        oldPermutation.pop("LoopUnroll", None )
        oldPermutation.pop("SplitU", None )

    updatedHardcodedParameters = []
    for newPermutation in update:
      # if its a hybrid param, delete primitives from hard-coded;
      # primitives will be recorded as "determined" later
      for oldPermutation in self.hardcodedParameters:
        permutation = {}
        permutation.update(oldPermutation)
        permutation.update(newPermutation)
        if permutation not in updatedHardcodedParameters: # "set"
          updatedHardcodedParameters.append(permutation)
    # convert to set and back to list to remove duplicates
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
# - check if this step needs to be performed based on redo
# - which problem sizes will be benchmarked
# - which solutions will be benchmarked
#   - know prior step to read results from
#   - simple algorithm to choose winner, and record it
# - Write
#   - solution files
#   - solution iteration file
#   - problem iteration file
#   - CMakeLists.txt
# - Copy static files to build directory
# - compile and run executable
# - read in csv data and write winner.yaml
################################################################################
class BenchmarkStep:

  def __init__(self, hardcodedParameters, prevParameters, \
      benchmarkParameters, initialSolutionParameters, problemSizes, idx):
    # what is my step Idx
    self.stepIdx = idx

    # what parameters don't need to be benchmarked because hard-coded or forked
    # it's a list of dictionaries, each element a permutation
    self.hardcodedParameters = deepcopy(hardcodedParameters)

    # what parameters have been previously determined
    self.prevParameters = deepcopy(prevParameters)

    # what parameters will I benchmark
    self.benchmarkParameters = deepcopy(benchmarkParameters)

    # what solution parameters do I use for what hasn't been benchmarked
    self.initialSolutionParameters = initialSolutionParameters

    # what problem sizes do I benchmark
    self.problemSizes = deepcopy(problemSizes)

    # what winners will I parse from my data

  def __str__(self):
    string = "%02u" % self.stepIdx
    for param in self.benchmarkParameters:
      string += "_%s" % str(param)
    return string
  def __repr__():
    return self.__str__()




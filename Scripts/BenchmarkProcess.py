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
  # Init
  ##############################################################################
  def __init__(self, config):
    # read problem type
    if "ProblemType" in config:
      problemTypeConfig = config["ProblemType"]
    else:
      problemTypeConfig = {}
      printDefault("No ProblemType in config: %s; using defaults." % str(config) )
    self.problemType = ProblemType(problemTypeConfig)
    printStatus("BenchmarkProcess beginning %s" % str(self.problemType))

    # read initial solution parameters
    solutionConfig = { "ProblemType": problemTypeConfig }
    if "InitialSolutionParameters" not in config:
      printDefault("No InitialSolutionParameters; using defaults.")
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
    self.hardcodedParameters = [{}]
    self.singleValueParameters = {}

    # (I)
    self.fillInMissingStepsWithDefaults(config)

    # convert list of parameters to list of steps
    self.currentProblemSizes = []
    self.benchmarkStepIdx = 0

    # (II)
    self.convertParametersToSteps()


  ##############################################################################
  # (I) Create lists of param, filling in missing params from defaults
  ##############################################################################
  def fillInMissingStepsWithDefaults(self, config):
    print ""
    print "####################################################################"
    print "# Filling in Parameters With Defaults"
    print "####################################################################"
    print ""

    # TODO - print warning when config contains a parameter
    # that doesn't have a default; that means they probably spelled it wrong

    ############################################################################
    # (I-0) get 6 phases from config
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
    configBenchmarkFinalParameters = config["BenchmarkFinalParameters"][0] \
        if "BenchmarkFinalParameters" in config \
        else {"ProblemSizes": defaultBenchmarkFinalProblemSizes}


    ############################################################################
    # (I-1) get current problem sizes
    if configBenchmarkCommonParameters != None:
      if len(configBenchmarkCommonParameters) > 0:
        if "ProblemSizes" in configBenchmarkCommonParameters[0]:
          # user specified, so use it, remove it from config and insert later
          currentProblemSizes = \
          configBenchmarkCommonParameters[0]["ProblemSizes"]
          del configBenchmarkCommonParameters[0]
    else:
      currentProblemSizes = defaultProblemSizes
    # into common we put in all Dcommon that
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
    if configBenchmarkCommonParameters != None:
      for paramDict in configBenchmarkCommonParameters:
        self.benchmarkCommonParameters.append(paramDict)
    else: # make empty
      self.benchmarkCommonParameters = [{"ProblemSizes": currentProblemSizes}]

    ############################################################################
    # (I-2) into fork we put in all Dfork that
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
    if configForkParameters != None:
      for paramDict in configForkParameters:
        self.forkParameters.append(paramDict)
    else: # make empty
      self.forkParameters = []

    ############################################################################
    # (I-3) get current problem sizes
    if configBenchmarkForkParameters != None:
      if len(configBenchmarkForkParameters) > 0:
        if "ProblemSizes" in configBenchmarkForkParameters[0]:
          # user specified, so use it, remove it from config and insert later
          currentProblemSizes = configBenchmarkForkParameters[0]["ProblemSizes"]
          del configBenchmarkForkParameters[0]
    # into Bfork we put in all DBfork that
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
    if configBenchmarkForkParameters != None:
      for paramDict in configBenchmarkForkParameters:
        self.benchmarkForkParameters.append(paramDict)
    else: # make empty
      self.benchmarkForkParameters = [{"ProblemSizes": currentProblemSizes}]

    ############################################################################
    # (I-4) into join we put in all non-derrived Djoin that
    # don't show up in Bcommon/Bfork/CBfork/Cjoin/CBjoin
    # followed by CBforked
    self.joinParameters = []
    for paramName in defaultJoinParameters:
      if not hasParam( paramName, [ self.benchmarkCommonParameters, \
          self.forkParameters, self.benchmarkForkParameters, \
          configJoinParameters, configBenchmarkJoinParameters]) \
          or paramName == "ProblemSizes":
        if "JoinParameters" not in config \
            or (paramName != "MacroTile" and paramName != "DepthU"):
          self.joinParameters.append(paramName)
    if configJoinParameters != None:
      for paramName in configJoinParameters:
        self.joinParameters.append(paramName)
    else: # make empty
        self.joinParameters = []

    ############################################################################
    # (I-5) benchmark join
    if configBenchmarkJoinParameters != None:
      if len(configBenchmarkJoinParameters) > 0:
        if "ProblemSizes" in configBenchmarkJoinParameters[0]:
          # user specified, so use it, remove it from config and insert later
          currentProblemSizes = configBenchmarkJoinParameters[0]["ProblemSizes"]
          del configBenchmarkJoinParameters[0]
    # into Bjoin we put in all DBjoin that
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
    if configBenchmarkJoinParameters != None:
      for paramDict in configBenchmarkJoinParameters:
        self.benchmarkJoinParameters.append(paramDict)
    else: # make empty
      self.benchmarkJoinParameters = [{"ProblemSizes": currentProblemSizes}]
    #print "0: ", self.benchmarkJoinParameters

    ############################################################################
    # (I-6) benchmark final sizes
    self.benchmarkFinalParameters = configBenchmarkFinalParameters
    # no other parameters besides problem sizes


    ############################################################################
    # (I-7) any default param with 1 value will be hardcoded; move to beginning
    for stepList in [self.benchmarkCommonParameters, \
        self.forkParameters, self.benchmarkForkParameters, \
        self.benchmarkJoinParameters]:
      for paramDict in copy(stepList):
        for paramName in copy(paramDict):
          paramValues = paramDict[paramName]
          if len(paramValues) < 2:
            paramDict.pop(paramName)
            #self.benchmarkCommonParameters.insert(0, {paramName: paramValues })
            self.hardcodedParameters[0][paramName] = paramValues[0]
            self.singleValueParameters[paramName] = [ paramValues[0] ]
            if len(paramDict) == 0:
              stepList.remove(paramDict)

    ############################################################################
    # (I-8) if fork and join, but no benchmark fork, append dummy benchmarkFork
    if len(self.forkParameters) > 0 and len(self.joinParameters) > 0 \
        and (len(self.benchmarkForkParameters) == 0 \
        or (len(self.benchmarkForkParameters) == 1 \
        and hasParam("ProblemSizes", self.benchmarkForkParameters)) ):
      self.benchmarkForkParameters.append({"BenchmarkFork": [0]})

    ############################################################################
    # (I-9) if join, but no benchmark join, append dummy benchmarkJoin
    #if len(self.joinParameters) > 0 \
    #    and (len(self.benchmarkJoinParameters) == 0 \
    #    or (len(self.benchmarkJoinParameters) == 1 \
    #    and hasParam("ProblemSizes", self.benchmarkJoinParameters)) ):
    #  self.benchmarkJoinParameters.append({"BenchmarkJoin": [0]})
    # No, this is handles by Final Benchmark

    ############################################################################
    # (I-10) Print Parameter Lists
    # benchmarkCommonParameters
    print "HardcodedParameters:"
    for paramName in self.hardcodedParameters[0]:
      paramValues = self.hardcodedParameters[0][paramName]
      print "    %s: %s" % (paramName, paramValues)
    print "BenchmarkCommonParameters:"
    for step in self.benchmarkCommonParameters:
      print "    %s" % step
    # forkParameters
    print "ForkParameters:"
    for param in self.forkParameters:
      print "    %s" % param
    # benchmarkForkParameters
    print "BenchmarkForkParameters:"
    for step in self.benchmarkForkParameters:
      print "    %s" % step
    # joinParameters
    print "JoinParameters:"
    for param in self.joinParameters:
      print "    %s" % param
    # benchmarkJoinParameters
    print "BenchmarkJoinParameters:"
    for step in self.benchmarkJoinParameters:
      print "    %s" % step
    # benchmarkJoinParameters
    print "BenchmarkfinalParameters:"
    for step in self.benchmarkFinalParameters:
      print "    %s" % step


  ##############################################################################
  # (II) convert lists of parameters to benchmark steps
  ##############################################################################
  def convertParametersToSteps(self):
    print ""
    print "####################################################################"
    print "# Convert Parameters to Steps"
    print "####################################################################"
    print ""

    ############################################################################
    # (II-1) benchmark common parameters
    print ""
    print "####################################################################"
    print "# Benchmark Common Parameters"
    self.addStepsForParameters( self.benchmarkCommonParameters  )

    ############################################################################
    # (II-2) fork parameters
    # calculate permutations of
    print ""
    print "####################################################################"
    print "# Fork Parameters"
    print self.forkParameters
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
      #print Solution.getNameFull(forkPermutations[i])
    if len(forkPermutations) > 0:
      self.forkHardcodedParameters(forkPermutations)

    ############################################################################
    # (II-3) benchmark common parameters
    print ""
    print "####################################################################"
    print "# Benchmark Fork Parameters"
    self.addStepsForParameters( self.benchmarkForkParameters  )

    ############################################################################
    # (II-4.1) join parameters
    # answer should go in hard-coded parameters
    # does it remove the prior forks? Yes.
    print ""
    print "####################################################################"
    print "# Join Parameters"
    macroTileJoinSet = set()
    depthUJoinSet = set()
    totalPermutations = 1
    for joinName in self.joinParameters:
      # joining a parameter with only a single value
      if hasParam(joinName, self.singleValueParameters):
        pass

      elif hasParam(joinName, self.forkParameters):
        # count permutations
        for param in self.forkParameters:
          for name in param: # only 1
            values = param[name]
            localPermutations = len(values)
            print "JoinParameter %s has %u possibilities" % (joinName, localPermutations)
            totalPermutations *= localPermutations

      ##########################################################################
      # (II-4.2) Join MacroTile
      elif joinName == "MacroTile":
        print "JoinParam: MacroTile"
        # get possible WorkGroupEdges from forked
        print "currentForkParameters = %s" % str(self.forkParameters)
        workGroupEdgeValues = []
        workGroupShapeValues = []
        threadTileEdgeValues = []
        threadTileShapeValues = []
        # todo having MacroTile as join parameter causes trouble if
        # one parameter is benchmarked rather than forked
        # however, this may still be the right way to do it

        # count permutations
        for paramList in [self.benchmarkCommonParameters, \
            self.forkParameters, self.benchmarkForkParameters, \
            self.benchmarkJoinParameters, self.singleValueParameters ]:
          if hasParam("WorkGroupEdge", paramList):
            workGroupEdgeValues = getParamValues("WorkGroupEdge", paramList)
          if hasParam("WorkGroupShape", paramList):
            workGroupShapeValues = getParamValues("WorkGroupShape", paramList)
          if hasParam("ThreadTileEdge", paramList):
            threadTileEdgeValues = getParamValues("ThreadTileEdge", paramList)
          if hasParam("ThreadTileShape", paramList):
            threadTileShapeValues = getParamValues("ThreadTileShape", paramList)
        macroTilePermutations = len(workGroupEdgeValues) \
            * len(workGroupShapeValues) * len(threadTileEdgeValues) \
            * len(threadTileShapeValues)
        printStatus("Total JoinMacroTile Permutations: %u" % macroTilePermutations)
        #print "MacroTile Parameters Found:"
        #print workGroupEdgeValues, workGroupShapeValues, threadTileEdgeValues, threadTileShapeValues

        # enumerate permutations
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

      ##########################################################################
      # (II-4.3) Join DepthU
      elif joinName == "DepthU":
        unrollValues = []
        splitUValues = []
        # count permutations
        for paramList in [self.benchmarkCommonParameters, \
            self.forkParameters, self.benchmarkForkParameters, \
            self.benchmarkJoinParameters, self.singleValueParameters ]:
          if hasParam("LoopUnroll", paramList):
            unrollValues = getParamValues("LoopUnroll", paramList)
          if hasParam("SplitU", paramList):
            splitUValues = getParamValues("SplitU", paramList)
        depthUPermutations = len(unrollValues)*len(splitUValues)
        printStatus("Total JoinDepthU Permutations: %u" % depthUPermutations)
        # enumerate permutations
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

    ############################################################################
    # (II-4.4) Enumerate Permutations Other * MacroTile * DepthU
    macroTiles = list(macroTileJoinSet)
    depthUs = list(depthUJoinSet)
    printStatus("TotalJoinPermutations = %u" % ( totalPermutations) )
    joinPermutations = []
    for i in range(0, totalPermutations):
      joinPermutations.append({})
      pIdx = i
      for joinName in self.joinParameters:
        if hasParam(joinName, self.forkParameters):
          for paramDict in self.forkParameters: # hardcodedPermutations
            if joinName in paramDict:
              paramValues = paramDict[joinName]
              valueIdx = pIdx % len(paramValues)
              joinPermutations[i][joinName] = paramValues[valueIdx]
              pIdx /= len(paramValues)
              break
        elif joinName == "MacroTile":
          valueIdx = pIdx % len(macroTiles)
          pIdx /= len(macroTiles)
          joinPermutations[i]["MacroTile0"] = macroTiles[valueIdx][0]
          joinPermutations[i]["MacroTile1"] = macroTiles[valueIdx][1]
          #Solution.assignDimsFromEdgeAndShape(joinPermutations[i])
        elif joinName == "DepthU":
          valueIdx = pIdx % len(depthUs)
          pIdx /= len(depthUs)
          joinPermutations[i][joinName] = depthUs[valueIdx]
      #print joinPermutations[i]
    #self.hardcodedParameters.append(joinPermutations)
    print "JoinPermutations: "
    for perm in joinPermutations:
      print Solution.getNameFull(perm)
    if len(joinPermutations) > 0:
      self.joinHardcodedParameters(joinPermutations)


    ############################################################################
    # (II-5) benchmark join parameters
    print ""
    print "####################################################################"
    print "# Benchmark Join Parameters"
    self.addStepsForParameters( self.benchmarkJoinParameters  )

    ############################################################################
    # (II-6) benchmark final
    print ""
    print "####################################################################"
    print "# Benchmark Final"
    self.currentProblemSizes = ProblemSizes(self.problemType, \
        self.benchmarkFinalParameters["ProblemSizes"])
    currentBenchmarkParameters = {}
    print "Adding BenchmarkStep for Final"
    benchmarkStep = BenchmarkStep(
        self.hardcodedParameters,
        currentBenchmarkParameters,
        self.initialSolutionParameters.state,
        self.currentProblemSizes,
        self.benchmarkStepIdx )
    self.benchmarkSteps.append(benchmarkStep)
    self.benchmarkStepIdx+=1


  ##############################################################################
  # For list of config parameters convert to steps and append to steps list
  ##############################################################################
  def addStepsForParameters(self, configParameterList):
    print configParameterList
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
        print "Adding BenchmarkStep for %s" % str(currentBenchmarkParameters)
        benchmarkStep = BenchmarkStep(
            self.hardcodedParameters,
            currentBenchmarkParameters,
            self.initialSolutionParameters.state,
            self.currentProblemSizes,
            self.benchmarkStepIdx )
        self.benchmarkSteps.append(benchmarkStep)
        self.benchmarkStepIdx+=1


  ##############################################################################
  # Add new permutations of hardcoded parameters to old permutations of params
  ##############################################################################
  def forkHardcodedParameters( self, update ):
    updatedHardcodedParameters = []
    for oldPermutation in self.hardcodedParameters:
      for newPermutation in update:
        permutation = {}
        permutation.update(oldPermutation)
        permutation.update(newPermutation)
        updatedHardcodedParameters.append(permutation)
    self.hardcodedParameters = updatedHardcodedParameters


  ##############################################################################
  # contract old permutations of hardcoded parameters based on new
  ##############################################################################
  def joinHardcodedParameters( self, update ):
    self.hardcodedParameters = update
    return


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
    #printStatus("Creating BenchmarkStep BP=%u HCP=%s" % ( str(benchmarkParameters), str(hardcodedParameters)))
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

    printStatus("Creating BenchmarkStep [BP]=%u [HCP]=%u [P]=%u" \
        % ( len(benchmarkParameters), len(hardcodedParameters), \
        problemSizes.totalProblemSizes))

  def abbreviation(self):
    string = "%02u" % self.stepIdx
    if len(self.benchmarkParameters) > 0:
      for param in self.benchmarkParameters:
        string += "_%s" % Solution.getParameterNameAbbreviation(param)
    else:
      string += "_Final"
    return string

  def __str__(self):
    string = "%02u" % self.stepIdx
    if len(self.benchmarkParameters) > 0:
      for param in self.benchmarkParameters:
        string += "_%s" % str(param)
    else:
      string += "_Final"
    return string
  def __repr__():
    return self.__str__()




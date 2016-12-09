import sys
from copy import *

from Common import *
from Structs import *

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

  def __init__(self, hardcodedParameters, readParameters, benchmarkParameters, initialSolutionParameters, problemSizes, idx):
    # what is my step Idx
    self.stepIdx = idx

    # what parameters don't need to be benchmarked because hard-coded or forked
    self.hardcodedParameters = deepcopy(hardcodedParameters)

    # what parameters do I need to read from prior steps
    self.readParameters = deepcopy(readParameters)

    # what parameters will I benchmark
    self.benchmarkParameters = deepcopy(benchmarkParameters)

    # what solution parameters do I use for what hasn't been benchmarked
    self.initialSolutionParameters = initialSolutionParameters

    # what problem sizes do I benchmark
    self.problemSizes = deepcopy(problemSizes)

    # what winners will I parse from my data

  def __str__(self):
    string = "  BenchmarkStep %u\n" % self.stepIdx
    string += "    HardCoded: %s\n" % self.hardcodedParameters
    string += "    Read: %s\n" % self.readParameters
    string += "    Benchmark: %s\n" % self.benchmarkParameters
    string += "    ProblemSizes: %s\n" % self.problemSizes
    return string
  def __repr__():
    return self.__str__()



################################################################################
# Benchmark Process
# steps in config need to be expanded and
# missing elements need to be assigned a default
################################################################################
class BenchmarkProcess:



  ##############################################################################
  def __init__(self, config):
    printStatus("Beginning")
    # read problem type
    if "ProblemType" in config:
      problemTypeConfig = config["ProblemType"]
    else:
      problemTypeConfig = {}
      printWarning("No ProblemType in config: %s; using defaults." % str(config) )
    self.problemType = ProblemType(problemTypeConfig)
    printStatus("Beginning %s" % str(self.problemType))

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
    self.benchmarkSteps = []
    self.fillInMissingStepsWithDefaults(config)

    # convert list of parameters to list of steps
    self.readParameters = []
    self.currentProblemSizes = []
    self.hardcodedParameters = []
    self.benchmarkStepIdx = 0
    self.convertParametersToSteps()


  ##############################################################################
  # convert lists of parameters to benchmark steps
  def convertParametersToSteps(self):

    # (1) benchmark common parameters
    self.addStepsForParameters( self.benchmarkCommonParameters  )

    # (2) fork parameters
    # calculate permutations of
    totalPermutations = 1
    for param in self.forkParameters:
      for name in param: # only 1
        values = param[name]
        totalPermutations *= len(values)
    permutations = []
    for i in range(0, totalPermutations):
      permutations.append([])
      pIdx = i
      for param in self.forkParameters:
        for name in param:
          values = param[name]
          valueIdx = pIdx % len(values)
          permutation = [name, values[valueIdx]]
          permutations[i].append(permutation)
          pIdx /= len(values)
      print permutations[i]
    self.hardcodedParameters.append(permutations)

    # (3) benchmark common parameters
    self.addStepsForParameters( self.benchmarkForkParameters  )

    # (4) join parameters
    # answer should go in hard-coded parameters
    # does it remove the prior forks? Yes.
    macroTileJoinSet = set()
    depthUJoinSet = set()
    totalPermutations = 1
    for joinName in self.joinParameters:
      # find in hardcoded; that's where forked will be
      if inListOfDictionaries(joinName, self.forkParameters):
        for param in self.forkParameters:
          for name in param: # only 1
            values = param[name]
            totalPermutations += len(values)
            print "JoinParameter %s has %u possibilities" % (joinName, len(values))
      elif joinName == "MacroTile":
        print "JoinParam: MacroTile"
        # get possible WorkGroupEdges from forked
        print self.forkParameters
        workGroupEdgeValues = []
        workGroupShapeValues = []
        threadTileEdgeValues = []
        threadTileShapeValues = []
        for paramList in [self.benchmarkCommonParameters, self.forkParameters, self.benchmarkForkParameters, self.benchmarkJoinParameters]:
          if inListOfDictionaries("WorkGroupEdge", paramList):
            workGroupEdgeValues = getValuesInListOfDictionaries("WorkGroupEdge", paramList)
          if inListOfDictionaries("WorkGroupShape", paramList):
            workGroupShapeValues = getValuesInListOfDictionaries("WorkGroupShape", paramList)
          if inListOfDictionaries("ThreadTileEdge", paramList):
            threadTileEdgeValues = getValuesInListOfDictionaries("ThreadTileEdge", paramList)
          if inListOfDictionaries("ThreadTileShape", paramList):
            threadTileShapeValues = getValuesInListOfDictionaries("ThreadTileShape", paramList)
        totalPermutations = len(workGroupEdgeValues)*len(workGroupShapeValues)*len(threadTileEdgeValues)*len(threadTileShapeValues)
        printStatus("TotalJoinPermutations: %u" % totalPermutations)

        for i in range(0, totalPermutations):
          pIdx = i
          workGroupEdgeIdx = pIdx % len(workGroupEdgeValues)
          pIdx /= len(workGroupEdgeValues)
          workGroupShapeIdx = pIdx % len(workGroupShapeValues)
          pIdx /= len(workGroupShapeValues)
          threadTileEdgeIdx = pIdx % len(threadTileEdgeValues)
          pIdx /= len(threadTileEdgeValues)
          threadTileShapeIdx = pIdx % len(threadTileShapeValues)
          pIdx /= len(threadTileShapeValues)
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
        printStatus("JoinSetSize: %u" % len(macroTileJoinSet) )
        print macroTileJoinSet



        # add macrotile to set
      elif joinName == "DepthU":
        print "JoinParam: DepthU"
        # get possible splitU
        # get possible unroll
        # add splitU*unroll to set
        pass
      else:
        validJoinNames = ["MacroTile", "DepthU"]
        for validParam in self.forkParameters:
          for validName in validParam: # only 1
            validJoinNames.append(validName)
        printExit("JoinParameter \"%s\" not in %s" % (joinName, validJoinNames) )





    # (5) benchmark common parameters
    self.addStepsForParameters( self.benchmarkJoinParameters  )
    """
    self.benchmarkForkParameters = []
    self.joinParameters = []
    self.benchmarkJoinParameters = []
    """

  ##############################################################################
  # for list of config parameters convert to steps and append to steps list
  def addStepsForParameters(self, configParameterList):
    for paramConfig in configParameterList:
      if isinstance(paramConfig, dict):
        if "ProblemSizes" in paramConfig:
          self.currentProblemSizes = ProblemSizes(self.problemType, paramConfig["ProblemSizes"])
          continue
      currentBenchmarkParameters = []
      for paramName in paramConfig:
        paramValues = paramConfig[paramName]
        if len(paramValues) == 1:
          self.hardcodedParameters.append([paramName, paramValues[0]])
        else:
          currentBenchmarkParameters.append([paramName, paramValues])
      if len(currentBenchmarkParameters) > 0:
        benchmarkStep = BenchmarkStep(
            self.hardcodedParameters,
            self.readParameters,
            currentBenchmarkParameters,
            self.initialSolutionParameters,
            self.currentProblemSizes,
            self.benchmarkStepIdx )
        self.benchmarkSteps.append(benchmarkStep)
        self.readParameters.append(currentBenchmarkParameters)
        self.benchmarkStepIdx+=1


  ##############################################################################
  # create thorough lists of parameters, filling in missing info from defaults
  def fillInMissingStepsWithDefaults(self, config):

    # get benchmark steps from config
    configBenchmarkCommonParameters = config["BenchmarkCommonParameters"] \
        if "BenchmarkCommonParameters" in config else [{"ProblemSizes": defaultProblemSizes}]
    configForkParameters = config["ForkParameters"] \
        if "ForkParameters" in config else []
    configBenchmarkForkParameters = config["BenchmarkForkParameters"] \
        if "BenchmarkForkParameters" in config else [{"ProblemSizes": defaultProblemSizes}]
    configJoinParameters = config["JoinParameters"] \
        if "JoinParameters" in config else []
    configBenchmarkJoinParameters = config["BenchmarkJoinParameters"] \
        if "BenchmarkJoinParameters" in config else [{"ProblemSizes": defaultProblemSizes}]

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
    for param in defaultBenchmarkCommonParameters:
      paramName = param[0]
      if not inListOfListOfDictionaries( paramName, [
          configBenchmarkCommonParameters, configForkParameters, configBenchmarkForkParameters,
          configJoinParameters, configBenchmarkJoinParameters]) \
          or paramName == "ProblemSizes":
        self.benchmarkCommonParameters.append({param[0]: param[1]})
    for param in configBenchmarkCommonParameters:
      self.benchmarkCommonParameters.append(param)

    ########################################
    # (2) into fork we put in all Dfork that
    # don't show up in Bcommon/Cfork/CBfork/Cjoin/CBjoin
    # followed by Cfork
    self.forkParameters = []
    for param in defaultForkParameters:
      paramName = param[0]
      if not inListOfListOfDictionaries( paramName, [
          self.benchmarkCommonParameters,
          configForkParameters, configBenchmarkForkParameters, configJoinParameters, configBenchmarkJoinParameters]) \
          or paramName == "ProblemSizes":
        self.forkParameters.append({param[0]: param[1]})
    for param in configForkParameters:
      self.forkParameters.append(param)

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
    for param in defaultBenchmarkForkParameters:
      paramName = param[0]
      if not inListOfListOfDictionaries( paramName, [
          self.benchmarkCommonParameters, self.forkParameters,
          configBenchmarkForkParameters, configJoinParameters, configBenchmarkJoinParameters]) \
          or paramName == "ProblemSizes":
        self.benchmarkForkParameters.append({param[0]: param[1]})
    for param in configBenchmarkForkParameters:
      self.benchmarkForkParameters.append(param)

    ########################################
    # (4) into join we put in all Djoin that
    # don't show up in Bcommon/Bfork/CBfork/Cjoin/CBjoin
    # followed by CBforked
    self.joinParameters = []
    for param in defaultJoinParameters:
      paramName = param[0]
      if not inListOfListOfDictionaries( paramName, [
          self.benchmarkCommonParameters, self.forkParameters, self.benchmarkForkParameters,
          configJoinParameters, configBenchmarkJoinParameters]) \
          or paramName == "ProblemSizes":
        self.joinParameters.append(param)
    for param in configJoinParameters:
      self.joinParameters.append(param)

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
    for param in defaultBenchmarkJoinParameters:
      paramName = param[0]
      if not inListOfListOfDictionaries( paramName, [
          self.benchmarkCommonParameters, self.forkParameters, self.benchmarkForkParameters, self.joinParameters,
          configBenchmarkJoinParameters]) \
          or paramName == "ProblemSizes":
        self.benchmarkJoinParameters.append({param[0]: param[1]})
    for param in configBenchmarkJoinParameters:
      self.benchmarkJoinParameters.append(param)


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

  def __str__(self):
    string = "BenchmarkProcess:\n"
    for step in self.benchmarkSteps:
      string += str(step)
    return string
  def __repr__(self):
    return self.__str__()






################################################################################
def benchmarkProblemType( config ):



  # convert cnofig to full benchmark process (resolves defaults)
  benchmarkProcess = BenchmarkProcess(config)
  # for step in benchmarkProcess:


  # default order of parameters
  # default of which values to benchmark
  # default fork
  # default join
  # parameters not mentioned anywhere are moved to before first fork


    # if Fork
      # expand winners
    # if Join
      # contract winners
    # if Problems
      # update problems
    # if Parameter
    # check if parameter already benchmarked and redo=False
    # create solution candidates
      # InitialSolutionParameters
      # DeterminedSolutionParameters
        # analyse prior step and clearly write out winners or prior step here
      # this step's list of parameters to benchmark
    # create benchmark executable
      # kernel and solution files
      # problem iteration file
      # cmake for generated files
      # copy static files
    # compile benchmark executable
    # run benchmark executable
  # Final BenchmarkSolutions

# benchmarking directory structure
# build/
#   BenchmarkProblemTypes
#     Cij_Aik_Bjk_SBOI
#       Step1
#       Step2
#       Final
#       Data
#         step1.csv
#         step1.yaml
#         step2.csv
#         step2.yaml
#         final.csv
#     Cij_Aik_Bkj_SBOI
#       ...
#   Analysis
#     Cij_Aik_Bjk_SBOI.yaml
#     Cij_Aik_Bjk_SBOI.yaml
#     LibName.yaml

def main(  config ):
  printStatus("Beginning")
  for problemType in config:
    if problemType is None:
      benchmarkProblemType({})
    else:
      benchmarkProblemType(problemType)
    if globalParameters["DebugPrintLevel"] >= 1:
      print ""

  printStatus("DONE.")

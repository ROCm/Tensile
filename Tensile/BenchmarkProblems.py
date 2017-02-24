import sys
import os
from copy import deepcopy
from copy import copy as shallowcopy
from shutil import copy as shutil_copy
from shutil import rmtree
import csv
from subprocess import Popen

from BenchmarkStructs import BenchmarkProcess
from Common import globalParameters, HR, pushWorkingPath, popWorkingPath, print1, print2, printExit, printWarning, ensurePath
from SolutionStructs import Solution, ProblemType
from SolutionWriter import SolutionWriter
from KernelWriter import KernelWriter
from ClientWriter import writeRunScript, writeClientParameters
from TensileCreateLibrary import writeSolutionsAndKernels, writeCMake
import YAMLIO



################################################################################
# Benchmark Problem Type
################################################################################
def benchmarkProblemType( config ):

  # convert config to full benchmark process (resolves defaults)
  print1("")
  print1(HR)
  print1("# Converting Config to BenchmarkProcess Object")
  print1(HR)
  print1("")
  benchmarkProcess = BenchmarkProcess(config)

  problemTypeName = str(benchmarkProcess.problemType)
  pushWorkingPath(problemTypeName)
  ensurePath(os.path.join(globalParameters["WorkingPath"],"Data"))

  totalBenchmarkSteps = len(benchmarkProcess)
  winners = WinningParameterDict()
  print1("# NumBenchmarkSteps: %u" % totalBenchmarkSteps)
  print1("")
  print1(HR)
  print1("# Done Creating BenchmarkProcess Object")
  print1(HR)

  ##############################################################################
  # For Each Benchmark Step
  ##############################################################################
  for benchmarkStepIdx in range(0, totalBenchmarkSteps):

    benchmarkStep = benchmarkProcess[benchmarkStepIdx]
    resultingHardcodedParameterList = \
        winners.update( benchmarkStep.hardcodedParameters )
    benchmarkStep.hardcodedParameters = resultingHardcodedParameterList
    numHardcoded = len(benchmarkStep.hardcodedParameters)
    stepName = str(benchmarkStep)
    shortName = benchmarkStep.abbreviation()
    print1("\n")
    print1(HR)
    print1("# %s\n# %s" % (problemTypeName, stepName))
    print1("# NumProblems: %u" % benchmarkStep.problemSizes.totalProblemSizes)
    print1("# BenchmarkParameters:")
    for paramName in benchmarkStep.benchmarkParameters:
      paramValues = benchmarkStep.benchmarkParameters[paramName]
      printStr = "#     %s = { %s" % (paramName, paramValues[0])
      for paramValueIdx in range(1, len(paramValues)):
        printStr += ", %s" % str(paramValues[paramValueIdx])
      printStr += " }"
      print1(printStr)

    if False:
    # print1(hardcoded parameters and their winners
      print1("# HardcodedParameters | WinningParameters:")
      paramDictIdx = 0
      hardcodedMinNaming = \
          Solution.getMinNaming(benchmarkStep.hardcodedParameters)
      for paramDict in benchmarkStep.hardcodedParameters:
        winningParameters = winners[paramDict]
        print1("#    (%u) %s | %s" % (paramDictIdx, \
            Solution.getNameMin(paramDict, hardcodedMinNaming), \
            Solution.getNameFull(winningParameters) ))
        paramDictIdx += 1
    pushWorkingPath(shortName)

    ############################################################################
    # Copy Files to Benchmark Source Directory
    ############################################################################
    pushWorkingPath("source")
    filesToCopy = [
        "Client.cpp",
        "Client.h",
        "CMakeLists.txt",
        "MathTemplates.cpp",
        "MathTemplates.h",
        "SetupTeardown.cpp",
        "TensileTypes.h",
        "ReferenceCPU.h",
        "SolutionHelper.cpp",
        "SolutionHelper.h",
        "Tools.cpp",
        "Tools.h",
        ]

    for f in filesToCopy:
      shutil_copy(
          os.path.join(globalParameters["SourcePath"], f),
          globalParameters["WorkingPath"] )
    #shutil_copy(
    #    os.path.join(globalParameters["SourcePath"], \
    #    "TensileBenchmark_CMakeLists.txt"),
    #    os.path.join(globalParameters["WorkingPath"], "CMakeLists.txt" ) )
    if globalParameters["Backend"] == "OCL":
      shutil_copy(
          os.path.join(globalParameters["SourcePath"], "FindOpenCL.cmake"),
          globalParameters["WorkingPath"] )
    else:
      shutil_copy(
          os.path.join(globalParameters["SourcePath"], "FindHIP.cmake"),
          globalParameters["WorkingPath"] )
      shutil_copy(
          os.path.join(globalParameters["SourcePath"], "FindHCC.cmake"),
          globalParameters["WorkingPath"] )

    ############################################################################
    # Enumerate Benchmark Permutations
    ############################################################################
    solutions = []
    totalBenchmarkPermutations = 1
    for benchmarkParamName in benchmarkStep.benchmarkParameters:
      totalBenchmarkPermutations *= len(benchmarkStep.benchmarkParameters[benchmarkParamName])
    maxPossibleSolutions = totalBenchmarkPermutations*numHardcoded
    print1("# MaxPossibleSolutions: %u = %u (hardcoded) * %u (benchmark)" % \
        (maxPossibleSolutions, numHardcoded, totalBenchmarkPermutations))

    benchmarkPermutations = []
    for i in range(0, totalBenchmarkPermutations):
      permutation = {}
      pIdx = i
      for benchmarkParamName in benchmarkStep.benchmarkParameters:
        benchmarkParamValues = deepcopy( \
            benchmarkStep.benchmarkParameters[benchmarkParamName])
        valueIdx = pIdx % len(benchmarkParamValues)
        permutation[benchmarkParamName] = benchmarkParamValues[valueIdx]
        pIdx /= len(benchmarkParamValues)
      benchmarkPermutations.append(permutation)

    ############################################################################
    # Enumerate Solutions = Hardcoded * Benchmark
    ############################################################################
    sys.stdout.write("# Enumerating Solutions")
    for hardcodedIdx in range(0, numHardcoded):
      solutions.append([])
      hardcodedParamDict = benchmarkStep.hardcodedParameters[hardcodedIdx]
      for benchmarkIdx in range(0, len(benchmarkPermutations)):
        benchmarkPermutation = benchmarkPermutations[benchmarkIdx]
        solution = {"ProblemType": deepcopy(benchmarkProcess.problemType.state)}
        solution.update(benchmarkPermutation)
        solution.update(hardcodedParamDict)
        winningParameters = winners[hardcodedParamDict]
        if winningParameters == None:
          # this is a joined parameter that didn't have a winner, that's okay
          continue
        solution.update(winningParameters)

        # append default parameters where necessary
        for initialSolutionParameterName in benchmarkStep.initialSolutionParameters:
          if initialSolutionParameterName not in solution:
            solution[initialSolutionParameterName] = \
                benchmarkStep.initialSolutionParameters[initialSolutionParameterName]
        # TODO check if solution matches problem size for exact tile kernels
        solutionObject = Solution(solution)
        if solutionObject["Valid"]:
          hasSolution = False
          for hardcodedSolutions in solutions:
            for hardcodedSolution in hardcodedSolutions:
              if hardcodedSolution == solutionObject:
                hasSolution = True
          if hasSolution:
            if globalParameters["PrintLevel"] >= 1:
              sys.stdout.write(":")
          else:
            solutions[hardcodedIdx].append(solutionObject)
            if globalParameters["PrintLevel"] >= 1:
              sys.stdout.write("|")
        else:
          if globalParameters["PrintLevel"] >= 1:
            sys.stdout.write(".")
        if globalParameters["PrintLevel"] >= 1:
          sys.stdout.flush()
        #else:
        #  print2("rejecting solution %s" % str(solutionObject))
    if globalParameters["PrintLevel"] >= 1:
      sys.stdout.write("\n")
      sys.stdout.flush()

    # remove hardcoded that don't have any valid benchmarks
    removeHardcoded = []
    for hardcodedIdx in range(0, numHardcoded):
      if len(solutions[hardcodedIdx]) == 0:
        hardcodedParamDict = benchmarkStep.hardcodedParameters[hardcodedIdx]
        removeHardcoded.append(hardcodedParamDict)
    removesExist = len(removeHardcoded) > 0
    for hardcodedParam in removeHardcoded:
      benchmarkStep.hardcodedParameters.remove(hardcodedParam)
    if removesExist:
      winners.update( benchmarkStep.hardcodedParameters )
      numHardcoded = len(benchmarkStep.hardcodedParameters )
      # remove from solution 2D list also
      for solutionList in shallowcopy(solutions):
        if len(solutionList) == 0:
          solutions.remove(solutionList)


    print1("# ActualSolutions: %u / %u" % ( len(solutions), \
        maxPossibleSolutions ))
    # create linear list
    solutionList = []
    for i in range(0, len(solutions)):
      solutionsForHardcoded = solutions[i]
      for j in range(0, len(solutionsForHardcoded)):
        solution = solutionsForHardcoded[j]
        solutionList.append(solution)
    if globalParameters["PrintLevel"] >= 1:
      for i in range(0, len(solutions)):
        solutionsForHardcoded = solutions[i]
        for j in range(0, len(solutionsForHardcoded)):
          solution = solutionsForHardcoded[j]
          print2("#    (%u:%u) %s" % (i, j, \
              Solution.getNameFull(solution) ))
      print2(HR)

    # write benchmarkFiles
    writeBenchmarkFiles(solutionList, benchmarkStep.problemSizes, \
        shortName, filesToCopy)
    popWorkingPath() # source

    ############################################################################
    # Run Benchmark Script
    ############################################################################
    resultsFileBase = os.path.normpath(os.path.join( \
        globalParameters["WorkingPath"], "../Data", shortName))
    resultsFileName = resultsFileBase + ".csv"
    solutionsFileName = resultsFileBase + ".yaml"
    if not os.path.exists(resultsFileName) or \
        globalParameters["ForceRedoBenchmarkProblems"]:
      # if redo=true, clobber the build directory
      if globalParameters["ForceRedoBenchmarkProblems"]:
        rmtree(os.path.join(globalParameters["WorkingPath"], "build"), \
            ignore_errors=True)
      pushWorkingPath("build")

      # write runScript
      libraryLogicPath = None
      path = globalParameters["WorkingPath"]
      forBenchmark = True
      runScriptName = writeRunScript(path, libraryLogicPath, forBenchmark)

      # run runScript
      process = Popen(runScriptName, cwd=globalParameters["WorkingPath"])
      process.communicate()
      if process.returncode:
        printWarning("Benchmark Process exited with code %u" % process.returncode)
      popWorkingPath() # build


    ############################################################################
    # Winners -> Determined Parameters
    ############################################################################
    results = getResults(resultsFileName, solutions)
    print2("CSV Results: %s" % results)
    winners.addResults(benchmarkStep.hardcodedParameters, \
        benchmarkPermutations, solutions, results)

    ############################################################################
    # Write Solutions YAML
    ############################################################################
    YAMLIO.writeSolutions(solutionsFileName, benchmarkStep.problemSizes, \
        solutions )

    # End Iteration
    popWorkingPath() # stepName
    print1("%s\n# %s\n# %s: End\n%s\n" \
        % (HR, problemTypeName, shortName, HR))

  popWorkingPath() # ProblemType
  return resultsFileBase
# End benchmarkProblemType()


################################################################################
# Read GFlop/s from file
################################################################################
def getResults(resultsFileName, solutions):
  try:
    resultsFile = open(resultsFileName, "r")
  except IOError:
    printExit("Can't open \"%s\" to get results" % resultsFileName )

  # setup data structures
  numSolutions = 0
  results = []
  for solutionsForHardcoded in solutions:
    results.append([])
    for solution in solutionsForHardcoded:
      problemSizeIdx = solution["ProblemType"]["TotalIndices"] + 1
      results[-1].append([])
      numSolutions += 1

  # read results in gflops
  csvFile = csv.reader(resultsFile)
  startIdx = problemSizeIdx + 1
  rowLength = startIdx + numSolutions

  rowIdx = 0
  for row in csvFile:
    rowIdx+=1
    if rowIdx == 1:
      continue
    else:
      if len(row) < rowLength:
        printWarning("CSV File %s row %u doesn't have %u elements; ignoring remainer of file." \
            % (resultsFileName, rowIdx, rowLength) )
        break
      idx = startIdx
      #for i in range(0, len(numBenchmarksPerHardcoded)):
      #  for j in range(0, numBenchmarksPerHardcoded[i]):
      for i in range(0, len(solutions)):
        solutionsForHardcoded = solutions[i]
        for j in range(0, len(solutionsForHardcoded)):
          solution = solutionsForHardcoded[j]
          gflops = float(row[idx])
          results[i][j].append(gflops)
          idx += 1
  if rowIdx < 2:
    printExit("CSV File %s only has %u row(s); prior benchmark must not have run long enough to produce data." \
        % (resultsFileName, rowIdx) )
  return results


################################################################################
# Write Benchmark Files
################################################################################
def writeBenchmarkFiles(solutions, problemSizes, stepName, filesToCopy):
  if not globalParameters["MergeFiles"]:
    ensurePath(os.path.join(globalParameters["WorkingPath"], "Solutions"))
    ensurePath(os.path.join(globalParameters["WorkingPath"], "Kernels"))

  ##############################################################################
  # Min Naming
  ##############################################################################
  kernels = []
  for solution in solutions:
    solutionKernels = solution.getKernels()
    for kernel in solutionKernels:
      if kernel not in kernels:
        kernels.append(kernel)

  solutionSerialNaming = Solution.getSerialNaming(solutions)
  kernelSerialNaming = Solution.getSerialNaming(kernels)
  solutionMinNaming = Solution.getMinNaming(solutions)
  kernelMinNaming = Solution.getMinNaming(kernels)
  solutionWriter = SolutionWriter( \
      solutionMinNaming, solutionSerialNaming, \
      kernelMinNaming, kernelSerialNaming)
  kernelWriter = KernelWriter( \
      kernelMinNaming, kernelSerialNaming)

  # write solution, kernels and CMake
  writeSolutionsAndKernels( \
      globalParameters["WorkingPath"], solutions, solutionWriter, kernelWriter)


  ##############################################################################
  # Write CMake
  ##############################################################################

  clientName = "TensileBenchmark_%s" % stepName
  writeCMake(globalParameters["WorkingPath"], solutions, filesToCopy, clientName)

  forBenchmark = True
  writeClientParameters(forBenchmark, solutions, problemSizes, stepName, filesToCopy)


################################################################################
# FrozenDictionary
################################################################################
class FrozenDictionary:
  def __init__(self, parameters):
    self.parameters = deepcopy(parameters)
    self.hashValue = hash(Solution.getNameFull(self.parameters))

  def __len__(self):
    return len(self.parameters)

  def __iter__(self):
    return iter(self.parameters)

  def __getitem__(self, key):
    return self.parameters[key]

  def __hash__(self):
    return self.hashValue

  def __str__(self):
    return Solution.getNameFull(self.parameters)
  def __repr__(self):
    return self.__str__();


################################################################################
# Winning Parameters For Hardcoded Parameters
################################################################################
class WinningParameterDict:

  ##########################################################
  # Init
  def __init__(self):
    self.winners = {}

  ##########################################################
  # Add Winning Parameters For Hardcoded Parameters
  def addResults( self, hardcodedParameterList, benchmarkPermutations, \
      solutions, results):
    for hardcodedIdx in range(0, len(results)):
      hardcodedResults = results[hardcodedIdx]
      hardcodedParameters = hardcodedParameterList[hardcodedIdx]
      winningIdx = -1
      winningScore = -9999 # -1 is score of invalid
      # find fastest benchmark parameters for this hardcoded
      for benchmarkIdx in range(0, len(hardcodedResults)):
        benchmarkResult = hardcodedResults[benchmarkIdx]
        benchmarkScore = max(benchmarkResult) # take fastest regardless of size
        if benchmarkScore > winningScore:
          winningScore = benchmarkScore
          winningIdx = benchmarkIdx
      winningSolution = solutions[hardcodedIdx][winningIdx]
      winningParameters = {}
      for paramName in benchmarkPermutations[0]:
        winningParameters[paramName] = winningSolution[paramName]
      print2("HCP[%u] Winner: idx=%u, gflops=%f, param=%s" \
          % ( hardcodedIdx, winningIdx, winningScore, winningParameters))
      matches = WinningParameterDict.get(hardcodedParameters, self.winners)
      if len(matches) != 1:
        printExit("Didn't find exactly 1 match")
      hardcodedParametersKey = matches[0][0]
      #oldWinningParameters = matches[0][1]
      #oldScore = matches[0][2]
      self.winners[hardcodedParametersKey][0].update(winningParameters)
      self.winners[hardcodedParametersKey][1] = winningScore


  ##########################################################
  # Get Winning Parameters For Hardcoded Parameters
  def __getitem__( self, hardcodedParameters ):
    #(hardcodedParametersKey, winningParameters, score) = \
    matches = WinningParameterDict.get(hardcodedParameters, self.winners)
    if len(matches) == 1:
      return matches[0][1]
    elif len(matches) == 0:
      return None
    else:
      printExit("Didn't find exactly 1 match")

  ##########################################################
  # Update Hardcoded Parameters In Winning Parameters
  # could be forking, joining or adding parameters to same hardcodeds
  def update(self, newHardcodedParameterList ):
    # TODO when new list is joining, we need to choose the fastest
    oldWinners = self.winners
    self.winners = {}

    # if this is first time, populate with dummies and early exit
    if len(oldWinners) == 0:
      for newHardcodedParameters in newHardcodedParameterList:
        self.winners[FrozenDictionary(newHardcodedParameters)] = [{},-1]
    else:
      for newHardcodedParameters in newHardcodedParameterList:
        #(oldHardcodedParameters, winningParameters, score) = \
        matches = WinningParameterDict.get(newHardcodedParameters, oldWinners)
        if len(matches) == 1: # plain update
          hardcodedFrozen = matches[0][0]
          winningParameters = matches[0][1]
          score = matches[0][2]
          #if winningParameters != None:
          newHardcodedParameters.update(hardcodedFrozen.parameters)
          self.winners[FrozenDictionary(newHardcodedParameters)] = \
              [ winningParameters, score ]
        elif len(matches) > 1: # join
          fastestScore = -1
          fastestHardcodedParameters = {}
          fastestWinningParameters = {}
          for matchIdx in range(0, len(matches)):
            match = matches[matchIdx]
            hardcodedFrozen = match[0]
            winningParameters = match[1]
            score = match[2]
            if score > fastestScore:
              fastestScore = score
              fastestWinningParameters = winningParameters
              fastestHardcodedParameters = hardcodedFrozen.parameters
          newHardcodedParameters.update(fastestHardcodedParameters)
          self.winners[FrozenDictionary(newHardcodedParameters)] = \
              [ fastestWinningParameters, fastestScore ]
        #else:
          #    % Solution.getNameFull(newHardcodedParameters)
          # TODO can I avoid reaching this code by not allowing join DepthU
          # probably not
          #printWarning("No Winners found for %s" \
          #    % Solution.getNameFull(newHardcodedParameters))
          #printExit("AVOID ME")
          #self.winners[FrozenDictionary(newHardcodedParameters)] = [{},-1]

    # return resulting hardcodedParameterList
    returnHardcodedParameterList = []
    for hardcodedFrozen in self.winners:
      returnHardcodedParameterList.append(hardcodedFrozen.parameters)
    return returnHardcodedParameterList

  ##########################################################
  # Get Winning Parameters For Hardcoded Parameters
  # need to match MacroTile and DepthU also
  @staticmethod
  def get( lookupHardcodedParameters, winners ):
    matches = []
    for hardcodedFrozen in winners:
      winningParameters = winners[hardcodedFrozen][0]
      #    % (Solution.getNameFull(lookupHardcodedParameters), \
      #    Solution.getNameFull(hardcodedFrozen.parameters), \
      #    Solution.getNameFull(winningParameters))
      score = winners[hardcodedFrozen][1]
      frozenMatch = True
      # a match if no key in hardcoded has a different value than lookup
      for paramName in hardcodedFrozen:
        if paramName in lookupHardcodedParameters:
          if lookupHardcodedParameters[paramName] != \
              hardcodedFrozen[paramName]:
            #    % (paramName, \
            #    lookupHardcodedParameters[paramName], hardcodedFrozen[paramName])
            frozenMatch = False
            break
      if frozenMatch:
        matchMacroTile = True
        matchDepthU = True
        matchUnion = {}
        matchUnion.update(hardcodedFrozen.parameters)
        matchUnion.update(winningParameters)
        if "MacroTile0" in lookupHardcodedParameters:
          lookupMacroTile0 = lookupHardcodedParameters["MacroTile0"]
          lookupMacroTile1 = lookupHardcodedParameters["MacroTile1"]
          #for paramName in hardcodedFrozen:
          #  paramValue = hardcodedFrozen[paramName]
          #  matchUnion[paramName] = paramValue
          Solution.assignProblemIndependentDerivedParameters(matchUnion)
          Solution.assignProblemIndependentDerivedParameters(hardcodedFrozen.parameters)
          if matchUnion["MacroTile0"] != lookupMacroTile0 \
              or matchUnion["MacroTile1"] != lookupMacroTile1:
            matchMacroTile = False
        if "DepthU" in lookupHardcodedParameters:
          lookupDepthU = lookupHardcodedParameters["DepthU"]
          matchDepthU = 1
          if "LoopUnroll" in matchUnion:
            matchDepthU *= matchUnion["LoopUnroll"]
          if "SplitU" in matchUnion:
            matchDepthU *= matchUnion["SplitU"]
          if matchDepthU != lookupDepthU:
            matchDepthU = False
          else:
            hardcodedFrozen.parameters["DepthU"] = lookupDepthU
        if matchMacroTile and matchDepthU:
          matches.append([hardcodedFrozen, winningParameters, score])
      else:
        pass

    return matches

  ##########################################################
  # To String
  def __str__(self):
    state = ""
    idx = 0
    for hardcodedParameters in self.winners:
      winningParameters = self.winners[hardcodedParameters][0]
      score = self.winners[hardcodedParameters][1]
      state += "  %2u: %s -> %s %f GFlop/s\n" % (idx, hardcodedParameters, \
          Solution.getNameFull(winningParameters), score)
      idx += 1
    return state
  def __repr__(self):
    return self.__str__()


################################################################################
# Main
################################################################################
def main( config ):
  dataPath = os.path.join(globalParameters["WorkingPath"], \
      globalParameters["BenchmarkDataPath"])
  pushWorkingPath(globalParameters["BenchmarkProblemsPath"])
  ensurePath(dataPath)
  for benchmarkProblemTypeConfig in config:
    problemTypeConfig = benchmarkProblemTypeConfig["ProblemType"]
    print2("ProblemTypeConfig: %s" % problemTypeConfig)
    problemTypeObj = ProblemType(problemTypeConfig)

    # Benchmark Problem Type
    if benchmarkProblemTypeConfig is None:
      resultsFileBase = benchmarkProblemType({})
    else:
      resultsFileBase = benchmarkProblemType(benchmarkProblemTypeConfig)

    # Copy Data
    resultsFileName = resultsFileBase + ".csv"
    solutionsFileName = resultsFileBase + ".yaml"
    newResultsFileName = os.path.join(dataPath, "%s.csv" % str(problemTypeObj))
    newSolutionsFileName = os.path.join(dataPath, "%s.yaml" % str(problemTypeObj))
    shutil_copy( resultsFileName, newResultsFileName )
    shutil_copy( solutionsFileName, newSolutionsFileName )

  popWorkingPath()

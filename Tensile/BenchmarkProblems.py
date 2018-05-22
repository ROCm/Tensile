################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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
import os, sys
from copy import deepcopy
from copy import copy as shallowcopy
import shutil
from shutil import copy as shutil_copy
from shutil import rmtree
import filecmp
import csv
from subprocess import Popen
import time

from BenchmarkStructs import BenchmarkProcess
from Common import globalParameters, HR, pushWorkingPath, popWorkingPath, print1, print2, printExit, printWarning, ensurePath, startTime, ProgressBar
from SolutionStructs import Solution, ProblemType
from SolutionWriter import SolutionWriter
from KernelWriterSource import KernelWriterSource
from KernelWriterAssembly import KernelWriterAssembly
from ClientWriter import writeRunScript, writeClientParameters
from TensileCreateLibrary import writeSolutionsAndKernels, writeCMake
import YAMLIO



################################################################################
# Benchmark Problem Type
################################################################################
def benchmarkProblemType( problemTypeConfig, problemSizeGroupConfig, \
    problemSizeGroupIdx ):

  benchmarkTestFails = 0

  # convert config to full benchmark process (resolves defaults)
  print1("")
  print1(HR)
  print1("# Converting Config to BenchmarkProcess Object")
  print1(HR)
  print1("")
  benchmarkProcess = BenchmarkProcess( problemTypeConfig, \
      problemSizeGroupConfig )

  problemTypeName = str(benchmarkProcess.problemType)
  problemSizeGroupName = "%s_%02u" % (problemTypeName, problemSizeGroupIdx)
  pushWorkingPath(problemSizeGroupName)
  ensurePath(os.path.join(globalParameters["WorkingPath"],"Data"))

  totalBenchmarkSteps = len(benchmarkProcess)
  resultsFileBaseFinal = None
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
    currentTime = time.time()
    elapsedTime = currentTime - startTime
    print1("# BenchmarkStep: %s - %s %.3fs" % (problemSizeGroupName, stepName, elapsedTime))
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
    sourceDir = \
      os.path.join(globalParameters["WorkingPath"], "source" )
    ensurePath(sourceDir)
    pushWorkingPath("sourceTmp")
    filesToCopy = [
        "Client.cpp",
        "Client.h",
        "CMakeLists.txt",
        "DeviceStats.h",
        "TensorUtils.h",
        "MathTemplates.cpp",
        "MathTemplates.h",
        "TensileTypes.h",
        "KernelHeader.h",
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
    if globalParameters["RuntimeLanguage"] == "OCL":
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
    print1("# Enumerating Solutions")
    if globalParameters["PrintLevel"] >= 1:
      progressBar = ProgressBar(maxPossibleSolutions)
    solutionSet = set() # avoid duplicates for nlca=-1, 1
    for hardcodedIdx in range(0, numHardcoded):
      solutions.append([])
      hardcodedParamDict = benchmarkStep.hardcodedParameters[hardcodedIdx]
      for benchmarkIdx in range(0, len(benchmarkPermutations)):
        benchmarkPermutation = benchmarkPermutations[benchmarkIdx]
        solution = {"ProblemType": deepcopy(benchmarkProcess.problemType.state)}
        solution.update(benchmarkPermutation)
        solution.update(hardcodedParamDict)
        if benchmarkStepIdx > 0:
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
          if solutionObject not in solutionSet:
            solutionSet.add(solutionObject)
            solutions[hardcodedIdx].append(solutionObject)
        else:
          if globalParameters["PrintSolutionRejectionReason"]:
            print1("rejecting solution %s" % str(solutionObject))
        if globalParameters["PrintLevel"] >= 1:
          progressBar.increment()

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
      if globalParameters["PrintLevel"] >= 1:
        print1("")
      numHardcoded = len(benchmarkStep.hardcodedParameters )
      # remove from solution 2D list also
      for solutionList in shallowcopy(solutions):
        if len(solutionList) == 0:
          solutions.remove(solutionList)
    print1("# Actual Solutions: %u / %u\n" % ( len(solutions), \
        maxPossibleSolutions ))


    # create linear list
    solutionList = []
    for i in range(0, len(solutions)):
      solutionsForHardcoded = solutions[i]
      for j in range(0, len(solutionsForHardcoded)):
        solution = solutionsForHardcoded[j]
        solutionList.append(solution)
    if len(solutionList) == 0:
        msg = "Your parameters resulted in 0 valid solutions."
        if globalParameters["PrintSolutionRejectionReason"]:
            msg += "\nExamine reject and backtrace messages above to see why and where solutions were rejected."
        else:
            msg += "\nYou should re-run with \"PrintSolutionRejectionReason: True\" to see why each parameter combination was rejected."
        printExit(msg)
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

    sourceTmp = globalParameters["WorkingPath"]
    files = os.listdir(sourceTmp)
    for f in files:
      f0 = os.path.join(sourceTmp, f)
      f1 = os.path.join(sourceDir, f)
      if os.path.isdir(f0):
        print "cpDir:", f0, f1
        if os.path.isdir(f1):
          shutil.rmtree( f1, True )
        shutil.copytree( f0, f1 )
      elif not os.path.exists(f1) or not filecmp.cmp(f0, f1):
        print "cp:", f0, f1
        shutil.copy( f0, f1 )

    popWorkingPath() # source

    ############################################################################
    # Run Benchmark Script
    ############################################################################
    resultsFileBase = os.path.normpath(os.path.join( \
        globalParameters["WorkingPath"], "../Data", shortName))
    if benchmarkStep.isFinal():
      resultsFileBaseFinal = resultsFileBase
    resultsFileName = resultsFileBase + ".csv"
    solutionsFileName = resultsFileBase + ".yaml"
    if not os.path.exists(resultsFileName) or \
        globalParameters["ForceRedoBenchmarkProblems"]:
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
        benchmarkTestFails += 1
        printWarning("BenchmarkProblems: Benchmark Process exited with code %u" % process.returncode)
      popWorkingPath() # build
    else:
      print1("# Already benchmarked; skipping.")


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
    currentTime = time.time()
    elapsedTime = currentTime - startTime
    print1("%s\n# %s\n# %s: End - %.3fs\n%s\n" \
        % (HR, problemSizeGroupName, shortName, elapsedTime, HR))

  popWorkingPath() # ProblemType
  return (resultsFileBaseFinal, benchmarkTestFails)
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
  kernelsBetaOnly = []
  for solution in solutions:
    solutionKernels = solution.getKernels()
    for kernel in solutionKernels:
      if kernel not in kernels:
        kernels.append(kernel)
    solutionKernelsBetaOnly = solution.getKernelsBetaOnly()
    for kernel in solutionKernelsBetaOnly:
      if kernel not in kernelsBetaOnly:
        kernelsBetaOnly.append(kernel)

  solutionSerialNaming = Solution.getSerialNaming(solutions)
  kernelSerialNaming = Solution.getSerialNaming(kernels)
  solutionMinNaming = Solution.getMinNaming(solutions)
  kernelMinNaming = Solution.getMinNaming(kernels)
  solutionWriter = SolutionWriter( \
      solutionMinNaming, solutionSerialNaming, \
      kernelMinNaming, kernelSerialNaming)
  kernelWriterSource = KernelWriterSource( \
      kernelMinNaming, kernelSerialNaming)
  kernelWriterAssembly = KernelWriterAssembly( \
      kernelMinNaming, kernelSerialNaming)

  # write solution, kernels and CMake
  writeSolutionsAndKernels( \
      globalParameters["WorkingPath"], solutions, kernels, kernelsBetaOnly, \
      solutionWriter, kernelWriterSource, kernelWriterAssembly )

  ##############################################################################
  # Write CMake
  ##############################################################################

  clientName = "TensileBenchmark_%s" % stepName
  writeCMake(globalParameters["WorkingPath"], solutions, kernels, filesToCopy, \
      clientName)

  forBenchmark = True
  writeClientParameters(forBenchmark, solutions, problemSizes, stepName, \
      filesToCopy)


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
    if globalParameters["PrintLevel"] >= 1:
      print1("# Adding Results to Solution Database")
      progressBar = ProgressBar(len(results))
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
      #print2("HCP[%u] Winner: idx=%u, gflops=%f, param=%s" \
      #    % ( hardcodedIdx, winningIdx, winningScore, winningParameters))
      matches = WinningParameterDict.get(hardcodedParameters, self.winners)
      if len(matches) != 1:
        printExit("Didn't find exactly 1 match")
      hardcodedParametersKey = matches[0][0]
      #oldWinningParameters = matches[0][1]
      #oldScore = matches[0][2]
      self.winners[hardcodedParametersKey][0].update(winningParameters)
      self.winners[hardcodedParametersKey][1] = winningScore
      if globalParameters["PrintLevel"] >= 1:
        progressBar.increment()


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
      if globalParameters["PrintLevel"] >= 1:
        print1("# Updating Solution Database")
        progressBar = ProgressBar(len(newHardcodedParameterList))
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
        if globalParameters["PrintLevel"] >= 1:
          progressBar.increment()


    # return resulting hardcodedParameterList
    returnHardcodedParameterList = []
    for hardcodedFrozen in self.winners:
      returnHardcodedParameterList.append(hardcodedFrozen.parameters)
    return returnHardcodedParameterList

  ##########################################################
  # Get Winning Parameters For Hardcoded Parameters
  # need to match MacroTile also
  @staticmethod
  def get( lookupHardcodedParameters, winners ):
    matches = []

    # only 1 winner, when benchmarking 1 solution
    if len(winners) == 1:
      hardcodedFrozen = winners.keys()[0]
      winningParameters = winners[hardcodedFrozen][0]
      score = winners[hardcodedFrozen][1]
      matches.append([hardcodedFrozen, winningParameters, score])
      return matches

    for hardcodedFrozen in winners:
      winningParameters = winners[hardcodedFrozen][0]
      score = winners[hardcodedFrozen][1]
      frozenMatch = True
      # a match if no key in hardcoded has a different value than lookup
      for paramName in hardcodedFrozen:
        if paramName in lookupHardcodedParameters:
          if lookupHardcodedParameters[paramName] != \
              hardcodedFrozen[paramName]:
            frozenMatch = False
            break
      if frozenMatch:
        matchMacroTile = True
        matchUnion = {}
        matchUnion.update(hardcodedFrozen.parameters)
        matchUnion.update(winningParameters)
        if "MacroTile0" in lookupHardcodedParameters:
          lookupMacroTile0 = lookupHardcodedParameters["MacroTile0"]
          lookupMacroTile1 = lookupHardcodedParameters["MacroTile1"]
          Solution.assignProblemIndependentDerivedParameters(matchUnion)
          Solution.assignProblemIndependentDerivedParameters(hardcodedFrozen.parameters)
          if matchUnion["MacroTile0"] != lookupMacroTile0 \
              or matchUnion["MacroTile1"] != lookupMacroTile1:
            matchMacroTile = False
        if matchMacroTile:
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
  totalTestFails = 0
  for benchmarkProblemTypeConfig in config:
    problemTypeConfig = benchmarkProblemTypeConfig[0]
    if len(benchmarkProblemTypeConfig) < 2:
      problemSizeGroupConfigs = [{}]
    else:
      problemSizeGroupConfigs = benchmarkProblemTypeConfig[1:]
    for problemSizeGroupIdx in range(0, len(problemSizeGroupConfigs)):
      problemSizeGroupConfig = problemSizeGroupConfigs[problemSizeGroupIdx]
      print2("ProblemTypeConfig: %s" % problemTypeConfig)
      problemTypeObj = ProblemType(problemTypeConfig)
      globalParameters["EnableHalf"] = problemTypeObj["DataType"].isHalf()

      # results files will be named
      newResultsFileName = os.path.join(dataPath, "%s_%02u.csv" \
          % (str(problemTypeObj), problemSizeGroupIdx) )
      newSolutionsFileName = os.path.join(dataPath, "%s_%02u.yaml" \
          % (str(problemTypeObj), problemSizeGroupIdx) )

      # skip if possible
      if globalParameters["ForceRedoBenchmarkProblems"] or \
          not os.path.exists(newResultsFileName):

        # Benchmark Problem Size Group
        (resultsFileBaseFinal, benchmarkErrors) = benchmarkProblemType(problemTypeConfig, \
            problemSizeGroupConfig, problemSizeGroupIdx)
        totalTestFails += benchmarkErrors
        print "totalTestFails=", totalTestFails

        # Copy Data
        resultsFileBase = resultsFileBaseFinal
        resultsFileName = "%s.csv" % (resultsFileBase)
        solutionsFileName = "%s.yaml" % (resultsFileBase)
        shutil_copy( resultsFileName, newResultsFileName )
        shutil_copy( solutionsFileName, newSolutionsFileName )
      else:
        print1("# %s_%02u already benchmarked; skipping." % (str(problemTypeObj), problemSizeGroupIdx) )

  popWorkingPath()

  if globalParameters["ExitOnFails"] and totalTestFails:
    sys.exit(1)

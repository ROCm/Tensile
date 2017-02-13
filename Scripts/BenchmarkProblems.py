import sys
import os
from copy import deepcopy
from copy import copy as shallowcopy
from shutil import copy as shutil_copy
from shutil import rmtree
import csv
from subprocess import Popen

from BenchmarkStructs import *
from Common import *
from SolutionStructs import *
from SolutionWriter import *
from KernelWriter import *
from ClientWriter import *
import LibraryWriter
import YAMLIO


################################################################################
# Benchmark Problem Type
################################################################################
def benchmarkProblemType( config ):

  # convert config to full benchmark process (resolves defaults)
  print ""
  print HR
  print "# Converting Config to BenchmarkProcess Object"
  print HR
  print ""
  benchmarkProcess = BenchmarkProcess(config)

  problemTypeName = str(benchmarkProcess.problemType)
  pushWorkingPath(problemTypeName)
  ensurePath(os.path.join(globalParameters["WorkingPath"],"Data"))

  totalBenchmarkSteps = len(benchmarkProcess)
  winners = WinningParameterDict()
  determinedParameters = [{}] # winner chosen from benchmark
  printStatus("NumBenchmarkSteps: %u" % totalBenchmarkSteps)
  print ""
  print HR
  print "# Done Creating BenchmarkProcess Object"
  print HR

  ##############################################################################
  # For Each Benchmark Step
  ##############################################################################
  for benchmarkStepIdx in range(0, totalBenchmarkSteps):

    # Print Step Name
    benchmarkStep = benchmarkProcess[benchmarkStepIdx]
    #print "Updating winners with new hardcodeds"
    resultingHardcodedParameterList = \
        winners.update( benchmarkStep.hardcodedParameters )
    benchmarkStep.hardcodedParameters = resultingHardcodedParameterList
    numHardcoded = len(benchmarkStep.hardcodedParameters)
    stepName = str(benchmarkStep)
    shortName = benchmarkStep.abbreviation()
    print "\n\n"
    print HR
    print "# %s\n# %s" % (problemTypeName, stepName)
    print "# NumProblems: %u" % benchmarkStep.problemSizes.totalProblemSizes
    print "# BenchmarkParameters:"
    for paramName in benchmarkStep.benchmarkParameters:
      paramValues = benchmarkStep.benchmarkParameters[paramName]
      printStr = "#     %s = { %s" % (paramName, paramValues[0])
      for paramValueIdx in range(1, len(paramValues)):
        printStr += ", %s" % str(paramValues[paramValueIdx])
      printStr += " }"
      print printStr

    if True:
    # print hardcoded parameters and their winners
      print "# HardcodedParameters | WinningParameters:"
      paramDictIdx = 0
      #print "hardcodedParameters: %s" % benchmarkStep.hardcodedParameters
      hardcodedMinNaming = \
          Solution.getMinNaming(benchmarkStep.hardcodedParameters)
      for paramDict in benchmarkStep.hardcodedParameters:
        winningParameters = winners[paramDict]
        print "#    (%u) %s | %s" % (paramDictIdx, \
            Solution.getNameMin(paramDict, hardcodedMinNaming), \
            Solution.getNameFull(winningParameters) )
        paramDictIdx += 1
      #for paramName in paramDict:
      #  paramValue = paramDict[paramName]
      #  printStr += "%s: %s, " % (paramName, str(paramValue))
      #print printStr
    #print HR
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
    currentSolution = {"ProblemType": deepcopy(benchmarkProcess.problemType.state) }
    totalBenchmarkPermutations = 1
    for benchmarkParamName in benchmarkStep.benchmarkParameters:
      totalBenchmarkPermutations *= len(benchmarkStep.benchmarkParameters[benchmarkParamName])
    maxPossibleSolutions = totalBenchmarkPermutations*numHardcoded
    print "# MaxPossibleSolutions: %u = %u (hardcoded) * %u (benchmark)" % \
        (maxPossibleSolutions, numHardcoded, totalBenchmarkPermutations)

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
      #print "BenchmarkPermutation: %s" % permutation
      benchmarkPermutations.append(permutation)

    ############################################################################
    # Enumerate Solutions = Hardcoded * Benchmark
    ############################################################################
    for hardcodedIdx in range(0, numHardcoded):
      solutions.append([])
      hardcodedParamDict = benchmarkStep.hardcodedParameters[hardcodedIdx]
      for benchmarkIdx in range(0, len(benchmarkPermutations)):
        benchmarkPermutation = benchmarkPermutations[benchmarkIdx]
        solution = {"ProblemType": deepcopy(benchmarkProcess.problemType.state)}
        solution.update(benchmarkPermutation)
        solution.update(hardcodedParamDict)
        #print "SolutionParameters: %s" % solution
        winningParameters = winners[hardcodedParamDict]
        if winningParameters == None:
          # this is a joined parameter that didn't have a winner, that's okay
          continue
        solution.update(winningParameters)
        # print solution

        # append default parameters where necessary
        for initialSolutionParameterName in benchmarkStep.initialSolutionParameters:
          if initialSolutionParameterName not in solution:
            solution[initialSolutionParameterName] = benchmarkStep.initialSolutionParameters[initialSolutionParameterName]
        # TODO check if solution matches problem size for exact tile kernels
        solutionObject = Solution(solution)
        if SolutionWriter.solutionParametersConsistent(solutionObject):
          #printStatus("appending solution %s" % str(solutionObject))
          solutions[hardcodedIdx].append(solutionObject)
        #else:
        #  printStatus("rejecting solution %s" % str(solutionObject))

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
      #print "Updating winners after eliminating some hardcodeds"
      winners.update( benchmarkStep.hardcodedParameters )
      numHardcoded = len(benchmarkStep.hardcodedParameters )
      # remove from solution 2D list also
      for solutionList in shallowcopy(solutions):
        if len(solutionList) == 0:
          solutions.remove(solutionList)


    print "# NumActualSolutions: %u / %u" % ( len(solutions), \
        maxPossibleSolutions )
    # create linear list
    solutionList = []
    for i in range(0, len(solutions)):
      solutionsForHardcoded = solutions[i]
      for j in range(0, len(solutionsForHardcoded)):
        solution = solutionsForHardcoded[j]
        solutionList.append(solution)
    solutionsMinNaming = Solution.getMinNaming(solutionList)
    for i in range(0, len(solutions)):
      solutionsForHardcoded = solutions[i]
      #print "HC: %s" \
      #    % Solution.getNameMin(benchmarkStep.hardcodedParameters[i], \
      #    hardcodedMinNaming)
      for j in range(0, len(solutionsForHardcoded)):
        solution = solutionsForHardcoded[j]
        print "#    (%u:%u) %s" % (i, j, \
            Solution.getNameMin(solution, solutionsMinNaming) )
    print HR

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

      # create run.bat or run.sh which builds and runs
      runScriptName = os.path.join(globalParameters["WorkingPath"], \
        "run.%s" % ("bat" if os.name == "nt" else "sh") )
      runScriptFile = open(runScriptName, "w")
      echoLine = "@echo." if os.name == "nt" else "echo"
      if os.name != "nt":
        runScriptFile.write("#!/bin/sh\n")
      runScriptFile.write("%s & echo %s & echo # %s & echo # %s: Configuring CMake & echo %s\n" \
          % (echoLine, HR, problemTypeName, stepName, HR))
      if os.name == "nt":
        runScriptFile.write("cmake -DTensile_CLIENT_BENCHMARK=ON -DTensile_MERGE_FILES=%s -DCMAKE_GENERATOR_PLATFORM=x64 ../source\n" % ("ON" if globalParameters["MergeFiles"] else "OFF") )
      else:
        runScriptFile.write("cmake ../source\n")
      runScriptFile.write("%s & echo %s & echo # %s & echo # %s: Building Benchmark & echo %s\n" \
          % (echoLine, HR, problemTypeName, stepName, HR))
      runScriptFile.write("cmake --build . --config %s%s\n" \
          % (globalParameters["CMakeBuildType"], " -- -j 8" if os.name != "nt" else "") )
      runScriptFile.write("%s & echo %s & echo # %s & echo # %s: Running Benchmark & echo %s\n" \
          % (echoLine, HR, problemTypeName, stepName, HR))
      if os.name == "nt":
        runScriptFile.write(os.path.join(globalParameters["CMakeBuildType"],"TensileBenchmark_%s.exe") \
            % (shortName) )
      else:
        runScriptFile.write("./TensileBenchmark_%s" % (shortName))
      runScriptFile.close()
      if os.name != "nt":
        os.chmod(runScriptName, 0777)
      # wait for python to finish printing
      process = Popen(runScriptName, cwd=globalParameters["WorkingPath"])
      status = process.communicate()
      popWorkingPath() # build


    ############################################################################
    # Winners -> Determined Parameters
    ############################################################################
    results = getResults(resultsFileName, solutions)
    print "CSV Results: %s" % results
    winners.addResults(benchmarkStep.hardcodedParameters, \
        benchmarkPermutations, solutions, results)

    ############################################################################
    # Write Solutions YAML
    ############################################################################
    YAMLIO.writeSolutions(solutionsFileName, benchmarkStep.problemSizes, \
        solutions )

    #solutionsFromFile = YAMLIO.readSolutions(solutionYAMLFileName)
    #solutionsMinNaming = Solution.getMinNaming(solutionsFromFile)
    #for solution in solutionsFromFile:
    #  print Solution.getNameMin(solution, solutionsMinNaming)

    # End Iteration
    popWorkingPath() # stepName
    print "%s\n# %s\n# %s: End\n%s\n" \
        % (HR, problemTypeName, shortName, HR)

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
      totalFlops = float(row[problemSizeIdx])
      idx = startIdx
      #for i in range(0, len(numBenchmarksPerHardcoded)):
      #  for j in range(0, numBenchmarksPerHardcoded[i]):
      for i in range(0, len(solutions)):
        solutionsForHardcoded = solutions[i]
        for j in range(0, len(solutionsForHardcoded)):
          solution = solutionsForHardcoded[j]
          gflops = float(row[idx])
          #time_ms = float(row[idx])
          #flops = totalFlops / (time_ms / 1000)
          #gflops = flops / (1000*1000*1000)
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
  printStatus("Beginning")
  if not globalParameters["MergeFiles"]:
    ensurePath(os.path.join(globalParameters["WorkingPath"], "Solutions"))
    ensurePath(os.path.join(globalParameters["WorkingPath"], "Kernels"))

  ##############################################################################
  # Min Naming
  ##############################################################################
  solutionFileNames = []
  kernelNames = []
  kernels = []
  for solution in solutions:
    solutionKernels = solution.getKernels()
    for kernel in solutionKernels:
      if kernel not in kernels:
        kernels.append(kernel)

  if globalParameters["ShortFileNames"] and not globalParameters["MergeFiles"]:
    solutionSerialNaming = Solution.getSerialNaming(solutions)
    kernelSerialNaming = Solution.getSerialNaming(kernels)
  else:
    solutionSerialNaming = None
    kernelSerialNaming = None
  solutionMinNaming = Solution.getMinNaming(solutions)
  kernelMinNaming = Solution.getMinNaming(kernels)
  solutionWriter = SolutionWriter( \
      solutionMinNaming, solutionSerialNaming, \
      kernelMinNaming, kernelSerialNaming)
  kernelWriter = KernelWriter( \
      kernelMinNaming, kernelSerialNaming)

  # write solution, kernels and CMake
  LibraryWriter.writeSolutionsAndKernels( \
      globalParameters["WorkingPath"], solutions, solutionWriter, kernelWriter)


  ##############################################################################
  # Write CMake
  ##############################################################################
  generatedFile = open(os.path.join(globalParameters["WorkingPath"], \
      "Generated.cmake"), "w")
  generatedFile.write(CMakeHeader)
  generatedFile.write("set( TensileBenchmark_Solutions\n")
  # write solution names
  if globalParameters["MergeFiles"]:
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions.h\n")
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions.cpp\n")
  else:
    for solutionFileName in solutionFileNames:
      generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions/%s.h\n" \
          % (solutionFileName) )
      generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions/%s.cpp\n" \
          % (solutionFileName) )
  generatedFile.write("  )\n")

  # write kernel names
  generatedFile.write("set( TensileBenchmark_Kernels\n")
  if globalParameters["MergeFiles"]:
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Kernels.h\n")
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Kernels.cpp\n")
  else:
    for kernelName in kernelNames:
      generatedFile.write("  ${CMAKE_SOURCE_DIR}/Kernels/%s.h\n" % (kernelName))
      generatedFile.write("  ${CMAKE_SOURCE_DIR}/Kernels/%s.cpp\n" % kernelName)
  generatedFile.write("  )\n")

  generatedFile.write("set( TensileBenchmark_Source\n")
  for fileName in filesToCopy:
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/%s\n" % fileName)
  generatedFile.write("  )\n\n")

  # benchmark parameters
  generatedFile.write("set( ClientName TensileBenchmark_%s)\n" \
      % (stepName) )
  generatedFile.write("set( Tensile_BACKEND \"%s\")\n" \
      % (globalParameters["Backend"]) )

  # build parameters
  generatedFile.write("set( CMAKE_BUILD_TYPE \"%s\")\n" \
      % (globalParameters["CMakeBuildType"]) )

  # close generated cmake
  generatedFile.close()

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
    printStatus("beginning")
    #print benchmarkPermutations
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
      print "HCP[%u] Winner: idx=%u, gflops=%f, param=%s" \
          % ( hardcodedIdx, winningIdx, winningScore, winningParameters)
      matches = WinningParameterDict.get(hardcodedParameters, self.winners)
      if len(matches) != 1:
        printExit("Didn't find exactly 1 match")
      hardcodedParametersKey = matches[0][0]
      oldWinningParameters = matches[0][1]
      oldScore = matches[0][2]
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
      happy += 1
      printExit("Didn't find exactly 1 match")

  ##########################################################
  # Update Hardcoded Parameters In Winning Parameters
  # could be forking, joining or adding parameters to same hardcodeds
  def update(self, newHardcodedParameterList ):
    #print "WinnerObj::update()"
    # TODO when new list is joining, we need to choose the fastest
    oldWinners = self.winners
    self.winners = {}

    # if this is first time, populate with dummies and early exit
    if len(oldWinners) == 0:
      for newHardcodedParameters in newHardcodedParameterList:
        self.winners[FrozenDictionary(newHardcodedParameters)] = [{},-1]
    else:
      for newHardcodedParameters in newHardcodedParameterList:
        #print "New: ", Solution.getNameFull(newHardcodedParameters)
        #(oldHardcodedParameters, winningParameters, score) = \
        matches = WinningParameterDict.get(newHardcodedParameters, oldWinners)
        #print "Found %u matches" % len(matches)
        if len(matches) == 1: # plain update
          #print "Update Plain"
          hardcodedFrozen = matches[0][0]
          winningParameters = matches[0][1]
          score = matches[0][2]
          #if winningParameters != None:
          newHardcodedParameters.update(hardcodedFrozen.parameters)
          #print "Score: %u" % score
          self.winners[FrozenDictionary(newHardcodedParameters)] = \
              [ winningParameters, score ]
        elif len(matches) > 1: # join
          #print "Update Join %u matches for %s" % (len(matches), Solution.getNameFull(newHardcodedParameters) )
          fastestIdx = -1
          fastestScore = -1
          fastestHardcodedParameters = {}
          fastestWinningParameters = {}
          for matchIdx in range(0, len(matches)):
            match = matches[matchIdx]
            hardcodedFrozen = match[0]
            winningParameters = match[1]
            score = match[2]
            #print "    %s -> %s %f GFlop/s" % (Solution.getNameFull(hardcodedParameters), Solution.getNameFull(winningParameters), score)
            if score > fastestScore:
              fastestScore = score
              fastestIdx = matchIdx
              fastestWinningParameters = winningParameters
              fastestHardcodedParameters = hardcodedFrozen.parameters
          newHardcodedParameters.update(fastestHardcodedParameters)
          #print "FastestScore: %u" % fastestScore
          #print "FastestHardcoded: %s" % Solution.getNameFull(fastestHardcodedParameters)
          self.winners[FrozenDictionary(newHardcodedParameters)] = \
              [ fastestWinningParameters, fastestScore ]
        #else:
          #print "No Winners found for %s" \
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
    #print "  GET looks: %s" % Solution.getNameFull(lookupHardcodedParameters)
    matches = []
    for hardcodedFrozen in winners:
      winningParameters = winners[hardcodedFrozen][0]
      #print "Evaluating %s == %s+%s" \
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
            #print "%s: lookup %s != hardcoded %s" \
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
          #print hardcodedFrozen
          #for paramName in hardcodedFrozen:
          #  paramValue = hardcodedFrozen[paramName]
          #  matchUnion[paramName] = paramValue
          Solution.assignDimsFromEdgeAndShape(matchUnion)
          Solution.assignDimsFromEdgeAndShape(hardcodedFrozen.parameters)
          if matchUnion["MacroTile0"] != lookupMacroTile0 \
              or matchUnion["MacroTile1"] != lookupMacroTile1:
            #print "MacroTile NOT Matched"
            matchMacroTile = False
          #else:
            #print "MacroTile Matched"
        if "DepthU" in lookupHardcodedParameters:
          lookupDepthU = lookupHardcodedParameters["DepthU"]
          matchDepthU = 1
          if "LoopUnroll" in matchUnion:
            matchDepthU *= matchUnion["LoopUnroll"]
          if "SplitU" in matchUnion:
            matchDepthU *= matchUnion["SplitU"]
          if matchDepthU != lookupDepthU:
            #print "DepthU NOT Matched"
            matchDepthU = False
          else:
            hardcodedFrozen.parameters["DepthU"] = lookupDepthU
            #print "DepthU Matched"
        if matchMacroTile and matchDepthU:
          #print "Match: %s" % Solution.getNameFull(hardcodedFrozen.parameters)
          matches.append([hardcodedFrozen, winningParameters, score])
      else:
        #print "Frozen NOT Matched"
        pass

    return matches

  ##########################################################
  # To String
  def __str__(self):
    state = ""
    idx = 0
    for hardcodedParameters in self.winners:
      #print self.winners[hardcodedParameters]
      winningParameters = self.winners[hardcodedParameters][0]
      score = self.winners[hardcodedParameters][1]
      #print score
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
  printStatus("Beginning")
  dataPath = os.path.join(globalParameters["WorkingPath"], \
      globalParameters["BenchmarkDataPath"])
  pushWorkingPath(globalParameters["BenchmarkProblemsPath"])
  ensurePath(dataPath)
  for benchmarkProblemTypeConfig in config:
    problemTypeConfig = benchmarkProblemTypeConfig["ProblemType"]
    print problemTypeConfig
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

  printStatus("DONE.")
  popWorkingPath()

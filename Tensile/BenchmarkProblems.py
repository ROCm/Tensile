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

import csv
import itertools
import os
import shutil
import sys
import time

from copy import deepcopy

from . import ClientExecutable
from . import SolutionLibrary
from . import LibraryIO
from . import Utils
from .BenchmarkStructs import BenchmarkProcess, constructForkPermutations, checkForValidParameters
from .ClientWriter import runClient, writeClientConfig
from .Common import globalParameters, HR, pushWorkingPath, popWorkingPath, print1, print2, printExit, printWarning, ensurePath, \
                    startTime, validParameters
from .KernelWriterAssembly import KernelWriterAssembly
from .KernelWriterSource import KernelWriterSource
from .SolutionStructs import Solution, ProblemType, ProblemSizes
from .SolutionWriter import SolutionWriter
from .TensileCreateLibrary import writeSolutionsAndKernels, writeCMake, buildObjectFileNames
from .CustomKernels import getCustomKernelConfig

############################################################################
# generateForkedSolutions
############################################################################
def generateForkedSolutions (problemType, hardcodedParameters, benchmarkPermutations, initialSolutionParameters=None):
  """this creates a set or solutions based on the forked parameters using
     a set of common parameters from which to fork from

  Parameters:
  problemType the problem type
  hardcodedParameters the set of parameters which overrides the baseline parameters
  benchmarkPermutations set of baseline parameters from which the the updates are branched form
  winners previous winning parameters which overrides the derived parameters
  initialSolutionParameters set of parameters which fills in missing params default parameters

  Returns:
  list: Soutions list

  """

  solutions = []
  numHardcoded = len(hardcodedParameters)

  print1("# Enumerating Solutions")
  solutionSet = set()

  for hardcodedIdx in Utils.tqdm(range(0, numHardcoded), "Enumerating Solutions"):
    solutions.append([])
    hardcodedParamDict = hardcodedParameters[hardcodedIdx]
    for benchmarkPermutation in benchmarkPermutations:
      solution = {"ProblemType": deepcopy(problemType.state)}
      solution.update(benchmarkPermutation)
      solution.update(hardcodedParamDict)

      # append default parameters where necessary
      if initialSolutionParameters:
        for initialSolutionParameterName in initialSolutionParameters:
          if initialSolutionParameterName not in solution:
            solution[initialSolutionParameterName] = \
              initialSolutionParameters[initialSolutionParameterName]

      # TODO check if solution matches problem size for exact tile kernels
      solutionObject = Solution(solution)
      if solutionObject["Valid"]:
        if solutionObject not in solutionSet:
          solutionSet.add(solutionObject)
          solutions[hardcodedIdx].append(solutionObject)
      else:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("rejecting solution %s" % str(solutionObject))

  return solutions

def generateCustomKernelSolution(kernelName, directory=globalParameters["CustomKernelDirectory"]):
    kernelConfig = getCustomKernelConfig(kernelName, directory)
    checkForValidParameters({p: [kernelConfig[p]] for p in kernelConfig if p != "ProblemType"}, set(validParameters.keys()))
    # test if problem type matches with configuration file
    kernelConfig["KernelLanguage"] = "Assembly"   # replacement kernels are always assembly kernels?
    kernelConfig["CustomKernelName"] = kernelName

    return Solution(kernelConfig)

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

  enableTileSelection = benchmarkProcess.problemType["TileAwareSelection"]
  problemTypeName = str(benchmarkProcess.problemType)
  problemSizeGroupName = "%s_%02u" % (problemTypeName, problemSizeGroupIdx)
  pushWorkingPath(problemSizeGroupName)
  ensurePath(os.path.join(globalParameters["WorkingPath"],"Data"))

  totalBenchmarkSteps = len(benchmarkProcess)
  resultsFileBaseFinal = None

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

    pushWorkingPath(shortName)

    ############################################################################
    # Copy Files to Benchmark Source Directory
    ############################################################################
    stepBaseDir = globalParameters["WorkingPath"]
    sourceDir = \
      os.path.join(stepBaseDir, "source" )
    ensurePath(sourceDir)

    filesToCopy = []
    pushWorkingPath("source")
    filesToCopy = [
        "TensileTypes.h",
        "tensile_bfloat16.h",
        "KernelHeader.h",
        ]

    for f in filesToCopy:
      shutil.copy(
          os.path.join(globalParameters["SourcePath"], f),
          globalParameters["WorkingPath"] )
    if globalParameters["RuntimeLanguage"] == "OCL":
      shutil.copy(
          os.path.join(globalParameters["SourcePath"], "FindOpenCL.cmake"),
          globalParameters["WorkingPath"] )
    else:
      shutil.copy(
          os.path.join(globalParameters["SourcePath"], "FindHIP.cmake"),
          globalParameters["WorkingPath"] )

    ############################################################################
    # Enumerate Benchmark Permutations
    ############################################################################
    solutions = []

    benchmarkPermutations = constructForkPermutations(benchmarkStep.benchmarkParameters)
    maxPossibleSolutions = len(benchmarkPermutations) * numHardcoded

    ############################################################################
    # Enumerate Solutions = Hardcoded * Benchmark
    ############################################################################
    solutions = generateForkedSolutions(benchmarkProcess.problemType, \
        benchmarkStep.hardcodedParameters, benchmarkPermutations, \
        benchmarkStep.initialSolutionParameters)

    # remove hardcoded that don't have any valid benchmarks
    removeHardcoded = list([x for i, x in enumerate(benchmarkStep.hardcodedParameters) \
        if len(solutions[i]) == 0])
    validHardcoded =  list([x for i, x in enumerate(benchmarkStep.hardcodedParameters) \
        if len(solutions[i]) > 0])

    removesExist = len(removeHardcoded) > 0
    benchmarkStep.hardcodedParameters = validHardcoded

    # add custom kernels to list of solutions
    customKernelList = problemSizeGroupConfig.get("CustomKernels", [])
    customKernelWildcard = False
    if customKernelList == ["*"]:
      customKernelList = \
          [fname[:-2] for fname in os.listdir(globalParameters["CustomKernelDirectory"]) \
          if fname.endswith(".s")]
      customKernelWildcard = True

    for kernelName in customKernelList:
      print1("# Processing custom kernel {}".format(kernelName))
      customSolution = generateCustomKernelSolution(kernelName)
      if customSolution["ProblemType"] != benchmarkProcess.problemType:
        # Raise error if this kernel was specifically requested and problem type doesn't match
        if not customKernelWildcard:
          missingParams = [p for p in benchmarkProcess.problemType if p not in customSolution["ProblemType"]]
          extraParams   = [p for p in customSolution["ProblemType"] if p not in benchmarkProcess.problemType]
          msg  = "The problem type in the config file does not match" \
              "that of the custom kernel, {0}.".format(kernelName)
          msg += "\nMissing config parameters:\n" + str(missingParams)
          msg += "\nExtra custom kernel parameters:\n" + str(extraParams)
          raise RuntimeError(msg)
        else:
          print1("# Rejected {}: Problem Type doesn't match".format(kernelName))
      else:
        print1("# Added {} to solutions".format(kernelName))
        maxPossibleSolutions += 1
        if customSolution["Valid"]:
          solutions.append([customSolution])
          benchmarkStep.hardcodedParameters.append(customSolution._state)
        elif globalParameters["PrintSolutionRejectionReason"]:
          print1("rejecting solution %s" % str(customSolution))

    numHardcoded = len(benchmarkStep.hardcodedParameters )
    if removesExist:
      solutions = list([s for s in solutions if len(s) > 0])

    print1("# Actual Solutions: %u / %u after SolutionStructs\n" % ( len(solutions), \
        maxPossibleSolutions ))

    # create linear list
    solutionList = list(itertools.chain.from_iterable(solutions))

    if len(solutionList) == 0:
        msg = "Your parameters resulted in 0 valid solutions."
        if globalParameters["PrintSolutionRejectionReason"]:
            msg += "\nExamine reject and backtrace messages above to see why" \
                "and where solutions were rejected."
        else:
            msg += "\nYou should re-run with \"PrintSolutionRejectionReason: True\"" \
                "to see why each parameter combination was rejected."
        printExit(msg)
    if globalParameters["PrintLevel"] >= 1:
      for i,solutionsForHardcoded in enumerate(solutions):
        for j, solution in enumerate(solutionsForHardcoded):
          print2("#    (%u:%u) %s" % (i, j, \
              Solution.getNameFull(solution) ))
      print2(HR)

    # write benchmarkFiles
    writeBenchmarkFiles(stepBaseDir, solutionList, benchmarkStep.problemSizes, \
        shortName, filesToCopy, benchmarkProcess.solutionSummationSizes)

    removeSolutions = []
    for i in range(0, len(solutions)):
      solutionsForHardcoded = solutions[i]
      removeSolutions.append([])
      for j in range(0, len(solutionsForHardcoded)):
        solution = solutionsForHardcoded[j]
        if solutionList.count(solution) == 0:
          removeSolutions[i].append(solution)

    for i in range(0, len(solutions)):
      solutionsForHardcoded = solutions[i]
      for j in range(0, len(removeSolutions[i])):
          solutionsForHardcoded.remove(removeSolutions[i][j])

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
      numHardcoded = len(benchmarkStep.hardcodedParameters )
      # remove from solution 2D list also
      prevCount = len(solutions)
      solutions = list([s for s in solutions if len(s) > 0])
      print1("# Actual Solutions: %u / %u after KernelWriter\n" \
            % (len(solutions), prevCount ))

    popWorkingPath() # source

    ############################################################################
    # Run Benchmark Script
    ############################################################################
    resultsFileBase = os.path.normpath(os.path.join( \
        globalParameters["WorkingPath"], "../Data", shortName))
    if benchmarkStep.isFinal():
      resultsFileBaseFinal = resultsFileBase
    resultsFileName = resultsFileBase + ".csv"
    newResultsFileName = None # Add another results file to generate diff
    solutionsFileName = resultsFileBase + ".yaml"
    if not os.path.exists(resultsFileName) or \
        globalParameters["ForceRedoBenchmarkProblems"]:


      libraryLogicPath = None
      forBenchmark = True
      returncode = runClient(libraryLogicPath, forBenchmark, enableTileSelection)

      if returncode:
        benchmarkTestFails += 1
        printWarning("BenchmarkProblems: Benchmark Process exited with code %u" % returncode)
    else:
      print1("# Already benchmarked; skipping.")

    ############################################################################
    # Write Solutions YAML
    ############################################################################
    LibraryIO.writeSolutions(solutionsFileName, benchmarkStep.problemSizes, \
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

def compareResults(old, new, name):
    import math
    if name == " WinnerIdx":
      return 0

    try:
        old = float(old)
    except (ValueError, TypeError):
        old = -1

    try:
        new = float(new)
    except (ValueError, TypeError):
        new = -1

    def isbad(x):
        return x <= 0 or math.isnan(x) or math.isinf(x)

    if isbad(old) and isbad(new):
        return 0
    if isbad(old):
        return 1
    if isbad(new):
        raise ValueError("Old is good ({}) and new is bad ({}). Name: {}".format(old, new, name))

    return abs((old-new)/old)

################################################################################
# Read GFlop/s from file
################################################################################
def getResults(resultsFileName, solutions, enableTileSelection, newResultsFileName=None):

  print1("# Get Results from CSV")
  try:
    resultsFile = open(resultsFileName, "r")
  except IOError:
    printExit("Can't open \"%s\" to get results" % resultsFileName )

  newCSV = itertools.repeat(None)
  if newResultsFileName is not None:
    newFile = open(newResultsFileName, 'r')
    newCSV = csv.reader(newFile)

    diffFile = open(newResultsFileName+'-diff.csv', 'w')
    diffCSV = csv.writer(diffFile)

  # setup data structures
  results = []
  numSolutions = 0
  for solutionsForHardcoded in solutions:
    results.append([])
    for solution in solutionsForHardcoded:
      numColForProblemSize = solution["ProblemType"]["TotalIndices"]
      results[-1].append([])
      numSolutions += 1

  # read results in gflops
  csvFile = csv.reader(resultsFile)

  if globalParameters["CSVExportWinner"]:
    # in both old/new clients, csv files always output "GFlops" ,...., "LDD" "LDC" "LDA" "LDB" "TotalFlops" "WinnerGFlops" "WinnerTimeUS" "WinnerIdx" "WinnerName" columns
    startIdx = numColForProblemSize + 10
  else:
    # in both old/new clients, csv files always output "GFlops" ,...., "LDD" "LDC" "LDA" "LDB"columns
    # old client, non-GEMM csv files don't contain "LDD" "LDC" "LDA" "LDB", so we output an "N/A" text (in csv only) for alignment purpose (-diff.csv)
    startIdx = numColForProblemSize + 5

  rowLength = startIdx + numSolutions

  rowIdx = 0
  for row,newRow in zip(csvFile, newCSV):
    rowIdx+=1
    if rowIdx == 1:
      if newRow is not None:
        diffCSV.writerow(row)
        diffCSV.writerow(newRow)
        headerRow = row
      continue
    else:
      if len(row) < rowLength:
        printWarning("CSV File %s row %u doesn't have %u elements; ignoring remainer of file." \
            % (resultsFileName, rowIdx, rowLength) )
        break
      if newRow is not None:
        diffCSV.writerow([compareResults(old,new,name) for old,new,name in itertools.zip_longest(row, newRow, headerRow)])

      idx = startIdx
      for i,solutionsForHardcoded in enumerate(solutions):
        for j,solution in enumerate(solutionsForHardcoded):
          gflops = float(row[idx])

          results[i][j].append(gflops)
          idx += 1
  if rowIdx < 2 and not enableTileSelection:
    printExit("CSV File %s only has %u row(s); prior benchmark must not have run long enough to produce data." \
        % (resultsFileName, rowIdx) )

  resultsFile.close()
  if newResultsFileName is not None:
    newFile.close()
    diffFile.close()
  return results


################################################################################
# Write Benchmark Files
################################################################################
def writeBenchmarkFiles(stepBaseDir, solutions, problemSizes, stepName, filesToCopy, solutionSummationSizes):
  if not globalParameters["MergeFiles"] or globalParameters["NumMergedFiles"] > 1:
    ensurePath(os.path.join(globalParameters["WorkingPath"], "Solutions"))
    ensurePath(os.path.join(globalParameters["WorkingPath"], "Kernels"))

  ##############################################################################
  # Min Naming
  ##############################################################################

  kernels = []
  kernelHelperOjbs = []

  kernelNames = set()
  kernelHelperNames = set()

  for solution in Utils.tqdm(solutions, "Finding unique solutions"):
    solutionKernels = solution.getKernels()
    for kernel in solutionKernels:
      kName = Solution.getNameFull(kernel)
      if kName not in kernelNames:
        kernels.append(kernel)
        kernelNames.add(kName)

    solutionHelperKernels = solution.getHelperKernelObjects()
    for ko in solutionHelperKernels:
      kname = ko.getKernelName()
      if kname not in kernelHelperNames:
        kernelHelperOjbs.append(ko)
        kernelHelperNames.add(kname)


  solutionSerialNaming = Solution.getSerialNaming(solutions)
  kernelSerialNaming   = Solution.getSerialNaming(kernels)
  solutionMinNaming    = Solution.getMinNaming(solutions)
  kernelMinNaming      = Solution.getMinNaming(kernels)
  solutionWriter       = SolutionWriter(solutionMinNaming, solutionSerialNaming, kernelMinNaming, kernelSerialNaming)
  kernelWriterSource   = KernelWriterSource(kernelMinNaming, kernelSerialNaming)
  kernelWriterAssembly = KernelWriterAssembly(kernelMinNaming, kernelSerialNaming)

  # write solution, kernels and CMake
  problemType = solutions[0]["ProblemType"]
  codeObjectFiles = writeSolutionsAndKernels( \
      globalParameters["WorkingPath"], globalParameters["CxxCompiler"], [problemType], solutions, kernels, kernelHelperOjbs, \
      solutionWriter, kernelWriterSource, kernelWriterAssembly, errorTolerant=True )

  newLibraryDir = ensurePath(os.path.join(globalParameters["WorkingPath"], 'library'))
  newLibraryFile = os.path.join(newLibraryDir, "TensileLibrary")
  newLibrary = SolutionLibrary.MasterSolutionLibrary.BenchmarkingLibrary(solutions)
  newLibrary.applyNaming(kernelMinNaming)
  LibraryIO.write(newLibraryFile, Utils.state(newLibrary), globalParameters["LibraryFormat"])

  codeObjectFiles = [os.path.relpath(f, globalParameters["WorkingPath"]) for f in codeObjectFiles]

  writeClientConfig(True, solutions, problemSizes, stepName, stepBaseDir, newLibrary, codeObjectFiles, False)

  if "TileAwareSelection" in problemType and problemType["TileAwareSelection"]:
    maxMacroTile0 = 0
    maxMacroTile1 = 0
    for solution in solutions:
      macroTile0 = solution["MacroTile0"]
      macroTile1 = solution["MacroTile1"]
      if macroTile0 > maxMacroTile0:
        maxMacroTile0 = macroTile0
      if macroTile1 > maxMacroTile1:
        maxMacroTile1 = macroTile1
    idealM = 36 * maxMacroTile0
    idealN = 36 * maxMacroTile1
    idealSizes = []
    if problemType["Batched"]:
        for idealK in solutionSummationSizes:
          idealSize = {"Exact": [idealM, idealN, 1, idealK]}
          idealSizes.append(idealSize)
    else:
        for idealK in solutionSummationSizes:
          idealSize = {"Exact": [idealM, idealN, idealK]}
          idealSizes.append(idealSize)
    idealProblemSizes = ProblemSizes(problemType, idealSizes)
    writeClientConfig(True, solutions, idealProblemSizes, stepName, stepBaseDir, newLibrary, codeObjectFiles, True)

  if len(solutions) == 0:
    printExit("write solutions and kernels results 0 valid soultion.")

  ##############################################################################
  # Write CMake
  ##############################################################################
  outputPath = globalParameters["WorkingPath"]

  (solutionFiles,
   sourceKernelFiles,
   asmKernelFiles,
   sourceLibFiles,
   asmLibFiles) = buildObjectFileNames(solutionWriter, kernelWriterSource, \
    kernelWriterAssembly, solutions, kernels, kernelHelperOjbs)

  writeCMake(outputPath, solutionFiles, sourceKernelFiles, filesToCopy)

  for fileName in filesToCopy:
    shutil.copy( os.path.join(globalParameters["SourcePath"], fileName), \
      outputPath )


################################################################################
# Main
################################################################################
def main(config):
  """Entry point for the "BenchmarkProblems" section of a Tensile config yaml"""
  ClientExecutable.getClientExecutable()

  dataPath = os.path.join(globalParameters["WorkingPath"], globalParameters["BenchmarkDataPath"])
  pushWorkingPath(globalParameters["BenchmarkProblemsPath"])
  ensurePath(dataPath)

  totalTestFails = 0
  for benchmarkProblemTypeConfig in config:
    problemTypeConfig = benchmarkProblemTypeConfig[0]
    if len(benchmarkProblemTypeConfig) < 2:
      problemSizeGroupConfigs = [{}]
    else:
      problemSizeGroupConfigs = benchmarkProblemTypeConfig[1:]

    for problemSizeGroupIdx, problemSizeGroupConfig in enumerate(problemSizeGroupConfigs):
      print2("ProblemTypeConfig: {}".format(problemTypeConfig))
      problemTypeObj = ProblemType(problemTypeConfig)
      globalParameters["EnableHalf"] = problemTypeObj["DataType"].isHalf()

      # using a suffix to check the csv version (for later addFromCSV())
      csvSuffix = "_CSVWinner" if globalParameters["CSVExportWinner"] else ""
      # results files will be named
      newResultsFileName = os.path.join(dataPath, "{}_{:02d}{}.csv" \
          .format(str(problemTypeObj), problemSizeGroupIdx, csvSuffix) )
      newSolutionsFileName = os.path.join(dataPath, "{}_{:02d}{}.yaml" \
          .format(str(problemTypeObj), problemSizeGroupIdx, csvSuffix) )
      newGranularityFileName = os.path.join(dataPath, "{}_{:02d}{}.gsp" \
          .format(str(problemTypeObj), problemSizeGroupIdx, csvSuffix) )

      # skip if possible
      if globalParameters["ForceRedoBenchmarkProblems"] \
          or not os.path.exists(newResultsFileName):

        # benchmark problem size group
        (resultsFileBaseFinal, benchmarkErrors) = \
            benchmarkProblemType(problemTypeConfig, problemSizeGroupConfig, problemSizeGroupIdx)
        totalTestFails += benchmarkErrors

        print("clientExit={} {} for {}" \
            .format(totalTestFails, "(ERROR)" if totalTestFails else "(PASS)", \
            globalParameters["ConfigPath"]) )

        # copy data
        resultsFileBase = resultsFileBaseFinal
        resultsFileName = resultsFileBase + ".csv"
        solutionsFileName = resultsFileBase + ".yaml"
        granularityFileName = resultsFileBase + "_Granularity.csv"
        shutil.copy( resultsFileName, newResultsFileName )
        shutil.copy( solutionsFileName, newSolutionsFileName )
        if os.path.isfile(granularityFileName):
          shutil.copy( granularityFileName, newGranularityFileName )
      else:
        print1("# {}_{:02d} already benchmarked; skipping." \
            .format(str(problemTypeObj), problemSizeGroupIdx) )

  popWorkingPath()

  if globalParameters["ExitOnFails"] and totalTestFails:
    sys.exit(1)

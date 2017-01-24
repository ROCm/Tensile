import sys
import os
from copy import deepcopy
from shutil import copy as shutil_copy
from shutil import rmtree
import csv

from BenchmarkProcess import *
from Common import *
from Structs import *
from SolutionWriter import *
from KernelWriter import *
from subprocess import Popen


################################################################################
# Benchmark Problem Type
################################################################################
def benchmarkProblemType( config ):

  hr = "#######################################################################"

  # convert config to full benchmark process (resolves defaults)
  print ""
  print hr
  print "# Converting Config to BenchmarkProcess Object"
  print hr
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
  print hr
  print "# Done Creating BenchmarkProcess Object"
  print hr

  ##############################################################################
  # For Each Benchmark Step
  ##############################################################################
  for benchmarkStepIdx in range(0, totalBenchmarkSteps):

    # Print Step Name
    benchmarkStep = benchmarkProcess[benchmarkStepIdx]
    resultingHardcodedParameterList = \
        winners.update( benchmarkStep.hardcodedParameters )
    benchmarkStep.hardcodedParameters = resultingHardcodedParameterList
    stepName = str(benchmarkStep)
    print "\n\n"
    print hr
    print "# %s\n# %s" % (problemTypeName, stepName)
    print "#  -NumProblems: %u" % benchmarkStep.problemSizes.totalProblemSizes
    print "#  -BenchmarkParameters:"
    for paramName in benchmarkStep.benchmarkParameters:
      paramValues = benchmarkStep.benchmarkParameters[paramName]
      printStr = "#     %s = { %s" % (paramName, paramValues[0])
      for paramValueIdx in range(1, len(paramValues)):
        printStr += ", %s" % str(paramValues[paramValueIdx])
      printStr += " }"
      print printStr
    print "#  -HardcodedParameters:"
    paramDictIdx = 0
    hardcodedParametersMinNaming = \
        Solution.getMinNaming(benchmarkStep.hardcodedParameters)
    for paramDict in benchmarkStep.hardcodedParameters:
      print "#   (%u) %s" % (paramDictIdx, \
          Solution.getNameMin(paramDict, hardcodedParametersMinNaming))
      paramDictIdx += 1
      #for paramName in paramDict:
      #  paramValue = paramDict[paramName]
      #  printStr += "%s: %s, " % (paramName, str(paramValue))
      #print printStr
    print hr
    print ""
    pushWorkingPath(stepName)

    ############################################################################
    # Copy Files to Benchmark Source Directory
    ############################################################################
    pushWorkingPath("source")
    filesToCopy = [
        "TensileBenchmark_Main.cpp",
        "TensileBenchmark_Main.h",
        "MathTemplates.cpp",
        "MathTemplates.h",
        "Tensile.cpp",
        "Tensile.h",
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
    shutil_copy(
        os.path.join(globalParameters["SourcePath"], "TensileBenchmark_CMakeLists.txt"),
        os.path.join(globalParameters["WorkingPath"], "CMakeLists.txt" ) )
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
    #print "prevParameters = %s" % str(benchmarkStep.prevParameters)
    totalBenchmarkPermutations = 1
    for benchmarkParamName in benchmarkStep.benchmarkParameters:
      totalBenchmarkPermutations *= len(benchmarkStep.benchmarkParameters[benchmarkParamName])
    print "MaxPossibleSolutions: %u = %u (hardcoded) * %u (benchmark)" % \
        (totalBenchmarkPermutations*len(benchmarkStep.hardcodedParameters), \
        len(benchmarkStep.hardcodedParameters), totalBenchmarkPermutations)

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
    for hardcodedParamDict in benchmarkStep.hardcodedParameters:
      for benchmarkPermutation in benchmarkPermutations:
        solution = {"ProblemType": deepcopy(benchmarkProcess.problemType.state)}
        solution.update(benchmarkPermutation)
        solution.update(hardcodedParamDict)
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
          solutions.append(solutionObject)
        #else:
          #printStatus("rejecting solution %s" % str(solutionObject))
    print "NumActualSolutions: %u / %u" % ( len(solutions), \
        totalBenchmarkPermutations*len(benchmarkStep.hardcodedParameters) )
    # write benchmarkFiles
    writeBenchmarkFiles(solutions, benchmarkStep.problemSizes, \
        stepName, filesToCopy)
    popWorkingPath() # source

    ############################################################################
    # Run Benchmark Script
    ############################################################################
    resultsFileName = os.path.join(globalParameters["WorkingPath"], \
        "../Data", "%s.csv" % stepName)
    if not os.path.exists(resultsFileName) or globalParameters["ForceRedo"]:
      # if redo=true, clobber the build directory
      if globalParameters["ForceRedo"]:
        rmtree(os.path.join(globalParameters["WorkingPath"], "build"), ignore_errors=True)
      pushWorkingPath("build")

      # create run.bat or run.sh which builds and runs
      runScriptName = os.path.join(globalParameters["WorkingPath"], \
        "run.%s" % "bat" if os.name == "nt" else "sh")
      runScriptFile = open(runScriptName, "w")
      runScriptFile.write("@echo. & echo %s & echo # %s & echo # %s: Configuring CMake & echo %s\n" \
          % (hr, problemTypeName, stepName, hr))
      runScriptFile.write("cmake -DCMAKE_GENERATOR_PLATFORM=x64 ../source\n")
      runScriptFile.write("@echo. & echo %s & echo # %s & echo # %s: Building Benchmark & echo %s\n" \
          % (hr, problemTypeName, stepName, hr))
      runScriptFile.write("cmake --build . --config %s\n" \
          % globalParameters["CMakeBuildType"] )
      runScriptFile.write("@echo. & echo %s & echo # %s & echo # %s: Running Benchmark & echo %s\n" \
          % (hr, problemTypeName, stepName, hr))
      runScriptFile.write(os.path.join(globalParameters["CMakeBuildType"],"TensileBenchmark_%s%s") \
          % (stepName, ".exe" if os.name == "nt" else "") )
      runScriptFile.close()
      # wait for python to finish printing
      process = Popen(runScriptName, cwd=globalParameters["WorkingPath"])
      status = process.communicate()
      popWorkingPath() # build


    ############################################################################
    # Winners -> Determined Parameters
    ############################################################################
    results = getResults(resultsFileName, len(solutions), \
        len(benchmarkStep.hardcodedParameters), \
        solutions[0])
    winners.addResults(benchmarkStep.hardcodedParameters, \
        benchmarkPermutations, results)
    #for hardcodedParameterIdx in range(0, len(winnerIndices)):
    #  hardcodedParameters = \
    #      benchmarkStep.hardcodedParameters[hardcodedParameterIdx]
    #  (winnerIdx, score) = winnerIndices[hardcodedParameterIdx]
    #  winningParameters = benchmarkPermutations[winnerIdx]
    #  winners[hardcodedParameters] = (winningParameters, score) # does update

    #print "Winners Updated Winners\n%s" % winners

    popWorkingPath() # stepName
    print "%s%s\n# %s\n# %s: End\n%s%s\n" \
        % (hr, hr, problemTypeName, stepName, hr, hr)

  popWorkingPath()




################################################################################
# Read GFlop/s from file
################################################################################
def getResults(resultsFileName, numSolutions, numHardcodedParameters, \
    solution):
  try:
    resultsFile = open(resultsFileName, "r")
  except IOError:
    printExit("Can't open \"%s\" to get results" % resultsFileName )

  numSolutionsPerHardcodedParameter = numSolutions / numHardcodedParameters

  # setup data structures
  results = []
  #numWins = []
  for i in range(0, numHardcodedParameters):
    results.append([])
    #numWins.append([])
    for j in range(0, numSolutionsPerHardcodedParameter):
      results[i].append([])
      #numWins[i].append(0)

  # read results in gflops
  csvFile = csv.reader(resultsFile)
  problemSizeIdx = solution["ProblemType"]["TotalIndices"] + 1
  startIdx = problemSizeIdx + 1
  rowIdx = 0
  for row in csvFile:
    if rowIdx == 0:
      rowIdx+=1
      continue
    else:
      totalFlops = float(row[problemSizeIdx])
      for i in range(0, numHardcodedParameters):
        for j in range(0, numSolutionsPerHardcodedParameter):
          time_ms = float(row[startIdx+j \
              + i*numSolutionsPerHardcodedParameter])
          flops = totalFlops / (time_ms / 1000)
          gflops = flops / (1000*1000*1000)
          results[i][j].append(gflops)
  return results

  """
  # count wins for each problem size
  for r in range(0, rowIdx):
    for i in range(0, numHardcodedParameters):
      winnerIdx = -1
      winnerTime = 1e100
      for j in range(0, numSolutionsPerHardcodedParameter):
        solutionTime = times[i][j][r]
        if solutionTime < winnerTime:
          winnerIdx = j
          winnerTime = solutionTime
      numWins[i][winnerIdx] += 1

  # determine winnerIndices
  winnerIndices = []
  for i in range(0, numHardcodedParameters):
    winnerIdx = -1
    winnerNumWins = 0
    for j in range(0, numSolutionsPerHardcodedParameter):
      if numWins[i][j] > winnerNumWins:
        winnerIdx = j
        winnerNumWins = numWins[i][j]
    winnerIndices.append(winnerIdx)
  return winnerIndices
  """


################################################################################
# Write Benchmark Files
################################################################################
def writeBenchmarkFiles(solutions, problemSizes, stepName, filesToCopy):
  printStatus("Beginning")
  if not globalParameters["MergeFiles"]:
    ensurePath(os.path.join(globalParameters["WorkingPath"], "Solutions"))
    ensurePath(os.path.join(globalParameters["WorkingPath"], "Kernels"))

  solutionFileNames = []
  kernelNames = []
  kernels = []

  ##############################################################################
  # Min Naming
  ##############################################################################
  for solution in solutions:
    solutionKernels = solution.getKernels()
    for kernel in solutionKernels:
      if kernel not in kernels:
        kernels.append(kernel)

  if globalParameters["ShortFileNames"] and not globalParameters["MergeFiles"] :
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



  ##############################################################################
  # Write Solutions
  ##############################################################################
  if globalParameters["MergeFiles"]:
    solutionSourceFile = open(os.path.join(globalParameters["WorkingPath"], \
        "GeneratedSolutions.cpp"), "w")
    solutionHeaderFile = open(os.path.join(globalParameters["WorkingPath"], \
        "GeneratedSolutions.h"), "w")
    solutionSourceFile.write("#include \"GeneratedSolutions.h\"\n")
    solutionHeaderFile.write("#include \"Tensile.h\"\n")
    solutionHeaderFile.write("#include \"GeneratedKernels.h\"\n")
    solutionHeaderFile.write("#include \"SolutionHelper.h\"\n")
    solutionHeaderFile.write("#include \"Tools.h\"\n")
  for solution in solutions:
    # get solution name
    if not globalParameters["MergeFiles"]:
      if globalParameters["ShortFileNames"]:
        solutionFileName = \
            Solution.getNameSerial(solution, solutionSerialNaming)
      else:
        solutionFileName = Solution.getNameMin(solution, solutionMinNaming)
      solutionFileNames.append(solutionFileName)
    #printStatus("Writing files for solution %s" % solutionFileName )

    # write solution.cpp
    if not globalParameters["MergeFiles"]:
      solutionSourceFile = open(os.path.join(globalParameters["WorkingPath"], \
          "Solutions", solutionFileName+".cpp"), "w")
    solutionSourceFile.write( \
        solutionWriter.getSourceFileString(solution))
    if not globalParameters["MergeFiles"]:
      solutionSourceFile.close()

    # write solution.h
    if not globalParameters["MergeFiles"]:
      solutionHeaderFile = open(os.path.join(globalParameters["WorkingPath"], \
          "Solutions", solutionFileName+".h"), "w")
    solutionHeaderFile.write( \
        solutionWriter.getHeaderFileString(solution))
    if not globalParameters["MergeFiles"]:
      solutionHeaderFile.close()
  # close merged
  if not globalParameters["MergeFiles"]:
    solutionHeaderFile.close()

  ##############################################################################
  # Write Kernels
  ##############################################################################
  if globalParameters["MergeFiles"]:
    kernelSourceFile = open(os.path.join(globalParameters["WorkingPath"], \
        "GeneratedKernels.cpp"), "w")
    kernelHeaderFile = open(os.path.join(globalParameters["WorkingPath"], \
        "GeneratedKernels.h"), "w")
    kernelSourceFile.write("#include \"GeneratedKernels.h\"\n")
    kernelHeaderFile.write("#pragma once\n")
    if globalParameters["Backend"] != "OCL":
      kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
  for kernel in kernels:
    # get kernel name
    if not globalParameters["MergeFiles"]:
      if globalParameters["ShortFileNames"]:
        kernelName = Solution.getNameSerial(kernel, kernelSerialNaming)
      else:
        kernelName = Solution.getNameMin(kernel, kernelMinNaming)
      kernelNames.append(kernelName)

    # write kernel.cpp
    if not globalParameters["MergeFiles"]:
      kernelSourceFile = open(os.path.join(globalParameters["WorkingPath"], \
          "Kernels", kernelName+".cpp"), "w")
    kernelSourceFile.write( kernelWriter.getSourceFileString(kernel))
    if not globalParameters["MergeFiles"]:
      kernelSourceFile.close()

    # write kernel.h
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile = open(os.path.join(globalParameters["WorkingPath"], \
          "Kernels", kernelName+".h"), "w")
    kernelHeaderFile.write( kernelWriter.getHeaderFileString(kernel))
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile.close()
  # close merged
  if globalParameters["MergeFiles"]:
    kernelHeaderFile.close()

  ##############################################################################
  # Write CMake
  ##############################################################################
  generatedFile = open(os.path.join(globalParameters["WorkingPath"], \
      "Generated.cmake"), "w")
  generatedFile.write(globalParameters["CMakeHeader"])
  generatedFile.write("set( TensileBenchmark_Solutions\n")
  # write solution names
  if globalParameters["MergeFiles"]:
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/GeneratedSolutions.h\n")
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/GeneratedSolutions.cpp\n")
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
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/GeneratedKernels.h\n")
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/GeneratedKernels.cpp\n")
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
  generatedFile.write("set( TensileBenchmark_NAME TensileBenchmark_%s)\n" \
      % (stepName) )
  generatedFile.write("set( TensileBenchmark_BACKEND \"%s\")\n" \
      % (globalParameters["Backend"]) )

  # build parameters
  generatedFile.write("set( CMAKE_BUILD_TYPE \"%s\")\n" \
      % (globalParameters["CMakeBuildType"]) )

  # close generated cmake
  generatedFile.close()

  ##############################################################################
  # Write Generated Benchmark Parameters
  ##############################################################################
  benchmarkParametersFile = open(os.path.join(globalParameters["WorkingPath"], \
      "GeneratedBenchmarkParameters.h"), "w")
  benchmarkParametersFile.write(globalParameters["CHeader"])

  h = ""
  if globalParameters["MergeFiles"]:
    h += "#include \"GeneratedSolutions.h\"\n"
  else:
    for solutionFileName in solutionFileNames:
      h += "#include \"" + solutionFileName + ".h\"\n"
  h += "\n"

  ##############################################################################
  # Problem Type
  ##############################################################################
  h += "/* problem type */\n"
  typeName = solution["ProblemType"]["DataType"].toCpp()
  h += "typedef %s DataType;\n" % (typeName)
  h += "DataType alpha = tensileGetOne<DataType>();\n"
  h += "DataType beta = tensileGet%s<DataType>();\n" % ("One" if solution["ProblemType"]["UseBeta"] else "Zero")
  h += "static const unsigned int bytesPerElement = %u;\n" \
      % (solutions[0]["ProblemType"]["DataType"].numBytes())
  h += "const int numFlopsPerMac = %u;\n" \
      % (2 if solution["ProblemType"]["DataType"].isReal() else 8)
  h += "const unsigned int numIndicesC = %u;\n" % solution["ProblemType"]["NumIndicesC"]
  h += "const unsigned int numIndicesAB = %u;\n" % len(solution["ProblemType"]["IndexAssignmentsA"])
  h += "const unsigned int indexAssignmentsA[numIndicesAB] = {"
  h += "  %u" % solution["ProblemType"]["IndexAssignmentsA"][0]
  for i in range(1, len(solution["ProblemType"]["IndexAssignmentsA"])):
    h += ", %u" % solution["ProblemType"]["IndexAssignmentsA"][i]
  h += "};\n"
  h += "const unsigned int indexAssignmentsB[numIndicesAB] = {"
  h += "  %u" % solution["ProblemType"]["IndexAssignmentsB"][0]
  for i in range(1, len(solution["ProblemType"]["IndexAssignmentsB"])):
    h += ", %u" % solution["ProblemType"]["IndexAssignmentsB"][i]
  h += "};\n"
  h += "const bool complexConjugateA = %s;\n" % ("true" if solution["ProblemType"]["ComplexConjugateA"] else "false" )
  h += "const bool complexConjugateB = %s;\n" % ("true" if solution["ProblemType"]["ComplexConjugateB"] else "false" )
  h += "\n"

  ##############################################################################
  # Problem Sizes
  ##############################################################################
  h += "/* problem sizes */\n"
  h += "static const unsigned int totalIndices = %u;\n" \
      % (problemSizes.totalIndices)
  h += "static const bool indexIsSized[totalIndices] = {"
  for i in range(0, problemSizes.totalIndices):
    h += "  %s" % ("true" if problemSizes.indexIsSized[i] else "false")
    if i < problemSizes.totalIndices-1:
      h += ","
  h += " };\n"

  h += "static const unsigned int numIndicesSized = %u;\n" \
      % len(problemSizes.indicesSized)
  h += "static const unsigned int indicesSized[numIndicesSized][4] = {\n"
  h += "// { min, stride, stride_incr, max }\n"
  for i in range(0, len(problemSizes.indicesSized)):
    r = problemSizes.indicesSized[i]
    h += "  { %u, %u, %u, %u }" % (r[0], r[1], r[2], r[3])
    if i < len(problemSizes.indicesSized)-1:
      h += ","
    h += "\n"
  h += "  };\n"

  numIndicesMapped = len(problemSizes.indicesMapped)
  h += "static const unsigned int numIndicesMapped = %u;\n" % numIndicesMapped
  if numIndicesMapped > 0:
    h += "static const unsigned int indicesMapped[numIndicesMapped] = {"
    for i in range(0, numIndicesMapped):
      h += " %u" % problemSizes.indicesMapped[i]
      if i < numIndicesMapped-1:
        h += ","
    h += " };\n"
  else:
    h += "static const unsigned int indicesMapped[1] = { 0 }; // dummy\n"

  # max problem sizes
  sizeC = 1
  sizeA = 1
  sizeB = 1
  for idx in range(0, solution["ProblemType"]["NumIndicesC"]):
    sizeC *= problemSizes.indexMax[idx]
  for idx in solution["ProblemType"]["IndexAssignmentsA"]:
    sizeA *= problemSizes.indexMax[idx]
  for idx in solution["ProblemType"]["IndexAssignmentsB"]:
    sizeB *= problemSizes.indexMax[idx]
  h += "size_t maxSizeC = %u;\n" % (sizeC)
  h += "size_t maxSizeA = %u;\n" % (sizeA)
  h += "size_t maxSizeB = %u;\n" % (sizeB)
  h += "\n"

  ##############################################################################
  # Current Problem Size
  ##############################################################################
  h += "/* current problem size */\n"
  h += "const unsigned int numProblems = %u;\n" % problemSizes.totalProblemSizes
  h += "unsigned int fullSizes[totalIndices];\n"
  h += "unsigned int currentSizedIndexSizes[numIndicesSized];\n"
  h += "unsigned int currentSizedIndexIncrements[numIndicesSized];\n"
  h += "\n"

  ##############################################################################
  # Solutions
  ##############################################################################
  h += "/* solutions */\n"
  h += "const unsigned int numSolutions = %u;\n" % len(solutions)
  h += "float solutionTimes[numProblems][numSolutions]; // milliseconds\n"
  h += "\n"
  h += "typedef TensileStatus (*SolutionFunctionPointer)(\n"
  argList = solutionWriter.getArgList(solutions[0])
  for i in range(0, len(argList)):
    h += "  %s%s" % (argList[i], ",\n" if i < len(argList)-1 else ");\n\n")
  h += "static const SolutionFunctionPointer solutions[numSolutions] = {\n"
  for i in range(0, len(solutions)):
    solution = solutions[i]
    solutionName = Solution.getNameMin(solution, solutionMinNaming)
    h += "  %s" % solutionName
    if i < len(solutions)-1:
      h += ","
    h += "\n"
  h += "};\n"
  h += "\n"

  h += "const char *solutionNames[numSolutions] = {\n"
  for i in range(0, len(solutions)):
    solution = solutions[i]
    solutionName = Solution.getNameMin(solution, solutionMinNaming)
    h += "  \"%s\"" % solutionName
    if i < len(solutions)-1:
      h += ","
    h += "\n"
  h += "};\n"
  h += "\n"

  ##############################################################################
  # Runtime Structures
  ##############################################################################
  h += "/* runtime structures */\n"
  h += "DataType *initialC;\n"
  h += "DataType *referenceC;\n"
  h += "DataType *deviceOnHostC;\n"
  h += "DataType *initialA;\n"
  h += "DataType *initialB;\n"
  h += "TensileStatus status;\n"
  if globalParameters["Backend"] == "OCL":
    h += "unsigned int platformIdx = %u;\n" \
        % (globalParameters["PlatformIdx"])
    h += "unsigned int deviceIdx = %u;\n" \
        % (globalParameters["DeviceIdx"])
    h += "cl_platform_id platform;\n"
    h += "cl_device_id device;\n"
    h += "cl_context context;\n"
    h += "cl_command_queue stream;\n"
    h += "cl_mem deviceC;\n"
    h += "cl_mem deviceA;\n"
    h += "cl_mem deviceB;\n"
  else:
    h += "DataType *deviceC;\n"
    h += "static hipError_t status;\n"
    h += "static int deviceIdx = %u;\n" \
        % (globalParameters["DeviceIdx"])
  h += "\n"

  ##############################################################################
  # Benchmarking and Validation Parameters
  ##############################################################################
  h += "/* benchmarking parameters */\n"
  h += "const int numEnqueuesPerSync = %u;\n" \
      % (globalParameters["EnqueuesPerSync"])
  h += "const int numSyncsPerBenchmark = %u;\n" \
      % (globalParameters["SyncsPerBenchmark"])
  h += "const unsigned int numElementsToValidate = %s;\n" \
      % (str(globalParameters["NumElementsToValidate"]) \
      if globalParameters["NumElementsToValidate"] >= 0 \
      else "0xFFFFFFFF" )
  h += "const unsigned int validationMaxToPrint = %u;\n" \
      % globalParameters["ValidationMaxToPrint"]
  h += "const bool validationPrintValids = %s;\n" \
      % ("true" if globalParameters["ValidationPrintValids"] else "false")
  h += "size_t validationStride;\n"
  h += "\n"



  ##############################################################################
  # Generated Call to Reference
  ##############################################################################
  h += "/* generated call to reference */\n"
  h += "void generatedCallToReferenceCPU( unsigned int *sizes) {\n"
  h += "  TensileStatus status = tensileReferenceCPU(\n"
  h += "      referenceC,\n"
  h += "      initialA,\n"
  h += "      initialB,\n"
  h += "      alpha,\n"
  h += "      beta,\n"
  h += "      totalIndices,\n"
  h += "      sizes,\n"
  h += "      numIndicesC,\n"
  h += "      numIndicesAB,\n"
  h += "      indexAssignmentsA,\n"
  h += "      indexAssignmentsB,\n"
  h += "      complexConjugateA,\n"
  h += "      complexConjugateB,\n"
  h += "      validationStride );\n"
  h += "};\n"
  h += "\n"

  ##############################################################################
  # Generated Call to Solution
  ##############################################################################
  h += "/* generated call to solution */\n"
  h += "void generatedCallToSolution(\n"
  h += "    unsigned int solutionIdx,\n"
  h += "    unsigned int *sizes) {\n"
  h += "\n"
  h += "  // calculate parameters assuming packed data\n"
  # strides
  indexChars = globalParameters["IndexChars"]
  firstStride = 1
  if solution["ProblemType"]["UseInitialStrides"]:
    firstStride = 0
  lastStrideC = solution["ProblemType"]["NumIndicesC"]
  lastStrideA = len(solution["ProblemType"]["IndexAssignmentsA"])
  lastStrideB = len(solution["ProblemType"]["IndexAssignmentsB"])

  # calculate strides
  for i in range(0,lastStrideC):
    h += "  unsigned int strideC%u%s = 1" % (i, indexChars[i])
    for j in range(0, i):
      h += "*sizes[%i]" % j
    h += ";\n"
  for i in range(0,lastStrideA):
    h += "  unsigned int strideA%u%s = 1" % (i, \
        indexChars[solution["ProblemType"]["IndexAssignmentsA"][i]])
    for j in range(0, i):
      h += "*sizes[%i]" % \
        solution["ProblemType"]["IndexAssignmentsA"][j]
    h += ";\n"
  for i in range(0,lastStrideB):
    h += "  unsigned int strideB%u%s = 1" % (i, \
        indexChars[solution["ProblemType"]["IndexAssignmentsB"][i]])
    for j in range(0, i):
      h += "*sizes[%i]" % \
        solution["ProblemType"]["IndexAssignmentsB"][j]
    h += ";\n"
  for i in range(0, solution["ProblemType"]["TotalIndices"]):
    h += "  unsigned int size%s = sizes[%u];\n" % (indexChars[i], i)
  h += "\n"

  # function call
  h += "  // call solution function\n"
  h += "  solutions[solutionIdx]( deviceC, deviceA, deviceB,\n"
  h += "      alpha,\n"
  if solution["ProblemType"]["UseBeta"]:
    h += "      beta,\n"
  h += "      0, 0, 0, // offsets\n"
  for i in range(firstStride,lastStrideC):
    h += "      strideC%u%s,\n" % (i, indexChars[i])
  for i in range(firstStride,lastStrideA):
    h += "      strideA%u%s,\n" % (i, \
        indexChars[solution["ProblemType"]["IndexAssignmentsA"][i]])
  for i in range(firstStride,lastStrideB):
    h += "      strideB%u%s,\n" % (i, \
        indexChars[solution["ProblemType"]["IndexAssignmentsB"][i]])
  for i in range(0, solution["ProblemType"]["TotalIndices"]):
    h += "      size%s,\n" % indexChars[i]
  h += "      stream,\n"
  h += "      0, NULL, NULL); // events\n"
  h += "};\n"
  h += "\n"

  ##############################################################################
  # Results File Name
  ##############################################################################
  h += "/* results file name */\n"
  resultsFileName = os.path.join(globalParameters["WorkingPath"],"../../Data","%s.csv" % stepName)
  resultsFileName = resultsFileName.replace("\\", "\\\\")

  h += "const char *resultsFileName = \"%s\";\n" % resultsFileName

  benchmarkParametersFile.write(h)
  benchmarkParametersFile.close()



################################################################################
# Main
################################################################################
def main( config ):
  printStatus("Beginning")
  pushWorkingPath("1_BenchmarkProblemTypes")
  for problemType in config:
    if problemType is None:
      benchmarkProblemType({})
    else:
      benchmarkProblemType(problemType)
    if globalParameters["DebugPrintLevel"] >= 1:
      print ""

  printStatus("DONE.")
  popWorkingPath()


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
      results):
    #printStatus("beginning")
    #print benchmarkPermutations
    for hardcodedIdx in range(0, len(results)):
      hardcodedResults = results[hardcodedIdx]
      hardcodedParameters = hardcodedParameterList[hardcodedIdx]
      winningIdx = -1
      winningScore = -1
      # find fastest benchmark parameters for this hardcoded
      for benchmarkIdx in range(0, len(hardcodedResults)):
        benchmarkResult = hardcodedResults[benchmarkIdx]
        benchmarkScore = max(benchmarkResult) # take fastest regardless of size
        if benchmarkScore > winningScore:
          winningScore = benchmarkScore
          winningIdx = benchmarkIdx
      #print winningIdx, winningScore
      winningParameters = benchmarkPermutations[winningIdx]
      # (hardcodedParametersKey, oldWinningParameters, oldScore) = \
      matches = WinningParameterDict.get(hardcodedParameters, self.winners)
      if len(matches) != 1:
        printExit("Didn't find exactly 1 match")
      hardcodedParametersKey = matches[0][0]
      oldWinningParameters = matches[0][1]
      oldScore = matches[0][2]
      #print self.winners[hardcodedParametersKey]
      #print winningParameters
      self.winners[hardcodedParametersKey][0].update(winningParameters)
      self.winners[hardcodedParametersKey][1] = winningScore


    #(hardcodedParametersKey, winningParameters, score) = \
    #    WinningParameterDict.get(hardcodedParameters, self.winners)
    #print "Found %s -> %s" % ( hardcodedParametersKey, tmp)
    #self.winners[hardcodedParametersKey].update( \
    #    newWinningParameters)

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
    # TODO when new list is joining, we need to choose the fastest
    oldWinners = self.winners
    self.winners = {}

    # if this is first time, populate with dummies and early exit
    if len(oldWinners) == 0:
      for newHardcodedParameters in newHardcodedParameterList:
        self.winners[FrozenDictionary(newHardcodedParameters)] = [{},-1]
    else:
      for newHardcodedParameters in newHardcodedParameterList:
        # print newHardcodedParameters
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
          #print "Update Join %s" % Solution.getNameFull(newHardcodedParameters)
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
          matchDepthU = matchUnion["LoopUnroll"] * matchUnion["SplitU"]
          if matchDepthU != lookupDepthU:
            #print "DepthU NOT Matched"
            matchDepthU = False
          else:
            hardcodedFrozen.parameters["DepthU"] = lookupDepthU
            #print "DepthU Matched"
        if matchMacroTile and matchDepthU:
          #print "Match: %s" % Solution.getNameFull(hardcodedFrozen.parameters)
          matches.append([hardcodedFrozen, winningParameters, score])

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

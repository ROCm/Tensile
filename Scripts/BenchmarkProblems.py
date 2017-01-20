import sys
import os
from copy import deepcopy
from shutil import copy as shutil_copy
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

  ##############################################################################
  # For Each Benchmark Step
  ##############################################################################
  for benchmarkStepIdx in range(0, totalBenchmarkSteps):

    # Print Step Name
    benchmarkStep = benchmarkProcess[benchmarkStepIdx]
    stepName = str(benchmarkStep)
    print "\n\n"
    print hr
    print "# %s" % (stepName)
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
    print "prevParameters = %s" % str(benchmarkStep.prevParameters)
    totalBenchmarkPermutations = 1
    for benchmarkParamName in benchmarkStep.benchmarkParameters:
      totalBenchmarkPermutations *= len(benchmarkStep.benchmarkParameters[benchmarkParamName])
    print "totalSolutions = %u = %u (hardcoded) * %u (benchmark)" % \
        (totalBenchmarkPermutations*len(benchmarkStep.hardcodedParameters), \
        len(benchmarkStep.hardcodedParameters), totalBenchmarkPermutations)

    benchmarkPermutations = []
    for i in range(0, totalBenchmarkPermutations):
      permutation = {}
      pIdx = i
      for benchmarkParamName in benchmarkStep.benchmarkParameters:
        benchmarkParamValues = deepcopy(benchmarkStep.benchmarkParameters[benchmarkParamName])
        valueIdx = pIdx % len(benchmarkParamValues)
        permutation[benchmarkParamName] = benchmarkParamValues[valueIdx]
        pIdx /= len(benchmarkParamValues)
      benchmarkPermutations.append(permutation)

    ############################################################################
    # Enumerate Hardcoded Permutations
    ############################################################################
    print "Hardcoded Parameters"
    for hardcodedParamDict in benchmarkStep.hardcodedParameters:
      print Solution.getNameFull(hardcodedParamDict)
    print "Winners Initial\n%s" % winners
    winners.update( benchmarkStep.hardcodedParameters )
    print "Winners Updated Hardcodeds\n%s" % winners

    ############################################################################
    # Enumerate Solutions = Hardcoded * Benchmark
    ############################################################################
    for hardcodedParamDict in benchmarkStep.hardcodedParameters:
      for benchmarkPermutation in benchmarkPermutations:
        solution = {"ProblemType": deepcopy(benchmarkProcess.problemType.state)}
        solution.update(benchmarkPermutation)
        solution.update(hardcodedParamDict)
        winningParameters = winners[hardcodedParamDict]
        solution.update(winningParameters)
        # print solution

        # append default parameters where necessary
        for initialSolutionParameterName in benchmarkStep.initialSolutionParameters:
          if initialSolutionParameterName not in solution:
            solution[initialSolutionParameterName] = benchmarkStep.initialSolutionParameters[initialSolutionParameterName]
        # TODO check if solution matches problem size for exact tile kernels
        solutionObject = Solution(solution)
        if SolutionWriter.solutionParametersConsistent(solutionObject.state):
          printStatus("appending solution %s" % str(solutionObject))
          solutions.append(solutionObject)
        else:
          printStatus("rejecting solution %s" % str(solutionObject))
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
      pushWorkingPath("build")
      # create run.bat or run.sh which builds and runs
      runScriptName = os.path.join(globalParameters["WorkingPath"], \
        "run.%s" % "bat" if os.name == "nt" else "sh")
      runScriptFile = open(runScriptName, "w")
      runScriptFile.write("@echo. & echo %s & echo # %s: Configuring CMake & echo %s\n" \
          % (hr, stepName, hr))
      runScriptFile.write("cmake ../source\n")
      runScriptFile.write("@echo. & echo %s & echo # %s: Building Benchmark & echo %s\n" \
          % (hr, stepName, hr))
      runScriptFile.write("cmake --build . --config %s\n" \
          % globalParameters["CMakeBuildType"] )
      runScriptFile.write("@echo. & echo %s & echo # %s: Running Benchmark & echo %s\n" \
          % (hr, stepName, hr))
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
    try:
      resultsFile = open(resultsFileName, "r")
    except IOError:
      printExit("Can't open \"%s\" after BenchmarkStep %u: %s" \
          % (resultsFileName, benchmarkStepIdx, stepName) )
    winnerIndices = determineWinnersFromResults(resultsFile, solutions, \
        benchmarkStep.hardcodedParameters)
    print "NumwinnerIndices: %u" % len(winnerIndices)
    #print "NumOldDetermineds: %u" % len(determinedParameters)
    #print "OldDetermineds:"
    #determinedParametersMinNaming = Solution.getMinNaming(determinedParameters)
    # hardcodedParametersMinNaming = Solution.getMinNaming(hardcodedParameters)
    #for detParam in determinedParameters:
    #  print "  %s" % Solution.getNameMin(detParam, \
    #      determinedParametersMinNaming)
    print "NumHardcoded: %u" % len(benchmarkStep.hardcodedParameters)
    print "NumBenchmarked: %u" % len(benchmarkPermutations)
    #oldDeterminedParameters = deepcopy(determinedParameters)
    #determinedParameters = []
    for hardcodedParameterIdx in range(0, len(winnerIndices)):
      hardcodedParameters = \
          benchmarkStep.hardcodedParameters[hardcodedParameterIdx]
      winnerIdx = winnerIndices[hardcodedParameterIdx]
      winningParameters = benchmarkPermutations[winnerIdx]
      #print "Trying %s -> %s" % (Solution.getNameFull(hardcodedParameters), \
      #    Solution.getNameFull(winningParameters))
      winners[hardcodedParameters] = winningParameters # does update

      #print "Winner[%u:%s] is %u:%s" % (hardcodedParameterIdx, \
      #    Solution.getNameMin(hardcodedParameters, \
      #    hardcodedParametersMinNaming), winnerIdx, \
      #    Solution.getNameFull(benchmarkParameters) )
      #winningParameters = {}
      #winningParameters.update(hardcodedParameters)
      #winningParameters.update( \
      #    oldDeterminedParameters[hardcodedParameterIdx \
      #    % len(oldDeterminedParameters)])
      #winningParameters.update(benchmarkParameters)
      #determinedParameters.append(winningParameters)


    #print "NumNewDetermineds: %u" % len(determinedParameters)
    #determinedParametersMinNaming = Solution.getMinNaming(determinedParameters)
    #print "NewDetermineds:"
    #for detParam in determinedParameters:
    #  print "  %s" % Solution.getNameMin(detParam, \
    #      determinedParametersMinNaming)
    print "Winners Updated Winners\n%s" % winners


    popWorkingPath() # stepName
    print "%s%s\n# %s: End\n%s%s\n" % (hr, hr, stepName, hr, hr)
    #printExit("ON PURPOSE")

  popWorkingPath()




################################################################################
# Determine Winners From Results
################################################################################
def determineWinnersFromResults(resultsFile, solutions, hardcodedParameters):
  numHardcodedParameters = len(hardcodedParameters)
  totalSolutions = len(solutions)
  numSolutionsPerHardcodedParameter = totalSolutions / numHardcodedParameters
  print "%u solutions for each of %u hardcoded parameters\n" \
      % (numSolutionsPerHardcodedParameter, numHardcodedParameters)

  # setup data structures
  times = []
  numWins = []
  for i in range(0, numHardcodedParameters):
    times.append([])
    numWins.append([])
    for j in range(0, numSolutionsPerHardcodedParameter):
      times[i].append([])
      numWins[i].append(0)

  # read times
  csvFile = csv.reader(resultsFile)
  startIdx = solutions[0]["ProblemType"]["TotalIndices"] + 2
  rowIdx = 0
  for row in csvFile:
    if rowIdx == 0:
      rowIdx+=1
      continue
    else:
      for i in range(0, numHardcodedParameters):
        for j in range(0, numSolutionsPerHardcodedParameter):
          times[i][j].append(float(row[startIdx+j \
              + i*numSolutionsPerHardcodedParameter]))

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


################################################################################
# Write Benchmark Files
################################################################################
def writeBenchmarkFiles(solutions, problemSizes, stepName, filesToCopy):
  printStatus("Beginning")
  ensurePath(os.path.join(globalParameters["WorkingPath"], "Solutions"))
  ensurePath(os.path.join(globalParameters["WorkingPath"], "Kernels"))

  solutionNames = []
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

  solutionMinNaming = Solution.getMinNaming(solutions)
  kernelMinNaming = Solution.getMinNaming(kernels)
  solutionWriter = SolutionWriter(globalParameters["Backend"], \
      solutionMinNaming, kernelMinNaming)
  kernelWriter = KernelWriter(globalParameters["Backend"], kernelMinNaming)

  ##############################################################################
  # Write Solutions
  ##############################################################################
  for solution in solutions:
    # get solution name
    solutionName = Solution.getNameMin(solution.state, solutionMinNaming)
    solutionNames.append(solutionName)
    printStatus("Writing files for solution %s" % solutionName )

    # write solution.cpp
    solutionSourceFile = open(os.path.join(globalParameters["WorkingPath"], \
        "Solutions", solutionName+".cpp"), "w")
    solutionSourceFile.write( \
        solutionWriter.getSourceFileString(solution))
    solutionSourceFile.close()

    # write solution.h
    solutionHeaderFile = open(os.path.join(globalParameters["WorkingPath"], \
        "Solutions", solutionName+".h"), "w")
    solutionHeaderFile.write( \
        solutionWriter.getHeaderFileString(solution))
    solutionHeaderFile.close()

  ##############################################################################
  # Write Kernels
  ##############################################################################
  for kernel in kernels:
    # get kernel name
    kernelName = Solution.getNameMin(kernel, kernelMinNaming)
    kernelNames.append(kernelName)

    # write kernel.cpp
    kernelSourceFile = open(os.path.join(globalParameters["WorkingPath"], \
        "Kernels", kernelName+".cpp"), "w")
    kernelSourceFile.write( kernelWriter.getSourceFileString(kernel))
    kernelSourceFile.close()

    # write kernel.h
    kernelHeaderFile = open(os.path.join(globalParameters["WorkingPath"], \
        "Kernels", kernelName+".h"), "w")
    kernelHeaderFile.write( kernelWriter.getHeaderFileString(kernel))
    kernelHeaderFile.close()

  ##############################################################################
  # Write CMake
  ##############################################################################
  generatedFile = open(os.path.join(globalParameters["WorkingPath"], \
      "Generated.cmake"), "w")
  generatedFile.write(globalParameters["CMakeHeader"])
  generatedFile.write("set( TensileBenchmark_Solutions\n")
  # write solution names
  for solutionName in solutionNames:
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions/%s.h\n" \
        % (solutionName) )
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions/%s.cpp\n" \
        % (solutionName) )
  generatedFile.write("  )\n")

  # write kernel names
  generatedFile.write("set( TensileBenchmark_Kernels\n")
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
  for solutionName in solutionNames:
    h += "#include \"" + solutionName + ".h\"\n"
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
  for i in range(0, len(solutionNames)):
    solutionName = solutionNames[i]
    h += "  %s" % solutionName
    if i < len(solutionNames)-1:
      h += ","
    h += "\n"
  h += "};\n"
  h += "\n"
  h += "const char *solutionNames[numSolutions] = {\n"
  for i in range(0, len(solutionNames)):
    solutionName = solutionNames[i]
    h += "  \"%s\"" % solutionName
    if i < len(solutionNames)-1:
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
  def __setitem__( self, hardcodedParameters, newWinningParameters ):
    (hardcodedParametersKey, tmp) = \
        WinningParameterDict.get(hardcodedParameters, self.winners)
    #print "Found %s -> %s" % ( hardcodedParametersKey, tmp)
    self.winners[hardcodedParametersKey].update( \
        newWinningParameters)

  ##########################################################
  # Get Winning Parameters For Hardcoded Parameters
  def __getitem__( self, hardcodedParameters ):
    (tmp, winningParameters) = \
        WinningParameterDict.get(hardcodedParameters, self.winners)
    return winningParameters

  ##########################################################
  # Update Hardcoded Parameters In Winning Parameters
  # could be forking, joining or adding parameters to same hardcodeds
  def update(self, newHardcodedParameterList ):
    oldWinners = self.winners
    self.winners = {}
    for newHardcodedParameters in newHardcodedParameterList:
      # print newHardcodedParameters
      (oldHardcodedParameters, winningParameters) = \
          WinningParameterDict.get(newHardcodedParameters, oldWinners)
      if winningParameters != None:
        self.winners[FrozenDictionary(newHardcodedParameters)] = \
            winningParameters
      else:
        self.winners[FrozenDictionary(newHardcodedParameters)] = {}

  ##########################################################
  # Get Winning Parameters For Hardcoded Parameters
  # a match if no key in hardcoded has a different value than lookup
  @staticmethod
  def get( lookupHardcodedParameters, winners ):
    #print "  GET looks: %s" % Solution.getNameFull(lookupHardcodedParameters)
    for hardcodedFrozen in winners:
      winningParameters = winners[hardcodedFrozen]
      match = True
      for paramName in hardcodedFrozen:
        if paramName in lookupHardcodedParameters:
          #print "comparing %s: %s == %s" % (paramName, \
          #lookupHardcodedParameters[paramName], \
          #    hardcodedFrozen[paramName])
          if lookupHardcodedParameters[paramName] != \
              hardcodedFrozen[paramName]:
            #print "mismatch"
            match = False
          #else:
          #  print "match"
        #else:
        #  print "%s not in lookup" % (paramName)

      if match:
        #print "  GET found: %s" \
        #    % Solution.getNameFull(lookupHardcodedParameters)
        return (hardcodedFrozen, winningParameters)
    return (None, None)
    printExit("WinningParameterDict::get didn't find %s" \
        % str(lookupHardcodedParameters))

  ##########################################################
  # To String
  def __str__(self):
    state = ""
    idx = 0
    for hardcodedParameters in self.winners:
      winningParameters = self.winners[hardcodedParameters]
      state += "  %2u: %s -> %s\n" % (idx, hardcodedParameters, \
          Solution.getNameFull(winningParameters))
      idx += 1
    return state
  def __repr__(self):
    return self.__str__()

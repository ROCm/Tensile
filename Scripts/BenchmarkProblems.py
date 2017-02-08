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
        "BenchmarkClient.cpp",
        "BenchmarkClient.h",
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
    shutil_copy(
      os.path.join(globalParameters["SourcePath"], "BenchmarkClient.cmake"),
      os.path.join(globalParameters["WorkingPath"], "CMakeLists.txt" ) )

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
        runScriptFile.write("cmake -DCMAKE_GENERATOR_PLATFORM=x64 ../source\n")
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


  # write solution, kernels and CMake
  LibraryWriter.writeSolutionsAndKernels( \
      globalParameters["WorkingPath"], solutions)


  """
  ##############################################################################
  # Write Solutions
  ##############################################################################
  if globalParameters["MergeFiles"]:
    solutionSourceFile = open(os.path.join(globalParameters["WorkingPath"], \
        "GeneratedSolutions.cpp"), "w")
    solutionHeaderFile = open(os.path.join(globalParameters["WorkingPath"], \
        "GeneratedSolutions.h"), "w")
    solutionSourceFile.write("#include \"GeneratedSolutions.h\"\n")
    solutionHeaderFile.write("#include \"TensileTypes.h\"\n")
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
    solutionSourceFile.write(CHeader)
    solutionSourceFile.write( \
        solutionWriter.getSourceFileString(solution))
    if not globalParameters["MergeFiles"]:
      solutionSourceFile.close()

    # write solution.h
    if not globalParameters["MergeFiles"]:
      solutionHeaderFile = open(os.path.join(globalParameters["WorkingPath"], \
          "Solutions", solutionFileName+".h"), "w")
    solutionHeaderFile.write(CHeader)
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
    kernelSourceFile.write(CHeader)
    kernelSourceFile.write( kernelWriter.getSourceFileString(kernel))
    if not globalParameters["MergeFiles"]:
      kernelSourceFile.close()

    # write kernel.h
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile = open(os.path.join(globalParameters["WorkingPath"], \
          "Kernels", kernelName+".h"), "w")
    kernelHeaderFile.write(CHeader)
    kernelHeaderFile.write( kernelWriter.getHeaderFileString(kernel))
    if not globalParameters["MergeFiles"]:
      kernelHeaderFile.close()
  # close merged
  if globalParameters["MergeFiles"]:
    kernelHeaderFile.close()
  """

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
  # Write CMake
  ##############################################################################
  generatedFile = open(os.path.join(globalParameters["WorkingPath"], \
      "Generated.cmake"), "w")
  generatedFile.write(CMakeHeader)
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
  generatedFile.write("set( TensileClient TensileBenchmark_%s)\n" \
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
  benchmarkParametersFile.write(CHeader)

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
  h += "float solutionPerf[numProblems][numSolutions]; // milliseconds\n"
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
    h += "DataType *deviceA;\n"
    h += "DataType *deviceB;\n"
    h += "hipStream_t stream;\n"
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
  h += "unsigned int dataInitType = %s;\n" % globalParameters["DataInitType"]
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
      winningScore = -1
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
  pushWorkingPath(globalParameters["BenchmarkProblemsPath"])
  dataPath = os.path.join(globalParameters["WorkingPath"], globalParameters["BenchmarkPath"])
  ensurePath(dataPath)
  for problemType in config:
    problemTypeObj = ProblemType(problemType)

    # Benchmark Problem Type
    if problemType is None:
      resultsFileBase = benchmarkProblemType({})
    else:
      resultsFileBase = benchmarkProblemType(problemType)

    # Copy Data
    resultsFileName = resultsFileBase + ".csv"
    solutionsFileName = resultsFileBase + ".yaml"
    newResultsFileName = os.path.join(globalParameters["WorkingPath"], \
        "Results", "%s.csv" % str(problemTypeObj))
    newSolutionsFileName = os.path.join(globalParameters["WorkingPath"], \
        "Results", "%s.yaml" % str(problemTypeObj))
    shutil_copy( resultsFileName, newResultsFileName )
    shutil_copy( solutionsFileName, newSolutionsFileName )

  printStatus("DONE.")
  popWorkingPath()

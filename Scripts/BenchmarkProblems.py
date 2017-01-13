import sys
import os
from copy import deepcopy
from shutil import copy as shutil_copy

from BenchmarkProcess import *
from Common import *
from Structs import *
from SolutionWriter import *
from KernelWriter import *
from subprocess import Popen


################################################################################
def benchmarkProblemType( config ):


  # convert config to full benchmark process (resolves defaults)
  benchmarkProcess = BenchmarkProcess(config)
  problemTypeName = str(benchmarkProcess.problemType)
  pushWorkingPath(problemTypeName)
  ensurePath(os.path.join(globalParameters["WorkingPath"],"Data"))

  totalBenchmarkSteps = len(benchmarkProcess)
  determinedParameters = {} # winner chosen from benchmark
  for benchmarkStepIdx in range(0, totalBenchmarkSteps):
    benchmarkStep = benchmarkProcess[benchmarkStepIdx]
    print "BenchmarkStepIdx: %u" % benchmarkStepIdx
    stepName = str(benchmarkStep)
    pushWorkingPath(stepName)
    # copy files to benchmark source directory
    pushWorkingPath("source")
    filesToCopy = [
        "TensileBenchmark_Main.cpp",
        "TensileBenchmark_Main.h",
        "MathTemplates.cpp",
        "MathTemplates.h",
        "Tensile.cpp",
        "Tensile.h",
        "ReferenceCPU.cpp",
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

    # (1) create list of solutions
    solutions = []
    currentSolution = {"ProblemType": deepcopy(benchmarkProcess.problemType.state) }
    print "prevParameters = %s" % str(benchmarkStep.prevParameters)
    # append previously determined values
    for prevParamDict in benchmarkStep.prevParameters:
      for prevParamName in prevParamDict:
        if prevParamName in determinedParameters:
          paramValue = determinedParameters[prevParamName]
          currentSolution[prevParamName] = deepcopy(paramValue)
        else:
          printWarning("Parameter %s should have been determined, but wasn't" % prevParamName)
    # multiplicity of benchmark params
    totalBenchmarkPermutations = 1
    for benchmarkParamName in benchmarkStep.benchmarkParameters:
      totalBenchmarkPermutations *= len(benchmarkStep.benchmarkParameters[benchmarkParamName])
    print "totalSolutions = %u = %u (hardcoded) * %u (benchmark)" % \
        (totalBenchmarkPermutations*len(benchmarkStep.hardcodedParameters), \
        len(benchmarkStep.hardcodedParameters), totalBenchmarkPermutations)
    for i in range(0, totalBenchmarkPermutations):
      pIdx = i
      for benchmarkParamName in benchmarkStep.benchmarkParameters:
        benchmarkParamValues = deepcopy(benchmarkStep.benchmarkParameters[benchmarkParamName])
        valueIdx = pIdx % len(benchmarkParamValues)
        currentSolution[benchmarkParamName] = deepcopy(benchmarkParamValues[valueIdx])
        pIdx /= len(benchmarkParamValues)

      # multiplicity of hardcoded params
      for hardcodedParamDict in benchmarkStep.hardcodedParameters:
        fullSolution = {}
        fullSolution.update(currentSolution)
        fullSolution.update(hardcodedParamDict)

        # append default parameters where necessary
        for initialSolutionParameterName in benchmarkStep.initialSolutionParameters:
          if initialSolutionParameterName not in fullSolution:
            fullSolution[initialSolutionParameterName] = benchmarkStep.initialSolutionParameters[initialSolutionParameterName]
        # TODO check if solution matches problem size for exact tile kernels
        solutionObject = Solution(fullSolution, solutions)
        printStatus("appending solution %s" % str(solutionObject))
        solutions.append(solutionObject)

    # write benchmarkFiles
    writeBenchmarkFiles(solutions, benchmarkStep.problemSizes, stepName, filesToCopy)

    popWorkingPath() # source
    pushWorkingPath("build")
    # create run.bat or run.sh which builds and runs
    hr = "#####################################################################"
    runScriptName = os.path.join(globalParameters["WorkingPath"], \
      "run.%s" % "bat" if os.name == "nt" else "sh")
    runScriptFile = open(runScriptName, "w")
    runScriptFile.write("@echo. & echo %s & echo # Configuring CMake & echo %s\n" \
        % (hr, hr))
    runScriptFile.write("cmake ../source\n")
    runScriptFile.write("@echo. & echo %s & echo # Building TensileBenchmark & echo %s\n" \
        % (hr, hr))
    runScriptFile.write("cmake --build . --config %s\n" \
        % globalParameters["CMakeBuildType"] )
    runScriptFile.write("@echo. & echo %s & echo # Running TensileBenchmark & echo %s\n" \
        % (hr, hr))
    runScriptFile.write(os.path.join("Release","TensileBenchmark_%s%s") \
        % (stepName, ".exe" if os.name == "nt" else "") )
    runScriptFile.close()
    print "\n\n"
    print "####################################################################"
    print "# Executing Benchmark Step %u: %s" % (benchmarkStepIdx, stepName)
    print "####################################################################"
    print "\n\n"
    process = Popen(runScriptName, cwd=globalParameters["WorkingPath"])
    status = process.wait()

    printExit("%s returned %u" % (runScriptName, status) )

    # build benchmark
    # execute benchmark
    popWorkingPath() # build

    popWorkingPath() # benchmark

  popWorkingPath()

def writeBenchmarkFiles(solutions, problemSizes, stepName, filesToCopy):
  printStatus("Beginning")
  ensurePath(os.path.join(globalParameters["WorkingPath"], "Solutions"))
  ensurePath(os.path.join(globalParameters["WorkingPath"], "Kernels"))

  solutionNames = []
  kernelNames = []
  kernels = []

  # get all unique kernels for solutions
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

  # open generated.cmake
  generatedFile = open(os.path.join(globalParameters["WorkingPath"], \
      "Generated.cmake"), "w")
  generatedFile.write(globalParameters["CMakeHeader"])
  generatedFile.write("set( TensileBenchmark_Solutions\n")
  # open solutions.h
  #allSolutionsHeaderFile = open(os.path.join(globalParameters["WorkingPath"],\
  #    "GeneratedSolutions.h"), "w")
  #allSolutionsHeaderFile.write(globalParameters["CHeader"])
  # write solution names
  for solutionName in solutionNames:
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions/%s.h\n" \
        % (solutionName) )
    generatedFile.write("  ${CMAKE_SOURCE_DIR}/Solutions/%s.cpp\n" \
        % (solutionName) )
    #allSolutionsHeaderFile.write("#include \"" + solutionName + ".h\"\n")
  generatedFile.write("  )\n")
  # close solutions header
  #allSolutionsHeaderFile.close()

  # open kernels.cmake
  generatedFile.write("set( TensileBenchmark_Kernels\n")
  # write kernel names
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


  # generated benchmark parameters
  benchmarkParametersFile = open(os.path.join(globalParameters["WorkingPath"], \
      "GeneratedBenchmarkParameters.h"), "w")
  benchmarkParametersFile.write(globalParameters["CHeader"])

  h = ""
  for solutionName in solutionNames:
    h += "#include \"" + solutionName + ".h\"\n"

  h += "\n"

  h += "/* does this index have independent size range or map to same value as another index */\n"
  h += "static const unsigned int totalIndices = %u;\n" \
      % (problemSizes.totalIndices)
  h += "static const bool indexIsSized[totalIndices] = {\n"
  for i in range(0, problemSizes.totalIndices):
    h += "  %s" % ("true" if problemSizes.indexIsSized[i] else "false")
    if i < problemSizes.totalIndices-1:
      h += ","
    h += "\n"
  h += "  };\n\n"


  h += "/* min, stride, stride_incr, max */\n"
  h += "static const unsigned int numIndicesSized = %u;\n" \
      % len(problemSizes.indicesSized)
  h += "static const unsigned int indicesSized[numIndicesSized][4] = {\n"
  for i in range(0, len(problemSizes.indicesSized)):
    r = problemSizes.indicesSized[i]
    h += "  { %u, %u, %u, %u }" % (r[0], r[1], r[2], r[3])
    if i < len(problemSizes.indicesSized)-1:
      h += ","
    h += "\n"
  h += "  };\n\n"

  h += "/* which other index does this index map (for its size) */\n"
  h += "static const unsigned int numIndicesMapped = %u;\n" \
      % len(problemSizes.indicesMapped)
  h += "static const unsigned int indicesMapped[numIndicesMapped] = {\n"
  for i in range(0, len(problemSizes.indicesMapped)):
    h += "  %u" % problemSizes.indicesMapped[i]
    if i < len(problemSizes.indicesMapped)-1:
      h += ","
    h += "\n"
  h += "  };\n"

  h += "unsigned int problemSize[totalIndices];\n"

  h += "static const unsigned int bytesPerElements = %u;\n" \
      % (solutions[0]["ProblemType"]["DataType"].numBytes())

  h += "unsigned int numSolutions = %u;\n" % len(solutions)
  h += "typedef TensileStatus (*SolutionFunctionPointer)(\n"
  argList = solutionWriter.getArgList(solutions[0])
  for i in range(0, len(argList)):
    h += "  %s%s" % (argList[i], ",\n" if i < len(argList)-1 else ");\n\n")
  h += "static const SolutionFunctionPointer solutions[%u] = {\n" \
      % (len(solutions))
  for i in range(0, len(solutionNames)):
    solutionName = solutionNames[i]
    h += "  %s" % solutionName
    if i < len(solutionNames)-1:
      h += ","
    h += "\n"
  h += "};\n"
  h += "\n"
  h += "const char *solutionNames[%u] = {\n" \
      % (len(solutions))
  for i in range(0, len(solutionNames)):
    solutionName = solutionNames[i]
    h += "  \"%s\"" % solutionName
    if i < len(solutionNames)-1:
      h += ","
    h += "\n"
  h += "};\n"
  h += "\n"
  h += "unsigned int numProblems = %u;\n" % problemSizes.totalProblemSizes
  h += "float solutionTimes[%u][%u];\n" \
      % (problemSizes.totalProblemSizes, len(solutions))

  sizeC = 1
  sizeA = 1
  sizeB = 1
  for idx in range(0, solution["ProblemType"]["NumIndicesC"]):
    sizeC *= problemSizes.indexMax[idx]
  for idx in solution["ProblemType"]["IndexAssignmentsA"]:
    sizeA *= problemSizes.indexMax[idx]
  for idx in solution["ProblemType"]["IndexAssignmentsB"]:
    sizeB *= problemSizes.indexMax[idx]
  typeName = solution["ProblemType"]["DataType"].toCpp()
  h += "typedef %s DataType;\n" % (typeName)
  h += "size_t maxSizeC = %u;\n" % (sizeC)
  h += "DataType *initialC;\n"
  h += "DataType *referenceC;\n"
  h += "DataType *deviceOnHostC;\n"
  h += "size_t maxSizeA = %u;\n" % (sizeA)
  h += "DataType *initialA;\n"
  h += "size_t maxSizeB = %u;\n" % (sizeB)
  h += "DataType *initialB;\n"
  h += "DataType alpha;\n"
  h += "DataType beta;\n"
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

  h += "const int numEnqueuesPerSync = %u;\n" \
      % (globalParameters["EnqueuesPerSync"])
  h += "const int numSyncsPerBenchmark = %u;\n" \
      % (globalParameters["SyncsPerBenchmark"])


  # problem description
  h += "const unsigned int numIndicesC = %u;\n" % solution["ProblemType"]["NumIndicesC"]
  h += "const unsigned int indexAssignmentsA[%u] = {\n" \
      % len(solution["ProblemType"]["IndexAssignmentsA"])
  h += "  %u" % solution["ProblemType"]["IndexAssignmentsA"][0]
  for i in range(1, len(solution["ProblemType"]["IndexAssignmentsA"])):
    h += ",\n  %u" % solution["ProblemType"]["IndexAssignmentsA"][i]
  h += "\n};\n"
  h += "const unsigned int indexAssignmentsB[%u] = {\n" \
      % len(solution["ProblemType"]["IndexAssignmentsB"])
  h += "  %u" % solution["ProblemType"]["IndexAssignmentsB"][0]
  for i in range(1, len(solution["ProblemType"]["IndexAssignmentsB"])):
    h += ",\n  %u" % solution["ProblemType"]["IndexAssignmentsB"][i]
  h += "\n};\n"


  # enqueue solution for problem size
  h += "void generatedCallToReferenceCPU(\n"
  h += "  unsigned int *sizes) {\n"
  h += "};\n\n"

  h += "void generatedCallToSolution(\n"
  h += "  unsigned int solutionIdx,\n"
  h += "  unsigned int *sizes) {\n"
  h += "};\n"

  # output filename
  resultsFileName = os.path.join(globalParameters["WorkingPath"],"..","%s.csv" % stepName)
  resultsFileName = resultsFileName.replace("\\", "\\\\")

  h += "const char *resultsFileName = \"%s\";\n" % resultsFileName

  benchmarkParametersFile.write(h)
  benchmarkParametersFile.close()

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

import sys
from copy import deepcopy
from shutil import copy as shutil_copy

from BenchmarkProcess import *
from Common import *
from Structs import *
from SolutionWriter import *
from KernelWriter import *


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
        "StructOperations.cpp",
        "StructOperations.h",
        "Tensile.cpp",
        "Tensile.h",
        "Solution.cpp",
        "Solution.h",
        "SolutionTensorContractionCPU.cpp",
        "SolutionTensorContractionCPU.h",
        "Tools.cpp",
        "Tools.h",

        ]
    for f in filesToCopy:
      shutil_copy(
          os.path.join(globalParameters["SourcePath"], f),
          globalParameters["WorkingPath"] )
    shutil_copy(
        os.path.join(globalParameters["SourcePath"], "TensileBenchmark_CMakeLists.txt"),
        os.path.join(globalParameters["WorkingPath"], "CmakeLists.txt" ) )
    #print "hardcodedParameters = %s" % str(benchmarkStep.hardcodedParameters)

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
      #print "0-solutionList=%s" % str(solutions)
      pIdx = i
      for benchmarkParamName in benchmarkStep.benchmarkParameters:
        #print "1-solutionList=%s" % str(solutions)
        benchmarkParamValues = deepcopy(benchmarkStep.benchmarkParameters[benchmarkParamName])
        valueIdx = pIdx % len(benchmarkParamValues)
        currentSolution[benchmarkParamName] = deepcopy(benchmarkParamValues[valueIdx])
        pIdx /= len(benchmarkParamValues)
      #print "CurrentSolution (before hardcoding)"
      #print currentSolution

      # multiplicity of hardcoded params
      for hardcodedParamDict in benchmarkStep.hardcodedParameters:
        #print "2-solutionList=%s" % str(solutions)
        fullSolution = {}
        fullSolution.update(currentSolution)
        #print "fullSolution %s" % str(fullSolution)
        #print "3-solutionList=%s" % str(solutions)
        fullSolution.update(hardcodedParamDict)
        #print "fullSolution %s" % str(fullSolution)

        # append default parameters where necessary
        #print benchmarkStep.initialSolutionParameters
        for initialSolutionParameterName in benchmarkStep.initialSolutionParameters:
          if initialSolutionParameterName not in fullSolution:
            fullSolution[initialSolutionParameterName] = benchmarkStep.initialSolutionParameters[initialSolutionParameterName]
        # TODO check if solution matches problem size for exact tile kernels
        #print "4-solutionList=%s" % str(solutions)
        solutionObject = Solution(fullSolution, solutions)
        #print "5-solutionList=%s" % str(solutions)
        #print solutionObject.state
        printStatus("appending solution %s" % str(solutionObject))
        solutions.append(solutionObject)
        #print "6-solutionList=%s" % str(solutions)
      #print ""

    # write benchmarkFiles
    writeBenchmarkFiles(solutions, benchmarkStep.problemSizes)

    popWorkingPath() # source
    pushWorkingPath("build")
    # create run.bat or run.sh which builds and runs
    # build benchmark
    # execute benchmark
    popWorkingPath() # build

    popWorkingPath() # benchmark

  popWorkingPath()

def writeBenchmarkFiles(solutions, problemSizes):
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

    """
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
    """

  # open solutions.cmake
  solutionsCMakeFile = open(os.path.join(globalParameters["WorkingPath"], \
      "GeneratedSolutions.cmake"), "w")
  solutionsCMakeFile.write(globalParameters["CMakeHeader"])
  solutionsCMakeFile.write("set( SolutionFiles\n")
  # open solutions.h
  allSolutionsHeaderFile = open(os.path.join(globalParameters["WorkingPath"],\
      "GeneratedSolutions.h"), "w")
  allSolutionsHeaderFile.write(globalParameters["CHeader"])
  # write solution names
  for solutionName in solutionNames:
    solutionHeaderFilePath = os.path.join( \
        globalParameters["WorkingPath"], "Solutions", solutionName+".h")
    solutionsCMakeFile.write("  " + solutionHeaderFilePath + "\n" )
    allSolutionsHeaderFile.write("#include \"" + solutionName + ".h\"\n")
  # close solutions
  solutionsCMakeFile.close()
  allSolutionsHeaderFile.close()

  # open kernels.cmake
  kernelsCMakeFile = open(os.path.join(globalParameters["WorkingPath"], \
      "GeneratedKernels.cmake"), "w")
  kernelsCMakeFile.write(globalParameters["CMakeHeader"])
  kernelsCMakeFile.write("set( KernelFiles\n")
  # write kernel names
  for kernelName in kernelNames:
    kernelHeaderFilePath = os.path.join( \
        globalParameters["WorkingPath"], "Kernels", kernelName+".h")
    kernelsCMakeFile.write("  " + kernelHeaderFilePath + "\n" )
  # close kernels
  kernelsCMakeFile.close()


  # generated benchmark parameters
  benchmarkParametersFile = open(os.path.join(globalParameters["WorkingPath"], \
      "GeneratedBenchmarkParameters.h"), "w")
  benchmarkParametersFile.write(globalParameters["CHeader"])

  h = ""

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
  h += "static const unsigned int indicesMapped[numIndicesMapped][4] = {\n"
  for i in range(0, len(problemSizes.indicesMapped)):
    h += "  %u" % problemSizes.indicesMapped[i]
    if i < len(problemSizes.indicesMapped)-1:
      h += ","
    h += "\n"
  h += "  };\n"

  h += "static const unsigned int bytesPerElements = %u;\n" \
      % (solutions[0]["ProblemType"]["DataType"].numBytes())

  h += "typedef void( *SolutionFunctionPointer)();\n"
  h += "static const SolutionFunctionPointer solutions[%u] = {\n" \
      % (len(solutions))
  for i in range(0, len(solutionNames)):
    solutionName = solutionNames[i]
    h += "  %s" % solutionName
    if i < len(solutionNames)-1:
      h += ","
    h += "\n"
  h += "};\n"

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

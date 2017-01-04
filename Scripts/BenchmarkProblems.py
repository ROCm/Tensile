import sys
from copy import deepcopy
from shutil import copy as shutil_copy

from BenchmarkProcess import *
from Common import *
from Structs import *


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
          currentSolution[prevParamName] = paramValue
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
        benchmarkParamValues = benchmarkStep.benchmarkParameters[benchmarkParamName]
        valueIdx = pIdx % len(benchmarkParamValues)
        currentSolution[benchmarkParamName] = benchmarkParamValues[valueIdx]
        pIdx /= len(benchmarkParamValues)

      # multiplicity of hardcoded params
      for hardcodedParamDict in benchmarkStep.hardcodedParameters:
        fullSolution = deepcopy(currentSolution)
        # TODO dict is showing up as list of dicts sometimes
        currentSolution.update(hardcodedParamDict)

        # append default parameters where necessary
        #print benchmarkStep.initialSolutionParameters
        for initialSolutionParameterName in benchmarkStep.initialSolutionParameters:
          if initialSolutionParameterName not in fullSolution:
            fullSolution[initialSolutionParameterName] = benchmarkStep.initialSolutionParameters[initialSolutionParameterName]
        # TODO check if solution matches problem size for exact tile kernels
        solutionObject = Solution(fullSolution)
        printStatus("appending solution %s" % str(solutionObject))
        solutions.append(solutionObject)
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

  solutionNames = []
  kernelNames = []
  kernels = set()

  solutionMinNaming = Solution.getMinNaming(solutions)
  for solution in solutions:
    # write solution .cpp, .h
    # append to solution names
    # append kernels to set
    pass

  for kernel in kernels:
    pass
    # write kernel .cpp, .h
    # append to kernel names


    # Kernels.cmake
    # Solutions.cmake
    # ProblemSizeRange (numDims, array of stride/incr/min/max

    #
    # FileWriter:
    #   WriteBenchmarkFiles
    #     initSolutionForProblem rewrite
    #     max tensor size
    #     include solutions
    #
    #   WriteBackendFiles
    #     SolutionSelection
    #   writeKernelFiles(kernelSet)
    #   writeSolutionFiles(solutionSet)
    #   getKernelSourceFileString
    #   getKernelHeaderFileString
    #

# benchmarking directory structure
# build/
#   1_BenchmarkProblemTypes
#     Cij_Aik_Bjk_SBOI
#       1_ParamName
#       2_ParamName
#       3_Fork
#       4_Join
#       5_Final
#       0_Data
#         step1.csv
#         step2.csv
#         final.csv
#     Cij_Aik_Bkj_SBOI
#       ...
#   2_Analysis
#     Cij_Aik_Bjk_SBOI.yaml
#     Cij_Aik_Bjk_SBOI.yaml
#     LibName.yaml

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

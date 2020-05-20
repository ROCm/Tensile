################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import globalParameters, HR, pushWorkingPath, popWorkingPath, print1, CHeader, printWarning, listToInitializer, ClientExecutionLock
from . import ClientExecutable
from . import YAMLIO

import os
import subprocess
from shutil import copy as shutil_copy
from shutil import rmtree

from .Contractions import FreeIndex


################################################################################
# Main
################################################################################
def main( config ):
  libraryLogicPath = os.path.join(globalParameters["WorkingPath"], \
      globalParameters["LibraryLogicPath"])
  stepBaseDir = pushWorkingPath(globalParameters["LibraryClientPath"])


  ##############################################################################
  # Copy Source Files
  ##############################################################################
  pushWorkingPath("source")
  filesToCopy = [
      "SolutionMapper.h",
      "Client.cpp",
      "Client.h",
      "DeviceStats.h",
      "ReferenceCPU.h",
      "TensorUtils.h",
      "MathTemplates.cpp",
      "MathTemplates.h",
      "KernelHeader.h",
      "Tools.h",
      "TensileCreateLibrary.cmake",
      ]

  for f in filesToCopy:
    shutil_copy(
        os.path.join(globalParameters["SourcePath"], f),
        globalParameters["WorkingPath"] )
  #if globalParameters["NewClient"] == 2:
    #shutil_copy(
    #    os.path.join(globalParameters["SourcePath"], "CMakeListsCreateLibrary.txt"),
    #    os.path.join(globalParameters["WorkingPath"], "CMakeLists.txt") )
  #else:
  if globalParameters["NewClient"] < 2:
    shutil_copy(
        os.path.join(globalParameters["SourcePath"], "CMakeLists.txt"),
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

  ##############################################################################
  # Read Logic Files
  ##############################################################################
  logicFiles = [os.path.join(libraryLogicPath, f) for f \
      in os.listdir(libraryLogicPath) \
      if (os.path.isfile(os.path.join(libraryLogicPath, f)) \
      and os.path.splitext(f)[1]==".yaml")]
  print1("LogicFiles: %s" % logicFiles)
  functions = []
  functionNames = []
  enableHalf = False
  for logicFileName in logicFiles:
    (scheduleName, deviceNames, problemType, solutionsForType, \
        indexOrder, exactLogic, rangeLogic, newLibrary, architectureName) \
        = YAMLIO.readLibraryLogicForSchedule(logicFileName)
    if problemType["DataType"].isHalf():
        enableHalf = True
    functions.append((scheduleName, problemType))
    functionNames.append("tensile_%s" % (problemType))
  globalParameters["EnableHalf"] = enableHalf

  ##############################################################################
  # Write Generated Header
  ##############################################################################
  forBenchmark = False
  solutions = None
  problemSizes = None
  stepName = None
  solutionSummationSizes = None
  if globalParameters["NewClient"] != 2:
    if logicFiles:
      writeClientParameters(forBenchmark, solutions, problemSizes, stepName, \
          functions, solutionSummationSizes, stepBaseDir)
  popWorkingPath() # source

  ##############################################################################
  # Run Build Script
  ##############################################################################
  # if redo=true, clobber the build directory
  if globalParameters["ForceRedoLibraryClient"]:
    rmtree(os.path.join(globalParameters["WorkingPath"], "build"), \
        ignore_errors=True)

  forBenchmark = False
  enableTileSelection = False
  returncode = runClient(libraryLogicPath, forBenchmark, enableTileSelection)

  popWorkingPath() # LibraryClient

  return returncode


################################################################################
# Write Run Script
################################################################################

def runClient(libraryLogicPath, forBenchmark, enableTileSelection):
  # write runScript

  pushWorkingPath("build")
  path = globalParameters["WorkingPath"]
  if globalParameters["NewClient"] < 2:
    buildScriptName = writeBuildOldClientScript(path, libraryLogicPath, forBenchmark, enableTileSelection)
    runScriptName = writeRunScript(path, libraryLogicPath, forBenchmark, enableTileSelection)

    subprocess.check_call(buildScriptName, cwd=path)

    # run runScript
    with ClientExecutionLock():
      process = subprocess.Popen(runScriptName, cwd=path)
      process.communicate()

    if process.returncode:
      printWarning("ClientWriter Benchmark Process exited with code %u" % process.returncode)
    popWorkingPath() # build
    return process.returncode
  else:
    if not forBenchmark:
      buildScriptName = writeBuildNewClientLibraryScript(path, libraryLogicPath, forBenchmark, enableTileSelection)
      with ClientExecutionLock():
        process = subprocess.Popen(buildScriptName, cwd=path)
        process.communicate()
    else:
      runScriptName = writeRunScript(path, libraryLogicPath, forBenchmark, enableTileSelection)

      with ClientExecutionLock():
        process = subprocess.Popen(runScriptName, cwd=path)
        process.communicate()

    if process.returncode:
      printWarning("ClientWriter Benchmark Process exited with code %u" % process.returncode)
    popWorkingPath() # build
    return process.returncode

def getBuildOldClientScript(libraryLogicPath, forBenchmark):
  import io
  runScriptFile = io.StringIO()
  q = "" if os.name == "nt" else "\""
  echoLine = "@echo." if os.name == "nt" else "echo"
  runScriptFile.write("%s && echo %s%s%s && echo %s# Configuring CMake for Client%s && echo %s%s%s\n" \
      % (echoLine, q, HR, q, q, q, q, HR, q))
  runScriptFile.write("cmake")
  # runtime and kernel language
  runScriptFile.write(" -DTensile_RUNTIME_LANGUAGE=%s" % globalParameters["RuntimeLanguage"])
  runScriptFile.write(" -DTensile_CODE_OBJECT_VERSION=%s" % globalParameters["CodeObjectVersion"])
  runScriptFile.write(" -DTensile_COMPILER=%s" % globalParameters["CxxCompiler"])
  runScriptFile.write(" -DTensile_ARCHITECTURE=%s" % globalParameters["Architecture"])
  if globalParameters["EnableHalf"]:
    runScriptFile.write(" -DTensile_ENABLE_HALF=ON")
  if "ResumeBenchmarkProblem" in globalParameters and globalParameters["ResumeBenchmarkProblem"]:
    runScriptFile.write(" -DTensile_RESUME_BENCHMARK=ON")
  else:
    runScriptFile.write(" -DTensile_RESUME_BENCHMARK=OFF")
  if forBenchmark:
    # for benchmark client
    runScriptFile.write(" -DTensile_CLIENT_BENCHMARK=ON")
  else:
    # for library client
    runScriptFile.write(" -DTensile_ROOT=%s" % globalParameters["ScriptPath"] )
    runScriptFile.write(" -DTensile_CLIENT_BENCHMARK=OFF")
    runScriptFile.write(" -DTensile_LOGIC_PATH=%s" % libraryLogicPath)
    runScriptFile.write(" -DTensile_LIBRARY_PRINT_DEBUG=%s" \
        % ("ON" if globalParameters["LibraryPrintDebug"] else "OFF"))
    runScriptFile.write(" -DTensile_SHORT_FILE_NAMES=%s" \
        % ("ON" if globalParameters["ShortNames"] else "OFF"))
  if globalParameters["CMakeCXXFlags"]:
    runScriptFile.write("  -DCMAKE_CXX_FLAGS=%s" % globalParameters["CMakeCXXFlags"] )
  if globalParameters["CMakeCFlags"]:
    runScriptFile.write("  -DCMAKE_C_FLAGS=%s" % globalParameters["CMakeCFlags"] )
  if globalParameters["NewClient"] == 2:
    runScriptFile.write(" -DTENSILE_NEW_CLIENT=ON")
  else:
    runScriptFile.write(" -DTENSILE_NEW_CLIENT=OFF")
  runScriptFile.write("  -DCMAKE_BUILD_TYPE=%s" % (globalParameters["CMakeBuildType"]))
  # for both
  if os.name == "nt":
    runScriptFile.write(" -DCMAKE_GENERATOR_PLATFORM=x64")
  runScriptFile.write(" -DTensile_MERGE_FILES=%s" \
      % ("ON" if globalParameters["MergeFiles"] else "OFF"))
  runScriptFile.write(" ../source\n")
  runScriptFile.write("%s && echo %s%s%s && echo %s# Building Client%s && echo %s%s%s\n" \
      % (echoLine, q, HR, q, q, q, q, HR, q))
  runScriptFile.write("cmake --build . --config %s%s\n" \
      % (globalParameters["CMakeBuildType"], " -- -j 8" \
      if os.name != "nt" else "") )

  return runScriptFile.getvalue()

def getBuildNewClientLibraryScript(buildPath, libraryLogicPath, forBenchmark):
  import io
  runScriptFile = io.StringIO()

  callCreateLibraryCmd = globalParameters["ScriptPath"] + "/bin/TensileCreateLibrary"


  if globalParameters["MergeFiles"]:
    callCreateLibraryCmd += " --merge-files"
  else:
    callCreateLibraryCmd += " --no-merge-files"

  callCreateLibraryCmd += " --no-legacy-components"

  if globalParameters["ShortNames"]:
    callCreateLibraryCmd += " --short-file-names"
  else:
    callCreateLibraryCmd += " --no-short-file-names"

  if globalParameters["LibraryPrintDebug"]:
    callCreateLibraryCmd += " --library-print-debug"
  else:
    callCreateLibraryCmd += " --no-library-print-debug"

  callCreateLibraryCmd += " --architecture=" + globalParameters["Architecture"]
  callCreateLibraryCmd += " --code-object-version=" + globalParameters["CodeObjectVersion"]
  callCreateLibraryCmd += " --cxx-compiler=" + globalParameters["CxxCompiler"]

  callCreateLibraryCmd += " %s" % libraryLogicPath
  callCreateLibraryCmd += " %s" % buildPath #" ../source"
  callCreateLibraryCmd += " %s\n" % globalParameters["RuntimeLanguage"]

  runScriptFile.write(callCreateLibraryCmd)

  return runScriptFile.getvalue()

def writeBuildNewClientLibraryScript(path, libraryLogicPath, forBenchmark, enableTileSelection):
  filename = os.path.join(path, \
    "build.%s" % ("bat" if os.name == "nt" else "sh") )
  with open(filename, "w") as file:
    file.write("#!/bin/bash\n\n")
    file.write("set -ex\n")
    file.write(getBuildNewClientLibraryScript(path, libraryLogicPath, forBenchmark))

  if os.name != "nt":
    os.chmod(filename, 0o777)
  return filename

def writeBuildOldClientScript(path, libraryLogicPath, forBenchmark, enableTileSelection):
  filename = os.path.join(path, \
    "build.%s" % ("bat" if os.name == "nt" else "sh") )
  with open(filename, "w") as file:
    file.write("#!/bin/bash\n\n")
    file.write("set -ex\n")
    file.write(getBuildOldClientScript(libraryLogicPath, forBenchmark))

  if os.name != "nt":
    os.chmod(filename, 0o777)
  return filename


def writeRunScript(path, libraryLogicPath, forBenchmark, enableTileSelection):
  # create run.bat or run.sh which builds and runs
  runScriptName = os.path.join(path, \
    "run.%s" % ("bat" if os.name == "nt" else "sh") )
  runScriptFile = open(runScriptName, "w")
  echoLine = "@echo." if os.name == "nt" else "echo"
  if os.name != "nt":
    runScriptFile.write("#!/bin/bash\n\n")
  q = "" if os.name == "nt" else "\""

  runScriptFile.write("set -ex\n")


  if forBenchmark:
    if os.name == "nt":
      runScriptFile.write(os.path.join(globalParameters["CMakeBuildType"], \
          "client.exe") )
    else:
      if globalParameters["PinClocks"] and globalParameters["ROCmSMIPath"]:
        runScriptFile.write("%s -d 0 --setfan 255 --setsclk 7\n" % globalParameters["ROCmSMIPath"])
        runScriptFile.write("sleep 1\n")
        runScriptFile.write("%s -d 0 -a\n" % globalParameters["ROCmSMIPath"])

      runScriptFile.write("set +e\n")


    if globalParameters["DataInitTypeA"] == -1 :
        globalParameters["DataInitTypeA"] = globalParameters["DataInitTypeAB"]
    if globalParameters["DataInitTypeB"] == -1 :
        globalParameters["DataInitTypeB"] = globalParameters["DataInitTypeAB"]

    if globalParameters["NewClient"] < 2:
      runScriptFile.write("./client")
      clp = ""
      clp += " --platform-idx %u" % globalParameters["Platform"]
      clp += " --device-idx %u" % globalParameters["Device"]
      clp += " --init-alpha %u" % globalParameters["DataInitTypeAlpha"]
      clp += " --init-beta %u" % globalParameters["DataInitTypeBeta"]
      clp += " --init-d %u" % globalParameters["DataInitTypeD"]
      clp += " --init-c %u" % globalParameters["DataInitTypeC"]
      clp += " --init-a %u" % globalParameters["DataInitTypeA"]
      clp += " --init-b %u" % globalParameters["DataInitTypeB"]
      clp += " --c-equal-d %u" % globalParameters["CEqualD"]
      clp += " --print-valids %u" % globalParameters["ValidationPrintValids"]
      clp += " --print-max %u" % globalParameters["ValidationMaxToPrint"]
      clp += " --num-benchmarks %u" % globalParameters["NumBenchmarks"]
      clp += " --num-elements-to-validate %u" % globalParameters["NumElementsToValidate"]
      clp += " --num-enqueues-per-sync %u" % globalParameters["EnqueuesPerSync"]
      clp += " --num-syncs-per-benchmark %u" % globalParameters["SyncsPerBenchmark"]
      clp += " --use-gpu-timer %u" % globalParameters["KernelTime"]
      clp += " --sleep-percent %u" % globalParameters["SleepPercent"]
      clp += " --benchmark-solutions %u" % enableTileSelection
      if "ClientArgs" in globalParameters:
        clientParams = globalParameters["ClientArgs"]
        if clientParams:
          clp += " " + globalParameters["ClientArgs"]
      runScriptFile.write(clp)
      runScriptFile.write("\n")
      runScriptFile.write("ERR1=$?\n")
    else:
      runScriptFile.write("ERR1=0\n")

    if globalParameters["NewClient"]:
      newClientExe = ClientExecutable.getClientExecutable()
      configFile = os.path.join(globalParameters['WorkingPath'], '../source/ClientParameters.ini')
      runScriptFile.write("{} --config-file {} {}\n".format(newClientExe, configFile, globalParameters["NewClientArgs"]))
      if enableTileSelection and (globalParameters["NewClient"] == 2):
        configFileGranularity = os.path.join(globalParameters['WorkingPath'], '../source/ClientParameters_Granularity.ini')
        runScriptFile.write("{} --config-file {} {}\n".format(newClientExe, configFileGranularity, globalParameters["NewClientArgs"]))
      runScriptFile.write("ERR2=$?\n\n")
    else:
      runScriptFile.write("ERR2=0\n")

    runScriptFile.write("""
ERR=0
if [[ $ERR1 -ne 0 ]]
then
    echo one
    ERR=$ERR1
fi
if [[ $ERR2 -ne 0 ]]
then
    echo two
    ERR=$ERR2
fi
""")

    if os.name != "nt":
      if globalParameters["PinClocks"] and globalParameters["ROCmSMIPath"]:
        runScriptFile.write("%s -d 0 --resetclocks\n" % globalParameters["ROCmSMIPath"])
        runScriptFile.write("%s -d 0 --setfan 50\n" % globalParameters["ROCmSMIPath"])
  else:
    executablePath = os.path.join(globalParameters["WorkingPath"])
    if os.name == "nt":
      executablePath = os.path.join(executablePath, \
          globalParameters["CMakeBuildType"], \
          "client.exe")
    else:
      executablePath = os.path.join(executablePath, "client")
    runScriptFile.write("%s && echo %s%s%s && echo %s# Library Client:%s && echo %s# %s%s && %s\n" \
        % (echoLine, q, HR, q, q, q, q, executablePath, q, executablePath) )
  if os.name != "nt":
    runScriptFile.write("exit $ERR\n")
  runScriptFile.close()
  if os.name != "nt":
    os.chmod(runScriptName, 0o777)
  return runScriptName


def toCppBool(yamlBool):
  return "true" if yamlBool else "false"

def getMaxSolutionSizes(solutions, solutionSummationSizes):

  maxK = max(solutionSummationSizes)
  maxMT0 = 0
  maxMT1 = 0
  for solution in solutions:

    wg = solution["WorkGroup"]
    tt = solution["ThreadTile"]
    mt0 = wg[0] * tt[0]
    mt1 = wg[1] * tt[1]

    if (mt0 > maxMT0):
      maxMT0 = mt0

    if (mt1 > maxMT1):
      maxMT1 = mt1

  return [maxMT0, maxMT1, maxK]

def checkConstStride(constStrideMap, keyIdx):
  finalVal = None
  for (mapIdx, val) in constStrideMap:
    if keyIdx == mapIdx:
      finalVal = val
  #print ("idx=", keyIdx, "=", finalVal)
  return finalVal

def problemSizeParams(solution, problem):

    numIndices = len(solution.problemType.indices)
    rv = []

    if problem.stridesA:
        astrides = list(problem.stridesA)
    else:
        astrides = [-1] * solution.problemType.aDims
    for sc in solution.problemType.setConstStrideA:
        index = solution.problemType.indices[sc[0]]
        if type(index) == FreeIndex:
            assert(index.isA)
            astrides[index.i] = sc[1]
        else:
            astrides[index.a] = sc[1]

    if problem.stridesB:
      bstrides = list(problem.stridesB)
    else:
      bstrides = [-1] * solution.problemType.bDims
    for sc in solution.problemType.setConstStrideB:
        index = solution.problemType.indices[sc[0]]
        if type(index) == FreeIndex:
            assert(not index.isA)
            bstrides[index.i] = sc[1]
        else:
            bstrides[index.b] = sc[1]


    cstrides = problem.stridesC
    dstrides = problem.stridesD
    if len(problem.sizes) == numIndices:
        None
    elif len(problem.sizes) == numIndices + 4:
        # FIXME-problem, this is Exact format with strides tacked onto sizes as 4 extra pams
        # should just set problem.stride* appropriately when reading the Yaml and not deal with extra fields here
        if astrides[1] == -1:
          astrides[1] = problem.sizes[numIndices+2]
        else:
          raise RuntimeError("problem-specified lda(%u) conflicts with setConstStrideA(%u)" % \
              (astrides[1], problem.sizes[numIndices+2]))

        if bstrides[1] == -1:
          bstrides[1] = problem.sizes[numIndices+3]
        else:
          raise RuntimeError("problem-specified ldb(%u) conflicts with setConstStrideB(%u)" % \
              (bstrides[1], problem.sizes[numIndices+3]))

        cstrides = (-1, problem.sizes[numIndices+1])
        dstrides = (-1, problem.sizes[numIndices+0])
    else:
        raise RuntimeError(
            "Invalid number of problem type indices: {0} - Indices: {1}, problemSize: {2}".format(len(problem.sizes), numIndices,
            ', '.join(map(str, problem.sizes))))

    problemSizeArg = ('problem-size', ','.join(map(str, problem.sizes[:numIndices])))
    rv.insert(0, problemSizeArg)

    rv.append(('a-strides', ",".join(map(str, astrides))))
    rv.append(('b-strides', ",".join(map(str, bstrides))))
    if cstrides:
      rv.append(('c-strides', ",".join(map(str, cstrides))))
    if dstrides:
      rv.append(('d-strides', ",".join(map(str, dstrides))))

    if problem.zeroPadA:
        rv.append(('a-zero-pads', ';'.join([','.join(map(str,zp)) for zp in problem.zeroPadA])))
    if problem.zeroPadB:
        rv.append(('b-zero-pads', ';'.join([','.join(map(str,zp)) for zp in problem.zeroPadB])))

    return rv


def dataInitName(num):
    if num == 0: return 'Zero'
    if num == 1: return 'One'
    if num == 2: return 'Two'
    if num == 3: return 'Random'
    if num == 4: return 'NaN'
    if num == 5: return 'Inf'
    if num == 6: return 'BadInput'
    if num == 7: return 'BadOutput'
    if num == 8: return 'SerialIdx'
    if num == 9: return 'SerialDim0'
    if num == 10: return 'SerialDim1'
    if num == 11: return 'Identity'

def dataInitParams(problemType):
    initA = globalParameters['DataInitTypeA']
    initB = globalParameters['DataInitTypeB']
    initC = globalParameters['DataInitTypeC']
    initD = globalParameters['DataInitTypeD']
    initAlpha = globalParameters['DataInitTypeAlpha']
    initBeta  = globalParameters['DataInitTypeBeta']
    # import pdb
    # pdb.set_trace()

    if not problemType.useBeta:
        initBeta = 0

    if initA == -1: initA = globalParameters['DataInitTypeAB']
    if initB == -1: initB = globalParameters['DataInitTypeAB']

    return [('init-a',     dataInitName(initA)),
            ('init-b',     dataInitName(initB)),
            ('init-c',     dataInitName(initC)),
            ('init-d',     dataInitName(initD)),
            ('init-alpha', dataInitName(initAlpha)),
            ('init-beta',  dataInitName(initBeta))]

def writeClientConfig(forBenchmark, solutions, problemSizes, stepName, stepBaseDir, newLibrary, codeObjectFiles, tileAwareSelection = False):

    if tileAwareSelection:
      filename = os.path.join(globalParameters["WorkingPath"], "ClientParameters_Granularity.ini")
    else:
      filename = os.path.join(globalParameters["WorkingPath"], "ClientParameters.ini")


    if len(newLibrary.solutions)==0:
      raise RuntimeError ("No valid solutions found")
    with open(filename, "w") as f:
        def param(key, value):
            f.write("{}={}\n".format(key, value))

        sourceDir = os.path.join(stepBaseDir, "source")
        libraryFile = os.path.join(sourceDir, "library", "TensileLibrary.yaml")
        param("library-file", libraryFile)

        currentGFXName = "gfx%x%x%x" % globalParameters["CurrentISA"]
        for coFile in codeObjectFiles:
            if (currentGFXName in coFile):
                param("code-object", os.path.join(sourceDir,coFile))

        if tileAwareSelection:
          param('results-file', os.path.join(stepBaseDir, "../Data", stepName+"_Granularity.csv"))
        else:
          if globalParameters["NewClient"] == 1:
            param('results-file', os.path.join(stepBaseDir, "../Data", stepName+"-new.csv"))
          else:
            param('results-file', os.path.join(stepBaseDir, "../Data", stepName+".csv"))

        newSolution = next(iter(newLibrary.solutions.values()))
        if newSolution.problemType.convolution and globalParameters["ConvolutionVsContraction"]:
            param('convolution-identifier', newSolution.problemType.convolution.identifier())
        param('problem-identifier', newSolution.problemType.operationIdentifier)
        param('a-type',     newSolution.problemType.aType.toEnum())
        param('b-type',     newSolution.problemType.bType.toEnum())
        param('c-type',     newSolution.problemType.cType.toEnum())
        param('d-type',     newSolution.problemType.dType.toEnum())
        param('alpha-type', newSolution.problemType.alphaType.toEnum())
        param('beta-type',  newSolution.problemType.betaType.toEnum())

        param('high-precision-accumulate',  newSolution.problemType.highPrecisionAccumulate)

        for problem in problemSizes.problems:
            for key,value in problemSizeParams(newSolution, problem):
                param(key,value)
            #param('problem-size', ','.join(map(str,problemSize)))

        param("device-idx",               globalParameters["Device"])

        for key,value in dataInitParams(newSolution.problemType):
            param(key, value)

        param("c-equal-d",                globalParameters["CEqualD"])

        if globalParameters["PrintTensorA"]:
          param("print-tensor-a",         1)
        if globalParameters["PrintTensorB"]:
          param("print-tensor-b",         1)
        if globalParameters["PrintTensorC"]:
          param("print-tensor-c",         1)
        if globalParameters["PrintTensorD"]:
          param("print-tensor-d",         1)
        if globalParameters["PrintTensorRef"]:
          param("print-tensor-ref",         1)

        if globalParameters["BoundsCheck"]:
          param("bounds-check", 1)

        param("print-valids",             globalParameters["ValidationPrintValids"])
        param("print-max",                globalParameters["ValidationMaxToPrint"])
        param("num-benchmarks",           globalParameters["NumBenchmarks"])
        param("num-elements-to-validate", globalParameters["NumElementsToValidate"])
        param("num-enqueues-per-sync",    globalParameters["EnqueuesPerSync"])
        param("num-syncs-per-benchmark",  globalParameters["SyncsPerBenchmark"])
        param("use-gpu-timer",            globalParameters["KernelTime"])
        if globalParameters["ConvolutionVsContraction"]:
            assert(newSolution.problemType.convolution)
            param("convolution-vs-contraction", globalParameters["ConvolutionVsContraction"])
        if not globalParameters["KernelTime"]:
            param("num-warmups", 1)
        param("sleep-percent",            globalParameters["SleepPercent"])
        param("perf-l2-read-hits",        globalParameters["PerfModelL2ReadHits"])
        param("perf-l2-write-hits",       globalParameters["PerfModelL2WriteHits"])
        param("perf-l2-read-bw-mul",      globalParameters["PerfModelL2ReadBwMul"])
        param("perf-read-efficiency",     globalParameters["PerfModelReadEfficiency"])



################################################################################
# Write Generated Benchmark Parameters
################################################################################
def writeClientParameters(forBenchmark, solutions, problemSizes, stepName, \
    functionList, stepBaseDir, solutionSummationSizes, solutionWriter = None):
  h = ""

  ##############################################################################
  # Min Naming
  ##############################################################################
  """
  if forBenchmark:
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
  """

  if forBenchmark:
    if globalParameters["MergeFiles"]:
      h += "#include \"Solutions.h\"\n"
    else:
      for solution in solutions:
        solutionName = solutionWriter.getSolutionName(solution)
        h += "#include \"" + solutionName + ".h\"\n"
        h += "#include \"Solutions.h\"\n"
    h += "#include \"ReferenceCPU.h\"\n"
    h += "\n"
  else:
    h += "#include \"Solutions.h\"\n"
    h += "#include \"Tensile.h\"\n"


  h += "typedef enum {\n"
  h += "    enum_float,\n"
  h += "    enum_double,\n"
  h += "    enum_TensileComplexFloat,\n"
  h += "    enum_TensileComplexDouble\n"
  h += "#ifdef Tensile_ENABLE_HALF\n"
  h += "    ,enum_TensileHalf\n"
  h += "#endif\n"
  h += "    ,enum_TensileInt8x4\n"
  h += "    ,enum_TensileInt32\n"
  h += "    ,enum_tensile_bfloat16\n"
  h += "} DataTypeEnum;\n"
  h += "\n"

  h += "// Debug Params\n"
  h += "const unsigned printTensorA=%x;\n" % int(globalParameters["PrintTensorA"])
  h += "const unsigned printTensorB=%x;\n" % int(globalParameters["PrintTensorB"])
  h += "const unsigned printTensorC=%x;\n" % int(globalParameters["PrintTensorC"])
  h += "const unsigned printTensorD=%x;\n" % int(globalParameters["PrintTensorD"])

  h += "const bool printWinnersOnly=%s;\n" % toCppBool(globalParameters["PrintWinnersOnly"])
  h += "\n"

  h += "const char indexChars[%u] = \"%s" \
      % (len(globalParameters["IndexChars"])+1, \
      globalParameters["IndexChars"][0])
  for i in range(1, len(globalParameters["IndexChars"])):
    h += globalParameters["IndexChars"][i]
  h += "\";\n"

  h += "unsigned int functionIdx;\n"
  h += "unsigned int dataTypeIdx;\n"
  h += "unsigned int problemTypeIdx;\n"
  h += "\n"

  ##############################################################################
  # Problem Types
  ##############################################################################
  #dataTypes = []
  #problemTypes = []
  #functionSerialToDataTypeAndIdx = []
  dataTypes = []
  problemTypes = []
  destDataTypes = {}
  computeDataTypes = {}
  problemTypesForDataType = {} # for data type
  schedulesForProblemType = {} # for problem type
  functionInfo = [] # dataTypeIdx, problemTypeIdx, idxWithinDataType, idxWithinProblemType
  #tileSelection = False

  if forBenchmark:
    problemType = solutions[0]["ProblemType"]
    dataType = problemType["DataType"]
    #tileSelection = problemType["TileAwareSelection"]

    destDataType = problemType["DestDataType"]
    destDataTypes[dataType] = destDataType

    computeDataType = problemType["ComputeDataType"]
    computeDataTypes[dataType] = computeDataType

    dataTypes.append(dataType)

    problemTypes.append(problemType)
    problemTypesForDataType[dataType] = [problemType]
    schedulesForProblemType[problemType] = solutions
    numProblemTypes = 1
    for solution in solutions:
      functionInfo.append([ 0, 0, 0, 0, 0, 0 ])
  else:
    for functionIdx in range(0, len(functionList)):
      function = functionList[functionIdx]
      scheduleName = function[0]
      problemType = function[1]
      dataType = problemType["DataType"]
      destDataType = problemType["DestDataType"]
      computeDataType = problemType["ComputeDataType"]
      if dataType not in dataTypes:
        dataTypes.append(dataType)
        destDataTypes[dataType] = destDataType
        computeDataTypes[dataType] = computeDataType
        problemTypesForDataType[dataType] = []
      if problemType not in problemTypesForDataType[dataType]:
        problemTypesForDataType[dataType].append(problemType)
        schedulesForProblemType[problemType] = []
      schedulesForProblemType[problemType].append(scheduleName)

    # sort
    dataTypes = sorted(dataTypes)
    for dataType in dataTypes:
      problemTypesForDataType[dataType] = \
          sorted(problemTypesForDataType[dataType],key=str)
      for problemType in problemTypesForDataType[dataType]:
        schedulesForProblemType[problemType] = \
            sorted(schedulesForProblemType[problemType],key=str)

    # assign info
    functionIdxSerial = 0
    problemTypeIdxSerial = 0
    for dataTypeIdxSerial in range(0, len(dataTypes)):
      dataType = dataTypes[dataTypeIdxSerial]
      functionIdxForDataType = 0
      for problemTypeIdxForDataType in range(0, \
          len(problemTypesForDataType[dataType])):
        problemType = \
            problemTypesForDataType[dataType][problemTypeIdxForDataType]
        problemTypes.append(problemType)
        functionIdxForProblemType = 0
        for functionIdxForProblemType in range(0, \
            len(schedulesForProblemType[problemType])):
          functionInfo.append([ \
              dataTypeIdxSerial, \
              problemTypeIdxForDataType, \
              problemTypeIdxSerial, \
              functionIdxSerial,\
              functionIdxForDataType,\
              functionIdxForProblemType, \
              ])
          functionIdxForProblemType += 1
          functionIdxForDataType += 1
          functionIdxSerial += 1
        problemTypeIdxSerial += 1
    numProblemTypes = problemTypeIdxSerial
    numFunctions = functionIdxSerial
    h += "const unsigned int numFunctions = %u;\n" % numFunctions

  ##############################################################################
  # Data Types
  ##############################################################################
  h += "/* data types */\n"
  numDataTypes = len(dataTypes)
  h += "const unsigned int numDataTypes = %u;\n" % numDataTypes
  h += "const DataTypeEnum dataTypeEnums[numDataTypes] = { enum_%s" \
      % dataTypes[0].toCpp()
  for dataTypeIdx in range(1, numDataTypes):
    h += ", enum_%s" % dataTypes[dataTypeIdx].toCpp()
  h += " };\n"
  # bytes per elements
  h += "const unsigned int bytesPerElement[numDataTypes] = { %u" \
      % (dataTypes[0].numBytes())
  for dataTypeIdx in range(1, numDataTypes):
    dataType = dataTypes[dataTypeIdx]
    h += ", %u" % dataType.numBytes()
  h += " };\n"
  # flops per mac
  if dataTypes[0].isInt8x4():
    h += "const unsigned int numFlopsPerMac[numDataTypes] = { %u" % (8 if dataTypes[0].isReal() else 32)
  else:
    h += "const unsigned int numFlopsPerMac[numDataTypes] = { %u" % (2 if dataTypes[0].isReal() else 8)
  for dataTypeIdx in range(1, numDataTypes):
    dataType = dataTypes[dataTypeIdx]
    h += ", %u" % (2 if dataType.isReal() else 8)
  h += " };\n"
  for dataTypeIdx in range(0, numDataTypes):
    h += "#define Tensile_DATA_TYPE_%s\n" \
        % dataTypes[dataTypeIdx].toCpp().upper()

  ##############################################################################
  # Problem Types
  ##############################################################################
  h += "/* problem types */\n"
  h += "const unsigned int numProblemTypes = %u;\n" % numProblemTypes
  # Num C Indices
  h += "const unsigned int numIndicesC[numProblemTypes] = { %u" \
      % problemTypes[0]["NumIndicesC"]
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    h += ", %u" % problemType["NumIndicesC"]
  h += " };\n"

  # Num AB Indices
  maxNumIndicesA = len(problemTypes[0]["IndexAssignmentsA"])
  maxNumIndicesB = len(problemTypes[0]["IndexAssignmentsB"])
  h += "const unsigned int numIndicesA[numProblemTypes] = { %u" \
      % len(problemTypes[0]["IndexAssignmentsA"])
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    numIndicesA = len(problemType["IndexAssignmentsA"])
    h += ", %u" % numIndicesA
    maxNumIndicesA = max(numIndicesA, maxNumIndicesA)
  h += " };\n"
  h += "const unsigned int maxNumIndicesA = %u;\n" % maxNumIndicesA

  h += "const unsigned int numIndicesB[numProblemTypes] = { %u" \
      % len(problemTypes[0]["IndexAssignmentsB"])
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    numIndicesB = len(problemType["IndexAssignmentsB"])
    h += ", %u" % numIndicesB
    maxNumIndicesB = max(numIndicesB, maxNumIndicesB)
  h += " };\n"
  h += "const unsigned int maxNumIndicesB = %u;\n" % maxNumIndicesB

  # Index Assignments A
  h += "const unsigned int indexAssignmentsA[numProblemTypes][maxNumIndicesA] = {\n"
  for problemTypeIdx in range(0, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    indices = problemType["IndexAssignmentsA"]
    h += "  { %u" % indices[0]
    for i in range(1, maxNumIndicesA):
      if i < len(indices):
        h += ", %u" % indices[i]
      else:
        h += ", static_cast<unsigned int>(-1)"
    if problemTypeIdx < numProblemTypes-1:
      h += " },\n"
    else:
      h += " }\n"
  h += "};\n"
  # Index Assignments B
  h += "const unsigned int indexAssignmentsB[numProblemTypes][maxNumIndicesB] = {\n"
  for problemTypeIdx in range(0, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    indices = problemType["IndexAssignmentsB"]
    h += "  { %u" % indices[0]
    for i in range(1, maxNumIndicesB):
      if i < len(indices):
        h += ", %u" % indices[i]
      else:
        h += ", static_cast<unsigned int>(-1)"
    if problemTypeIdx < numProblemTypes-1:
      h += " },\n"
    else:
      h += " }\n"
  h += "};\n"
  # Index Assignments LD
  h += "const unsigned int numIndicesLD = %u;\n" % problemType["NumIndicesLD"]
  h += "const unsigned int indexAssignmentsLD[numIndicesLD] = {"
  if problemType["NumIndicesLD"] > 0:
    h += " %u" % problemType["IndexAssignmentsLD"][0]
    for ldIdx in range(1, len(problemType["IndexAssignmentsLD"])):
      h += ", %u" % problemType["IndexAssignmentsLD"][ldIdx]
  h += "};\n"
  # beta
  h += "bool useBeta[numProblemTypes] = { %s" \
      % ("true" if problemTypes[0]["UseBeta"] else "false")
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    h += ", %s" % ("true" if problemType["UseBeta"] else "false")
  h += " };\n"
  # Complex Conjugates
  h += "const bool complexConjugateA[numProblemTypes] = { %s" \
      % ("true" if problemTypes[0]["ComplexConjugateA"] else "false" )
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    h += ", %s" % ("true" if problemTypes[0]["ComplexConjugateA"] else "false" )
  h += " };\n"
  h += "const bool complexConjugateB[numProblemTypes] = { %s" \
      % ("true" if problemTypes[0]["ComplexConjugateB"] else "false" )
  for problemTypeIdx in range(1, numProblemTypes):
    problemType = problemTypes[problemTypeIdx]
    h += ", %s" % ("true" if problemTypes[0]["ComplexConjugateB"] else "false" )
  h += " };\n"
  h += "\n"

  if not forBenchmark:
    h += "// dataTypeIdxSerial, problemTypeIdxForDataType, problemTypeIdxSerial, functionIdxSerial, functionIdxForDataType, functionIdxForProblemType\n"
    first = True
    h += "const unsigned int functionInfo[numFunctions][6] = {\n"
    for info in functionInfo:
      h += "%s{ %u, %u, %u, %u, %u, %u }" % ("  " if first else ",\n  ", \
          info[0], info[1], info[2], info[3], info[4], info[5] )
      first = False
    h += " };\n"


  ##############################################################################
  # Problem Sizes
  ##############################################################################
  maxNumIndices = problemTypes[0]["TotalIndices"]
  if not forBenchmark:
    for problemType in problemTypes:
      maxNumIndices = max(problemType["TotalIndices"], maxNumIndices)
  h += "const unsigned int maxNumIndices = %u;\n" % maxNumIndices
  h += "const unsigned int totalIndices[numProblemTypes] = { %u" \
      % problemTypes[0]["TotalIndices"]
  for problemTypeIdx in range(1, numProblemTypes):
      h += ", %u" % problemTypes[problemTypeIdx]["TotalIndices"]
  h += " };\n"
  if forBenchmark:
    h += "const unsigned int numProblems = %u;\n" \
        % problemSizes.totalProblemSizes
    h += "const unsigned int problemSizes[numProblems][%u] = {\n" \
        % (problemTypes[0]["TotalIndices"] + problemType["NumIndicesLD"])
    for i in range(problemSizes.totalProblemSizes):
      #assert problemSizes.problems[i].stridesA == None # new stride functionality only supported on new client, not here
      problemSize = problemSizes.problems[i].sizes
      line = "  {%5u" %problemSize[0]
      for j in range(1, problemTypes[0]["TotalIndices"] + problemType["NumIndicesLD"]):
        line += ",%5u" % problemSize[j]
      line += " }"
      h += line
      if i < problemSizes.totalProblemSizes-1:
        h += ","
      else:
        h += ""
    h += "};\n"
    h += "const unsigned int minStrides[%u] = {" \
        % problemTypes[0]["TotalIndices"]
    for i in range(0, len(problemSizes.minStrides)):
      if (i!=0):
        h += ", "
      h += str(problemSizes.minStrides[i])
    h += "};\n"
  else:
    h += "unsigned int userSizes[maxNumIndices];\n"
    h += "unsigned int minStrides[%u] = {" \
        % maxNumIndices
    for i in range(0, maxNumIndices):
      if (i!=0):
        h += ", "
      h += str(0); # always use 0 for minStrides in benchmark mode
    h += "};\n"

  if forBenchmark:
    h += "/* problem sizes */\n"
    """
    h += "const bool indexIsSized[maxNumIndices] = {"
    for i in range(0, problemSizes.totalIndices):
      h += " %s" % ("true" if problemSizes.indexIsSized[i] else "false")
      if i < problemSizes.totalIndices-1:
        h += ","
    h += " };\n"

    h += "const unsigned int numIndicesSized = %u;\n" \
        % len(problemSizes.indicesSized)
    h += "const unsigned int indicesSized[numIndicesSized][4] = {\n"
    h += "// { min, stride, stride_incr, max }\n"
    for i in range(0, len(problemSizes.indicesSized)):
      r = problemSizes.indicesSized[i]
      h += "  { %u, %u, %u, %u }" % (r[0], r[1], r[2], r[3])
      if i < len(problemSizes.indicesSized)-1:
        h += ","
      h += "\n"
    h += "  };\n"

    numIndicesMapped = len(problemSizes.indicesMapped)
    h += "const unsigned int numIndicesMapped = %u;\n" % numIndicesMapped
    if numIndicesMapped > 0:
      h += "#define Tensile_INDICES_MAPPED 1\n"
      h += "const unsigned int indicesMapped[numIndicesMapped] = {"
      for i in range(0, numIndicesMapped):
        h += " %u" % problemSizes.indicesMapped[i]
        if i < numIndicesMapped-1:
          h += ","
      h += " };\n"
    else:
      h += "#define Tensile_INDICES_MAPPED 0\n"
    """

  ##############################################################################
  # Max Problem Sizes
  ##############################################################################
  if forBenchmark:
    maximumD = problemSizes.maxD
    maximumC = problemSizes.maxC
    maximumA = problemSizes.maxA
    maximumB = problemSizes.maxB

    maxMT = getMaxSolutionSizes(solutions, solutionSummationSizes)

    maxMN = 1296 * maxMT[0] * maxMT[1]
    maxMK = 36 * maxMT[0] * maxMT[2]
    maxNK = 36 * maxMT[1] * maxMT[2]

    maximumA = max(maximumA, maxMK)
    maximumB = max(maximumB, maxNK)
    maximumC = max(maximumC, maxMN)
    maximumD = max(maximumD, maxMN)

    h += "size_t maxSizeD = %u;\n" % (maximumD)
    h += "size_t maxSizeC = %u;\n" % (maximumC)
    h += "size_t maxSizeA = %u;\n" % (maximumA)
    h += "size_t maxSizeB = %u;\n" % (maximumB)
    h += "\n"
  else:
    h += "size_t maxSizeD;\n"
    h += "size_t maxSizeC;\n"
    h += "size_t maxSizeA;\n"
    h += "size_t maxSizeB;\n"
    h += "\n"

  ##############################################################################
  # Current Problem Size
  ##############################################################################
  h += "/* current problem size */\n"
  #h += "unsigned int fullSizes[maxNumIndices];\n"
    #h += "unsigned int currentSizedIndexSizes[numIndicesSized];\n"
    #h += "unsigned int currentSizedIndexIncrements[numIndicesSized];\n"
  h += "\n"

  ##############################################################################
  # Solutions
  ##############################################################################
  if forBenchmark:
    # Solution Ptrs
    h += "/* solutions */\n"
    # Problem Type Indices
    h += "const unsigned int maxNumSolutions = %u;\n" % len(solutions)
    h += "float solutionPerf[numProblems][maxNumSolutions]; // milliseconds\n"
    h += "\n"

    h += "static const SolutionInfo solutions[maxNumSolutions] = {\n"
    for i in range(0, len(solutions)):
      solution = solutions[i]
      solutionName = solutionWriter.getSolutionName(solution)
      h += "  {(void*)%s, \"%s\", {%d, %d, %d, %d, %s, %d, %d} }" % \
        (solutionName, solutionName,
          solution["AssertSummationElementMultiple"],
          solution["AssertFree0ElementMultiple"],
          solution["AssertFree1ElementMultiple"],
          solution["AssertMinApproxSize"],
          "true" if solution["LdcEqualsLdd"] else "false",
          solution["PackBatchDims"]==2, \
          solution["PackBatchDims"]==1, \
          )
      if i < len(solutions)-1:
        h += ","
      h += "\n"
    h += " };\n"
    h += "\n"

    numSummations = len(solutionSummationSizes)
    h += "const unsigned int numSummations = %d;\n" % (numSummations)

    h += "const unsigned int summations[numSummations] = {%d" % (solutionSummationSizes[0])
    for i in range(1, numSummations):
      h += ", %d" % (solutionSummationSizes[i])
    h += "};\n"

  ##############################################################################
  # Solution meta data
  ##############################################################################

    transA = solutions[0]["ProblemType"]["TransposeA"]
    transB = solutions[0]["ProblemType"]["TransposeB"]
    h += "const unsigned int solutionMetaData[maxNumSolutions][10] = {\n"
    for i in range(0, len(solutions)):
      solution = solutions[i]

      wg = solution["WorkGroup"]
      tt = solution["ThreadTile"]
      mt0 = wg[0] * tt[0]
      mt1 = wg[1] * tt[1]
      gsu = solution["GlobalSplitU"]
      lsu = wg[2]

      h += "  {%d, %d, %d, %d, %d, %d, %d, %d, %d, %d}" % (mt0,mt1,tt[0],tt[1],wg[0],wg[1],transA,transB,gsu,lsu)

      if (i < len(solutions) - 1):
        h += ",\n"
      else:
        h += "\n"
    h += " };\n"
    h += "\n"



  else:
    # Function Names
    functionNames = []
    for dataType in dataTypes:
      for problemType in problemTypesForDataType[dataType]:
        # example scheduleName is fiji, vega10, etc
        for scheduleName in schedulesForProblemType[problemType]:
          functionNames.append("tensile_%s" % (problemType))
    h += "const char *functionNames[numFunctions] = {\n"
    for functionIdx in range(0, len(functionNames)):
      functionName = functionNames[functionIdx]
      h += "    \"%s\"%s\n" % (functionName, \
          "," if functionIdx < len(functionNames)-1 else "" )
    h += " };\n"

  ##############################################################################
  # Runtime Structures
  ##############################################################################
  h += "/* runtime structures */\n"
  h += "TensileStatus status;\n"
  if globalParameters["RuntimeLanguage"] == "OCL":
    h += "cl_platform_id platform;\n"
    h += "cl_device_id device;\n"
    h += "cl_context context;\n"
    h += "cl_command_queue stream;\n"
  else:
    h += "hipStream_t stream;\n"
    #h += "int deviceIdx = %u;\n" \
    #    % (globalParameters["Device"])
  h += "\n"
  h += "void *deviceD;\n"
  h += "void *deviceC;\n"
  h += "void *deviceA;\n"
  h += "void *deviceB;\n"

  ##############################################################################
  # Benchmarking and Validation Parameters
  ##############################################################################
  h += "\n/* benchmarking parameters */\n"
  #h += "const bool measureKernelTime = %s;\n" \
  #    % ("true" if globalParameters["KernelTime"] else "false")
  #h += "const unsigned int numEnqueuesPerSync = %u;\n" \
  #    % (globalParameters["EnqueuesPerSync"])
  #h += "const unsigned int numSyncsPerBenchmark = %u;\n" \
  #    % (globalParameters["SyncsPerBenchmark"])
  #h += "unsigned int numElementsToValidate = %s;\n" \
  #    % (str(globalParameters["NumElementsToValidate"]) \
  #    if globalParameters["NumElementsToValidate"] >= 0 \
  #    else "0xFFFFFFFF" )
  #h += "unsigned int validationMaxToPrint = %u;\n" \
  #    % globalParameters["ValidationMaxToPrint"]
  #h += "bool validationPrintValids = %s;\n" \
  #    % ("true" if globalParameters["ValidationPrintValids"] else "false")
  h += "size_t validationStride;\n"
  if problemType["HighPrecisionAccumulate"]:
    h += "static bool useHighPrecisionAccumulate = true;\n"
  else:
    h += "static bool useHighPrecisionAccumulate = false;\n"
  #h += "unsigned int dataInitTypeC = %s;\n" % globalParameters["DataInitTypeC"]
  #h += "unsigned int dataInitTypeAB = %s;\n" % globalParameters["DataInitTypeAB"]
  h += "\n"

  ##############################################################################
  # Generated Call to Reference
  ##############################################################################
  h += "/* generated call to reference */\n"
  h += "template<typename DataType, typename DestDataType, typename ComputeDataType>\n"
  h += "TensileStatus generatedCallToReferenceCPU(\n"
  h += "    const unsigned int *sizes,\n"
  h += "    const unsigned int *minStrides,\n"
  h += "    DestDataType *referenceD,\n"
  h += "    DestDataType *referenceC,\n"
  h += "    DataType *initialA,\n"
  h += "    DataType *initialB,\n"
  h += "    const unsigned int lda,\n"
  h += "    const unsigned int ldb,\n"
  h += "    const unsigned int ldc,\n"
  h += "    const unsigned int ldd,\n"
  h += "    const unsigned int stride_a,\n"
  h += "    const unsigned int stride_b,\n"
  h += "    const unsigned int stride_c,\n"
  h += "    const unsigned int stride_d,\n"
  h += "    ComputeDataType alpha,\n"
  h += "    ComputeDataType beta,\n"
  h += "    bool useHighPrecisionAccumulate) {\n"
  h += "  return tensileReferenceCPU(\n"
  h += "      referenceD,\n"
  h += "      referenceC,\n"
  h += "      initialA,\n"
  h += "      initialB,\n"
  h += "      lda,\n"
  h += "      ldb,\n"
  h += "      ldc,\n"
  h += "      ldd,\n"
  h += "      stride_a,\n"
  h += "      stride_b,\n"
  h += "      stride_c,\n"
  h += "      stride_d,\n"
  h += "      alpha,\n"
  h += "      beta,\n"
  h += "      totalIndices[problemTypeIdx],\n"
  h += "      sizes,\n"
  h += "      minStrides,\n"
  h += "      numIndicesC[problemTypeIdx],\n"
  h += "      numIndicesA[problemTypeIdx],\n"
  h += "      numIndicesB[problemTypeIdx],\n"
  h += "      indexAssignmentsA[problemTypeIdx],\n"
  h += "      indexAssignmentsB[problemTypeIdx],\n"
  h += "      complexConjugateA[problemTypeIdx],\n"
  h += "      complexConjugateB[problemTypeIdx],\n"
  h += "      validationStride,\n"
  h += "      useHighPrecisionAccumulate);\n"
  h += "};\n"
  h += "\n"

  ##############################################################################
  # Generated Call to Solution
  ##############################################################################
  if forBenchmark:
    problemType = solutions[0]["ProblemType"]
    h += "/* generated call to solution */\n"
    h += "template<typename DataType, class SolutionInfoType>\n"
    h += "TensileStatus generatedCallToSolution(\n"
    h += "    const SolutionInfoType &solution,\n"
    h += "    SolutionLock *solutionLock,\n"
    h += "    const unsigned int *sizes,\n"
    h += "    const unsigned int *minStrides,\n"
    h += "    const unsigned int lda,\n"
    h += "    const unsigned int ldb,\n"
    h += "    const unsigned int ldc,\n"
    h += "    const unsigned int ldd,\n"
    h += "    const unsigned int stride_a,\n"
    h += "    const unsigned int stride_b,\n"
    h += "    const unsigned int stride_c,\n"
    h += "    const unsigned int stride_d,\n"
    h += "    DataType alpha,\n"
    h += "    DataType beta,\n"
    h += "    unsigned int numEvents = 0,\n"
    if globalParameters["RuntimeLanguage"] == "OCL":
      h += "    cl_event *event_wait_list = NULL,\n"
      h += "    cl_event *outputEvent = NULL ) {\n"
    else:
      h += "    hipEvent_t *startEvent = NULL,\n"
      h += "    hipEvent_t *stopEvent = NULL ) {\n"

    h += "  // calculate parameters assuming packed data\n"
    # strides
    indexChars = globalParameters["IndexChars"]
    firstStride = 1
    #assert(not problemType["UseInitialStridesCD"]) # not supported in old client
    if problemType["UseInitialStridesAB"]:
      firstStride = 0
    lastStrideD = problemType["NumIndicesC"]
    lastStrideC = problemType["NumIndicesC"]
    lastStrideA = len(problemType["IndexAssignmentsA"])
    lastStrideB = len(problemType["IndexAssignmentsB"])

    # calculate strides
    for i in range(0,lastStrideD):
      h += "  unsigned int strideD%u%s = 1" % (i, indexChars[i])
      for j in range(0, i):
        h += " * ("
        if j == 0:
          h += "(ldd != std::numeric_limits<unsigned int>::max()) ? ldd : "
        h += "std::max(minStrides[%i], sizes[%i]))" % (j,j)
      h += ";\n"
    h += "  if (stride_d != std::numeric_limits<unsigned int>::max())  strideD%u%s = stride_d;\n" % (lastStrideD-1, indexChars[lastStrideD-1])
    for i in range(0,lastStrideC):
      h += "  unsigned int strideC%u%s = 1 " % (i, indexChars[i])
      for j in range(0, i):
        h += " * ("
        if j == 0:
          h += "(ldc != std::numeric_limits<unsigned int>::max()) ? ldc : "
        h+= "std::max(minStrides[%i], sizes[%i]))" % (j,j)
      h += ";\n"
    h += "  if (stride_c != std::numeric_limits<unsigned int>::max())  strideC%u%s = stride_c;\n" % (lastStrideC-1, indexChars[lastStrideC-1])

    constStride = None
    for i in range(0,lastStrideA):
      idx = problemType["IndexAssignmentsA"][i]
      constStride = checkConstStride(problemType["SetConstStrideA"], idx)
      if constStride != None:
        h += "  unsigned int strideA%u%s = %d; //SetConstStrideA\n" % (i,
          indexChars[problemType["IndexAssignmentsA"][i]],
          constStride)
      else:
        h += "  unsigned int strideA%u%s = 1" % (i, \
            indexChars[problemType["IndexAssignmentsA"][i]])
        for j in range(0, i):
          h += " * ("
          if j == 0:
            h += "(lda != std::numeric_limits<unsigned int>::max()) ? lda : "
          h += "std::max(minStrides[%i], sizes[%i]))" % \
            (problemType["IndexAssignmentsA"][j],
             problemType["IndexAssignmentsA"][j])
        h += ";\n"
    if constStride == None:
      h += "  if (stride_a != std::numeric_limits<unsigned int>::max())  strideA%u%s = stride_a;\n" % (lastStrideA-1, indexChars[problemType["IndexAssignmentsA"][lastStrideA-1]])

    for i in range(0,lastStrideB):
      idx = problemType["IndexAssignmentsB"][i]
      constStride = checkConstStride(problemType["SetConstStrideB"], idx)
      if constStride != None:
        h += "  unsigned int strideB%u%s = %d; //SetConstStrideB\n" % (i,
          indexChars[problemType["IndexAssignmentsB"][i]],
          constStride)
      else:
        h += "  unsigned int strideB%u%s = 1" % (i, \
            indexChars[problemType["IndexAssignmentsB"][i]])
        for j in range(0, i):
          h += " * ("
          if j == 0:
            h += "(ldb != std::numeric_limits<unsigned int>::max()) ? ldb : "
          h+= "std::max(minStrides[%i], sizes[%i]))" % \
            (problemType["IndexAssignmentsB"][j],
             problemType["IndexAssignmentsB"][j])
        h += ";\n"
    h += "  if (stride_b != std::numeric_limits<unsigned int>::max())  strideB%u%s = stride_b;\n" % (lastStrideB-1, indexChars[problemType["IndexAssignmentsB"][lastStrideB-1]])

    for i in range(0, problemType["TotalIndices"]):
      h += "  unsigned int size%s = sizes[%u];\n" % (indexChars[i], i)
    h += "\n"


    # function call
    h += "  // Check assertions,\n"
    assert(not problemType["UseInitialStridesCD"]) # not supported in old client
    firstStride = 0 if problemType["UseInitialStridesAB"] else 1
    lastStrideD = problemType["NumIndicesC"]
    lastStrideC = problemType["NumIndicesC"]
    lastStrideA = len(problemType["IndexAssignmentsA"])
    lastStrideB = len(problemType["IndexAssignmentsB"])
    numSizes = problemType["TotalIndices"]
    h += "  typedef ProblemDims<%u,%u,%u,%u,%u,%u> ProblemDims_%s;\n" \
        % (firstStride, lastStrideD, lastStrideC, lastStrideA, lastStrideB, numSizes, problemType)
    # TODO - this should be initialized somewhere once?
    h += "  static const ProblemType problemType( "
    h += listToInitializer(problemType["IndicesFree"]) + ", "
    h += listToInitializer(problemType["IndicesSummation"]) + ", "
    h += listToInitializer(problemType["IndicesBatch"]) + ', '
    h += listToInitializer(problemType["IndexAssignmentsA"]) + ', '
    h += listToInitializer(problemType["IndexAssignmentsB"])
    h += ");\n"
    # create problem size - TODO could move this up to the caller
    h += "  ProblemDims_%s pdims(" % problemType
    indexChars = globalParameters["IndexChars"]
    for i in range(firstStride,lastStrideD):
      if i != firstStride: h += ", "
      h += "strideD%u%s" % (i, indexChars[i])
    for i in range(firstStride,lastStrideC):
      h += ", strideC%u%s" % (i, indexChars[i])
    for i in range(firstStride,lastStrideA):
      h += ", strideA%u%s" % (i, \
          indexChars[problemType["IndexAssignmentsA"][i]])
    for i in range(firstStride,lastStrideB):
      h += ", strideB%u%s" % (i, \
          indexChars[problemType["IndexAssignmentsB"][i]])
    for i in range(0, problemType["TotalIndices"]):
      h += ", size%s" % indexChars[i]
    h += ");\n"
    h += "  if (!ProblemProperties(pdims,&problemType).validForSolution(solution._assertionRequirements))\n"
    h += "    return tensileStatusAssertFailure;  // problem dims did not meet requirements for solution\n"
    h += "\n"

    h += "  // call solution function\n"
    h += "  TensileSolutionPointer_%s f = reinterpret_cast<TensileSolutionPointer_%s> (solution._functionPtr);\n" \
            % (problemType, problemType)
    if globalParameters["RuntimeLanguage"] == "OCL":
      h += "  return f(solutionLock, static_cast<cl_mem>(deviceD), static_cast<cl_mem>(deviceC), static_cast<cl_mem>(deviceA), static_cast<cl_mem>(deviceB),\n"
    else:
      typeName = dataTypes[0].toCpp()
      destTypeName = destDataTypes[dataType].toCpp()
      computeTypeName = computeDataTypes[dataType].toCpp()
      h += "  return f(solutionLock, static_cast<%s *>(deviceD), static_cast<%s *>(deviceC), static_cast<%s *>(deviceA), static_cast<%s *>(deviceB),\n" \
          % (destTypeName, destTypeName, typeName, typeName)
    h += "      alpha,\n"
    if problemType["UseBeta"]:
      h += "      beta,\n"
    for i in range(firstStride,lastStrideD):
      h += "      strideD%u%s,\n" % (i, indexChars[i])
    for i in range(firstStride,lastStrideC):
      h += "      strideC%u%s,\n" % (i, indexChars[i])
    for i in range(firstStride,lastStrideA):
      h += "      strideA%u%s,\n" % (i, \
          indexChars[problemType["IndexAssignmentsA"][i]])
    for i in range(firstStride,lastStrideB):
      h += "      strideB%u%s,\n" % (i, \
          indexChars[problemType["IndexAssignmentsB"][i]])
    for i in range(0, problemType["TotalIndices"]):
      h += "      size%s,\n" % indexChars[i]
    h +=   "      stream,\n"
    if globalParameters["RuntimeLanguage"] == "OCL":
       h += "      numEvents, event_wait_list, outputEvent ); // events\n"
    else:
       h += "      numEvents, startEvent, stopEvent); // events\n"

    h += "};\n"
    h += "\n"
  else:
    ############################################################################
    # Generated Call to Function
    ############################################################################
    for enqueue in [True, False]:
      functionName = "tensile" if enqueue else "tensileGetSolutionName"
      returnName = "TensileStatus" if enqueue else "const char *"
      h += "/* generated call to function */\n"
      h += "template<typename DataType, typename DestDataType, typename ComputeDataType>\n"
      h += "%s generatedCallTo_%s(\n" % (returnName, functionName)
      h += "    unsigned int *sizes,\n"
      h += "    unsigned int *minStrides,\n"
      h += "    ComputeDataType alpha,\n"
      h += "    ComputeDataType beta,\n"
      h += "    unsigned int lda,\n"
      h += "    unsigned int ldb,\n"
      h += "    unsigned int ldc,\n"
      h += "    unsigned int ldd,\n"
      h += "    unsigned int strideA,\n"
      h += "    unsigned int strideB,\n"
      h += "    unsigned int strideC,\n"
      h += "    unsigned int strideD,\n"
      h += "    unsigned int numEvents = 0,\n"

      if globalParameters["RuntimeLanguage"] == "OCL":
        h += "    cl_event *event_wait_list = NULL,\n"
        h += "    cl_event *outputEvent = NULL );\n\n"
      else:
        h += "    hipEvent_t *startEvent = NULL,\n"
        h += "    hipEvent_t *stopEvent = NULL );\n\n"


#need to get DestDataType in here
      for dataType in dataTypes:
        typeName = dataType.toCpp()
        destDataType = destDataTypes[dataType]
        destTypeName = destDataType.toCpp()
        computeDataType = computeDataTypes[dataType]
        computeTypeName = computeDataType.toCpp()
        functionsForDataType = []
        for problemType in problemTypesForDataType[dataType]:
          for scheduleName in schedulesForProblemType[problemType]:
            functionsForDataType.append([scheduleName, problemType])
        h += "template<>\n"
        h += "inline %s generatedCallTo_%s<%s, %s, %s>(\n" \
            % (returnName, functionName, typeName, destTypeName, computeTypeName)
        h += "    unsigned int *sizes,\n"
        h += "    unsigned int *minStrides,\n"
        h += "    %s alpha,\n" % computeTypeName
        h += "    %s beta,\n" % computeTypeName
        h += "    unsigned int lda,\n"
        h += "    unsigned int ldb,\n"
        h += "    unsigned int ldc,\n"
        h += "    unsigned int ldd,\n"
        h += "    unsigned int strideA,\n"
        h += "    unsigned int strideB,\n"
        h += "    unsigned int strideC,\n"
        h += "    unsigned int strideD,\n"
        h += "    unsigned int numEvents, \n"

        if globalParameters["RuntimeLanguage"] == "OCL":
          h += "    cl_event *event_wait_list,\n"
          h += "    cl_event *outputEvent ) {\n\n"
        else:
          h += "    hipEvent_t *startEvent,\n"
          h += "    hipEvent_t *stopEvent ) {\n\n"

        h += "    unsigned int functionIdxForDataType = functionInfo[functionIdx][4];\n"

        for functionIdx in range(0, len(list(functionsForDataType))):
          function = functionsForDataType[functionIdx]
          scheduleName = function[0]
          problemType = function[1]
          if len(list(functionsForDataType))> 1:
            if functionIdx == 0:
              h += "  if (functionIdxForDataType == %u) {\n" % functionIdx
            elif functionIdx == len(list(functionsForDataType))-1:
              h += "  } else {\n"
            else:
              h += "  } else if (functionIdxForDataType == %u) {\n" \
                  % functionIdx

          # strides
          indexChars = globalParameters["IndexChars"]
          firstStride = 1
          assert(not problemType["UseInitialStridesCD"]) # not supported in old client
          if problemType["UseInitialStridesAB"]:
            firstStride = 0
          lastStrideD = problemType["NumIndicesC"]
          lastStrideC = problemType["NumIndicesC"]
          lastStrideA = len(problemType["IndexAssignmentsA"])
          lastStrideB = len(problemType["IndexAssignmentsB"])

          # calculate strides
          for i in range(0,lastStrideD):
            h += "    unsigned int strideD%u%s = 1" % (i, indexChars[i])
            for j in range(0, i):
              h += "*sizes[%i]" % j
            h += ";\n"
          h += "    if (strideD != std::numeric_limits<unsigned int>::max())  strideD%u%s = strideD;\n" % (lastStrideD-1, indexChars[lastStrideD-1])
          for i in range(0,lastStrideC):
            h += "    unsigned int strideC%u%s = 1" % (i, indexChars[i])
            for j in range(0, i):
              h += "*sizes[%i]" % j
            h += ";\n"
          h += "    if (strideC != std::numeric_limits<unsigned int>::max())  strideC%u%s = strideC;\n" % (lastStrideC-1, indexChars[lastStrideC-1])

          for i in range(0,lastStrideA):
            h += "    unsigned int strideA%u%s = 1" % (i, \
                indexChars[problemType["IndexAssignmentsA"][i]])
            for j in range(0, i):
              h += "*sizes[%i]" % \
                problemType["IndexAssignmentsA"][j]
            h += ";\n"
          h += "    if (strideA != std::numeric_limits<unsigned int>::max())  strideA%u%s = strideA;\n" % (lastStrideA-1, indexChars[problemType["IndexAssignmentsA"][lastStrideA-1]])
          for i in range(0,lastStrideB):
            h += "    unsigned int strideB%u%s = 1" % (i, \
                indexChars[problemType["IndexAssignmentsB"][i]])
            for j in range(0, i):
              h += "*sizes[%i]" % \
                problemType["IndexAssignmentsB"][j]
            h += ";\n"
          h += "    if (strideB != std::numeric_limits<unsigned int>::max())  strideB%u%s = strideB;\n" % (lastStrideB-1, indexChars[problemType["IndexAssignmentsB"][lastStrideB-1]])
          for i in range(0, problemType["TotalIndices"]):
            h += "    unsigned int size%s = sizes[%u];\n" % (indexChars[i], i)

          # function call
          h += "    // call solution function\n"
          h += "    return %s_%s(\n" % (functionName, problemType)
          if enqueue:
            if globalParameters["RuntimeLanguage"] == "OCL":
              h += "        static_cast<cl_mem>(deviceD),\n"
              h += "        static_cast<cl_mem>(deviceC),\n"
              h += "        static_cast<cl_mem>(deviceA),\n"
              h += "        static_cast<cl_mem>(deviceB),\n"
            else:
              h += "        static_cast<%s *>(deviceD),\n" % destTypeName
              h += "        static_cast<%s *>(deviceC),\n" % destTypeName
              h += "        static_cast<%s *>(deviceA),\n" % typeName
              h += "        static_cast<%s *>(deviceB),\n" % typeName
            h += "        alpha,\n"
            if problemType["UseBeta"]:
              h += "        beta,\n"
          for i in range(firstStride,lastStrideD):
            h += "        strideD%u%s,\n" % (i, indexChars[i])
          for i in range(firstStride,lastStrideC):
            h += "        strideC%u%s,\n" % (i, indexChars[i])
          for i in range(firstStride,lastStrideA):
            h += "        strideA%u%s,\n" % (i, \
                indexChars[problemType["IndexAssignmentsA"][i]])
          for i in range(firstStride,lastStrideB):
            h += "        strideB%u%s,\n" % (i, \
                indexChars[problemType["IndexAssignmentsB"][i]])
          for i in range(0, problemType["TotalIndices"]):
            h += "        size%s%s\n" % (indexChars[i], "," if i != problemType["TotalIndices"]-1 else "")
          if enqueue:
            if globalParameters["RuntimeLanguage"] == "OCL":
               h += ", stream, numEvents, event_wait_list, outputEvent"
            else:
               h += ", stream, numEvents, startEvent, stopEvent"
          h += ");\n"

        if len(functionsForDataType) > 1:
          h += "  }\n" # close last if
        h += "};\n" # close callToFunction

  ##############################################################################
  # Results File Name
  ##############################################################################
  if forBenchmark:
    h += "/* results file name */\n"
    resultsFileName = os.path.join(stepBaseDir, \
        "../Data","%s.csv" % stepName)
    resultsFileName = resultsFileName.replace("\\", "\\\\")
    h += "const char *resultsFileName = \"%s\";\n" % resultsFileName

    granularityFileName = os.path.join(stepBaseDir, \
        "../Data","%s_Granularity.csv" % stepName)

    granularityFileName = granularityFileName.replace("\\", "\\\\")
    h += "const char *granularityFileName = \"%s\";\n" % granularityFileName

  ##############################################################################
  # Write File
  ##############################################################################
  clientParametersFile = open(os.path.join(globalParameters["WorkingPath"], \
      "ClientParameters.h"), "w")
  clientParametersFile.write(CHeader)
  clientParametersFile.write(h)
  clientParametersFile.close()

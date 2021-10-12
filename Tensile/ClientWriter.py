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

from .Common import globalParameters, pushWorkingPath, popWorkingPath, printExit, printWarning, ClientExecutionLock
from . import ClientExecutable
from . import Common
from . import LibraryIO

import os
import subprocess
from enum import Enum

from .Contractions import FreeIndex
from .Contractions import ProblemType as ContractionsProblemType

class DataInitName(Enum):
  Zero = 0
  One = 1
  Two = 2
  Random = 3
  NaN = 4
  Inf = 5
  BadInput = 6
  BadOutput = 7
  SerialIdx = 8
  SerialDim0 = 9
  SerialDim1 = 10
  Identity = 11
  TrigSin = 12
  TrigCos = 13
  TrigAbsSin = 14
  TrigAbsCos = 15
  RandomNarrow = 16
  NegOne = 17

class ClientLogLevel(Enum):
  Error = 0
  Terse = 1
  Verbose = 2
  Debug = 3

################################################################################
# Write Run Script
################################################################################
def runNewClient(scriptPath, clientParametersPath, clientBuildDir=None):

  clientExe = ClientExecutable.getClientExecutable(clientBuildDir)
  iniFile = "--config-file={}".format(clientParametersPath)
  args = [clientExe, iniFile]

  try:
    subprocess.run(args, check=True)
  except (subprocess.CalledProcessError, OSError) as e:
    printWarning("ClientWriter Benchmark Process exited with error: {}".format(e))

def runClient(libraryLogicPath, forBenchmark, enableTileSelection, configPaths=None):
  # write runScript
  pushWorkingPath("build")
  path = globalParameters["WorkingPath"]

  runScriptName = writeRunScript(path, forBenchmark, enableTileSelection, configPaths)
  with ClientExecutionLock():
    process = subprocess.Popen(runScriptName, cwd=path)
    process.communicate()

  if process.returncode:
    printWarning("ClientWriter Benchmark Process exited with code %u" % process.returncode)
  popWorkingPath() # build

  return process.returncode

def writeRunScript(path, forBenchmark, enableTileSelection, configPaths=None):
  if configPaths is None:
    configPaths = []
    configPaths.append(os.path.join(globalParameters["WorkingPath"], "../source/ClientParameters.ini"))
    if enableTileSelection is True:
      configPaths.append(os.path.join(globalParameters["WorkingPath"], "../source/ClientParameters_Granularity.ini"))

  # create run.bat or run.sh which builds and runs
  runScriptName = os.path.join(path, \
    "run.%s" % ("bat" if os.name == "nt" else "sh") )
  runScriptFile = open(runScriptName, "w")
  if os.name != "nt":
    runScriptFile.write("#!/bin/bash\n\n")

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

    runScriptFile.write("ERR1=0\n")

    clientExe = ClientExecutable.getClientExecutable()
    for configFile in configPaths:
      runScriptFile.write("{} --config-file {} {}\n".format(clientExe, configFile, globalParameters["ClientArgs"]))
    runScriptFile.write("ERR2=$?\n\n")

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
    for configFile in configPaths:
      runScriptFile.write("{} --config-file {} {} --best-solution 1\n".format(ClientExecutable.getClientExecutable(), configFile, globalParameters["ClientArgs"]))
  if os.name != "nt":
    runScriptFile.write("exit $ERR\n")
  runScriptFile.close()
  if os.name != "nt":
    os.chmod(runScriptName, 0o777)
  return runScriptName

def problemSizeParams(problemType, problem):

    numIndices = len(problemType.indices)
    rv = []

    if problem.stridesA:
        astrides = list(problem.stridesA)
    else:
        astrides = [-1] * problemType.aDims
    for sc in problemType.setConstStrideA:
        index = problemType.indices[sc[0]]
        if type(index) == FreeIndex:
            assert(index.isA)
            astrides[index.i] = sc[1]
        else:
            astrides[index.a] = sc[1]

    if problem.stridesB:
      bstrides = list(problem.stridesB)
    else:
      bstrides = [-1] * problemType.bDims
    for sc in problemType.setConstStrideB:
        index = problemType.indices[sc[0]]
        if type(index) == FreeIndex:
            assert(not index.isA)
            bstrides[index.i] = sc[1]
        else:
            bstrides[index.b] = sc[1]

    if problem.stridesC:
      cstrides = list(problem.stridesC)
    else:
      cstrides = [-1] * problemType.cDims

    if problem.stridesD:
      dstrides = list(problem.stridesD)
    else:
      dstrides = [-1] * problemType.dDims

    if len(problem.sizes) == numIndices:
        None
    elif len(problem.sizes) == numIndices + 4:
        # FIXME-problem, this is Exact format with strides tacked onto sizes as 4 extra pams
        # should just set problem.stride* appropriately when reading the Yaml and not deal with extra fields here
        if astrides[1] == -1:
          astrides[1] = problem.sizes[numIndices+2]
        elif astrides[1] != problem.sizes[numIndices+2]:
          raise RuntimeError("problem-specified lda(%u) conflicts with setConstStrideA(%u)" % \
              (astrides[1], problem.sizes[numIndices+2]))

        if bstrides[1] == -1:
          bstrides[1] = problem.sizes[numIndices+3]
        elif bstrides[1] != problem.sizes[numIndices+3]:
          raise RuntimeError("problem-specified ldb(%u) conflicts with setConstStrideB(%u)" % \
              (bstrides[1], problem.sizes[numIndices+3]))

        if cstrides[1] == -1:
          cstrides[1] = problem.sizes[numIndices+1]

        if dstrides[1] == -1:
          dstrides[1] = problem.sizes[numIndices+0]

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

def dataInitParams(problemType):
    initA = globalParameters['DataInitTypeA']
    initB = globalParameters['DataInitTypeB']
    initC = globalParameters['DataInitTypeC']
    initD = globalParameters['DataInitTypeD']
    initAlpha = globalParameters['DataInitTypeAlpha']
    initBeta  = globalParameters['DataInitTypeBeta']

    if not problemType.useBeta:
        initBeta = 0

    if initA == -1: initA = globalParameters['DataInitTypeAB']
    if initB == -1: initB = globalParameters['DataInitTypeAB']

    return [('init-a',     DataInitName(initA).name),
            ('init-b',     DataInitName(initB).name),
            ('init-c',     DataInitName(initC).name),
            ('init-d',     DataInitName(initD).name),
            ('init-alpha', DataInitName(initAlpha).name),
            ('init-beta',  DataInitName(initBeta).name)]

def boundsCheckName(mode):
    if mode == 0: return 'Disable'
    if mode == 1: return 'NaN'
    if mode == 2: return 'GuardPageFront'
    if mode == 3: return 'GuardPageBack'
    if mode == 4: return 'GuardPageAll'


def writeClientConfigIni(problemSizes, problemType, sourceDir, codeObjectFiles, resultsFileName, parametersFilePath, libraryFile=None):

    with open(parametersFilePath, "w") as f:
        def param(key, value):
            f.write("{}={}\n".format(key, value))

        if libraryFile is None:
          libraryFilename = "TensileLibrary.yaml" if globalParameters["LibraryFormat"] == "yaml" else "TensileLibrary.dat"
          libraryFile = os.path.join(sourceDir, "library", libraryFilename)
        param("library-file", libraryFile)

        currentGFXName = Common.gfxName(globalParameters["CurrentISA"])
        for coFile in codeObjectFiles:
            if 'gfx' not in coFile or currentGFXName in coFile:
                param("code-object", os.path.join(sourceDir,coFile))

        param('results-file', resultsFileName)
        convValidation = problemType.convolution and globalParameters["ConvolutionVsContraction"];
        if convValidation:
            param('convolution-identifier', problemType.convolution.identifier())
        param('performance-metric', globalParameters["PerformanceMetric"])
        param('problem-identifier', problemType.operationIdentifier)
        param('a-type',     problemType.aType.toEnum())
        param('b-type',     problemType.bType.toEnum())
        param('c-type',     problemType.cType.toEnum())
        param('d-type',     problemType.dType.toEnum())
        param('alpha-type', problemType.alphaType.toEnum())
        param('beta-type',  problemType.betaType.toEnum())

        param('high-precision-accumulate', problemType.highPrecisionAccumulate)
        param('strided-batched', problemType.stridedBatched)

        for problem in problemSizes.problems:
            for key,value in problemSizeParams(problemType, problem):
                param(key,value)
            if convValidation:
              param('convolution-problem', problemType.convolution.identifier(problem))

        param("device-idx",               globalParameters["Device"])

        for key,value in dataInitParams(problemType):
            param(key, value)

        param("c-equal-d",                globalParameters["CEqualD"])

        param("offset-a",                 globalParameters["BufferOffsetA"])
        param("offset-b",                 globalParameters["BufferOffsetB"])
        param("offset-c",                 globalParameters["BufferOffsetC"])
        param("offset-d",                 globalParameters["BufferOffsetD"])

        if globalParameters["PrintTensorA"]:
          param("print-tensor-a",         1)
        if globalParameters["PrintTensorB"]:
          param("print-tensor-b",         1)
        if globalParameters["PrintTensorC"]:
          param("print-tensor-c",         1)
        if globalParameters["PrintTensorD"]:
          param("print-tensor-d",         1)
        if globalParameters["PrintTensorRef"]:
          param("print-tensor-ref",       1)
        if globalParameters["DumpTensors"]:
          param("dump-tensors",           1)
        if globalParameters["ExitOnFails"] > 1:
          param("exit-on-error", 1)

        param("bounds-check",             boundsCheckName(int(globalParameters["BoundsCheck"])))
        param("print-valids",             globalParameters["ValidationPrintValids"])
        param("print-max",                globalParameters["ValidationMaxToPrint"])
        param("num-benchmarks",           globalParameters["NumBenchmarks"])
        param("num-elements-to-validate", globalParameters["NumElementsToValidate"])
        param("num-enqueues-per-sync",    globalParameters["EnqueuesPerSync"])
        param("num-syncs-per-benchmark",  globalParameters["SyncsPerBenchmark"])
        param("use-gpu-timer",            globalParameters["KernelTime"])
        param("hardware-monitor",         globalParameters["HardwareMonitor"])
        if convValidation:
            param("convolution-vs-contraction", globalParameters["ConvolutionVsContraction"])
        if not globalParameters["KernelTime"]:
            param("num-warmups", 1)
        param("sleep-percent",            globalParameters["SleepPercent"])
        param("perf-l2-read-hits",        globalParameters["PerfModelL2ReadHits"])
        param("perf-l2-write-hits",       globalParameters["PerfModelL2WriteHits"])
        param("perf-l2-read-bw-mul",      globalParameters["PerfModelL2ReadBwMul"])
        param("perf-read-efficiency",     globalParameters["PerfModelReadEfficiency"])
        param("csv-export-extra-cols",    globalParameters["CSVExportWinner"])
        param("csv-merge-same-problems",  globalParameters["CSVMergeSameProblemID"])
        param("log-level",                ClientLogLevel(globalParameters["ClientLogLevel"]).name)
        param("max-workspace-size",       globalParameters["MaxWorkspaceSize"])
        param("granularity-threshold",    globalParameters["GranularityThreshold"])
        param("pristine-on-gpu",          globalParameters["PristineOnGPU"])

        param("library-update-file",      globalParameters["LibraryUpdateFile"])
        param("library-update-comment",   globalParameters["LibraryUpdateComment"])

def writeClientConfig(forBenchmark, solutions, problemSizes, stepName, stepBaseDir, newLibrary, codeObjectFiles, tileAwareSelection, configBase = "ClientParameters", libraryFile = None):

    if tileAwareSelection:
      filename = os.path.join(globalParameters["WorkingPath"], "%s_Granularity.ini"%configBase)
    else:
      filename = os.path.join(globalParameters["WorkingPath"], "%s.ini"%configBase)

    if len(newLibrary.solutions)==0:
      raise RuntimeError ("No valid solutions found")

    resultsFileName = None
    if tileAwareSelection:
      resultsFileName = os.path.join(stepBaseDir, "../Data", stepName+"_Granularity.csv")
    else:
      resultsFileName = os.path.join(stepBaseDir, "../Data", stepName+".csv")

    newSolution = next(iter(newLibrary.solutions.values()))
    sourceDir = os.path.join(stepBaseDir, "source")
    writeClientConfigIni(problemSizes, newSolution.problemType, sourceDir, codeObjectFiles, resultsFileName, filename, libraryFile)

    return filename

def CreateBenchmarkClientParametersForSizes(libraryRootPath, problemSizes, dataFilePath, configFile, problemTypeDict=None):

    libraryPath = os.path.join(libraryRootPath, "library")
    libraryFiles = [os.path.join(libraryPath, f) for f in os.listdir(libraryPath)]
    codeObjectFiles = [f for f in libraryFiles if f.endswith("co")]

    if problemTypeDict:
      problemType = ContractionsProblemType.FromOriginalState(problemTypeDict)
    else:
      # if the we can library contains meta data then we can get the problem type this data
      metaDataFilePath = os.path.join(libraryPath, "metadata.yaml")
      if not os.path.exists(metaDataFilePath):
        printExit ("meta data file %s does not exist" % metaDataFilePath)
      metaData = LibraryIO.readYAML(metaDataFilePath)
      problemTypeDict = metaData["ProblemType"]
      problemType = ContractionsProblemType.FromOriginalState(problemTypeDict)

    writeClientConfigIni(problemSizes, problemType, libraryRootPath, codeObjectFiles, dataFilePath, configFile)

################################################################################
# Copyright (C) 2016-2019 Advanced Micro Devices, Inc. All rights reserved.
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

from __future__ import print_function
import os
import sys
import argparse


def printExit(message):
  print ("Tensile::FATAL: %s" % message)
  sys.stdout.flush()
  sys.exit(-1)

try:
  import yaml
except ImportError:
  printExit("You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions.")

#HR = "################################################################################"


def ensurePath( path ):
  if not os.path.exists(path):
    os.makedirs(path)
  return path

################################################################################
# Define Constants
################################################################################

def constant(f):
  def fset(self, value):
    raise TypeError
  def fget(self):
    return f(self)
  return property(fget, fset)

class _Const(object):
  @constant
  def GlobalParameters(self):
    return "GlobalParameters"
  
  @constant
  def BenchmarkProblems(self):
    return "BenchmarkProblems"

  @constant
  def LibraryLogic(self):
    return "LibraryLogic"

  @constant
  def LibraryClient(self):
    return "LibraryClient"

CONST = _Const()

defaultHeader = {}

defaultHeader["MinimumRequiredVersion"] = "4.2.0"
defaultHeader["ForceRedoBenchmarkProblems"] = True
defaultHeader["ForceRedoLibraryLogic"] = True
defaultHeader["ForceRedoLibraryClient"] = True
defaultHeader["CMakeBuildType"] = "Release"
defaultHeader["EnqueuesPerSync"] = 1
defaultHeader["SyncsPerBenchmark"] = 1
defaultHeader["LibraryPrintDebug"] = False
defaultHeader["NumElementsToValidate"] = 0
defaultHeader["ValidationMaxToPrint"] = 4
defaultHeader["ValidationPrintValids"] = False
defaultHeader["ShortNames"] = False
defaultHeader["MergeFiles"] = True
defaultHeader["Platform"] = 0
defaultHeader["Device"] = 0
defaultHeader["KernelTime"] = True
defaultHeader["PinClocks"] = False
defaultHeader["SleepPercent"] = 20
defaultHeader["DataInitTypeBeta"] = 0
defaultHeader["SolutionSelectionAlg"] = 1
defaultHeader["PrintWinnersOnly"] = 1
defaultHeader["DataInitTypeAB"] = 0
defaultHeader["NewClient"] = 2

################################################################################
# Tuning Configuration Container
################################################################################
class TuningConfiguration(object):
    #__slots__ = ['__globalParameters','__benchmarkProblems','__libraryLogic','__libraryClient']
    def __init__(self,fileName=None):
        self.__globalParameters = None
        self.__benchmarkProblems = None
        self.__libraryLogic = None
        self.__libraryClient = None

        if fileName is not None:
            print("reading configuration: %s" % fileName)
            try:
                stream = open(fileName, "r")
            except IOError:
                printExit("Cannot open file: %s" % filename )

            data = yaml.load(stream, yaml.SafeLoader)

            if CONST.GlobalParameters in data:
                self.__globalParameters = data[CONST.GlobalParameters]
            else:
                self.__globalParameters = None

            if CONST.BenchmarkProblems in data:
                self.__benchmarkProblems = data[CONST.BenchmarkProblems]
            else:
                self.__benchmarkProblems = None

            if CONST.LibraryLogic in data:
                self.__libraryLogic = data[CONST.LibraryLogic]
            else:
                self.__libraryLogic = None

            if CONST.LibraryClient in data:
                self.__libraryClient = True
            else:
                self.__libraryClient = None

            stream.close()

    @property
    def globalParameters(self):
        return self.__globalParameters

    @globalParameters.setter
    def globalParameters(self, value):
        self.__globalParameters = value

    @property
    def benchmarkProblems(self):
        return self.__benchmarkProblems

    @benchmarkProblems.setter
    def benchmarkProblems(self, value):
        self.__benchmarkProblems = value

    @property
    def libraryLogic(self):
        return self.__libraryLogic

    @libraryLogic.setter
    def libraryLogic(self, value):
        self.__libraryLogic = value

    @property
    def libraryClient(self):
        return self.__libraryClient

    @libraryClient.setter
    def libraryClient(self, value):
        self.__libraryClient = value    

    def writeLibraryLogic(self,filename):
  
  # work around to output data in order
        dataGlobal = {}
        dataBenchmark = {}
        dataLibraryLogic = {}

        try:
            stream = open(filename, "w")

            if self.globalParameters:
                dataGlobal[CONST.GlobalParameters] = self.globalParameters
                yaml.dump(dataGlobal, stream, default_flow_style=None, width=1024)
                stream.flush()

            if self.benchmarkProblems:
                dataBenchmark[CONST.BenchmarkProblems] = self.benchmarkProblems     
                #yaml.dump(dataBenchmark, stream, default_flow_style=None, default_style='', width=1024)
                yaml.safe_dump(dataBenchmark, stream, default_flow_style=None)
                stream.flush()

            if self.libraryLogic:
                dataLibraryLogic[CONST.LibraryLogic] = self.libraryLogic
                yaml.dump(dataLibraryLogic, stream, default_flow_style=None, width=1024)
                stream.flush()

            if self.libraryClient:
                stream.write("LibraryClient:\n")
                stream.flush()
        
            stream.close()
        except IOError:
            printExit("Cannot open file: %s" % filename)


def generateProblemType(initialParams, tileAware= "true"):

    if tileAware == "true":
        problemType = {
            "OperationType": "GEMM",
            "DataType": "s",
            "Batched": True,
            "UseBeta": True,
            "TransposeA": False,
            "TransposeB": True,
            "TileAwareSelection": True
        }
    else:
        problemType = {
            "OperationType": "GEMM",
            "DataType": "s",
            "Batched": True,
            "UseBeta": True,
            "TransposeA": False,
            "TransposeB": True,
        }

    if initialParams:
        keys = list(initialParams.keys())
        for key in keys:
            problemType[key] = initialParams[key]

    return problemType


arcturusLibraryLogic={'ArchitectureName': 'gfx908', 'DeviceNames': ['Device 7380', 'Device 7388', 'Device 738c', 'Device 7390', 'Device 731f'], 'ScheduleName': 'arcturus'}
vega20LibraryLogic={'ArchitectureName': 'gfx906', 'DeviceNames': ['Device 66a0', 'Device 66a1', 'Device 66a7', 'Device 66af', 'Vega 20'], 'ScheduleName': 'vega20'}
vega10LibraryLogic={'ArchitectureName': 'gfx900', 'DeviceNames': ['Device 6863', 'Device 6862', 'Device 687f', 'Device 6860', 'Device 6861', 'Vega 10 XTX [Radeon Vega Frontier Edition]', 'Vega [Radeon RX Vega]'], 'ScheduleName': 'vega10'}
mi25LibraryLogic={'ArchitectureName': 'gfx900', 'DeviceNames': ['Device 6860'], 'ScheduleName': 'mi25'}
r9nanoLibraryLogic={'ArchitectureName': 'gfx803', 'DeviceNames': ['Device 7300'], 'ScheduleName': 'r9nano'}
hipLibraryLogic={'ArchitectureName': 'fallback', 'DeviceNames': ['Device 0000'], 'ScheduleName': 'hip'}

libraryLogicMapper={'arcturus': arcturusLibraryLogic, 'vega20': vega20LibraryLogic, 'vega10': vega10LibraryLogic, 'mi25': mi25LibraryLogic, 'r9nano': r9nanoLibraryLogic, 'hip': hipLibraryLogic}

def getLibraryLogic(logicType):
    libraryLogic = libraryLogicMapper[logicType]
    return libraryLogic
  
def appendThreadTiles(benchmarkGroup, threadTiles):
    forkedParams = benchmarkGroup["ForkParameters"]
    forkedParams.append({"ThreadTile": threadTiles})

def appendWorkGroups(benchmarkGroup, workGroups):
    forkedParams = benchmarkGroup["ForkParameters"]
    forkedParams.append({"WorkGroup": workGroups})

def appendSizes(benchmarkGroup, sizes, tileAware="true"):
    benchmarkFinalParams = benchmarkGroup["BenchmarkFinalParameters"]
    problemSizes = []
    for size in sizes:
        problemSizes.append({"Exact": size})

    if not benchmarkFinalParams:
        benchmarkFinalParams = []
        benchmarkGroup["BenchmarkFinalParameters"] = benchmarkFinalParams

    if tileAware == "false":
        benchmarkFinalParams.append({"ProblemSizes":problemSizes})
    
def generateEmptyBenchmarkGroup():
    benchmarkGroup={"InitialSolutionParameters":None,"BenchmarkCommonParameters":None,"ForkParameters":None,"BenchmarkForkParameters":None,"JoinParameters":None,
                    "BenchmarkJoinParameters":None,"BenchmarkFinalParameters":None}

    return benchmarkGroup


def generateDefaultScheme():
    scheme={"EdgeType": ["ShiftPtr"],
            "KernelLanguage": ["Assembly"],
            "LoopTail": [True],
            "WorkGroupMapping": [1, 8],
            "DepthU": [16],
            "VectorWidth": [-1],
            "GlobalSplitU": [1],
            "GlobalReadVectorWidth": [-1],
            "FractionalLoad": [1],
            "PrefetchGlobalRead": [ False ],
            "PrefetchLocalRead": [ False, True]}
    return scheme

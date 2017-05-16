################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
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
import os.path
import sys
import inspect
from __init__ import __version__
from collections import OrderedDict
import time

startTime = time.time()

# print level
# 0 - user wants no printing
# 1 - user wants limited prints
# 2 - user wants full prints

################################################################################
# Global Parameters
################################################################################
globalParameters = OrderedDict()
globalParameters["IndexChars"] =  "IJKLMNOPQRSTUVWXYZ"
if os.name == "nt":
  globalParameters["RuntimeLanguage"] = "OCL"
  globalParameters["KernelLanguage"] = "OCL"
else:
  globalParameters["RuntimeLanguage"] = "HIP"
  globalParameters["KernelLanguage"] = "HIP"
# print level
globalParameters["PrintLevel"] = 1
globalParameters["LibraryPrintDebug"] = False
globalParameters["PrintSolutionRejectionReason"] = False
# paths
globalParameters["ScriptPath"] = os.path.dirname(os.path.realpath(__file__))
globalParameters["SourcePath"] = os.path.join(globalParameters["ScriptPath"], "Source")
globalParameters["WorkingPath"] = os.getcwd()
globalParameters["BenchmarkProblemsPath"] = "1_BenchmarkProblems"
globalParameters["BenchmarkDataPath"] = "2_BenchmarkData"
globalParameters["LibraryLogicPath"] = "3_LibraryLogic"
globalParameters["LibraryClientPath"] = "4_LibraryClient"
# device
globalParameters["Platform"] = 0
globalParameters["Device"] = 0
# benchmark behavior
globalParameters["EnableHalf"] = False
globalParameters["CMakeBuildType"] = "Release" # Debug
globalParameters["CMakeCXXFlags"] = ""
globalParameters["CMakeCFlags"] = ""
globalParameters["ForceRedoBenchmarkProblems"] = True
globalParameters["ForceRedoLibraryLogic"] = True
globalParameters["ForceRedoLibraryClient"] = True
globalParameters["EnqueuesPerSync"] = 1
globalParameters["SyncsPerBenchmark"] = 4
globalParameters["PinClocks"] = False
globalParameters["KernelTime"] = False

# file heirarchy
globalParameters["ShortNames"] = False
globalParameters["MergeFiles"] = True
# validation
globalParameters["NumElementsToValidate"] = 128
globalParameters["ValidationMaxToPrint"] = 4
globalParameters["ValidationPrintValids"] = False
globalParameters["DataInitTypeAB"] = 0 # 0=rand, 1=1, 2=serial, 3=0
globalParameters["DataInitTypeC"]  = 0 # 0=rand, 1=1, 2=serial, 3=0
# protect against invalid kernel
globalParameters["MaxLDS"] = 32768
globalParameters["DeviceLDS"] = 32768
globalParameters["MinimumRequiredVersion"] = "0.0.0"

################################################################################
# Default Benchmark Parameters
################################################################################
validParameters = {
    "LoopDoWhile":                [ False, True ],
    "LoopTail":                   [ False, True ],
    "LocalWriteCoalesceGroupA":   [ False, True ],
    "LocalWriteCoalesceGroupB":   [ False, True ],
    "GlobalReadCoalesceVectorA":  [ False, True ],
    "GlobalReadCoalesceVectorB":  [ False, True ],
    "PrefetchGlobalRead":         [ False, True ],
    "PrefetchLocalRead":          [ False, True ],
    "UnrollMemFence":             [ False, True ],
    "GlobalSplitUWorkGroupMappingRoundRobin":     [ False, True ],
    "GlobalSplitUSummationAssignmentRoundRobin":  [ False, True ],

    "WorkGroupMapping":           [1]+range(-1024,0)+range(2,1025),
    "MaxOccupancy":               [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ], # wg / CU
    "GroupShape":                 [ -64, -32, -16, -8, -4, -2,1,2,4,8,16,32,64],
    "ThreadTileShape":            [ -64, -32, -16, -8, -4, -2,1,2,4,8,16,32,64],
    "NumLoadsCoalescedA":         [ -1, 1, 2, 3, 4, 6, 8, 16, 32, 64 ],
    "NumLoadsCoalescedB":         [ -1, 1, 2, 3, 4, 6, 8, 16, 32, 64 ],
    "ThreadTileNumElements":      [ 1, 2, 4, 8, 16, 32, 64, 36],
    "DepthU":                     [ 1, 2, 4, 8, 16, 32, 64, 128, 256 ],
    "LocalSplitU":                [ 1, 2, 4, 8, 16, 32, 64 ],
    "GlobalSplitU":               [ 1, 2, 4, 8, 16, 32, 64 ],
    "NumThreads":                 [ 64, 128, 256 ],
    "VectorWidth":                [ -1, 1, 2, 4 ],
    "LdsPad":                     [ 0, 1 ],
    "MacroTileShapeMin":          [ 1, 2, 4, 8, 16, 32, 64 ],
    "MacroTileShapeMax":          [ 1, 2, 4, 8, 16, 32, 64 ],

    "EdgeType":                   [ "Branch", "Shift", "None" ],

    }
# same parameter for all solution b/c depends only on compiler
defaultBenchmarkCommonParameters = [
    {"LoopDoWhile":               [ False ] },
    {"LoopTail":                  [ True ] },
    {"EdgeType":                  [ "Branch" ] },
    {"LdsPad":                    [ 0 ] },
    {"MaxOccupancy":              [ 10 ] },
    {"VectorWidth":               [ 1 ] }, # =2 once fixed
    {"GlobalReadCoalesceVectorA": [ True ] },
    {"GlobalReadCoalesceVectorB": [ True ] },
    {"LocalWriteCoalesceGroupA":  [ True ] },
    {"LocalWriteCoalesceGroupB":  [ True ] },
    {"NumThreads":                [ 16*16] },
    {"GroupShape":                [ 1 ] },
    {"ThreadTileShape":           [ 1 ] },
    {"PrefetchGlobalRead":        [ False ] },
    {"PrefetchLocalRead":         [ False ] },
    {"UnrollMemFence":            [ False ] },
    {"LocalSplitU":               [ 1 ] },
    {"GlobalSplitU":              [ 1 ] },
    {"GlobalSplitUWorkGroupMappingRoundRobin":    [ True ] },
    {"GlobalSplitUSummationAssignmentRoundRobin": [ True ] },
    {"MacroTileShapeMin":          [ 1 ] },
    {"MacroTileShapeMax":          [ 4 ] },
    ]
# benchmark these solution independently
defaultForkParameters = [
    {"ThreadTileNumElements":   [ 4*4, 2*2, 6*6, 8*8 ] },
    {"NumLoadsCoalescedA":      [ 1, -1 ] },
    {"NumLoadsCoalescedB":      [ 1, -1 ] },
    {"DepthU":                  [ 4, 8, 16 ] },
    ]
# keep one winner per solution and it affects which will win
defaultBenchmarkForkParameters = [
    {"WorkGroupMapping":        [ 1 ] },
    ]
# final list of solutions
defaultJoinParameters = [
    "MacroTile" ]
# keep one winner per solution and it would affect which solutions fastest
defaultBenchmarkJoinParameters = [
    ]

# dictionary of defaults comprised for 1st option for each parameter
defaultSolution = {}
for paramList in [defaultBenchmarkCommonParameters, defaultForkParameters, \
    defaultBenchmarkForkParameters,defaultBenchmarkJoinParameters]:
  for paramDict in paramList:
    for key, value in paramDict.iteritems():
      defaultSolution[key] = value[0]
# other non-benchmark options for solutions

################################################################################
# Default Problem Type
################################################################################
defaultProblemType = {
    "OperationType":            "GEMM",
    "UseBeta":                  True,
    "UseInitialStrides":        False,
    "HighPrecisionAccumulate":  False,
    "TransposeA":               False,
    "TransposeB":               True,
    "ComplexConjugateA":        False,
    "ComplexConjugateB":        False,
    "Batched":                  False,
    "IndexAssignmentsA":        [0, 2],
    "IndexAssignmentsB":        [1, 2],
    "NumDimensionsC":           2,
    "DataType":                 0,
    }
defaultProblemSizes = "[{Range:[ [2880], 0, 0 ]}]"
defaultBenchmarkFinalProblemSizes = """[{Range:[
    [64, 64, 64, 2880],
    [64, 64, 64, 2880],
    [64, 64, 64, 2880] ]}]"""


################################################################################
# Default Analysis Parameters
################################################################################
defaultAnalysisParameters = {
    "ScheduleName":       "Default",
    "DeviceNames":  ["Unspecified"],
    "BranchPenalty":              0, # microseconds / kernel
    "SmoothOutliers":         False, # enforce monotonic data
    "SolutionImportanceMin":      0, # = keep all solutions 0.01=1% of wins
    }


################################################################################
# Searching Nested Lists / Dictionaries
################################################################################
# param name in structures?
def inListOfDictionaries(param, dictionaries):
  for dictionary in dictionaries:
    if param in dictionary:
      return True
  return False
def inListOfListOfDictionaries(param, dictionaries):
  for dictionaryList in dictionaries:
    if inListOfDictionaries(param, dictionaryList):
      return True
  return False
def inListOfLists(param, lists):
  for l in lists:
    if param in l:
      return True
  return False

# get param values from structures.
def hasParam( name, structure ):
  if isinstance(structure, list):
    for l in structure:
      if hasParam(name, l):
        return True
    return False
  elif isinstance(structure, dict):
    return name in structure
  else:
    return name == structure
    #printExit("structure %s is not list or dict" % structure)

def getParamValues( name, structure ):
  if isinstance(structure, list):
    for l in structure:
      param = getParamValues(name, l)
      if param != None:
        return param
    return None
  elif isinstance(structure, dict):
    if name in structure:
      return structure[name]
    else:
      return None
  else:
    printExit("structure %s is not list or dict" % structure)

################################################################################
# Print Debug
################################################################################
def print1(message):
  if globalParameters["PrintLevel"] >= 1:
    print message
def print2(message):
  if globalParameters["PrintLevel"] >= 2:
    print message

def printWarning(message):
  print "Tensile::WARNING: %s" % message
def printExit(message):
  print "Tensile::FATAL: %s" % message
  sys.exit(-1)



################################################################################
# Assign Global Parameters
################################################################################
def assignGlobalParameters( config ):
  global globalParameters

  if "MinimumRequiredVersion" in config:
    if not versionIsCompatible(config["MinimumRequiredVersion"]):
      printExit("Benchmark.yaml file requires version=%s is not compatible with current Tensile version=%s" \
          % (config["MinimumRequiredVersion"], __version__) )

  print2("GlobalParameters:")
  for key in globalParameters:
    defaultValue = globalParameters[key]
    if key in config:
      configValue = config[key]
      if configValue == defaultValue:
        print2(" %24s: %8s (same)" % (key, configValue))
      else:
        print2(" %24s: %8s (overriden)" % (key, configValue))
    else:
      print2(" %24s: %8s (unspecified)" % (key, defaultValue))

  for key in config:
    value = config[key]
    if key not in globalParameters:
      printWarning("Global parameter %s = %s unrecognised." % ( key, value ))
    globalParameters[key] = value


################################################################################
# Assign Parameters
################################################################################
def assignParameterWithDefault(destinationDictionary, key, sourceDictionary, \
    defaultDictionary):
  if key in sourceDictionary:
    destinationDictionary[key] = sourceDictionary[key]
  else:
    destinationDictionary[key] = defaultDictionary[key]

def assignParameterRequired(destinationDictionary, key, sourceDictionary):
  if key in sourceDictionary:
    destinationDictionary[key] = sourceDictionary[key]
  else:
    printExit("Parameter \"%s\" must be defined in dictionary %s" % (key, sourceDictionary) )


################################################################################
# Push / Pop Working Path
################################################################################
def pushWorkingPath( foldername ):
  globalParameters["WorkingPath"] = \
      os.path.join(globalParameters["WorkingPath"], foldername )
  ensurePath( globalParameters["WorkingPath"] )
def popWorkingPath():
  globalParameters["WorkingPath"] = \
      os.path.split(globalParameters["WorkingPath"])[0]
def ensurePath( path ):
  if not os.path.exists(path):
    os.makedirs(path)

################################################################################
# Is query version compatible with current version
################################################################################
def versionIsCompatible(queryVersionString):
  (qMajor, qMinor, qPatch) = queryVersionString.split(".")
  (tMajor, tMinor, tPatch) = __version__.split(".")

  # major version must match exactly
  if qMajor != tMajor:
    return False

  # minor.patch version must be >=
  if qMinor > tMinor:
    return False
  if qMinor == tMinor:
    if qPatch > tPatch:
      return False
  return True

# TODO
CMakeHeader = "# Header\n\n"
CHeader = "// Header\n\n"
HR = "################################################################################"

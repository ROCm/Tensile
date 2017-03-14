import os.path
import sys
import inspect
from collections import OrderedDict

# print level
# 0 - user wants no printing
# 1 - user wants limited prints
# 2 - user wants full prints

################################################################################
# Global Parameters
################################################################################
globalParameters = OrderedDict()
globalParameters["IndexChars"] =  "IJKLMNOPQRSTUVWXYZ"
globalParameters["Name"] = "Tensile"
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
globalParameters["CMakeBuildType"] = "Release" # Debug
globalParameters["ForceRedoBenchmarkProblems"] = True
globalParameters["ForceRedoLibraryLogic"] = True
globalParameters["ForceRedoLibraryClient"] = True
globalParameters["EnqueuesPerSync"] = 1
globalParameters["SyncsPerBenchmark"] = 4
globalParameters["PinClocks"] = False
# file heirarchy
globalParameters["ShortNames"] = False
globalParameters["MergeFiles"] = True
# validation
globalParameters["NumElementsToValidate"] = 16
globalParameters["ValidationMaxToPrint"] = 16
globalParameters["ValidationPrintValids"] = False
globalParameters["DataInitType"] = 0 # 0=rand, 1=1, 2=serial
# protect against invalid kernel
globalParameters["MaxLDS"] = 32768
globalParameters["MaxMacroTileRatio"] = 4

# BF00 

################################################################################
# Default Benchmark Parameters
################################################################################
validParameters = {
    "LoopDoWhile":            [ False, True ],
    "LoopTail":               [ False, True ],
    "Prefetch":               [ False, True ] ,

    "WorkGroupMapping":       [1]+range(-1024,0)+range(2,1025),
    "GroupShape":             [ -64, -32, -16, -8, -4, -2, 0, 2, 4, 8, 16, 32, 64 ],
    "ThreadTileShape":        [ -64, -32, -16, -8, -4, -2, 0, 2, 4, 8, 16, 32, 64 ],
    "NumLoadsCoalescedA":     [ -1, 1, 2, 3, 4, 6, 8, 16, 32, 64 ],
    "NumLoadsCoalescedB":     [ -1, 1, 2, 3, 4, 6, 8, 16, 32, 64 ],
    "ThreadTileNumElements":  [ 1, 2, 4, 8, 16, 32, 64, 36],
    "DepthU":                 [ 1, 2, 4, 8, 16, 32, 64, 128, 256 ],
    "SplitU":                 [ 1, 2, 4, 8, 16, 32, 64 ],
    "NumThreads":             [ 64, 128, 256 ],
    "VectorWidth":            [ -1, 1, 2, 4 ],
    "LdsPad":                 [ 0, 1 ],

    "EdgeType":               [ "Branch", "Shift", "None" ],

    }
# same parameter for all solution b/c depends only on compiler
defaultBenchmarkCommonParameters = [
    {"LoopDoWhile":             [ False ] },
    {"LoopTail":                [ True ] },
    {"EdgeType":                [ "Branch" ] }, # Shift
    {"LdsPad":                  [ 1 ] }, # 0
    {"Prefetch":                [ False ] },
    {"VectorWidth":             [ -1 ] },
    ]
# benchmark these solution independently
defaultForkParameters = [
    {"NumThreads":              [ 16*16, 8*8 ] },
    {"GroupShape":              [ 0 ] }, # -4, -2, 0, 2, 4
    {"ThreadTileNumElements":   [ 4*4, 2*2, 6*6, 8*8 ] },
    {"ThreadTileShape":         [ 0 ] }, # -4, -2, 0, 2, 4
    {"NumLoadsCoalescedA":      [ 1, -1 ] },
    {"NumLoadsCoalescedB":      [ 1, -1 ] },
    {"DepthU":                  [ 4, 8, 16, 32 ] },
    {"SplitU":                  [ 1, 4, 16 ] },
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

# derived parameters may show up in solution dict but don't use for naming
"""
derivedParameters = [
    "SubGroup0",
    "SubGroup1",
    "ThreadTile0",
    "ThreadTile1",
    "Valid",
    "MacroTile0",
    "MacroTile1",
    "NumElementsPerThread",
    "NumLoadsA",
    "NumLoadsB",
    "NumLoadsPerpendicularA",
    "NumLoadsPerpendicularB",
    "LdsOffsetB",
    "LdsNumElements",
    "LoopUnroll",
    "AssignedDerivedParameters",
    "AssignedProblemIndependentDerivedParameters",
    "BenchmarkFork"
    ]
"""

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
defaultProblemSizes = [ [128], 0, 0 ]
defaultBenchmarkFinalProblemSizes = [
    [16, 16, 16, 128],
    [16, 16, 16, 128],
    [16, 16, 16, 128] ]


################################################################################
# Default Analysis Parameters
################################################################################
defaultAnalysisParameters = {
    "InitialSolutionWindow":      4,
    "BranchPenalty":            100, # microseconds / kernel
    "SmoothOutliers":         False, # enforce monotonic data
    "SolutionImportanceMin":   0.01, # = 1%
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
  #f = inspect.currentframe().f_back.f_code
  #filebase = os.path.splitext(os.path.basename(f.co_filename))[0]
  #print "Tensile::%s::%s - WARNING - %s" % (filebase, f.co_name, message)
  print "Tensile::WARNING: %s" % message
def printExit(message):
  #f = inspect.currentframe().f_back.f_code
  #filebase = os.path.splitext(os.path.basename(f.co_filename))[0]
  #print "Tensile::%s::%s - FATAL - %s" % (filebase, f.co_name, message)
  print "Tensile::FATAL: %s" % message
  sys.exit(-1)



################################################################################
# Assign Global Parameters
################################################################################
def assignGlobalParameters( config ):
  global globalParameters

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


# TODO
CMakeHeader = "# Header\n\n"
CHeader = "// Header\n\n"
HR = "################################################################################"


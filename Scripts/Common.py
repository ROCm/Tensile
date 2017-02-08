import os.path
import sys
import inspect
from collections import OrderedDict
from sets import Set

# debug print level
# 0 - user wants no printing
# 1 - user wants basic status
# 2 - user wants debugging

################################################################################
# Global Parameters
################################################################################
globalParameters = OrderedDict()
globalParameters["IndexChars"] =  "IJKLMNOPQRSTUVWXYZ"
globalParameters["Name"] = "Tensile"
if os.name == "nt":
  globalParameters["Backend"] = "OCL"
else:
  globalParameters["Backend"] = "HIP"
# print debug
globalParameters["DebugPrintLevel"] = 1
globalParameters["LibraryPrintDebug"] = False
# paths
globalParameters["ScriptPath"] = os.path.dirname(os.path.realpath(__file__))
globalParameters["SourcePath"] = os.path.join(globalParameters["ScriptPath"], "..", "Source")
globalParameters["WorkingPath"] = os.getcwd()
globalParameters["BenchmarkProblemsPath"] = "1_BenchmarksProblems"
globalParameters["BenchmarkDataPath"] = "2_BenchmarkData"
globalParameters["LibraryLogicPath"] = "3_LibraryLogic"
globalParameters["LibraryClientPath"] = "4_LibraryClient"
# device
globalParameters["PlatformIdx"] = 0
globalParameters["DeviceIdx"] = 0
# benchmark behavior
globalParameters["CMakeBuildType"] = "Release" # Debug
globalParameters["ForceRedoBenchmarkProblems"] = True
globalParameters["ForceRedoLibraryLogic"] = True
globalParameters["ForceRedoLibraryClient"] = True
globalParameters["EnqueuesPerSync"] = 1
globalParameters["SyncsPerBenchmark"] = 4
# file heirarchy
globalParameters["ShortFileNames"] = False
globalParameters["MergeFiles"] = True
# validation
globalParameters["NumElementsToValidate"] = 16
globalParameters["ValidationMaxToPrint"] = 16
globalParameters["ValidationPrintValids"] = False
globalParameters["DataInitType"] = 0 # 0=rand, 1=1, 2=serial
# protect against invalid kernel
globalParameters["MaxThreads"] = 256
globalParameters["MaxRegisters"] = 256
globalParameters["MaxLDS"] = 32768


################################################################################
# Default Benchmark Parameters
################################################################################
# same parameter for all solution b/c depends only on compiler
defaultBenchmarkCommonParameters = [
    {"KernelMaxSizes":          [ [0, 0, 0] ] }, # infinite
    {"KernelSerial":            [ True ] },
    {"LoopDoWhile":             [ True ] },
    {"LoopTail":                [ False ] },
    {"LoadMacInterleave":       [ 4 ] },
    {"AtomicAccumulate":        [ False ] },
    {"EdgeType":                [ "Branch" ] }, # Shift
    {"EdgeMultiKernel":         [ False ] },
    {"PadLDS":                  [ 1 ] },
    ]
# benchmark these solution independently
defaultForkParameters = [
    {"WorkGroupEdge":           [ 16, 8 ] },
    {"WorkGroupShape":          [ 0 ] }, # -1, 0, 1
    {"ThreadTileEdge":          [ 1, 2, 4, 6, 8 ] },
    {"ThreadTileShape":         [ 0 ] }, # -1, 0, 1
    {"SplitU":                  [ 1 ] },
    {"Prefetch":                [ False ] },
    ]
# keep one winner per solution and it affects which will win
defaultBenchmarkForkParameters = [
    {"WorkGroupMapping":        [ 1 ] },
    {"LoopUnroll":              [ 16, 8, 4 ] },
    ]
# final list of solutions
defaultJoinParameters = [
    "MacroTile", "DepthU"
    ]
# keep one winner per solution and it would affect which solutions fastest
defaultBenchmarkJoinParameters = [
    {"NumLoadsCoalescedA":       [ 1, 2, 3, 4, 6, 8 ] },
    {"NumLoadsCoalescedB":       [ 1, 2, 3, 4, 6, 8 ] },
    {"VectorWidthGlobalLoad":   [ 4 ] },
    {"VectorWidthGlobalStore":  [ 4 ] },
    {"VectorWidthLocalLoad":    [ 4 ] },
    {"VectorWidthLocalStore":   [ 4 ] },
    ]

# derrived parameters may show up in solution dict but don't use for naming
derrivedParameters = [
    "MacroTile0",
    "MacroTile1",
    "WorkGroup0",
    "WorkGroup1",
    "ThreadTile0",
    "ThreadTile1",
    "NumLoadsA",
    "NumLoadsB",
    "NumLoadsPerpendicularA",
    "NumLoadsPerpendicularB",
    ]

# dictionary of defaults comprised for 1st option for each parameter
defaultSolution = {}
for paramList in [defaultBenchmarkCommonParameters, defaultForkParameters, \
    defaultBenchmarkForkParameters,defaultBenchmarkJoinParameters]:
  for paramDict in paramList:
    for key, value in paramDict.iteritems():
      defaultSolution[key] = value[0]
# other non-benchmark options for solutions
defaultSolution["MacroTileMaxRatio"] = 2

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
    "Dilation":                 3,
    "Threshold":                0.1,
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
def printStatus( message): # 0
  f = inspect.currentframe().f_back.f_code
  filebase = os.path.splitext(os.path.basename(f.co_filename))[0]
  print "Tensile::%s::%s - %s" % (filebase, f.co_name, message)
def printExtra( message): # 1
  f = inspect.currentframe().f_back.f_code
  filebase = os.path.splitext(os.path.basename(f.co_filename))[0]
  print "Tensile::%s::%s - %s" % (filebase, f.co_name, message)
def printDebug( message): # 2
  f = inspect.currentframe().f_back.f_code
  filebase = os.path.splitext(os.path.basename(f.co_filename))[0]
  print "Tensile::%s::%s - %s" % (filebase, f.co_name, message)
def printWarning( message): # 1
  f = inspect.currentframe().f_back.f_code
  filebase = os.path.splitext(os.path.basename(f.co_filename))[0]
  print "Tensile::%s::%s - WARNING - %s" % (filebase, f.co_name, message)
def printDefault( message): # 1
  f = inspect.currentframe().f_back.f_code
  filebase = os.path.splitext(os.path.basename(f.co_filename))[0]
  print "Tensile::%s::%s - DEFAULT - %s" % (filebase, f.co_name, message)
def printExit( message): # 2
  f = inspect.currentframe().f_back.f_code
  filebase = os.path.splitext(os.path.basename(f.co_filename))[0]
  print "Tensile::%s::%s - FATAL - %s" % (filebase, f.co_name, message)
  sys.exit(-1)



################################################################################
# Assign Global Parameters
################################################################################
def assignGlobalParameters( config ):
  global globalParameters

  print "GlobalParameters:"
  for key in globalParameters:
    defaultValue = globalParameters[key]
    if key in config:
      configValue = config[key]
      if configValue == defaultValue:
        print " %24s: %8s (same)" % (key, configValue)
      else:
        print " %24s: %8s (overriden)" % (key, configValue)
    else:
      print " %24s: %8s (unspecified)" % (key, defaultValue)

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
  #print globalParameters["WorkingPath"]
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


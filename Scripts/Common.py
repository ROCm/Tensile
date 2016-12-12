import os.path
import sys
import inspect

from collections import OrderedDict
from sets import Set

# debug print level
# 0 - user wants no printing
# 1 - user wants basic status
# 2 - user wants debugging
indexChars = "ijklmnopqrstuvwxyz"

globalParameters = OrderedDict()
globalParameters["Name"] = "Tensile"
globalParameters["Backend"] = "OCL" # OCL, HIP, ASM
globalParameters["DebugPrintLevel"] = 1
globalParameters["ScriptPath"] = os.path.dirname(os.path.realpath(__file__))
globalParameters["WorkingPath"] = os.getcwd()
globalParameters["Redo"] = "Changed" # Force None

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


defaultProblemSizes = [ [5760], 0, 0 ]

# same parameter for all solution b/c depends only on compiler
defaultBenchmarkCommonParameters = [
    {"KernelGrid":              [ [1, 1, 1] ] },
    {"KernelSerial":            [ True, False ] },
    {"LoopFor":                 [ True, False ] },
    {"LoopTail":                [ True ] },
    {"LoadMacInterleave":       [ 4, 8, 16 ] },
    {"AtomicAccumulate":        [ False ] },
    {"EdgeType":                [ "Shift", "Branch", "None", "MultiShift", "MultiBranch" ] },
    ]
# benchmark these solution independently
defaultForkParameters = [
    {"WorkGroupEdge":           [ 16, 8 ] },
    {"WorkGroupShape":          [ 0, -1, 1 ] },
    {"ThreadTileEdge":          [ 1, 2, 4, 6, 8 ] },
    {"ThreadTileShape":         [ 0, -1, 1 ] },
    {"SplitU":                  [ 1, 4, 16, 64 ] },
    {"Prefetch":                [ True, False ] },
    ]
# keep one winner per solution and it affects which will win
defaultBenchmarkForkParameters = [
    {"WorkGroupOrder":          [ 1, -1, 4, -4, 8, -8 ] },
    {"LoopUnroll":              [ 16, 8, 4, 32 ] },
    ]
# final list of solutions
defaultJoinParameters = [
    "MacroTile", "DepthU"
    ]
# keep one winner per solution and it would affect which solutions fastest
defaultBenchmarkJoinParameters = [
    {"NumLoadsParaA":           [ 1, 2, 3, 4, 6, 8 ] },
    {"NumLoadsParaB":           [ 1, 2, 3, 4, 6, 8 ] },
    {"GlobalLoadVectorWidth":   [ 4, 2, 1 ] },
    {"LocalStoreVectorWidth":   [ 4, 2, 1 ] },
    {"LocalLoadVectorWidth":    [ 4, 2, 1 ] },
    {"GlobalStoreVectorWidth":  [ 4, 2, 1 ] },
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

# default problem
defaultProblemType = {
    "OperationType":            "GEMM",
    "UseBeta":                  True,
    "UseOffsets":               True,
    "UseInitialStrides":        False,
    "HighPrecisionAccumulate":  False,
    "TransposeA":               False,
    "TransposeB":               True,
    "Batched":                  False,
    "IndexAssignmentsA":        [0, 2],
    "IndexAssignmentsB":        [1, 2],
    "NumDimensionsC":           2,
    "DataType":                 0,

    }

# printing for status and debugging
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
def printExit( message): # 2
  f = inspect.currentframe().f_back.f_code
  filebase = os.path.splitext(os.path.basename(f.co_filename))[0]
  print "Tensile::%s::%s - FATAL - %s" % (filebase, f.co_name, message)
  sys.exit(-1)



################################################################################
# Global Parameters
################################################################################
def assignGlobalParameters( config ):
  global globalParameters
  for key in config:
    value = config[key]
    if key not in globalParameters:
      printWarning("Global parameter %s = %s unrecognised." % ( key, value ))
    globalParameters[key] = value

  print "Tensile::Common::assignGlobalParameters::globalParameters"
  for key in globalParameters:
    value = globalParameters[key]
    if key in config:
      if config[key] == globalParameters[key]:
        printExtra(" %16s: %8s (same)" % (key, value) )
      else:
        printExtra(" %16s: %8s (overriden)" % (key, value) )
    else:
      printExtra(" %16s: %8s" % (key, value) )



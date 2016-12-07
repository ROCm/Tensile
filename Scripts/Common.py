import os.path
import inspect

from collections import OrderedDict

# debug print level
# 0 - user wants no printing
# 1 - user wants basic status
# 2 - user wants debugging
globalParameters = OrderedDict()
globalParameters["Name"] = "Tensile"
globalParameters["Backend"] = "OCL" # OCL, HIP, ASM
globalParameters["DebugPrintLevel"] = 1
globalParameters["ScriptPath"] = os.path.dirname(os.path.realpath(__file__))
globalParameters["WorkingPath"] = os.getcwd()
globalParameters["Redo"] = "Changed" # Force None

indexChars = "ijklmnopqrstuvwxyz"

solutionDefaults = {
    # solution parameters
    "KernelGrid0":              1,
    "KernelGrid1":              1,
    "KernelGridU":              1,
    "KernelsSerial":            True,
    # kernel parameters
    "WorkGroupOrder":           1,
    "MicroTileEdge":            4,
    "MicroTileShape":           0,
    "WorkGroupEdge":            16,
    "WorkGroupShape":           0,
    "LoopFor":                  True,
    "LoopUnroll":               16,
    "LoopTail":                 True,
    "NumLoadsParaA":            1,
    "NumLoadsParaB":            1,
    "GlobalLoadVectorWidth":    4,
    "LocalStoreVectorWidth":    4,
    "LocalLoadVectorWidth":     4,
    "GlobalStoreVectorWidth":   4,
    "LoadMacInterleave":        4,
    "SplitK":                   1,
    "Prefetch":                 True,
    "AtomicAccumulate":         False,
    "EdgeType":                 "Shift",
    }

problemTypeDefaults = {
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
  sys.exit(message)



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



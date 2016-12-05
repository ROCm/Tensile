import os.path
from collections import OrderedDict

# debug print level
# 0 - user wants no printing
# 1 - user wants basic status
# 2 - user wants debugging
globalParameters = OrderedDict()
globalParameters["Name"] = "Tensile"
globalParameters["Backend"] = "OpenCL"
globalParameters["DebugPrintLevel"] = 1
globalParameters["ScriptPath"] = os.path.dirname(os.path.realpath(__file__))
globalParameters["WorkingPath"] = os.getcwd()
globalParameters["indexChars"] = "ijklmnopqrstuvwxyz"

def printDebug( level, message):
  if globalParameters["DebugPrintLevel"] >= level:
    print message


################################################################################
# Global Parameters
################################################################################
def assignGlobalParameters( config ):
  global globalParameters
  for key in config:
    value = config[key]
    if key not in globalParameters:
      printDebug(1,"Tensile::Common::assignGlobalParameters: WARNING: Global parameter has no default: %s = %s" % ( key, value ))
    globalParameters[key] = value

  if globalParameters["DebugPrintLevel"] >= 1:
    print "Tensile::Common::assignGlobalParameters::globalParameters"
    for key in globalParameters:
      value = globalParameters[key]
      if key in config:
        if config[key] == globalParameters[key]:
          print " %16s: %8s (same)" % (key, value)
        else:
          print " %16s: %8s (overriden)" % (key, value)
      else:
        print " %16s: %8s" % (key, value)



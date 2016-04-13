import Structs

class SolutionSelectionWriter:

  def __init__(self, backend):
    self.backend = backend

  
  #############################################################################
  # write top level getSolution
  # chooses amongst devices
  #############################################################################
  def writeGetSolutionTop(self, psMap):
    functionName = "getSolutionTop"
    # source file
    s = ""
    s += "#include \"CobaltGetSolution.h\"\n"
    for deviceProfile, exactMatches in psMap.iteritems():
      s += "#include \"CobaltGetSolution_" + deviceProfile.libString() + ".h\"\n"
    s += "\n"
    s += "CobaltSolution " + functionName + "( const CobaltProblem & problem, CobaltStatus *status ) {\n"
    # if match device
    for deviceProfile, exactMatches in psMap.iteritems():
      s += "  if ( problem.deviceProfile.devices[0] == \"" + deviceProfile.devices[0].name + "\""
      for i in range(1, len(deviceProfile.devices)):
        s += " && problem.deviceProfile.devices[" + str(i) + "] == \"" + deviceProfile.devices[i] + "\""
      s += " ) {\n"
      s += "    return getSolutionFor_" + deviceProfile.libString() + "(problem, status);\n"
      s += "  }\n"
    # else doesn't match any device
    for deviceProfile, exactMatches in psMap.iteritems():
      s += "  /* doesn't match any known device; return a default */\n"
      s += "  return getSolutionFor_" + deviceProfile.libString() + "(problem, status);\n"
    s += "}\n"
    s += "\n"

    # header file
    h = ""
    h += "#ifndef COBALTGETSOLUTION_H\n"
    h += "#define COBALTGETSOLUTION_H\n"
    h += "\n"
    h += "#include \"Cobalt.h\"\n"
    h += "\n"
    h += "CobaltSolution " + functionName + "( const CobaltProblem & problem, CobaltStatus *status);\n"
    h += "\n"
    h += "#endif\n"
    h += "\n"

    return (s, h)

  
  #############################################################################
  # write device-level getSolution
  # chooses amongst exact matches
  #############################################################################
  def writeGetSolutionForDevice( self, deviceProfile, exactMatches):
    functionName = "getSolution_" + deviceProfile.libString()
    s = ""
    s += "#include \"CobaltGetSolution_" + deviceProfile.libString() + ".h\"\n"
    for exactMatch, problems in exactMatches.iteritems():
      s += "#include \"CobaltGetSolution_" + exactMatch.libString() + ".h\"\n"
    s += "\n"
    s += "CobaltSolution " + functionName + "( const CobaltProblem & problem, CobaltStatus *status ) {\n"
    
    for exactMatch, problems in exactMatches.iteritems():
      # if problem exactly matches EXACT_MATCH
      s += "  if (\n"
      s += "     problem->pimpl.getDataTypeC() == " + exactMatch.typeC.getLibString() + "\n"

      s += "  ) {\n"
      s += "    return getSolution_" + exactMatch.libString() + "( const CobaltProblem & problem, CobaltStatus *status);\n"
      s += "  }\n"
    
    s += "  *status = cobaltStatusProblemNotSupported;\n"
    s += "}\n"
    s += "\n"
    
    # header file
    h = ""
    h += "#ifndef COBALT" + functionName.upper() + "_H\n"
    h += "#define COBALT" + functionName.upper() + "_H\n"
    h += "\n"
    h += "#include \"Cobalt.h\"\n"
    h += "\n"
    h += "CobaltSolution " + functionName + "( const CobaltProblem & problem, CobaltStatus *status);\n"
    h += "\n"
    h += "#endif\n"
    h += "\n"
    return (s, h)

  
  #############################################################################
  # write exact match level getSolution
  # chooses amongst sizes and mods
  #############################################################################
  def writeGetSolutionForExactMatch(self, exactMatch, problems):
    s = ""
    h = ""
    return (s, h)

  
  #############################################################################
  # write cmake file for CobaltLib solution selection
  #############################################################################
  def writeCobaltLibCMake(self, psMap, subdirectory):
    s = "# CobaltLib.cmake\n"
    s += "\n"
    s += "include( ${CobaltLib_KernelFiles_CMAKE_DYNAMIC} )\n"
    s += "include( ${CobaltLib_SolutionFiles_CMAKE_DYNAMIC} )\n"
    s += "\n"
    s += "set( CobaltLib_SRC_GENERATED_DYNAMIC\n"
    
    for deviceProfile, exactMatches in psMap.iteritems():
      print str(deviceProfile), str(exactMatches)
      # (2) Write Device-Level Solution Selection files
      baseName = "CobaltGetSolution_" + deviceProfile.libString()
      s += "  ${CobaltLib_DIR_GENERATED}" + subdirectory + baseName + ".cpp\n"
      s += "  ${CobaltLib_DIR_GENERATED}" + subdirectory + baseName + ".h\n"

      for exactMatch, problems in exactMatches.iteritems():
        baseName = "CobaltGetSolution_" + exactMatch.libString()
      s += "  ${CobaltLib_DIR_GENERATED}" + subdirectory + baseName + ".cpp\n"
      s += "  ${CobaltLib_DIR_GENERATED}" + subdirectory + baseName + ".h\n"
    s += ")\n"
    s += "\n"
    s += "source_group(CobaltGen\\\\Backend FILES\n"
    s += "  ${CobaltLib_SRC_GENERATED_STATIC}\n"
    s += "  ${CobaltLib_SRC_GENERATED_DYNAMIC} )\n"
    s += "\n"
    return s
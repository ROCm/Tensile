from Common import *
import YAMLIO

import sys
import os
from subprocess import Popen
from shutil import copy as shutil_copy
from shutil import rmtree


################################################################################
# Main
################################################################################
def main(  config ):
  libraryLogicPath = os.path.join(globalParameters["WorkingPath"], \
      globalParameters["LibraryLogicPath"])
  pushWorkingPath(globalParameters["LibraryClientPath"])
  printStatus("Beginning")


  ##############################################################################
  # Copy Source Files
  ##############################################################################
  pushWorkingPath("source")
  filesToCopy = [
      "Client.cpp",
      "Client.h",
      "CMakeLists.txt",
      "CreateTensile.cmake"
      ]

  for f in filesToCopy:
    filename = os.path.join(globalParameters["SourcePath"], f)
    shutil_copy(
        os.path.join(globalParameters["SourcePath"], f),
        globalParameters["WorkingPath"] )
  if globalParameters["Backend"] == "OCL":
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
  # Write Generated Header
  ##############################################################################
  logicFiles = [os.path.join(libraryLogicPath, f) for f \
      in os.listdir(libraryLogicPath) \
      if os.path.isfile(os.path.join(libraryLogicPath, f))]
  print logicFiles
  functions = []
  functionNames = []
  for logicFileName in logicFiles:
    (scheduleName, problemType, solutionsForType, skinnyLogic0, skinnyLogic1, \
        diagonalLogic) = YAMLIO.readLibraryLogicForProblemType(logicFileName)
    functions.append((scheduleName, problemType))
    functionNames.append("tensile_%s_%s" % (scheduleName, problemType))

  # open file
  generated = open(os.path.join(globalParameters["WorkingPath"],
    "GeneratedHeader.h" ), "w" )
  g = ""
  g += "const unsigned int numFunctions = %u;\n" % len(functions)
  g += "char *functionNames[numFunctions] = {\n"
  for functionIdx in range(0, len(functionNames)):
    functionName = functionNames[functionIdx]

    g += "    \"%s\"%s\n" % (functionName, \
        "," if functionIdx < len(functionNames)-1 else "" )
  g += "};\n"


  # close file
  generated.write(g)
  generated.close()
  popWorkingPath() # source

  ##############################################################################
  # Run Build Script
  ##############################################################################
  # if redo=true, clobber the build directory
  if globalParameters["ForceRedoLibraryClient"]:
    rmtree(os.path.join(globalParameters["WorkingPath"], "build"), \
        ignore_errors=True)
  pushWorkingPath("build")

  # create run.bat or run.sh which builds and runs
  runScriptName = os.path.join(globalParameters["WorkingPath"], \
    "run.%s" % ("bat" if os.name == "nt" else "sh") )
  runScriptFile = open(runScriptName, "w")
  echoLine = "@echo." if os.name == "nt" else "echo"
  if os.name != "nt":
    runScriptFile.write("#!/bin/sh\n")
  runScriptFile.write("%s & echo %s & echo # Configuring CMake & echo %s\n" \
      % (echoLine, HR, HR))
  runScriptFile.write("cmake")
  if os.name == "nt":
    runScriptFile.write(" -DCMAKE_GENERATOR_PLATFORM=x64")
  runScriptFile.write(" -DTensile_CLIENT_BENCHMARK=OFF")
  runScriptFile.write(" -DTensile_LOGIC_PATH=%s" % libraryLogicPath)
  runScriptFile.write(" -DTensile_ROOT=%s" \
      % os.path.join(globalParameters["ScriptPath"], ".."))
  runScriptFile.write(" -DTensile_BACKEND=%s" \
      % globalParameters["Backend"])
  runScriptFile.write(" -DTensile_MERGE_FILES=%s" \
      % ("ON" if globalParameters["MergeFiles"] else "OFF"))
  runScriptFile.write(" -DTensile_SHORT_FILE_NAMES=%s" \
      % ("ON" if globalParameters["ShortFileNames"] else "OFF"))
  runScriptFile.write(" -DTensile_LIBRARY_PRINT_DEBUG=%s" \
      % ("ON" if globalParameters["LibraryPrintDebug"] else "OFF"))
  runScriptFile.write(" ../source\n")
  runScriptFile.write("%s & echo %s & echo # Building Library Client & echo %s\n" \
      % (echoLine, HR, HR))
  runScriptFile.write("cmake --build . --config %s%s\n" \
      % (globalParameters["CMakeBuildType"], " -- -j 8" if os.name != "nt" else "") )
  runScriptFile.write("%s & echo %s & echo # Running Library Client & echo %s\n" \
      % (echoLine, HR, HR))
  if os.name == "nt":
    runScriptFile.write(os.path.join(globalParameters["CMakeBuildType"], \
        "LibraryClient.exe") )
  else:
    runScriptFile.write("./LibraryClient")
  runScriptFile.close()
  if os.name != "nt":
    os.chmod(runScriptName, 0777)
  # wait for python to finish printing
  process = Popen(runScriptName, cwd=globalParameters["WorkingPath"])
  status = process.communicate()
  popWorkingPath() # build


  printStatus("DONE.")
  popWorkingPath()

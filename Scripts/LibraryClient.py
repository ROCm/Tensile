from Common import *

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
      "LibraryClient.cpp",
      "LibraryClient.h",
      "CreateTensile.cmake"
      ]

  shutil_copy(
      os.path.join(globalParameters["SourcePath"], "LibraryClient.cmake"),
      os.path.join(globalParameters["WorkingPath"], "CMakeLists.txt" ) )
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

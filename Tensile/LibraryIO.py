################################################################################
# Copyright 2016-2020 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import print2, printExit, printWarning, versionIsCompatible
from .SolutionStructs import Solution, ProblemSizes, ProblemType
from . import __version__
from . import Common
from . import SolutionLibrary
import os

try:
  import yaml
except ImportError:
  printExit("You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions.")

try:
  import msgpack
except ImportError:
  printExit("You must install MessagePack for Python to use Tensile (to parse config files). See https://github.com/msgpack/msgpack-python for installation instructions.")

################################################################################
# Read Benchmark Config from YAML Files
################################################################################
def readConfig( filename ):
  try:
    stream = open(filename, "r")
  except IOError:
    printExit("Cannot open file: %s" % filename )
  config = yaml.load(stream, yaml.SafeLoader)
  stream.close()
  return config

################################################################################
# Write List of Solutions to YAML File
################################################################################
def writeSolutions( filename, problemSizes, solutions ):
  # convert objects to nested dictionaries
  solutionStates = []
  for hardcoded in solutions:
    for solution in hardcoded:
      solutionState = solution.getAttributes()
      solutionState["ProblemType"] = solutionState["ProblemType"].state
      solutionState["ProblemType"]["DataType"] = \
          solutionState["ProblemType"]["DataType"].value
      solutionState["ProblemType"]["DestDataType"] = \
          solutionState["ProblemType"]["DestDataType"].value
      solutionState["ProblemType"]["ComputeDataType"] = \
          solutionState["ProblemType"]["ComputeDataType"].value
      solutionStates.append(solutionState)
  # write dictionaries
  try:
    stream = open(filename, "w")
  except IOError:
    printExit("Cannot open file: %s" % filename)
  stream.write("- MinimumRequiredVersion: %s\n" % __version__ )
  stream.write("- ProblemSizes:\n")
  for sizeRange in problemSizes.ranges:
    stream.write("  - Range: %s\n" % sizeRange)
  for problemExact in problemSizes.exacts:
    #FIXME-problem, this ignores strides:
    stream.write("  - Exact: %s\n" % str(problemExact))
  yaml.dump(solutionStates, stream, default_flow_style=None)
  stream.close()


################################################################################
# Read List of Solutions from YAML File
################################################################################
def readSolutions( filename ):
  try:
    stream = open(filename, "r")
  except IOError:
    printExit("Cannot open file: %s" % filename )
  solutionStates = yaml.load(stream, yaml.SafeLoader)
  stream.close()

  # verify
  if len(solutionStates) < 2:
    printExit("len(%s) %u < 2" % (filename, len(solutionStates)))
  versionString = solutionStates[0]["MinimumRequiredVersion"]
  if not versionIsCompatible(versionString):
    printWarning("File \"%s\" version=%s does not match current Tensile version=%s" \
        % (filename, versionString, __version__) )

  if "ProblemSizes" not in solutionStates[1]:
    printExit("%s doesn't begin with ProblemSizes" % filename)
  else:
    problemSizesConfig = solutionStates[1]["ProblemSizes"]

  solutions = []
  for i in range(2, len(solutionStates)):
    solutionState = solutionStates[i]
    solutionObject = Solution(solutionState)
    solutions.append(solutionObject)
  problemType = solutions[0]["ProblemType"]
  problemSizes = ProblemSizes(problemType, problemSizesConfig)
  return (problemSizes, solutions)

################################################################################
# Read Library Logic from YAML
################################################################################
def readLibraryLogicForSchedule( filename ):
  #print1("# Reading Library Logic: %s" % ( filename ))
  try:
    stream = open(filename, "r")
  except IOError:
    printExit("Cannot open file: %s" % filename )
  data = yaml.load(stream, yaml.SafeLoader)
  stream.close()

  # verify
  if len(data) < 6:
    printExit("len(%s) %u < 7" % (filename, len(data)))

  # parse out objects
  versionString     = data[0]["MinimumRequiredVersion"]
  scheduleName      = data[1]
  architectureName  = data[2]
  deviceNames       = data[3]
  problemTypeState  = data[4]
  solutionStates    = data[5]
  indexOrder        = data[6]
  exactLogic        = data[7]
  rangeLogic        = data[8]

  # does version match
  if not versionIsCompatible(versionString):
    printWarning("File \"%s\" version=%s does not match Tensile version=%s" \
        % (filename, versionString, __version__) )

  # unpack problemType
  problemType = ProblemType(problemTypeState)
  # unpack solutions
  solutions = []
  for i in range(0, len(solutionStates)):
    solutionState = solutionStates[i]
    if solutionState["KernelLanguage"] == "Assembly":
      solutionState["ISA"] = Common.gfxArch(architectureName)
    else:
      solutionState["ISA"] = [0, 0, 0]
    solutionObject = Solution(solutionState)
    if solutionObject["ProblemType"] != problemType:
      printExit("ProblemType of file doesn't match solution: %s != %s" \
          % (problemType, solutionObject["ProblemType"]))
    solutions.append(solutionObject)

  newLibrary = SolutionLibrary.MasterSolutionLibrary.FromOriginalState(data, solutions)

  return (scheduleName, deviceNames, problemType, solutions, indexOrder, \
      exactLogic, rangeLogic, newLibrary, architectureName)

################################################################################
# Data-specific writers
################################################################################

def configWriter(libraryFormat = "yaml"):
  return YAMLWriter() if libraryFormat == "yaml" else MessagePackWriter()

class Writer:
  def write(self, filename, data):
    """ Write data to a given file. """
    pass

  def writeLibraryLogicForSchedule(self, filePath, schedulePrefix, architectureName, \
      deviceNames, logicTuple):
    """ Encode library logic """
    pass

  def _getLibraryLogicForSchedule(self, schedulePrefix, architectureName, deviceNames, \
      logicTuple):
    problemType   = logicTuple[0]
    solutions     = logicTuple[1]
    indexOrder    = logicTuple[2]
    exactLogic    = logicTuple[3]
    rangeLogic    = logicTuple[4]

    tileSelection = False
    if len(logicTuple) > 5 and logicTuple[5]:
      tileSelection = True

    data = []
    # Tensile version
    data.append({"MinimumRequiredVersion":__version__})
    # schedule name
    data.append(schedulePrefix)     # change from Tensile to vega10
    data.append(architectureName)
    # schedule device names
    data.append(deviceNames)
    # problem type
    problemTypeState = problemType.state
    problemTypeState["DataType"] = \
        problemTypeState["DataType"].value
    problemTypeState["DestDataType"] = \
        problemTypeState["DestDataType"].value
    problemTypeState["ComputeDataType"] = \
        problemTypeState["ComputeDataType"].value
    data.append(problemTypeState)
    # solutions
    solutionList = []
    for solution in solutions:
      solutionState = solution.getAttributes()
      solutionState["ProblemType"] = solutionState["ProblemType"].state
      solutionState["ProblemType"]["DataType"] = \
          solutionState["ProblemType"]["DataType"].value
      solutionState["ProblemType"]["DestDataType"] = \
          solutionState["ProblemType"]["DestDataType"].value
      solutionState["ProblemType"]["ComputeDataType"] = \
          solutionState["ProblemType"]["ComputeDataType"].value
      solutionList.append(solutionState)

    if tileSelection:
      tileSolutions = logicTuple[5]
      for solution in tileSolutions:
        solutionState = solution.getAttributes()
        solutionState["ProblemType"] = solutionState["ProblemType"].state
        solutionState["ProblemType"]["DataType"] = \
            solutionState["ProblemType"]["DataType"].value
        solutionState["ProblemType"]["DestDataType"] = \
            solutionState["ProblemType"]["DestDataType"].value
        solutionState["ProblemType"]["ComputeDataType"] = \
            solutionState["ProblemType"]["ComputeDataType"].value
        solutionList.append(solutionState)

    data.append(solutionList)
    # index order
    data.append(indexOrder)

    # exactLogic
    exactLogicList = []
    for key in exactLogic:
      exactLogicList.append([list(key), exactLogic[key]])
    data.append(exactLogicList)

    # rangeLogic
    data.append(rangeLogic)

    if tileSelection:
      tileSelectionLogic = {}
      tileSelectionIndices = logicTuple[6]
      tileSelectionLogic["TileSelectionIndices"] = tileSelectionIndices
      data.append(tileSelectionLogic)

    return data

class YAMLWriter(Writer):
  def write(self, filename, data):
    with open(filename, 'w') as f:
        yaml.dump(data, f, explicit_start=True, explicit_end=True, default_flow_style=None)

  def writeLibraryLogicForSchedule(self, filePath, schedulePrefix, architectureName, \
      deviceNames, logicTuple):
    problemType = logicTuple[0]
    filename = os.path.join(filePath, "%s_%s.yaml" \
        % (schedulePrefix, str(problemType)))
    print2("# writeLogic( %s )" % ( filename ))

    data = self._getLibraryLogicForSchedule(schedulePrefix, architectureName, \
      deviceNames, logicTuple)

    try:
      stream = open(filename, "w")
      yaml.dump(data, stream, default_flow_style=None)
      stream.close()
    except IOError:
      printExit("Cannot open file: %s" % filename)

class MessagePackWriter(Writer):
  def write(self, filename, data):
    with open(filename, 'wb') as f:
        msgpack.pack(data, f)

  def writeLibraryLogicForSchedule(self, filePath, schedulePrefix, architectureName, \
      deviceNames, logicTuple):
    problemType = logicTuple[0]
    filename = os.path.join(filePath, "%s_%s.dat" \
        % (schedulePrefix, str(problemType)))
    print2("# writeLogic( %s )" % ( filename ))

    data = self._getLibraryLogicForSchedule(schedulePrefix, architectureName, \
      deviceNames, logicTuple)

    try:
      stream = open(filename, "wb")
      msgpack.pack(data, stream)
      stream.close()
    except IOError:
      printExit("Cannot open file: %s" % filename)

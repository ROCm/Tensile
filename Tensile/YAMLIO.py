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
from Common import globalParameters, print1, print2, printExit, printWarning
from SolutionStructs import Solution, ProblemSizes, ProblemType
from __init__ import __version__

import os
try:
  import yaml
except ImportError:
  printExit("You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions.")


################################################################################
# Read Benchmark Config from YAML Files
################################################################################
def readConfig( filename ):
  #print "Tensile::YAMLIO::readConfig( %s )" % ( filename )
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
  #print "Tensile::YAMLIO::writeSolutions( %s, %u )" % ( filename, len(solutions) )
  # convert objects to nested dictionaries
  solutionStates = []
  for hardcoded in solutions:
    for solution in hardcoded:
      solutionState = solution.state
      solutionState["ProblemType"] = solutionState["ProblemType"].state
      solutionState["ProblemType"]["DataType"] = \
          solutionState["ProblemType"]["DataType"].value
      solutionStates.append(solutionState)
  # write dictionaries
  try:
    stream = open(filename, "w")
  except IOError:
    printExit("Cannot open file: %s" % filename)
  stream.write("- %s\n" % __version__ )
  stream.write("- ProblemSizes: %s\n" % str(problemSizes))
  yaml.dump(solutionStates, stream, default_flow_style=False)
  stream.close()


################################################################################
# Read List of Solutions from YAML File
################################################################################
def readSolutions( filename ):
  #print "Tensile::YAMLIO::readSolutions( %s )" % ( filename )
  try:
    stream = open(filename, "r")
  except IOError:
    printExit("Cannot open file: %s" % filename )
  solutionStates = yaml.load(stream, yaml.SafeLoader)
  stream.close()

  # verify
  if len(solutionStates) < 2:
    printExit("len(%s) %u < 2" % (filename, len(solutionStates)))
  version = solutionStates[0]
  if version != __version__:
    printWarning("File \"%s\" version=%s does not match Tensile version=%s" \
        % (filename, version, __version__) )

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
  #print problemType
  #print problemType.state
  #print problemSizesConfig
  #print problemSizes

  return (problemSizes, solutions)


################################################################################
# Encode Library Logic to YAML
# 1 yaml per problem type
# problemType, skinny0, skinny1, diagonal
################################################################################
def writeLibraryLogicForSchedule( filePath, schedulePrefix, deviceNames, \
    logicTuple):
  problemType   = logicTuple[0]
  solutions     = logicTuple[1]
  indexOrder    = logicTuple[2]
  logic         = logicTuple[3]
  filename = os.path.join(filePath, "%s_%s.yaml" \
      % (schedulePrefix, str(problemType)))
  print2("# writeLogic( %s )" % ( filename ))

  data = []
  # Tensile version
  data.append(__version__)
  # schedule name
  data.append(schedulePrefix)
  # schedule device names
  data.append(deviceNames)
  # problem type
  problemTypeState = problemType.state
  problemTypeState["DataType"] = \
      problemTypeState["DataType"].value
  data.append(problemTypeState)
  # solutions
  solutionList = []
  for solution in solutions:
    solutionState = solution.state
    solutionState["ProblemType"] = solutionState["ProblemType"].state
    solutionState["ProblemType"]["DataType"] = \
        solutionState["ProblemType"]["DataType"].value
    solutionList.append(solutionState)
  data.append(solutionList)
  # index order
  data.append(indexOrder)
  # logic
  data.append(logic)

  # open & write file
  try:
    stream = open(filename, "w")
    #yaml.dump(data, stream, default_flow_style=False)
    yaml.dump(data, stream)
    stream.close()
  except IOError:
    printExit("Cannot open file: %s" % filename)

def readLibraryLogicForSchedule( filename ):
  print1("# Reading Library Logic: %s" % ( filename ))
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
  version           = data[0]
  scheduleName      = data[1]
  deviceNames       = data[2]
  problemTypeState  = data[3]
  solutionStates    = data[4]
  indexOrder        = data[5]
  logic             = data[6]

  # does version match
  if version != __version__:
    printWarning("File \"%s\" version=%s does not match Tensile version=%s" \
        % (filename, version, __version__) )

  # unpack problemType
  problemType = ProblemType(problemTypeState)
  # unpack solutions
  solutions = []
  for i in range(0, len(solutionStates)):
    solutionState = solutionStates[i]
    solutionObject = Solution(solutionState)
    if solutionObject["ProblemType"] != problemType:
      printExit("ProblemType of file doesn't match solution: %s != %s" \
          % (problemType, solutionObject["ProblemType"]))
    solutions.append(solutionObject)

  return (scheduleName, deviceNames, problemType, solutions, indexOrder, logic )

from Common import globalParameters, print1, print2, printExit
from SolutionStructs import Solution, ProblemSizes, ProblemType

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
  if "ProblemSizes" not in solutionStates[0]:
    printExit("%s doesn't begin with ProblemSizes" % filename)

  solutions = []
  for i in range(1, len(solutionStates)):
    solutionState = solutionStates[i]
    solutionObject = Solution(solutionState)
    solutions.append(solutionObject)
  problemType = solutions[0]["ProblemType"]
  #print problemType
  #print problemType.state
  problemSizesConfig = solutionStates[0]["ProblemSizes"]
  #print problemSizesConfig
  problemSizes = ProblemSizes(problemType, problemSizesConfig)
  #print problemSizes

  return (problemSizes, solutions)


################################################################################
# Encode Library Backend to YAML
# 1 yaml per problem type
# problemType, skinny0, skinny1, diagonal
################################################################################
def writeLibraryLogicForProblemType( filePath, schedulePrefix, logic):
  problemType   = logic[0]
  solutions     = logic[1]
  skinnyLogic0  = logic[2]
  skinnyLogic1  = logic[3]
  diagonalLogic = logic[4]
  filename = os.path.join(filePath, "%s_%s.yaml" \
      % (schedulePrefix, str(problemType)))
  print2("# writeLogic( %s )" % ( filename ))

  # open file
  try:
    stream = open(filename, "w")
  except IOError:
    printExit("Cannot open file: %s" % filename)

  # write problem type
  problemTypeState = problemType.state
  problemTypeState["DataType"] = \
      problemTypeState["DataType"].value
  data = [ globalParameters["Name"], problemTypeState, [], [], [], [] ]
  for solution in solutions:
    solutionState = solution.state
    solutionState["ProblemType"] = solutionState["ProblemType"].state
    solutionState["ProblemType"]["DataType"] = \
        solutionState["ProblemType"]["DataType"].value
    data[2].append(solutionState)
  for rule in skinnyLogic0:
    data[3].append(rule)
  for rule in skinnyLogic1:
    data[4].append(rule)
  for rule in diagonalLogic:
    data[5].append(rule)

  #stream.write(data)
  yaml.dump(data, stream, default_flow_style=False)
  stream.close()


def readLibraryLogicForProblemType( filename ):
  print1("# Reading Library Logic: %s" % ( filename ))
  try:
    stream = open(filename, "r")
  except IOError:
    printExit("Cannot open file: %s" % filename )
  data = yaml.load(stream, yaml.SafeLoader)
  stream.close()

  # verify
  if len(data) < 6:
    printExit("len(%s) %u < 6" % (filename, len(solutionStates)))

  # parse out objects
  scheduleName = data[0]
  problemTypeState = data[1]
  solutionStates = data[2]
  skinnyLogic0 = data[3]
  skinnyLogic1 = data[4]
  diagonalLogic = data[5]

  solutions = []
  problemType = ProblemType(problemTypeState)
  for i in range(0, len(solutionStates)):
    solutionState = solutionStates[i]
    solutionObject = Solution(solutionState)
    if solutionObject["ProblemType"] != problemType:
      printExit("ProblemType of file doesn't match solution: %s != %s" \
          % (problemType, solutionObject["ProblemType"]))
    solutions.append(solutionObject)

  return (scheduleName, problemType, solutions, skinnyLogic0, skinnyLogic1, \
      diagonalLogic)

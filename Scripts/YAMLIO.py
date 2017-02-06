from Common import *
import sys
try:
  import yaml
except ImportError:
  printExit("You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions.")

from Structs import *

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
def writeLibraryConfigForProblemType( filePath, schedulePrefix, \
    logic):
  problemType   = logic[0]
  solutions     = logic[1]
  skinnyLogic0  = logic[2]
  skinnyLogic1  = logic[3]
  diagonalLogic = logic[4]
  filename = os.path.join(filePath, "%s_%s.yaml" % (schedulePrefix, str(problemType)))
  print "writeLogic( %s )" % ( filename )

  # open file
  try:
    stream = open(filename, "w")
  except IOError:
    printExit("Cannot open file: %s" % filename)

  # write problem type
  problemTypeState = problemType.state
  problemTypeState["DataType"] = \
      problemTypeState["DataType"].value
  data = [ problemTypeState, [], [], [], [] ]
  for solution in solutions:
    solutionState = solution.state
    solutionState["ProblemType"] = solutionState["ProblemType"].state
    solutionState["ProblemType"]["DataType"] = \
        solutionState["ProblemType"]["DataType"].value
    data[1].append(solutionState)
  for rule in skinnyLogic0:
    data[2].append(rule)
  for rule in skinnyLogic1:
    data[3].append(rule)
  for rule in diagonalLogic:
    data[4].append(rule)

  #stream.write(data)
  yaml.dump(data, stream, default_flow_style=False)
  stream.close()



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
      solutionState["ProblemType"]["DataType"] = solutionState["ProblemType"]["DataType"].value
      solutionStates.append(solutionState)
  # write dictionaries
  try:
    stream = open(filename, "w")
  except IOError:
    printExit("Cannot open file: %s" % filename)
  stream.write("- ProblemSizes: %s\n" % str(problemSizes))
  yaml.dump(solutionStates, stream, default_flow_style=False)
  stream.close()
  
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

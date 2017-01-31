import os
import os.path
from Common import *
from Structs import *
import YAMLIO

def analyzeProblemType( problemTypeTuple ):
  problemType = problemTypeTuple[0]
  problemSizes = problemTypeTuple[1]
  dataFileName = problemTypeTuple[2]
  solutionsFileName = problemTypeTuple[3]
  print HR
  print "# %s" % problemType
  #print "#  %s" % dataFileName
  #print "#  %s" % solutionsFileName

  (problemSizes, solutions) = YAMLIO.readSolutions(solutionsFileName)
  print "# ProblemSizes: %s" % problemSizes
  solutionMinNaming = Solution.getMinNaming(solutions)
  print "# Solutions:"
  solutionIdx = 0
  for solution in solutions:
    print "#  (%u) %s" % (solutionIdx, Solution.getNameMin(solution, solutionMinNaming))
    solutionIdx += 1
  print HR

  

def main(  config ):
  pushWorkingPath(globalParameters["AnalyzePath"])
  print ""
  print HR
  print "# Analysing data in %s." % config["DataPath"]
  print HR
  print ""

  ##############################################################################
  # Determine Which Problem Types
  ##############################################################################
  problemTypeTuples = []
  for fileName in os.listdir(config["DataPath"]):
    if ".csv" in fileName:
      fileBase = os.path.splitext(os.path.join(config["DataPath"], fileName))[0]
      dataFileName = fileBase + ".csv"
      solutionsFileName = fileBase + ".yaml"
      if not os.path.exists(dataFileName):
        printExit("%s doesn't exist for %s" % (dataFileName, fileBase) )
      if not os.path.exists(solutionsFileName):
        printExit("%s doesn't exist for %s" % (solutionsFileName, fileBase) )
      (problemSizes, solutions) = YAMLIO.readSolutions(solutionsFileName)
      if len(solutions) == 0:
        printExit("%s doesn't contains any solutions." % (solutionsFileName) )
      problemType = solutions[0]["ProblemType"]
      problemTypeTuple = ( problemType, problemSizes, \
          dataFileName, solutionsFileName)
      if problemTypeTuple not in problemTypeTuples:
        print "adding tuple"
        problemTypeTuples.append(problemTypeTuple)
      else:
        print "not adding tuple"

  # Run Analysis
  for problemTypeTuple in problemTypeTuples:
    analyzeProblemType( problemTypeTuple )




  printStatus("DONE.")
  popWorkingPath()

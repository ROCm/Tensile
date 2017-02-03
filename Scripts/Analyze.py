import os
import os.path
import array
import csv

from Common import *
from Structs import *
import YAMLIO

################################################################################
# Analyze Problem Type
################################################################################
def analyzeProblemType( problemTypeTuple, analysisParameters ):
  problemType = problemTypeTuple[0]
  problemSizes = problemTypeTuple[1]
  dataFileName = problemTypeTuple[2]
  solutionsFileName = problemTypeTuple[3]
  print HR
  print "# %s" % problemType
  #print "#  %s" % dataFileName
  #print "#  %s" % solutionsFileName

  # Read Solutions
  (problemSizes, solutions) = YAMLIO.readSolutions(solutionsFileName)
  print "# ProblemSizes: %s" % problemSizes
  solutionMinNaming = Solution.getMinNaming(solutions)
  print "# Solutions:"
  solutionIdx = 0
  for solution in solutions:
    print "#  (%u) %s" % (solutionIdx, Solution.getNameMin(solution, solutionMinNaming))
    solutionIdx += 1
  print HR

  # Read Data
  numProblemSizes = problemSizes.numProblemSizes
  data = BenchmarkData(problemType, problemSizes, solutions, analysisParameters)
  data.populateFromCSV(dataFileName)
  #data.printData()
  diagonalSolutions = data.getFastestSolutionsAlongDiagonal()

################################################################################
# BenchmarkData
################################################################################

################################################################################
# BenchmarkData
################################################################################
class BenchmarkData:
  # numProblemSizes[i] = num0, num1, ...

  def __init__(self, problemType, problemSizes, solutions, analysisParameters):
    self.problemType = problemType
    self.problemSizes = problemSizes
    self.analysisParameters = analysisParameters
    print "ProblemSizes: %s" % self.problemSizes
    # TODO verify that data is symmetric for diagonal
    #if self.problemSizes[self.problemType["Index0"]] \
    #    != self.problemSizes[self.problemType["Index1"]]:
    #  printExit("d0 / d1 must be symmetric for analysis.")
    self.numProblemSizes = problemSizes.numProblemSizes # native order
    print "NumProblemSizes: %s" % self.numProblemSizes
    self.numIndices = len(self.numProblemSizes)
    self.solutions = solutions
    self.numSolutions = len(self.solutions)
    self.solutionMinNaming = Solution.getMinNaming(solutions)
    self.solutionNames = []
    for solution in self.solutions:
      self.solutionNames.append(Solution.getNameMin(solution, \
          self.solutionMinNaming))

    # special indices
    self.idx0 = self.problemType["Index0"]
    self.idx1 = self.problemType["Index1"]
    self.idxU = self.problemType["IndexUnroll"]

    # total size of data array
    self.totalProblems = 1
    for numProblems in self.numProblemSizes:
      self.totalProblems *= numProblems
    self.totalSize = self.totalProblems * self.numSolutions
    print "TotalProblems: %u" % self.totalProblems
    print "TotalSolutions: %u" % self.numSolutions
    print "TotalSize: %u" % self.totalSize
    self.data = array.array('f', [0]*self.totalSize)

    # map problem sizes -> index
    self.problemSizeToIndex = []
    self.problemIndexToSize = []
    sizedIdx = 0
    mappedIdx = 0
    for i in range(0, self.numIndices):
      self.problemSizeToIndex.append({})
      self.problemIndexToSize.append([])
      if self.problemSizes.indexIsSized[i]:
        index = self.problemSizes.indicesSized[sizedIdx]
        sizedIdx += 1
      else:
        index = self.problemSizes.indicesSized[ \
          self.problemSizes.indicesMapped[mappedIdx]]
        mappedIdx += 1
      currentSize = index[0]
      currentStride = index[1]
      idx = 0
      while currentSize <= index[3]:
        self.problemSizeToIndex[i][currentSize] = idx
        self.problemIndexToSize[i].append(currentSize)
        currentSize += currentStride
        currentStride += index[2]
        idx += 1
    #print "S->I %s" % self.problemSizeToIndex
    #print "I->S %s" % self.problemIndexToSize



  ##############################################################################
  # Read In CSV
  def populateFromCSV(self, dataFileName):

    # open file
    try:
      dataFile = open(dataFileName, "r")
    except IOError:
      printExit("Can't open \"%s\" to get data" % dataFileName )

    # column indices
    csvFile = csv.reader(dataFile)
    problemSizeStartIdx = 1
    totalSizeIdx = problemSizeStartIdx + self.numIndices
    solutionStartIdx = totalSizeIdx + 1
    rowLength = solutionStartIdx + self.numSolutions

    # iterate over rows
    rowIdx = 0
    for row in csvFile:
      rowIdx+=1
      if rowIdx == 1:
        continue
      else:
        if len(row) < rowLength:
          printWarning("CSV File %s row %u doesn't have %u elements; ignoring remainer of file." \
              % (dataFileName, rowIdx, rowLength) )
          break

        # get problem size
        problemSize = []
        for i in range(problemSizeStartIdx, totalSizeIdx):
          problemSize.append(int(row[i]))
        problemIndices = []
        for i in range(0, self.numIndices):
          problemIndices.append(self.problemSizeToIndex[i][problemSize[i]])
        serialIdx = self.indicesToSerial(0, problemIndices)
        #print "%s -> %s -> %u" % (problemSize, problemIndices, serialIdx)

        # total size
        totalFlops = float(row[totalSizeIdx])

        # data
        solutionIdx = 0
        for i in range(solutionStartIdx, rowLength):
          gflops = float(row[i])
          self.data[serialIdx+solutionIdx] = gflops
          solutionIdx += 1
    if rowIdx < 2:
      printExit("CSV File %s only has %u row(s); prior benchmark must not have run long enough to produce data." \
          % (dataFileName, rowIdx) )


  ##############################################################################
  # Get Fastest Solutions Along Diagonal (d0=d1) for largest sizes
  ##############################################################################
  def getFastestSolutionsAlongDiagonal(self):
    print "Determining fastest solution along diagonal"
    # abstract to multidimensions
    # what is the diagonal
    dilation = self.analysisParameters["Dilation"]
    threshold = self.analysisParameters["Threshold"]
    numProblems0 = self.numProblemSizes[self.idx0]
    print "NumProblemsDiag: %u" % numProblems0

    ############################################################################
    # determine winner at largest size
    problemIndices = []
    for numProblemsForIndex in self.numProblemSizes:
      problemIndices.append(numProblemsForIndex-1)
    solutionNumWins = [0]*self.numSolutions
    solutionGFlops = [0]*self.numSolutions
    for problemSizeIdx in range(numProblems0-dilation*2, numProblems0):
      problemIndices[self.idx0] = problemSizeIdx
      problemIndices[self.idx1] = problemSizeIdx
      problemIdx = self.indicesToSerial(0, problemIndices)
      winnerIdx = -1
      winnerGFlops = -1
      for solutionIdx in range(0, self.numSolutions):
        solutionSerialIdx = problemIdx + solutionIdx
        solutionTmpGFlops = self.data[solutionSerialIdx]
        if solutionTmpGFlops > winnerGFlops:
          winnerIdx = solutionIdx
          winnerGFlops = solutionTmpGFlops
        #print "updated winner: ", winnerIdx
      #print winnerIdx
      solutionNumWins[winnerIdx] += 1
      if winnerGFlops > solutionGFlops[winnerIdx]:
        solutionGFlops[winnerIdx] = winnerGFlops
    largestWinnerIdx = -1
    largestWinnerNumWins = -1
    largestWinnerGFlops = -1
    #print "FastestWins:"
    for i in range(0, self.numSolutions):
      #print "sol[%u] = %u wins @ %.0f GFlops" \
      #    % (i, solutionNumWins[i], solutionGFlops[i])
      if solutionNumWins[i] > largestWinnerNumWins:
        largestWinnerIdx = i
        largestWinnerNumWins = solutionNumWins[i]
        largestWinnerGFlops = solutionGFlops[i]
    print "Winner at Largest Problem: S[%u] @ %.0f GFlops with %u/%u wins" \
        % (largestWinnerIdx, largestWinnerGFlops, largestWinnerNumWins, \
        dilation*2)
    problemIndices[self.idx0] = numProblems0-1
    problemIndices[self.idx1] = numProblems0-1
    largestWinnerAtLargestProblemIdx = self.indicesToSerial(largestWinnerIdx, \
        problemIndices)
    largestWinnerGFlopsAtLargestSize = \
        self.data[largestWinnerAtLargestProblemIdx]

    ############################################################################
    # Diagonal Rule
    # [ solutionIdx, minSizeThresholdIdx, gflops at minSize, maxGFlops ]
    numRules = 1
    diagonalRules = [ [largestWinnerIdx, numProblems0-1, \
        largestWinnerGFlopsAtLargestSize, largestWinnerGFlops] ]

    ############################################################################
    # For largest to smallest, determine fastest solution
    for problemSizeIdx in range(numProblems0-2, -1, -1):
      problemIndices[self.idx0] = problemSizeIdx
      problemIndices[self.idx1] = problemSizeIdx
      problemIdx = self.indicesToSerial(0, problemIndices)

      # current rule winner performance at this problemSizeIdx
      ruleWinnerIdx = diagonalRules[-1][0]
      ruleWinnerGFlopsForSize = self.data[problemIdx + ruleWinnerIdx]

      #determine fastest at this problemSizeIdx
      winnerForSizeIdx = -1
      winnerForSizeGFlops = -1
      for solutionIdx in range(0, self.numSolutions):
        solutionSerialIdx = problemIdx + solutionIdx
        solutionGFlops = self.data[solutionSerialIdx]
        if solutionGFlops > winnerForSizeGFlops:
          #print "%f > %f" % (solutionGFlops, winnerGFlops)
          winnerForSizeIdx = solutionIdx
          winnerForSizeGFlops = solutionGFlops
      problemSize0 = self.problemIndexToSize[self.idx0][problemSizeIdx]
      problemSizeIdxU = problemIndices[self.idxU]
      #print "%4ux%4ux%4u: S[%u] @%5.0f GFlops" % (problemSize0, problemSize0, \
      #    self.problemIndexToSize[self.idxU][problemSizeIdxU], \
      #    winnerForSizeIdx, winnerForSizeGFlops)

      # ruleWinner also wins at this problem size (at least by threshold)
      if winnerForSizeIdx == ruleWinnerIdx \
          or ruleWinnerGFlopsForSize > (1-threshold)*winnerForSizeGFlops:
        # just update rule
        diagonalRules[numRules-1][1] = problemSizeIdx
        diagonalRules[numRules-1][2] = ruleWinnerGFlopsForSize
        diagonalRules[numRules-1][3] = max(diagonalRules[numRules-1][3], \
            ruleWinnerGFlopsForSize)

      # we have a new candidate winner
      # only keep it if don't revert back to ruleWinner over next Dilation
      else:

        # check if we don't revert back to ruleWinner over next Dilation probs
        revert = False
        for dilationIdx in range(problemSizeIdx-1, \
            problemSizeIdx-dilation, -1):
          ruleWinnerGFlopsForDilation = self.data[dilationIdx \
              + ruleWinnerIdx]
          #determine fastest at this problemSizeIdx
          winnerForDilationIdx = -1
          winnerForDilationGFlops = -1
          for solutionIdx in range(0, self.numSolutions):
            solutionSerialIdx = problemIdx + solutionIdx
            solutionGFlops = self.data[solutionSerialIdx]
            if solutionGFlops > winnerForDilationGFlops:
              #print "%f > %f" % (solutionGFlops, winnerGFlops)
              winnerForDilationIdx = solutionIdx
              winnerForDilationGFlops = solutionGFlops
          # ruleWinner also wins at dilation size (at least by threshold)
          if winnerForDilationIdx == ruleWinnerIdx \
              or ruleWinnerGFlopsForDilation \
              > (1-threshold)*winnerForSizeGFlops:
            # yes, within Dilation, we've returned to same winner
            revert = True
            # so update rule for this size
            diagonalRules[numRules-1][1] = dilationIdx
            diagonalRules[numRules-1][2] = winnerForDilationGFlops
            diagonalRules[numRules-1][3] = max(diagonalRules[numRules-1][3], \
                winnerForSizeGflops)
            # resume outer loop after dilation
            problemSizeIdx = dilationIdx
            break
          else:
            # different winner at this dilation size
            # don't need to do anything
            pass

        # if we never revert to rule during dilation, create new rule
        if not revert:
          # [ solutionIdx, minSizeThresholdIdx, gflops at minSize, maxGFlops ]
          newRule = [ winnerForSizeIdx, problemSizeIdx, \
              winnerForSizeGFlops, winnerForSizeGFlops]
          diagonalRules.append(newRule)
          numRules += 1
          #print "Added new rule: %s" % newRule

    print "Diagonal Rules:"
    for rule in diagonalRules:
      # [ solutionIdx, minSizeThresholdIdx, gflops at minSize, maxGFlops ]
      print "  d0,1 >=%5u: S[%u] @ %5.0f GFlops is %s" \
          % (self.problemIndexToSize[self.idx0][rule[1]], rule[0], rule[3], \
          self.solutionNames[rule[0]])
    self.diagonalRules = diagonalRules
    #end diagonal rules


  ##############################################################################
  # Print Data
  ##############################################################################
  def printData(self):
    print "serial; idxD0, idxD1, idxDU, idxOthers; sizeD0, sizeD1, sizeDU, sizeOthers; sol0, sol1, sol2, ..."
    indices = [0]*self.numIndices
    for serial in range(0, self.totalProblems):
      s = "[%4u] [%2u" % (serial, indices[0])
      for i in range(1, self.numIndices):
        s += ", %2u" % indices[i]
      s += "] [%4u" % self.problemIndexToSize[0][indices[0]]
      for i in range(1, self.numIndices):
        s += ", %4u" % self.problemIndexToSize[i][indices[i]]
      s += "]: %9.3f" % self.data[serial*self.numSolutions+0]
      for i in range(1, self.numSolutions):
        s += ", %9.3f" % self.data[serial*self.numSolutions+i]
      print s
      indices[0] += 1
      for i in range(1, self.numIndices):
        if indices[i-1] >= self.numProblemSizes[i-1]:
          indices[i-1] = 0
          indices[i] += 1

  ##############################################################################
  # Get Item
  ##############################################################################
  def __getitem__(self, indexTuple):
    indices = indexTuple[0] # in analysis order
    solutionIdx = indexTuple[1]
    serial = self.indicesToSerial(solutionIdx, indices)
    return self.data[serial]

  ##############################################################################
  # Get Item
  ##############################################################################
  def __setitem__(self, indexTuple, value):
    indices = indexTuple[0] # in analysis order
    solutionIdx = indexTuple[1]
    serial = self.indicesToSerial(solutionIdx, indices )
    self.data[serial] = value

  ##############################################################################
  # Indices -> Serial
  ##############################################################################
  def indicesToSerial(self, solutionIdx, indices ):
    serial = 0
    stride = 1
    serial += solutionIdx * stride
    stride *= self.numSolutions
    for i in range(0, self.numIndices):
      serial += indices[i] * stride
      stride *= self.numProblemSizes[i]
    return serial


################################################################################
# Main
################################################################################
def main(  config ):
  print config
  print defaultAnalysisParameters
  pushWorkingPath(globalParameters["AnalyzePath"])

  # Assign Defaults
  analysisParameters = {}
  for parameter in defaultAnalysisParameters:
    assignParameterWithDefault(analysisParameters, parameter, config, \
        defaultAnalysisParameters)

  print ""
  print HR
  print "# Analysing data in %s." % config["DataPath"]
  for parameter in analysisParameters:
    print "#   %s: %s" % (parameter, analysisParameters[parameter])
  print HR
  print ""



  ##############################################################################
  # Determine Which Problem Types
  ##############################################################################
  problemTypeTuples = []
  for fileName in os.listdir(config["DataPath"]):
    if os.path.splitext(fileName)[1] == ".csv":
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
        problemTypeTuples.append(problemTypeTuple)

  # Run Analysis
  for problemTypeTuple in problemTypeTuples:
    analyzeProblemType( problemTypeTuple, analysisParameters )

  printStatus("DONE.")
  popWorkingPath()

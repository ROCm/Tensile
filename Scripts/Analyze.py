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
def analyzeProblemType( problemTypeTuple ):
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
  print numProblemSizes
  data = BenchmarkData(problemType, problemSizes, solutions)
  data.populateFromCSV(dataFileName)
  data.printData()


################################################################################
# BenchmarkData
################################################################################
class BenchmarkData:
  # indexNativeToOrdered[i] = d0, d1, du, others for analysis and solution selection
  # indicesNative[i] = 0, 1, 2, 3... # superfluous
  # everything else in native order and we use indexNativeToOrdered to address into them

  # numProblemSizes[i] = num0, num1, ...

  def __init__(self, problemType, problemSizes, solutions):
    self.problemType = problemType
    self.problemSizes = problemSizes
    self.numProblemSizes = problemSizes.numProblemSizes # native order
    self.numIndices = len(self.numProblemSizes)
    self.solutions = solutions
    self.numSolutions = len(self.solutions)

    # indexNativeToOrdered
    self.indexNativeToOrdered = [] # [d0, d1, dU, others]
    self.indexNativeToOrdered.append(self.problemType["Index0"])
    self.indexNativeToOrdered.append(self.problemType["Index1"])
    self.indexNativeToOrdered.append(self.problemType["IndexUnroll"])

    self.indexOrderedToNative = [0]*self.numIndices # [native idx of d0, ]
    self.indexOrderedToNative[self.problemType["Index0"]] = 0
    self.indexOrderedToNative[self.problemType["Index1"]] = 1
    self.indexOrderedToNative[self.problemType["IndexUnroll"]] = 2
    for i in range(0, self.numIndices):
      if i not in self.indexNativeToOrdered:
        self.indexNativeToOrdered.append(i)
        self.indexOrderedToNative[i] = len(self.indexNativeToOrdered)-1
    print "IndexNativeToOrdered: %s" % self.indexNativeToOrdered
    print "IndexOrderedToNative: %s" % self.indexOrderedToNative

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

    print "S->I %s" % self.problemSizeToIndex
    print "I->S %s" % self.problemIndexToSize

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
        serialIdx = self.indicesNativeToSerial(problemIndices, 0)
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
  # Print Data
  ##############################################################################
  def printData(self):
    print "serial; idxD0, idxD1, idxDU, idxOthers; sizeD0, sizeD1, sizeDU, sizeOthers; sol0, sol1, sol2, ..."
    indices = [0]*self.numIndices
    for serial in range(0, self.totalProblems):
      s = "[%4u] [%2u" % (serial, indices[0])
      for i in range(1, self.numIndices):
        s += ", %2u" % indices[i]
      s += "] [%4u" % self.problemIndexToSize[i][indices[i]]
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
    serial = self.indicesToSerial(indices, solutionIdx)
    return self.data[serial]

  ##############################################################################
  # Get Item
  ##############################################################################
  def __setitem__(self, indexTuple, value):
    indices = indexTuple[0] # in analysis order
    solutionIdx = indexTuple[1]
    serial = self.indicesToSerial(indices, solutionIdx)
    self.data[serial] = value

  ##############################################################################
  # Indices -> Serial
  ##############################################################################
  def indicesNativeToSerial(self, indices, solutionIdx):
    serial = 0
    stride = 1
    serial += solutionIdx * stride
    stride *= self.numSolutions
    for i in range(0, self.numIndices):
      serial += indices[i] * stride
      stride *= self.numProblemSizes[i]
    return serial
  def indicesOrderedToSerial(self, indices, solutionIdx):
    serial = 0
    stride = 1
    serial += solutionIdx * stride
    stride *= self.numSolutions
    for i in range(0, self.numIndices):
      serial += indices[i] * stride
      stride *= self.numProblemSizes[self.indexOrderedToNative[i]]
    return serial


################################################################################
# Main
################################################################################
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
    analyzeProblemType( problemTypeTuple )

  printStatus("DONE.")
  popWorkingPath()

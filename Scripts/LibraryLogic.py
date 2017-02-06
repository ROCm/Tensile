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

  # Read Data From CSV
  numProblemSizes = problemSizes.numProblemSizes
  data = BenchmarkDataAnalyzer(problemType, problemSizes, solutions, \
      analysisParameters)
  data.populateFromCSV(dataFileName)

  ##############################################################################
  # Determine Solutions Along Diagonal
  # roughly same splitting regardless of sizeU
  problemIndices = []
  for numProblemsForIndex in data.numProblemSizes:
    problemIndices.append(numProblemsForIndex-1)
  diagonalRules = data.getFastestSolutionsAlongDiagonal(problemIndices)
  if True:
    print "Diagonal Rules:"
    for rule in diagonalRules:
      string = "  if freeSize >=%4u" % data.problemIndexToSize[0][rule[1][0]]
      for i in range(1, data.numIndices):
        string += "x%4u" % data.problemIndexToSize[i][rule[1][i]]
      string += " return S[%u] @ %5.0f-%5.0f>%5.0f GFlops is %s" \
          % (rule[0], rule[2], rule[3], rule[4], \
          data.solutionNames[rule[0]])
      print string

  ##############################################################################
  # Determine Skinny0 Solutions
  skinnyRules01 = data.getSkinnySolutions(diagonalRules, problemIndices, \
      data.idx0, data.idx1)
  #print "Skinny Rules:"
  #for rule in skinnyRules01:
  #  string = "  if freeSize >=%4u" % data.problemIndexToSize[0][rule[1][0]]
  #  for i in range(1, data.numIndices):
  #    string += "x%4u" % data.problemIndexToSize[i][rule[1][i]]
  #  string += " return S[%u] @ %5.0f-%5.0f>%5.0f GFlops is %s" \
  #      % (rule[0], rule[2], rule[3], rule[4], \
  #      data.solutionNames[rule[0]])

  ##############################################################################
  # Determine Skinny1 Solutions
  skinnyRules10 = data.getSkinnySolutions(diagonalRules, problemIndices, \
      data.idx1, data.idx0)

  # list solutions that actually get used
  solutionIndicesUsed = []
  for rule in skinnyRules01:
    pass
  for rule in skinnyRules10:
    pass
  for rule in diagonalRules:
    solutionIdx = rule[0]
    solution = solutions[solutionIdx]
    MT0 = solution["MacroTile0"]
    MT1 = solution["MacroTile1"]
    DU = solution["DepthU"]
    #print "Rule Tile S[%u]: %ux%ux%u" % (solutionIdx, MT0, MT1, DU)
    # is this solution in the list
    inList = False
    for solutionUsed in solutionIndicesUsed:
      if solutionUsed[0] == solutionIdx:
        inList = True
        break
    if not inList:
      insertIdx = len(solutionIndicesUsed)
      for i in range(0, len(solutionIndicesUsed)):
        iMT0 = solutionIndicesUsed[i][1]
        iMT1 = solutionIndicesUsed[i][2]
        iDU  = solutionIndicesUsed[i][3]
        #print "  compare S[%u]: %ux%ux%u" % (solutionIndicesUsed[i][0], \
        #    iMT0, iMT1, iDU)
        if MT0*MT1 < iMT0*iMT1:
          insertIdx = i
          break
        elif MT0*MT1 > iMT0*iMT1:
          continue
        else: # MT == MT
          if DU < iDU:
            insertIdx = i
            break
          else:
            continue

        # if i'm smaller than i, insert me before i
      #print "insert: %u" % insertIdx
      solutionIndicesUsed.insert(insertIdx, [solutionIdx, MT0, MT1, DU])
  print solutionIndicesUsed

  # list of solutions used
  solutionsUsed = []
  for solutionIndexUsed in solutionIndicesUsed:
    solutionsUsed.append(solutions[solutionIndexUsed[0]])

  # translate rules to new solution indices
  for rule in skinnyRules01:
    pass
  for rule in skinnyRules10:
    pass
  for ruleIdx in range(0, len(diagonalRules)):
    solutionIdx = diagonalRules[ruleIdx][0]
    for i in range(0, len(solutionIndicesUsed)):
      solutionIndexUsed = solutionIndicesUsed[i]
      if solutionIdx == solutionIndexUsed[0]:
        diagonalRules[ruleIdx][0] = i
        break
    # change problemSizeIndices to sizes
    for i in range(0, 3):
      diagonalRules[ruleIdx][1][i] = \
          data.problemIndexToSize[i][ diagonalRules[ruleIdx][1][i] ]

  print "New Rules: %s" % diagonalRules


  #return (skinnyRules01, skinnyRules10, diagonalRules)
  return (problemType, solutionsUsed, [], [], diagonalRules )



################################################################################
# BenchmarkDataAnalyzer
################################################################################
class BenchmarkDataAnalyzer:

  ########################################
  # diagonal rule looks like
  # 0: solutionIdx
  # 1: problemIndices for minThreshold problem
  # 2: gflops at above minSize
  # 3: maxGFlops for this solution along diagonal in interval it won
  # 4: gflops of prior winner at minSize, i.e., what performance did it beat

  ########################################
  # skinny rule looks like
  # 0: solutionIdx
  # 1: problemIndices for minThreshold problem
  # 2: gflops at above minSize


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
  def getFastestSolutionsAlongDiagonal(self, problemIndices):
    print "\nFastest Diagonal idxU: %u" % problemIndices[self.idxU]
    # abstract to multidimensions
    # what is the diagonal
    dilation = self.analysisParameters["Dilation"]
    threshold = self.analysisParameters["Threshold"]
    numProblems0 = self.numProblemSizes[self.idx0]

    ############################################################################
    # determine winner at largest size
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
    #print "Winner at Largest Problem: S[%u] @ %.0f GFlops with %u/%u wins" \
    #    % (largestWinnerIdx, largestWinnerGFlops, largestWinnerNumWins, \
    #    dilation*2)
    problemIndices[self.idx0] = numProblems0-1
    problemIndices[self.idx1] = numProblems0-1
    largestWinnerAtLargestProblemIdx = self.indicesToSerial(largestWinnerIdx, \
        problemIndices)
    largestWinnerGFlopsAtLargestSize = \
        self.data[largestWinnerAtLargestProblemIdx]

    ############################################################################
    # Diagonal Rule
    # solutionIdx, minSizeThresholdIdx, gflops at minSize, maxGFlops, oldGFlops
    numRules = 1
    diagonalRules = [ [largestWinnerIdx, deepcopy(problemIndices), \
        largestWinnerGFlopsAtLargestSize, largestWinnerGFlops, -1] ]

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
      (winnerForSizeIdx, winnerForSizeGFlops) = \
          self.getWinnerForProblem( problemIndices )

      # ruleWinner also wins at this problem size (at least by threshold)
      if winnerForSizeIdx == ruleWinnerIdx \
          or ruleWinnerGFlopsForSize > (1-threshold)*winnerForSizeGFlops:
        # just update rule
        diagonalRules[numRules-1][1] = deepcopy(problemIndices)
        diagonalRules[numRules-1][2] = ruleWinnerGFlopsForSize
        diagonalRules[numRules-1][3] = max(diagonalRules[numRules-1][3], \
            ruleWinnerGFlopsForSize)

      # we have a new candidate winner
      # only keep it if don't revert back to ruleWinner over next Dilation
      else:

        # check if we don't revert back to ruleWinner over next Dilation probs
        revert = False
        endDilationIdx = max(-1, problemSizeIdx-dilation)
        for dilationSizeIdx in range(problemSizeIdx-1, \
            endDilationIdx, -1):
          problemIndices[self.idx0] = dilationSizeIdx
          problemIndices[self.idx1] = dilationSizeIdx
          dilationIdx = self.indicesToSerial(0, problemIndices)
          ruleWinnerGFlopsForDilation = self.data[dilationIdx \
              + ruleWinnerIdx]
          #determine fastest at this problemSizeIdx
          (winnerForDilationIdx, winnerForDilationGFlops) = \
              self.getWinnerForProblem(problemIndices)

          # ruleWinner also wins at dilation size (at least by threshold)
          if winnerForDilationIdx == ruleWinnerIdx \
              or ruleWinnerGFlopsForDilation \
              > (1-threshold)*winnerForSizeGFlops:
            # yes, within Dilation, we've returned to same winner
            revert = True
            # so update rule for this size
            diagonalRules[numRules-1][1] = deepcopy(problemIndices)
            diagonalRules[numRules-1][2] = winnerForDilationGFlops
            diagonalRules[numRules-1][3] = max(diagonalRules[numRules-1][3], \
                winnerForSizeGFlops)
            # resume outer loop after dilation
            problemSizeIdx = dilationSizeIdx
            break
          else:
            # different winner at this dilation size
            # don't need to do anything
            pass

        # if we never revert to rule during dilation, create new rule
        if not revert:
          # solutionIdx, minSizeThresholdIdx, gflops at minSize, maxGFlops, old
          newRule = [ winnerForSizeIdx, deepcopy(problemIndices), \
              winnerForSizeGFlops, winnerForSizeGFlops, ruleWinnerGFlopsForSize]
          diagonalRules.append(newRule)
          numRules += 1
          #print "Added new rule: %s" % newRule

    return diagonalRules
    #end diagonal rules


  ##############################################################################
  # Skinny Solutions
  ##############################################################################
  def getSkinnySolutions(self, diagonalRules, problemIndices, \
      idxLarge, idxSmall):
    idx0 = self.idx0
    idx1 = self.idx1
    idxU = self.idxU
    dilation = self.analysisParameters["Dilation"]
    threshold = self.analysisParameters["Threshold"]

    skinnyRules = []

    # for each size threshold along diagonal
    for diagonalRuleIdx in range(0, len(diagonalRules)):
      diagonalRule = diagonalRules[diagonalRuleIdx]
      diagonalRuleWinnerIdx = diagonalRule[0]
      diagonalRuleThresholdProblem = diagonalRule[1]
      diagonalRuleGFlops = diagonalRule[2] # perf at threshold
      thresholdSizeFree = self.getSizeFree(diagonalRuleThresholdProblem)
      print "ThresholdSizeFree[%u][%u]: %u" \
          % (diagonalRuleThresholdProblem[idx0], \
          diagonalRuleThresholdProblem[idx1], \
          thresholdSizeFree)

      # check skinny d0<<d1 (large d0, small d1)
      skinnyProblemIndices = deepcopy(problemIndices)
      for sizeIdxSmall in range( diagonalRuleThresholdProblem[idxSmall]-1, -1, -1):
        skinnyProblemIndices[idxSmall] = sizeIdxSmall
        for sizeIdxLarge in range( diagonalRuleThresholdProblem[idxLarge], \
            self.numProblemSizes[idxLarge]):
          skinnyProblemIndices[idxLarge] = sizeIdxLarge


          skinnySizeFree = self.getSizeFree(skinnyProblemIndices)
          if skinnySizeFree > thresholdSizeFree:
            #print "SkinnySizeFree[%u][%u]: %u" % (sizeIdxSmall, sizeIdxLarge, \
            #  skinnySizeFree)

            # rule winner's performance at this skinnyness
            skinnyProblemIdx = self.indicesToSerial(0, skinnyProblemIndices)
            diagonalWinnerGFlopsForSkinny = self.data[skinnyProblemIdx \
                + diagonalRuleWinnerIdx]

            # which solution wins here?
            (winnerIdx, winnerGFlops) = \
                self.getWinnerForProblem(skinnyProblemIndices)
            #print winnerIdx, winnerGFlops
            if winnerIdx == diagonalRuleWinnerIdx \
                or diagonalWinnerGFlopsForSkinny > (1-threshold)*winnerGFlops:
              # diagonal rule also wins here
              print "if dS <%5u and dL >%5u diagnl S[%2u] %5.0f == S[%2u] %5.0f GFlops" \
                  % (self.problemIndexToSize[idxSmall][sizeIdxSmall], \
                  self.problemIndexToSize[idxLarge][sizeIdxLarge], \
                  winnerIdx, winnerGFlops, diagonalRuleWinnerIdx, \
                  diagonalWinnerGFlopsForSkinny )
              pass
            else:
              # we're so skinny that diagonal rule no longer applies
              print "if dS <%5u and dL >%5u skinny S[%2u] %5.0f >> S[%2u] %5.0f GFlops" \
                  % (self.problemIndexToSize[idxSmall][sizeIdxSmall], \
                  self.problemIndexToSize[idxLarge][sizeIdxLarge], \
                  winnerIdx, winnerGFlops, diagonalRuleWinnerIdx, \
                  diagonalWinnerGFlopsForSkinny )
              skinnyRule = [deepcopy(skinnyProblemIndices), winnerIdx, \
                  winnerGFlops]
              skinnyRules.append(skinnyRule)
              # TODO need to use dilate parameter to make sure we've switched
              # TODO data along this size may not agree with
              #   data along different sizes (but perhaps it should
              # TODO need extra loop here, to iterate idxSmall to
              # smaller sizes to see if the solution changes further

            # does the diagonalRuleWinner also win here?
            break # only check the problem size closest to ruleSize
      print

    return skinnyRules
    # end skinny solutions

  ##############################################################################
  # Get Size Free and Summation
  ##############################################################################
  def getWinnerForProblem(self, problemIndices):
    problemIdx = self.indicesToSerial(0, problemIndices)
    winnerIdx = -1
    winnerGFlops = -1
    for solutionIdx in range(0, self.numSolutions):
      solutionSerialIdx = problemIdx + solutionIdx
      solutionGFlops = self.data[solutionSerialIdx]
      if solutionGFlops > winnerGFlops:
        #print "%f > %f" % (solutionGFlops, winnerGFlops)
        winnerIdx = solutionIdx
        winnerGFlops = solutionGFlops
    return (winnerIdx, winnerGFlops)


  ##############################################################################
  # Get Size Free and Summation
  ##############################################################################
  def getSizeFree(self, problemIndices):
    sizeFree = 1
    for i in range(0, self.problemType["NumIndicesC"]):
      sizeFree *= self.problemIndexToSize[i][problemIndices[i]]
    return sizeFree

  def getSizeSummation(self, problemIndices):
    sizeSummation = 1
    for i in range(self.problemType["NumIndicesC"], \
        self.problemType["TotalIndices"]):
      sizeSummation *= self.problemIndexToSize[i][problemIndices[i]]
    return sizeSummation

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
  benchmarkDataPath = os.path.join(globalParameters["WorkingPath"], globalParameters["BenchmarkDataPath"])
  pushWorkingPath(globalParameters["LibraryLogicPath"])

  # Assign Defaults
  analysisParameters = {}
  for parameter in defaultAnalysisParameters:
    assignParameterWithDefault(analysisParameters, parameter, config, \
        defaultAnalysisParameters)

  print ""
  print HR
  print "# Analysing data in %s." % globalParameters["BenchmarkDataPath"]
  for parameter in analysisParameters:
    print "#   %s: %s" % (parameter, analysisParameters[parameter])
  print HR
  print ""



  ##############################################################################
  # Determine Which Problem Types
  ##############################################################################
  problemTypeTuples = []
  if not os.path.exists(benchmarkDataPath):
    printExit("Path doesn't exist: %s" % benchmarkDataPath)
  for fileName in os.listdir(benchmarkDataPath):
    if os.path.splitext(fileName)[1] == ".csv":
      fileBase = os.path.splitext( \
          os.path.join(benchmarkDataPath, \
          fileName))[0]
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
  schedulePrefix = globalParameters["Name"]
  for problemTypeTuple in problemTypeTuples:
    logic = analyzeProblemType( problemTypeTuple, analysisParameters )
    YAMLIO.writeLibraryConfigForProblemType(globalParameters["WorkingPath"], \
        schedulePrefix, logic)

  printStatus("DONE.")
  popWorkingPath()

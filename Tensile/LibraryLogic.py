import os
import os.path
import array
import csv

from copy import deepcopy

from Common import print1, print2, printWarning, HR, printExit, defaultAnalysisParameters, globalParameters, pushWorkingPath, popWorkingPath, assignParameterWithDefault
from SolutionStructs import Solution
import YAMLIO

################################################################################
# Analyze Problem Type
################################################################################
def analyzeProblemType( problemTypeTuple, inputParameters ):
  problemType = problemTypeTuple[0]
  problemSizes = problemTypeTuple[1]
  dataFileName = problemTypeTuple[2]
  solutionsFileName = problemTypeTuple[3]
  print2(HR)
  print1("# %s" % problemType)

  #print "#  %s" % dataFileName
  #print "#  %s" % solutionsFileName

  ######################################
  # Read Solutions
  (problemSizes, solutions) = YAMLIO.readSolutions(solutionsFileName)
  print2("# ProblemSizes: %s" % problemSizes)
  solutionMinNaming = Solution.getMinNaming(solutions)
  print2("# Solutions:")
  solutionIdx = 0
  for solution in solutions:
    print2("#  (%u) %s" % (solutionIdx, Solution.getNameMin(solution, \
        solutionMinNaming)))
    solutionIdx += 1
  print2(HR)

  ######################################
  # Read Data From CSV
  logic = LogicAnalyzer(problemType, problemSizes, solutions, inputParameters)
  logic.populateFromCSV(dataFileName)

  ######################################
  # Remove invalid solutions
  logic.removeInvalidSolutions()

  ######################################
  # Remove least important solutions
  logic.removeLeastImportantSolutions()

  ######################################
  # Correct outliers
  # logic.smooth()
  logic.print2D([0, 0])

  ######################################
  # Create Rules
  logic.enRule(0, logic.globalIndexRange)



  #return (skinnyRules01, skinnyRules10, diagonalRules)
  #return (problemType, logic.solutionsUsed, [], [], logic.diagonalRules )
  return (problemType, [], [], [], [] )



################################################################################
# LogicAnalyzer
################################################################################
class LogicAnalyzer:

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

  ##############################################################################
  ##############################################################################
  ###
  ###  Entry / Top-Level Functions
  ###
  ##############################################################################
  ##############################################################################

  ##############################################################################
  # ENTRY: Init
  ##############################################################################
  def __init__(self, problemType, problemSizes, solutions, inputParameters):
    self.problemType = problemType
    self.problemSizes = problemSizes
    self.parameters = inputParameters
    print2("ProblemSizes: %s" % self.problemSizes)
    # TODO verify that data is symmetric for diagonal
    #if self.problemSizes[self.problemType["Index0"]] \
    #    != self.problemSizes[self.problemType["Index1"]]:
    #  printExit("d0 / d1 must be symmetric for analysis.")
    self.numProblemSizes = problemSizes.numProblemSizes # native order
    print1("NumProblemSizes: %s" % self.numProblemSizes)
    self.numIndices = len(self.numProblemSizes)
    self.solutions = solutions
    self.numSolutions = len(self.solutions)
    self.solutionMinNaming = Solution.getMinNaming(solutions)
    self.solutionNames = []
    self.solutionTiles = []
    for solution in self.solutions:
      self.solutionNames.append(Solution.getNameMin(solution, \
          self.solutionMinNaming))
      self.solutionTiles.append("%ux%u"%(solution["MacroTile0"], solution["MacroTile1"]))
    self.flopsPerMac = self.problemType["DataType"].flopsPerMac()

    # special indices
    self.idx0 = self.problemType["Index0"]
    self.idx1 = self.problemType["Index1"]
    self.idxU = self.problemType["IndexUnroll"]

    # total size of data array
    self.totalProblems = 1
    for numProblems in self.numProblemSizes:
      self.totalProblems *= numProblems
    self.totalSize = self.totalProblems * self.numSolutions
    print2("TotalProblems: %u" % self.totalProblems)
    print2("TotalSolutions: %u" % self.numSolutions)
    print2("TotalSize: %u" % self.totalSize)
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
    self.rangeIndicesFree = range(0, self.problemType["NumIndicesC"])
    self.rangeIndicesSummation = range(self.problemType["NumIndicesC"], \
        self.problemType["TotalIndices"])
    self.w0 = self.parameters["Weight0"]
    self.w1 = self.parameters["Weight1"]
    self.w2 = self.parameters["Weight2"]
    #print "S->I %s" % self.problemSizeToIndex
    #print "I->S %s" % self.problemIndexToSize
    self.indexOrder = self.recommendedIndexOrder()
    print2("IndexOrder: %s" % self.indexOrder)
    self.globalIndexRange = []
    for i in range(0, self.numIndices):
      self.globalIndexRange.append([0, self.numProblemSizes[i]])
    self.problemIndicesForGlobalRange \
        = self.problemIndicesForRange(self.globalIndexRange)



  ##############################################################################
  # ENTRY: Read In CSV
  ##############################################################################
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
        #totalFlops = float(row[totalSizeIdx])

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
  # ENTRY: Remove Invalid Solutions
  ##############################################################################
  def removeInvalidSolutions(self):
    #problemIndices = [0]*self.numIndices
    allSolutionValid = False
    while not allSolutionValid:
      moreProblems = True
      invalidIdx = -1
      for problemIndices in self.problemIndicesForGlobalRange:
        problemIdx = self.indicesToSerial(0, problemIndices)
        for solutionIdx in range(0, self.numSolutions):
          gflops = self.data[problemIdx+solutionIdx]
          if gflops == 0:
            invalidIdx = solutionIdx
            break
      if invalidIdx >= 0:
        print1("# Removing Invalid Solution: %u %s" \
            % (invalidIdx, self.solutionNames[invalidIdx]) )
        self.removeSolution(invalidIdx)
      else:
        allSolutionValid = True


  ##############################################################################
  # ENTRY: Remove Least Important Solutions
  ##############################################################################
  def removeLeastImportantSolutions(self):
    # Remove least important solutions
    while True:
      (lisIdx, lisPercSaved, lisPercWins, lisPercExec) \
          = self.leastImportantSolution()
      if lisPercSaved < self.parameters["FractionTimeSavedMin"]:
        self.removeSolution(lisIdx)
        continue
      else:
        break


  ##############################################################################
  # ENTRY: Smooth - correct outliers
  ##############################################################################
  def smooth(self):
    outlierThreshold = self.parameters["OutlierThreshold"]
    problemSizes = [0]*self.numIndices
    for problemIndices in self.problemIndicesForGlobalRange:
      problemIdx = self.indicesToSerial(0, problemIndices)

      for solutionIdx in range(0, self.numSolutions):
        gflops = self.data[problemIdx+solutionIdx]
        neighborGFlops = []
        smoothProblem = False
        for iIdx in range(0, self.numIndices):
          if problemIndices[iIdx] > 0 \
              and problemIndices[iIdx] < self.numProblemSizes[iIdx]-1:
            neighborBeforeIndices = deepcopy(problemIndices)
            neighborAfterIndices = deepcopy(problemIndices)
            neighborBeforeIndices[iIdx] -= 1
            neighborAfterIndices[iIdx] += 1
            neighborBeforeIdx = self.indicesToSerial(0, neighborBeforeIndices)
            neighborAfterIdx = self.indicesToSerial(0, neighborAfterIndices)
            neighborBeforeGFlops = self.data[neighborBeforeIdx+solutionIdx]
            neighborAfterGFlops = self.data[neighborAfterIdx+solutionIdx]
            neighborGFlops.append(neighborBeforeGFlops)
            neighborGFlops.append(neighborAfterGFlops)
            if neighborBeforeGFlops > gflops * (1+outlierThreshold) \
                and neighborAfterGFlops * (1+outlierThreshold) < gflops :
              smoothProblem = True
        if smoothProblem:
          s = ""
          for i in range(0, self.numIndices):
            problemSizes[i] = self.problemIndexToSize[i][problemIndices[i]]
            s += "%u, " % problemSizes[i]
          new = sum(neighborGFlops)/len(neighborGFlops)
          old = self.data[problemIdx+solutionIdx]
          s += "%f -> %f" % (old, new)
          print s
          self.data[problemIdx+solutionIdx] \
              = sum(neighborGFlops)/len(neighborGFlops)


  ##############################################################################
  # ENTRY: En Rule
  # currentIndexIndex = 0, 1, 2, 3...
  ##############################################################################
  def enRule(self, currentIndexIndex, currentIndexRange):
    currentIndex = self.indexOrder[currentIndexIndex]
    lastIndex = currentIndexIndex == self.numIndices-1

    # if there's only 1 problem size here
    if currentIndexRange[currentIndex][1] \
        - currentIndexRange[currentIndex][0] == 1:
      # this is last index, so just return fastest solution
      if lastIndex:
        scores = scoreRangeForSolutions(currentIndexRange)
        winnerIdx = 0
        for solutionIdx in range(1, self.numSolution):
          if scores[solutionIdx] < scores[winnerIdx]:
            winnerIdx = solutionIdx
        rule = [ -1, winnerIdx ]
      # this isn't last index, so just return next index
      else:
        newIndexIndex = currentIndexIndex+1
        newIndexRange = deepcopy(currentIndexRange)
        rule = [ -1, self.enRule(newIndexIndex, newIndexRange) ]

    # create rule for smallest size

    # for all problem indices in this index
    for problemIndex in range(currentIndexRange[currentIndex][0], \
        currentIndexRange[currentIndex][1]):
    # rules = seed with smallest rule
    # for dimIdx = 0 -> numSizes
      # if newRule
        # score range using newRule
        # score range using priorRule
        # accept/reject based on score
    # current index is dimOrder[0]





    sumValues = []
    totalSummationSizes = 1
    for i in self.rangeIndicesSummation:
      totalSummationSizes *= self.numProblemSizes[i]
    summationPermutations = []
    for permutationIdx in range(0, totalSummationSizes):
      permutation = []
      permutationSize = 1
      pIdx = permutationIdx
      for i in self.rangeIndicesSummation:
        idx = pIdx % self.numProblemSizes[i]
        permutation.append(idx)
        permutationSize *= self.problemIndexToSize[i][idx]
        pIdx /= self.numProblemSizes[i]
      # insert permutation in sorted order
      insertIdx = len(summationPermutations)-1
      for pIdx in range(0, len(summationPermutations)):
        size = 1
        for i in self.rangeIndicesSummation:
          size *= self.problemIndexToSize[i][summationPermutations[pIdx][i]]
        if permutationSize > size:
          insertIdx = pIdx
          break
      summationPermutations.insert(insertIdx, permutation)
    print "SummationPermutations:", summationPermutations


    if len(summationPermutations) == 1:
      rules = [ 0, self.createRules01(summationPermutations[0]) ]
      return rules
    else:
      printExit("No Logic to support multiple summation sizes.")
      # iterate over summation permutations
# for each serial pair, scoreA, scoreB, scoreAB
# keep rule AB if scoreAB isn't much slower than scoreA + scoreB

    """
    sizeSummation *= self.problemIndexToSize[i][problemIndices[i]]

    firstProblemIndices = []
    lastProblemIndices = []
    for i in range(0, self.numIndices):
      firstProblemIndices.append(0)
      lastProblemIndices.append(self.numProblems[i]-1)
    minSumValue = self.getSizeSummation(firstProblemIndices)
    maxSumValue = self.getSizeSummation(lastProblemIndices)
    numSumValues =


    rule = [
        [
          minU,                             # k threshold
          [[min01,s], [0,s]],               # diagonals
          [0, max0, [[min1,s], [min1,s]]],  # skinny0's
          [1, max1, [[min0,s], [min0,s]]],  # skinny1's
        ],
        [
          minU,                             # k threshold
          [[min01,s], [0,s]],               # diagonals
          [0, max0, [[min1,s], [min1,s]]],  # skinny0's
          [1, max1, [[min0,s], [min0,s]]],  # skinny1's
        ],
    ]

    ruleA = createRules01()
    ruleB = createRules01()

    minSumValue = 0
    maxSumValue = self.numProblems


    sizeSummation = 1
    for i in range(self.problemType["NumIndicesC"], \
        self.problemType["TotalIndices"]):
      sizeSummation *= self.problemIndexToSize[i][problemIndices[i]]
    return sizeSummation
    """



  ##############################################################################
  ##############################################################################
  ###
  ###  Mid-Level Functions
  ###
  ##############################################################################
  ##############################################################################



  ##############################################################################
  # Create Rules dim0 / dim1
  ##############################################################################
  def createRules01(self, problemSizeSummation ):

    diagonalRules = self.createRulesDiagonal(problemSizeSummation)


  ##############################################################################
  # Create Rules Diagonal
  ##############################################################################
  def createRulesDiagonal(self, problemSizeSummation):
    thresholdForDiagonality = 1.5 # slightly fewer problems than 2
    numProblemSizesFastestDiagonal = 16
    problemIndices = [0]*self.numIndices
    for i in self.rangeIndicesSummation:
      problemIndices[i] = problemSizeSummation[i \
          - self.problemType["NumIndicesC"]]
    print2("\nDiagonalRules for %s" % problemIndices)
    problemSizes = [0]*self.numIndices
    totalFlopsPerSizeFree = self.flopsPerMac
    for i in self.rangeIndicesSummation:
      totalFlopsPerSizeFree *= self.problemIndexToSize[i][problemIndices[i]]
    print "totalFlopsPerSizeFree", totalFlopsPerSizeFree

    ########################################
    # transform data into serial list of "diagonal problem sizes"
    diagonalData = []
    moreProblems = True
    while moreProblems:

      # size free
      for i in range(0, self.numIndices):
        problemSizes[i] = self.problemIndexToSize[i][problemIndices[i]]
      size0 = problemSizes[self.idx0]
      size1 = problemSizes[self.idx1]

      # if diagonal
      if size0 < size1*thresholdForDiagonality \
          and size1 < size0*thresholdForDiagonality:
        sizeFree = self.getSizeFree(problemIndices)

        problemIdx = self.indicesToSerial(0, problemIndices)
        solutionGFlops = []
        for i in range(0, self.numSolutions):
          solutionGFlops.append(self.data[problemIdx+i])

        diagonalData.append([ sizeFree, solutionGFlops ])

      # next problem
      problemIndices[0] += 1
      for i in self.rangeIndicesFree:
        if problemIndices[i] >= self.numProblemSizes[i]:
          if i == self.problemType["NumIndicesFree"]-1:
            moreProblems = False
            break
          else:
            problemIndices[i] = 0
            problemIndices[i+1] += 1
        else:
          break

    diagonalData.sort(key=lambda x: x[0], reverse=True)
    for dd in diagonalData:
      print "DD[%u]: %s" % (dd[0], dd[1])
    print len(diagonalData)


    ########################################
    # create first rule
    sizeFree = diagonalData[0][0]
    relativeTime = [0]*self.numSolutions
    for i in range(0, numProblemSizesFastestDiagonal):
      for j in range(0, self.numSolutions):
        gflops = diagonalData[i][1][j]
        relativeTime[j] += 1 / gflops
    winnerIdx = 0
    winnerRelativeTime = relativeTime[0]
    for i in range(1, self.numSolutions):
      if relativeTime[i] < winnerRelativeTime:
        winnerIdx = i
        winnerRelativeTime = relativeTime[i]
    print "FastestDiagonalSolution:", winnerIdx, self.solutionNames[winnerIdx]
    fastestGFlops = 0
    for i in range(0, numProblemSizesFastestDiagonal):
      gflops = diagonalData[i][1][winnerIdx]
      if gflops > fastestGFlops:
        fastestGFlops = gflops

    rules = []
    #                                  minGFlops      maxGFlops      oldGFlops?
    rules.append([winnerIdx, sizeFree, fastestGFlops, fastestGFlops, -1])
    print "Winner[%3u]: %u" % (0, winnerIdx)
# we can't just pay attention to single winner
# we need to compute scores for all solutions over a window
# b/c 441115111333
#   = 441555555333
#
# we can do a smoothing pass to get rid of bogus data; if a data point is more than x% slower than 4 surrounding points, than its bogus, just set it equal to average of 4 surrounding points
#

    ########################################
    # create subsequent rules for smaller sizes
    for diagonalDataIdx in range(1, len(diagonalData)):
      print "DiagonalDataIdx:", diagonalDataIdx
      # prior rule
      priorRule = rules[len(rules)-1]
      priorWinnerIdx = priorRule[0]
      # candidate winner
      candidateWinnerIdx = 0
      candidateWinnerGFlops = diagonalData[diagonalDataIdx][1][0]
      for j in range(1, self.numSolutions):
        gflops = diagonalData[diagonalDataIdx][1][j]
        if gflops > candidateWinnerGFlops:
          candidateWinnerIdx = j
          candidateWinnerGFlops = gflops
      if candidateWinnerIdx == priorWinnerIdx:
        # update prior rule to include this sizeFree
        rules[len(rules)-1][1] = diagonalData[diagonalDataIdx][0] # size free
        rules[len(rules)-1][2] = \
            diagonalData[diagonalDataIdx][1][priorWinnerIdx] # perf at size
        continue
      else:
        # candidate rule
        sizeFree = diagonalData[diagonalDataIdx][0]
        totalFlops = sizeFree*totalFlopsPerSizeFree
        candidateGFlops = diagonalData[diagonalDataIdx][1][candidateWinnerIdx]
        priorGFlops = diagonalData[diagonalDataIdx][1][priorWinnerIdx]
        candidateRule = [ candidateWinnerIdx, sizeFree, candidateGFlops, \
            candidateGFlops, -1 ]
        # candidate and prior scores
        candidateTimeUs = totalFlops / candidateGFlops / 1000
        priorTimeUs = totalFlops / priorGFlops / 1000
        candidateScore = 1*self.w2 + candidateTimeUs
        priorScore = 0*self.w2 + priorTimeUs
        print "DDI[%3u] Prior[%2u]: %.0fus vs Candi[%2u]: %.0fus" \
            % (diagonalDataIdx, priorWinnerIdx, priorScore, candidateWinnerIdx, candidateScore)
        checkMoreProblems = True
        for newDiagonalDataIdx in range(diagonalDataIdx+1, len(diagonalData)):
          newWinnerIdx = 0
          newWinnerGFlops = diagonalData[newDiagonalDataIdx][1][0]
          for j in range(1, self.numSolutions):
            gflops = diagonalData[newDiagonalDataIdx][1][j]
            if gflops > newWinnerGFlops:
              newWinnerIdx = j
              newWinnerGFlops = gflops
          # update candidate and prior scores
          sizeFree = diagonalData[newDiagonalDataIdx][0]
          totalFlops = sizeFree*totalFlopsPerSizeFree
          candidateGFlops = \
              diagonalData[newDiagonalDataIdx][1][candidateWinnerIdx]
          priorGFlops = diagonalData[newDiagonalDataIdx][1][priorWinnerIdx]
          candidateTimeUs = totalFlops / candidateGFlops / 1000
          priorTimeUs = totalFlops / priorGFlops / 1000
          candidateScore += candidateTimeUs
          priorScore += priorTimeUs
          print "  NDDI[%3u] Prior[%2u]: %.0fus vs Candi[%2u]: %.0fus" \
              % (newDiagonalDataIdx, priorWinnerIdx, priorScore, \
              candidateWinnerIdx, candidateScore)
          if newWinnerIdx == candidateWinnerIdx:
            print "    newWinnerIdx == candidateWinnerIdx"
            if candidateScore < priorScore:
              # candidate rule accepted
              rules.append(candidateRule)
              print "      accepting"
              break
            else:
              # candidate rule not yet accepted
              candidateRule[1] = sizeFree
              candidateRule[2] = candidateGFlops
              print "      continuing"
              continue
          elif newWinnerIdx == priorWinnerIdx:
            print "    newWinnerIdx == priorWinnerIdx"
            # returned to original winner, decide now to accept/reject
            if candidateScore < priorScore:
              # candidate rule accepted
              rules.append(candidateRule)
              print "      accepting"
              break
            else:
              # candidate rule rejected; update prior, continue at newSize
              rules[len(rules)-1][1] = sizeFree
              rules[len(rules)-1][2] = priorGFlops
              diagonalDataIdx = newDiagonalDataIdx
              print "      rejecting"
              break
          else:
            print "    newWinnerIdx is %u" % newWinnerIdx
            # new winner was a 3rd solution; decide now (same as above)
            if candidateScore < priorScore:
              # candidate rule accepted
              rules.append(candidateRule)
              print "      accepting"
              break
            else:
              # candidate rule rejected; update prior, continue at newSize
              rules[len(rules)-1][1] = diagonalData[newDiagonalDataIdx][0]
              rules[len(rules)-1][2] = \
                  diagonalData[newDiagonalDataIdx][1][priorWinnerIdx]
              diagonalDataIdx = newDiagonalDataIdx
              print "      rejecting"
              break

      return

        # go farther forward, does candidate rule keep winning, or does priorRule keep winning?
        # the new rule should start at a loss b/c of Weight2
        # a few problems in the future
            # if new rule is better, W2 gets amortized, Wt improves
            # if new rule is worse, W2 gets amortized, Wt worsens
        # continue to future problems until, and make final decision
          # newRule gets better score; accept
          # return to priorRule winner; accept/reject
          # Yet a new winner
            # easy: make final accept/reject including this new problem size
            # hard: recure?
          #
        # is the num problems in future vary with W2,Wt?
# Wt = 1
# W2 = 1 means we would rather lose 1us per kernel rather than adding another split (actually they're equal)
# so, in order for candidate to be accepted immediately, it must improve all kernels by more than 1us, or after 2 sizes, improve by 0.5us per kernel
#
#
# 0 0 1 0 0
# 0 0 1 1 0
# 0 0 1 4 0
# 0 0 1 4 1 0
#

      print "Winner[%3u]: %u" % (i, winnerIdx)


    return

















    # abstract to multidimensions
    # what is the diagonal
    dilation = self.self.parameters["Dilation"]
    threshold = self.self.parameters["Threshold"]
    numProblems0 = self.numProblemSizes[self.idx0]

    ############################################################################
    # determine winner at largest size
    solutionNumWins = [0]*self.numSolutions
    solutionGFlops = [0]*self.numSolutions
    for problemSizeIdx in range(max(0,numProblems0-dilation*2), numProblems0):
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
    #idxU = self.idxU
    #dilation = self.self.parameters["Dilation"]
    threshold = self.self.parameters["Threshold"]

    skinnyRules = []

    # for each size threshold along diagonal
    for diagonalRuleIdx in range(0, len(diagonalRules)):
      diagonalRule = diagonalRules[diagonalRuleIdx]
      diagonalRuleWinnerIdx = diagonalRule[0]
      diagonalRuleThresholdProblem = diagonalRule[1]
      #diagonalRuleGFlops = diagonalRule[2] # perf at threshold
      thresholdSizeFree = self.getSizeFree(diagonalRuleThresholdProblem)
      print2("ThresholdSizeFree[%u][%u]: %u" \
          % (diagonalRuleThresholdProblem[idx0], \
          diagonalRuleThresholdProblem[idx1], \
          thresholdSizeFree))

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
              print2("if dS <%5u and dL >%5u diagnl S[%2u] %5.0f == S[%2u] %5.0f GFlops" \
                  % (self.problemIndexToSize[idxSmall][sizeIdxSmall], \
                  self.problemIndexToSize[idxLarge][sizeIdxLarge], \
                  winnerIdx, winnerGFlops, diagonalRuleWinnerIdx, \
                  diagonalWinnerGFlopsForSkinny ))
              pass
            else:
              # we're so skinny that diagonal rule no longer applies
              print2("if dS <%5u and dL >%5u skinny S[%2u] %5.0f >> S[%2u] %5.0f GFlops" \
                  % (self.problemIndexToSize[idxSmall][sizeIdxSmall], \
                  self.problemIndexToSize[idxLarge][sizeIdxLarge], \
                  winnerIdx, winnerGFlops, diagonalRuleWinnerIdx, \
                  diagonalWinnerGFlopsForSkinny ))
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

    return skinnyRules
    # end skinny solutions


  ##############################################################################
  # Determine Logic Along U
  ##############################################################################
  def determineLogicAlongU(self):
    globalRange = []
    for i in range(0, self.numIndices):
      globalRange.append( [0, self.numProblemSizes[i]] )




    self.print2D([0, 0])

    ############################################################################
    # Determine Solutions Along Diagonal
    # roughly same splitting regardless of sizeU
    problemIndices = []
    for numProblemsForIndex in self.numProblemSizes:
      problemIndices.append(numProblemsForIndex-1)
    print problemIndices
    self.diagonalRules = self.getFastestSolutionsAlongDiagonal(problemIndices)
    if True:
      print2("Diagonal Rules:")
      for rule in self.diagonalRules:
        string = "  if freeSize >=%4u" % self.problemIndexToSize[0][rule[1][0]]
        for i in range(1, self.numIndices):
          string += "x%4u" % self.problemIndexToSize[i][rule[1][i]]
        string += " return S[%u] @ %5.0f-%5.0f>%5.0f GFlops is %s" \
            % (rule[0], rule[2], rule[3], rule[4], \
            self.solutionNames[rule[0]])
        print2(string)

    ############################################################################
    # Determine Skinny0 Solutions
    skinnyRules01 = self.getSkinnySolutions(self.diagonalRules, problemIndices, \
        self.idx0, self.idx1)
    #print "Skinny Rules:"
    #for rule in skinnyRules01:
    #  string = "  if freeSize >=%4u" % data.problemIndexToSize[0][rule[1][0]]
    #  for i in range(1, data.numIndices):
    #    string += "x%4u" % data.problemIndexToSize[i][rule[1][i]]
    #  string += " return S[%u] @ %5.0f-%5.0f>%5.0f GFlops is %s" \
    #      % (rule[0], rule[2], rule[3], rule[4], \
    #      data.solutionNames[rule[0]])

    ############################################################################
    # Determine Skinny1 Solutions
    skinnyRules10 = self.getSkinnySolutions(self.diagonalRules, problemIndices, \
        self.idx1, self.idx0)

    # list solutions that actually get used
    solutionIndicesUsed = []
    for rule in skinnyRules01:
      pass
    for rule in skinnyRules10:
      pass
    for rule in self.diagonalRules:
      solutionIdx = rule[0]
      solution = self.solutions[solutionIdx]
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
    #print solutionIndicesUsed

    # list of solutions used
    self.solutionsUsed = []
    for solutionIndexUsed in solutionIndicesUsed:
      self.solutionsUsed.append(self.solutions[solutionIndexUsed[0]])

    # translate rules to new solution indices
    for rule in skinnyRules01:
      pass
    for rule in skinnyRules10:
      pass
    for ruleIdx in range(0, len(self.diagonalRules)):
      solutionIdx = self.diagonalRules[ruleIdx][0]
      for i in range(0, len(solutionIndicesUsed)):
        solutionIndexUsed = solutionIndicesUsed[i]
        if solutionIdx == solutionIndexUsed[0]:
          self.diagonalRules[ruleIdx][0] = i
          break
      # change problemSizeIndices to sizes
      for i in range(0, 3):
        self.diagonalRules[ruleIdx][1][i] = \
            self.problemIndexToSize[i][ self.diagonalRules[ruleIdx][1][i] ]

    print2("# New Rules: %s" % self.diagonalRules)



  ##############################################################################
  ##############################################################################
  ###
  ###  Helper / Low-Level Functions
  ###
  ##############################################################################
  ##############################################################################



  ##############################################################################
  # Print2D
  def print2D(self, indices ):
    indicesIdx = 0
    problemIndices = []
    for i in range(0, self.numIndices):
      if i == self.idx0:
        problemIndices.append(-1)
      elif i == self.idx1:
        problemIndices.append(-1)
      else:
        problemIndices.append(indices[indicesIdx])
        indicesIdx += 1

    winnerIndices = []
    w = "winner"
    g = "gflops"
    f = "faster"
    s = "second"
    sss = []
    for sIdx in range(0, self.numSolutions):
      sss.append("Sol[%u]" % sIdx)
    for j in range(0, self.numProblemSizes[1]):
      w += ",%4u" % self.problemIndexToSize[1][j]
      g += ",%4u" % self.problemIndexToSize[1][j]
      f += ",%4u" % self.problemIndexToSize[1][j]
      s += ",%4u" % self.problemIndexToSize[1][j]
      for sIdx in range(0, self.numSolutions):
        sss[sIdx] += ",%4u" % self.problemIndexToSize[1][j]
    w += "\n"
    g += "\n"
    f += "\n"
    s += "\n"
    for sIdx in range(0, self.numSolutions):
      sss[sIdx] += "\n"
    for i in range(0, self.numProblemSizes[0]):
      problemIndices[self.idx0] = i
      w += "%4u" % self.problemIndexToSize[0][i]
      g += "%4u" % self.problemIndexToSize[0][i]
      f += "%4u" % self.problemIndexToSize[0][i]
      s += "%4u" % self.problemIndexToSize[0][i]
      for sIdx in range(0, self.numSolutions):
        sss[sIdx] += "%4u" % self.problemIndexToSize[0][i]
      for j in range(0, self.numProblemSizes[1]):
        problemIndices[self.idx1] = j
        problemIdx = self.indicesToSerial(0, problemIndices)
        for sIdx in range(0, self.numSolutions):
          sss[sIdx] += ",%f" % self.data[problemIdx+sIdx]

        if self.data[problemIdx+0] > self.data[problemIdx+1]:
          winnerIdx = 0
          winnerGFlops = self.data[problemIdx+0]
          secondIdx = 1
          secondGFlops = self.data[problemIdx+1]
        else:
          winnerIdx = 1
          winnerGFlops = self.data[problemIdx+1]
          secondIdx = 0
          secondGFlops = self.data[problemIdx+0]
        for solutionIdx in range(2, self.numSolutions):
          solutionSerialIdx = problemIdx + solutionIdx
          solutionGFlops = self.data[solutionSerialIdx]
          if solutionGFlops > winnerGFlops:
            #print "%f > %f" % (solutionGFlops, winnerGFlops)
            secondIdx = winnerIdx
            secondGFlops = winnerGFlops
            winnerIdx = solutionIdx
            winnerGFlops = solutionGFlops


        if winnerIdx not in winnerIndices:
          winnerIndices.append(winnerIdx)
        w += ",%4u" % winnerIdx
        g += ",%f" % winnerGFlops
        f += ",%f" % (winnerGFlops/secondGFlops)
        s += ",%4u" % (secondIdx)
      w += "\n"
      g += "\n"
      f += "\n"
      s += "\n"
      for sIdx in range(0, self.numSolutions):
        sss[sIdx] += "\n"

    w += "\n\n"
    g += "\n\n"
    f += "\n\n"
    s += "\n\n"
    for sIdx in range(0, self.numSolutions):
      sss[sIdx] += "\n\n"
    w += "Winners:\n"
    for winnerIdx in winnerIndices:
      w += "%4u, %s, %s\n" % (winnerIdx, self.solutionTiles[winnerIdx], self.solutionNames[winnerIdx])

    printFileName = "Winner2D"
    for idx in indices:
      printFileName += "_%u" % idx
    printFileName += ".csv"
    printFile = open(os.path.join(globalParameters["WorkingPath"], printFileName), "w")
    printFile.write( w )
    printFile.write( g )
    printFile.write( f )
    printFile.write( s )
    for sIdx in range(0, self.numSolutions):
      printFile.write( sss[sIdx] )
    printFile.close()


  ##############################################################################
  # Least Important Solution
  def leastImportantSolution(self):
    solutionImportance = []
    for i in range(0, self.numSolutions):
      solutionImportance.append([i, 0, 0, 0])
    problemSizes = [0]*self.numIndices
    totalSavedMs = 0
    totalExecMs = 0
    totalWins = 0
    for problemIndices in self.problemIndicesForGlobalRange:
      for i in range(0, self.numIndices):
        problemSizes[i] = self.problemIndexToSize[i][problemIndices[i]]
      totalFlops = self.flopsPerMac
      for size in problemSizes:
        totalFlops *= size

      problemIdx = self.indicesToSerial(0, problemIndices)
      if self.data[problemIdx+0] > self.data[problemIdx+1]:
        winnerIdx = 0
        winnerGFlops = self.data[problemIdx+0]
        secondIdx = 1
        secondGFlops = self.data[problemIdx+1]
      else:
        winnerIdx = 1
        winnerGFlops = self.data[problemIdx+1]
        secondIdx = 0
        secondGFlops = self.data[problemIdx+0]

      for solutionIdx in range(2, self.numSolutions):
        solutionSerialIdx = problemIdx + solutionIdx
        solutionGFlops = self.data[solutionSerialIdx]
        if solutionGFlops > winnerGFlops:
          secondIdx = winnerIdx
          secondGFlops = winnerGFlops
          winnerIdx = solutionIdx
          winnerGFlops = solutionGFlops
      winnerTimeMs = totalFlops / winnerGFlops / 1000000
      secondTimeMs = totalFlops / secondGFlops / 1000000
      solutionImportance[winnerIdx][1] += (secondTimeMs - winnerTimeMs)
      solutionImportance[winnerIdx][2] += 1
      solutionImportance[winnerIdx][3] += winnerTimeMs

      totalSavedMs += secondTimeMs - winnerTimeMs
      totalExecMs += winnerTimeMs
      totalWins += 1
    solutionImportance.sort(key=lambda x: x[1])
    return ( solutionImportance[0][0], \
        solutionImportance[0][1] / totalSavedMs, \
        solutionImportance[0][2] / totalWins, \
        solutionImportance[0][3] / totalExecMs )


  ##############################################################################
  # Score Range For Logic
  def scoreRangeForLogic(self, indexRange, logic):
    pass

  ##############################################################################
  # Score (microseconds) Range For Solutions
  def scoreRangeForSolutions(self, indexRange):
    scores = [0]*self.numSolutions
    for problemIndices in self.problemIndicesForRange(indexRange):
      problemSerial = self.indicesToSerial(0, problemIndices)
      totalFlops = self.totalFlopsForProblemIndices(problemIndices)
      for solutionIdx in range(0, self.numSolutions):
        gflops = self.data[problemSerial+solutionIdx]
        timeUs = totalFlops / gflops / 1000
        scores[solutionIdx] += timeUs
    return scores

  ##############################################################################
  # Total Flops For Problem Indices
  def totalFlopsForProblemIndices(self, problemIndices):
    totalFlops = self.flopsPerMac
    for i in range(0, self.numIndices):
      totalFlops *= self.problemIndexToSize[i][problemIndices[i]]
    return totalFlops

  ##############################################################################
  # Remove Solution
  def removeSolution(self, removeSolutionIdx):

    # temporarily move current to old
    oldSolutions = self.solutions
    oldNumSolutions = self.numSolutions
    oldData = self.data
    oldTotalSize = self.totalSize

    # update solutions
    self.solutions = []
    for i in range(0, oldNumSolutions):
      if i != removeSolutionIdx:
        self.solutions.append(oldSolutions[i])
    self.solutionMinNaming = Solution.getMinNaming(self.solutions)
    self.solutionNames = []
    self.solutionTiles = []
    for solution in self.solutions:
      self.solutionNames.append(Solution.getNameMin(solution, \
          self.solutionMinNaming))
      self.solutionTiles.append("%ux%u"%(solution["MacroTile0"], \
          solution["MacroTile1"]))
    self.numSolutions = len(self.solutions)

    # update data
    self.totalSize = self.totalProblems * self.numSolutions
    self.data = array.array('f', [0]*self.totalSize)
    for problemIdx in range(0, self.totalProblems):
      newSolutionIdx = 0
      for oldSolutionIdx in range(0, oldNumSolutions):
        if oldSolutionIdx != removeSolutionIdx:
          self.data[problemIdx*self.numSolutions+newSolutionIdx] \
              = oldData[problemIdx*oldNumSolutions+oldSolutionIdx]
          newSolutionIdx += 1

  ##############################################################################
  # Recommended Index Order
  # TODO, this may depend on transposes
  def recommendedIndexOrder(self):
    order = []
    for i in range(0, self.numIndices):
      if i != self.idxU and i != self.idx1 and i != self.idx0:
        order.append(i)
    order.append(self.idxU)
    order.append(self.idx0)
    order.append(self.idx1)
    return order

  ##############################################################################
  # Print Data
  def printData(self):
    print2("serial; idxD0, idxD1, idxDU, idxOthers; sizeD0, sizeD1, sizeDU, sizeOthers; sol0, sol1, sol2, ...")
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
      print2(s)
      indices[0] += 1
      for i in range(1, self.numIndices):
        if indices[i-1] >= self.numProblemSizes[i-1]:
          indices[i-1] = 0
          indices[i] += 1

  ##############################################################################
  # Problem Indices For Range
  def problemIndicesForRange(self, indexRange):
    problemIndexList = []
    problemIndices = []
    for idx in indexRange:
      problemIndices.append(idx[0])
    moreProblems = True
    while moreProblems:
      problemIndexList.append(deepcopy(problemIndices))
      # next problem
      problemIndices[0] += 1
      for i in range(0, self.numIndices):
        if problemIndices[i] >= indexRange[i][1]:
          if i == self.numIndices-1:
            moreProblems = False
            break
          else:
            problemIndices[i] = indexRange[i][0]
            problemIndices[i+1] += 1
        else:
          break
    return problemIndexList

  ##############################################################################
  # Get Winner For Problem
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
  # Get Size Free
  def getSizeFree(self, problemIndices):
    sizeFree = 1
    for i in self.rangeIndicesFree:
      sizeFree *= self.problemIndexToSize[i][problemIndices[i]]
    return sizeFree


  ##############################################################################
  # Get Size Summation
  def getSizeSummation(self, problemIndices):
    sizeSummation = 1
    for i in self.rangeIndicesSummation:
      sizeSummation *= self.problemIndexToSize[i][problemIndices[i]]
    return sizeSummation


  ##############################################################################
  # Get Item
  def __getitem__(self, indexTuple):
    indices = indexTuple[0] # in analysis order
    solutionIdx = indexTuple[1]
    serial = self.indicesToSerial(solutionIdx, indices)
    return self.data[serial]


  ##############################################################################
  # Set Item
  def __setitem__(self, indexTuple, value):
    indices = indexTuple[0] # in analysis order
    solutionIdx = indexTuple[1]
    serial = self.indicesToSerial(solutionIdx, indices )
    self.data[serial] = value


  ##############################################################################
  # Indices -> Serial
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
################################################################################
###
###   Main
###
################################################################################
################################################################################
def main(  config ):
  print2("# LibraryLogic config: %s" % config)
  print2("# DefaultAnalysisParameters: " % defaultAnalysisParameters)
  benchmarkDataPath = os.path.join(globalParameters["WorkingPath"], \
      globalParameters["BenchmarkDataPath"])
  pushWorkingPath(globalParameters["LibraryLogicPath"])

  # Assign Defaults
  analysisParameters = {}
  for parameter in defaultAnalysisParameters:
    assignParameterWithDefault(analysisParameters, parameter, config, \
        defaultAnalysisParameters)

  print1("")
  print1(HR)
  print1("# Analysing data in %s" % globalParameters["BenchmarkDataPath"])
  for parameter in analysisParameters:
    print2("#   %s: %s" % (parameter, analysisParameters[parameter]))
  print1(HR)
  print1("")



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
    YAMLIO.writeLibraryLogicForProblemType(globalParameters["WorkingPath"], \
        schedulePrefix, logic)

  popWorkingPath()

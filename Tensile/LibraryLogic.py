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
import os
import os.path
import array
import csv
from sys import stdout

from copy import deepcopy

from Common import print1, print2, printWarning, HR, printExit, defaultAnalysisParameters, globalParameters, pushWorkingPath, popWorkingPath, assignParameterWithDefault
from SolutionStructs import Solution
import YAMLIO

################################################################################
# Analyze Problem Type
################################################################################
def analyzeProblemType( problemType, problemSizeGroups, inputParameters ):
  print2(HR)
  print1("# %s" % problemType)

  solutionsList = []
  problemSizesList = []
  dataFileNameList = []
  for problemSizeGroup in problemSizeGroups:
    problemSizes = problemSizeGroup[0]
    dataFileName = problemSizeGroup[1]
    dataFileNameList.append(dataFileName)
    solutionsFileName = problemSizeGroup[2]
    #print "  problemSizes:", problemSizes
    #print "# DataFileName:", dataFileName
    #print "  solutionsFileName:", solutionsFileName

    ######################################
    # Read Solutions
    (problemSizes, solutions) = YAMLIO.readSolutions(solutionsFileName)
    problemSizesList.append(problemSizes)
    solutionsList.append(solutions)
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
  # Create Logic Analyzer
  logicAnalyzer = LogicAnalyzer( problemType, problemSizesList, solutionsList, \
      dataFileNameList, inputParameters)

  ######################################
  # Remove invalid solutions
  logicAnalyzer.removeInvalidSolutions()

  ######################################
  # Remove least important solutions
  logicAnalyzer.removeLeastImportantSolutions()

  ######################################
  # Print solutions used
  print1("Solutions Used:")
  for i in range(0, len(logicAnalyzer.solutions)):
    print1("(%2u) %s" % (i, Solution.getNameFull(logicAnalyzer.solutions[i])))

  ######################################
  # Correct outliers
  if inputParameters["SmoothOutliers"]:
    logicAnalyzer.smoothOutliers()
  numProblemSizes = logicAnalyzer.numProblemSizes
  # print all 2D
  numPermutations = 1
  permutations = []
  for i in range(0, logicAnalyzer.numIndices):
    if i != logicAnalyzer.idx0 and i != logicAnalyzer.idx1:
      numPermutations *= numProblemSizes[i]
  #print numPermutations
  for j in range(0, numPermutations):
    pIdx = j
    permutation = []
    for i in range(0, logicAnalyzer.numIndices):
      if i != logicAnalyzer.idx0 and i != logicAnalyzer.idx1:
        npsi = numProblemSizes[i]
        permutation.append(pIdx%npsi)
        pIdx /= numProblemSizes[i]
    permutations.append(permutation)
  #print permutations
  for permutation in permutations:
    logicAnalyzer.print2D(permutation)

  ######################################
  # Range Logic
  rangeLogic = logicAnalyzer.enRule(0, logicAnalyzer.globalIndexRange)
  print2("# Final Range Logic:")
  print2(rangeLogic)
  logicComplexity = [0]*logicAnalyzer.numIndices
  logicAnalyzer.scoreLogicComplexity(rangeLogic, logicComplexity)
  print2("# Range Logic Complexity: %s" % logicComplexity)
  score = logicAnalyzer.scoreRangeForLogic( \
      logicAnalyzer.globalIndexRange, rangeLogic)
  print1("\n# Score: %.0f ms" % (score/1000))
  logicAnalyzer.prepareLogic(rangeLogic) # convert indices to sizes, -1

  ######################################
  # Range Logic
  exactLogic = logicAnalyzer.exactWinners
  print1("# Exact Logic:\n")
  print1("%s"%exactLogic)

  return (problemType, logicAnalyzer.solutions, logicAnalyzer.indexOrder, \
       exactLogic, rangeLogic )



################################################################################
# LogicAnalyzer
################################################################################
class LogicAnalyzer:

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
  def __init__(self, problemType, problemSizesList, solutionsList, \
      dataFileNameList, inputParameters):

    # parameters
    self.parameters = inputParameters

    # problem type
    self.problemType = problemType
    self.idx0 = self.problemType["Index0"]
    self.idx1 = self.problemType["Index1"]
    self.idxU = self.problemType["IndexUnroll"]

    # merge solutions from size groups
    # solutions needs to be a set, and offset needs to be mapping
    self.numSolutionsPerGroup = []
    self.solutionGroupMap = []
    self.solutions = []
    for solutionGroupIdx in range(0, len(solutionsList)):
      solutionGroup = solutionsList[solutionGroupIdx]
      self.numSolutionsPerGroup.append(len(solutionGroup))
      self.solutionGroupMap.append({})
      for solutionIdx in range(0, len(solutionGroup)):
        solution = solutionGroup[solutionIdx]
        if solution not in self.solutions:
          self.solutions.append(solution)
        sIdx = self.solutions.index(solution)
        self.solutionGroupMap[solutionGroupIdx][solutionIdx] = sIdx
    # print "SolutionGroupMap", self.solutionGroupMap
    self.numSolutions = len(self.solutions)
    self.solutionMinNaming = Solution.getMinNaming(self.solutions)
    self.solutionNames = []
    self.solutionTiles = []
    for solution in self.solutions:
      self.solutionNames.append(Solution.getNameMin(solution, \
          self.solutionMinNaming))
      self.solutionTiles.append("%ux%u"%(solution["MacroTile0"], \
          solution["MacroTile1"]))
    self.flopsPerMac = self.problemType["DataType"].flopsPerMac()

    # merge problem sizes from size groups
    #self.numIndices = len(problemSizesList[0].numProblemSizes)
    self.numIndices = self.problemType["TotalIndices"]
    unifiedProblemSizes = []
    for i in range(0, self.numIndices):
      unifiedProblemSizes.append(set())
    self.exactProblemSizes = set()
    self.rangeProblemSizes = set()
    for problemSizes in problemSizesList:
      # add exacts
      for exactSize in problemSizes.exacts:
        self.exactProblemSizes.add(tuple(exactSize))

      # add ranges
      self.rangeProblemSizes.update(problemSizes.sizes)
      for rangeSize in problemSizes.ranges:
        sizedIdx = 0
        mappedIdx = 0
        for i in range(0, self.numIndices):
          if rangeSize.indexIsSized[i]:
            index = rangeSize.indicesSized[sizedIdx]
            sizedIdx += 1
          else:
            index = rangeSize.indicesSized[ \
              rangeSize.indicesMapped[mappedIdx]]
            mappedIdx += 1
          currentSize = index[0]
          currentStride = index[1]
          while currentSize <= index[3]:
            unifiedProblemSizes[i].add(currentSize)
            currentSize += currentStride
            currentStride += index[2]
    for i in range(0, len(unifiedProblemSizes)):
      unifiedProblemSizes[i] = sorted(list(unifiedProblemSizes[i]))
    print2("UnifiedProblemSizes: %s" % unifiedProblemSizes)
    print2("ExactProblemSizes: %s" % self.exactProblemSizes)
    print2("RangeProblemSizes: %s" % self.rangeProblemSizes)

    # problem size index <-> size
    self.problemSizeToIndex = []
    self.problemIndexToSize = []
    self.numProblemSizes = []
    for i in range(0, self.numIndices):
      self.problemSizeToIndex.append({})
      self.problemIndexToSize.append([])
      for j in range(0, len(unifiedProblemSizes[i])):
        size = unifiedProblemSizes[i][j]
        self.problemSizeToIndex[i][size] = j
        self.problemIndexToSize[i].append(size)
      self.numProblemSizes.append(len(unifiedProblemSizes[i]))
    print1("NumProblemSizes: %s" % self.numProblemSizes)

    # total size of data array
    self.totalProblems = 1
    for numProblems in self.numProblemSizes:
      self.totalProblems *= numProblems
    self.totalSize = self.totalProblems * self.numSolutions
    print2("TotalProblems: %u" % self.totalProblems)
    print2("TotalSolutions: %u" % self.numSolutions)
    print2("TotalSize: %u" % self.totalSize)
    self.data = array.array('f', [-2]*self.totalSize)
    self.exactWinners = {}

    """
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
    """
    #self.rangeIndicesFree = range(0, self.problemType["NumIndicesC"])
    #self.rangeIndicesSummation = range(self.problemType["NumIndicesC"], \
    #    self.problemType["TotalIndices"])
    self.indexOrder = self.recommendedIndexOrder()
    print2("IndexOrder: %s" % self.indexOrder)
    self.globalIndexRange = []
    for i in range(0, self.numIndices):
      self.globalIndexRange.append([0, self.numProblemSizes[i]])
    self.problemIndicesForGlobalRange \
        = self.problemIndicesForRange(self.globalIndexRange)
    self.tab = [""]*self.numIndices

    ######################################
    # Read Data From CSV
    for fileIdx in range(0, len(dataFileNameList)):
      dataFileName = dataFileNameList[fileIdx]
      self.addFromCSV(dataFileName, self.numSolutionsPerGroup[fileIdx], \
          self.solutionGroupMap[fileIdx])

    #print self.data
    # map exact problem sizes to solutions
    print "ExactWinners", self.exactWinners


  ##############################################################################
  # ENTRY: Add From CSV
  ##############################################################################
  def addFromCSV(self, dataFileName, numSolutions, solutionMap):

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
    rowLength = solutionStartIdx + numSolutions

    # iterate over rows
    rowIdx = 0
    for row in csvFile:
      rowIdx+=1
      if rowIdx == 1:
        continue
      else:
        #if len(row) < rowLength:
        #  printWarning("CSV File %s row %u doesn't have %u elements; ignoring remainer of file." \
        #      % (dataFileName, rowIdx, rowLength) )
        #  break

        # get problem size
        problemSize = []
        for i in range(problemSizeStartIdx, totalSizeIdx):
          problemSize.append(int(row[i]))
        problemSize = tuple(problemSize)

        # Exact Problem Size
        if problemSize in self.exactProblemSizes:
          print "hi"
          # solution gflops
          solutionIdx = 0
          winnerIdx = -1
          winnerGFlops = -1
          for i in range(solutionStartIdx, rowLength):
            gflops = float(row[i])
            if gflops > winnerGFlops:
              winnerIdx = solutionIdx
              winnerGFlops = gflops
            solutionIdx += 1
          if problemSize in self.exactWinners:
            if winnerGFlops > self.exactWinners[problemSize][1]:
              self.exactWinners[problemSize] = [solutionMap[winnerIdx], winnerGFlops]
          else:
            self.exactWinners[problemSize] = [solutionMap[winnerIdx], winnerGFlops]

        # Range Problem Size
        elif problemSize in self.rangeProblemSizes:
          problemIndices = []
          for i in range(0, self.numIndices):
            problemIndices.append(self.problemSizeToIndex[i][problemSize[i]])
          print problemIndices
          serialIdx = self.indicesToSerial(0, problemIndices)
          # solution gflops
          solutionIdx = 0
          for i in range(solutionStartIdx, rowLength):
            gflops = float(row[i])
            self.data[serialIdx+solutionMap[solutionIdx]] = gflops
            solutionIdx += 1

        # Unknown Problem Size
        else:
          printExit("Huh? %s has ProblemSize %s which isn't in its yaml" \
              % ( dataFileName, list(problemSize)) )
    if rowIdx < 2:
      printExit("CSV File %s only has %u row(s); prior benchmark must not have run long enough to produce data." \
          % (dataFileName, rowIdx) )
    print self.data


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
        problemSerial = self.indicesToSerial(0, problemIndices)
        for solutionIdx in range(0, self.numSolutions):
          gflops = self.data[problemSerial+solutionIdx]
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
    while len(self.solutions) > 1:
      lisTuple = self.leastImportantSolution()
      if lisTuple != None:
        lisIdx = lisTuple[0]
        lisPercSaved = lisTuple[1]
        if lisPercSaved < self.parameters["SolutionImportanceMin"]:
          print1("# Removing Unimportant Solution: %u %s" \
              % (lisIdx, self.solutionNames[lisIdx]) )
          self.removeSolution(lisIdx)
          continue
        else:
          break
      else: # no more lis, remainders are exact winner
        break


  ##############################################################################
  # ENTRY: Smooth Outliers
  ##############################################################################
  def smoothOutliers(self):
    problemSizes = [0]*self.numIndices
    for problemIndices in self.problemIndicesForGlobalRange:
      problemSerial = self.indicesToSerial(0, problemIndices)

      for solutionIdx in range(0, self.numSolutions):
        gflops = self.data[problemSerial+solutionIdx]
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
            if neighborBeforeGFlops > gflops \
                and neighborAfterGFlops < gflops :
              smoothProblem = True
        if smoothProblem:
          s = ""
          for i in range(0, self.numIndices):
            problemSizes[i] = self.problemIndexToSize[i][problemIndices[i]]
            s += "%u, " % problemSizes[i]
          new = sum(neighborGFlops)/len(neighborGFlops)
          old = self.data[problemSerial+solutionIdx]
          s += "%f -> %f" % (old, new)
          self.data[problemSerial+solutionIdx] \
              = sum(neighborGFlops)/len(neighborGFlops)


  ##############################################################################
  # ENTRY: En Rule
  # currentIndexIndex = 0, 1, 2, 3...
  # currentIndexRange will have only 1 size for prior indices (unless initial)
  #
  # Rule:
  # [128, [
  #         [64, [
  #                [16, 0],
  #                [2880,1]
  #              ]
  #         ],
  #         [96, [
  #                [16, 0],
  #                [64, 1]
  #              ]
  #         ]
  #       ]
  # ], another
  #
  #
  ##############################################################################
  def enRule(self, currentIndexIndex, currentIndexRange):
    cii = currentIndexIndex
    if currentIndexIndex == 0:
      self.tab[cii] = "[] "
    elif currentIndexIndex == 1:
      self.tab[cii] = "[%2u] " % ( \
          currentIndexRange[self.indexOrder[0]][0])
    elif currentIndexIndex == 2:
      self.tab[cii] = "[%2u,%2u] " % ( \
          currentIndexRange[self.indexOrder[0]][0], \
          currentIndexRange[self.indexOrder[1]][0])
    elif currentIndexIndex == 3:
      self.tab[cii] = "[%2u,%2u,%2u] " % ( \
          currentIndexRange[self.indexOrder[0]][0], \
          currentIndexRange[self.indexOrder[1]][0], \
          currentIndexRange[self.indexOrder[2]][0])
    elif currentIndexIndex == 4:
      self.tab[cii] = "[%2u,%2u,%2u,%2u] " % ( \
          currentIndexRange[self.indexOrder[0]][0], \
          currentIndexRange[self.indexOrder[1]][0], \
          currentIndexRange[self.indexOrder[2]][0], \
          currentIndexRange[self.indexOrder[3]][0])
    tab = self.tab[cii]
    if globalParameters["PrintLevel"] == 1:
      stdout.write("\n%s"%tab)
    currentIndex = self.indexOrder[currentIndexIndex]
    print2("%senRule(%s)" % (tab, currentIndexRange))
    nextIndexIndex = currentIndexIndex+1
    nextIndexRange = deepcopy(currentIndexRange)
    isLastIndex = currentIndexIndex == self.numIndices-1
    ruleList = []

    ########################################
    # if there's only 1 problem size here
    ########################################
    if currentIndexRange[currentIndex][1] \
        - currentIndexRange[currentIndex][0] == 1:

      ########################################
      # this is last index, so just return fastest solution
      if isLastIndex:
        winnerIdx = self.winnerForRange(currentIndexRange)
        if winnerIdx < 0:
          return None
        ruleList.append([-1, winnerIdx])
        if globalParameters["PrintLevel"] == 1:
          stdout.write("%")

      ########################################
      # this isn't last index, so just recursively return next index
      else:
        #print2("%sreturning early enRule(%s)" \
        #    % (tab, nextIndexRange) )
        nextRule = self.enRule(nextIndexIndex, nextIndexRange)
        if nextRule == None:
          return None
        rule = [ -1, nextRule ]
        ruleList.append(rule)
        if globalParameters["PrintLevel"] == 1:
          stdout.write("%")

    ########################################
    # full iterative rule list
    ########################################
    else:
      #print tab, "Initial Rule"

      ########################################
      # create initial rule
      if isLastIndex:
        for problemIndex in range(currentIndexRange[currentIndex][0], \
            currentIndexRange[currentIndex][1]):
          nextIndexRange[currentIndex][0] = problemIndex
          nextIndexRange[currentIndex][1] = problemIndex+1
          winnerIdx = self.winnerForRange(nextIndexRange)
          #print "InitialWinner:", winnerIdx, " @ ", problemIndex
          initialRule = [ currentIndexRange[currentIndex][0], winnerIdx]
          if winnerIdx >= 0:
            break
          # print initial winner
        if winnerIdx < 0:
          return None
        """
        print2("Winner@ %u, %u, %u, %u is S[%u]: %s" % ( \
            self.problemIndexToSize[0][nextIndexRange[0][0]], \
            self.problemIndexToSize[1][nextIndexRange[1][0]], \
            self.problemIndexToSize[2][nextIndexRange[2][0]], \
            self.problemIndexToSize[3][nextIndexRange[3][0]], \
            winnerIdx, \
            self.solutionNames[winnerIdx] ) )
        """
        print "InitialRule", initialRule
      else:
        #print2("%sinitialRule(%s)" % (tab, nextIndexRange))
        nextRule = self.enRule(nextIndexIndex, nextIndexRange)
        if nextRule == None:
          return None
        initialRule = [ currentIndexRange[currentIndex][0], nextRule ]
        #print2("%sinitialRule(%s) DONE" % (tab, nextIndexRange))
      ruleList.append(initialRule)
      print tab, ruleList
      if globalParameters["PrintLevel"] == 1:
        stdout.write("#")

      ########################################
      # for all problem indices in this index
      #print tab, "Improve Rule"
      for problemIndex in range(currentIndexRange[currentIndex][0]+1, \
          currentIndexRange[currentIndex][1]):
        nextIndexRange[currentIndex][0] = problemIndex
        nextIndexRange[currentIndex][1] = problemIndex+1
        priorRule = ruleList[len(ruleList)-1]
        priorRuleForSize = deepcopy(priorRule)
        priorRuleForSize[0] = problemIndex

        if isLastIndex:
          winnerIdx = self.winnerForRange(nextIndexRange)
          # if no solutions benchmarked for this problem size, continue
          if winnerIdx < 0:
            ruleList[len(ruleList)-1][0] = problemIndex # NO_UPDATE
            print tab, ruleList, "Updating b/c None"
            if globalParameters["PrintLevel"] == 1:
              stdout.write(" ")
            continue
          else:
            candidateRule = [ problemIndex, winnerIdx]
        else:
          nextRule = self.enRule(nextIndexIndex, nextIndexRange)
          if nextRule == None:
            ruleList[len(ruleList)-1][0] = problemIndex # NO_UPDATE
            print tab, ruleList, "Updating b/c None"
            if globalParameters["PrintLevel"] == 1:
              stdout.write(" ")
            continue
          else:
            candidateRule = [ problemIndex, nextRule ]

        ########################################
        # candidate same as prior
        if candidateRule[1] == priorRule[1]:
          #print2("%sP[%2u]: same" % (tab, problemIndex))
          ruleList[len(ruleList)-1][0] = problemIndex # NO_UPDATE
          print tab, ruleList, "Updating b/c Same"
          if globalParameters["PrintLevel"] == 1:
            stdout.write(" ")
          continue

        ########################################
        # compare candidate vs prior
        else:
          #print2("%sScoring P:%s for Prior=%s, Cand=%s" \
          #    % ( tab, nextIndexRange, priorRuleForSize, candidateRule))
          # score prior

          """
          priorRuleScore = self.scoreRangeForLogic(nextIndexRange, \
              [priorRuleForSize])
          logicComplexity = [0]*self.numIndices
          self.scoreLogicComplexity( \
              [priorRuleForSize], logicComplexity)
          priorRuleScore += self.parameters["BranchPenalty"] \
              * sum(logicComplexity)
          # score candidate
          candidateRuleScore = self.scoreRangeForLogic(nextIndexRange, \
              [candidateRule])
          #print "CRS", candidateRuleScore
          logicComplexity = [0]*self.numIndices
          self.scoreLogicComplexity( \
              [candidateRule], logicComplexity)
          candidateRuleScore += self.parameters["BranchPenalty"] \
              * sum(logicComplexity)
          candidateRuleScore += self.parameters["BranchPenalty"] # penalize
          candidateFaster = candidateRuleScore < priorRuleScore
          print2("%sP[%2u]: %s %s~%.0fus < %s~%.0fus" % (tab, problemIndex, \
              "wins" if candidateFaster else "same", \
              candidateRule, candidateRuleScore, priorRuleForSize, \
              priorRuleScore ))
          """

          ########################################
          # candidate wins
          if True: # or candidateRuleScore < priorRuleScore:
            ruleList.append(candidateRule)
            print tab, ruleList, "Appending b/c Different"
            if globalParameters["PrintLevel"] == 1:
              stdout.write("#")

          ########################################
          # prior wins
          else:
            print "PRIOR_WINS"
            if globalParameters["PrintLevel"] == 1:
              stdout.write(".")
            ruleList[len(ruleList)-1][0] = problemIndex # NO_UPDATE
            print tab, ruleList

    print2("%sReturning RuleList: %s" % (tab, ruleList))
    return ruleList



  ##############################################################################
  ##############################################################################
  ###
  ###  Mid-Level Functions
  ###
  ##############################################################################
  ##############################################################################



  ##############################################################################
  # Prepare Logic
  # convert threshold indices to sizes
  # last threshold = -1
  ##############################################################################
  def prepareLogic(self, logic):
    depth = self.getLogicDepth(logic)
    if depth == 0: return
    indexIndex = self.numIndices - depth
    index = self.indexOrder[indexIndex]
    for i in range(0, len(logic)):
      if i == len(logic)-1:
        logic[i][0] = -1
      else:
        logic[i][0] = self.problemIndexToSize[index][logic[i][0]]
      self.prepareLogic(logic[i][1])


  ##############################################################################
  # Print2D
  ##############################################################################
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
        problemSerial = self.indicesToSerial(0, problemIndices)
        for sIdx in range(0, self.numSolutions):
          sss[sIdx] += ",%f" % self.data[problemSerial+sIdx]
        winnerIdx = 0
        secondIdx = 1
        winnerGFlops = self.data[problemSerial+0]
        secondGFlops = 1e-9
        for solutionIdx in range(1, self.numSolutions):
          solutionSerialIdx = problemSerial + solutionIdx
          solutionGFlops = self.data[solutionSerialIdx]
          if solutionGFlops > winnerGFlops:
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
  ##############################################################################
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

      problemSerial = self.indicesToSerial(0, problemIndices)
      if self.data[problemSerial+0] > self.data[problemSerial+1]:
        winnerIdx = 0
        winnerGFlops = self.data[problemSerial+0]
        secondIdx = 1
        secondGFlops = self.data[problemSerial+1]
      else:
        winnerIdx = 1
        winnerGFlops = self.data[problemSerial+1]
        secondIdx = 0
        secondGFlops = self.data[problemSerial+0]

      for solutionIdx in range(2, self.numSolutions):
        solutionSerialIdx = problemSerial + solutionIdx
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
    totalSavedMs = max(1, totalSavedMs)
    solutionImportance.sort(key=lambda x: x[1])
    for i in range(0, self.numSolutions):
      solutionIdx = solutionImportance[0][0]
      canRemove = True
      for j in self.exactWinners:
        winnerIdx = self.exactWinners[j]
        if solutionIdx == winnerIdx: # exact winners are important
          canRemove = False
          break
      if canRemove:
        return ( solutionImportance[0][0], \
            solutionImportance[0][1] / totalSavedMs, \
            solutionImportance[0][2] / totalWins, \
            solutionImportance[0][3] / totalExecMs )
    return None


  ##############################################################################
  # Remove Solution
  ##############################################################################
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
    for problemIndex in range(0, self.totalProblems):
      newSolutionIdx = 0
      for oldSolutionIdx in range(0, oldNumSolutions):
        if oldSolutionIdx != removeSolutionIdx:
          self.data[problemIndex*self.numSolutions+newSolutionIdx] \
              = oldData[problemIndex*oldNumSolutions+oldSolutionIdx]
          newSolutionIdx += 1


  ##############################################################################
  # Score Range For Logic
  ##############################################################################
  def scoreRangeForLogic(self, indexRange, logic):
    depth = self.getLogicDepth(logic)
    depth = self.numIndices - depth
    fullLogic = deepcopy(logic)
    for i in range(0, depth):
      fullLogic = [[-1, fullLogic]]
    fullLogic = fullLogic
    return self.scoreRangeForFullLogic(depth, indexRange, fullLogic)

  ##############################################################################
  # Score Range For Full Logic
  ##############################################################################
  def scoreRangeForFullLogic(self, depth, indexRange, logic):
    score = 0
    for problemIndices in self.problemIndicesForRange(indexRange):
      problemSerial = self.indicesToSerial(0, problemIndices)
      totalFlops = self.totalFlopsForProblemIndices(problemIndices)
      solutionIdx = self.getSolutionForProblemIndicesUsingLogic( \
          problemIndices, logic)
      solutionGFlops = self.data[problemSerial + solutionIdx]
      solutionGFlops = max(1E-9, solutionGFlops)
      timeUs = totalFlops / solutionGFlops / 1000
      score += timeUs
    return score

  ##############################################################################
  # Get Solution For Problem Indices Using Logic
  ##############################################################################
  def getSolutionForProblemIndicesUsingLogic(self, problemIndices, logic):
    currentProblemIndices = self.toIndexOrder(problemIndices)
    currentLogic = logic
    for i in range(0, self.numIndices):
      currentSizeIndex = currentProblemIndices[0]
      for j in range(0, len(currentLogic)):
        if currentLogic[j][0] < 0:
          currentProblemIndices = currentProblemIndices[1:]
          currentLogic = currentLogic[j][1]
          break
        if currentLogic[j][0] >= 0:
          if currentSizeIndex <= currentLogic[j][0]:
            currentProblemIndices = currentProblemIndices[1:]
            currentLogic = currentLogic[j][1]
            break
    return currentLogic


  ##############################################################################
  ##############################################################################
  ###
  ###  Helper / Low-Level Functions
  ###
  ##############################################################################
  ##############################################################################


  ##############################################################################
  # Get Winner For Problem
  def getWinnerForProblem(self, problemIndices):
    problemSerial = self.indicesToSerial(0, problemIndices)
    winnerIdx = -1
    winnerGFlops = -1
    for solutionIdx in range(0, self.numSolutions):
      solutionSerialIdx = problemSerial + solutionIdx
      solutionGFlops = self.data[solutionSerialIdx]
      solutionGFlops = max(1E-9, solutionGFlops)
      if solutionGFlops > winnerGFlops:
        winnerIdx = solutionIdx
        winnerGFlops = solutionGFlops
    # print "Winner %u %f" % (winnerIdx, winnerGFlops)
    return (winnerIdx, winnerGFlops)


  ##############################################################################
  # Winner For Range, -1 if nothing benchmarked
  def winnerForRange(self, indexRange):
    if self.numSolutions == 1:
      return 0
    else:
      scores = self.scoreRangeForSolutions(indexRange)
      # print "WinnerForRange", indexRange, scores
      winnerIdx = 0
      hasWinner = False
      for solutionIdx in range(1, self.numSolutions):
        if scores[solutionIdx] < scores[winnerIdx]:
          winnerIdx = solutionIdx
          hasWinner = True
        elif scores[solutionIdx] > scores[winnerIdx]:
          hasWinner = True
        else:
          pass # still no winner

      return winnerIdx if hasWinner else -1


  ##############################################################################
  # Score (microseconds) Range For Solutions
  def scoreRangeForSolutions(self, indexRange):
    scores = [0]*self.numSolutions
    for problemIndices in self.problemIndicesForRange(indexRange):
      problemSerial = self.indicesToSerial(0, problemIndices)
      totalFlops = self.totalFlopsForProblemIndices(problemIndices)
      for solutionIdx in range(0, self.numSolutions):
        gflops = self.data[problemSerial+solutionIdx]
        gflops = max(1E-9, gflops)
        timeUs = totalFlops / gflops / 1000
        scores[solutionIdx] += timeUs
    return scores


  ##############################################################################
  # Score Logic Complexity
  def scoreLogicComplexity(self, logic, logicComplexity):
    depth = self.getLogicDepth(logic)
    if depth == 0: return
    depth = self.numIndices - depth
    currentLogic = logic
    for i in range(0, len(logic)):
      logicComplexity[depth] += 1
      self.scoreLogicComplexity(logic[i][1], logicComplexity)


  ##############################################################################
  # Get Logic Depth
  def getLogicDepth(self, logic):
    obj = logic
    depth = 0
    while isinstance(obj, list):
      obj = obj[0][1]
      depth += 1
    return depth


  ##############################################################################
  # To Index Order
  def toIndexOrder(self, problemIndices):
    ordered = []
    for i in self.indexOrder:
      ordered.append(problemIndices[i])
    return ordered


  ##############################################################################
  # Total Flops For Problem Indices
  def totalFlopsForProblemIndices(self, problemIndices):
    totalFlops = self.flopsPerMac
    for i in range(0, self.numIndices):
      totalFlops *= self.problemIndexToSize[i][problemIndices[i]]
    return totalFlops


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
  # Get Size Free
  #def getSizeFree(self, problemIndices):
  #  sizeFree = 1
  #  for i in self.rangeIndicesFree:
  #    sizeFree *= self.problemIndexToSize[i][problemIndices[i]]
  #  return sizeFree


  ##############################################################################
  # Get Size Summation
  #def getSizeSummation(self, problemIndices):
  #  sizeSummation = 1
  #  for i in self.rangeIndicesSummation:
  #    sizeSummation *= self.problemIndexToSize[i][problemIndices[i]]
  #  return sizeSummation


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
  #problemTypeTuples = []
  problemTypes = {}
  if not os.path.exists(benchmarkDataPath):
    printExit("Path doesn't exist: %s" % benchmarkDataPath)
  fileNames = os.listdir(benchmarkDataPath)
  fileNames = sorted(fileNames)
  for fileName in fileNames:
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
      if problemType not in problemTypes:
        problemTypes[problemType] = []
      problemTypes[problemType].append( (problemSizes, \
          dataFileName, solutionsFileName) )
      #if problemTypeTuple not in problemTypeTuples:
      #  problemTypeTuples.append(problemTypeTuple)

  # Run Analysis
  #schedulePrefix = globalParameters["Name"]
  schedulePrefix = config["ScheduleName"]
  deviceNamesForSchedule = config["DeviceNames"]
  for problemType in problemTypes:
    logicTuple = analyzeProblemType( problemType, problemTypes[problemType], \
        analysisParameters )
    YAMLIO.writeLibraryLogicForSchedule(globalParameters["WorkingPath"], \
        schedulePrefix, deviceNamesForSchedule, logicTuple)

  popWorkingPath()


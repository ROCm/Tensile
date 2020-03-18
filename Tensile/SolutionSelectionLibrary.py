################################################################################
# Copyright (C) 2016-2019 Advanced Micro Devices, Inc. All rights reserved.
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

from .Common import printExit
from .SolutionStructs import Solution
from . import YAMLIO

import csv

def getSummationKeys(header):
  keys=[]
  for i in range(7, len(header)):
    keystr = header[i].split("=")[1].strip()
    key = int(keystr)
    keys.append(key)
  return keys

def makeKey(row):
  key=row[3]
  for i in range(4, 7):
    key += "_%s" % row[i].strip()
  return key

def getSolutionBaseKey (solution):

  macroTile0 = solution["MacroTile0"]
  macroTile1 = solution["MacroTile1"]
  globalSplitU = solution["GlobalSplitU"]
  localSplitU = solution["WorkGroup"][2]

  key = "%s_%s_%s_%s" % (macroTile0, macroTile1, localSplitU, globalSplitU)

  return key

def updateIfGT(theDicrionary, theKey, theValue):
  if not theKey in theDicrionary:
    theDicrionary[theKey] = theValue
  else:
    theOldValue = theDicrionary[theKey]
    if theValue > theOldValue:
      theDicrionary[theKey] = theValue


def updateValidSolutions(validSolutions, analyzerSolutions, solutionMinNaming):
  solutionsStartIndex = len(analyzerSolutions)
  validSelectionSolutionsIncluded = []
  validSelectionSolutionsRemainder = []
  selectionSolutionsIds = set([])
  for validSelectionSolution in validSolutions:
    (validSolution, validSolutionInfo) = validSelectionSolution
    if validSolution in analyzerSolutions:
      validExactSolutionIndex = analyzerSolutions.index(validSolution)
      selectionSolutionsIds.add(validExactSolutionIndex)
      validExactSolution = analyzerSolutions[validExactSolutionIndex]
      validSelectionSolutionsIncluded.append((validExactSolution, validSolutionInfo))
    else:
      validSelectionSolutionsRemainder.append(validSelectionSolution)

  selectionSolutions = []
  for i in range(0 ,len(validSelectionSolutionsIncluded)):
    validSelectionSolution = validSelectionSolutionsIncluded[i]
    (validSolution, validSolutionInfo) = validSelectionSolution
    #selectionSolutionIndex = solutionsStartIndex + i
    #selectionSolutionIndex = validSolution["SolutionIndex"]
    #selectionSolutionsIds.add(selectionSolutionIndex)
    validSolution["Ideals"] = validSolutionInfo
    #selectionSolutions.append(validSolution)
    analyzerSolutions.append(validSolution)

  solutionsStartIndex = len(analyzerSolutions)


  ######################################
  # Print solutions used
  #print1("# Solutions Used:")
  #for i in range(0, len(analyzerSolutions)):
    #s = analyzerSolutions[i]
    #s["SolutionIndex"] = i
    #s["SolutionNameMin"] = Solution.getNameMin(s, solutionMinNaming)
    #print1("(%2u) %s : %s" % (i, \
    #  Solution.getNameMin(s, solutionMinNaming), \
    #  Solution.getNameFull(s)))  # this is the right name

  for i in range(0, len(validSelectionSolutionsRemainder)):
    validSelectionSolution = validSelectionSolutionsRemainder[i]
    (validSolution, validSolutionInfo) = validSelectionSolution
    selectionSolutionIndex = solutionsStartIndex + i
    #validSolution["SolutionIndex"] = selectionSolutionIndex
    selectionSolutionsIds.add(selectionSolutionIndex)
    validSolution["SolutionNameMin"] = Solution.getNameMin(validSolution, solutionMinNaming)
    validSolution["Ideals"] = validSolutionInfo
    selectionSolutions.append(validSolution)

  selectionSolutionsIdsList = list(selectionSolutionsIds)

  return selectionSolutionsIdsList


def analyzeSolutionSelectionOldClient( problemType, problemSizeGroups):

  dataFileNameList = []
  performanceMap = {}
  solutionsHash = {}
  #solutionData = {}
  #allSolutions = []

  for problemSizeGroup in problemSizeGroups:
    dataFileName = problemSizeGroup[3]
    dataFileNameList.append(dataFileName)
    solutionsFileName = problemSizeGroup[2]

    (_, solutions) = YAMLIO.readSolutions(solutionsFileName)
    if len(solutions) == 0:
      printExit("%s doesn't contains any solutions." % (solutionsFileName) )

    #solutionMinNaming = Solution.getMinNaming(solutions)

    #for solution in solutions:
    #  solutionName = Solution.getNameMin(solution, solutionMinNaming)
    #  solution["SolutionNameMin"] = solutionName
    #  allSolutions.append(solution)
      
    dataFile = open(dataFileName, "r") 
    csvFile = csv.reader(dataFile)

    rowIdx = 0
    summationKeys = None

    for row in csvFile:
      #print (row)
      if rowIdx == 0:
        print(rowIdx)
        summationKeys = getSummationKeys(row)
        #rowIdx = rowIdx + 1
      else:
        #print (len(row))
        #solution = solutions[rowIdx - 1]
        #rowIdx = rowIdx + 1

        if len(row) > 1:
          solution = solutions[rowIdx - 1]
          keyBase = makeKey(row)
          idx=7
          name = row[0]
          perfData = {}
          for summationKey in summationKeys:
            key = "%s_%s" % (keyBase,summationKey)
            value = float(row[idx])
            perfData[summationKey] = value
            idx+=1

            if not solution in solutionsHash:
              dataMap = {}
              solutionsHash[solution] = dataMap

            updateIfGT(solutionsHash[solution], summationKey, value)

            if not key in performanceMap:
              performanceMap[key] = (solution, value)
            else:
              _,valueOld = performanceMap[key]
              if value > valueOld:
                performanceMap[key] = (solution, value)
            #if key not in performanceMap:
            #  performanceMap[key] = (name,value)
            #else:
            #  (name1,value1) = performanceMap[key]
            #  if value > value1:
            #    performanceMap[key] = (name,value)
          #solutionData[name]=perfData
      rowIdx+=1

    dataFile.close()

  #validSolutionsNames = set([])
  #for key in performanceMap:
  #  (name,_) = performanceMap[key]      
  #  validSolutionsNames.add(name)
  #validSolutions = []
  #for solution in allSolutions:
  #  sname = solution["SolutionNameMin"]
  #  if sname in validSolutionsNames:
  #    solutionInfo = solutionData[sname]
  #    validSolutions.append((solution, solutionInfo))

  validSolutions = []
  validSolutionSet = set([])

  for key in performanceMap:
    solution, _ = performanceMap[key]
    validSolutionSet.add(solution)
  
  for validSolution in validSolutionSet:
    dataMap = solutionsHash[validSolution]
    validSolutions.append((validSolution,dataMap))
  return validSolutions

def analyzeSolutionSelection(problemType, selectionFileNameList, numSolutionsPerGroup, solutionGroupMap, solutionsList):

  performanceMap = {}
  solutionsHash = {}

  #numIndices = self.problemType["TotalIndices"] + problemType["NumIndicesLD"]
  totalIndices = problemType["TotalIndices"]
  summationIndex = totalIndices
  numIndices = totalIndices + problemType["NumIndicesLD"]
  problemSizeStartIdx = 1
  totalSizeIdx = problemSizeStartIdx + numIndices
  solutionStartIdx = totalSizeIdx + 1
  for fileIdx in range(0, len(selectionFileNameList)):
    solutions = solutionsList[fileIdx]
    selectionFileName = selectionFileNameList[fileIdx]
    solutionsMap = solutionGroupMap[fileIdx]
    numSolutions = numSolutionsPerGroup[fileIdx]
    rowLength = solutionStartIdx + numSolutions
    solutionBaseKeys = []

#    solutionIdx = 0
#    winnerIdx = -1
#    winnerGFlops = -1
#    for i in range(solutionStartIdx, rowLength):
#      gflops = float(row[i])
#      if gflops > winnerGFlops:
#        winnerIdx = solutionIdx
#        winnerGFlops = gflops
#      solutionIdx += 1
#    if winnerIdx != -1:
#      if problemSize in self.exactWinners:
#        if winnerGFlops > self.exactWinners[problemSize][1]:
#          #print "update exact", problemSize, "CSV index=", winnerIdx, self.exactWinners[problemSize], "->", solutionMap[winnerIdx], winnerGFlops
#          self.exactWinners[problemSize] = [solutionMap[winnerIdx], winnerGFlops]
#      else:
#        self.exactWinners[problemSize] = [solutionMap[winnerIdx], winnerGFlops]
#        #print "new exact", problemSize, "CSV index=", winnerIdx, self.exactWinners[problemSize]

    #for solutionIdx in solutionsMap:
    #  sIdx = solutionsMap[solutionIdx]
    #  solution = solutions[sIdx]
    #  baseKey = getSolutionBaseKey(solution)
    #  solutionBaseKeys.append(baseKey)

    for solution in solutions:
      baseKey = getSolutionBaseKey(solution)
      solutionBaseKeys.append(baseKey)

    selectionfFile = open(selectionFileName, "r") 
    csvFile = csv.reader(selectionfFile)

    firstRow = 0
    for row in csvFile:
      if firstRow == 0:
        firstRow += 1
      else:
        sumationId = row[summationIndex].strip()

        solutionIndex = 0
        for i in range(solutionStartIdx, rowLength):
          baseKey = solutionBaseKeys[solutionIndex]
          key = "%s_%s" % (baseKey, sumationId)
          #sIdx = solutionsMap[solutionIndex]
          #solution = solutions[sIdx]
          solution = solutions[solutionIndex]
          solutionIndex += 1
          value = float(row[i])
          if not solution in solutionsHash:
            dataMap = {}
            solutionsHash[solution] = dataMap

          updateIfGT(solutionsHash[solution], sumationId, value)
          if not key in performanceMap:
            performanceMap[key] = (solution, value)
          else:
            _,valueOld = performanceMap[key]
            if value > valueOld:
              performanceMap[key] = (solution, value)


  validSolutions = []
  validSolutionSet = set([])

  for key in performanceMap:
    solution, _ = performanceMap[key]
    validSolutionSet.add(solution)
  
  for validSolution in validSolutionSet:
    dataMap = solutionsHash[validSolution]
    validSolutions.append((validSolution,dataMap))

  return validSolutions


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

from .Common import print1, print2, HR, printExit, defaultAnalysisParameters, globalParameters, pushWorkingPath, popWorkingPath, assignParameterWithDefault, startTime, ProgressBar, printWarning
from .SolutionStructs import Solution
from . import SolutionLibrary
from . import Utils
from . import YAMLIO
from . import __version__

from copy import deepcopy
from sys import stdout
import array
import csv
import os
import time

try:
  import yaml
except ImportError:
  printExit("You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions.")

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

def analyzeSolutionSelection( problemType, problemSizeGroups):

  dataFileNameList = []
  performanceMap = {}
  solutionData = {}
  allSolutions = []

  for problemSizeGroup in problemSizeGroups:
    problemSizes = problemSizeGroup[0]
    dataFileName = problemSizeGroup[3]
    dataFileNameList.append(dataFileName)
    solutionsFileName = problemSizeGroup[2]

    (_, solutions) = YAMLIO.readSolutions(solutionsFileName)
    if len(solutions) == 0:
      printExit("%s doesn't contains any solutions." % (solutionsFileName) )

    solutionMinNaming = Solution.getMinNaming(solutions)

    
    for solution in solutions:
      solutionName = Solution.getNameMin(solution, solutionMinNaming)
      solution["SolutionNameMin"] = solutionName
      allSolutions.append(solution)
      
    dataFile = open(dataFileName, "r") 
    csvFile = csv.reader(dataFile)

    rowIdx = 0
    summationKeys = None

    for row in csvFile:
      if rowIdx == 0:
        print(rowIdx)
        summationKeys = getSummationKeys(row)
      else:
        if len(row) > 0:
          keyBase = makeKey(row)
          idx=7
          name = row[0]
          perfData = {}
          for summationKey in summationKeys:
            key = "%s_%s" % (keyBase,summationKey)
            value = float(row[idx])
            perfData[summationKey] = value
            idx+=1
            if key not in performanceMap:
              performanceMap[key] = (name,value)
            else:
              (name1,value1) = performanceMap[key]
              if value > value1:
                performanceMap[key] = (name,value)
          solutionData[name]=perfData
      rowIdx+=1

    dataFile.close()

  validSolutionsNames = set([])
  for key in performanceMap:
    (name,_) = performanceMap[key]      
    validSolutionsNames.add(name)
  validSolutions = []
  for solution in allSolutions:
    sname = solution["SolutionNameMin"]
    if sname in validSolutionsNames:
      solutionInfo = solutionData[sname]
      validSolutions.append((sname, solution, solutionInfo))
  return validSolutions


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

def writeSelectionLibraryForSchedule( filePath, schedulePrefix, architectureName, deviceNames, \
    logicTuple, solutionIdx, validSolutionsNames, solutionData):
  problemType   = deepcopy(logicTuple[0])
  solutions     = deepcopy(logicTuple[1])
  indexOrder    = deepcopy(logicTuple[2])
  exactLogic    = deepcopy(logicTuple[3])
  rangeLogic    = deepcopy(logicTuple[4])

  filename = os.path.join(filePath, "%s_%s_TensileLibrary.ylib" \
      % (schedulePrefix, str(problemType)))
  print2("# writeLogic( %s )" % ( filename ))

  data = []
  # Tensile version
  data.append({"MinimumRequiredVersion":__version__})
  # schedule name
  data.append(schedulePrefix)     # change from Tensile to vega10
  data.append(architectureName)
  # schedule device names
  data.append(deviceNames)
  # problem type
  problemTypeState = problemType.state
  problemTypeState["DataType"] = \
      problemTypeState["DataType"].value
  problemTypeState["DestDataType"] = \
      problemTypeState["DestDataType"].value
  problemTypeState["ComputeDataType"] = \
      problemTypeState["ComputeDataType"].value
  data.append(problemTypeState)
  # solutions
  solutionList = []
  for solution in solutions:
    solutionState = solution.getAttributes()
    solutionState["ProblemType"] = solutionState["ProblemType"].state
    solutionState["ProblemType"]["DataType"] = \
        solutionState["ProblemType"]["DataType"].value
    solutionState["ProblemType"]["DestDataType"] = \
        solutionState["ProblemType"]["DestDataType"].value
    solutionState["ProblemType"]["ComputeDataType"] = \
        solutionState["ProblemType"]["ComputeDataType"].value
    solutionList.append(solutionState)
  data.append(solutionList)
  # index order
  data.append(indexOrder)

  # exactLogic
  exactLogicList = []
  for key in exactLogic:
    exactLogicList.append([list(key), exactLogic[key]])
  data.append(exactLogicList)

  # rangeLogic
  data.append(rangeLogic)

  data.append(solutionIdx)
  data.append(validSolutionsNames)
  data.append(solutionData)

  newLibrary = SolutionLibrary.MasterSolutionLibrary.FromOriginalState(data, libraryOrder = ['Hardware', 'OperationIdentifier', 'Predicates', 'Matching'])
  newLibrary.applyNaming()
  YAMLIO.write(filename, Utils.state(newLibrary))

  
################################################################################
################################################################################
###
###   Main
###
################################################################################
################################################################################
def main(problemSizeGroups , schedulePrefix, architectureName, deviceNames, logicTuple ):

  performanceMap = {}
  solutionData = {}
  solutionIdx = {}
  for problemSizeGroup in problemSizeGroups:
    solutionsFileName = problemSizeGroup[2]
    selectionFileName = problemSizeGroup[3]

    if not os.path.exists(solutionsFileName):
      printExit("%s doesn't exist for %s" % (solutionsFileName, fileBase))

    if not os.path.exists(selectionFileName):
      printExit("%s doesn't exist for %s" % (selectionFileName, fileBase))

    (problemSizes, solutions) = YAMLIO.readSolutions(solutionsFileName)

    if len(solutions) == 0:
      printExit("%s doesn't contains any solutions." % (solutionsFileName) )

    selectionFile = open(selectionFileName, "r") 
    selectionFileCSV = csv.reader(selectionFile)

    rowIdx = 0
    summationKeys = None 
    for row in selectionFileCSV:
      if rowIdx == 0:
        summationKeys = getSummationKeys(row)
      else:
        if len(row) > 0:
          keyBase = makeKey(row)
          idx=7
          name = row[0]
          solutionIdx[name] = rowIdx
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
  validSolutionsNames = set([])
  for key in performanceMap:
    (name,_) = performanceMap[key]      
    validSolutionsNames.add(name)
  writeSelectionLibraryForSchedule(globalParameters["WorkingPath"], schedulePrefix, architectureName, \
    deviceNames, logicTuple, solutionIdx, validSolutionsNames, solutionData)



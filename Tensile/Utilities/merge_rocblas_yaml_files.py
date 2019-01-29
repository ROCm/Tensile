
#from copy import deepcopy
#from Common import print1, print2, printExit, HR, ensurePath

#from SolutionStructs import Solution

#from __init__ import __version__

import os
import sys
import argparse

HR = "################################################################################"

################################################################################
# Print Debug
################################################################################

def printWarning(message):
  print "Tensile::WARNING: %s" % message
  sys.stdout.flush()

def printExit(message):
  print "Tensile::FATAL: %s" % message
  sys.stdout.flush()
  sys.exit(-1)

try:
  import yaml
except ImportError:
  printExit("You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions.")

#import YAMLIO

def ensurePath( path ):
  if not os.path.exists(path):
    os.makedirs(path)
  return path

################################################################################
# Library Logic Container
################################################################################
class LibraryLogic:

  def __init__(self,filename=None):

    if filename is not None:
      print ("# Reading Library Logic: " + filename)
      try:
        stream = open(filename, "r")
      except IOError:
        printExit("Cannot open file: %s" % filename )
      data = yaml.load(stream, yaml.SafeLoader)

      self.__set_versionString(data[0]["MinimumRequiredVersion"])
      self.__set_scheduleName(data[1])
      self.__set_architectureName(data[2])
      self.__set_deviceNames(data[3])
      self.__set_problemType(data[4])
      self.__set_solutionStates(data[5])
      self.__set_indexOrder(data[6])
      self.__set_exactLogic(data[7])
      self.__set_rangeLogic(data[8])
    
      stream.close()

    else:
      self.__set_versionString(None)
      self.__set_scheduleName(None)
      self.__set_architectureName(None)
      self.__set_deviceNames(None)
      self.__set_problemType(None)
      self.__set_solutionStates(None)
      self.__set_indexOrder(None)
      self.__set_exactLogic(None)
      self.__set_rangeLogic(None)

  #versionString
  def __get_versionString(self):
    return self.__versionString

  def __set_versionString(self,value):
    self.__versionString = value

  versionString = property(__get_versionString,__set_versionString)

  #scheduleName
  def __get_scheduleName(self):
    return self.__scheduleName

  def __set_scheduleName(self, value):
    self.__scheduleName = value

  scheduleName = property(__get_scheduleName,__set_scheduleName)

  #architectureName
  def __get_architectureName(self):
    return self.__architectureName

  def __set_architectureName(self,value):
    self.__architectureName = value

  architectureName = property(__get_architectureName,__set_architectureName)

  #deviceNames
  def __get_deviceNames(self):
    return self.__deviceNames

  def __set_deviceNames(self,value):
    self.__deviceNames = value

  deviceNames = property(__get_deviceNames,__set_deviceNames)

  
  #problemTypeState
  def __get_problemType(self):
    return self.__problemType

  def __set_problemType(self,value):
    self.__problemType = value

  problemType = property(__get_problemType,__set_problemType)

  #solutionStates
  def __get_solutionStates(self):
    return self.__solutionStates

  def __set_solutionStates(self,value):
    self.__solutionStates = value

  solutionStates = property(__get_solutionStates,__set_solutionStates)

  #indexOrder
  def __get_indexOrder(self):
    return self.__indexOrder

  def __set_indexOrder(self,value):
    self.__indexOrder = value
  
  indexOrder = property(__get_indexOrder,__set_indexOrder)


  #exactLogic
  def __get_exactLogic(self):
    return self.__exactLogic

  def __set_exactLogic(self,value):
    self.__exactLogic = value

  exactLogic = property(__get_exactLogic,__set_exactLogic)

  #rangeLogic
  def __get_rangeLogic(self):
    return self.__rangeLogic

  def __set_rangeLogic(self,value):
    self.__rangeLogic = value

  rangeLogic = property(__get_rangeLogic,__set_rangeLogic)

  def writeLibraryLogic(self,filename):
  
    data = []

    if self.versionString is not None:
      data.append({"MinimumRequiredVersion":self.versionString})
    
    if self.scheduleName is not None:
      data.append(self.scheduleName)     
    
    if self.architectureName is not None:
       data.append(self.architectureName)
    
    if self.deviceNames is not None:
      data.append(self.deviceNames)

    if self.problemType is not None:
      data.append(self.problemType)

    if self.solutionStates is not None:
      data.append(self.solutionStates)
    
    if self.indexOrder is not None:
      data.append(self.indexOrder)
    
    if self.exactLogic is not None:
      data.append(self.exactLogic)
    
    if self.rangeLogic is not None:
      data.append(self.rangeLogic)

    if not data:
      printExit("No data to output")
    else:
      try:
        stream = open(filename, "w")
        yaml.safe_dump(data, stream)
        stream.close()
      except IOError:
        printExit("Cannot open file: %s" % filename)


def MergeTensileLogicFiles(origionalLibraryLogic, exactLibraryLogic):
  
  mergedLibraryLogic = LibraryLogic()

  solutionList = origionalLibraryLogic.solutionStates
  solutionListExact = exactLibraryLogic.solutionStates

  newSolutionOffset = len(solutionList)

  filterdSolutionExactList = []
  replicationMapping = {}
  idx = 0
  idxMapping = newSolutionOffset

  mergedSolutionList = []
  for solution in solutionList:
    mergedSolutionList.append(solution)

  # construct the mappings from the old exact kernal configurations
  # to their definitions in the merged files
  for solution in solutionListExact:
    if solution in solutionList:
      # if solution exists in the origional configuration the
      # its placement in the merged kernel configurations list
      # gets mapped to the pre-existing configuration
      idxOrg = solutionList.index(solution)
      replicationMapping[idx] = idxOrg

    else:
      filterdSolutionExactList.append(solution)
      # if the solution does not exist in the origional configurations
      # it gets mapped to the new offset
      replicationMapping[idx] = idxMapping
      mergedSolutionList.append(solution)
      idxMapping += 1

    idx += 1

  exactLogic = origionalLibraryLogic.exactLogic
  exactLogicExact = exactLibraryLogic.exactLogic

  filteredExactLogicExact = []
  
  # use the mapping from above to remap the exact logic
  # in the merged file
  for exact in exactLogicExact:
    # example exact entry [[123,124,1,123], [5, 4312.3]]
    # the first fiedl in [5, 4312.3] is the mapping to the 
    # kernel configuration
    kernelIndex = exact[1][0]
    
    if replicationMapping.has_key(kernelIndex):
      exact[1][0] = replicationMapping[kernelIndex]
    
    filteredExactLogicExact.append(exact)

  #sizeList, _ = zip(*exactLogicExact)
  sizeList, _ = zip(*filteredExactLogicExact)

  mergedExactLogic = []
  for logicMapping in exactLogic:
    if logicMapping[0] not in sizeList:
      mergedExactLogic.append(logicMapping)

  for logicMapping in filteredExactLogicExact:
    mergedExactLogic.append(logicMapping)

  mergedLibraryLogic.versionString = origionalLibraryLogic.versionString
  mergedLibraryLogic.scheduleName = origionalLibraryLogic.scheduleName
  mergedLibraryLogic.architectureName = origionalLibraryLogic.architectureName
  mergedLibraryLogic.deviceNames = origionalLibraryLogic.deviceNames
  mergedLibraryLogic.problemType = origionalLibraryLogic.problemType
  mergedLibraryLogic.solutionStates = mergedSolutionList
  mergedLibraryLogic.indexOrder = origionalLibraryLogic.indexOrder
  mergedLibraryLogic.exactLogic = mergedExactLogic
  mergedLibraryLogic.rangeLogic  = origionalLibraryLogic.rangeLogic

  return mergedLibraryLogic


def ProcessMergeLogicFile(exactFileName, origionalFileName, outputFileName):
  
  _, fileName = os.path.split(exactFileName)

  print ("processing file: " + fileName)

  libraryLogic = LibraryLogic(origionalFileName)
  libraryLogicExact = LibraryLogic(exactFileName)

  mergedLibraryLogic = MergeTensileLogicFiles(libraryLogic,libraryLogicExact)

  mergedLibraryLogic.writeLibraryLogic(outputFileName)

def RunMergeTensileLogicFiles():

  print ""
  print HR
  print "# Merge Library Logic"
  print HR
  print ""
  
  ##############################################################################
  # Parse Command Line Arguments
  ##############################################################################
  
  argParser = argparse.ArgumentParser()
  argParser.add_argument("OrigionalLogicPath", help="Path to the origional LibraryLogic.yaml input files.")
  argParser.add_argument("ExactLogicPath", help="Path to the exact LibraryLogic.yaml input files.")
  argParser.add_argument("OutputPath", help="Where to write library files?")

  args = argParser.parse_args()

  origionalLogicPath = args.OrigionalLogicPath
  exactLogicPath = args.ExactLogicPath
  outputPath = args.OutputPath
  print ("Origional Logic Path: " + origionalLogicPath)
  print ("Exact Logic Path: " + exactLogicPath)
  print ("OutputPath: " + outputPath)

  print ""
  ensurePath(outputPath)
  if not os.path.exists(exactLogicPath):
    printExit("LogicPath %s doesn't exist" % exactLogicPath)

  exactLogicFiles = [os.path.join(exactLogicPath, f) for f in os.listdir(exactLogicPath) \
      if (os.path.isfile(os.path.join(exactLogicPath, f)) \
      and os.path.splitext(f)[1]==".yaml")]

  #print1("# LibraryLogicFiles:" % exactLogicFiles)
  #for logicFile in logicFiles:
  #  print1("#   %s" % logicFile)

  for exactLogicFilePath in exactLogicFiles:
    _, fileName = os.path.split(exactLogicFilePath)
    #print1("#   %s" % fileName)
    origionalLogicFilePath = os.path.join(origionalLogicPath, fileName)
    #print1("#   %s" % origionalLogicFilePath)
    if os.path.isfile(origionalLogicFilePath):
      
      outputLogicFilePath = os.path.join(outputPath, fileName)

      try:
        ProcessMergeLogicFile(exactLogicFilePath, origionalLogicFilePath, outputLogicFilePath)
      except Exception as ex:
        print("Exception: {0}".format(ex))

    else:
      print ("# file does not exist in origional directory " + origionalLogicFilePath)
    

################################################################################
# Main
################################################################################
if __name__ == "__main__":
    RunMergeTensileLogicFiles()

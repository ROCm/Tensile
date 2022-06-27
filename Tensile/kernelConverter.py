from . import __version__
from . import LibraryIO
from .Common import print1,ensurePath,HR,defaultBenchmarkCommonParameters,defaultProblemType

import yaml                    
import argparse
import copy
import os
import shutil
import sys
import re

#accumulates the yaml file data to be printed in tensile input yaml file
YamlFileData = []

dataType = {
    0:"s",
    1:"d",
    2:"c",
    3:"z",
    4:"h",
    5:"4xi8",
    6:"i",
    7:"b",
    8:"i8"
} 

def kernelConverter(userArgs):
    print1("")
    print1(HR)
    print1("#")
    print1("#  Tensile Retune Library v{}".format(__version__))
    
     # argument parsing and related setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("LibLogicFile", type=os.path.realpath,
                           help="<Library logic file> to be converted to tensile input yaml file")
    argParser.add_argument("SolutionIndex", type=checkPositive,help="<Solution Index> from Library Logic File "
                           ,
                           default=None)                               
    argParser.add_argument("OutputPath", type=os.path.realpath,
                           help="<output file path> where tensile yaml file to be created")
                           
    argParser.add_argument('--skipMI','-s', action="store_true", help="Skips the MatrixInstruction field in the tensile yaml file"
                                "i.e Thread Tile and Work Group parameters without MI",required=False)
                                
    args = argParser.parse_args(userArgs)
    logicFilePath = args.LibLogicFile
    solutionIndex = args.SolutionIndex
    TensileYamlFile = args.OutputPath
    
    print1("#  Library Logic: {}".format(logicFilePath))
    print1("#  Solution Index: {}".format(solutionIndex))
    print1("#  Usage: python3 ../Tensile/bin/kernelConverter <LibraryLogicFile> <SolutionIndex> <OutputTensileYamlFile> [--SkipMI] \n")
    print1("#")
    print1(HR)
    print1("")
    
    #TensileYamlFile = ensurePath(os.path.abspath(args.OutputPath))
    libYaml = LibraryIO.readYAML(logicFilePath)
    if libYaml == "":
     raise RuntimeError("Yaml file data is empty, read yaml file :{} failed".format(logicFilePath))
    
    #reads library logic file. AllsolutionStates=>has solution data for all solution index 
    fields = LibraryIO.rawLibraryLogic(libYaml)
    (versionString, scheduleName,architectureName,deviceNames,problemTypeState,AllsolutionStates,indexOrder,exactLogic,rangeLogic, otherFields) = fields
    
    #Extract the solution data for the user specified solution Index
    currentIndexSolution  = AllsolutionStates[solutionIndex]
    if currentIndexSolution == "":
      raise RuntimeError("Could not find the matching data for the solution index:{} from the library logic file, Try different solution Index".format(solutionIndex))
    
    CheckMacroTileThreadTileWorkGroupMatches(currentIndexSolution)
    if args.skipMI != True:
       MIInstruction9Bits = Form9BitMIInstruction(currentIndexSolution)

    #Extract the problem type parameters 
    currentIndexProblemType = currentIndexSolution["ProblemType"]
    
    #Iterate over problem type parameters and form the Yaml data
    FormProblemTypeYamlData(currentIndexProblemType,versionString)
    if args.skipMI != True:
        #Iterate over Fork  parameters and form the Yaml data
        FormForkParametersYamlData(currentIndexSolution, MIInstruction9Bits)
    else:
        print1("--skipMI option specified so skipping the Matrix Instruction parameter ...\n")   
        FormForkParametersDataForNonMI(currentIndexSolution)
    
    #Forms the Library logic string
    FormProblemSizeYamlData(exactLogic,solutionIndex,scheduleName,deviceNames,architectureName)
    
    #Write the Formed Yaml data into the Yaml File 
    WriteToTensileYamlFile(TensileYamlFile)
    
def checkPositive(value):
    try:
        value = int(value)
        if value <= 0:
            raise argParser.ArgumentTypeError("{} is not a positive integer".format(value))
    except ValueError:
        raise Exception("{} is not an integer".format(value))
    return value

def FormProblemTypeYamlData(currentIndexProblemType, versionString):
    if len(currentIndexProblemType) == 0:
       raise RuntimeError("Length of problem Type Parameters is empty!!, Please re-check the library logic file !")
    
    # Form Global parameters section as well    
    YamlFileData.append("GlobalParameters:\n")
    YamlFileData.append("  MinimumRequiredVersion: {}\n\n".format(versionString['MinimumRequiredVersion']))
    YamlFileData.append("BenchmarkProblems:\n  -\n    - # ProblemType\n")       

    for problemTypeKey, problemTypeValue in currentIndexProblemType.items():
     # always print DataType, DestDataType, ComputeDataType, HighPrecisionAccumulate, OperationType fields
      if problemTypeKey == "DataType" or problemTypeKey == "DestDataType" or problemTypeKey == "ComputeDataType":
         YamlFileData.append("      {}: {}\n".format(problemTypeKey, dataType[problemTypeValue]))
         continue
      if problemTypeKey == "HighPrecisionAccumulate" or problemTypeKey == "OperationType":
         YamlFileData.append("      {}: {}\n".format(problemTypeKey, problemTypeValue))
         continue
      if problemTypeKey in defaultProblemType:
         if problemTypeValue != defaultProblemType[problemTypeKey]:
            YamlFileData.append("      {}: {}\n".format(problemTypeKey, problemTypeValue))
            
def FormForkParametersDataForNonMI(currentIndexSolution):

    YamlFileData.append("    - # BenchmarkProblemSizeGroup\n") 
    YamlFileData.append("      InitialSolutionParameters:\n") 
    YamlFileData.append("      BenchmarkCommonParameters:\n") 
    
    YamlFileData.append("      ForkParameters:\n")
    macInstruction = ""
    for forkKey, forkValue in currentIndexSolution.items():
      if forkKey == "MACInstruction":
        if forkValue == "MAC": #This is a bug in the library logic file
          macInstruction ="         - {}: ['MAD']\n".format(forkKey)
          YamlFileData.append(macInstruction)
          continue
        else: 
          macInstruction ="         - {}: {}\n".format(forkKey,[forkValue])
          YamlFileData.append(macInstruction)
          continue
        
      if forkKey != "ProblemType" and forkKey != "MatrixInstruction":
        # Find the matching index for fork key name from list of dictionaries => defaultBenchmarkCommonParameters
        index =  next((i for i,d in enumerate(defaultBenchmarkCommonParameters) if forkKey in d), None)
        if index != None:
          #Strip the leading and trailing [] brackets, trim the leading and trailing spaces
          DefaultValue = str(defaultBenchmarkCommonParameters[index][forkKey])[1:-1].strip()
          
          # remove the "" from the string
          DefaultValue = re.sub("[\"\']", "", DefaultValue)
          if str(forkValue) != DefaultValue:
            if forkKey == "ThreadTile" or forkKey == "WorkGroup":
              YamlFileData.append("         - {}: {}\n".format(forkKey, forkValue))
            else:
              YamlFileData.append("         - {}: {}\n".format(forkKey, [forkValue]))


def FormForkParametersYamlData(currentIndexSolution, MIInstruction9Bits):

    YamlFileData.append("    - # BenchmarkProblemSizeGroup\n") 
    YamlFileData.append("      InitialSolutionParameters:\n") 
    YamlFileData.append("      BenchmarkCommonParameters:\n") 
    
    YamlFileData.append("      ForkParameters:\n")
    macInstruction = ""
    for forkKey, forkValue in currentIndexSolution.items():
      if forkKey == "MACInstruction":
        if forkValue == "MAC": #This is a bug in the library logic file
          macInstruction ="         - {}: ['MAD']\n".format(forkKey)
          YamlFileData.append(macInstruction)
          continue
        else: 
          macInstruction ="         - {}: {}\n".format(forkKey,[forkValue])
          YamlFileData.append(macInstruction)
          continue
        
      if forkKey != "ProblemType" and forkKey != "ThreadTile" and forkKey != "WorkGroup":
        # Find the matching index for fork key name from list of dictionaries => defaultBenchmarkCommonParameters
        index =  next((i for i,d in enumerate(defaultBenchmarkCommonParameters) if forkKey in d), None)
        if index != None:
          if forkKey == "MatrixInstruction":
            YamlFileData.append(MIInstruction9Bits)
            YamlFileData.remove(macInstruction)
            continue
          #Strip the leading and trailing [] brackets, trim the leading and trailing spaces
          DefaultValue = str(defaultBenchmarkCommonParameters[index][forkKey])[1:-1].strip()
          
          # remove the "" from the string
          DefaultValue = re.sub("[\"\']", "", DefaultValue)
          if str(forkValue) != DefaultValue:
            YamlFileData.append("         - {}: {}\n".format(forkKey, [forkValue]))
  
def Form9BitMIInstruction(currentSolutionState):
   
   MIBlock = currentSolutionState["MIBlock"]
   #print1(" MIBlock : {0} ".format(MIBlock))
  
   MIWaveTile =  currentSolutionState["MIWaveTile"]
   MIWaveGroup = currentSolutionState["MIWaveGroup"]
   #print1(" MIWaveTile : {0} ".format(MIWaveTile))
   #print1(" MIWaveGroup : {0} ".format(MIWaveGroup))
    
   if len(MIBlock) == 0 or len(MIWaveTile) == 0  or len(MIWaveGroup) == 0:
    raise RuntimeError("Length of MIBlock:{0}, MIWave Tile: {1},MIWaveGroup : {2} cannot be empty".format(len(MIBlock),len(MIWaveTile),len(MIWaveGroup)))
    
   MIBlock1 = [MIBlock[i] for i in (0,1,2,3,4)]
   MIBlock5bits = ','.join([str(item) for item in MIBlock1])
   #print1(" MIBlock5bits : {0} ".format(MIBlock5bits))
    
   MIWaveTile2Bits = ','.join([str(item) for item in MIWaveTile])
   #print1(" MIWaveTile2Bits : {0} ".format(MIWaveTile2Bits))
    
   MIWaveGroup2Bits = ','.join([str(item) for item in MIWaveGroup])
   #print1(" MIWaveGroup2Bits : {0} ".format(MIWaveGroup2Bits))
   
   if len(MIBlock5bits) == 0 or len(MIWaveTile2Bits) == 0  or len(MIWaveGroup2Bits) == 0:
       raise RuntimeError("Length of MIBlock5bits:{0}, MIWaveGroup2Bits : {1},MIWaveGroup2Bits:{2} cannot be empty".format(len(MIBlock5bits),len(MIWaveTile2Bits),len(MIWaveGroup2Bits)))
    
   MIInstruction9Bits = "         - MatrixInstruction:\n           - ["+ MIBlock5bits + "," + MIWaveTile2Bits + "," + MIWaveGroup2Bits + "]\n"
   return MIInstruction9Bits
    
def FormProblemSizeYamlData(exactLogic,solutionIndex,scheduleName,deviceNames,architectureName):
    
    YamlFileData.append("      BenchmarkJoinParameters:\n")
    YamlFileData.append("      BenchmarkFinalParameters:\n")
    
    YamlFileData.append("         - ProblemSizes:\n")
    #Form the problem Size 
    for (size, mapping) in exactLogic:
      if mapping[0] == solutionIndex: 
        YamlFileData.append("           - Exact: %s             # Eff: %s  Solution Index: %s\n" % (size,mapping[1],mapping[0]))
        
     #Form final library logic string
    YamlFileData.append("#########################################################################################\n")
    YamlFileData.append("LibraryLogic:\n")
    YamlFileData.append("    ScheduleName: %s\n"%scheduleName)
    YamlFileData.append("    DeviceNames: %s\n"%deviceNames)
    YamlFileData.append("    ArchitectureName: \"%s\"\n"%architectureName)
        
def CheckMacroTileThreadTileWorkGroupMatches(currentSolution):

  if not currentSolution.keys() & {'MIBlock','MIWaveTile','MIWaveGroup'}:
    raise RuntimeError("one or more of these fields => MIBlock, MIWaveTile, MIWaveGroup is missing in the library logic file!! ..\n")
            
  MIBlock = currentSolution["MIBlock"]
  MIWaveTile =  currentSolution["MIWaveTile"]
  MIWaveGroup = currentSolution["MIWaveGroup"]
  
  if MIBlock == "" or MIWaveTile == "" or MIWaveGroup == "":
    raise RuntimeError("Length of MIBlock:{0}, MIWaveTile : {1},MIWaveGroup:{2} cannot be empty, Check the library logic file !\n".format(len(MIBlock),len(MIWaveTile),len(MIWaveGroup)))
    
  MatrixM,MatrixN,TT0,TT1,MT0,MT1,WG0,WG1,Waves = calculateMatrixMNThreadTileMacroTileWorkGroupParameters(MIBlock,MIWaveTile,MIWaveGroup)
 
  if not currentSolution.keys() & {'MacroTile0','MacroTile1','ThreadTile','WorkGroup','MatrixInstM','MatrixInstN'}:
    raise RuntimeError("one or more of these fields =>MacroTile0, MacroTile1, ThreadTile, WorkGroup,MatrixInstM,MatrixInstN is missing in the library logic file!! ..\n")
   
  if MT0 != currentSolution["MacroTile0"]:
   raise RuntimeError("Macro Tile0 {0} doesnot match LibLogic value {1}".format(MT0, currentSolution["MacroTile0"]))
  
  if MT1 !=  currentSolution["MacroTile1"]:
   raise RuntimeError("Macro Tile1 {0} doesnot match LibLogic value {1}".format(MT1, currentSolution["MacroTile1"]))
    
  if TT0 != int(currentSolution["ThreadTile"][0]):
   raise RuntimeError("ThreadTile0 {0} doesnot match LibLogic value {1}".format(TT0, currentSolution["ThreadTile"][0]))
   
  #print1("ThreadTile0 {0} matches LibLogic value {1}\n".format(TT0, currentSolution["ThreadTile"][0])) 
    
  if TT1 !=  int(currentSolution["ThreadTile"][1]):
   raise RuntimeError("ThreadTile1 {0} doesnot match LibLogic value {1}".format(TT1, currentSolution["ThreadTile"][1]))
   
  #print1("ThreadTile0 {1} matches LibLogic value {1}\n".format(TT1, currentSolution["ThreadTile"][1]))
    
  if WG0 !=  int(currentSolution["WorkGroup"][0]):
   raise RuntimeError("WorkGroup0 {0} doesnot match LibLogic value {1}\n".format(WG0, currentSolution["WorkGroup"][0]))
  
  #print1("WorkGroup0 {0} matches LibLogic value {1}".format(WG0, currentSolution["WorkGroup"][0]))
    
  if WG1 !=  int(currentSolution["WorkGroup"][1]):
   raise RuntimeError("WorkGroup1 {1} doesnot match LibLogic value {1}".format(WG1, currentSolution["WorkGroup"][1]))
  
  #print1("WorkGroup1 {1} does matches LibLogic value {1}\n".format(WG1, currentSolution["WorkGroup"][1]))
  
  if MatrixM !=  int(currentSolution["MatrixInstM"]):
   raise RuntimeError("MatrixInstM {0} doesnot match LibLogic value {1}".format(MatrixM, currentSolution["MatrixInstM"]))
  
  if MatrixN !=  int(currentSolution["MatrixInstN"]):
   raise RuntimeError("MatrixInstN {0} doesnot match LibLogic value {1}".format(MatrixN, currentSolution["MatrixInstN"]))

  #print1("Thread Tile, Macro Tile ,WG , Matrix M, Matrix N parameter matches from the lib logic file\n")

def calculateMatrixMNThreadTileMacroTileWorkGroupParameters(MIBlock,MIWaveTile,MIWaveGroup):

  MatrixM = int(MIBlock[0]) *int(MIBlock[4])
  MatrixN = int(MIBlock[1])
  TT0 = int(MIWaveTile[0]) 
  TT1 = int(MIWaveTile[1])*int(MIBlock[1])
  MT0 =  MatrixM*int(MIWaveTile[0])*int(MIWaveGroup[0])
  MT1 =  MatrixN*int(MIWaveTile[1])*int(MIWaveGroup[1])
  WG0 =  int(MIBlock[0]) * int(MIBlock[4])*int(MIWaveGroup[0])
  WG1 = int(MIWaveGroup[0]) * int(MIWaveGroup[1]) *64 // int(WG0)
  Waves = int(MIWaveGroup[0])*int(MIWaveGroup[1])
  
  return MatrixM,MatrixN,TT0,TT1,MT0,MT1,WG0,WG1,Waves

def WriteToTensileYamlFile(TensileYamlFile):
  try:
    os.makedirs(os.path.dirname(TensileYamlFile), exist_ok=True)
    with open(TensileYamlFile, "w") as fileHandle:
      fileHandle.writelines("%s" % place for place in YamlFileData)
      fileHandle.close()
      print1("Sucessfully created the Tensile Input Yaml File {} from library logic file !!.\n".format(TensileYamlFile))
  except (OSError, IOError) as e:
    print1("Error: Creating file {} Please provide file name in this format <filename>.yaml.".format(TensileYamlFile))
    
def main():
  kernelConverter(sys.argv[1:])

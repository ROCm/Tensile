import argparse
import Structs
import SolutionCandidateGenerator

import xml.sax
import copy
import sys
import os

def addTimeToMap( psMap, exactMatch, problem, solution, time ):
  if exactMatch.deviceProfile not in psMap:
    psMap[exactMatch.deviceProfile] = {}
  if exactMatch not in psMap[exactMatch.deviceProfile]:
    psMap[exactMatch.deviceProfile][exactMatch] = {}
  if problem not in psMap[exactMatch.deviceProfile][exactMatch]:
    psMap[exactMatch.deviceProfile][exactMatch][problem] = {}
  if solution not in psMap[exactMatch.deviceProfile][exactMatch][problem]:
    psMap[exactMatch.deviceProfile][exactMatch][problem][solution] = Structs.SolutionBenchmark()
  psMap[exactMatch.deviceProfile][exactMatch][problem][solution].times.append(time)

def addValidationToMap( psMap, exactMatch, problem, solution, validationStatus ):
  if exactMatch not in psMap:
    psMap[exactMatch] = {}
  if problem not in psMap[exactMatch]:
    psMap[exactMatch][problem] = {}
  if solution not in psMap[exactMatch][problem]:
    psMap[exactMatch][problem][solution] = Structs.SolutionBenchmark()

  if psMap[exactMatch][problem][solution].validationStatus == 0:
    psMap[exactMatch][problem][solution].validationStatus = validationStatus
  elif psMap[exactMatch][problem][solution].validationStatus != validationStatus:
    print "ERROR: conflicting validation reports"

################################################################################
# CobaltHandler
################################################################################
class CobaltHandler( xml.sax.ContentHandler ):
  def __init__(self, data, readSolutions):
    self.data = data
    self.readSolutions = readSolutions # read problems and solutions for GenBackend
    self.readProblems = not readSolutions # read problems only for GenBenchmark

    # for reading problems
    self.numProblemsAdded = 0
    self.problem = Structs.Problem()

    self.currentTensor = 0
    self.tensor = Structs.Tensor()

    self.currentIndexAssignments = 0
    self.indexAssignments = []

    # for reading solutions
    self.solution = Structs.Solution()

  def startElement(self, tag, attributes):
    if tag == "Problem": # DONE
      self.problem = Structs.Problem()
    elif tag == "Tensor": # DONE
      self.tensor = Structs.Tensor()
      dataTypeString = attributes["dataType"]
      if dataTypeString == "cobaltDataTypeHalf":
        self.tensor.dataType.value = Structs.DataType.half
      elif dataTypeString == "cobaltDataTypeSingle":
        self.tensor.dataType.value = Structs.DataType.single
      elif dataTypeString == "cobaltDataTypeDouble":
        self.tensor.dataType.value = Structs.DataType.double
      elif dataTypeString == "cobaltDataTypeComplexHalf":
        self.tensor.dataType.value = Structs.DataType.complexHalf
      elif dataTypeString == "cobaltDataTypeComplexSingle":
        self.tensor.dataType.value = Structs.DataType.complexSingle
      elif dataTypeString == "cobaltDataTypeComplexDouble":
        self.tensor.dataType.value = Structs.DataType.complexDouble
      elif dataTypeString == "cobaltDataTypeComplexConjugateHalf":
        self.tensor.dataType.value = Structs.DataType.complexConjugateHalf
      elif dataTypeString == "cobaltDataTypeComplexConjugateSingle":
        self.tensor.dataType.value = Structs.DataType.complexConjugateSingle
      elif dataTypeString == "cobaltDataTypeComplexConjugateDouble":
        self.tensor.dataType.value = Structs.DataType.complexConjugateDouble
      elif dataTypeString == "cobaltDataTypeNone":
        self.tensor.dataType.value = Structs.DataType.none
      pass
    elif tag == "Dimension": # DONE
      dim = Structs.Dimension()
      dim.stride = int(attributes["stride"])
      dim.size = int(attributes["size"])
      self.tensor.dimensions.append( dim )
      pass
    elif tag == "Operation":
      self.problem.operation.useAlpha = int(attributes["useAlpha"])
      self.problem.operation.alphaType = \
          Structs.DataType(int(attributes["alphaType"]))
      self.problem.operation.useBeta = int(attributes["useBeta"])
      self.problem.operation.betaType = \
          Structs.DataType(int(attributes["betaType"]))
      self.problem.operation.useOffsets = int(attributes["useOffsets"])
      self.problem.operation.numIndicesFree = \
          int(attributes["numIndicesFree"])
      self.problem.operation.numIndicesBatch = \
          int(attributes["numIndicesBatch"])
      self.problem.operation.numIndicesSummation = \
          int(attributes["numIndicesSummation"])
      pass
    elif tag == "Type":
      operationTypeStr = attributes["string"]
      if operationTypeStr == "cobaltOperationTypeContraction":
        self.problem.operation.type = Structs.OperationType(0)
      elif operationTypeStr == "cobaltOperationTypeConvolution":
        self.problem.operation.type = Structs.OperationType(1)
      elif operationTypeStr == "cobaltOperationTypeCorrelation":
        self.problem.operation.type = Structs.OperationType(2)
      pass
    elif tag == "IndexAssignments":
      self.indexAssignments = []
      pass
    elif tag == "IndexAssignment":
      self.indexAssignments.append(int(attributes["indexAssignment"]))
      pass
    elif tag == "DeviceProfile":
      pass
    elif tag == "Device":
      device = Structs.Device(attributes["name"] )
      self.problem.deviceProfile.devices.append( device )
      pass
    elif tag == "ImplementationDetails":
      self.solution.kernels = []
      self.solution.kernelGrid = [ int(attributes["kernelGrid0"]), int(attributes["kernelGrid1"]), int(attributes["kernelGrid2"]) ]
      self.solution.branch = [ Structs.BranchType(int(attributes["branch0"])), Structs.BranchType(int(attributes["branch1"])) ]
      self.solution.ppdOffsets = attributes["ppdOffsets"] == "True"
      self.solution.ppdLeadingStride = attributes["ppdLeadingStride"] == "True"
      self.solution.ppdAll = attributes["ppdAll"] == "True"
      pass
    elif tag == "Kernel":
      # read data from xml
      if attributes["name"] != "None":
        kernel = Structs.Kernel()
        kernel.tile.workGroup = [int(attributes["workGroup0"]), int(attributes["workGroup1"])]
        kernel.tile.microTile = [int(attributes["microTile0"]), int(attributes["microTile1"])]
        kernel.tile.branch = [ Structs.BranchType(int(attributes["branch0"])), Structs.BranchType(int(attributes["branch1"])) ]
        kernel.unrolls = [ int(attributes["unroll0"]) ]
        secondUnroll = int(attributes["unroll1"])
        if secondUnroll > 0:
          kernel.unrolls.append( secondUnroll )
        # pull data from problem and solution
        kernel.dataTypeC = self.problem.tensorC.dataType
        kernel.dataTypeA = self.problem.tensorA.dataType
        kernel.dataTypeB = self.problem.tensorB.dataType
        kernel.operation = self.problem.operation
        kernel.problem = self.problem
        kernel.ppdOffsets = self.solution.ppdOffsets
        kernel.ppdLeadingStride = self.solution.ppdLeadingStride
        kernel.ppdAll = self.solution.ppdAll
        # make index assignments (rather than storing in xml)
        SolutionCandidateGenerator.makeIndexAssignments(kernel, self.problem)
        # append kernel to current solution
        self.solution.kernels.append(kernel)
      else:
        self.solution.kernels.append(None)
      pass
    elif tag == "Benchmark":
      # basically end of TraceEntry
      time = float(attributes["time"])
      exactMatch = Structs.ExactMatch()
      self.assignExactMatch(exactMatch)
      addTimeToMap( self.data, exactMatch, copy.deepcopy(self.problem), copy.deepcopy(self.solution), time )
    elif tag == "Validation":
      valid = 1 if attributes["status"] == "True" else -1
      exactMatch = Structs.ExactMatch()
      self.assignExactMatch(exactMatch)
      addValidationToMap( self.data, exactMatch, self.problem, self.solution, valid )


  def endElement(self, tag):
    if tag == "Problem": # DONE
      if self.readProblems:
        self.data.add(copy.deepcopy(self.problem))
        self.numProblemsAdded += 1
    elif tag == "Tensor": # DONE
      if self.currentTensor == 0: # C
        self.problem.tensorC = copy.deepcopy(self.tensor)
      elif self.currentTensor == 1: # A
        self.problem.tensorA = copy.deepcopy(self.tensor)
      elif self.currentTensor == 2: # B
        self.problem.tensorB = copy.deepcopy(self.tensor)
      self.currentTensor = (self.currentTensor+1)%3
      pass
    elif tag == "Dimension": # DONE
      pass
    elif tag == "Operation": # DONE
      pass
    elif tag == "Type": # DONE
      pass
    elif tag == "IndexAssignments": # DONE
      #print "Completed IndexAssignments:"
      #print str(self.indexAssignments)
      if self.currentIndexAssignments == 0: # A
        self.problem.operation.indexAssignmentsA = self.indexAssignments
      elif self.currentIndexAssignments == 1: # B
        self.problem.operation.indexAssignmentsB = self.indexAssignments
      self.currentIndexAssignments = (self.currentIndexAssignments+1)%2
      pass
    elif tag == "IndexAssignment": # DONE
      pass
    elif tag == "DeviceProfile":
      pass
    elif tag == "Device":
      pass
    elif tag == "ImplementationDetails":
      pass
    elif tag == "Kernel":
      pass
    elif tag == "Benchmark":
      pass
    elif tag == "TraceEntry":
      pass
    elif tag == "CobaltLog":
      pass

  def characters(self, content):
    pass

  def assignExactMatch(self, exactMatch):
    exactMatch.deviceProfile = self.problem.deviceProfile
    exactMatch.numIndicesFree = len(self.problem.tensorC.dimensions)
    exactMatch.indexAssignmentsA = self.problem.operation.indexAssignmentsA
    exactMatch.indexAssignmentsB = self.problem.operation.indexAssignmentsB
    exactMatch.operationType = self.problem.operation.type
    exactMatch.ppdOffsets = self.solution.ppdOffsets
    exactMatch.ppdLeadingStride = self.solution.ppdLeadingStride
    exactMatch.ppdAll = self.solution.ppdAll
    exactMatch.typeC = self.problem.tensorC.dataType
    exactMatch.typeA = self.problem.tensorA.dataType
    exactMatch.typeB = self.problem.tensorB.dataType
    exactMatch.typeAlpha = self.problem.operation.alphaType
    exactMatch.typeBeta = self.problem.operation.betaType



################################################################################
# getProblemsFromXML
################################################################################
def getProblemsFromXML( inputFile, problemSet ):
  parser = xml.sax.make_parser()
  parser.setFeature(xml.sax.handler.feature_namespaces, 0)
  readSolutions = False
  appProblemsHandler = CobaltHandler(problemSet, readSolutions)
  parser.setContentHandler( appProblemsHandler )
  try:
    parser.parse( inputFile )
    print "  + " + str(appProblemsHandler.numProblemsAdded) \
        + " problem(s) from " + os.path.basename(inputFile)
  except:
    print inputFile + " error"

################################################################################
# getProblemsFromXML
################################################################################
def getSolutionsFromXML( inputFile, psMap ):
  print "creating parser\n"
  parser = xml.sax.make_parser()
  parser.setFeature(xml.sax.handler.feature_namespaces, 0)
  readSolutions = True
  print "creating handler\n"
  solutionsHandler = CobaltHandler(psMap, readSolutions)
  print "setting handler\n"
  parser.setContentHandler( solutionsHandler )
  try:
    print "parsing file\n"
    parser.parse( inputFile )
    print "parsing file - done\n"
  except:
    print inputFile + " error"
  

################################################################################
# Main
################################################################################
if __name__ == "__main__":

  # arguments
  ap = argparse.ArgumentParser(description="FileReader")
  ap.add_argument("--input-file", dest="inputFiles", action="append" )
  args = ap.parse_args()

  # parse xml
  for inputFile in args.inputFiles:
    problemSet = set()
    getProblemsFromXML( inputFile, problemSet )
  print problemSet

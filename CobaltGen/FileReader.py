import argparse
import Structs

import xml.sax
import copy

################################################################################
# AppProblemsHandler
################################################################################
class AppProblemsHandler( xml.sax.ContentHandler ):
  def __init__(self, problemSet):
    self.problemSet = problemSet
    self.numProblemsAdded = 0
    self.inSummaryGetSolution = False
    self.problem = Structs.Problem()

    self.currentTensor = 0
    self.tensor = Structs.Tensor()

    self.currentIndexAssignments = 0
    self.indexAssignments = []

  def startElement(self, tag, attributes):
    if tag == "SummaryGetSolution": # DONE
      self.inSummaryGetSolution = True
    elif not self.inSummaryGetSolution: # DONE
      return
    elif tag == "Problem": # DONE
      self.problem = Structs.Problem()
    elif tag == "Tensor": # DONE
      self.tensor = Structs.Tensor()
      dataTypeString = attributes["dataType"]
      if dataTypeString == "cobaltDataTypeSingle":
        self.tensor.dataType.value = 0
      elif dataTypeString == "cobaltDataTypeDouble":
        self.tensor.dataType.value = 1
      elif dataTypeString == "cobaltDataTypeSingleComplex":
        self.tensor.dataType.value = 2
      elif dataTypeString == "cobaltDataTypeDoubleComplex":
        self.tensor.dataType.value = 3
      pass
    elif tag == "Dimension": # DONE
      dim = Structs.Dimension()
      dim.stride = int(attributes["stride"])
      dim.size = int(attributes["size"])
      self.tensor.dimensions.append( dim )
      pass
    elif tag == "Operation":
      self.problem.operation.alphaType = \
          Structs.DataType(int(attributes["alphaType"]))
      self.problem.operation.alpha = \
          int(attributes["alpha"])
      self.problem.operation.betaType = \
          Structs.DataType(int(attributes["betaType"]))
      self.problem.operation.beta = \
          int(attributes["beta"])
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
      device = Structs.Device(attributes["name"], \
          int(attributes["numComputeUnits"]), \
          int(attributes["clockFrequency"]) )
      self.problem.deviceProfile.devices.append( device )
      pass

  def endElement(self, tag):
    if tag == "SummaryGetSolution": # DONE
      self.inSummaryGetSolution = False
    elif not self.inSummaryGetSolution: # DONE
      return
    elif tag == "Problem": # DONE
      #print "Completed Problem:"
      #print str(self.problem)
      self.problemSet.add(copy.deepcopy(self.problem))
      self.numProblemsAdded += 1
    elif tag == "Tensor": # DONE
      #print "Completed Tensor: " + str(self.currentTensor)
      #print str(self.tensor)
      #print "tensorC: " + str(self.problem.tensorC)
      #print "tensorA: " + str(self.problem.tensorA)
      #print "tensorB: " + str(self.problem.tensorB)
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

  def characters(self, content):
    pass









"""
from xml.dom.minidom import parse
import xml.dom.minidom

################################################################################
# getTensorFromElement
################################################################################
def getTensorFromElement( element ):
  dataType = Structs.DataType(-1)
  dataTypeString = element.getAttribute("dataType")
  if dataTypeString == "cobaltDataTypeSingle":
    dataType = Structs.DataType(0)
  elif dataTypeString == "cobaltDataTypeDouble":
    dataType = Structs.DataType(1)
  elif dataTypeString == "cobaltDataTypeSingleComplex":
    dataType = Structs.DataType(2)
  elif dataTypeString == "cobaltDataTypeDoubleComplex":
    dataType = Structs.DataType(3)
  else:
    print "Tensor dataType string \"" + dataTypeString + "\" unrecognized"
  dims = []
  dimElements = element.getElementsByTagName("Dimension")
  for dimElement in dimElements:
    dims.append( Structs.Dimension( \
        int(dimElement.getAttribute("stride")), \
        int(dimElement.getAttribute("size")) ) )
  return Structs.Tensor( dataType, dims )


################################################################################
# getOperationFromElement
################################################################################
def getOperationFromElement( element ):

  # type
  operationTypeStr = \
    element.getElementsByTagName("Type")[0].getAttribute("string")
  operationType = Structs.OperationType(-1)
  if operationTypeStr == "cobaltOperationTypeContraction":
    operationType = Structs.OperationType(0)
  elif operationTypeStr == "cobaltOperationTypeConvolution":
    operationType = Structs.OperationType(1)
  elif operationTypeStr == "cobaltOperationTypeCorrelation":
    operationType = Structs.OperationType(2)
  else:
    print "OperationType " + operationTypeStr + " unrecognized."

  # alpha, beta, numIndices
  alphaType = Structs.DataType(int(element.getAttribute("alphaType")))
  alpha = int(element.getAttribute("alpha"))
  betaType = Structs.DataType( int(element.getAttribute("betaType")))
  beta = int(element.getAttribute("beta"))
  numIndicesFree = int(element.getAttribute("numIndicesFree"))
  numIndicesBatch = int(element.getAttribute("numIndicesBatch"))
  numIndicesSummation = int(element.getAttribute("numIndicesSummation"))

  # indexAssignmentsA
  indexAssignmentsElements = element.getElementsByTagName("IndexAssignments")
  indexAssignmentElementsA = \
      indexAssignmentsElements[0].getElementsByTagName("IndexAssignment")
  indexAssignmentsA = []
  for indexAssignmentElement in indexAssignmentElementsA:
    indexAssignmentsA.append( \
        int(indexAssignmentElement.getAttribute("indexAssignment")) )

  # indexAssignmentsB
  indexAssignmentElementsB = \
      indexAssignmentsElements[1].getElementsByTagName("IndexAssignment")
  indexAssignmentsB = []
  for indexAssignmentElement in indexAssignmentElementsB:
    indexAssignmentsB.append( \
        int(indexAssignmentElement.getAttribute("indexAssignment")) )

  return Structs.Operation( \
      operationType, \
      alphaType, \
      alpha, \
      betaType, \
      beta, \
      numIndicesFree, \
      numIndicesBatch, \
      numIndicesSummation, \
      indexAssignmentsA, \
      indexAssignmentsB )


################################################################################
# getDeviceProfileFromElement
################################################################################
def getDeviceProfileFromElement( element ):
  devices = []
  deviceElements = \
      element.getElementsByTagName("Device")
  for deviceElement in deviceElements:
    devices.append( Structs.Device( \
        deviceElement.getAttribute("name"), \
        int(deviceElement.getAttribute("numComputeUnits")), \
        int(deviceElement.getAttribute("clockFrequency")) ) )
  return Structs.DeviceProfile( devices )
"""


################################################################################
# getProblemsFromXML
################################################################################
def getProblemsFromXML( inputFile, problemSet ):
  """
  DOMTree = xml.dom.minidom.parse( inputFile )
  log = DOMTree.documentElement
  getSolutionCalls = log.getElementsByTagName("GetSolution")
  for getSolutionCall in getSolutionCalls:
    solution = getSolutionCall.getElementsByTagName("Solution")[0]
    problem = solution.getElementsByTagName("Problem")[0]

    # tensors
    tensorElements = problem.getElementsByTagName("Tensor")
    tensorC = getTensorFromElement(tensorElements[0])
    tensorA = getTensorFromElement(tensorElements[1])
    tensorB = getTensorFromElement(tensorElements[2])

    # device profile
    deviceProfileElement = problem.getElementsByTagName("DeviceProfile")[0]
    deviceProfile = getDeviceProfileFromElement(deviceProfileElement)

    # operation
    operationElement = problem.getElementsByTagName("Operation")[0]
    operation = getOperationFromElement( operationElement )

    # problem
    problem = Structs.Problem( \
        tensorA, \
        tensorB, \
        tensorC, \
        operation, \
        deviceProfile )
    problemSet.add(problem)
  """
  print "getProblemsFromXML"
  parser = xml.sax.make_parser()
  parser.setFeature(xml.sax.handler.feature_namespaces, 0)
  appProblemsHandler = AppProblemsHandler(problemSet)
  parser.setContentHandler( appProblemsHandler )
  parser.parse( inputFile )
  print "AppProblemsHandler::NumProblemsAdded = " + str(appProblemsHandler.numProblemsAdded)


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

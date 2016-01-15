import argparse
from xml.dom.minidom import parse
import xml.dom.minidom
import Structs

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
        dimElement.getAttribute("stride"), \
        dimElement.getAttribute("size") ) )
  return Structs.Tensor( dataType, dims )


################################################################################
# getOperationFromElement
################################################################################
def getOperationFromElement( element ):

  # type
  operationTypeStr = \
    element.getElementsByTagName("Type")[0].getAttribute("string")
  operationType = Structs.OperationType(-1)
  if operationTypeStr == "cobaltOperationTypeTensorContraction":
    operationType = Structs.OperationType(0)
  elif operationTypeStr == "cobaltOperationTypeConvolution":
    operationType = Structs.OperationType(1)
  elif operationTypeStr == "cobaltOperationTypeCorrelation":
    operationType = Structs.OperationType(2)
  else:
    print "OperationType " + operationTypeStr + " unrecognized."

  # numIndices
  numIndicesFree = element.getAttribute("numIndicesFree")
  numIndicesBatch = element.getAttribute("numIndicesBatch")
  numIndicesSummation = element.getAttribute("numIndicesSummation")

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
        deviceElement.getAttribute("numComputeUnits"), \
        deviceElement.getAttribute("clockFrequency") ) )
  return Structs.DeviceProfile( devices )


################################################################################
# getProblemsFromXML
################################################################################
def getProblemsFromXML( inputFile, problemSet ):
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

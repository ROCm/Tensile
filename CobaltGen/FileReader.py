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
      if dataTypeString == "cobaltDataTypeHalf":
        self.tensor.dataType.value = 0
      elif dataTypeString == "cobaltDataTypeSingle":
        self.tensor.dataType.value = 1
      elif dataTypeString == "cobaltDataTypeDouble":
        self.tensor.dataType.value = 2
      elif dataTypeString == "cobaltDataTypeComplexHalf":
        self.tensor.dataType.value = 3
      elif dataTypeString == "cobaltDataTypeComplexSingle":
        self.tensor.dataType.value = 4
      elif dataTypeString == "cobaltDataTypeComplexDouble":
        self.tensor.dataType.value = 5
      elif dataTypeString == "cobaltDataTypeNone":
        self.tensor.dataType.value = 7
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
      if self.problem in self.problemSet:
        print "Oops; problem already in set: " + str(self.problem)
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



################################################################################
# getProblemsFromXML
################################################################################
def getProblemsFromXML( inputFile, problemSet ):
  parser = xml.sax.make_parser()
  parser.setFeature(xml.sax.handler.feature_namespaces, 0)
  appProblemsHandler = AppProblemsHandler(problemSet)
  parser.setContentHandler( appProblemsHandler )
  try:
    parser.parse( inputFile )
    print inputFile + " added " + str(appProblemsHandler.numProblemsAdded) \
        + " problems"
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

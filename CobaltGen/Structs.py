#from enum import Enum

################################################################################
# Status - Enum
################################################################################
class Status:
  success = 0

################################################################################
# Data Type - Enum
################################################################################
class DataType:
  single = 0
  double = 1
  singleComplex = 2
  doubleComplex = 3

  def __init__( self, value ):
    self.value = value

  def toChar(self):
    if self.value == self.single:
      return "S"
    elif self.value == self.double:
      return "D"
    elif self.value == self.singleComplex:
      return "C"
    elif self.value == self.doubleComplex:
      return "Z"
    else:
      return "ERROR"

  def toOpenCL(self):
    if self.value == self.single:
      return "float"
    elif self.value == self.double:
      return "double"
    elif self.value == self.singleComplex:
      return "float2"
    elif self.value == self.doubleComplex:
      return "double2"
    else:
      return "ERROR"

  def isReal(self):
    if self.value == self.single or self.value == self.double:
      return True
    else:
      return False

  def isComplex(self):
    return not self.isReal()

  def numRegistersPerElement( self ):
    if self.value == self.single:
      return 1
    elif self.value == self.double:
      return 2
    elif self.value == self.singleComplex:
      return 2
    elif self.value == self.doubleComplex:
      return 4
    else:
      return "ERROR"



################################################################################
# Dimension
################################################################################
class Dimension:
  def __init__( self ):
    self.stride = 0
    self.size = 0
  def __init__( self, stride, size ):
    self.stride = stride
    self.size = size
  def __str__(self):
    return "["+str(self.stride)+","+str(self.size)+"]"
  def __repr__(self):
    return self.__str__()


################################################################################
# Tensor
################################################################################
class Tensor:
  def __init__(
      self, \
      dataType, \
      dimensions ):
    self.dataType = dataType
    self.dimensions = dimensions

  def __str__(self):
    state = "[ Tensor " + self.dataType.toChar() + ", " + self.dimensions.__str__() + " ]"
    return state

################################################################################
# Backend - Enum
################################################################################
class Backend:
  opencl = 0
  hsa = 1
  hcc = 2

  def __init__( self, value ):
    self.value = value

  def toString(self):
    if self.value == self.opencl:
      return "OCL"
    elif self.value == self.hsa:
      return "HSA"
    elif self.value == self.hcc:
      return "HCC"
    else:
      return "ERROR"


################################################################################
# Device
################################################################################
class Device:
  def __init__(
      self, \
      name, \
      numComputeUnits, \
      clockFrequency ):
    self.name = name
    self.numComputeUnits = numComputeUnits
    self.clockFrequency = clockFrequency; # MHz

  def __str__(self):
    return self.name

  def __repr__(self):
    return self.__str__()


################################################################################
# DeviceProfile
################################################################################
class DeviceProfile:
  def __init__(self):
    self.devices = []
  def __init__(self, devices):
    self.devices = devices

  def __str__(self):
    return str(self.devices)

  def __repr__(self):
    return self.__str__()


################################################################################
# OperationType - Enum
################################################################################
class OperationType:
  contraction = 0
  convolution = 1
  correlation = 2

  def __init__(self, value):
    self.value = value

  def __str__(self):
    if self.value == self.contraction:
      return "CT"
    elif self.value == self.convolution:
      return "CV"
    elif self.value == self.correlation:
      return "CR"
    else:
      return "ERROR"

  def __repr__(self):
    return self.__str__()


################################################################################
# Operation
################################################################################
class Operation:
  def __init__( \
      self, \
      type = OperationType(-1), \
      alpha = -1, \
      beta = -1, \
      numIndicesFree = -1, \
      numIndicesBatch = -1, \
      numIndicesSummation = -1, \
      indexAssignmentsA = [], \
      indexAssignmentsB = []):
    self.type = type
    self.alpha = alpha
    self.beta = beta
    self.numIndicesFree = numIndicesFree
    self.numIndicesBatch = numIndicesBatch
    self.numIndicesSummation = numIndicesSummation
    self.indexAssignmentsA = indexAssignmentsA
    self.indexAssignmentsB = indexAssignmentsB
    self.pad = []
    self.stride = []

  def __str__(self):
    state = ""
    state += "[ Operation "
    state += "ty=" + str(self.type) + "; "
    state += "al=" + str(self.alpha) + "; "
    state += "be=" + str(self.beta) + "; "
    state += "nF=" + str(self.numIndicesFree) + "; "
    state += "nB=" + str(self.numIndicesBatch) + "; "
    state += "nS=" + str(self.numIndicesSummation) + "; "
    state += "iA=" + str(self.indexAssignmentsA) + "; "
    state += "iB=" + str(self.indexAssignmentsB) + " ]"
    return state

  def __repr__(self):
    return self.__str__()


################################################################################
# Problem
# - some problem descriptors get passed in as kernel argument and
#   Don't need to be exactly matched to solution
#   - Tensor dimensions[i].size
#   - Tensor dimensions[i].stride
#   - alpha
#   - beta
# - some problem descriptors get compiled/written into kernel and
#   Do need to be exactly matched to solution
#   - Tensor data types
#   - dimensionality of tensors and operation
#   - operation
# - other
#   - Device - determined through benchmarking / file reading
################################################################################
class Problem:
  def __init__( \
      self, \
      tensorC, \
      tensorA, \
      tensorB, \
      operation, \
      deviceProfile ):
    self.tensorC = tensorC
    self.tensorA = tensorA
    self.tensorB = tensorB
    self.operation = operation
    self.deviceProfile = deviceProfile

  def __str__(self):
    state = "[ Problem "
    state += "C" + str(self.tensorC) + "; "
    state += "A" + str(self.tensorA) + "; "
    state += "B" + str(self.tensorB) + "; "
    state += "O" + str(self.operation) + "; "
    state += "D" + str(self.deviceProfile) + " ]"
    return state

  def __repr__(self):
    return self.__str__()


################################################################################
# ProblemRange
# - contiguous range of problems uniterrupted by any other ProblemRange
# - after benchmarking, MAP[Solution]->ProblemList, ProblemList needs to be
#   broken down into ProblemRanges such that:
#   - if Problem is in ProblemRange, use associated Solution
################################################################################
class ProblemRange:
  def __init__(self):
    self.mins = []
    self.maxs = []
    self.mods = []
    self.freeMin = -1
    self.freeMax = -1
    self.freeMod = -1
    self.summationMin = -1
    self.summationMax = -1
    self.summationMod = -1
# contains one set of SolutionCorrectnessParameters
# work-item min, max (dimA*dimB outside of summation)
# summation size, "k"


################################################################################
# Tile
################################################################################
class Tile:
  def __init__( self ):
    self.workGroupDim0 = -1
    self.workGroupDim1 = -1
    self.microTileDim0 = -1
    self.microTileDim1 = -1
    self.macroTileDim0 = -1
    self.macroTileDim1 = -1

  def __str__(self):
    state = "[ " + str(self.workGroupDim0) + "x" + str(self.workGroupDim1)
    state += " " + str(self.microTileDim0) + "x" + str(self.microTileDim1)
    state += " " + str(self.macroTileDim0) + "x" + str(self.macroTileDim1)
    state += " ]"

  def __repr__(self):
    return self.__str__()


################################################################################
# Kernel
################################################################################
class Kernel:
  def __init__( self ):
    self.tile = Tile()
    self.unrolls = []
    #self.gridLocation = [ 1, 1 ]
    self.edge = [ False, False ]


################################################################################
# Solution
################################################################################
class Solution:
  def __init__(self):
    # Solution Correctness Parameters
    self.operation = Operation()
    self.dataTypeC = DataType(-1)
    self.dataTypeA = DataType(-1)
    self.dataTypeB = DataType(-1)
    self.numIndicesFree = -1
    self.numIndicesBatch = -1
    self.numIndicesSummation = -1
    # Index Assignments
    self.indexOrderC = []
    self.indexOrderSummation = []
    self.indexAssignmentTileDim0 = -1
    self.indexAssignmentTileDim1 = -1
    # Problem Characteristics affecting performance
    self.problemSizeDim0 = -1
    self.problemSizeDim1 = -1
    self.indexUnroll = -1
    self.problemSizeUnroll = -1
    self.tensorStrideDim0 = -1
    self.tensorStrideDim1 = -1
    # Kernels
    self.kernelGrid = [ -1, -1 ]
    self.kernels = []

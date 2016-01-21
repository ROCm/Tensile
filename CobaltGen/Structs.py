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

  def __str__(self):
    return self.toChar()

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return (self.value)
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, DataType) and self.getAttributes() == other.getAttributes()



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

  def getAttributes(self):
    return ( self.stride, self.size )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Dimension) and self.getAttributes() == other.getAttributes()


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
    state = "[Tensor"
    state += "; " + self.dataType.toChar()
    state += "; " + str(self.dimensions)
    state += "]"
    return state

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( self.dataType, frozenset(self.dimensions) )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Tensor) and self.getAttributes() == other.getAttributes()


################################################################################
# Backend - Enum
################################################################################
class Backend:
  opencl12 = 0
  hip = 1

  def __init__( self, value=0 ):
    self.value = value

  def __str__(self):
    if self.value == self.opencl12:
      return "OpenCL 1.2"
    elif self.value == self.hip:
      return "HIP"
    else:
      return "ERROR"

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( self.value )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Backend) and self.getAttributes() == other.getAttributes()


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
    state = "[Device"
    state += "; " + self.name
    state += "; " + str(self.numComputeUnits)
    state += "; " + str(self.clockFrequency)
    state += "]"
    return self.name

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        self.name, \
        self.numComputeUnits, \
        self.clockFrequency, \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Device) and self.getAttributes() == other.getAttributes()


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

  def getAttributes(self):
    return (frozenset(self.devices))
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, DeviceProfile) and self.getAttributes() == other.getAttributes()


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

  def getAttributes(self):
    return (self.value)
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, OperationType) and self.getAttributes() == other.getAttributes()


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
    state += "[Operation"
    state += "; " + str(self.type)
    state += "; " + str(self.alpha)
    state += "; " + str(self.beta)
    state += "; " + str(self.numIndicesFree)
    state += "; " + str(self.numIndicesBatch)
    state += "; " + str(self.numIndicesSummation)
    state += "; " + str(self.indexAssignmentsA)
    state += "; " + str(self.indexAssignmentsB)
    state += "]"
    return state

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        self.type, \
        self.alpha, \
        self.beta, \
        self.numIndicesFree, \
        self.numIndicesBatch, \
        self.numIndicesSummation, \
        frozenset(self.indexAssignmentsA), \
        frozenset(self.indexAssignmentsB), \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Operation) and self.getAttributes() == other.getAttributes()


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
    state = "[Problem"
    state += "; " + str(self.tensorC)
    state += "; " + str(self.tensorA)
    state += "; " + str(self.tensorB)
    state += "; " + str(self.operation)
    state += "; " + str(self.deviceProfile)
    state += "]"
    return state

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        self.tensorC, \
        self.tensorA, \
        self.tensorB, \
        self.operation, \
        self.deviceProfile, \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Problem) and self.getAttributes() == other.getAttributes()


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
    state = "[Tile; " + str(self.workGroupDim0) + "x" + str(self.workGroupDim1)
    state += "; " + str(self.microTileDim0) + "x" + str(self.microTileDim1)
    state += "; " + str(self.macroTileDim0) + "x" + str(self.macroTileDim1)
    state += "]"
    return state

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        self.workGroupDim0, \
        self.workGroupDim1, \
        self.microTileDim0, \
        self.microTileDim1, \
        self.macroTileDim0, \
        self.macroTileDim1, \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Tile) and self.getAttributes() == other.getAttributes()


################################################################################
# Kernel
################################################################################
class Kernel:
  def __init__( self ):
    self.dataTypeC = DataType(-1)
    self.dataTypeA = DataType(-1)
    self.dataTypeB = DataType(-1)
    self.operation = Operation()
    # Index Assignments
    self.indexOrderC = []
    self.indexOrderSummation = []
    self.indexAssignmentDim0 = -1
    self.indexAssignmentDim1 = -1
    self.indexUnroll = -1
    # Tile
    self.tile = Tile()
    self.unrolls = []

  def __str__(self):
    state = "[Kernel; " + str(self.tile)
    state += "; " + str(self.dataTypeC)
    state += "; " + str(self.dataTypeA)
    state += "; " + str(self.dataTypeB)
    state += "; " + str(self.operation)
    state += "; " + str(self.indexOrderC)
    state += "; " + str(self.indexOrderSummation)
    state += "; " + str(self.indexAssignmentDim0)
    state += "; " + str(self.indexAssignmentDim1)
    state += "; " + str(self.indexUnroll)
    state += "; " + str(self.tile)
    state += "; " + str(self.unrolls)
    state += "]"
    return state

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        self.dataTypeC, \
        self.dataTypeA, \
        self.dataTypeB, \
        self.operation, \
        frozenset(self.indexOrderC), \
        frozenset(self.indexOrderSummation), \
        self.indexAssignmentDim0, \
        self.indexAssignmentDim1, \
        self.indexUnroll, \
        self.tile, \
        frozenset(self.unrolls), \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Kernel) and self.getAttributes() == other.getAttributes()


################################################################################
# Solution
################################################################################
class Solution:
  def __init__(self):
    # Solution Correctness Parameters
    # Kernels
    self.kernelGrid = [ -1, -1 ]
    self.kernels = []

  def __str__(self):
    state = "[Solution"
    state += "; " + str(self.kernelGrid)
    state += "; " + str(self.kernels)
    state += "]"
    return state

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        frozenset(self.kernels), \
        self.kernelGrid[0], \
        self.kernelGrid[1] )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Solution) and self.getAttributes() == other.getAttributes()


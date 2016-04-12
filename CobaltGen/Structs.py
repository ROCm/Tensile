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
  half          = 0
  single        = 1
  double        = 2
  complexHalf   = 3
  complexSingle = 4
  complexDouble = 5
  complexConjugateHalf    = 6
  complexConjugateSingle  = 7
  complexConjugateDouble    = 8
  # num         = 9
  none          = 10

  def __init__( self, value ):
    self.value = value

  def toChar(self):
    if self.value == self.half:
      return "H"
    if self.value == self.single:
      return "S"
    elif self.value == self.double:
      return "D"
    elif self.value == self.complexHalf:
      return "Q"
    elif self.value == self.complexSingle:
      return "C"
    elif self.value == self.complexDouble:
      return "Z"
    elif self.value == self.complexConjugateHalf:
      return "W"
    elif self.value == self.complexConjugateSingle:
      return "X"
    elif self.value == self.complexConjugateDouble:
      return "Y"
    elif self.value == self.none:
      return "0"
    else:
      return "ERROR(" + str(self.value) + ")"

  def toOpenCL(self):
    if self.value == self.single:
      return "float"
    elif self.value == self.double:
      return "double"
    elif self.value == self.complexSingle or self.value == self.complexConjugateSingle:
      return "float2"
    elif self.value == self.complexDouble or self.value == self.complexConjugateDouble:
      return "double2"
    else:
      return "ERROR(" + str(self.value) + ")"

  def toCpp(self):
    if self.value == self.single:
      return "float"
    elif self.value == self.double:
      return "double"
    elif self.value == self.complexSingle or self.value == self.complexConjugateSingle:
      return "CobaltComplexFloat"
    elif self.value == self.complexDouble or self.value == self.complexConjugateDouble:
      return "CobaltComplexDouble"
    elif self.value == self.none:
      return "void"
    else:
      return "ERROR(" + str(self.value) + ")"

  def getLibString(self):
    if self.value == self.half:
      return "cobaltDataTypeHalf"
    if self.value == self.single:
      return "cobaltDataTypeSingle"
    elif self.value == self.double:
      return "cobaltDataTypeDouble"
    elif self.value == self.complexHalf:
      return "cobaltDataTypeComplexHalf"
    elif self.value == self.complexSingle:
      return "cobaltDataTypeComplexSingle"
    elif self.value == self.complexDouble:
      return "cobaltDataTypeComplexDouble"
    elif self.value == self.complexConjugateHalf:
      return "cobaltDataTypeComplexConjugateHalf"
    elif self.value == self.complexConjugateSingle:
      return "cobaltDataTypeComplexConjugateSingle"
    elif self.value == self.complexConjugateDouble:
      return "cobaltDataTypeComplexConjugateDouble"
    elif self.value == self.none:
      return "cobaltDataTypeNone"
    else:
      return "ERROR(" + str(self.value) + ")"

  def zeroStringOpenCL(self):
    zeroString = "("
    zeroString += self.toOpenCL()
    zeroString += ")("
    if self.isReal():
      zeroString += "0.0"
    else:
      zeroString += "0.0, 0.0"
    zeroString += ")"
    return zeroString

  def isReal(self):
    if self.value == self.half or self.value == self.single or self.value == self.double:
      return True
    else:
      return False

  def isComplex(self):
    return not self.isReal()

  def isConjugate(self):
    if self.value == self.complexConjugateHalf or self.value == self.complexConjugateSingle or self.value == self.complexConjugateDouble:
      return True
    else:
      return False

  def numRegisters( self ):
    if self.value == self.single:
      return 1
    elif self.value == self.double:
      return 2
    elif self.value == self.complexSingle or self.value == self.complexConjugateSingle:
      return 2
    elif self.value == self.complexDouble or self.value == self.complexConjugateDouble:
      return 4
    else:
      return "ERROR(" + str(self.value) + ")"

  def numBytes( self ):
    return self.numRegisters() * 4

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
  #def __init__( self, stride, size ):
  #  self.stride = stride
  #  self.size = size

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
  def __init__( self ):
    self.dataType = DataType(-1)
    self.dimensions = []
    #print "Tensor::__init__" + str(self)

  def __str__(self):
    state = "[Tensor"
    state += "; " + self.dataType.toChar()
    state += "; " + str(self.dimensions)
    state += "]"
    return state

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( self.dataType, tuple(self.dimensions))
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Tensor) \
        and self.getAttributes() == other.getAttributes()


################################################################################
# Backend - Enum
################################################################################
class Backend:
  opencl12 = 0
  hip = 1

  def __init__( self ):
    self.value = 0

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
      name):
    self.name = name

  def __str__(self):
    state = "[Device"
    state += "; " + self.name
    state += "]"
    return self.name

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        self.name, \
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

  def __str__(self):
    return str(self.devices)

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return (tuple(self.devices))
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

  def getLibString(self):
    if self.value == self.contraction:
      return "cobaltOperationTypeContraction"
    elif self.value == self.convolution:
      return "cobaltOperationTypeConvolution"
    elif self.value == self.correlation:
      return "cobaltOperationTypeCorrelation"
    else:
      return "ERROR"


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
  def __init__( self ):
    self.type = OperationType(-1)
    self.useAlpha = -1
    self.alphaType = DataType(-1)
    self.useBeta = -1
    self.betaType = DataType(-1)
    self.numIndicesFree = -1
    self.numIndicesBatch = -1
    self.numIndicesSummation = -1
    self.indexAssignmentsA = []
    self.indexAssignmentsB = []
    self.pad = []
    self.stride = []

  def __str__(self):
    state = ""
    state += "[Operation"
    state += "; " + str(self.type)
    state += "; " + str(self.useAlpha)
    state += "; " + str(self.alphaType)
    state += "; " + str(self.useBeta)
    state += "; " + str(self.betaType)
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
        self.useAlpha, \
        self.alphaType, \
        self.useBeta, \
        self.betaType, \
        self.numIndicesFree, \
        self.numIndicesBatch, \
        self.numIndicesSummation, \
        tuple(self.indexAssignmentsA), \
        tuple(self.indexAssignmentsB), \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Operation) and self.getAttributes() == other.getAttributes()


################################################################################
# ExactMatch - parameters which must exactly match between problem and solution
################################################################################
class ExactMatch:
  def __init__(self):
    self.deviceProfile = DeviceProfile()
    self.typeC = DataType(-1)
    self.typeA = DataType(-1)
    self.typeB = DataType(-1)
    self.typeAlpha = DataType(-1)
    self.typeBeta = DataType(-1)
    self.operationType = OperationType(-1)
    self.indexAssignmentsA = []
    self.indexAssignmentsB = []
    self.ppdOffsets = False # if true, solution must allow offset parameters; if false, enqueue must not use offsets
    self.ppdInitialStrides = False # if true, solution must allow non-1 initial strides; if false, problem must have size=1 initial strides
    # self.ppdAll = False # to actually support all parameters being compiled into kernel, all tensor dimensions must become part of exact match

  def getAttributes(self):
    return ( \
      self.deviceProfile, \
      self.typeC, \
      self.typeA, \
      self.typeB, \
      self.typeAlpha, \
      self.typeBeta, \
      self.operationType, \
      tuple(self.indexAssignmentsA), \
      tuple(self.indexAssignmentsB), \
      self.ppdOffsets, \
      self.ppdInitialStrides \
      )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, ExactMatch) and self.getAttributes() == other.getAttributes()

class SolutionBenchmark:
  def __init__(self):
    self.times = []
    self.validationStatus = 0 # -1 invalid, 0 unspecified, +1 valid

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
  def __init__( self ):
    self.tensorC = Tensor()
    self.tensorA = Tensor()
    self.tensorB = Tensor
    self.operation = Operation()
    self.deviceProfile = DeviceProfile()

  def __str__(self):
    #state = "[Problem"
    #state += "; " + str(self.tensorC)
    #state += "; " + str(self.tensorA)
    #state += "; " + str(self.tensorB)
    #state += "; " + str(self.operation)
    #state += "; " + str(self.deviceProfile)
    #state += "]"

    state = ""
    indexChars = "ijklmnopqrstuvwxyz"

    # device
    for device in self.deviceProfile.devices:
      state += device.name + "_"
    # operation type
    state += self.operation.type.__str__() + "_"
    # precisions
    state += self.tensorC.dataType.toChar()
    state += self.tensorA.dataType.toChar()
    state += self.tensorB.dataType.toChar()
    state += self.operation.alphaType.toChar()
    state += self.operation.betaType.toChar()
    # C
    state += "_C_"
    state += indexChars[0]
    state += str(self.tensorC.dimensions[0].stride) + "_" + str(self.tensorC.dimensions[0].size)
    for i in range(1, len(self.tensorC.dimensions)):
      state += "_"
      state += indexChars[i]
      state += str(self.tensorC.dimensions[i].stride) + "_" + str(self.tensorC.dimensions[i].size)
    # Sum
    state += "_Sum_"
    state += indexChars[len(self.tensorC.dimensions)]
    for j in range(0, len(self.tensorA.dimensions)):
      if self.operation.indexAssignmentsA[j] == 0 + len(self.tensorC.dimensions):
        state += str(self.tensorA.dimensions[j].size)
    for i in range(1, self.operation.numIndicesSummation):
      state += "_"
      state += self.indexChars[tensorA.numDims()+i]
      for j in range( 0, len(self.tensorA.dimensions)):
        if self.operation.indexAssignmentsA[j] == i+len(self.tensorC.dimensions):
          state += str(self.tensorA.dimensions[j].size)
    # A
    state += "_A_"
    for i in range(0, len(self.tensorA.dimensions)):
      state += indexChars[self.operation.indexAssignmentsA[i]]
      state += str(self.tensorA.dimensions[i].stride)
      if i < len(self.tensorA.dimensions)-1:
        state += "_"
    # B
    state += "_B_";
    for i in range(0, len(self.tensorB.dimensions)):
      state += indexChars[self.operation.indexAssignmentsB[i]];
      state += str(self.tensorB.dimensions[i].stride)
      if i < len(self.tensorB.dimensions)-1:
        state += "_"
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
# BranchType - Enum
################################################################################
class BranchType:
  none = 0
  multiple = 1
  branched = 2

  def __init__(self, value):
    self.value = value

  def __str__(self):
    if self.value == self.none:
      return "none"
    elif self.value == self.multiple:
      return "multiple"
    elif self.value == self.branched:
      return "branched"
    else:
      return "ERROR"

  def getChar(self):
    if self.value == self.multiple:
      return "m"
    elif self.value == self.branched:
      return "b"
    else:
      return "x"

  def isNone(self):
    return self.value == self.none
  def isMultiple(self):
    return self.value == self.multiple
  def isBranched(self):
    return self.value == self.branched

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return (self.value)
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, BranchType) \
        and self.getAttributes() == other.getAttributes()


################################################################################
# Tile
################################################################################
class Tile:
  def __init__( self ):
    self.workGroup = [ -1, -1]
    self.microTile = [ -1, -1]
    self.branch = [ BranchType(-1), BranchType(-1)]

  def __str__(self):
    state = "[Tile; " + str(self.workGroup[0]) + "x" + str(self.workGroup[1])
    state += "; " + str(self.microTile[0]) + "x" + str(self.microTile[1])
    state += "; " + str(self.branch[0]) + "x" + str(self.branch[1])
    state += "]"
    return state

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        self.workGroup[0], \
        self.workGroup[1], \
        self.microTile[0], \
        self.microTile[1], \
        self.branch[0], \
        self.branch[1], \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Tile) \
        and self.getAttributes() == other.getAttributes()


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
    self.unrollDimStride0 = -1
    self.unrollDimStride1 = -1
    self.unrollDimSize = -1
    self.unrollDimStrideGreaterThanTileDimStrideA = False
    self.unrollDimStrideLessThanTileDimStrideB = False

    # a kernel holds a copy of the problem so it can #define strides if necessary
    self.problem = Problem()

    # Tile
    self.tile = Tile()
    self.unrolls = []

    # Pre-Processor definition optimizations
    self.ppdOffsets = False # offsets are #defined and not arguments
    self.ppdLeadingStride = False #leading strides are #defined and not arguments
    self.ppdAll = False #everything is #defined and not arguments

    # frozens
    # TODO remove these
    self.indexOrderC_Frozen = []
    self.indexOrderSummation_Frozen = []
    self.unrolls_Frozen = []

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
        tuple(self.indexOrderC), \
        tuple(self.indexOrderSummation), \
        self.indexAssignmentDim0, \
        self.indexAssignmentDim1, \
        self.tile, \
        tuple(self.unrolls), \
        self.ppdOffsets, \
        self.ppdLeadingStride, \
        self.ppdAll
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
    self.kernelGrid = [ -1, -1, -1 ]
    self.branch = [ BranchType(-1), BranchType(-1)]
    self.kernels = []

    # PreProcessor optimizations (#defining arguments)
    self.ppdOffsets = False # offsets are #defined and not arguments
    self.ppdLeadingStride = False #leading strides are #defined and not arguments
    self.ppdAll = False #everything is #defined and not arguments

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
        tuple(self.kernels), \
        self.kernelGrid[0], \
        self.kernelGrid[1], \
        self.branch[0], \
        self.branch[1], \
        self.ppdOffsets, \
        self.ppdLeadingStride, \
        self.ppdAll )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Solution) and self.getAttributes() == other.getAttributes()


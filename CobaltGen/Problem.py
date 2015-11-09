from enum import Enum

################################################################################
# Precision
################################################################################
class Precision(Enum):
  single = 0
  double = 1
  singleComplex = 2
  doubleComplex = 3

  def getChar(self):
    if self == single:
      return "s"
    elif self == double:
      return "d"
    elif self == singleComplex:
      return "c"
    elif self == doubleComplex:
      return "z"
    else
      return ""

  def getOpenCL(self):
    if self == single:
      return "float"
    elif self == double:
      return "double"
    elif self == singleComplex:
      return "float2"
    elif self == doubleComplex:
      return "double2"
    else:
      return ""

  def isReal(self):
    if self == single or self == double:
      return True
    else:
      return False
  def isComplex(self):
    return !self.isReal()


################################################################################
# DimensionDescriptor
################################################################################
class DimensionDescriptor:
  def __init__( self ):
    self.size = 0
    self.stride = 0
  def __init__( self, size, stride ):
    self.size = size
    self.stride = stride


################################################################################
# TensorDescriptor
################################################################################
class TensorDescriptor:
  def __init__(
      self, \
      precision, \
      dimensions ):
    self.precision = precision
    self.dimensions = dimensions


################################################################################
# OperationType
################################################################################
class OperationType(Enum):
  tensorContraction = 0
  convolution = 1


################################################################################
# OperationDescriptor
################################################################################
class OperationDescriptor:
  def __init__(
      self, \
      type, \
      dimensions ):
    self.type = type
    self.dimensions = dimensions


################################################################################
# DeviceDescriptor
################################################################################
class DeviceDescriptor:
  def __init__(
      name, \
      numComputeUnits, \
      clockFrequency ):
    self.name = name
    self.numComputeUnits = numComputeUnits
    self.clockFrequency = clockFrequency; # MHz


################################################################################
# DeviceProfile
################################################################################
class DeviceProfile:
  def __init__(self):
    self.devices = []
  def __init__(self, devices):
    self.devices = devices


################################################################################
# ProblemDescriptor
# - some problem descriptors get passed in as kernel argument and
#   Don't need to be exactly matched to solution
#   - Tensor dimensions[i].size
#   - Tensor dimensions[i].stride
#   - alpha
#   - beta
# - some problem descriptors get compiled/written into kernel and
#   Do need to be exactly matched to solution
#   - Tensor precisions
#   - dimensionality of tensors and operation
#   - operation
# - other
#   - Device - determined through benchmarking / file reading
################################################################################
class ProblemDescriptor:
  def __init__( \
      self, \
      tensorA, \
      tensorB, \
      tensorC, \
      operation, \
      device ):
    self.tensorA = tensorA
    self.tensorB = tensorB
    self.tensorC = tensorC
    self.operation = operation
    self.device = device



################################################################################
# ProblemRange
# - contiguous range of problems uniterrupted by any other ProblemRange
# - after benchmarking, MAP[Solution]->ProblemList, ProblemList needs to be
#   broken down into ProblemRanges such that:
#   - if Problem is in ProblemRange, use associated Solution
################################################################################
class ProblemRange:
  def __init__(self):
    self.devices = []
  def __init__(self, devices):
    self.devices = devices
# contains one set of SolutionCorrectnessParameters
# work-item min, max (dimA*dimB outside of summation)
# summation size, "k"


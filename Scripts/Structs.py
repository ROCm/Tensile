################################################################################
# Copyright (C) 2016 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

import SolutionCandidateGenerator

################################################################################
# Data Type - Enum
################################################################################
class DataType:
  single        = 0
  double        = 1
  complexSingle = 2
  complexDouble = 3
  half          = 4

  num           = 5
  none          = 6

  # data type properties
  toChar    = 0
  toReg     = 1
  toOpenCL  = 2
  toHIP     = 3
  toLibType = 4
  toLibEnum = 5
  #    char, reg, ocl,       hip,        libType,                libEnum
  properties = [
      [ "S", 1,   "float",   "float",   "float",                 "tensileDataTypeFloat"
      [ "D", 2,   "double",  "double",  "double",                "tensileDataTypeDouble"
      [ "C", 2,   "float2",  "float_2", "TensileComplexFloat",   "tensileDataTypeComplexFloat"
      [ "Z", 4,   "double2", "double_2", "TensileComplexDouble", "tensileDataTypeComplexDouble"
      [ "H", 0.5, "ERROR",   "fp16",     "TensileHalf",          "tensileDataTypeHalf"
  ]

  def __init__( self, value ):
    self.value = value

  def toChar(self):
    return "ERROR(" + str(self.value) + ")"

  def toOpenCL(self):
    return "ERROR(" + str(self.value) + ")"

  def toHIP(self):
    return properties[self.value][self.toOpenCL]

  def toDevice(self, backend):
    if backend.isOpenCL():
      return self.toOpenCL()
    else:
      return self.toHIP()

  def toCpp(self):
    return properties[self.value][self.toLibType]

  def getLibString(self):
    return properties[self.value][self.toLibEnum]

  def zeroString(self, backend):
    zeroString = "("
    zeroString += self.toDevice(backend)
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

  def isDouble(self):
    if self.value == self.double or self.value == self.complexDouble:
      return True
    else:
      return False

  def numRegisters( self ):
    return properties[self.value][self.toLibEnum]

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
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# Dimension
################################################################################
class Dimension:
  def __init__( self ):
    self.stride = 0
    self.size = 0

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
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# Tensor
################################################################################
class Tensor:
  def __init__( self ):
    self.dataType = DataType(-1)
    self.dimensions = []

  def __str__(self):
    name = "[Tensor"
    name += "; " + self.dataType.toChar()
    name += "; " + str(self.dimensions)
    name += "]"
    return name

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( self.dataType, tuple(self.dimensions))
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Tensor) \
        and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# Backend
################################################################################
class Backend:
  ocl = 0
  hip = 1
  asm = 2

  # property indices
  name = 0
  properties = [
      ["OCL"]
      ["HIP"]
      ["ASM"]
      ]
  def __init__( self ):
    self.value = 0

  def __str__(self):
    return self.properties[self.value][name]

  def isHIP(self):
    return self.value == self.hip

  def isOpenCL(self):
    return self.value == self.ocl

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( self.value )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Backend) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# Device
################################################################################
class Device:
  def __init__( self, name, numComputeUnits, clockFrequency, flopsPerClock):
    self.name = name
    self.numComputeUnits = numComputeUnits
    self.clockFrequency = clockFrequency
    self.flopsPerClock = flopsPerClock

  def __str__(self):
    print "Device.str"
    state = "[Device"
    state += "; " + self.name
    state += "; " + str(self.numComputeUnits)
    state += "; " + str(self.clockFrequency)
    state += "; " + str(self.flopsPerClock)
    state += "]"
    return state

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        self.name, \
        self.numComputeUnits, \
        self.clockFrequency, \
        self.flopsPerClock, \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Device) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

# ProblemSize
#  GEMM: M, N, K, [lda, ldb, ldc]
#  TensorContraction: sizeI, sizeJ, ...; [ stridesC, A, B ]


################################################################################
# ProblemType
class ProblemType:
    operationTypes = ["GEMM", "TensorContraction"]
    state = {}

  def __getitem__(self, key):
    return self.state[key]

  def __init__(self, config):
    self.assignWithDefault("OperationType", "GEMM", config)
    self.assignWithDefault("DataType", DataType(0), config)
    self.assignWithDefault("HighPrecisionAccumulate", False, config)
    self.assignWithDefault("UseBeta", True, config)
    self.assignWithDefault("UseOffsets", True, config)
    self.assignWithDefault("UseLeadingStrides", False, config)
    if self["OperationType"] == 0:
      self.initGEMM(config)
    elif self["OperationType"] == 1:
      self.initTensorContraction(config)

  def assignWithDefault(self, parameter, default, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      parameter = default

  def assign(self, parameter, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      print "Tensile::ProblemType::init ERROR - parameter \"%s\" must be defined" % parameter
      sys.exit(1)

  def initGEMM(self, config):
    self.assign("Transpose", [False, True], config)
    self.assign("Batched", False, config)
    sumIdx = 2 if self["Batched"] else 3
    self["IndexAssignmentsA"] = [0, sumIdx] # N
    self["IndexAssignmentsB"] = [sumIdx, 1] # N
    if self["Transpose"][0]: # transA
      self["IndexAssignmentsA"] = [sumIdx, 0] # T
    if self["Transpose"][1]: # transB
      self["IndexAssignmentsB"] = [1, sumIdx] # T
    if self["Batched"]:
      self["IndexAssignmentsA"].append(2)
      self["IndexAssignmentsB"].append(2)
      self["NumFreeIndices"] = 3
    else:
      self["NumFreeIndices"] = 2

  def initTensorContraction(self, config):
    self.assign("NumFreeIndices", config)
    self.assign("IndexAssignmentsA", config)
    self.assign("IndexAssignmentsB", config)

  def isGEMM(self):
    return self.operationType == 0

  def isTensorContraction(self):
    return self.operationType == 1


  def __str__(self):
    indexChars = Common.globalParameters["indexChars"]
    """
    S - single precision
    B - beta
    A - high precision accumulate
    O - offsets
    I - initial stride
    Cij_Aik_Bjk_SBAOI
    """
    # C dimensions
    name += "C"
    for i in range(0, self["NumFreeIndices"):
      name += indexChars[i].lower()
    # A dimensions
    name += "_A"
    for i in self.indexAssignmentsA:
      name += self.indexChars[i].lower()
    # B dimensions
    name += "_B"
    for i in self.indexAssignmentsB:
      name += self.indexChars[i].lower()

    # precision and other
    name += "_"
    name += self["DataType"].toChar()
    if self["HighPrecisionAccumulate"]: name += "A"
    if self["UseBeta"]: name += "B"
    if self["UseOffsets"]: name += "O"
    if self["UseInitialStrides"]: name += "I"
    return name

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return self.state
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, ExactMatch) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# Problem
class Problem:
  # sizeType=0 ranged
  # sizeType=1 exact
  def __init__( self ):
    self.tensorC = Tensor()
    self.tensorA = Tensor()
    self.tensorB = Tensor()
    self.operation = Operation()
    self.deviceProfile = DeviceProfile()
    self.sizeFree = 0
    self.sizeType = -1
    self.totalFlops = -1
    self.size0 = -1
    self.size1 = -1
    self.sizeU = -1

  def getSizeFree(self):
    if self.sizeFree == 0:
      self.sizeFree = 1
      for dimension in self.tensorC.dimensions:
        self.sizeFree *= dimension.size
    return self.sizeFree

  def getSize01U(self):
    if self.size0 < 0:
      kernel = Kernel()
      SolutionCandidateGenerator.makeIndexAssignments(kernel, self)
      self.size0 = self.tensorC.dimensions[ kernel.indexAssignmentDim0].size
      self.size1 = self.tensorC.dimensions[ kernel.indexAssignmentDim1].size
      self.sizeU = -1
      for i in range(len(self.operation.indexAssignmentsA)):
        if kernel.indexUnroll == self.operation.indexAssignmentsA[i]:
          self.sizeU = self.tensorA.dimensions[i].size
          break
    return (self.size0, self.size1, self.sizeU)

  def getSizeType(self):
    if self.sizeType < 0:
      # make index assignments
      kernel = Kernel()
      SolutionCandidateGenerator.makeIndexAssignments(kernel, self)
      # get key sizes
      problemSizeDim0 = self.tensorC.dimensions[ kernel.indexAssignmentDim0].size
      problemSizeDim1 = self.tensorC.dimensions[ kernel.indexAssignmentDim1].size
      problemSizeUnroll = -1
      for i in range(len(self.operation.indexAssignmentsA)):
        if kernel.indexUnroll == self.operation.indexAssignmentsA[i]:
          problemSizeUnroll = self.tensorA.dimensions[i].size
          break
      # if sizes are squarish, then type=0
      self.sizeType = 0
      if (not problemSizeDim0 % 16 == 0 and not (problemSizeDim0+1) % 16 == 0) or (not problemSizeDim1 % 16 == 0 and not (problemSizeDim1+1) % 16 == 0) or (not problemSizeUnroll % 16 == 0 and not (problemSizeUnroll+1) % 16 == 0):
        self.sizeType = 1
      if abs(problemSizeDim0-problemSizeDim1) > 1 or abs(problemSizeDim0-problemSizeUnroll) > 1 or abs(problemSizeDim1-problemSizeUnroll) > 1:
        self.sizeType = 1
    return self.sizeType

  def getNumFlops(self):
    if self.totalFlops < 0:
      self.totalFlops = self.getSizeFree()
      if self.tensorA.dataType.isReal():
        self.totalFlops *= 2
      else:
        self.totalFlops *= 8
      for i in range(0, len(self.operation.indexAssignmentsA)):
        index = self.operation.indexAssignmentsA[i]
        inC = index < len(self.tensorC.dimensions)
        inB = index in self.operation.indexAssignmentsB
        if inB and not inC: # is summation dimension
          self.totalFlops *= self.tensorA.dimensions[i].size
    return self.totalFlops

  def __str__(self):
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
      state += indexChars[len(self.tensorC.dimensions)+i]
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
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# EdgeType
################################################################################
class EdgeType:
  none        = 0
  branch      = 1
  shift       = 2
  multiBranch = 3
  multiShift  = 4

  toChar = 0
  toStr = 1
  properties = [
      [ "0", "None" ],
      [ "B", "Branch" ],
      [ "S", "Shift" ],
      [ "MB", "MultiBranch" ],
      [ "MS", "MultiShift" ],
      ]

  def __init__(self, value):
    self.value = value

  def __str__(self):
    return properties[self.value][toStr]

  def getChar(self):
    return properties[self.value][toChar]

  def isNone(self):
    return self.value == self.none
  def isBranch(self):
    return self.value == self.branch
  def isShift(self):
    return self.value == self.shift
  def isMultiBranch(self):
    return self.value == self.multiBranch
  def isMultiShift(self):
    return self.value == self.multiShift

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return (self.value)
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, EdgeType) \
        and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# Tile
################################################################################
class Tile:
  def __init__( self ):
    self.workGroup = [ -1, -1]
    self.microTile = [ -1, -1]
    self.edge = [ EdgeType(-1), EdgeType(-1)]

  def __str__(self):
    state = "[Tile; " + str(self.workGroup[0]) + "x" + str(self.workGroup[1])
    state += "; " + str(self.microTile[0]) + "x" + str(self.microTile[1])
    state += "; " + str(self.edge[0]) + "x" + str(self.edge[1])
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
        self.edge[0], \
        self.edge[1], \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Tile) \
        and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# Kernel
################################################################################
class Kernel:
  state = {
      "ProblemType":            ProblemType(),
      "IndexAssignmentDim0":    -1, # assigned ?
      "IndexAssignmentDim1":    -1, # assigned ?
      "TileDimCoalescedA":      True, # assigned ? "UnrollDimStrideGreaterThanTileDimStrideA": False # if true fast "N"
      "TileDimCoalescedB":      True, # assigned ? "UnrollDimStrideLessThanTileDimStrideB": False # if true slow "N"
      "WorkGroupOrder":         -4, # <0 means d1; >0 means d0, integer means blocking
      "MicroTileEdge":          4, # 1, 2, 4, 6, 8
      "MicroTileShape":         0, # -1 (d1=d0/2), 0(d1=d0), 1(d1=d0*2)
      "WorkGroupEdge":          16, # 8, 16
      "WorkGroupShape":         0, # -1 (d1=d0/2), 0(d1=d0), 1(d1=d0*2)
      "LoopFor":                True, # true = For, false = DoWhile
      "LoopUnroll":             16,
      "LoopTail":               True,
      "EdgeType":               EdgeType(2),
      "NumLoadsParaA":          -1,
      "NumLoadsParaB":          -1,
      "GlobalLoadVectorWidth":  4,
      "LocalStoreVectorWidth":  4,
      "LocalLoadVectorWidth":   4,
      "GlobalStoreVectorWidth": 4,
      "LoadMacInterleave":      4,
      "SplitK":                 2,
      "Prefetch":               True,
      "AtomicAccumulate":       False
      }





  def __getitem__(self, key):
    return self.state[key]

  def __init__(self, config):
    print "Tensile::Kernel::init not implemented"

  def assignWithDefault(self, parameter, default, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      parameter = default

  def assign(self, parameter, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      print "Tensile::Kernel::init ERROR - parameter \"%s\" must be defined" % parameter
      sys.exit(1)

  # create a dictionary with booleans on whether to include
  @classmethod
  def getMinNaming(kernels):
    requiredParameters = {}
    for key in kernels[0]:
      required = False
      for i in range(1, len(kernels)):
        if kernels[0][key] != kernels[i][key]:
          required = True
          break
      if required:
        requiredParameters[key] = True
    return requiredParameters

  def getNameFull(self):
    requiredParameters = {}
    for key in state:
      requiredParameters[key] = True
    return getNameMin(requiredParameters)

  def getNameMin(self, requiredParameters):
    name = ""
    for key in self.state:
      if requiredParameters[key]:
        name += getParameterAbbreviation(key)
        name += str(self.state[key])
        name += "_"
    return name

  @ classmethod
  def getParameterAbbreviation( parameter ):
    abbreviation = ''.join([c for c in s if c.isupper()])
    return abbreviation


  def __str__(self):
    return self.getNameFull()

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return self.state
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Kernel) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# Solution
################################################################################
class Solution:
  def __init__(self):
    # Solution Correctness Parameters
    # Kernels
    self.kernels = []
    self.kernelGrid = [ -1, -1, -1 ]
    self.kernelsConcurrent = True


  def __str__(self):
    name = "[Solution"
    name += "; " + str(self.kernelGrid)
    name += "; " + str(self.kernels)
    nmae += "]"
    return name

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        tuple(self.kernels), \
        self.kernelGrid[0], \
        self.kernelGrid[1], \
        self.edge[0], \
        self.edge[1], \
        self.ppdOffsets, \
        self.ppdLeadingStrides, \
        self.ppdAll )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Solution) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


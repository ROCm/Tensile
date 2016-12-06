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

import sys

import Common

################################################################################
# Data Type
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
  idxChar    = 0
  idxReg     = 1
  idxOpenCL  = 2
  idxHIP     = 3
  idxLibType = 4
  idxLibEnum = 5
  #    char, reg, ocl,       hip,        libType,                libEnum
  properties = [
      [ "S", 1,   "float",   "float",   "float",                 "tensileDataTypeFloat"         ],
      [ "D", 2,   "double",  "double",  "double",                "tensileDataTypeDouble"        ],
      [ "C", 2,   "float2",  "float_2", "TensileComplexFloat",   "tensileDataTypeComplexFloat"  ],
      [ "Z", 4,   "double2", "double_2", "TensileComplexDouble", "tensileDataTypeComplexDouble" ],
      [ "H", 0.5, "ERROR",   "fp16",     "TensileHalf",          "tensileDataTypeHalf"          ]
  ]

  def __init__( self, value ):
    if isinstance(value, int):
      self.value = value
    if isinstance(value, str):
      for propertiesIdx in range(0,6):
        for dataTypeIdx in range(0,self.num):
          if value.lower() == self.properties[dataTypeIdx][propertiesIdx].lower():
            self.value = dataTypeIdx
            return


  def toChar(self):
    return self.properties[self.value][self.idxChar]

  def toOpenCL(self):
    return self.properties[self.value][self.idxOpenCL]

  def toHIP(self):
    return self.properties[self.value][self.idxOpenCL]

  def toDevice(self, backend):
    if backend.isOpenCL():
      return self.toOpenCL()
    else:
      return self.toHIP()

  def toCpp(self):
    return self.properties[self.value][self.idxLibType]

  def getLibString(self):
    return self.properties[self.value][self.idxLibEnum]

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
  idxName = 0
  properties = [
      ["OCL"],
      ["HIP"],
      ["ASM"]
      ]
  def __init__( self ):
    self.value = 0

  def __str__(self):
    return self.properties[self.value][idxName]

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
  def __setitem__(self, key, value):
    self.state[key] = value

  def __init__(self, config):
    self.assignWithDefault("OperationType", "GEMM", config)
    self.assignWithDefault("HighPrecisionAccumulate", False, config)
    self.assignWithDefault("UseBeta", True, config)
    self.assignWithDefault("UseOffsets", True, config)
    self.assignWithDefault("UseInitialStrides", False, config)

    if "DataType" in config:
      self["DataType"] = DataType(config["DataType"])
    else:
      self["DataType"] = DataType(0)

    if self["OperationType"] == "GEMM":
      self.initGEMM(config)
    elif self["OperationType"] == "TensorContraction":
      self.initTensorContraction(config)

  def assignWithDefault(self, parameter, default, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      self[parameter] = default

  def assign(self, parameter, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      sys.exit("Tensile::ProblemType::init ERROR - parameter \"%s\" must be defined" % parameter)

  def initGEMM(self, config):
    self.assignWithDefault("TransposeA", False, config)
    self.assignWithDefault("TransposeB", True, config)
    self.assignWithDefault("Batched", False, config)
    sumIdx = 3 if self["Batched"] else 2
    self["IndexAssignmentsA"] = [0, sumIdx] # N
    self["IndexAssignmentsB"] = [sumIdx, 1] # N
    if self["TransposeA"]:
      self["IndexAssignmentsA"] = [sumIdx, 0] # T
    if self["TransposeB"]:
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
    name = "C"
    name += indexChars[:self["NumFreeIndices"]].lower()
    # A dimensions
    name += "_A"
    for i in self["IndexAssignmentsA"]:
      name += indexChars[i].lower()
    # B dimensions
    name += "_B"
    for i in self["IndexAssignmentsB"]:
      name += indexChars[i].lower()

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
# Solution
################################################################################
class Solution:
  state = {}

  def __init__(self, config):
    # problem type
    if "ProblemType" in config:
      self["ProblemType"] = ProblemType(config["ProblemType"])
    else:
      sys.exit("Tensile::%s::%s: ERROR - No ProblemType in config: %s" % ( __file__, __line__, str(config) ))

    # solution parameters
    self.assignWithDefault("KernelGrid0", 1, config)
    self.assignWithDefault("KernelGrid1", 1, config)
    self.assignWithDefault("KernelGridU", 1, config)
    self.assignWithDefault("KernelsSerial", True, config)
    # kernel parameters
    self.assignWithDefault("WorkGroupOrder", 1, config)
    self.assignWithDefault("MicroTileEdge", 4, config)
    self.assignWithDefault("MicroTileShape", 0, config)
    self.assignWithDefault("WorkGroupEdge", 16, config)
    self.assignWithDefault("WorkGroupShape", 0, config)
    self.assignWithDefault("LoopFor", True, config)
    self.assignWithDefault("LoopUnroll", 16, config)
    self.assignWithDefault("LoopTail", True, config)
    self.assignWithDefault("NumLoadsParaA", 1, config)
    self.assignWithDefault("NumLoadsParaB", 1, config)
    self.assignWithDefault("GlobalLoadVectorWidth", 4, config)
    self.assignWithDefault("LocalStoreVectorWidth", 4, config)
    self.assignWithDefault("LocalLoadVectorWidth", 4, config)
    self.assignWithDefault("GlobalStoreVectorWidth", 4, config)
    self.assignWithDefault("LoadMacInterleave", 4, config)
    self.assignWithDefault("SplitK", 1, config)
    self.assignWithDefault("Prefetch", True, config)
    self.assignWithDefault("AtomicAccumulate", False, config)
    self.assignWithDefault("EdgeType", "Shift", config) # None, Shift, Branch, MultiShift, MultiBranch
    # need problem sizes to assign - TODO ? reasonable default for Tensors?
    self.assignWithDefault("IndexAssignmentDim0", 0, config)
    self.assignWithDefault("IndexAssignmentDim1", 1, config)
    self.assignWithDefault("TileDimCoalescedA", True, config)
    self.assignWithDefault("TileDimCoalescedB", True, config)

  def assignWithDefault(self, parameter, default, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      self[parameter] = default

  def assign(self, parameter, config):
    if parameter in config:
      self[parameter] = config[parameter]
    else:
      sys.exit("Tensile::Solution::init: ERROR - parameter \"%s\" must be defined" % parameter)

  def __getitem__(self, key):
    return self.state[key]
  def __setitem__(self, key, value):
    self.state[key] = value

  # create a dictionary with booleans on whether to include parameter in name
  @staticmethod
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
    for key in self.state:
      requiredParameters[key] = True
    return self.getNameMin(requiredParameters)

  def getNameMin(self, requiredParameters):
    name = ""
    for key in self.state:
      if requiredParameters[key]:
        name += self.getParameterNameAbbreviation(key)
        name += self.getParameterValueAbbreviation(self[key])
        name += "_"
    return name

  @ staticmethod
  def getParameterNameAbbreviation( name ):
    return ''.join([c for c in name if c.isupper()])

  @ staticmethod
  def getParameterValueAbbreviation( value ):
    if isinstance(value, str):
      return ''.join([c for c in value if c.isupper()])
    elif isinstance(value, bool):
      return "1" if value else "0"
    elif isinstance(value, int):
      return str(value)
    elif isinstance(value, ProblemType):
      return str(value)
    else:
      sys.exit("Tensile::Solution::abbrev WARNING - parameter \"%s\" is new object type." % value)
      return str(value)

  def __str__(self):
    return self.getNameFull()

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return ( \
        tuple(self.kernels), \
        self.kernelGrid0, \
        self.kernelGrid1, \
        self.kernelGrid2, \
        )
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Solution) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


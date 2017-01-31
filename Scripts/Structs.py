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


from Common import *
import sys
from copy import deepcopy

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

  ########################################
  def __init__( self, value ):
    if isinstance(value, int):
      self.value = value
    elif isinstance(value, basestring):
      for propertiesIdx in range(0,6):
        for dataTypeIdx in range(0,self.num):
          if value.lower() == self.properties[dataTypeIdx][propertiesIdx].lower():
            self.value = dataTypeIdx
            return
    elif isinstance(value, DataType):
      self.value = value.value
    else:
      printExit("initializing DataType to %s %s" % (str(type(value)), str(value)) )


  ########################################
  def toChar(self):
    return self.properties[self.value][self.idxChar]
  def toOpenCL(self):
    return self.properties[self.value][self.idxOpenCL]
  def toHIP(self):
    return self.properties[self.value][self.idxOpenCL]
  def toDevice(self, backend):
    if backend == "OCL":
      return self.toOpenCL()
    else:
      return self.toHIP()
  def toCpp(self):
    return self.properties[self.value][self.idxLibType]
  def getLibString(self):
    return self.properties[self.value][self.idxLibEnum]

  ########################################
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

  ########################################
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

  ########################################
  def numRegisters( self ):
    return self.properties[self.value][self.idxReg]
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
# Device
################################################################################
class Device:

  ########################################
  def __init__( self, name, numComputeUnits, clockFrequency, flopsPerClock):
    self.name = name
    self.numComputeUnits = numComputeUnits
    self.clockFrequency = clockFrequency
    self.flopsPerClock = flopsPerClock

  ########################################
  def __str__(self):
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
# name of solution should begin with name of problemType, and arguments can be listed out explicitly
class ProblemType:
  operationTypes = ["GEMM", "TensorContraction"]

  ########################################
  def __init__(self, config):
    self.state = {}
    for key in defaultProblemType:
      self.assignWithDefault(key, defaultProblemType[key], config)

    if "DataType" in config:
      self["DataType"] = DataType(config["DataType"])
    else:
      self["DataType"] = DataType(0)

    if self["OperationType"] == "GEMM":
      self.initGEMM(config)
    elif self["OperationType"] == "TensorContraction":
      self.initTensorContraction(config)

    self.assignIndices()


  ########################################
  def initGEMM(self, config):
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
      self["NumIndicesC"] = 3
    else:
      self["NumIndicesC"] = 2

  ########################################
  def initTensorContraction(self, config):
    self.assign("NumIndicesC", config)
    self.assign("IndexAssignmentsA", config)
    self.assign("IndexAssignmentsB", config)

  ########################################
  def isGEMM(self):
    return self.operationType == 0

  ########################################
  def isTensorContraction(self):
    return self.operationType == 1

  ########################################
  # determine d0, d1, dU
  def assignIndices(self):
    self["TotalIndices"] = max(max(self["IndexAssignmentsA"])+1, max(self["IndexAssignmentsB"])+1)

    # determine num free, batch
    self["IndicesFree"] = []
    self["IndicesBatch"] = []
    self["IndicesSummation"] = []

    for i in range(0, self["NumIndicesC"]):
      inA = i in self["IndexAssignmentsA"]
      inB = i in self["IndexAssignmentsB"]
      if inA and inB:
        #self["NumIndicesBatch"] = (i+1)-self["NumIndicesFree"]
        self["IndicesBatch"].append(i)

      elif inA or inB:
        #self["NumIndicesFree"] = (i+1)
        self["IndicesFree"].append(i)
      else:
        printExit("invalid index %u" % i)

    # determine num summation
    for i in range(self["NumIndicesC"], self["TotalIndices"]):
      inA = i in self["IndexAssignmentsA"]
      inB = i in self["IndexAssignmentsB"]
      if inA and inB:
        #self["NumIndicesSummation"] = (i+1)-self["NumIndicesC"]
        self.state["IndicesSummation"].append(i)
      else:
        printExit("invalid index %u" % i)
    self["NumIndicesFree"] = len(self["IndicesFree"])
    self["NumIndicesBatch"] = len(self["IndicesBatch"])
    self["NumIndicesSummation"] = len(self["IndicesSummation"])


    # by default, unroll index will be the first summation index
    # TODO sort summation indices by "stride"
    self["IndexUnroll"] = self["IndicesSummation"][0]
    for i in range(0, len(self["IndexAssignmentsA"])):
      if self["IndexAssignmentsA"][i] == self["IndexUnroll"]:
        self["IndexUnrollA"] = i
        break
    for i in range(0, len(self["IndexAssignmentsB"])):
      if self["IndexAssignmentsB"][i] == self["IndexUnroll"]:
        self["IndexUnrollB"] = i
        break

    # assign d0, d1
    self["Index01A"] = -1
    self["Index01B"] = -1
    for i in self["IndexAssignmentsA"]:
      if i < self["NumIndicesC"]:
        self["Index01A"] = i
        break
    for i in self["IndexAssignmentsB"]:
      if i < self["NumIndicesC"]:
        self["Index01B"] = i
        break
    # whichever has lower stride in C (lower value), is 0, other is 1
    if self["Index01A"] < self["Index01B"]:
      self["Index0"]  = self["Index01A"]
      self["Index1"]  = self["Index01B"]
      self["Tensor0"] = 0
      self["Tensor1"] = 1
      self["TileA"] = 0
      self["TileB"] = 1
    else:
      self["Index0"]  = self["Index01B"]
      self["Index1"]  = self["Index01A"]
      self["Tensor0"] = 1
      self["Tensor1"] = 0
      self["TileA"] = 1
      self["TileB"] = 0

    # generalize transpose
    strideIdxA = self["IndexAssignmentsA"].index(self["Index01A"])
    strideIdxB = self["IndexAssignmentsB"].index(self["Index01B"])
    unrollIdxA = self["IndexAssignmentsA"].index(self["IndexUnroll"])
    unrollIdxB = self["IndexAssignmentsB"].index(self["IndexUnroll"])
    self["TLUA"] = strideIdxA < unrollIdxA
    self["TLUB"] = strideIdxB < unrollIdxB

    #unrollDimStrideGreaterThanTileDimStrideA = TLUA
    #unrollDimStrideLessThanTileDimStrideB    = !TLUB



  ########################################
  def __str__(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, self["NumIndicesC"]):
      name += indexChars[i].lower()
    # A dimensions
    name += "_A"
    for i in self["IndexAssignmentsA"]:
      name += indexChars[i].lower()
    if self["ComplexConjugateA"]:
      name += "C"
    # B dimensions
    name += "_B"
    for i in self["IndexAssignmentsB"]:
      name += indexChars[i].lower()
    if self["ComplexConjugateB"]:
      name += "C"

    # precision and other
    name += "_"
    name += self["DataType"].toChar()
    if self["UseBeta"]: name += "B"
    if self["HighPrecisionAccumulate"]: name += "H"
    if self["UseInitialStrides"]: name += "I"
    return name

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
  def __getitem__(self, key):
    return self.state[key]
  def __setitem__(self, key, value):
    self.state[key] = value
  def __repr__(self):
    return self.__str__()
  def getAttributes(self):
    return self.state
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, ProblemType) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# ProblemSizes
################################################################################
class ProblemSizes:

  ########################################
  def __init__(self, problemType, config):
    self.totalIndices = 1+max(problemType["IndexAssignmentsA"])
    if len(config) < self.totalIndices:
      #printDefault("SizeRange config (%s) has too few elements (%u < %u) than required by ProblemType (%s); appending defaults." \
          #% ( str(config), len(config), self.totalIndices, problemType ))
      for i in range(len(config), self.totalIndices):
        config.append(0)
    #if len(config) < self.totalIndices:
      #printDefault("SizeRange config (%s) has too many elements (%u > %u) than required by ProblemType (%s); ignoring remainder." \
          # % ( str(config), len(config), self.totalIndices, problemType ))
    self.indexMax = []
    self.indexIsSized = []
    self.indicesSized = []
    self.indicesMapped = []
    for i in range(0, self.totalIndices):
      dim = deepcopy(config[i])
      if isinstance(dim, list):
        if len(dim) == 1:
          self.indicesSized.append([dim[0], 1, 0, dim[0]])
        elif len(dim) == 2:
          self.indicesSized.append([dim[0], dim[0], 0, dim[1]])
        elif len(dim) == 3:
          self.indicesSized.append([dim[0], dim[1], 0, dim[2]])
        elif len(dim) == 4:
          self.indicesSized.append([dim[0], dim[1], dim[2], dim[3]])
        else:
          printExit("dimension[%u] config (%s) has %u descriptors rather than 1-4."
              % ( i, dim, len(dim) ))
        self.indexIsSized.append(True)
        self.indexMax.append(self.indicesSized[len(self.indicesSized)-1][3])

      elif isinstance(dim, int):
        self.indicesMapped.append(dim)
        self.indexIsSized.append(False)
        self.indexMax.append(self.indicesSized[self.indicesMapped[len(self.indicesMapped)-1]][3])

    # max num elements in each tensor
    self.maxNumElements = [ 1, 1, 1 ]
    for i in range(0, problemType["NumIndicesC"]):
      self.maxNumElements[0] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsA"]:
      self.maxNumElements[1] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsB"]:
      self.maxNumElements[2] *= self.indexMax[i]

    self.totalProblemSizes = 0
    currentSizedIndexSizes = []
    currentSizedIndexIncrements = []
    for index in self.indicesSized:
      currentSizedIndexSizes.append(index[0])
      currentSizedIndexIncrements.append(index[1])
    numIndicesSized = len(self.indicesSized)
    moreProblemSizes = True
    #print "Counting Problem Sizes: %s" % self.indicesSized
    while moreProblemSizes:
      #print "Size[%u]: sizes=%s incr=%s" % (self.totalProblemSizes, \
      #    currentSizedIndexSizes, currentSizedIndexIncrements)
      self.totalProblemSizes += 1
      currentSizedIndexSizes[0] += currentSizedIndexIncrements[0]
      currentSizedIndexIncrements[0] += self.indicesSized[0][2]
      for i in range(1, numIndicesSized+1):
        if currentSizedIndexSizes[i-1] > self.indicesSized[i-1][3]:
          # reset prior index size and incr
          currentSizedIndexSizes[i-1] = self.indicesSized[i-1][0]
          currentSizedIndexIncrements[i-1] = self.indicesSized[i-1][1]
          # increment next index
          if i == numIndicesSized:
            moreProblemSizes = False
          else:
            currentSizedIndexSizes[i] += currentSizedIndexIncrements[i];
            currentSizedIndexIncrements[i] += self.indicesSized[i][2];
    #print "ProblemSizes: %u" % self.totalProblemSizes


  def __str__(self):
    state = "[ "
    sizedIdx = 0
    mappedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      if self.indexIsSized[i]:
        indices = self.indicesSized[sizedIdx]
        state += "[ %u, %u, %u, %u ]" % (indices[0], indices[1], indices[2], indices[3])
        sizedIdx += 1
      else:
        indices = self.indicesSized[self.indicesMapped[mappedIdx]]
        state += str(self.indicesMapped[mappedIdx])
        mappedIdx += 1
      if i < len(self.indexIsSized)-1:
        state += ", "
    state += " ]"
    return state




# this will have a list of index size assignments
#order of assignments: i, j, k, l, m, ...


################################################################################
# Solution
################################################################################
class Solution:

  ########################################
  def __init__(self, config):
    self.state = {}
    # problem type
    if "ProblemType" in config:
      self["ProblemType"] = ProblemType(config["ProblemType"])
    else:
      self["ProblemType"] = ProblemType(defaultProblemType)

    # assign parameters with defaults
    for key in defaultSolution:
      self.assignWithDefault(key, defaultSolution[key], config)

    # assign parameters without defaults
    for key in config:
      if key != "ProblemType" and key not in self.state:
        #print "Solution::init() - WARNING: appending unrecognized %s=%s" \
        #    % (key, config[key])
        self.state[key] = config[key]

    Solution.assignDimsFromEdgeAndShape(self.state)

  ########################################
  # get a list of kernel parameters for this solution
  # kernels have edge0,1=T/F
  def getKernels(self):
    kernels = []
    if self.state["EdgeType"] == "MultiBranch" or self.state["EdgeType"] == "MultiShift":
      kernel00 = deepcopy(self.state)
      kernel00.update({"Edge0": False, "Edge1": False})
      kernel10 = deepcopy(self.state)
      kernel10.update({"Edge0": True, "Edge1": False})
      kernel01 = deepcopy(self.state)
      kernel01.update({"Edge0": False, "Edge1": True})
      kernels.append(kernel00)
      kernels.append(kernel10)
      kernels.append(kernel01)
    kernel11 = deepcopy(self.state)
    kernel11.update({"Edge0": True, "Edge1": True})
    kernels.append(kernel11)
    return kernels


  ########################################
  # assign Dim0, 1 based on edge and shape
  @staticmethod
  def assignDimsFromEdgeAndShape(state):
    # workgroup sizes
    state["WorkGroup0"] = state["WorkGroupEdge"]
    state["WorkGroup1"] = state["WorkGroupEdge"]
    if state["WorkGroupShape"] == 1:
      state["WorkGroup1"] *= 2
    elif state["WorkGroupShape"] == -1:
      state["WorkGroup0"] *= 2

    # thread tile sizes
    state["ThreadTile0"] = state["ThreadTileEdge"]
    state["ThreadTile1"] = state["ThreadTileEdge"]
    if state["ThreadTileShape"] == 1:
      state["ThreadTile1"] *= 2
    elif state["ThreadTileShape"] == -1:
      state["ThreadTile0"] *= 2

    # macro tile sizes
    state["MacroTile0"] = state["WorkGroup0"]*state["ThreadTile0"]
    state["MacroTile1"] = state["WorkGroup1"]*state["ThreadTile1"]


  ########################################
  # create a dictionary with booleans on whether to include parameter in name
  @staticmethod
  def getMinNaming(objs):
    # early return
    if len(objs) == 0:
      return {}
    # determine keys
    requiredParameters = {}
    if isinstance(objs[0], Solution):
      keys = list(objs[0].state.keys())
    else:
      keys = list(objs[0].keys())
    # only 1, rather than name being nothing, it'll be everything
    if len(objs) == 1:
      for key in keys:
        requiredParameters[key] = True
    else:
      for key in keys:
        required = False
        for i in range(1, len(objs)):
          if objs[0][key] != objs[i][key]:
            required = True
            break
        if required:
          requiredParameters[key] = True
        else:
          requiredParameters[key] = False
    requiredParameters["ProblemType"] = False # always prepended anyways
    # kernels need edge name to distinguish from solution name
    requiredParameters["Edge0"] = True
    requiredParameters["Edge1"] = True
    return requiredParameters

  ########################################
  @ staticmethod
  def getNameFull(state):
    requiredParameters = {}
    for key in state:
      requiredParameters[key] = True
    return Solution.getNameMin(state, requiredParameters)

  ########################################
  # TODO limit parameters to those in global, not derrived ones
  @ staticmethod
  def getNameMin(state, requiredParameters):
    name = ""
    first = True
    # put problem first
    if "ProblemType" in state:
      name += str(state["ProblemType"]) + "_"
    for key in sorted(state.keys()):
      if key in requiredParameters:
        if requiredParameters[key]:
          if not first:
            name += "_"
          else:
            first = False
          name += "%s%s" % ( Solution.getParameterNameAbbreviation(key), \
              Solution.getParameterValueAbbreviation(state[key]) )
      else:
        print "%s not in %s" % (key, requiredParameters)
    return name

  ########################################
  # create a dictionary of lists of parameter values
  @staticmethod
  def getSerialNaming(objs):
    data = {}
    for objIdx in range(0, len(objs)):
      #print "ObjIdx: %u" % objIdx
      obj = objs[objIdx]
      for paramName in sorted(obj.keys()):
        if paramName not in derrivedParameters:
          paramValue = obj[paramName]
          #if paramName == "ThreadTileEdge":
          #  print "%s = %s" % (paramName, paramValue)
          if paramName in data:
            if paramValue not in data[paramName]:
              data[paramName].append(paramValue)
          else:
            data[paramName] = [ paramValue ]
    maxObjs = 1
    #print "SerialNaming:"
    for paramName in data:
      data[paramName] = sorted(data[paramName])
      #print "%s: %s" % (paramName, data[paramName])
      maxObjs *= len(data[paramName])
    numDigits = len(str(maxObjs))
    print "MaxSerialNames: %u (%u)" % (maxObjs, numDigits)
    return [ data, numDigits ]

  ########################################
  # Get Name Serial
  @ staticmethod
  def getNameSerial(state, serialNaming):
    data = serialNaming[0]
    numDigits = serialNaming[1]

    serial = 0
    multiplier = 1
    for paramName in sorted(state.keys()):
      if paramName not in derrivedParameters:
        paramValue = state[paramName]
        paramData = data[paramName]
        paramNameMultiplier = len(paramData)
        if paramValue in paramData:
          paramValueIdx = paramData.index(paramValue)
        #else:
          #print "ERROR %s: %s not in %s" % ( paramName, paramValue, paramData )
          #print state
          #printExit()
        #if paramNameMultiplier > 1:
          #print "serial = %u*%u + %u; multiplier = %u * %u; %s::%s in %s" % ( \
          #    paramValueIdx, multiplier, serial, \
          #    paramNameMultiplier, multiplier, \
          #    paramName, paramValue, paramData[1] )

        serial += paramValueIdx * multiplier
        multiplier *= paramNameMultiplier
    #if serial == 0:
    #  print state
    name = "%s%0*u" % ("S" if isinstance(state, Solution) else "K", \
        numDigits, serial)
    #print "SerialName: %s" % name
    return name


  ########################################
  @ staticmethod
  def getParametersIndented(state, indent):
    s = ""
    first = True
    # put problem first
    s += "%sProblemType: %s\n" % (indent, str(state["ProblemType"]))
    for key in state:
      s += "%s%s: %s\n" % (indent, str(key), str(state[key]))
    return s

  ########################################
  @ staticmethod
  def getParameterNameAbbreviation( name ):
    return ''.join([c for c in name if not c.islower()])

  ########################################
  @ staticmethod
  def getParameterValueAbbreviation( value ):
    if isinstance(value, str):
      return ''.join([c for c in value if c.isupper()])
    elif isinstance(value, bool):
      return "1" if value else "0"
    elif isinstance(value, int):
      if value >= 0:
        return "%02u" % value
      else: # -1 -> n1
        return "n%01u" % abs(value)
    elif isinstance(value, ProblemType):
      return str(value)
    elif isinstance(value, list):
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += Solution.getParameterValueAbbreviation(value[i])
        if i < len(value)-1:
          abbrev += "_"
      return abbrev
    else:
      print value
      printExit("Parameter \"%s\" is new object type" % str(value) )
      return str(value)

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
  # make class look like dict
  def keys(self):
    return self.state.keys()
  def __len__(self):
    return len(self.state)
  def __iter__(self):
    return iter(self.state)

  def __getitem__(self, key):
    return self.state[key]
  def __setitem__(self, key, value):
    self.state[key] = value
  def __str__(self):
    return Solution.getNameFull(self.state)
  def __repr__(self):
    return self.__str__()
  def getAttributes(self):
    return state
  def __hash__(self):
    return hash(self.getAttributes())
  def __eq__(self, other):
    return isinstance(other, Solution) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


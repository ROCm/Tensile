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


from Common import globalParameters, defaultProblemType, assignParameterWithDefault, printExit, assignParameterRequired, defaultSolution, validParameters, print1
from copy import deepcopy
from math import ceil, log

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
  #    char, reg, ocl,       hip,       libType,                 libEnum
  properties = [
      [ "S", 1,   "float",   "float",   "float",                 "tensileDataTypeFloat"         ],
      [ "D", 2,   "double",  "double",  "double",                "tensileDataTypeDouble"        ],
      [ "C", 2,   "float2",  "float2",  "TensileComplexFloat",   "tensileDataTypeComplexFloat"  ],
      [ "Z", 4,   "double2", "double2", "TensileComplexDouble",  "tensileDataTypeComplexDouble" ],
      [ "H", 0.5, "ERROR",   "half",    "TensileHalf",           "tensileDataTypeHalf"          ]
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
    return self.properties[self.value][self.idxHIP]
  def toDevice(self, language):
    if language == "OCL":
      return self.toOpenCL()
    else:
      return self.toHIP()
  def toCpp(self):
    return self.properties[self.value][self.idxLibType]
  def getLibString(self):
    return self.properties[self.value][self.idxLibEnum]

  ########################################
  def zeroString(self, language, vectorWidth):
    if language == "HIP":
      if self.value == self.complexSingle:
        return "make_float2(0.f, 0.f)"
      if self.value == self.complexDouble:
        return "make_double2(0.0, 0.0)"

    zeroString = "("
    zeroString += self.toDevice(language)
    if vectorWidth > 1:
      zeroString += str(vectorWidth)
    zeroString += ")("

    """
    if self.value == self.half:
      single = "0"
      vectorWidth = 1
    elif self.value == self.single:
      single = "0.f"
    elif self.value == self.double:
      single = "0.0"
    elif self.value == self.complexSingle:
      single = "0.f, 0.f"
    elif self.value == self.complexDouble:
      single = "0.0, 0.0"
    """
    zeroString += "0"
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
    return self.value == self.double or self.value == self.complexDouble
  def isSingle(self):
    return self.value == self.single
  def isHalf(self):
    return self.value == self.half

  ########################################
  def numRegisters( self ):
    return self.properties[self.value][self.idxReg]
  def numBytes( self ):
    return self.numRegisters() * 4
  def flopsPerMac(self):
    return 2 if self.isReal() else 8

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
# ProblemType
# name of solution should begin with name of problemType, and arguments can be listed out explicitly
class ProblemType:
  operationTypes = ["GEMM", "TensorContraction"]

  ########################################
  def __init__(self, config):
    self.state = {}
    for key in defaultProblemType:
      assignParameterWithDefault(self.state, key, config, defaultProblemType)

    if "DataType" in config:
      self["DataType"] = DataType(config["DataType"])
    else:
      printExit("NO data type specified")
      self["DataType"] = DataType(0)

    if self["OperationType"] == "GEMM":
      self.initGEMM(config)
    elif self["OperationType"] == "TensorContraction":
      self.initTensorContraction(config)

    self.state["AssignedDerivedParameters"] = False
    ProblemType.assignDerivedParameters(self.state)


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
    assignParameterRequired(self.state, "NumIndicesC", config)
    assignParameterRequired(self.state, "IndexAssignmentsA", config)
    assignParameterRequired(self.state, "IndexAssignmentsB", config)

  ########################################
  def isGEMM(self):
    return self.operationType == 0

  ########################################
  def isTensorContraction(self):
    return self.operationType == 1

  ########################################
  # determine d0, d1, dU
  @staticmethod
  def assignDerivedParameters(state):
    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    state["TotalIndices"] = max(max(state["IndexAssignmentsA"])+1, \
        max(state["IndexAssignmentsB"])+1)

    # determine num free, batch
    state["IndicesFree"] = []
    state["IndicesBatch"] = []
    state["IndicesSummation"] = []

    for i in range(0, state["NumIndicesC"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        #state["NumIndicesBatch"] = (i+1)-state["NumIndicesFree"]
        state["IndicesBatch"].append(i)

      elif inA or inB:
        #state["NumIndicesFree"] = (i+1)
        state["IndicesFree"].append(i)
      else:
        printExit("invalid index %u" % i)

    # determine num summation
    for i in range(state["NumIndicesC"], state["TotalIndices"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        #state["NumIndicesSummation"] = (i+1)-state["NumIndicesC"]
        state["IndicesSummation"].append(i)
      else:
        printExit("invalid index %u" % i)
    state["NumIndicesFree"] = len(state["IndicesFree"])
    state["NumIndicesBatch"] = len(state["IndicesBatch"])
    state["NumIndicesSummation"] = len(state["IndicesSummation"])


    # by default, unroll index will be the first summation index
    # TODO sort summation indices by "stride"
    state["IndexUnroll"] = state["IndicesSummation"][0]
    for i in range(0, len(state["IndexAssignmentsA"])):
      if state["IndexAssignmentsA"][i] == state["IndexUnroll"]:
        state["IndexUnrollA"] = i
        break
    for i in range(0, len(state["IndexAssignmentsB"])):
      if state["IndexAssignmentsB"][i] == state["IndexUnroll"]:
        state["IndexUnrollB"] = i
        break

    # assign d0, d1
    state["Index01A"] = -1
    state["Index01B"] = -1
    for i in state["IndexAssignmentsA"]:
      if i < state["NumIndicesC"]:
        state["Index01A"] = i
        break
    for i in state["IndexAssignmentsB"]:
      if i < state["NumIndicesC"]:
        state["Index01B"] = i
        break
    # whichever has lower stride in C (lower value), is 0, other is 1
    if state["Index01A"] < state["Index01B"]:
      state["Index0"]  = state["Index01A"]
      state["Index1"]  = state["Index01B"]
      state["Tensor0"] = 0
      state["Tensor1"] = 1
      state["TileA"] = 0
      state["TileB"] = 1
    else:
      state["Index0"]  = state["Index01B"]
      state["Index1"]  = state["Index01A"]
      state["Tensor0"] = 1
      state["Tensor1"] = 0
      state["TileA"] = 1
      state["TileB"] = 0

    # generalize transpose
    strideIdxA = state["IndexAssignmentsA"].index(state["Index01A"])
    strideIdxB = state["IndexAssignmentsB"].index(state["Index01B"])
    unrollIdxA = state["IndexAssignmentsA"].index(state["IndexUnroll"])
    unrollIdxB = state["IndexAssignmentsB"].index(state["IndexUnroll"])
    state["TLUA"] = strideIdxA < unrollIdxA
    state["TLUB"] = strideIdxB < unrollIdxB

    #unrollDimStrideGreaterThanTileDimStrideA = TLUA = !transA = fast
    #!unrollDimStrideLessThanTileDimStrideB   = TLUB =  transB = fast
    state["AssignedDerivedParameters"] = True



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
  def __repr__(self):
    return self.__str__()
  def getAttributes(self):
    return self.state
  def __hash__(self):
    return hash(str(self))
  def __eq__(self, other):
    return isinstance(other, ProblemType) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


################################################################################
# ProblemSizeRange
################################################################################
class ProblemSizeRange:

  ########################################
  def __init__(self, problemType, config):
    self.totalIndices = 1+max(problemType["IndexAssignmentsA"])
    if len(config) < self.totalIndices:
      for i in range(len(config), self.totalIndices):
        config.append(0)
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
        self.indexMax.append(self.indicesSized[self.indicesMapped[ \
            len(self.indicesMapped)-1]][3])

    # max num elements in each tensor
    self.maxNumElements = [ 1, 1, 1 ]
    for i in range(0, problemType["NumIndicesC"]):
      self.maxNumElements[0] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsA"]:
      self.maxNumElements[1] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsB"]:
      self.maxNumElements[2] *= self.indexMax[i]

    self.totalProblemSizes = 1
    self.numProblemSizes = [] # per index
    self.problemSizeToIndex = []
    self.problemIndexToSize = []
    sizedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      self.problemSizeToIndex.append({})
      self.problemIndexToSize.append({})
      if self.indexIsSized[i]:
        self.numProblemSizes.append(0)
        index = self.indicesSized[sizedIdx]
        sizedIdx += 1
        currentSize = index[0]
        currentIncrement = index[1]
        while currentSize <= index[3]:
          currentSize += currentIncrement
          currentIncrement += index[2]
          self.numProblemSizes[i] += 1
      else:
        self.numProblemSizes.append(1)
      self.totalProblemSizes *= self.numProblemSizes[i]

    ########################################
    # enumerate problem sizes
    currentSizedIndexSizes = []
    currentSizedIndexIncrements = []
    for i in range(0, len(self.indicesSized)):
      currentSizedIndexSizes.append(self.indicesSized[i][0])
      currentSizedIndexIncrements.append(self.indicesSized[i][1])

    # iterate over all problem sizes
    self.problemSizes = []
    moreProblemSizes = True
    problemIdx = 0
    problemSize = [0]*self.totalIndices
    while moreProblemSizes:
      #/ convert current sized and mapped indices to full sizes
      currentSizedIdx = 0
      currentMappedIdx = 0
      for i in range(0, self.totalIndices):
        if self.indexIsSized[i]:
          problemSize[i] = currentSizedIndexSizes[currentSizedIdx]
          currentSizedIdx+=1
        else:
          problemSize[i] = problemSize[self.indicesMapped[currentMappedIdx]]
          currentMappedIdx+=1
      self.problemSizes.append(tuple(problemSize))

      #/ increment sizes for next benchmark
      currentSizedIndexSizes[0] += currentSizedIndexIncrements[0]
      currentSizedIndexIncrements[0] += self.indicesSized[0][2]
      for i in range(1, len(self.indicesSized)+1):
        # if prior index past max, reset to min and increment next index
        if currentSizedIndexSizes[i-1] > self.indicesSized[i-1][3]:
          #/ reset prior index
          currentSizedIndexSizes[i-1] = self.indicesSized[i-1][0]
          currentSizedIndexIncrements[i-1] = self.indicesSized[i-1][1]
          # increment next index
          if i >= len(self.indicesSized):
            moreProblemSizes = False
          else:
            currentSizedIndexSizes[i] += currentSizedIndexIncrements[i]
            currentSizedIndexIncrements[i] += self.indicesSized[i][2]

      problemIdx+=1

  ########################################
  # YAML format
  def __str__(self):
    state = "[ "
    sizedIdx = 0
    mappedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      if self.indexIsSized[i]:
        indices = self.indicesSized[sizedIdx]
        state += "[ %u, %u, %u, %u ]" \
            % (indices[0], indices[1], indices[2], indices[3])
        sizedIdx += 1
      else:
        indices = self.indicesSized[self.indicesMapped[mappedIdx]]
        state += str(self.indicesMapped[mappedIdx])
        mappedIdx += 1
      if i < len(self.indexIsSized)-1:
        state += ", "
    state += " ]"
    return state

################################################################################
# ProblemSizes
################################################################################
class ProblemSizes:

  ########################################
  def __init__(self, problemType, config):
    self.problemType = problemType
    self.ranges = []
    self.exacts = []
    for dictionary in config:
      for sizeTypeKey in dictionary:
        if sizeTypeKey == "Range":
          psr = ProblemSizeRange(problemType, dictionary[sizeTypeKey])
          self.ranges.append( psr )
        elif sizeTypeKey == "Exact":
          e = dictionary[sizeTypeKey]
          if len(e) != problemType["TotalIndices"]:
            printExit("ExactSize %s doesn't match indices of ProblemType %s" \
                % (e, problemType) )
          else:
            self.exacts.append(tuple(e))
        else:
          printExit("ProblemSize Type %s not supported"%sizeTypeKey)

    self.sizes = set()
    for sizeRange in self.ranges:
      self.sizes.update(sizeRange.problemSizes)
    self.sizes.update(self.exacts)
    self.sizes = sorted( list( self.sizes ) )
    self.totalProblemSizes = len(self.sizes)

    # max sizes
    self.maxC = 0
    self.maxA = 0
    self.maxB = 0
    for problemSize in self.sizes:
      sizeC = 1
      sizeA = 1
      sizeB = 1
      for i in range(0, problemType["NumIndicesC"]):
        sizeC *= problemSize[i]
      for i in self.problemType["IndexAssignmentsA"]:
        sizeA *= problemSize[i]
      for i in self.problemType["IndexAssignmentsB"]:
        sizeB *= problemSize[i]
      self.maxC = max(self.maxC, sizeC)
      self.maxA = max(self.maxA, sizeA)
      self.maxB = max(self.maxB, sizeB)



  def __str__(self):
    s = "ProblemSizes\n"
    for sizeRange in self.ranges:
      s += "  %s" % sizeRange
    return s























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
      assignParameterWithDefault(self.state, key, config, defaultSolution)

    # assign parameters without defaults
    for key in config:
      if key != "ProblemType" and key not in self.state:
        self.state[key] = config[key]
    self["Valid"] = True
    self["AssignedProblemIndependentDerivedParameters"] = False
    self["AssignedDerivedParameters"] = False
    Solution.assignDerivedParameters(self.state)

  ########################################
  # get a list of kernel parameters for this solution
  def getKernels(self):
    kernel = deepcopy(self.state)
    kernel.update({"Kernel": True})
    kernels = []
    kernels.append(kernel)
    return kernels

  ########################################
  # get a list of kernel parameters for this solution
  def getKernelsBetaOnly(self):
    kernels = []
    if self["GlobalSplitU"] < 2:
      return kernels
    betas = [False]
    if self["ProblemType"]["UseBeta"]:
      betas.append(True)
    for beta in betas:
      kernel = {}
      kernel["ProblemType"] = {}
      kernel["ProblemType"]["UseBeta"] = beta
      kernel["ProblemType"]["DataType"] = self["ProblemType"]["DataType"]
      kernel["ProblemType"]["Index0"] = self["ProblemType"]["Index0"]
      kernel["ProblemType"]["Index1"] = self["ProblemType"]["Index1"]
      kernel["ProblemType"]["UseInitialStrides"] = \
          self["ProblemType"]["UseInitialStrides"]
      kernel["ProblemType"]["NumIndicesC"] = self["ProblemType"]["NumIndicesC"]
      kernels.append(kernel)
    return kernels


  ########################################
  # assign tile sizes
  @staticmethod
  def assignProblemIndependentDerivedParameters(state):
    if "AssignedProblemIndependentDerivedParameters" in state:
      if state["AssignedProblemIndependentDerivedParameters"]:
        return
    state["AssignedProblemIndependentDerivedParameters"] = False
    if "Valid" not in state:
      state["Valid"] = True

    state["SubGroup0"] = state["WorkGroup"][0]
    state["SubGroup1"] = state["WorkGroup"][1]
    state["LocalSplitU"] = state["WorkGroup"][2]
    state["NumThreads"] = state["SubGroup0"] * state["SubGroup1"] * state["LocalSplitU"]

    state["ThreadTile0"] = state["ThreadTile"][0]
    state["ThreadTile1"] = state["ThreadTile"][1]

    # macro tile sizes
    if "SubGroup0" in state and "ThreadTile0" in state:
      state["MacroTile0"] = state["SubGroup0"]*state["ThreadTile0"]
    if "SubGroup1" in state and "ThreadTile1" in state:
      state["MacroTile1"] = state["SubGroup1"]*state["ThreadTile1"]
    if "LocalSplitU" in state and "DepthU" in state:
      state["LoopUnroll"] = state["DepthU"] / state["LocalSplitU"]
    if state["LoopUnroll"] * state["LocalSplitU"] != state["DepthU"]:
        state["Valid"] = False
    if "MacroTile" in state:
      if state["MacroTile0"] != state["MacroTile"][0] \
          or state["MacroTile1"] != state["MacroTile"][1]:
        state["Valid"] = False


    if state["Valid"] and "MacroTileShapeMax" in state \
        and "MacroTileShapeMin" in state:
      macroTileShape = max(state["MacroTile0"]/state["MacroTile1"], \
          state["MacroTile1"]/state["MacroTile0"])
      if macroTileShape > state["MacroTileShapeMax"] \
          or macroTileShape < state["MacroTileShapeMin"]:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("rejecting MacroTile Shape %u:%u for Min:Max %u:%u" \
              % (state["MacroTile0"], state["MacroTile1"], \
              state["MacroTileShapeMin"], state["MacroTileShapeMax"]))
        state["Valid"] = False

    if state["WorkGroupMappingType"] == "Z":
      if abs(state["WorkGroupMapping"]) > 2:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("WorkGroupMappingType=Z only supports WorkGroupMapping=1, 2")
        state["Valid"] = False

    # done
    state["AssignedProblemIndependentDerivedParameters"] = True


  ########################################
  # assign all derived parameters
  @staticmethod
  def assignDerivedParameters(state):
    Solution.assignProblemIndependentDerivedParameters(state)
    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    ProblemType.assignDerivedParameters(state["ProblemType"])
    if not state["Valid"]:
      return

    # VectorWidth
    if state["VectorWidth"] < 1:
      state["VectorWidth"] = 4 / state["ProblemType"]["DataType"].numRegisters()
      while state["ThreadTile0"] % state["VectorWidth"] != 0 \
          or state["ThreadTile1"] % state["VectorWidth"] != 0:
        state["VectorWidth"] /= 2
    # TT0,1 both must be multiples of VW, b/c of rC, rA, rB
    if state["ThreadTile0"] % state["VectorWidth"] != 0 \
        or state["ThreadTile1"] % state["VectorWidth"] != 0:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("ThreadTile0 %u or ThreadTile1 %u not a multiple of VectorWidth %u" \
            % (state["ThreadTile0"], state["ThreadTile1"], \
            state["VectorWidth"]))
      state["Valid"] = False

    # LocalSplitU too large?
    numElementsPerWorkGroup = state["MacroTile0"]*state["MacroTile1"]
    state["NumVectorsPerThread"] = numElementsPerWorkGroup / \
        state["NumThreads"] / state["VectorWidth"]
    if state["NumVectorsPerThread"] * state["NumThreads"] \
        * state["VectorWidth"] != numElementsPerWorkGroup:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("LocalSplitU %u too large; less than 1 vector per thread" \
            % (state["LocalSplitU"]))
      state["Valid"] = False
      return

    # LoopUnroll too small
    if state["LoopUnroll"] < 2:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("LoopUnroll %u is less than 2" \
            % (state["LoopUnroll"]))
      state["Valid"] = False

    # LocalSplitU but can't NumThreads%MacroTile doesn't support sideways load
    if state["LocalSplitU"] > 1:
      if state["NumThreads"] % state["MacroTile0"] != 0:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("LocalSplitU but NumThreads=%u not divisible by MT0=%u for sideways load" \
              % (state["NumThreads"], state["MacroTile0"]))
        state["Valid"] = False
        return

    # GlobalSplitU doesn't work with
    if state["GlobalSplitU"] > 1:
      if not state["GlobalSplitUSummationAssignmentRoundRobin"] \
          and state["LoopTail"]:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("GlobalSplitU and LoopTail require SummationAssignmentRoundRobin=True since strongly breaks Tensile kernel architecture")
        state["Valid"] = False
      if not state["ProblemType"]["DataType"].isSingle():
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("GlobalSplitU only compatible with single precision")
        state["Valid"] = False


    # how many elements to load
    if state["ProblemType"]["TLUA"]:
      totalElementsCoalescedA = state["MacroTile0"]
      totalElementsPerpA = state["DepthU"]
    else:
      totalElementsCoalescedA = state["DepthU"]
      totalElementsPerpA = state["MacroTile0"]

    if state["ProblemType"]["TLUB"]:
      totalElementsCoalescedB = state["MacroTile1"]
      totalElementsPerpB = state["DepthU"]
    else:
      totalElementsCoalescedB = state["DepthU"]
      totalElementsPerpB = state["MacroTile1"]

    totalElementsA = totalElementsCoalescedA * totalElementsPerpA
    totalElementsB = totalElementsCoalescedB * totalElementsPerpB

    # convert elements to vectors based on VectorWidth
    totalVectorsCoalescedA = totalElementsCoalescedA / state["VectorWidth"]
    totalVectorsCoalescedB = totalElementsCoalescedB / state["VectorWidth"]
    totalVectorsA = totalElementsA / state["VectorWidth"]
    totalVectorsB = totalElementsB / state["VectorWidth"]
    
    print "totalVectorsA", totalVectorsA
    print "totalVectorsB", totalVectorsB

    if totalVectorsA < state["NumThreads"]:
      if state["NumThreads"] % totalVectorsA == 0:
        state["PVA"] = state["NumThreads"] / totalVectorsA # partial vector
      else:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("NumThreads %u %% totalVectorsA %u != 0" \
              % (state["NumThreads"], totalVectorsA))
        state["Valid"] = False
    else:
      if totalVectorsA % state["NumThreads"] == 0:
        state["PVA"] = 1 # partial vector
      else:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("totalVectorsA %u %% NumThreads %u != 0" \
              % (totalVectorsA, state["NumThreads"]))
        state["Valid"] = False
    state["GlobalLoadVectorWidthA"] = state["VectorWidth"] / state["PVA"]
    state["NumLoadsA"] = totalVectorsA * state["PVA"] / state["NumThreads"]

    if totalVectorsB < state["NumThreads"]:
      if state["NumThreads"] % totalVectorsB == 0:
        state["PVB"] = state["NumThreads"] / totalVectorsB # partial vector
      else:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("NumThreads %u %% totalVectorsB %u != 0" \
              % (state["NumThreads"], totalVectorsB))
        state["Valid"] = False
    else:
      if totalVectorsB % state["NumThreads"] == 0:
        state["PVB"] = 1 # partial vector
      else:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("totalVectorsB %u %% NumThreads %u != 0" \
              % (totalVectorsB, state["NumThreads"]))
        state["Valid"] = False
    state["GlobalLoadVectorWidthB"] = state["VectorWidth"] / state["PVB"]
    state["NumLoadsB"] = totalVectorsB * state["PVB"] / state["NumThreads"]

    print "pva", state["PVA"]
    print "pvb", state["PVB"]

    """
    # how many load instructions
    if totalVectorsA % state["NumThreads"] != 0 or totalVectorsA < state["NumThreads"]:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("totalVectorsA %u %% NumThreads %u != 0" \
            % (totalVectorsA, state["NumThreads"]))
      state["Valid"] = False
      return
    else:
      state["NumLoadsA"] = totalVectorsA / state["NumThreads"]

    if totalVectorsB % state["NumThreads"] != 0 or totalVectorsB < state["NumThreads"]:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("totalVectorsB %u %% NumThreads %u != 0" \
            % (totalVectorsB, state["NumThreads"]))
      state["Valid"] = False
      return
    else:
      state["NumLoadsB"] = totalVectorsB / state["NumThreads"]
    """

    print "NumLoadsA", state["NumLoadsA"]
    # nlca = 1
    if state["NumLoadsCoalescedA"] == 1:
      foundValid = False
      for nlca in range(1, state["NumLoadsA"]+1):
        nlpa = state["NumLoadsA"] / nlca
        print nlca, nlpa
        if state["NumLoadsA"] % nlca == 0 \
            and totalVectorsCoalescedA % nlca == 0 \
            and totalElementsPerpA % nlpa == 0:
          state["NumLoadsCoalescedA"] = nlca
          state["NumLoadsPerpendicularA"] = nlpa
          foundValid = True
          break
      if not foundValid:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("No NumLoadsCoalescedA=1 found")
        state["Valid"] = False
        return

    # nlca = -1
    elif state["NumLoadsCoalescedA"] == -1:
      foundValid = False
      for nlca in range(state["NumLoadsA"], 0, -1):
        nlpa = state["NumLoadsA"] / nlca
        if state["NumLoadsA"] % nlca == 0 \
            and totalVectorsCoalescedA % nlca == 0 \
            and totalElementsPerpA % nlpa == 0:
          state["NumLoadsCoalescedA"] = nlca
          state["NumLoadsPerpendicularA"] = nlpa
          foundValid = True
          break
      if not foundValid:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("No NumLoadsCoalescedA=-1 found")
        state["Valid"] = False
        return

    # nlca = other
    else:
      if state["NumLoadsCoalescedA"] > state["NumLoadsA"]:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("NLCA > NLA")
        state["Valid"] = False
        return
      state["NumLoadsPerpendicularA"] = state["NumLoadsA"] \
          / state["NumLoadsCoalescedA"]

      if state["NumLoadsA"] % state["NumLoadsCoalescedA"] != 0:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("numLoadsA %u %% numLoadsParaA %u != 0" \
              % (state["NumLoadsA"], state["NumLoadsCoalescedA"]))
        state["Valid"] = False
      if totalVectorsCoalescedA % state["NumLoadsCoalescedA"] != 0:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("totalVectorsCoalescedA %u %% numLoadsParaA %u != 0" \
              % (totalVectorsCoalescedA, state["NumLoadsCoalescedA"]))
        state["Valid"] = False
        return
      if totalElementsPerpA % state["NumLoadsPerpendicularA"] != 0:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("totalElementsPerpA %u %% numLoadsPerpA %u != 0" \
              % (totalElementsPerpA, state["NumLoadsPerpendicularA"]))
        state["Valid"] = False
        return

    # nlcb = 1
    if state["NumLoadsCoalescedB"] == 1:
      foundValid = False
      for nlcb in range(1, state["NumLoadsB"]+1):
        nlpb = state["NumLoadsB"] / nlcb
        # pre-filter for VW*NLCB
        if state["VectorWidth"] > 1:
          if state["ProblemType"]["TLUB"]:
            if nlcb * state["VectorWidth"] \
                > state["ThreadTile1"]:
              continue
          else:
            if nlpb * state["VectorWidth"] \
                > state["ThreadTile1"]:
              continue
        if state["NumLoadsB"] % nlcb == 0 \
            and totalVectorsCoalescedB % nlcb == 0 \
            and totalElementsPerpB % nlpb == 0:
          state["NumLoadsCoalescedB"] = nlcb
          state["NumLoadsPerpendicularB"] = nlpb
          foundValid = True
          break
      if not foundValid:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("No NumLoadsCoalescedB=1 found")
        state["Valid"] = False
        return

    # nlcb = -1
    elif state["NumLoadsCoalescedB"] == -1:
      foundValid = False
      for nlcb in range(state["NumLoadsB"], 0, -1):
        nlpb = state["NumLoadsB"] / nlcb
        # pre-filter for VW*NLCB
        if state["VectorWidth"] > 1:
          if state["ProblemType"]["TLUB"]:
            if nlcb * state["VectorWidth"] \
                > state["ThreadTile1"]:
              continue
          else:
            if nlpb * state["VectorWidth"] \
                > state["ThreadTile1"]:
              continue
        if state["NumLoadsB"] % nlcb == 0 \
            and totalVectorsCoalescedB % nlcb == 0 \
            and totalElementsPerpB % nlpb == 0:
          state["NumLoadsCoalescedB"] = nlcb
          state["NumLoadsPerpendicularB"] = nlpb
          foundValid = True
          break
      if not foundValid:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("No NumLoadsCoalescedB=-1 found")
        state["Valid"] = False
        return

    # nlcb = other
    else:
      if state["NumLoadsCoalescedB"] > state["NumLoadsB"]:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("NLCB > NLB")
        state["Valid"] = False
        return

      state["NumLoadsPerpendicularB"] = state["NumLoadsB"] \
          / state["NumLoadsCoalescedB"]

      if state["NumLoadsB"] % state["NumLoadsCoalescedB"] != 0:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("numLoadsB %u %% numLoadsParaB %u != 0" \
            % (state["NumLoadsB"], state["NumLoadsCoalescedB"]))
        state["Valid"] = False
        return
      if totalVectorsCoalescedB % state["NumLoadsCoalescedB"] != 0:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("totalVectorsCoalescedB %u %% numLoadsParaB %u != 0" \
            % (totalVectorsCoalescedB, state["NumLoadsCoalescedB"]))
        state["Valid"] = False
        return
      if totalElementsPerpB % state["NumLoadsPerpendicularB"] != 0:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("totalElementsPerpB %u %% numLoadsPerpB %u != 0" \
            % (totalElementsPerpB, state["NumLoadsPerpendicularB"]))
        state["Valid"] = False
        return

    # lds buffer size for A, B
    ldsAlign = 64 / state["ProblemType"]["DataType"].numRegisters()
    ldsNumElementsA = state["DepthU"]*(state["MacroTile0"]+state["LdsPad"])
    ldsNumElementsAlignedA = ((ldsNumElementsA+ldsAlign-1)/ldsAlign)*ldsAlign
    ldsNumElementsB = state["DepthU"]*(state["MacroTile1"]+state["LdsPad"])
    ldsNumElementsAlignedB = ((ldsNumElementsB+ldsAlign-1)/ldsAlign)*ldsAlign
    # todo, can the alignment be a power of 2?
    if state["PrefetchGlobalRead"]:
      state["LdsNumElementsAlignedA"] = ldsNumElementsAlignedA
      state["LdsNumElementsAlignedB"] = ldsNumElementsAlignedB
      state["LdsOffsetA"] = 0
      state["LdsOffsetB"] = state["LdsOffsetA"] \
        + state["LdsNumElementsAlignedA"]

      offsetBlk = state["LdsOffsetB"] + state["LdsNumElementsAlignedB"]
      offsetBlk = int(2**(ceil(log(offsetBlk, 2))))

      state["LdsOffsetA_Blk"] = offsetBlk
      state["LdsOffsetB_Blk"] = state["LdsOffsetA_Blk"] \
        + state["LdsNumElementsAlignedA"]
      ldsNumElementsAB = state["LdsOffsetB_Blk"]+ ldsNumElementsB
    else:
      state["LdsOffsetB"] = ldsNumElementsAlignedA
      ldsNumElementsAB = ldsNumElementsAlignedA + ldsNumElementsB

    # lds buffer size for reduction
    ldsNumElementsReduction = state["LocalSplitU"]*state["MacroTile0"]*state["MacroTile1"] if state["LocalSplitU"] > 1 else 0

    # lds max occupancy
    ldsSizeOccupancy = globalParameters["DeviceLDS"] / state["MaxOccupancy"]
    ldsNumElementsOccupancy = ldsSizeOccupancy / state["ProblemType"]["DataType"].numBytes()

    # lds size is the greater of the two
    ldsNumElements = max(ldsNumElementsAB, ldsNumElementsReduction, ldsNumElementsOccupancy)
    state["LdsNumElements"] = ldsNumElements
    ldsSize = ldsNumElements * state["ProblemType"]["DataType"].numBytes()
    if ldsSize > globalParameters["MaxLDS"]:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("Kernel Uses %u > %u bytes of LDS" % ( ldsSize, globalParameters["MaxLDS"]))
      state["Valid"] = False
      return

    # compiler trips over these configurations
    """
    if globalParameters["KernelLanguage"] == "HIP" \
        and state["VectorWidth"] == 2 \
        and state["ProblemType"]["TLUA"] \
        and state["ProblemType"]["TLUB"] \
        and state["GlobalReadCoalesceVectorA"] \
        and state["GlobalReadCoalesceVectorB"]:
      if state["ThreadTile0"] == 4 and state["ThreadTile1"] == 8 \
          or state["ThreadTile0"] == 6 and state["ThreadTile1"] == 2\
          or state["ThreadTile0"] == 6 and state["ThreadTile1"] == 8:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("compiler bug")
        state["Valid"] = False

    if globalParameters["KernelLanguage"] == "HIP" \
        and state["VectorWidth"] == 4 \
        and state["ProblemType"]["TLUA"] \
        and state["ProblemType"]["TLUB"] \
        and state["GlobalReadCoalesceVectorA"] \
        and state["GlobalReadCoalesceVectorB"] \
        and state["ThreadTile0"] == 8 and state["ThreadTile1"] == 8:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("compiler bug")
      state["Valid"] = False

    if globalParameters["KernelLanguage"] == "OCL" \
        and state["VectorWidth"] == 2 \
        and state["ProblemType"]["TLUA"] \
        and state["ProblemType"]["TLUB"] \
        and state["GlobalReadCoalesceVectorA"] \
        and state["GlobalReadCoalesceVectorB"]:
      if state["ThreadTile0"] == 4 and state["ThreadTile1"] == 8 \
          or state["ThreadTile0"] == 8 and state["ThreadTile1"] == 4\
          or state["ThreadTile0"] == 8 and state["ThreadTile1"] == 4:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("compiler bug")
        state["Valid"] = False
    """



    state["AssignedDerivedParameters"] = True

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
        if key in validParameters.keys():
          requiredParameters[key] = False
    else:
      for key in keys:
        required = False
        if key in validParameters.keys():
          for i in range(1, len(objs)):
            if objs[0][key] != objs[i][key]:
              required = True
              break
        if required:
          requiredParameters[key] = True
        else:
          requiredParameters[key] = False
    requiredParameters["ProblemType"] = False # always prepended
    requiredParameters["MacroTile0"] = False # always prepended
    requiredParameters["MacroTile1"] = False # always prepended
    requiredParameters["DepthU"] = False # always prepended
    requiredParameters["Kernel"] = True # distinguish kernels from solutions
                                        # for single-source compilation
    return requiredParameters

  ########################################
  @ staticmethod
  def getNameFull(state):
    requiredParameters = {}
    for key in state:
      if key in validParameters.keys():
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
    if "MacroTile0" in state \
        and "MacroTile1" in state \
        and "DepthU" in state:
      name += "%s%03ux%03ux%02u_" \
          % ( Solution.getParameterNameAbbreviation("MacroTile"), \
          state["MacroTile0"], state["MacroTile1"], state["DepthU"] )
    for key in sorted(state.keys()):
      if key in requiredParameters:
        if requiredParameters[key]:
          if not first:
            name += "_"
          else:
            first = False
          name += "%s%s" % ( Solution.getParameterNameAbbreviation(key), \
              Solution.getParameterValueAbbreviation(state[key]) )
    return name

  ########################################
  # create a dictionary of lists of parameter values
  @staticmethod
  def getSerialNaming(objs):
    data = {}
    for objIdx in range(0, len(objs)):
      obj = objs[objIdx]
      for paramName in sorted(obj.keys()):
        if paramName in validParameters.keys():
          paramValue = obj[paramName]
          if paramName in data:
            if paramValue not in data[paramName]:
              data[paramName].append(paramValue)
          else:
            data[paramName] = [ paramValue ]
    maxObjs = 1
    for paramName in data:
      data[paramName] = sorted(data[paramName])
      maxObjs *= len(data[paramName])
    numDigits = len(str(maxObjs))
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
      if paramName in validParameters.keys():
        paramValue = state[paramName]
        paramData = data[paramName]
        paramNameMultiplier = len(paramData)
        if paramValue in paramData:
          paramValueIdx = paramData.index(paramValue)
        serial += paramValueIdx * multiplier
        multiplier *= paramNameMultiplier
    name = "%s%0*u" % ("S" if isinstance(state, Solution) else "K", \
        numDigits, serial)
    return name


  ########################################
  @ staticmethod
  def getParametersIndented(state, indent):
    s = ""
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
      printExit("Parameter \"%s\" is new object type" % str(value) )
      return str(value)

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
    return self.state
  def __hash__(self):
    return hash(str(self))
    #return hash(self.getAttributes())
  def __eq__(self, other):
    #return isinstance(other, Solution) and self.getAttributes() == other.getAttributes()
    return isinstance(other, Solution) and str(self) == str(other)
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result


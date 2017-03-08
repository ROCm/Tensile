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


from Common import globalParameters, defaultProblemType, assignParameterWithDefault, printExit, assignParameterRequired, defaultSolution, derivedParameters, print1, print2
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
    if backend == "HIP":
      if self.value == self.complexSingle:
        return "make_float2(0.f, 0.f)"
      if self.value == self.complexDouble:
        return "make_float2(0.0, 0.0)"

    zeroString = "(%s)(" % self.toDevice(backend)
    if self.value == self.single or self.value == self.half:
      zeroString += "0.f"
    elif self.value == self.double:
      zeroString += "0.0"
    elif self.value == self.complexSingle:
      zeroString += "0.f, 0.f"
    elif self.value == self.complexDouble:
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

    #unrollDimStrideGreaterThanTileDimStrideA = TLUA
    #unrollDimStrideLessThanTileDimStrideB    = !TLUB
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
  # assign tile sizes
  @staticmethod
  def assignProblemIndependentDerivedParameters(state):
    if "AssignedProblemIndependentDerivedParameters" in state:
      if state["AssignedProblemIndependentDerivedParameters"]:
        return
    state["AssignedProblemIndependentDerivedParameters"] = False

    (subGroup0, subGroup1, threadTile0, threadTile1) \
        = Solution.tileSizes(state["NumThreads"], state["SplitU"], \
        state["GroupShape"], state["ThreadTileNumElements"], state["ThreadTileShape"])

    state["SubGroup0"] = subGroup0
    state["SubGroup1"] = subGroup1
    state["ThreadTile0"] = threadTile0
    state["ThreadTile1"] = threadTile1
    if state["SubGroup0"]*state["SubGroup1"] \
        != state["NumThreads"]/state["SplitU"]:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("GroupSize %u * %u != %u / %u" % (state["SubGroup0"], state["SubGroup1"], state["NumThreads"], state["SplitU"]))
      state["Valid"] = False
    #print "Group:", state["SubGroup0"], state["SubGroup1"]

    if state["ThreadTile0"]*state["ThreadTile1"] != state["ThreadTileNumElements"]:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("ThreadTile %u * %u != %u" % (state["ThreadTile0"], state["ThreadTile1"], state["ThreadTileNumElements"]))
      state["Valid"] = False
    #print "ThreadTile:", state["ThreadTile0"], state["ThreadTile1"]

    # macro tile sizes
    if "SubGroup0" in state and "ThreadTile0" in state:
      state["MacroTile0"] = state["SubGroup0"]*state["ThreadTile0"]
    if "SubGroup1" in state and "ThreadTile1" in state:
      state["MacroTile1"] = state["SubGroup1"]*state["ThreadTile1"]
    if "SplitU" in state and "LoopUnroll" in state:
      state["DepthU"] = state["SplitU"] * state["LoopUnroll"]

    # tile shape
    if state["MacroTile0"]/state["MacroTile1"] > globalParameters["MaxMacroTileRatio"] \
        or state["MacroTile1"]/state["MacroTile0"] > globalParameters["MaxMacroTileRatio"]:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("rejecting ratio %u : %u" % (state["MacroTile0"], state["MacroTile1"]))
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

    # SplitU too large?
    numElementsPerWorkGroup = state["MacroTile0"]*state["MacroTile1"]
    state["NumElementsPerThread"] = numElementsPerWorkGroup / state["NumThreads"]
    if state["NumElementsPerThread"] * state["NumThreads"] != numElementsPerWorkGroup:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("SplitU %u too large; less than 1 element per thread" \
            % (state["SplitU"]))
      state["Valid"] = False
      return

    # SplitU but can't NumThreads%MacroTile doesn't support sideways load
    if state["NumThreads"] % state["MacroTile0"] != 0:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("SplitU but NumThreads=%u not divisible by MT0=%u for sideways load" \
            % (state["NumThreads"], state["MacroTile0"]))
      state["Valid"] = False
      return

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

    # how many load instructions
    if totalElementsA % state["NumThreads"] != 0:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("totalElementsA %u %% NumThreads %u != 0" \
            % (totalElementsA, state["NumThreads"]))
      state["Valid"] = False
      return
    else:
      state["NumLoadsA"] = totalElementsA / state["NumThreads"]

    if totalElementsB % state["NumThreads"] != 0:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("totalElementsB %u %% NumThreads %u != 0" \
            % (totalElementsB, state["NumThreads"]))
      state["Valid"] = False
      return
    else:
      state["NumLoadsB"] = totalElementsB / state["NumThreads"]
    #print "NumLoads:", state["NumLoadsA"], state["NumLoadsB"]

    # nlca = 1
    if state["NumLoadsCoalescedA"] == 1:
      foundValid = False
      for nlca in range(1, state["NumLoadsA"]+1):
        nlpa = state["NumLoadsA"] / nlca
        if state["NumLoadsA"] % nlca == 0 \
            and totalElementsCoalescedA % nlca == 0 \
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
            and totalElementsCoalescedA % nlca == 0 \
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
      if totalElementsCoalescedA % state["NumLoadsCoalescedA"] != 0:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("totalElementsCoalescedA %u %% numLoadsParaA %u != 0" \
              % (totalElementsCoalescedA, state["NumLoadsCoalescedA"]))
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
      for nlca in range(1, state["NumLoadsB"]+1):
        nlpa = state["NumLoadsB"] / nlca
        if state["NumLoadsB"] % nlca == 0 \
            and totalElementsCoalescedB % nlca == 0 \
            and totalElementsPerpB % nlpa == 0:
          state["NumLoadsCoalescedB"] = nlca
          state["NumLoadsPerpendicularB"] = nlpa
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
      for nlca in range(state["NumLoadsB"], 0, -1):
        nlpa = state["NumLoadsB"] / nlca
        if state["NumLoadsB"] % nlca == 0 \
            and totalElementsCoalescedB % nlca == 0 \
            and totalElementsPerpB % nlpa == 0:
          state["NumLoadsCoalescedB"] = nlca
          state["NumLoadsPerpendicularB"] = nlpa
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
      if totalElementsCoalescedB % state["NumLoadsCoalescedB"] != 0:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("totalElementsCoalescedB %u %% numLoadsParaB %u != 0" \
            % (totalElementsCoalescedB, state["NumLoadsCoalescedB"]))
        state["Valid"] = False
        return
      if totalElementsPerpB % state["NumLoadsPerpendicularB"] != 0:
        if globalParameters["PrintSolutionRejectionReason"]:
          print1("totalElementsPerpB %u %% numLoadsPerpB %u != 0" \
            % (totalElementsPerpB, state["NumLoadsPerpendicularB"]))
        state["Valid"] = False
        return

    # lds buffer size
    ldsAlign = 256 / state["ProblemType"]["DataType"].numRegisters()
    ldsNumElementsA = state["DepthU"]*(state["MacroTile0"]+state["LdsPad"])
    ldsNumElementsB = state["DepthU"]*(state["MacroTile1"]+state["LdsPad"])
    ldsNumElementsAlignedA = ((ldsNumElementsA+ldsAlign-1)/ldsAlign)*ldsAlign
    ldsNumElementsAlignedB = ((ldsNumElementsB+ldsAlign-1)/ldsAlign)*ldsAlign
    ldsNumElementsReduction = 0 if (state["SplitU"] == 1) \
        else (state["MacroTile0"]*state["MacroTile1"])
    ldsNumElements = max(ldsNumElementsAlignedA+ldsNumElementsB, ldsNumElementsReduction)
    state["LdsNumElements"] = ldsNumElements
    state["LdsOffsetB"] = ldsNumElementsAlignedA
    ldsSize = ldsNumElements * state["ProblemType"]["DataType"].numBytes()

    if ldsSize > globalParameters["MaxLDS"]:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("Kernel Uses %u > %u bytes of LDS" % ( ldsSize, globalParameters["MaxLDS"]))
      state["Valid"] = False
      return

    # Compiler may be causing incorrect spills on ROCm1.4 from DT on 2/21/17
    if globalParameters["Backend"] == "HIP":
      if state["ProblemType"]["DataType"].value == DataType.single:
        if state["MacroTile0"] == 128 or state["MacroTile1"] == 128:
          if state["NumLoadsCoalescedA"] != 1 and state["NumLoadsCoalescedB"] != 8:
            state["Valid"] = False
            #return
      elif state["ProblemType"]["DataType"].value == DataType.double:
        if globalParameters["Backend"] == "HIP":
          if state["MacroTile0"] >= 64 or state["MacroTile1"] >= 64:
            state["Valid"] = False
            #return
    state["AssignedDerivedParameters"] = True

    #print Solution.getNameFull(state)

# validation failures
# Cijk_Ailk_Bjlk_SB_DU16_LU16_MT064_MT164_NLA16_NLB16_NLCA02_NLCB01_NLPA08_NLPB16_TT008_TT108_TTE08_WG008_WG108_WGE08
# Cijk_Ailk_Bjlk_SB_DU16_LU16_MT064_MT164_NLA16_NLB16_NLCA04_NLCB02_NLPA04_NLPB08_TT008_TT108_TTE08_WG008_WG108_WGE08
# Cijk_Ailk_Bjlk_SB_DU16_LU16_MT064_MT164_NLA16_NLB16_NLCA02_NLCB04_NLPA08_NLPB04_TT008_TT108_TTE08_WG008_WG108_WGE08

# Cijk_Ailk_Bjlk_DB_DU16_LU16_MT064_MT164_NLA16_NLB16_NLCA04_NLCB01_NLPA04_NLPB16_TT008_TT108_TTE08_WG008_WG108_WGE08
# Cijk_Ailk_Bjlk_DB_DU08_LU08_MT064_MT164_NLA08_NLB08_NLCA01_NLCB01_NLPA08_NLPB08_TT008_TT108_TTE08_WG008_WG108_WGE08
# Cijk_Ailk_Bjlk_DB_DU08_LU08_MT064_MT164_NLA08_NLB08_NLCA08_NLCB01_NLPA01_NLPB08_TT008_TT108_TTE08_WG008_WG108_WGE08
# Cijk_Ailk_Bjlk_DB_DU08_LU08_MT064_MT164_NLA08_NLB08_NLCA08_NLCB08_NLPA01_NLPB01_TT008_TT108_TTE08_WG008_WG108_WGE08
# Cijk_Ailk_Bjlk_DB_DU16_LU16_MT064_MT164_NLA16_NLB16_NLCA08_NLCB08_NLPA02_NLPB02_TT008_TT108_TTE08_WG008_WG108_WGE08
# Cijk_Ailk_Bjlk_DB_DU08_LU08_MT064_MT164_NLA08_NLB08_NLCA01_NLCB08_NLPA08_NLPB01_TT008_TT108_TTE08_WG008_WG108_WGE08

  ########################################
  # compute tile sizes
  @staticmethod
  def tileSizes(numThreads, splitU, groupShape, \
      threadTileNumElements, threadTileShape):

    # group sizes
    subGroupSize = numThreads / splitU
    if groupShape == 0:
      subGroup0 = int(subGroupSize**0.5)
      subGroup1 = int(subGroupSize**0.5)
    elif groupShape > 0:
      subGroup0 = int((subGroupSize \
          / abs(groupShape))**0.5)
      subGroup1 = subGroup0 * abs(groupShape)
    elif groupShape < 0:
      subGroup1 = int((subGroupSize \
          / abs(groupShape))**0.5)
      subGroup0 = subGroup1 * abs(groupShape)

    # thread-tile sizes
    if threadTileShape == 0:
      threadTile0 = int(threadTileNumElements**0.5)
      threadTile1 = int(threadTileNumElements**0.5)
    elif threadTileShape > 0:
      threadTile0 = int((threadTileNumElements \
          / abs(threadTileShape))**0.5)
      threadTile1 = threadTile0 \
          * abs(threadTileShape)
    elif threadTileShape < 0:
      threadTile1 = int((threadTileNumElements \
          / abs(threadTileShape))**0.5)
      threadTile0 = threadTile1 \
          * abs(threadTileShape)

    return (subGroup0, subGroup1, threadTile0, threadTile1)

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
        if key not in derivedParameters:
          requiredParameters[key] = False
    else:
      for key in keys:
        required = False
        if key not in derivedParameters:
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
    return name

  ########################################
  # create a dictionary of lists of parameter values
  @staticmethod
  def getSerialNaming(objs):
    data = {}
    for objIdx in range(0, len(objs)):
      obj = objs[objIdx]
      for paramName in sorted(obj.keys()):
        if paramName not in derivedParameters:
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
      if paramName not in derivedParameters:
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


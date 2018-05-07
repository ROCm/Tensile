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


import sys,traceback
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
  def isNone(self):
    return self.value == self.none

  ########################################
  def numRegisters( self ):
    return self.properties[self.value][self.idxReg]
  def numBytes( self ):
    return int(self.numRegisters() * 4)
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

########################################
# Print a reject message :
def reject(state, *args):
  if globalParameters["PrintSolutionRejectionReason"]:
    sys.stdout.write("reject: ")
    for a in args:
      print(a)
    traceback.print_stack(None, 2)
  if state != None:
    state["Valid"] = False

# print a labled variable
def pvar(state, field):
  return field + "=" + str(state[field])

def roundupRatio(dividend, divisor):
  return int(ceil(float(dividend) / float(divisor)))

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
    # print index assignments
    #print2("IndicesFree:  %s" % state["IndicesFree"])
    #print2("IndicesBatch: %s" % state["IndicesBatch"])
    #print2("IndicesSum:   %s" % state["IndicesSummation"])
    state["NumIndicesFree"] = len(state["IndicesFree"])
    state["NumIndicesBatch"] = len(state["IndicesBatch"])
    state["NumIndicesSummation"] = len(state["IndicesSummation"])
    if state["NumIndicesFree"] != 2:
      printExit("Tensile can only handle 2 free indices; FreeIndices=%s."%state["IndicesFree"])

    # by default, unroll index will be the last/inner summation index
    state["IndexUnroll"] = state["IndicesSummation"][len(state["IndicesSummation"])-1]
    for i in range(0, len(state["IndexAssignmentsA"])):
      if state["IndexAssignmentsA"][i] == state["IndexUnroll"]:
        state["IndexUnrollA"] = i
        break
    for i in range(0, len(state["IndexAssignmentsB"])):
      if state["IndexAssignmentsB"][i] == state["IndexUnroll"]:
        state["IndexUnrollB"] = i
        break
    #print2("IndexUnrollA: %u" % state["IndexUnrollA"])
    #print2("IndexUnrollB: %u" % state["IndexUnrollB"])

    # assign d0, d1
    state["Index01A"] = -1
    state["Index01B"] = -1
    for i in state["IndexAssignmentsA"]:
      if i in state["IndicesFree"]:
        state["Index01A"] = i
        break
    for i in state["IndexAssignmentsB"]:
      if i in state["IndicesFree"]:
        state["Index01B"] = i
        break
    #print2("Index01A: %u" % state["Index01A"])
    #print2("Index01B: %u" % state["Index01B"])
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
    if self["HighPrecisionAccumulate"] and not self["SilentHighPrecisionAccumulate"]: name += "H"
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
    self.minStrides = None
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
        elif sizeTypeKey == "MinStride":
          e = dictionary[sizeTypeKey]
          if len(e) != problemType["TotalIndices"]:
            printExit("MinStride %s doesn't match indices of ProblemType %s" \
                % (e, problemType) )
          if self.minStrides:
            printExit("Only one MinStride command is allowed in a ProblemsSizes definition.  Previous minStrides:%s, New minstride:%s" \
                % (self.minStrides, e) )

          self.minStrides=(tuple(e))
        else:
          printExit("ProblemSize Type %s not supported"%sizeTypeKey)

    if not self.minStrides: 
      # set harmless default mins of 0
      self.minStrides = ([0]* problemType["TotalIndices"])

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
        sizeC *= max(self.minStrides[i], problemSize[i])
      for i in self.problemType["IndexAssignmentsA"]:
        sizeA *= max(self.minStrides[i], problemSize[i])
      for i in self.problemType["IndexAssignmentsB"]:
        sizeB *= max(self.minStrides[i], problemSize[i])
      self.maxC = max(self.maxC, sizeC)
      self.maxA = max(self.maxA, sizeA)
      self.maxB = max(self.maxB, sizeB)

  def __str__(self):
    s = "ProblemSizes\n"
    for sizeRange in self.ranges:
      s += "  %s" % sizeRange
    return s


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
        reject(state, "rejecting MacroTile Shape %u:%u for Min:Max %u:%u" \
            % (state["MacroTile0"], state["MacroTile1"], \
            state["MacroTileShapeMin"], state["MacroTileShapeMax"]))

    if "WorkGroupMappingType" in state:
      if state["WorkGroupMappingType"] == "Z":
        if abs(state["WorkGroupMapping"]) > 2:
          reject(state, "WorkGroupMappingType=Z only supports WorkGroupMapping=1, 2")


    # done
    state["AssignedProblemIndependentDerivedParameters"] = True

  ########################################
  # This is the "classic" algorithm which requires that each threads load the same number of bytes
  # Called with tc=A and then with tc=B
  # totalVectors is totalElements/GRVW, this is #vectors loaded by the LoadTile
  # Reduces the GlobalLoadVectorWidth if necessary if each thread has a small amount of work to do.
  # Output from this function:
  #  state[GlobalLoadVectorWidth*]
  #  state[NumLoads*] # only used in SolutionStructs, with classic alg
  @staticmethod
  def setGlobalLoadVectorWidth(state, tc, totalVectors):
    validDepthU = True
    if totalVectors < state["NumThreads"]:
      # Try to reduce size of vector so every thread has a load to do
      pv = state["NumThreads"] / totalVectors # partial vector
      if not state["FractionalLoad"]:
        if state["NumThreads"] % totalVectors != 0:
          reject(None, "NumThreads %u %% totalVectors %u != 0" \
              % (state["NumThreads"], totalVectors))
          validDepthU = False
        if pv * totalVectors != state["NumThreads"]:
          reject(None, "pv %u * totalVectors %u != NumThreads %u" \
              % (pv, totalVectors, state["NumThreads"]))
          validDepthU = False
        if state["GlobalReadVectorWidth"] % pv != 0:
          reject(None, "NumThreads %u %% totalVectors %u != 0" \
              % (state["NumThreads"], totalVectors))
          validDepthU = False
    else:
      pv = 1 # no partial vector required
      if totalVectors % state["NumThreads"] != 0:
        if not state["FractionalLoad"]:
          reject(None, "totalVectors %u %% NumThreads %u != 0" \
              % (totalVectors, state["NumThreads"]))
          validDepthU = False

    state["GlobalLoadVectorWidth%s"%tc] = state["GlobalReadVectorWidth"] / pv

    # NumLoads is NOT used on the fractional path
    # NumLoads is number of vector loads per-thread
    state["NumLoads%s"%tc] = totalVectors * pv / state["NumThreads"]

    #print "result: ", pvar(state, "GlobalLoadVectorWidth%s"%tc), \
    #        pvar(state, "NumLoads%s"%tc)

    return validDepthU

  ########################################
  # Sets the Global Read Tile dims (para, perp)
  # This information controls which threads read which addresses from global mem)
  # Output from this function:
  #   state[NumLoadsCoalescedA]
  #   state[NumLoadsPerpendicularA]
  #   state[LSCA]
  #   state[LSPA]
  @staticmethod
  def setGlobalLoadTileDimClassic(state, tc, numLoads, totalVectorsCoalesced, totalElementsPerp):
    # nlc = 1
    if state["NumLoadsCoalesced%s"%tc] == 1 :
      foundValid = False
      for nlc in range(1, state["NumLoads%s"%tc]+1):
        nlp = state["NumLoads%s"%tc] / nlc
        #print nlc, nlp
        if state["NumLoads%s"%tc] % nlc == 0 \
            and totalVectorsCoalesced % nlc == 0 \
            and totalElementsPerp % nlp == 0:
          state["NumLoadsCoalesced%s"%tc] = nlc
          state["NumLoadsPerpendicular%s"%tc] = nlp
          foundValid = True
          break
      if not foundValid:
        reject(state, "%s: No NumLoadsCoalesced=1 found"%tc)
        return False

    # nlc = -1
    elif state["NumLoadsCoalesced%s"%tc] == -1:
      foundValid = False
      for nlc in range(state["NumLoads%s"%tc], 0, -1):
        nlp = state["NumLoads%s"%tc] / nlc
        if state["NumLoads%s"%tc] % nlc == 0 \
            and totalVectorsCoalesced % nlc == 0 \
            and totalElementsPerp % nlp == 0:
          state["NumLoadsCoalesced%s"%tc] = nlc
          state["NumLoadsPerpendicular%s"%tc] = nlp
          foundValid = True
          break
      if not foundValid:
        reject(state, "%s: No NumLoadsCoalesced=-1 found"%tc)
        return False

    # nlc = other
    else:
      if state["NumLoadsCoalesced%s"%tc] > state["NumLoads%s"%tc]:
        reject(state, "%s nlc > numLoads"%tc)
        return False

      state["NumLoadsPerpendicular%s"%tc] = state["NumLoads%s"%tc] \
          / state["NumLoadsCoalesced%s"%tc]

      if state["NumLoads%s"%tc] % state["NumLoadsCoalesced%s"%tc] != 0:
        reject(state, "%s: numLoads %u %% numLoadsCoalesced %u != 0" \
            % (tc, state["NumLoads%s"%tc], state["NumLoadsCoalesced%s"%tc]))
        return False

      if totalVectorsCoalesced % state["NumLoadsCoalesced%s"%tc] != 0 :
        reject(state, "%s: totalVectorsCoalesced %u %% numLoadsPara %u != 0" \
              % (tc, totalVectorsCoalesced, state["NumLoadsCoalesced%s"%tc]))
        return False
      if totalElementsPerp % state["NumLoadsPerpendicular%s"%tc] != 0:
        reject(state, "%s: totalElementsPerp %u %% numLoadsPerp %u != 0" \
              % (tc, totalElementsPerp, state["NumLoadsPerpendicular%s"%tc]))
        return False

    if state["ProblemType"]["TLU%s"%tc]:
      state["LSC%s"%tc] = state["MacroTile%s"%tc] \
          / state["NumLoadsCoalesced%s"%tc]
      state["LSP%s"%tc] = int(ceil(float(state["DepthU"]) / state["NumLoadsPerpendicular%s"%tc]))
    else:
      state["LSC%s"%tc] = int(ceil(float(state["DepthU"]) / state["NumLoadsCoalesced%s"%tc]))
      state["LSP%s"%tc] = state["MacroTile%s"%tc] \
          / state["NumLoadsPerpendicular%s"%tc]

    return True


  ########################################
  # Sets the Global Read Tile dims (para, perp)
  # This information controls which threads read which addresses from global mem)
  # Output from this function:
  #   state[NumLoadsCoalesced*]
  #   state[NumLoadsPerpendicular*]
  #   state[LSC*]
  #   state[LSP*]
  #   state[GlobalReadVectorWidth]
  #
  # LSC and LSP define the shape of the PerLoadTile, measured in elements.
  #   LSC*LSP is the elements loaded by a single instruction across all
  #   threads in the group.
  #   LSC is the number of elements loaded in the para(coalesced) dimension
  #   PerLoadTile is always rectangular.
  #   The area (LSC*LSP) can be larger than NumThreads. In this case, some
  #   threads will generate a dummy OOB GRO.
  # NumLoadsCoalesced and NumLoadsPerpendicular define the number of times the
  #   PerLoadTile is loaded in each dimension to fetch the LoadTile
  # LoadTile = (LSC * NumLoadsCoalesced) * (LSP * NumLoadsPerpendicular).
  #   For Fractional, the LoadTile can be larger than the MacroTile. Buffer
  #   loads will clip any OOB references to 0 and will also avoid writing these
  #   into LDS.

  # Fractional load algorithm:
  #  - Each load instruction loads one or more (complete) rows of the load tile.
  #     - Each row is LSC elements wide
  #     - Rows are complete and do not wrap. This allows a single base GRO VGPR
  #       to be used for all loads in the tile.
  #     - Some work-items in the load may not perform useful work. These WI will
  #       set their GRO to a large OOB number so as to do no harm
  #     - Some G2L registers space may be unused as well.
  #     - The 'used' message at the bottom of this routine computes and prints the
  #       wasted register space.
  #     - The wasted space is removed when the data is written to LDS- the LWO
  #       for work-items beyond the valid ones are set to safely write to OOB locations.

  #     - The 
  @staticmethod
  def setGlobalLoadTileDimFractional(state, tc, depthU):

    assert(depthU > 0)

    # parDim, perpDim define the LoadTile and are measured in elements
    if state["ProblemType"]["TLU%s"%tc]:
      parDim  = state["MacroTile%s"%tc]
      perpDim = depthU
    else:
      parDim  = depthU
      perpDim = state["MacroTile%s"%tc]

    print "\ninfo: %s Fractional NumThreads=%u GRWV=%u MT=%ux%u" \
        % (tc, state["NumThreads"], state["GlobalReadVectorWidth"], \
           parDim, perpDim)

    # Try to find a GRVW which is smaller than the LSC and also does not force
    # the LSC to wrap - both of these conditions can be tested with lsc % grvw ==0.
    # Each iteration divides GRWV by 2 which provides finer granularity
    # and a possible opportunity to handle the lsc
    grvw = state["GlobalReadVectorWidth"]
    minVw = 2 if state["ProblemType"]["DataType"].isHalf() else 1
    bestVw = -1
    while grvw >= minVw:
      # Per instruction across the entire group:
      elementsLoadedPerInst = state["NumThreads"]*grvw
      # LSC, LSP - #elements loaded along specified dim with each load
      if parDim >= elementsLoadedPerInst:
        # entire work-group can work on (part) of the same row
        state["LSC%s"%tc] = elementsLoadedPerInst
        state["LSP%s"%tc] = 1
        state["NumLoadsCoalesced%s"%tc] = roundupRatio(parDim , state["LSC%s"%tc])
        state["NumLoadsPerpendicular%s"%tc] = 1
      else:
        # work-group exceeds read dimension so wraps to multiple rows
        state["LSC%s"%tc] = parDim
        state["LSP%s"%tc] = min(perpDim, elementsLoadedPerInst / parDim)
        state["NumLoadsCoalesced%s"%tc] = 1
        state["NumLoadsPerpendicular%s"%tc] = roundupRatio(perpDim , state["LSP%s"%tc])

      # Vector loads can't wrap to next P dim, so LSC must be divisible by vector elements;
      print "  lsc search : lsc(%u) %% grvw(%u) = %u (?0)" % (state["LSC%s"%tc], grvw, state["LSC%s"%tc] % grvw)
      if state["LSC%s"%tc] % grvw == 0:
        bestVw = grvw
        # Try to shrink GRVW if possible while keeping same LSC and LSP:
        # For example, avoid cases where we use a GRVW=4 with many empty addresses
        # when a GRVW=1 will do instead.
        validElementsLoadedPerInst = state["LSC%s"%tc] * state["LSP%s"%tc]
        grvw /= 2
        while grvw >= minVw:
          elementsLoadedPerInst = state["NumThreads"]*grvw
          if elementsLoadedPerInst < validElementsLoadedPerInst:
            break # Went too far, not enough load elements at this VW
          if state["LSC%s"%tc] % grvw == 0:
            print "  stepdown success (valid)elementsLoadedPerInst=", validElementsLoadedPerInst, "/", elementsLoadedPerInst, "grvw=", grvw, "lsc=", state["LSC%s"%tc]
            bestVw = grvw
          grvw /= 2
        break

      # TODO - could have this generate dwordx3 loads in addition, step down by 1 instead of div2
      # Would need to change asm code gen to generate x3
      grvw /= 2
      # end-- while loop

    if bestVw == -1:
      print "reject fractional - no acceptable tile dim? GlobalReadVectorWidth", state["GlobalReadVectorWidth"]
      return False  # could not find a solution, perhaps only possible for half ?

    state["GlobalLoadVectorWidth%s"%tc] = bestVw
    if bestVw != state["GlobalReadVectorWidth"]:
      print("  reducing GlobalLoadVectorWidth%s from %u to %u" \
          % (tc, state["GlobalReadVectorWidth"], bestVw))

    # How many loads per threads in each dimension.
    # threads which are outside the global read tile bounds will be clipped
    # in the assembly code generator.
    # Multiply the LSC*GRVW
    state["NumLoadsCoalesced%s"%tc] = roundupRatio(parDim, state["LSC%s"%tc])
    state["NumLoadsPerpendicular%s"%tc] = roundupRatio(perpDim , state["LSP%s"%tc])

    nlc = state["NumLoadsCoalesced%s"%tc]
    nlp = state["NumLoadsPerpendicular%s"%tc]

    # LoadTile must at least cover the MacroTile:
    assert(nlc*state["LSC%s"%tc] >= parDim)
    assert(nlp*state["LSP%s"%tc] >= perpDim)

    # how many threads compute Global Read Offsets (GRO) that are not used
    print "  PerLoadTile=%ux%u elements Loads/WI=%ux%u LoadTile/WI=%ux%u (MT=%ux%u), %u/%u = %.1f%% WI GRO used" \
        % (state["LSC%s"%tc], state["LSP%s"%tc], \
           nlc, nlp, \
           nlc*state["LSC%s"%tc], nlp*state["LSP%s"%tc], \
           parDim, perpDim, \
           parDim*perpDim, \
           nlc*nlp*state["NumThreads"]*state["GlobalLoadVectorWidth%s"%tc], \
           (float)(parDim*perpDim) / \
           (float)(nlc*nlp*state["NumThreads"]*state["GlobalLoadVectorWidth%s"%tc]) * 100.0 \
           )

    return True


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

    if state["ProblemType"]["Tensor0"]==0:
      state["ThreadTileA"] = state["ThreadTile0"]
      state["ThreadTileB"] = state["ThreadTile1"]
      state["SubGroupA"] = state["SubGroup0"]
      state["SubGroupB"] = state["SubGroup1"]
      state["MacroTileA"] = state["MacroTile0"]
      state["MacroTileB"] = state["MacroTile1"]
    else:
      state["ThreadTileB"] = state["ThreadTile0"]
      state["ThreadTileA"] = state["ThreadTile1"]
      state["SubGroupB"] = state["SubGroup0"]
      state["SubGroupA"] = state["SubGroup1"]
      state["MacroTileB"] = state["MacroTile0"]
      state["MacroTileA"] = state["MacroTile1"]

    # Init vars early since there are early-exit return statements below
    state["DirectToLdsA"] = False
    state["DirectToLdsB"] = False
    state["LocalWriteUseSgprA"] = False
    state["LocalWriteUseSgprB"] = False

    # VectorWidth default handling
    if state["VectorWidth"] < 1:
      state["VectorWidth"] = int(4 / state["ProblemType"]["DataType"].numRegisters())
      while state["ThreadTile0"] % state["VectorWidth"] != 0 \
          or state["ThreadTile1"] % state["VectorWidth"] != 0:
        state["VectorWidth"] /= 2
    # TT0,1 both must be multiples of VW, b/c of rC, rA, rB
    if state["ThreadTile0"] % state["VectorWidth"] != 0 \
        or state["ThreadTile1"] % state["VectorWidth"] != 0:
      reject(state, "ThreadTile0 %u or ThreadTile1 %u not a multiple of VectorWidth %u" \
          % (state["ThreadTile0"], state["ThreadTile1"], \
          state["VectorWidth"]))
      return


    # Default GlobalReadVectorWidth
    if state["GlobalReadVectorWidth"] == -1:
      state["GlobalReadVectorWidth"] = state["VectorWidth"]

    # Vector-width must be at least 2 for Half:
    if state["ProblemType"]["DataType"].isHalf() \
        and state["GlobalReadVectorWidth"] < 2:
      state["GlobalReadVectorWidth"] = 2

    if state["MinGlobalWriteVectorWidth"] == -1:
      state["MinGlobalWriteVectorWidth"] = 2 \
        if state["ProblemType"]["DataType"].isHalf() else 1

    if state["VectorWidth"]*state["ProblemType"]["DataType"].numBytes() > 16:
      # reject - VW too big
      state["Valid"] = False

    if state["GlobalReadVectorWidth"]*state["ProblemType"]["DataType"].numBytes() > 16:
      # reject - GRVW too big
      state["Valid"] = False


    # LocalSplitU too large?
    numElementsPerWorkGroup = state["MacroTile0"]*state["MacroTile1"]
    if numElementsPerWorkGroup < state["NumThreads"]:
      reject(state, "NumElementsPerWorkGroup %u < NumThreads %u; reduce LocalSplitU" \
          % (numElementsPerWorkGroup, state["NumThreads"]))
      return
    state["NumElementsPerThread"] = numElementsPerWorkGroup / \
        state["NumThreads"]
    state["GlobalWriteVectorWidth"] = min(state["VectorWidth"], state["NumElementsPerThread"] )
    if state["NumElementsPerThread"] % state["GlobalWriteVectorWidth"] != 0:
      reject(state, "LSU NumElementsPerThread %u not divisible into GWVW %u" \
          % (state["NumElementsPerThread"], state["GlobalWriteVectorWidth"]))
      return
    state["NumGlobalWriteVectorsPerThread"] = state["NumElementsPerThread"] \
        / state["GlobalWriteVectorWidth"]


    # LocalSplitU but can't NumThreads%MacroTile doesn't support sideways store
    if state["LocalSplitU"] > 1:
      if state["NumThreads"] % state["MacroTile0"] != 0:
        reject(state, "LocalSplitU but NumThreads=%u not divisible by MT0=%u for sideways store" \
            % (state["NumThreads"], state["MacroTile0"]))
        return
      if state["MacroTile0"]*state["MacroTile1"] % state["NumThreads"] != 0:
        reject(state, "LocalSplitU but MT0*MT1=%u elements doesn't divide into NumThreads=%u" \
            % (state["MacroTile0"]*state["MacroTile1"], state["NumThreads"]))
        return

    # GlobalSplitU doesn't work with some other things:
    if state["GlobalSplitU"] > 1:
      if not state["GlobalSplitUSummationAssignmentRoundRobin"] \
          and state["LoopTail"]:
        reject(state, "GlobalSplitU and LoopTail require SummationAssignmentRoundRobin=True since strongly breaks Tensile kernel architecture")
        return
      if not state["ProblemType"]["DataType"].isSingle():
        reject(state, "GlobalSplitU only compatible with single precision")
        return

    ########################################
    # Initial DepthU
    ########################################
    userDepthU = state["DepthU"]
    # DepthU == -1 means glvw=1
    if state["DepthU"] == -1:
      if state["MacroTile0"] != state["MacroTile1"]:
        reject(state, "DepthU=0 requires square MacroTile")
        return

    if userDepthU < 0:
      depthU = 2
      maxDepthU = globalParameters["MaxDepthU"]
    else:
      depthU = userDepthU
      maxDepthU = userDepthU

    ########################################
    # Search DepthU
    ########################################
    while True: # exit criteria at end
      validDepthU = True

      # how many elements to load
      if state["ProblemType"]["TLUA"]:
        totalElementsCoalescedA = state["MacroTile0"]
        totalElementsPerpA = depthU
      else:
        totalElementsCoalescedA = depthU
        totalElementsPerpA = state["MacroTile0"]

      if state["ProblemType"]["TLUB"]:
        totalElementsCoalescedB = state["MacroTile1"]
        totalElementsPerpB = depthU
      else:
        totalElementsCoalescedB = depthU
        totalElementsPerpB = state["MacroTile1"]

      totalElementsA = totalElementsCoalescedA * totalElementsPerpA
      totalElementsB = totalElementsCoalescedB * totalElementsPerpB


      if state["FractionalLoad"]:
        if not Solution.setGlobalLoadTileDimFractional(state, "A", depthU):
          validDepthU = False
        if not Solution.setGlobalLoadTileDimFractional(state, "B", depthU):
          validDepthU = False
      else:
        tva = totalElementsA / state["GlobalReadVectorWidth"]
        tvb = totalElementsB / state["GlobalReadVectorWidth"]
        if not Solution.setGlobalLoadVectorWidth(state, "A", tva):
          validDepthU = False
        if not Solution.setGlobalLoadVectorWidth(state, "B", tvb):
          validDepthU = False

      # Now convert elements to vectors based on GlobalReadVectorWidth
      totalVectorsCoalescedA = totalElementsCoalescedA / state["GlobalReadVectorWidth"]
      totalVectorsCoalescedB = totalElementsCoalescedB / state["GlobalReadVectorWidth"]
      totalVectorsA = totalElementsA / state["GlobalReadVectorWidth"]
      totalVectorsB = totalElementsB / state["GlobalReadVectorWidth"]

      print "info:", pvar(state, "NumThreads"), pvar(state, "DepthU"), \
                     pvar(state, "ThreadTile0"), pvar(state, "ThreadTile1"), \
                     "WG=%ux%u" % (state["WorkGroup"][0], state["WorkGroup"][1]), \
                     pvar(state, "MacroTileA"), pvar(state, "MacroTileB")
      print "info: totalElementsCoalescedA=", totalElementsCoalescedA, \
            " totalVectorsCoalescedA=", totalVectorsCoalescedA, " totalVectorsA=", totalVectorsA
      print "info: totalElementsCoalescedB=", totalElementsCoalescedB, \
            " totalVectorsCoalescedB=", totalVectorsCoalescedB, " totalVectorsB=", totalVectorsB

      # f16 can't load shorts from global->lds
      if state["ProblemType"]["DataType"].isHalf() \
          and (state["GlobalLoadVectorWidthA"] == 1 \
          or state["GlobalLoadVectorWidthB"] == 1):
        if "KernelLanguage" in state:
          if state["KernelLanguage"] == "Assembly":
            validDepthU = False
        else:
          validDepthU = False

      if not state["FractionalLoad"]:
        if userDepthU == -1: # no vectors
          if state["GlobalLoadVectorWidthA"] != 1 \
              or state["GlobalLoadVectorWidthB"] != 1:
            validDepthU = False
        elif userDepthU == -2:
          if max( state["GlobalLoadVectorWidthA"], \
              state["GlobalLoadVectorWidthB"]) \
              < state["GlobalReadVectorWidth"]:
            validDepthU = False
        elif userDepthU <= -3:
          if min( state["GlobalLoadVectorWidthA"], \
              state["GlobalLoadVectorWidthB"]) \
              < state["GlobalReadVectorWidth"]:
            validDepthU = False

      if not state["ProblemType"]["TLUA"]:
        if depthU < state["GlobalLoadVectorWidthA"]:
          validDepthU = False

      if not state["ProblemType"]["TLUB"]:
        if depthU < state["GlobalLoadVectorWidthB"]:
          validDepthU = False

      # this depthU is valid, done unless user wants to double (for TN)
      if validDepthU:
        if userDepthU < -3: # for every int below -3, use next doubled value
          userDepthU += 1
          depthU *= 2
          continue
        else: # use this found value
          state["DepthU"] = depthU
          break

      # this depthU not valid
      else:
        # keep looking
        if depthU < maxDepthU:
          depthU += 2
          continue
        # give up
        else:
          state["Valid"] = False
          return
    ########################################
    # end DepthU loop
    ########################################

    # f16 asm can't load shorts from global->lds
    if state["ProblemType"]["DataType"].isHalf() \
        and (state["GlobalLoadVectorWidthA"] == 1 \
        or state["GlobalLoadVectorWidthB"] == 1):
      if "KernelLanguage" in state:
        if state["KernelLanguage"] == "Assembly":
          reject(state, "f16 kernels can't load shorts from global->lds")
          return

    if not state["FractionalLoad"]:
      if not Solution.setGlobalLoadTileDimClassic(state, "A", state["NumLoadsA"], \
          totalVectorsCoalescedA, totalElementsPerpA):
        return
      if not Solution.setGlobalLoadTileDimClassic(state, "B", state["NumLoadsB"], \
          totalVectorsCoalescedB, totalElementsPerpB):
        return


    # TODO
    if (0 and state["LSCA"] % state["GlobalLoadVectorWidthA"] != 0):
      reject(state, "lsca % grvw != 0")
      return
    if (0 and state["LSPA"] % state["GlobalLoadVectorWidthA"] != 0):
      reject(state, "lspa % grvw != 0")
      return
    if (0 and state["LSCB"] % state["GlobalLoadVectorWidthB"] != 0):
      reject(state, "lscb % grvw != 0")
      return
    if (0 and state["LSPB"] % state["GlobalLoadVectorWidthB"] != 0):
      reject(state, "lspb % grvw != 0")
      return

    state["LVCA"] = roundupRatio(state["LSCA"] , state["GlobalLoadVectorWidthA"])
    state["LVPA"] = roundupRatio(state["LSPA"] , state["GlobalLoadVectorWidthA"])
    state["LVCB"] = roundupRatio(state["LSCB"] , state["GlobalLoadVectorWidthB"])
    state["LVPB"] = roundupRatio(state["LSPB"] , state["GlobalLoadVectorWidthB"])

    # Some of these might become 0?
    print "info: ", pvar(state, "LVCA"), pvar(state, "LVPA"), \
          pvar(state, "LVCB"), pvar(state, "LVPB")

    # lds buffer size for A, B
    if state["KernelLanguage"] == "Source" and \
       state["LdsPadA"] != state["LdsPadB"]:
      reject(state, "Source KernelLanguage only supports LdsPadA == LdsPadB")
      return

    if state["KernelLanguage"] == "Assembly" and state["PersistentKernel"]:
      if globalParameters["PrintSolutionRejectionReason"]:
        print1("Persistent only works on Source path")
      state["Valid"] = False
      return

    ldsAlign = int(64 / state["ProblemType"]["DataType"].numRegisters())
    ldsNumElementsA = state["DepthU"]*(state["MacroTile0"]+state["LdsPadA"])
    ldsNumElementsAlignedA = ((ldsNumElementsA+ldsAlign-1)/ldsAlign)*ldsAlign

    ldsNumElementsB = state["DepthU"]*(state["MacroTile1"]+state["LdsPadB"])
    ldsNumElementsAlignedB = ((ldsNumElementsB+ldsAlign-1)/ldsAlign)*ldsAlign
    # todo, can the alignment be a power of 2?
    state["LdsOffsetA"] = 0
    if state["PrefetchGlobalRead"]:
      state["LdsNumElementsAlignedA"] = ldsNumElementsAlignedA
      state["LdsNumElementsAlignedB"] = ldsNumElementsAlignedB
      state["LdsOffsetB"] = state["LdsOffsetA"] \
        + state["LdsNumElementsAlignedA"]

      offsetBlk = state["LdsOffsetB"] + ldsNumElementsAlignedB
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
      reject(state, "Kernel Uses %u > %u bytes of LDS" % ( ldsSize, globalParameters["MaxLDS"]))
      return

    # LoopUnroll  = DepthU / LocalSplitU
    if "LocalSplitU" in state and "DepthU" in state:
      state["LoopUnroll"] = state["DepthU"] / state["LocalSplitU"]
    if state["LoopUnroll"] * state["LocalSplitU"] != state["DepthU"]:
        state["Valid"] = False
    state["LoopUnroll"] /= state["InnerUnroll"]

    print "info: ", pvar(state, "LoopUnroll"), " LDS Stats:", pvar(state, "LdsOffsetA"), pvar(state, "LdsOffsetB")
    print "info: ", pvar(state["ProblemType"], "TLUA"), \
        pvar(state, "NumLoadsCoalescedA"), pvar(state, "NumLoadsPerpendicularA"), \
        pvar(state, "LSCA"), pvar(state, "LSPA")
    print "info:", pvar(state["ProblemType"], "TLUB"), \
        pvar(state, "NumLoadsCoalescedB"), pvar(state, "NumLoadsPerpendicularB"), \
        pvar(state, "LSCB"), pvar(state, "LSPB")

    # LoopUnroll too small
    if state["LoopUnroll"] < 2:
      reject(state, "LoopUnroll %u is less than 2" \
          % (state["LoopUnroll"]))

    if not state["BufferLoad"] or state["KernelLanguage"] != "Assembly":
      state["BufferLoad"] = False
      state["DirectToLds"] = False
      state["PreciseBoundsCheck"] = False
      state["UseSgprForGRO"] = False
      state["FractionalLoad"] = False

    # Determine if we can load directly-to-LDS.
    # Transpose requires a trip through registers to perform the transpose so can't use DirectToLdsA
    # LDS loads always write 4 bytes apart so can use only 4-byte operations
    #   TODO - for doubles we need to add something special here?
    # The matrix must not require transposing since that is done by reading to VGPR and writing in different order
    # The LSC (load size coalesced) must load some multiple of 256 bytes since that is what each DirectToLds load provides
    # Note for these matrices LSC is same as MacroTile dim
    if state["DirectToLds"]:
      # The tail loop requires half summation elements be a multiple of two to use DirectToLds feature
      elementMultipleOk = not state["ProblemType"]["DataType"].isHalf() \
                          or state["AssertSummationElementMultiple"] % 2 == 0

      wavefronts = state["NumThreads"] / globalParameters["WavefrontWidth"]
      numBytes = state["ProblemType"]["DataType"].numBytes()

      # DirectToLds loads return 256 bytes/wave
      # If fractional, ensure we are using all of the bytes that will be delivered

      if elementMultipleOk \
        and state["NumThreads"] % globalParameters["WavefrontWidth"] == 0:

        if (state["GlobalLoadVectorWidthA"] * numBytes == 4) \
          and not state["ProblemType"]["TransposeA"] \
          and state["LSCA"] * numBytes == 256 * wavefronts \
          and state["LSCA"] * numBytes == state["NumThreads"] * 4 :
          state["DirectToLdsA"] = True
          state["LocalWriteUseSgprA"] = True

        if (state["GlobalLoadVectorWidthB"] * state["ProblemType"]["DataType"].numBytes() == 4) \
          and state["ProblemType"]["TransposeB"] \
          and elementMultipleOk \
          and state["LSCB"] * numBytes == 256 * wavefronts \
          and state["LSCB"] * numBytes == state["NumThreads"] * 4 :
          state["DirectToLdsB"] = True
          state["LocalWriteUseSgprB"] = True

      if 1:
        print "A: TLU=", state["ProblemType"]["TLUA"], " MT=", state["MacroTile0"], \
               " LSCA=", state["LSCA"], "LSPA=", state["LSPA"], "GLVB_A=", state["GlobalLoadVectorWidthA"], \
               " dataTypeNumBytes=", state["ProblemType"]["DataType"].numBytes(), \
               "  ->DirectToLdsA=", state["DirectToLdsA"], \
               " NumLoadsCoalescedA=", state["NumLoadsCoalescedA"], \
               " NumLoadsPerpendicularA=", state["NumLoadsPerpendicularA"]
        print "B: TLU=", state["ProblemType"]["TLUB"], " MT=", state["MacroTile1"], \
               " LSCB=", state["LSCB"],"LSPB=", state["LSPB"],  "GLVB_B=", state["GlobalLoadVectorWidthB"], \
               " dataTypeNumBytes=", state["ProblemType"]["DataType"].numBytes(), \
               "  ->DirectToLdsB=", state["DirectToLdsB"], \
               " NumLoadsCoalescedB=", state["NumLoadsCoalescedB"], \
               " NumLoadsPerpendicularB=", state["NumLoadsPerpendicularB"]

      # Update parent variable so kernel display is accurate
      state["DirectToLds"] = state["DirectToLdsA"] or state["DirectToLdsB"]

    # Precise bounds check uses the "num_records" field in the buffer to
    # precisely detect when we are inbounds or not.  Only a one-dimensional
    # check is used since this is faster and also for computation we only
    # need to ensure that none of the loads fault.  threads which are
    # computing bogus sections of the C tile will later be ignored.
    # precise checking only works for vectorloads<=AssertSummationElementMultiple
    # else if the vload crosses boundary we ignore all components not just the
    # ones that are OOB.
    if state["PreciseBoundsCheck"]:
      if  state["GlobalLoadVectorWidthA"] > \
          state["AssertSummationElementMultiple"] \
          or state["GlobalLoadVectorWidthB"] > \
          state["AssertSummationElementMultiple"]:
        state["PreciseBoundsCheck"] = False

    # Use SGPR to store an offset from GlobalReadOffsetA+0.
    # (as opposed to using dedicated VGPR for each GRO
    # Requires preciseBounds check since we rely on the buffer bounds check, not
    # individual vector registers doing bounds compares.
    if not state["PreciseBoundsCheck"]:
      state["UseSgprForGRO"] = 0

    if state["UseSgprForGRO"] == -1:
      # Don't use SGPR if it looks like we might not have enough:
      # 40 is based on current SGPR usage, this may need to be tuned in the future:
      numLoadsA = state["NumLoadsCoalescedA"]*state["NumLoadsPerpendicularA"]
      numLoadsB = state["NumLoadsCoalescedB"]*state["NumLoadsPerpendicularB"]
      if numLoadsA + numLoadsB > 40:
        print "info: Disabling UseSgprForGRO since predicting too many SGPR will be used"
        state["UseSgprForGRO"] = 0
      else:
        state["UseSgprForGRO"] = 1

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
  # Get Name Min
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
    elif isinstance(value, tuple):
      abbrev = ""
      for i in range(0, len(value)):
        abbrev += str(value[i])
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


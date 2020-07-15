from copy import deepcopy

from .Common import globalParameters, CHeader
from .KernelWriterBase import KernelWriterBase

class KernelWriterBetaOnly(KernelWriterBase):

  def __init__(self, state):
    super().__init__()

    self.state["ProblemType"] = deepcopy(state["ProblemType"])
    self.state["_GlobalAccumulation"] = state["_GlobalAccumulation"]

    # derive parameter
    self.language = "HIP"
    self.kernelName = self.getKernelName()

    # determine chars for fast access
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[self.state["ProblemType"]["Index0"]] = "0" + self.indexChars[self.state["ProblemType"]["Index0"]]
    self.indexChars[self.state["ProblemType"]["Index1"]] = "1" + self.indexChars[self.state["ProblemType"]["Index1"]]
    self.tileChar0 = self.indexChars[self.state["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[self.state["ProblemType"]["Index1"]]


  def functionSignature(self):
    kStr = ""

    # self.state name
    kStr += self.endLine
    kStr += "extern \"C\"" + self.endLine
    kStr += "__global__ "
    kStr += "void %s" % ( self.kernelName )
    kStr += "(" + self.endLine

    # pointers
    restrictStr = "__restrict__"

    if self.state["_GlobalAccumulation"]:
      ptrStr = "float"
    else:
      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)

    kStr += "  " + ptrStr + " *D,"
    kStr += self.endLine
    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    kStr += "  " + ptrStr + " const * " + restrictStr + " C,"
    kStr += self.endLine

    # strides
    firstStrideCD = 1
    if self.state["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = self.state["ProblemType"]["NumIndicesC"]
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideD%s,%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideC%s,%s" % (self.indexChars[i], self.endLine)

    # sizes
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int const size%s" % self.indexChars[i]
      kStr += "," if (i < self.state["ProblemType"]["NumIndicesC"]-1 or self.state["ProblemType"]["UseBeta"]) else ")"
      kStr += self.endLine

    # beta
    if self.state["ProblemType"]["UseBeta"]:
      kStr += "  %s const beta)%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine )

    return kStr


  ##############################################################################
  # Kernel Body Beta-Only
  ##############################################################################
  def kernelBodyBetaOnly(self):
    problemType = self.state["ProblemType"]
    globalAccum = self.state["_GlobalAccumulation"]

    kStr = ""
    kStr += "{%s" % self.endLine

    ########################################
    # defined initial strides
    firstStride = 0
    if problemType["UseInitialStridesCD"]:
      # no strides #defined
      lastStrideC = 0
      assert 0  # need to fix beta-clear routine to pass initial stride parms
    else:
      # #define initial stride
      kStr += "/* hard-coded initial strides */%s" % self.endLine
      lastStrideC = 1
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideD" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + self.indexChars[i] + " 1" + self.endLine

    ########################################
    # GLOBAL_D()
    kStr += "#define GLOBAL_D(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideD%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideD%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    # GLOBAL_C()
    kStr += "#define GLOBAL_C(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideC%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideC%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    ########################################
    # wg d0, d1
    #kStr += "  unsigned int wg" + self.tileChar0 + " = " \
    #    + self.getGroupIdStr + "(0);" + self.endLine
    #kStr += "  unsigned int wg" + self.tileChar1 + " = " \
    #    + self.getGroupIdStr + "(1);" + self.endLine
    ########################################
    # wg other : batch dims
    freeIdxC0 = [idx for idx in range(problemType["NumIndicesC"]) \
                    if idx in problemType["IndexAssignmentsA"] and idx in problemType["IndicesFree"]]
    freeIdxC1 = [idx for idx in range(problemType["NumIndicesC"]) \
                    if idx in problemType["IndexAssignmentsB"] and idx in problemType["IndicesFree"]]

    batchSizes  = "*".join(["size%s"%self.indexChars[idx] for idx in problemType["IndicesBatch"]])
    freeSizesC0 = "*".join(["size%s"%self.indexChars[idx] for idx in freeIdxC0])
    freeSizesC1 = "*".join(["size%s"%self.indexChars[idx] for idx in freeIdxC1])

    t = []
    if freeSizesC0:
      t.append("(%s(0) >=  %s)" % (self.getGlobalIdStr, freeSizesC0))
    if freeSizesC1:
      t.append("(%s(1) >=  %s)" % (self.getGlobalIdStr, freeSizesC1))
    if batchSizes:
      t.append("(%s(2) >=  %s)" % (self.getGlobalIdStr, batchSizes))

    kStr += " if ("
    kStr += "\n   || ".join(t) + ")\n"
    kStr += "    return;\n"

    kStr += self.extractIndices(self.getGroupIdStr  + "(2)", "wg"     , problemType["IndicesBatch"])
    kStr += self.extractIndices(self.getGlobalIdStr + "(0)", "globalC", freeIdxC0)
    kStr += self.extractIndices(self.getGlobalIdStr + "(1)", "globalC", freeIdxC1)

    ########################################
    # D index
    kStr += "  %s idxD = GLOBAL_D( (%s)" % (self.uint64Str, self.uint64Str)
    kStr += ', '.join(["wg%s" % self.indexChars[i] if i in problemType["IndicesBatch"] else "globalC%s" % self.indexChars[i] \
                      for i in range(problemType["NumIndicesC"])])
    kStr += ");%s" % (self.endLine)

    # C index
    kStr += "  %s idxC = GLOBAL_C( (%s)" % (self.uint64Str, self.uint64Str)
    kStr += ', '.join(["wg%s" % self.indexChars[i] if i in problemType["IndicesBatch"] else "globalC%s" % self.indexChars[i] \
                      for i in range(problemType["NumIndicesC"])])
    kStr += ");%s" % (self.endLine)

    #kStr += "printf(\\\"%%09llu\\\\n\\\", idx);%s" % (self.endLine)

    ########################################
    # zero
    if globalAccum:
      ptrStr = "float"
    else:
      ptrStr = problemType["DataType"].toDevice(self.language)
    kStr += "#define SCALAR_ZERO ((%s)(0))%s" % (ptrStr, self.endLine )

    ########################################
    # zero
    computeType = problemType["ComputeDataType"].toDevice(self.language)
    if problemType["UseBeta"]:
      if problemType["DataType"].isComplex():
        kStr += "  if((beta.s0 == 0) && (beta.s1 == 0)) {%s" % self.endLine
      else:
        kStr += "  if(beta == SCALAR_ZERO) {%s" % self.endLine
      kStr += "    D[idxD] = SCALAR_ZERO;%s" % self.endLine
      kStr += "  } else {%s" % self.endLine
      kStr += "    D[idxD] = ((%s)(C[idxC])) * beta;%s" % (computeType, self.endLine)
      kStr += "  }%s" % self.endLine
    else:
      kStr += "  D[idxD] = SCALAR_ZERO;%s" % (self.endLine)

    ########################################
    # end
    kStr += "}%s" % self.endLine
    kStr += "#undef GLOBAL_D%s" % (self.endLine)
    kStr += "#undef GLOBAL_C%s" % (self.endLine)
    kStr += "#undef SCALAR_ZERO%s" % ( self.endLine)

    return kStr


  def getKernelName(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += self.state["ProblemType"]["DestDataType"].toChar()
    if self.state["ProblemType"]["UseBeta"]: name += "B"
    if self.state["_GlobalAccumulation"]: name += "_GA"

    return name


  def getSourceFileString(self):
    fileString = ""

    if not globalParameters["MergeFiles"]:
      fileString += "\n"
      fileString += "#include \"%s.h\"\n" % self.kernelName
      fileString += "\n"

    fileString += self.functionSignature()
    fileString += self.kernelBodyBetaOnly()

    return (0, fileString)

  def getHeaderFileString(self):
    fileString = "" # CHeader
    if not globalParameters["MergeFiles"]:
      fileString += CHeader
      fileString += "#pragma once\n\n"
      fileString += "\n"
      fileString += "#include <KernelHeader.h>\n\n"
      fileString += "#include <hip/hip_runtime.h>\n"
      fileString += "#include <hip/hip_fp16.h>\n"
      fileString += "\n"

    fileString += self.functionSignature()
    fileString += ";\n"

    return fileString

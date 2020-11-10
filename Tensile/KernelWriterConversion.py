from copy import deepcopy

from .Common import globalParameters, CHeader
from .KernelWriterBase import KernelWriterBase

class KernelWriterConversion(KernelWriterBase):

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

    # kernel name
    kStr += self.endLine
    kStr += "extern \"C\"\n"
    kStr += "__global__ "
    kStr += "void %s" % ( self.kernelName )
    kStr += "(" + self.endLine

    # pointers
    restrictStr = "__restrict__"
    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    kStr += "  " + ptrStr + " * D," + self.endLine
    kStr += "  " + "float * W," + self.endLine
    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    kStr += "  " + ptrStr + " const * " + restrictStr + " C," + self.endLine

    # alpha & beta
    kStr += "  %s const alpha,%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)
    kStr += "  %s const beta,%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)

    # strides
    firstStrideCD = 1
    if self.state["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = self.state["ProblemType"]["NumIndicesC"]
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideD%s,%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideW%s,%s" % (self.indexChars[i], self.endLine)
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideC%s,%s" % (self.indexChars[i], self.endLine)

    # sizes
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int const size%s,%s" % (self.indexChars[i], self.endLine)

    # gsu
    kStr += "  unsigned int const gsu)%s" % self.endLine

    return kStr


  def kernelBody(self):
    kStr = ""
    kStr += "{%s" % self.endLine
    problemType = self.state["ProblemType"]

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
      kStr += "#define strideW" + self.indexChars[i] + " 1" + self.endLine
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

    # GLOBAL_W()
    kStr += "#define GLOBAL_W(IDX%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideW%s" % (indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideW%s" % (indexChar, indexChar)
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
    # multi buffers GSU: Accumulate all GSU buffer
    indexChar = self.indexChars[0]
    kStr += "  uint64_t id = %s(0);%s" % (self.getGlobalIdStr, self.endLine)
    kStr += "  if (id >= (size%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += "*size%s" % self.indexChars[i]
    kStr += "))%s" % self.endLine
    kStr += "    return;%s" % self.endLine

    kStr += "  uint64_t id0"
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", id%d" % i
    kStr += ";%s" % self.endLine

    for i in range(0, problemType["NumIndicesC"]):
      kStr += "  id%d = id %% size%s;%s" % (i, self.indexChars[i], self.endLine)
      kStr += "  id  = id / size%s;%s" % (self.indexChars[i], self.endLine)

    kStr += self.endLine

    ########################################
    # D index
    kStr += "  %s idxD = GLOBAL_D( (%s)" % (self.uint64Str, self.uint64Str)
    kStr += ', '.join(["id%d" % i for i in range(problemType["NumIndicesC"])])
    kStr += ");%s" % (self.endLine)

    # W index
    kStr += "  %s idxW = GLOBAL_W( (%s)" % (self.uint64Str, self.uint64Str)
    kStr += ', '.join(["id%d" % i for i in range(problemType["NumIndicesC"])])
    kStr += ");%s" % (self.endLine)

    # D index
    kStr += "  %s idxC = GLOBAL_C( (%s)" % (self.uint64Str, self.uint64Str)
    kStr += ', '.join(["id%d" % i for i in range(problemType["NumIndicesC"])])
    kStr += ");%s" % (self.endLine)

    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    indexChar = self.indexChars[0]
    kStr += "  %s strideW = 1 + (size%s - 1) * strideW%s" % (self.uint64Str, indexChar, indexChar)
    for i in range(1, problemType["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (size%s - 1) * strideW%s" % (indexChar, indexChar)
    kStr += ";" + self.endLine

    kStr += "  float accum = 0.0f;%s" % self.endLine
    kStr += "  for (int i=0; i<gsu; i++) {%s" % self.endLine
    kStr += "    accum += W[idxW];%s" % self.endLine
    kStr += "    idxW  += strideW;%s" % self.endLine
    kStr += "  }%s" % self.endLine

    kStr += "  if( beta == (%s)0)%s" % (self.state["ProblemType"]["ComputeDataType"].toDevice(self.language), self.endLine)
    kStr += "    accum = ((float)alpha) * accum;%s" % (self.endLine)
    kStr += "  else%s" % self.endLine
    kStr += "    accum = (((float)alpha) * accum + ((float)beta) * ((float)C[idxC]));%s" % (self.endLine)

    typeStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    kStr += "  D[idxD] = (%s)accum;%s" % (typeStr, self.endLine)

    ########################################
    # end
    kStr += "}%s" % self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideD" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideW" + self.indexChars[i] + self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideC" + self.indexChars[i] + self.endLine
    kStr += "#undef GLOBAL_D%s" % (self.endLine)
    kStr += "#undef GLOBAL_W%s" % (self.endLine)
    kStr += "#undef GLOBAL_C%s" % (self.endLine)

    return kStr


  def getKernelName(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "C"
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += self.state["ProblemType"]["DestDataType"].toChar()
    name += "_PostGSU"

    return name


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


  def getSourceFileString(self):
    fileString = ""
    if not globalParameters["MergeFiles"]:
      fileString += "\n"
      fileString += "#include \"%s.h\"\n" % self.kernelName
      fileString += "\n"
    fileString += self.functionSignature()
    fileString += self.kernelBody()

    return (0, fileString)

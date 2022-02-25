################################################################################
# Copyright 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from copy import deepcopy

from .Common import globalParameters, CHeader
from .Activation import ActivationInline
from .KernelWriterBase import KernelWriterBase

class KernelWriterActivationOnly(KernelWriterBase):

  def __init__(self, state):
    super().__init__()

    self.state["ProblemType"] = deepcopy(state["ProblemType"])
    self.state["_GlobalAccumulation"] = state["_GlobalAccumulation"]
    self.state["WavefrontSize"] = state["WavefrontSize"]

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
    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    isStridedBuffer = self.state["ProblemType"]["StridedBatched"]
    ptrStr += "" if isStridedBuffer else "*"
    batch   = "" if isStridedBuffer else "Batch"
    kStr += "  " + ptrStr + " * " + batch + "D," + self.endLine

    ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    isStridedBuffer = self.state["ProblemType"]["StridedBatched"]
    ptrStr += "" if isStridedBuffer else "*"
    batch   = "" if isStridedBuffer else "Batch"

    if self.state["ProblemType"]["ActivationType"] != 'none':
      if self.state["ProblemType"]["ActivationType"] == 'all':
        kStr += "  Tensile::ActivationType const activationType,%s" % (self.endLine)
      activationCDataType = self.state["ProblemType"]["ComputeDataType"] if self.state["ProblemType"]["ActivationHPA"] else \
                            self.state["ProblemType"]["DestDataType"]
      for name in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        kStr += "  %s const %s,%s" % (activationCDataType.toDevice(self.language), name, self.endLine)

    # strides
    firstStrideCD = 1
    if self.state["ProblemType"]["UseInitialStridesCD"]:
      firstStrideCD = 0
    lastStrideC = self.state["ProblemType"]["NumIndicesC"]
    for i in range(firstStrideCD, lastStrideC):
      kStr += "  unsigned int const strideD%s,%s" % (self.indexChars[i], self.endLine)

    # sizes
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int const size%s,%s" % (self.indexChars[i], self.endLine)

    # offset
    kStr += "  unsigned int offsetD)%s" % self.endLine

    return kStr


  ##############################################################################
  # Kernel Body Beta-Only
  ##############################################################################
  def kernelBody(self):
    problemType = self.state["ProblemType"]

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

    ########################################
    # multi buffers GSU: Accumulate all GSU buffer
    indexChar = self.indexChars[0]
    kStr += "  uint64_t id = %s(0);%s" % (self.getGlobalIdStr, self.endLine)
    kStr += "  if (id >= (size%s" % self.indexChars[0]
    for i in range(1, problemType["NumIndicesC"]):
      kStr += "*size%s" % self.indexChars[i]
    kStr += "))%s" % self.endLine
    kStr += "    return;%s" % self.endLine

    kStr += self.endLine
    kStr += "  uint64_t id0"
    for i in range(1, problemType["NumIndicesC"]):
      kStr += ", id%d" % i
    kStr += ";%s" % self.endLine

    for i in range(0, problemType["NumIndicesC"]):
      kStr += "  id%d = id %% size%s;%s" % (i, self.indexChars[i], self.endLine)
      kStr += "  id  = id / size%s;%s" % (self.indexChars[i], self.endLine)

    nonTileFreeIndices = []

    # apply batch
    if not self.state["ProblemType"]["StridedBatched"]:
      nonTileFreeIndices = list(range(0, self.state["ProblemType"]["NumIndicesC"]))
      nonTileFreeIndices.remove(self.state["ProblemType"]["Index0"])
      nonTileFreeIndices.remove(self.state["ProblemType"]["Index1"])

      kStr += self.endLine
      kStr += "  uint64_t wg = 0"
      batchStride = "1"
      for i in nonTileFreeIndices:
        kStr += " + id%d * %s " % (i, batchStride)
        batchStride += " * size%s" % self.indexChars[i]
      kStr += ";" + self.endLine

      ptrStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
      kStr += "  " + ptrStr + " * D = BatchD[wg];" + self.endLine

    # apply offset
    kStr += self.endLine
    kStr += "  D = D + offsetD;" + self.endLine


    kStr += self.endLine
    ########################################
    # D index
    kStr += "  %s idxD = GLOBAL_D( (%s)" % (self.uint64Str, self.uint64Str)
    for i in range(problemType["NumIndicesC"]):
      tmpStr = ''
      if i in nonTileFreeIndices:
        tmpStr = '0'
      else:
        tmpStr = 'id%d' % i
      kStr += ', ' if i else ''
      kStr += tmpStr
    kStr += ");%s" % (self.endLine)

    ########################################
    # Activation
    typeStr = self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    typeActivationStr = self.state["ProblemType"]["ComputeDataType"].toDevice(self.language) if self.state["ProblemType"]["ActivationHPA"] else \
                        self.state["ProblemType"]["DestDataType"].toDevice(self.language)
    if self.state["ProblemType"]["ActivationType"] != 'none':
      names = ""
      if self.state["ProblemType"]["ActivationType"] == 'all':
        names += ", activationType"
      for name in self.state["ProblemType"]["ActivationType"].getAdditionalArgStringList():
        names += (", " + name)
      if self.state["ProblemType"]["DestDataType"].isInt8() and self.state["ProblemType"]["HighPrecisionAccumulate"]:
        kStr += "  D[idxD] = (%s)min(127, max(-128, activation((%s)D[idxD]%s)));%s" % (typeStr, typeActivationStr, names, self.endLine)
      else:
        kStr += "  D[idxD] = (%s)activation((%s)D[idxD]%s);%s" % (typeStr, typeActivationStr, names, self.endLine)

    ########################################
    # end
    kStr += "}%s" % self.endLine
    for i in range(firstStride, lastStrideC):
      kStr += "#undef strideD" + self.indexChars[i] + self.endLine
    kStr += "#undef GLOBAL_D%s" % (self.endLine)

    return kStr


  def getKernelName(self):
    indexChars = globalParameters["IndexChars"]
    # C dimensions
    name = "D"
    for i in range(0, self.state["ProblemType"]["NumIndicesC"]):
      name += indexChars[i].lower()
    name += "_"
    name += self.state["ProblemType"]["DestDataType"].toChar()
    if self.state["ProblemType"]["ActivationType"] == 'all':
      name += "_%s"%"A"
    elif self.state["ProblemType"]["ActivationType"] != 'none':
      name += "_%s"%str(self.state["ProblemType"]["ActivationType"]).upper()
    name += ("h" if self.state["ProblemType"]["ActivationHPA"] else "")

    return name


  def getSourceFileString(self):
    fileString = ""
    if not globalParameters["MergeFiles"]:
      fileString += "\n"
      fileString += "#include \"%s.h\"\n" % self.kernelName
      fileString += "\n"

    activationCDataType = self.state["ProblemType"]["ComputeDataType"] if self.state["ProblemType"]["ActivationHPA"] else \
                          self.state["ProblemType"]["DestDataType"]
    activation = ActivationInline(self.state["WavefrontSize"], \
                          activationCDataType)
    fileString += activation.generateInlineAssemblyFunction(self.state["ProblemType"]["ActivationType"])
    fileString += self.functionSignature()
    fileString += self.kernelBody()

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
      if self.state["ProblemType"]["ActivationType"] == 'all':
        fileString += "#include \"TensileActivationEnum.h\"\n"
      fileString += "\n"

    fileString += self.functionSignature()
    fileString += ";\n"

    return fileString

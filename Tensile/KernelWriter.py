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

from SolutionStructs import Solution, DataType
from Common import globalParameters

################################################################################
# Make OpenCL Kernel String
################################################################################
class KernelWriter:

  ##############################################################################
  # Make OpenCL Kernel String
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming ):
    self.language = globalParameters["KernelLanguage"]
    self.kernelMinNaming = kernelMinNaming
    self.kernelSerialNaming = kernelSerialNaming

    if self.language == "OCL":
      # everything escaped extra b/c string
      self.endLine = "\\n\"\n\""
      self.endLinePP = "\\\\" + self.endLine
    else:
      self.endLine = "\n"
      self.endLinePP =  "\\" + self.endLine

    if self.language == "OCL":
      self.getGroupIdStr = "get_group_id"
      self.getNumGroupsStr = "get_num_groups"
      self.getLocalIdStr = "get_local_id"
      self.getGlobalIdStr = "get_global_id"
      self.sharedDeclStr = "__local "
      self.sharedPtrStr = "__local "
      self.globalPtrStr = "__global "
      self.syncStr = "barrier(CLK_LOCAL_MEM_FENCE);"
      self.fenceStr = "mem_fence(CLK_LOCAL_MEM_FENCE);"
      self.macFStr = "mad"
      self.macDStr = "mad"
      self.int64Str = "long"
      self.uint64Str = "unsigned long"
      self.vectorComponents = ["s0", "s1", "s2", "s3"]
    else:
      self.getGroupIdStr = "hc_get_group_id"
      self.getNumGroupsStr = "hc_get_num_groups"
      self.getLocalIdStr = "hc_get_workitem_id"
      self.getGlobalIdStr = "hc_get_workitem_absolute_id"
      self.sharedDeclStr = "__shared__ "
      self.sharedPtrStr = ""
      self.globalPtrStr = ""
      self.syncStr = "__syncthreads();"
      self.fenceStr = self.syncStr
      self.macFStr = "fmaf"
      self.macDStr = "fma"
      self.int64Str = "int64_t"
      self.uint64Str = "uint64_t"
      self.vectorComponents = ["x", "y", "z", "w"]

    self.returnOnly = False
    self.tt2D = True

  ##############################################################################
  # get kernel name
  ##############################################################################
  def getKernelName(self, kernel):
    if globalParameters["ShortNames"]:
      kernelName = Solution.getNameSerial(kernel, self.kernelSerialNaming)
    else:
      kernelName = Solution.getNameMin(kernel, self.kernelMinNaming)
    return kernelName


  ##############################################################################
  # get kernel signature - DONE
  ##############################################################################
  def getSignature(self, kernel ):
    kernelName = self.getKernelName(kernel)

    # determine chars for fast access
    indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      indexChars.append(globalParameters["IndexChars"][i])
    indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + indexChars[kernel["ProblemType"]["Index0"]]
    indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + indexChars[kernel["ProblemType"]["Index1"]]
    tileChar0 = indexChars[kernel["ProblemType"]["Index0"]]
    tileChar1 = indexChars[kernel["ProblemType"]["Index1"]]

    s = ""
    # kernel name
    if self.language == "OCL":
      s += "__attribute__((reqd_work_group_size(NUM_THREADS,1,1)))"
      s += self.endLine
      s += "__kernel "
    else:
      s += "extern \"C\"\n"
      s += "__global__ "
    s += "void %s" % ( kernelName )
    s += "(" + self.endLine
    # pointers
    globalStr = "__global "
    if self.language == "HIP":
      s += "  hipLaunchParm lp," + self.endLine
      globalStr = ""
    restrictStr = "restrict"
    if self.language == "HIP":
      restrictStr = "__restrict__"
    ptrStr = kernel["ProblemType"]["DataType"].toDevice(self.language)
    s += "  " + globalStr + ptrStr \
        + " *C,"
    s += self.endLine
    s += "  " + globalStr + ptrStr \
        + " const * " + restrictStr + " A,"
    s += self.endLine
    s += "  " + globalStr + ptrStr \
        + " const * " + restrictStr + " B"

    # alpha & beta
    s += "," + self.endLine + "  " \
        + kernel["ProblemType"]["DataType"].toDevice(self.language) + " const alpha"
    if kernel["ProblemType"]["UseBeta"]:
      s += "," + self.endLine + "  " \
          + kernel["ProblemType"]["DataType"].toDevice(self.language) + " const beta"

    # offsets
    s += ( "," + self.endLine + "  unsigned int const offsetC,"
        + self.endLine +
        "  unsigned int const offsetA," + self.endLine +
        "  unsigned int const offsetB" )

    # strides
    firstStride = 1
    if kernel["ProblemType"]["UseInitialStrides"]:
      firstStride = 0
    lastStrideC = kernel["ProblemType"]["NumIndicesC"]
    lastStrideA = len(kernel["ProblemType"]["IndexAssignmentsA"])
    lastStrideB = len(kernel["ProblemType"]["IndexAssignmentsB"])
    for i in range(firstStride, lastStrideC):
      s += "," + self.endLine + "  unsigned int const strideC" + indexChars[i]
    for i in range(firstStride, lastStrideA):
      s += "," + self.endLine + "  unsigned int const strideA" \
          + indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
    for i in range(firstStride, lastStrideB):
      s += "," + self.endLine + "  unsigned int const strideB" \
          + indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]

    # sizes
    for i in range(0, kernel["ProblemType"]["TotalIndices"]):
      s += "," + self.endLine + "  unsigned int const size" + indexChars[i]
    s += " )"
    return s



  ##############################################################################
  # make kernel body
  ##############################################################################
  def getBody( self, kernel ):
    kernelName = self.getKernelName(kernel)

    # determine chars for fast access
    indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      indexChars.append(globalParameters["IndexChars"][i])
    indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + indexChars[kernel["ProblemType"]["Index0"]]
    indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + indexChars[kernel["ProblemType"]["Index1"]]

    # determine indices
    unrollChar = indexChars[kernel["ProblemType"]["IndicesSummation"][ \
        kernel["ProblemType"]["NumIndicesSummation"]-1]]
    tileChar0 = indexChars[kernel["ProblemType"]["Index0"]]
    tileChar1 = indexChars[kernel["ProblemType"]["Index1"]]
    tileCharA = tileChar0 if (kernel["ProblemType"]["Tensor0"]==0) else tileChar1
    tileCharB = tileChar0 if (kernel["ProblemType"]["Tensor0"]==1) else tileChar1
    tensorChar0 = "A" if (kernel["ProblemType"]["Tensor0"]==0) else "B"
    tensorChar1 = "A" if (kernel["ProblemType"]["Tensor1"]==0) else "B"

    ####################################
    # initializations
    kStr = ""
    kStr += self.endLine
    kStr += self.endLine
    kStr += "/****************************************/%s" % self.endLine
    kStr += "/* Preprocessor Definitions             */%s" % self.endLine
    kStr += "/****************************************/%s" % self.endLine

    ####################################
    # kernel preprocessor definitions
    kStr += self.endLine
    kStr += "/* tile parameters */" + self.endLine
    kStr += "#define NUM_THREADS %3d%s" \
        % (kernel["NumThreads"], self.endLine )
    kStr += "#define SG%s %d%s" \
        % (tileChar0, kernel["SubGroup0"], self.endLine )
    kStr += "#define SG%s %d%s" \
        % (tileChar1, kernel["SubGroup1"], self.endLine )
    kStr += "#define TT%s %d%s" \
        % (tileChar0, kernel["ThreadTile0"], self.endLine )
    kStr += "#define TT%s %d%s" \
        % (tileChar1, kernel["ThreadTile1"], self.endLine )
    kStr += "#define MT%s (SG%s*TT%s)%s" \
        % (tileChar0, tileChar0, tileChar0, self.endLine )
    kStr += "#define MT%s (SG%s*TT%s)%s" \
        % (tileChar1, tileChar1, tileChar1, self.endLine )
    kStr += self.endLine
    kStr += "/* DepthU parameters*/%s" % self.endLine
    kStr += "#define CPS (NUM_THREADS / MT%s * VECTOR_WIDTH)%s" \
        % (tileChar0, self.endLine)
    kStr += "#define SPLITU %d%s" \
        % (kernel["SplitU"], self.endLine )
    kStr += "#define UNROLL %d%s" \
        % (kernel["LoopUnroll"], self.endLine )
    kStr += "#define DEPTHU (SPLITU*UNROLL)%s" % (self.endLine )
    kStr += self.endLine
    kStr += "/* other */%s" % self.endLine
    kStr += "#define PAD %u%s" % (kernel["LdsPad"], self.endLine)
    kStr += "#define WORK_GROUP_MAPPING %u%s" % (abs(kernel["WorkGroupMapping"]), self.endLine)
    kStr += "#define VECTOR_WIDTH %u%s" % (kernel["VectorWidth"], self.endLine)
    kStr += self.endLine

    ####################################
    # num loads
    kStr += "/* num loads parallel and perpendicular to coalesced */" + self.endLine
    kStr += "#define NLCA %d%s" % (kernel["NumLoadsCoalescedA"], self.endLine )
    kStr += "#define NLCB %d%s" % (kernel["NumLoadsCoalescedB"], \
        self.endLine )

    kStr += "#define NLPA %d%s" % (kernel["NumLoadsPerpendicularA"], \
        self.endLine )
    kStr += "#define NLPB %d%s" % (kernel["NumLoadsPerpendicularB"], \
        self.endLine )
    kStr += self.endLine

    ####################################
    # load sizes
    kStr += "/* load sizes parallel and perpendicular to coalesced */%s" % self.endLine
    if kernel["ProblemType"]["TLUA"]:
      kStr += "#define LSCA (MT%s/NLCA)%s" \
          % (tileCharA, self.endLine)
      kStr += "#define LSPA (DEPTHU/NLPA)" + self.endLine
    else:
      kStr += "#define LSCA (DEPTHU/NLCA)%s" \
          % (self.endLine)
      kStr += "#define LSPA (MT%s/NLPA)%s" \
          % ( tileCharA, self.endLine)
    if kernel["ProblemType"]["TLUB"]:
      kStr += "#define LSCB (MT%s/NLCB)%s" \
          % (tileCharB, self.endLine)
      kStr += "#define LSPB (DEPTHU/NLPB)" + self.endLine
    else:
      kStr += "#define LSCB (DEPTHU/NLCB)%s" \
          % (self.endLine)
      kStr += "#define LSPB (MT%s/NLPB)%s" % (tileCharB, self.endLine)
    kStr += "#define LVCA (LSCA/VECTOR_WIDTH)%s" % (self.endLine)
    kStr += "#define LVCB (LSCB/VECTOR_WIDTH)%s" % (self.endLine)
    kStr += "#define LVPA (LSPA/VECTOR_WIDTH)%s" % (self.endLine)
    kStr += "#define LVPB (LSPB/VECTOR_WIDTH)%s" % (self.endLine)


    # lds buffer size
    kStr += "#define LDS_OFFSET_B %u%s" % (kernel["LdsOffsetB"], self.endLine)
    kStr += "#define LDS_NUM_ELEMENTS %u%s" % (kernel["LdsNumElements"], \
        self.endLine)


    ####################################
    # global memory indices
    kStr += self.endLine
    kStr += "/* global memory indices */" + self.endLine
    # C
    kStr += "#define GLOBAL_C(IDX%s" % indexChars[0]
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      kStr += ", IDX%s" % indexChars[i]
    indexChar = indexChars[0]
    kStr += ") (( (IDX%s)*strideC%s" % (indexChar, indexChar)
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      indexChar = indexChars[i]
      kStr += " + (IDX%s)*strideC%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine
    # A non-vector
    kStr += "#define GLOBAL_OFFSET_A(IDX%s" \
        % indexChars[kernel["ProblemType"]["IndexAssignmentsA"][0]]
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      kStr += ", IDX%s" \
          % indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
    indexChar = indexChars[kernel["ProblemType"]["IndexAssignmentsA"][0]]
    kStr += ") (( (IDX%s)*strideA%s" % (indexChar, indexChar)
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      indexChar = indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
      kStr += " + (IDX%s)*strideA%s" % (indexChar, indexChar)
    kStr += " ))%s" % self.endLine
    # B non-vector
    kStr += "#define GLOBAL_OFFSET_B(IDX%s" \
        % indexChars[kernel["ProblemType"]["IndexAssignmentsB"][0]]
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      kStr += ", IDX%s" \
          % indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]
    indexChar = indexChars[kernel["ProblemType"]["IndexAssignmentsB"][0]]
    kStr += ") (( (IDX%s)*strideB%s" % (indexChar, indexChar)
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      indexChar = indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]
      kStr += " + (IDX%s)*strideB%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine
    kStr += self.endLine

    ####################################
    # data types
    kStr += "/* data types */" + self.endLine
    kStr += "#define DATA_TYPE %s%s" \
        % (kernel["ProblemType"]["DataType"].toDevice(self.language), \
        self.endLine)
    vecStr = kernel["ProblemType"]["DataType"].toDevice(self.language)
    if kernel["VectorWidth"] > 1:
      vecStr += str(kernel["VectorWidth"])
    kStr += "#define VECTOR_TYPE %s%s" % (vecStr, self.endLine)

    if self.language == "OCL":
      kStr += "#define MAD(A,B,DST) mad(A,B,DST)"
    else:
      kStr += "#define MAD(A,B,DST) DST += A*B"
    kStr += self.endLine

    if self.language == "HIP" and kernel["ProblemType"]["DataType"].isComplex():
      kStr += "#define s0 x" + self.endLine
      kStr += "#define s1 y" + self.endLine
    kStr += self.endLine

    ####################################
    # MACs
    kStr += "/* MAC's */" + self.endLine
    if kernel["ProblemType"]["DataType"].isReal():
      # real data
      kStr += "#define TYPE_MAC(MULA,MULB,DST) " \
          + "DST = MAD(MULA,MULB,DST);" + self.endLine
      if kernel["ProblemType"]["UseBeta"]:
        # dst = alpha*reg + beta*dst
        kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG,BETA) " \
            + "DST = (ALPHA)*(REG) + (BETA)*(DST);" + self.endLine
      else:
        # dst = alpha*reg
        kStr += "#define TYPE_MAC_WRITE(DST,ALPHA,REG) " \
            + "DST = (ALPHA)*(REG);" + self.endLine
    else:
      # complex data
      if not kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]:
        # neither conjugate
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAD( -MULA.s1, MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]:
        # A conjugate (negate imaginary A.s1)
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s1, MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAD( -MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif not kernel["ProblemType"]["ComplexConjugateA"] and kernel["ProblemType"]["ComplexConjugateB"]:
        # B conjugate (negate imaginary B.s1)
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s0,  MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAD( -MULA.s1, -MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s0, -MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      else:
        # A & B conjugate (negate imaginary .s1)
        kStr += (
          "#define TYPE_MAC(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s0,  MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s1, -MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s0, -MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAD( -MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      if kernel["ProblemType"]["UseBeta"]:
        # dst = alpha*reg + beta*dst
        kStr += (
          "#define TYPE_MAC_WRITE( DST, ALPHA, REG, BETA ) "+self.endLinePP +
          "  /* (1) */ " + self.endLinePP +
          "  type_mac_tmp = REG.s0; " + self.endLinePP +
          "  REG.s0 *= ALPHA.s0; " + self.endLinePP +
          "  REG.s0 = MAD( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLinePP +
          "  REG.s1 *= ALPHA.s0; " + self.endLinePP +
          "  REG.s1 = MAD(  ALPHA.s1, type_mac_tmp, REG.s1 ); "+self.endLinePP+
          "  /* (2) */ " + self.endLinePP +
          "  REG.s0 = MAD(  BETA.s0, DST.s0, REG.s0 ); " + self.endLinePP +
          "  REG.s0 = MAD( -BETA.s1, DST.s1, REG.s0 ); " + self.endLinePP +
          "  REG.s1 = MAD(  BETA.s1, DST.s0, REG.s1 ); " + self.endLinePP +
          "  REG.s1 = MAD(  BETA.s0, DST.s1, REG.s1 ); " + self.endLinePP +
          "  /* (3) */ " + self.endLinePP +
          "  DST = REG;" + self.endLine )
      else:
        # dst = alpha*reg
        kStr += (
          "#define TYPE_MAC_WRITE( DST, ALPHA, REG ) "+self.endLinePP+
          "  /* (1) */ " + self.endLinePP +
          "  type_mac_tmp = REG.s0; " + self.endLinePP +
          "  REG.s0 *= ALPHA.s0; " + self.endLinePP +
          "  REG.s0 = MAD( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLinePP +
          "  REG.s1 *= ALPHA.s0; " + self.endLinePP +
          "  REG.s1 = MAD(  ALPHA.s1, type_mac_tmp, REG.s1 ); "+self.endLinePP+
          "  /* (3) */ " + self.endLinePP +
          "  DST = REG;" + self.endLine )

    ####################################
    # micro-tile
    kStr += self.endLine
    kStr += "/* %dx%d micro-tile */%s" % (kernel["ThreadTile0"], kernel["ThreadTile1"], self.endLine)

    kStr += "#define SUMMATION_UNROLL " + self.endLinePP
    for a in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
      kStr += "  rA[%d] = ldsReadIterA[%d*SG%s]; %s" \
          % (a, a, tileChar0, self.endLinePP)
    for b in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      kStr += "  rB[%d] = ldsReadIterB[%d*SG%s]; %s" \
          % (b, b, tileChar1, self.endLinePP)

    if False:
      if kernel["VectorWidth"] == 1:
        kStr += "  printf(\\\"MAC: T[%%02u]: %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f; %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f\\\\n\\\", serial, rA[0], rA[1], rA[2], rA[3], rA[4], rA[5], rA[6], rA[7], rB[0], rB[1], rB[2], rB[3], rB[4], rB[5], rB[6], rB[7]); %s" % (self.endLinePP)
      if kernel["VectorWidth"] == 2:
        kStr += "  printf(\\\"MAC: T[%%02u]: %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f; %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f\\\\n\\\", serial, rA[0].%s, rA[0].%s, rA[1].%s, rA[1].%s, rA[2].%s, rA[2].%s, rA[3].%s, rA[3].%s, rB[0].%s, rB[0].%s, rB[1].%s, rB[1].%s, rB[2].%s, rB[2].%s, rB[3].%s, rB[3].%s); %s" % ( \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.endLinePP)
      if kernel["VectorWidth"] == 4:
        kStr += "  printf(\\\"MAC: T[%%02u]: %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f; %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f\\\\n\\\", serial, rA[0].%s, rA[0].%s, rA[0].%s, rA[0].%s, rA[1].%s, rA[1].%s, rA[1].%s, rA[1].%s, rB[0].%s, rB[0].%s, rB[0].%s, rB[0].%s, rB[1].%s, rB[1].%s, rB[1].%s, rB[1].%s); %s" % ( \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.vectorComponents[0], self.vectorComponents[1], \
            self.vectorComponents[2], self.vectorComponents[3], \
            self.endLinePP)

    kStr += "  ldsReadIterA += SPLITU*(MT%s/VECTOR_WIDTH+PAD);%s" \
        % (tileChar0, self.endLinePP)
    kStr += "  ldsReadIterB += SPLITU*(MT%s/VECTOR_WIDTH+PAD);%s" \
        % (tileChar1, self.endLinePP)
    for b in range(0, kernel["ThreadTile1"]):
      for a in range(0, kernel["ThreadTile0"]):
        # a
        vecA = a / kernel["VectorWidth"]
        elemA = a % kernel["VectorWidth"]
        strA = "rA[%d]" % vecA
        if kernel["VectorWidth"] > 1:
          strA += ".%s" % self.vectorComponents[elemA]
        # b
        vecB = b / kernel["VectorWidth"]
        elemB = b % kernel["VectorWidth"]
        strB = "rB[%d]" % vecB
        if kernel["VectorWidth"] > 1:
          strB += ".%s" % self.vectorComponents[elemB]
        # c
        strC = "rC[(%d+%d*TT%s/VECTOR_WIDTH)]" % (vecA, b, tileChar0)
        elemC = elemA
        if kernel["VectorWidth"] > 1:
          strC += ".%s" % self.vectorComponents[elemA]
        """
        kStr += "  printf(\\\"T[%%u,%u,%u]: %s:%%.0f += %s:%%.0f * %s:%%.0f\\\\n\\\", serial, %s, %s, %s); %s" % (a, b, strC, strA, strB, strC, strA, strB, self.endLinePP)
        """
        kStr += "  TYPE_MAC(%s,%s,%s); %s" % (strA, strB, strC, self.endLinePP)

    kStr += "  " + self.fenceStr + self.endLine
    kStr += self.endLine

    ####################################
    # preprocessor definitions of kernel arguments
    firstStride = 0
    if kernel["ProblemType"]["UseInitialStrides"]:
      # no strides #defined
      lastStrideC = 0
      lastStrideA = 0
      lastStrideB = 0
    else:
      # #define initial stride
      kStr += "/* hard-coded initial strides */%s" \
          % self.endLine
      lastStrideC = 1
      lastStrideA = 1
      lastStrideB = 1

    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideA):
      kStr += "#define strideA" \
          + indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]] \
          + " 1" + self.endLine
    for i in range(firstStride, lastStrideB):
      kStr += "#define strideB" \
          + indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]] \
          + " 1" + self.endLine
    kStr += self.endLine

    ####################################
    # Function Signature
    ####################################
    kStr += "/****************************************/" + self.endLine
    kStr += "/* Begin Kernel                         */" + self.endLine
    kStr += "/****************************************/" + self.endLine
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic push" + self.endLine
      kStr += "#pragma clang diagnostic ignored \"-Wunused-parameter\"" + self.endLine
    kStr += self.getSignature(kernel)
    kStr += " {" + self.endLine
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic pop" + self.endLine

    ####################################
    # apply offsets
    kStr += self.endLine
    kStr += "  /* apply offsets */" + self.endLine
    kStr += "  C += offsetC;%s" % self.endLine
    kStr += "  A += offsetA;%s" % self.endLine
    kStr += "  B += offsetB;%s" % self.endLine

    kStr += self.endLine
    kStr += "  /***************************************/" + self.endLine
    kStr += "  /* Allocate Resources                  */" + self.endLine
    kStr += "  /***************************************/" + self.endLine

    ####################################
    # allocate registers
    kStr += self.endLine
    kStr += "  /* registers for MAC's */" + self.endLine
    kStr += "  VECTOR_TYPE rC[TT%s*TT%s/VECTOR_WIDTH] = {0};%s" \
        % (tileChar0, tileChar1, self.endLine )
    kStr += "  VECTOR_TYPE rA[TT%s/VECTOR_WIDTH];%s" \
        % (tileChar0, self.endLine)
    kStr += "  VECTOR_TYPE rB[TT%s/VECTOR_WIDTH];%s" \
        % (tileChar1, self.endLine)

    ####################################
    # registers for global -> local load
    kStr += self.endLine
    kStr += "  /* registers for global->local */" + self.endLine
    kStr += "  VECTOR_TYPE "
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        kStr += "a_" + str(para) + "_" + str(perp)
        if para == kernel["NumLoadsCoalescedA"]-1 and perp == kernel["NumLoadsPerpendicularA"]-1:
          kStr += ";" + self.endLine
        else:
          kStr += ", "
    kStr += "  VECTOR_TYPE "
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        kStr += "b_" + str(para) + "_" + str(perp)
        if para == kernel["NumLoadsCoalescedB"]-1 and perp == kernel["NumLoadsPerpendicularB"]-1:
          kStr += ";" + self.endLine
        else:
          kStr += ", "

    ####################################
    # allocate local memory
    kStr += self.endLine
    kStr += "  /* allocate LDS */" + self.endLine
    kStr += "  %sDATA_TYPE lds[LDS_NUM_ELEMENTS];%s" \
        % (self.sharedDeclStr, self.endLine )

    ####################################
    # Global Read Addresses
    ####################################
    kStr += self.endLine
    kStr += "  /***************************************/" + self.endLine
    kStr += "  /* Global Read Addresses               */" + self.endLine
    kStr += "  /***************************************/" + self.endLine
    kStr += self.endLine
    kStr += "  /* global read: work-group mapping */" + self.endLine
    if kernel["WorkGroupMapping"] == 1:
      kStr += "  unsigned int wg" + tileChar0 + " = " \
          + self.getGroupIdStr + "(0);" + self.endLine
      kStr += "  unsigned int wg" + tileChar1 + " = " \
          + self.getGroupIdStr + "(1);" + self.endLine
    else:
      dimCoal = (0 if kernel["WorkGroupMapping"] > 0 else 1)
      dimPerp = (1 if kernel["WorkGroupMapping"] > 0 else 0)

      # work-group free indices
      kStr += self.endLine
      kStr += "  unsigned int wg%s, wg%s;%s" % (tileChar0, tileChar1, self.endLine)
      kStr += "  %s groupSerial = %s(0) + %s(1) * %s(0);%s" \
          % (self.uint64Str, self.getGroupIdStr, self.getGroupIdStr, \
          self.getNumGroupsStr, self.endLine)
      kStr += "  %s superGroup = groupSerial / (%s(%u)*WORK_GROUP_MAPPING);%s" \
          % (self.uint64Str, self.getNumGroupsStr, dimCoal, self.endLine );
      kStr += "  unsigned int lastSuperGroupWidth = %s(%u) %% WORK_GROUP_MAPPING;%s" % \
          ( self.getNumGroupsStr, dimPerp, self.endLine )
      kStr += "  unsigned int numWorkGroupsBeforeLastSuperGroup = (%s(%u) - lastSuperGroupWidth)*%s(%u);%s" \
            % (self.getNumGroupsStr, dimPerp, self.getNumGroupsStr, dimCoal, \
            self.endLine)

      # if not in last super group
      kStr += "  if ( groupSerial < numWorkGroupsBeforeLastSuperGroup) {%s" \
              % (self.endLine)
      kStr += "    wg%s = (groupSerial/WORK_GROUP_MAPPING) %% %s(%s);%s" \
          % ((tileChar0 if kernel["WorkGroupMapping"] > 0 else tileChar1), \
          self.getNumGroupsStr, dimCoal, self.endLine)
      kStr += "    wg%s = superGroup*WORK_GROUP_MAPPING + groupSerial %% WORK_GROUP_MAPPING;%s" \
          % ((tileChar1 if kernel["WorkGroupMapping"] > 0 else tileChar0), \
          self.endLine)

      # if in last super group
      kStr += "  } else {%s" % self.endLine
      kStr += "    wg%s = (groupSerial-numWorkGroupsBeforeLastSuperGroup)/lastSuperGroupWidth;%s" \
          % ((tileChar0 if kernel["WorkGroupMapping"] > 0 else tileChar1), \
          self.endLine)
      kStr += "    wg%s = superGroup*WORK_GROUP_MAPPING + groupSerial %% lastSuperGroupWidth;%s" \
          % ((tileChar1 if kernel["WorkGroupMapping"] > 0 else tileChar0), \
          self.endLine)

      # if in last super group
      kStr += "  }%s" % self.endLine


    ####################################
    # global read: local c indices
    kStr += self.endLine
    kStr += "  /* global read: thread assignments */" + self.endLine
    # subgroup
    kStr += "  unsigned int serial = %s(0);%s" \
        % (self.getLocalIdStr, self.endLine)
    kStr += "  unsigned int sgId = serial / (SG%s*SG%s);%s" \
        % (tileChar0, tileChar1, self.endLine)

    # derrive global-read-coalesce-group from local in config
    if kernel["ProblemType"]["TLUA"]:
      globalReadCoalesceGroupA = kernel["LocalWriteCoalesceGroupA"]
    else:
      globalReadCoalesceGroupA = not kernel["LocalWriteCoalesceGroupA"]
    if kernel["ProblemType"]["TLUB"]:
      globalReadCoalesceGroupB = kernel["LocalWriteCoalesceGroupB"]
    else:
      globalReadCoalesceGroupB = not kernel["LocalWriteCoalesceGroupB"]

    # tile index assignment a
    kStr += "  unsigned int globalReadOffsetA%s = (serial%s" \
        % (tileCharA, ("%" if globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "/") )
    if globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorA"] == kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += " + (wg%s*MT%s);%s" \
        % (tileCharA, tileCharA, self.endLine)

    # tile index assignment b
    kStr += "  unsigned int globalReadOffsetB%s = (serial%s" \
        % (tileCharB, ("%" if globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "/") )
    if globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] == kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += " + (wg%s*MT%s);%s" \
        % (tileCharB, tileCharB, self.endLine)

    # summation index assignment a
    kStr += "  unsigned int globalReadOffsetA%s = (serial%s" \
        % (unrollChar, ("/" if globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "%") )
    if globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorA"] != kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine

    # summation index assignment b
    kStr += "  unsigned int globalReadOffsetB%s = (serial%s" \
        % (unrollChar, ("/" if globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "%") )
    if globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] != kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine

    ####################################
    # global read: other free indices
    nonTileFreeIndices = range(0, kernel["ProblemType"]["NumIndicesC"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index0"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index1"])
    if len(nonTileFreeIndices) > 0:
      kStr += self.endLine
      kStr += "  /* global read: other free indices */%s" % self.endLine
      for i in range(0, len(nonTileFreeIndices)):
        index = nonTileFreeIndices[i]
        kStr += "  unsigned int wg" + indexChars[index] \
            + " = ( " + self.getGroupIdStr + "(2)"
        for j in reversed( range( i+1, len(nonTileFreeIndices)) ):
          index2 = nonTileFreeIndices[j]
          kStr += " / size" + indexChars[index2]
        kStr += " ) % size" + indexChars[index] + ";" + self.endLine

    ####################################
    # global read: other non-unrolled summation indices
    if kernel["ProblemType"]["NumIndicesSummation"] > 1:
      kStr += self.endLine
      kStr += "  /* global read: other summation indices */%s" % self.endLine
      for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
        index = i
        kStr += "#define globalReadOffsetA%s 0%s" \
            % (indexChars[index], self.endLine)
        kStr += "#define globalReadOffsetB%s 0%s" \
            % (indexChars[index], self.endLine)

    ####################################
    # read / write vectors or vector components
    if kernel["ProblemType"]["TLUA"]: # NT no transpose
      numReadsTileA = kernel["NumLoadsCoalescedA"]
      numReadsUnrollA = kernel["NumLoadsPerpendicularA"]
      if kernel["GlobalReadCoalesceVectorA"]:
        readTileDimComponentsA = False # Vector
        readUnrollDimComponentsA = False # Scalar
        writeTileDimComponentsA = False # Vector
        writeUnrollDimComponentsA = False # Scalar
      else:
        readTileDimComponentsA = False # Scalar
        readUnrollDimComponentsA = kernel["VectorWidth"] > 1 # Components
        writeTileDimComponentsA = False # Scalar
        writeUnrollDimComponentsA = kernel["VectorWidth"] > 1 # Components
    else:
      numReadsTileA = kernel["NumLoadsPerpendicularA"]
      numReadsUnrollA = kernel["NumLoadsCoalescedA"]
      if kernel["GlobalReadCoalesceVectorA"]:
        readTileDimComponentsA = False # Scalar
        readUnrollDimComponentsA = False # Vector
        writeTileDimComponentsA = kernel["VectorWidth"] > 1 # Components
        writeUnrollDimComponentsA = False # Scalar
      else:
        readTileDimComponentsA = kernel["VectorWidth"] > 1 # Components
        readUnrollDimComponentsA = False # Scalar
        writeTileDimComponentsA = False # Vector
        writeUnrollDimComponentsA = False # Scalar

    numReadVectorComponentsA = kernel["VectorWidth"] \
        if (readTileDimComponentsA or readUnrollDimComponentsA) else 1
    numWriteVectorComponentsA = kernel["VectorWidth"] \
        if (writeTileDimComponentsA or writeUnrollDimComponentsA) else 1
    numReadTileVectorComponentsA = kernel["VectorWidth"] \
        if readTileDimComponentsA else 1 # for branches

    ####################################
    # global read: dimension offsets tile a
    kStr += self.endLine
    kStr += "  /* global read: dimension offsets a */%s" % self.endLine
    for l in range(0, numReadsTileA):
      if readTileDimComponentsA:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetA%s_%u_s%u = globalReadOffsetA%s + %u + %d*%s;%s" \
              % (tileCharA, l, s, tileCharA, s, l, \
              ("LSCA" if kernel["ProblemType"]["TLUA"] else "LSPA"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetA%s_%u = globalReadOffsetA%s + %d*%s;%s" \
            % (tileCharA, l, tileCharA, l, \
            ("LSCA" if kernel["ProblemType"]["TLUA"] else "LSPA"), \
            self.endLine)

    ####################################
    # global read: dimension offsets unroll a
    for l in range(0, numReadsUnrollA):
      if readUnrollDimComponentsA:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetA%s_%u_s%u = globalReadOffsetA%s + %u + %d*%s;%s" \
              % (unrollChar, l, s, unrollChar, s, l, \
              ("LSPA" if kernel["ProblemType"]["TLUA"] else "LSCA"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetA%s_%u = globalReadOffsetA%s + %d*%s;%s" \
            % (unrollChar, l, unrollChar, l, \
            ("LSPA" if kernel["ProblemType"]["TLUA"] else "LSCA"), \
            self.endLine)

    ####################################
    # read / write vectors or vector components b
    if kernel["ProblemType"]["TLUB"]: # NT no transpose
      numReadsTileB = kernel["NumLoadsCoalescedB"]
      numReadsUnrollB = kernel["NumLoadsPerpendicularB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        readTileDimComponentsB = False # Vector
        readUnrollDimComponentsB = False # Scalar
        writeTileDimComponentsB = False # Vector
        writeUnrollDimComponentsB = False # Scalar
      else:
        readTileDimComponentsB = False # Scalar
        readUnrollDimComponentsB = kernel["VectorWidth"] > 1 # Components
        writeTileDimComponentsB = False # Scalar
        writeUnrollDimComponentsB = kernel["VectorWidth"] > 1 # Components
    else:
      numReadsTileB = kernel["NumLoadsPerpendicularB"]
      numReadsUnrollB = kernel["NumLoadsCoalescedB"]
      if kernel["GlobalReadCoalesceVectorB"]:
        readTileDimComponentsB = False # Scalar
        readUnrollDimComponentsB = False # Vector
        writeTileDimComponentsB = kernel["VectorWidth"] > 1 # Components
        writeUnrollDimComponentsB = False # Scalar
      else:
        readTileDimComponentsB = kernel["VectorWidth"] > 1 # Components
        readUnrollDimComponentsB = False # Scalar
        writeTileDimComponentsB = False # Vector
        writeUnrollDimComponentsB = False # Scalar
    numReadVectorComponentsB = kernel["VectorWidth"] \
        if (readTileDimComponentsB or readUnrollDimComponentsB) else 1
    numWriteVectorComponentsB = kernel["VectorWidth"] \
        if (writeTileDimComponentsB or writeUnrollDimComponentsB) else 1
    numReadTileVectorComponentsB = kernel["VectorWidth"] \
        if readTileDimComponentsB else 1 # for branches

    ####################################
    # global read: dimension offsets tile b
    kStr += self.endLine
    kStr += "  /* global read: dimension offsets b */%s" % self.endLine
    for l in range(0, numReadsTileB):
      if readTileDimComponentsB:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetB%s_%u_s%u = globalReadOffsetB%s + %u + %d*%s;%s" \
              % (tileCharB, l, s, tileCharB, s, l, \
              ("LSCB" if kernel["ProblemType"]["TLUB"] else "LSPB"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetB%s_%u = globalReadOffsetB%s + %d*%s;%s" \
            % (tileCharB, l, tileCharB, l, \
            ("LSCB" if kernel["ProblemType"]["TLUB"] else "LSPB"), \
            self.endLine)

    ####################################
    # global read: dimension offsets unroll b
    for l in range(0, numReadsUnrollB):
      if readUnrollDimComponentsB:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetB%s_%u_s%u = globalReadOffsetB%s + %u + %d*%s;%s" \
              % (unrollChar, l, s, unrollChar, s, l, \
              ("LSPB" if kernel["ProblemType"]["TLUB"] else "LSCB"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetB%s_%u = globalReadOffsetB%s + %d*%s;%s" \
            % (unrollChar, l, unrollChar, l, \
            ("LSPB" if kernel["ProblemType"]["TLUB"] else "LSCB"), \
            self.endLine)

    ####################################
    # global read: branches
    if kernel["EdgeType"] != "None":
      kStr += self.endLine
      kStr += "  /* don't read out-of-bounds global a */%s" % self.endLine
      for l in range(0, numReadsTileA):
        gro = "(globalReadOffsetA%s_%u%s)" % (tileCharA, l, \
            ("_s0 + (VECTOR_WIDTH-1)" if readTileDimComponentsA else "") )
        limit = "size%s" % (tileCharA)

        if kernel["EdgeType"] == "Branch":
          kStr += "  bool inBoundsA_%u = %s < %s;%s" \
              % (l, gro, \
              limit, self.endLine)
        else:
          kStr += "  %s = (%s > %s) ? %s : %s;%s" \
              % (gro, gro, limit, limit, gro, self.endLine)

      kStr += self.endLine
      kStr += "  /* don't read out-of-bounds global b */%s" % self.endLine
      for l in range(0, numReadsTileB):
        gro = "(globalReadOffsetB%s_%u%s)" % (tileCharB, l, \
            ("_s0 + (VECTOR_WIDTH-1)" if readTileDimComponentsB else ""))
        limit = "size%s" % tileCharB

        if kernel["EdgeType"] == "Branch":
          kStr += "  bool inBoundsB_%u = %s < %s;%s" \
              % (l, gro, \
              limit, self.endLine)
        else:
          kStr += "  %s = (%s > %s) ? %s : %s;%s" \
              % (gro, gro, limit, limit, gro, self.endLine)

    ####################################
    # global read: offsets final a
    kStr += self.endLine
    kStr += "  /* global read: final offsets a */" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, numReadVectorComponentsA):
          kStr += "  %s globalReadOffsetA_%u_%u%s = GLOBAL_OFFSET_A( " \
              % (self.uint64Str, para, perp, \
              (("_s%u"%s) if (readTileDimComponentsA \
              or readUnrollDimComponentsA) else ""))
          for i in range(0, len(kernel["ProblemType"]["IndexAssignmentsA"])):
            index = kernel["ProblemType"]["IndexAssignmentsA"][i]
            if index < kernel["ProblemType"]["NumIndicesC"]:
              if index == kernel["ProblemType"]["TileA"]:
                kStr += "globalReadOffsetA%s_%u%s" \
                    % (tileCharA, \
                    (para if kernel["ProblemType"]["TLUA"] else perp), \
                    (("_s%u"%s) if readTileDimComponentsA else "") )
              else: # just a group index
                kStr += "wg" + indexChars[index]
            else: # summation index
              if index == kernel["ProblemType"]["IndexUnroll"]:
                kStr += "globalReadOffsetA%s_%u%s" \
                    % (unrollChar, \
                    (perp if kernel["ProblemType"]["TLUA"] else para), \
                    (("_s%u"%s) if readUnrollDimComponentsA else "") )
              else:
                kStr += "globalReadOffsetA%s" % indexChars[index]
            if i < len(kernel["ProblemType"]["IndexAssignmentsA"])-1:
              kStr += ", "
          kStr += " );%s" % self.endLine
          """
          kStr += "  printf(\\\"GRA T[%%02u] gROA_%u_%u%s = %%4u\\\\n\\\", serial, globalReadOffsetA_%u_%u%s);%s" \
              % (para, perp, \
              (("_s%u"%s) if (readTileDimComponentsA \
              or readUnrollDimComponentsA) else ""), \
              para, perp, \
              (("_s%u"%s) if (readTileDimComponentsA \
              or readUnrollDimComponentsA) else ""), \
              self.endLine )
          """

    ####################################
    # global read: offsets final b
    kStr += self.endLine
    kStr += "  /* global read: final offsets b */" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, numReadVectorComponentsB):
          kStr += "  %s globalReadOffsetB_%u_%u%s = GLOBAL_OFFSET_B( " \
              % (self.uint64Str, para, perp, \
              (("_s%u"%s) if (readTileDimComponentsB \
              or readUnrollDimComponentsB) else ""))
          for i in range(0, len(kernel["ProblemType"]["IndexAssignmentsB"])):
            index = kernel["ProblemType"]["IndexAssignmentsB"][i]
            if index < kernel["ProblemType"]["NumIndicesC"]:
              if index == kernel["ProblemType"]["TileB"]:
                kStr += "globalReadOffsetB%s_%u%s" \
                    % (tileCharB, \
                    (para if kernel["ProblemType"]["TLUB"] else perp), \
                    (("_s%u"%s) if readTileDimComponentsB else "") )
              else: # just a group index
                kStr += "wg" + indexChars[index]
            else: # summation index
              if index == kernel["ProblemType"]["IndexUnroll"]:
                kStr += "globalReadOffsetB%s_%u%s" \
                    % (unrollChar, \
                    (perp if kernel["ProblemType"]["TLUB"] else para), \
                    (("_s%u"%s) if readUnrollDimComponentsB else "") )
              else:
                kStr += "globalReadOffsetB%s" % indexChars[index]
            if i < len(kernel["ProblemType"]["IndexAssignmentsB"])-1:
              kStr += ", "
          kStr += " );%s" % self.endLine
          """
          kStr += "  printf(\\\"GRB T[%%02u] gROB_%u_%u%s = %%4u\\\\n\\\", serial, globalReadOffsetB_%u_%u%s);%s" \
              % (para, perp, \
              (("_s%u"%s) if (readTileDimComponentsB \
              or readUnrollDimComponentsB) else ""), \
              para, perp, \
              (("_s%u"%s) if (readTileDimComponentsB \
              or readUnrollDimComponentsB) else ""), \
              self.endLine )
          """

    ####################################
    # global read: addresses a
    kStr += self.endLine
    kStr += "  /* global read: addresses a */" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        if readTileDimComponentsA or readUnrollDimComponentsA:
          for s in range(0, numReadVectorComponentsA):
            kStr += "  %sDATA_TYPE const *globalReadA_%u_%u%s = A + globalReadOffsetA_%u_%u%s;%s" \
                % (self.globalPtrStr, para, perp, \
                (("_s%u"%s) if (readTileDimComponentsA \
                or readUnrollDimComponentsA) else ""), \
                para, perp, \
                (("_s%u"%s) if (readTileDimComponentsA \
                or readUnrollDimComponentsA) else ""), \
                self.endLine)
        else:
            kStr += "  %sVECTOR_TYPE const *globalReadA_%u_%u = (%sVECTOR_TYPE const *)(A + globalReadOffsetA_%u_%u);%s" \
                % (self.globalPtrStr, para, perp, self.globalPtrStr, \
                para, perp, self.endLine)

    ####################################
    # global read: addresses b
    kStr += self.endLine
    kStr += "  /* global read: addresses b */" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        if readTileDimComponentsB or readUnrollDimComponentsB:
          for s in range(0, numReadVectorComponentsB):
            kStr += "  %sDATA_TYPE const *globalReadB_%u_%u%s = B + globalReadOffsetB_%u_%u%s;%s" \
                % (self.globalPtrStr, para, perp, \
                (("_s%u"%s) if (readTileDimComponentsB \
                or readUnrollDimComponentsB) else ""), \
                para, perp, \
                (("_s%u"%s) if (readTileDimComponentsB \
                or readUnrollDimComponentsB) else ""), self.endLine)
        else:
            kStr += "  %sVECTOR_TYPE const *globalReadB_%u_%u = (%sVECTOR_TYPE const *)(B + globalReadOffsetB_%u_%u);%s" \
                % (self.globalPtrStr, para, perp, self.globalPtrStr, \
                para, perp, self.endLine)


    ####################################
    # LDS Write Addresses
    ####################################
    kStr += self.endLine
    kStr += "  /***************************************/" + self.endLine
    kStr += "  /* LDS Write Addresses                 */" + self.endLine
    kStr += "  /***************************************/" + self.endLine

    # free index assignment a
    kStr += "  unsigned int lwA%s = (serial%s" \
        % (tileCharA, ("%" if globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "/") )
    if globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")";
    if kernel["GlobalReadCoalesceVectorA"] == kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine

    # free index assignment b
    kStr += "  unsigned int lwB%s = (serial%s" \
        % (tileCharB, ("%" if globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "/") )
    if globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] == kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine

    # summation index assignment a
    kStr += "  unsigned int lwA%s = (serial%s" \
        % (unrollChar, ("/" if globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "%") )
    if globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")";
    if kernel["GlobalReadCoalesceVectorA"] != kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine

    # summation index assignment b
    kStr += "  unsigned int lwB%s = (serial%s" \
        % (unrollChar, ("/" if globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "%") )
    if globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] != kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine

    kStr += self.endLine
    kStr += "  /* lds write initial offsets */" + self.endLine
    kStr += "  unsigned int ldsWriteOffsetInitialA = lwA%s + lwA%s*(MT%s+PAD);%s" \
        % (tileCharA, unrollChar, tileCharA, self.endLine)
    kStr += "  unsigned int ldsWriteOffsetInitialB = lwB%s + lwB%s*(MT%s+PAD) + LDS_OFFSET_B;%s" \
        % (tileCharB, unrollChar, tileCharB, self.endLine)

    ####################################
    # lds write offsets a
    kStr += self.endLine
    kStr += "  /* lds write offsets */" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, numWriteVectorComponentsA):
          kStr += "  unsigned int ldsWriteOffsetA_%u_%u%s = ldsWriteOffsetInitialA + (%s%d*%s)" \
              % (para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsA \
              or writeUnrollDimComponentsA) else ""), \
              (("%u + "%s) if writeTileDimComponentsA else ""), \
              para, ("LSCA" if not kernel["ProblemType"]["TLUA"] else "LSCA") )
          if not kernel["ProblemType"]["TLUA"]:
            kStr += "*(MT%s+PAD)" % (tileCharA)
          kStr += " + (%s%d*%s)" % (
              (("%u + "%s) if writeUnrollDimComponentsA else ""), perp, \
              ("LSPA" if kernel["ProblemType"]["TLUA"] else "LSPA") )
          if kernel["ProblemType"]["TLUA"]:
            kStr += "*(MT%s+PAD)" % (tileCharA)
          kStr += ";%s" % self.endLine
          """
          kStr += "  printf(\\\"LWA T[%%02u] lWOA_%u_%u%s = %%4u\\\\n\\\", serial, ldsWriteOffsetA_%u_%u%s);%s" \
              % (para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsA \
              or writeUnrollDimComponentsA) else ""), \
              para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsA \
              or writeUnrollDimComponentsA) else ""), \
              self.endLine )
          """

    ####################################
    # lds write offsets b
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, numWriteVectorComponentsB):
          kStr += "  unsigned int ldsWriteOffsetB_%u_%u%s = ldsWriteOffsetInitialB + (%s%d*%s)" \
              % (para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsB \
              or writeUnrollDimComponentsB) else ""), \
              (("%u + "%s) if writeTileDimComponentsB else ""), para, \
              ("LSCB" if not kernel["ProblemType"]["TLUB"] else "LSCB") )
          if not kernel["ProblemType"]["TLUB"]:
            kStr += "*(MT%s+PAD)" % (tileCharB)
          kStr += " + (%s%d*%s)" % ( \
              (("%u + "%s) if writeUnrollDimComponentsB else ""), perp, \
              ("LSPB" if not kernel["ProblemType"]["TLUB"] else "LSPB") )
          if kernel["ProblemType"]["TLUB"]:
            kStr += "*(MT%s+PAD)" % (tileCharB)
          kStr += ";%s" % self.endLine
          """
          kStr += "  printf(\\\"LWB T[%%02u] lWOB_%u_%u%s = %%4u\\\\n\\\", serial, ldsWriteOffsetB_%u_%u%s);%s" \
             % (para, perp,
              (("_s%u"%s) if (writeTileDimComponentsB \
              or writeUnrollDimComponentsB) else ""), \
              para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsB \
              or writeUnrollDimComponentsB) else ""), \
              self.endLine )
          """


    ####################################
    # lds write addresses
    kStr += self.endLine
    kStr += "  /* lds write addresses */" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, numWriteVectorComponentsA):
          kStr += "  %s%s *ldsWriteA_%u_%u%s = (%s%s *)(lds + ldsWriteOffsetA_%u_%u%s);%s"\
              % (self.sharedPtrStr, ("DATA_TYPE" if (writeTileDimComponentsA \
              or writeUnrollDimComponentsA) else "VECTOR_TYPE"), para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsA \
              or writeUnrollDimComponentsA) else ""), \
              self.sharedPtrStr, ("DATA_TYPE" if (writeTileDimComponentsA \
              or writeUnrollDimComponentsA) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsA \
              or writeUnrollDimComponentsA) else ""), \
              self.endLine)
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, numWriteVectorComponentsB):
          kStr += "  %s%s *ldsWriteB_%u_%u%s = (%s%s *)(lds + ldsWriteOffsetB_%u_%u%s);%s"\
              % (self.sharedPtrStr, ("DATA_TYPE" if (writeTileDimComponentsB \
              or writeUnrollDimComponentsB) else "VECTOR_TYPE"), para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsB \
              or writeUnrollDimComponentsB) else ""), \
              self.sharedPtrStr, ("DATA_TYPE" if (writeTileDimComponentsB \
              or writeUnrollDimComponentsB) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsB \
              or writeUnrollDimComponentsB) else ""), \
              self.endLine)

    ####################################
    # LDS Read Addresses
    ####################################
    kStr += self.endLine
    kStr += "  /***************************************/" + self.endLine
    kStr += "  /* LDS Read Addresses                  */" + self.endLine
    kStr += "  /***************************************/" + self.endLine
    kStr += "  unsigned int lr%s = (serial %% SG%s);%s" \
        % (tileChar0, tileChar0, self.endLine)
    kStr += "  unsigned int lr%s = (serial / SG%s) %% SG%s;%s" \
        % (tileChar1, tileChar0, tileChar1, self.endLine)
    kStr += "  %sVECTOR_TYPE *ldsReadA = (%sVECTOR_TYPE *)(lds + lr%s*VECTOR_WIDTH + sgId*(MT%s+PAD));%s" \
          % (self.sharedPtrStr, self.sharedPtrStr, \
          tileChar0, tileChar0, self.endLine)
    kStr += "  %sVECTOR_TYPE *ldsReadB = (%sVECTOR_TYPE *)(lds + lr%s*VECTOR_WIDTH + sgId*(MT%s+PAD) + LDS_OFFSET_B);%s"\
          % (self.sharedPtrStr, self.sharedPtrStr, \
          tileChar1, tileChar1, self.endLine)
    kStr += "  %sVECTOR_TYPE *ldsReadIterA = ldsReadA;%s" \
          % (self.sharedPtrStr, self.endLine)
    kStr += "  %sVECTOR_TYPE *ldsReadIterB = ldsReadB;%s"\
          % (self.sharedPtrStr, self.endLine)
    kStr += self.endLine



    ###########################################################################
    # summations loops
    ###########################################################################
    indent = "  "
    kStr += self.endLine
    kStr += "%s/****************************************/%s" \
        % (indent, self.endLine)
    kStr += "%s/* summation loops(s)                   */%s" \
        % (indent, self.endLine)
    kStr += "%s/****************************************/%s" \
        % (indent, self.endLine)
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]):
      loopChar = indexChars[kernel["ProblemType"]["IndicesSummation"][i]]
      kStr += indent + "unsigned int sumIter" + loopChar \
          + " = size" + loopChar
      if i == kernel["ProblemType"]["NumIndicesSummation"]-1:
        kStr += " / DEPTHU"
      kStr += ";" + self.endLine
      if kernel["LoopDoWhile"]:
        kStr += indent + "do {" + self.endLine
      else:
        kStr += indent + "while (sumIter%s-- > 0) {%s" \
            % (loopChar, self.endLine)
      indent += "  "

    ####################################
    # global read A
    kStr += self.endLine
    kStr += indent + "/* read global */" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, numReadVectorComponentsA):
          kStr += "%sa_%u_%u%s = " % (indent, para, perp, \
              ((".%s"%self.vectorComponents[s]) if (readTileDimComponentsA \
              or readUnrollDimComponentsA) else "") )
          if kernel["EdgeType"] == "Branch":
            kStr += "( !inBoundsA_%u )" % ( \
                (para if kernel["ProblemType"]["TLUA"] else perp) )
            kStr += " ? %s : " % ( \
                kernel["ProblemType"]["DataType"].zeroString(self.language) )
          kStr += "*globalReadA_%u_%u%s;%s" \
              % (para, perp, \
              (("_s%u"%s) if (readTileDimComponentsA \
              or readUnrollDimComponentsA) else "" ), self.endLine)

    ####################################
    # global read B
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, numReadVectorComponentsB):
          kStr += "%sb_%u_%u%s = " % (indent, para, perp, \
              ((".%s"%self.vectorComponents[s]) if (readTileDimComponentsB \
              or readUnrollDimComponentsB) else "") )
          if kernel["EdgeType"] == "Branch":
            kStr += "( !inBoundsB_%u )" % ( \
                (para if kernel["ProblemType"]["TLUB"] else perp) )
            kStr += " ? %s : " \
                % kernel["ProblemType"]["DataType"].zeroString(self.language)
          kStr += "*globalReadB_%u_%u%s;%s" \
              % (para, perp, \
              (("_s%u"%s) if (readTileDimComponentsB \
              or readUnrollDimComponentsB) else "" ), self.endLine)

    ########################################
    # lds write a
    kStr += self.endLine
    kStr += indent + "/* lds write */" + self.endLine
    kStr += indent + self.syncStr + self.endLine
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic push" + self.endLine
      kStr += "#pragma clang diagnostic ignored \"-Wconditional-uninitialized\"" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, numWriteVectorComponentsA):
          kStr += "%s*ldsWriteA_%u_%u%s = a_%u_%u%s;%s" \
              % (indent, para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsA \
              or writeUnrollDimComponentsA) else "" ), \
              para, perp, \
              ((".%s"%self.vectorComponents[s]) if (writeTileDimComponentsA \
              or writeUnrollDimComponentsA) else "" ), \
              self.endLine)

    ########################################
    # lds write b
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, numWriteVectorComponentsB):
          kStr += "%s*ldsWriteB_%u_%u%s = b_%u_%u%s;%s" \
              % (indent, para, perp, \
              (("_s%u"%s) if (writeTileDimComponentsB \
              or writeUnrollDimComponentsB) else "" ), \
              para, perp, \
              ((".%s"%self.vectorComponents[s]) if (writeTileDimComponentsB \
              or writeUnrollDimComponentsB) else "" ), \
              self.endLine)
    kStr += indent + self.syncStr + self.endLine

    if self.language == "HIP":
      kStr += "#pragma clang diagnostic pop" + self.endLine

    # re-init lds read addresses
    kStr += self.endLine
    kStr += "%s/* re-init lds read addresses */%s" % (indent, self.endLine)
    kStr += "%sldsReadIterA = ldsReadA;%s" \
          % (indent, self.endLine)
    kStr += "%sldsReadIterB = ldsReadB;%s"\
          % (indent, self.endLine)

    # debug LDS state
    if False:
      kStr += "    /* print LDS state */" + self.endLine
      kStr += "    for (unsigned int i = serial; i < LDS_NUM_ELEMENTS; i+=NUM_THREADS) {%s" % self.endLine
      kStr += "      printf(\\\"lds[%%06u] = %%.0f\\\\n\\\", i, lds[i]);%s" \
          % self.endLine
      kStr += "    }" + self.endLine


    ####################################
    # do macs
    kStr += self.endLine
    kStr += indent + "/* do macs */" + self.endLine
    for u in range(0, kernel["LoopUnroll"]):
      kStr += indent + "SUMMATION_UNROLL" + self.endLine
    kStr += self.endLine


    ####################################
    # if has tail loop, close unrolled loop
    if kernel["LoopTail"]:
      loopChar = \
          indexChars[ \
          kernel["ProblemType"]["IndicesSummation"][ \
          kernel["ProblemType"]["NumIndicesSummation"]-1]]

      ####################################
      # increment global read addresses a
      kStr += "%s/* increment global read addresses */%s" \
          % (indent, self.endLine)
      for perp in range(0, kernel["NumLoadsPerpendicularA"]):
        for para in range(0, kernel["NumLoadsCoalescedA"]):
          for s in range(0, numReadVectorComponentsA):
            kStr += "%sglobalReadA_%u_%u%s += (%s)strideA%s*DEPTHU%s;%s" \
                % (indent, para, perp, \
                (("_s%u"%s) if (readTileDimComponentsA \
                or readUnrollDimComponentsA) else ""), \
                self.uint64Str, loopChar, "" if (readTileDimComponentsA \
                or readUnrollDimComponentsA) else "/VECTOR_WIDTH", self.endLine)

      ####################################
      # increment global read addresses b
      for perp in range(0, kernel["NumLoadsPerpendicularB"]):
        for para in range(0, kernel["NumLoadsCoalescedB"]):
          for s in range(0, numReadVectorComponentsB):
            kStr += "%sglobalReadB_%u_%u%s += (%s)strideB%s*DEPTHU%s;%s"\
                % (indent, para, perp, \
                (("_s%u"%s) if (readTileDimComponentsB \
                or readUnrollDimComponentsB) else ""), \
                self.uint64Str, loopChar, "" if (readTileDimComponentsB \
                or readUnrollDimComponentsB) else "/VECTOR_WIDTH", self.endLine)

      indent = indent[2:]
      # close unrolled loop
      if kernel["LoopDoWhile"]:
        kStr += indent + "} while (--sumIter" + loopChar \
            + " > 0);" + self.endLine
      else:
        kStr += "%s}%s" % (indent, self.endLine)
      kStr += self.endLine

      ########################################
      # Tail Loop
      ########################################
      kStr += "  /***************************************/" + self.endLine
      kStr += "  /* Tail Loop                           */" + self.endLine
      kStr += "  /***************************************/" + self.endLine

      ####################################
      # Tail: global read A
      ####################################
      kStr += self.endLine
      kStr += "%s/* global read */%s" % (indent, self.endLine)
      for perp in range(0, kernel["NumLoadsPerpendicularA"]):
        for para in range(0, kernel["NumLoadsCoalescedA"]):
          for s in range(0, numReadVectorComponentsA):
            kStr += "%sa_%u_%u%s = " % (indent, para, perp, \
                ((".%s"%self.vectorComponents[s]) if (readTileDimComponentsA \
                or readUnrollDimComponentsA) else "") )
            # guard around K
            kStr += "( globalReadOffsetA%s_%u%s >= (size%s %% DEPTHU) )" \
                % (unrollChar, \
                (perp if kernel["ProblemType"]["TLUA"] else para), \
                (("_s%u"%s) if readUnrollDimComponentsA else ""), unrollChar)
            # guard around edge
            if kernel["EdgeType"] == "Branch":
              kStr += " || "
              kStr += "( !inBoundsA_%u )" % ( \
                  (para if kernel["ProblemType"]["TLUA"] else perp) )
            kStr += " ? %s : " % \
                kernel["ProblemType"]["DataType"].zeroString(self.language)
            kStr += "*globalReadA_%u_%u%s;%s" % (para, perp, \
                (("_s%u"%s) if (readTileDimComponentsA \
                or readUnrollDimComponentsA) else ""), self.endLine)

      ####################################
      # Tail: global read B
      ####################################
      for perp in range(0, kernel["NumLoadsPerpendicularB"]):
        for para in range(0, kernel["NumLoadsCoalescedB"]):
          for s in range(0, numReadVectorComponentsB):
            kStr += "%sb_%u_%u%s = " % (indent, para, perp, \
                ((".%s"%self.vectorComponents[s]) if (readTileDimComponentsB \
                or readUnrollDimComponentsB) \
                else "") )
            # guard around k
            kStr += "( globalReadOffsetB%s_%u%s >= (size%s %% DEPTHU) )" \
                % (unrollChar, \
                (perp if kernel["ProblemType"]["TLUB"] else para), \
                (("_s%u"%s) if readUnrollDimComponentsB else ""), unrollChar)
            # guard around edge
            if kernel["EdgeType"] == "Branch":
              kStr += " || "
              kStr += "( !inBoundsB_%u )" % ( \
                  (para if kernel["ProblemType"]["TLUB"] else perp) )
            kStr += " ? %s : " % \
                kernel["ProblemType"]["DataType"].zeroString(self.language)
            kStr += "*globalReadB_%u_%u%s;%s" \
                % (para, perp, \
                (("_s%u"%s) if (readTileDimComponentsB \
                or readUnrollDimComponentsB) else ""), self.endLine)


      ########################################
      # Tail: lds write a
      kStr += self.endLine
      kStr += indent + "/* lds write */" + self.endLine
      kStr += indent + self.syncStr + self.endLine
      if self.language == "HIP":
        kStr += "#pragma clang diagnostic push" + self.endLine
        kStr += "#pragma clang diagnostic ignored \"-Wconditional-uninitialized\"" + self.endLine
      for perp in range(0, kernel["NumLoadsPerpendicularA"]):
        for para in range(0, kernel["NumLoadsCoalescedA"]):
          for s in range(0, numWriteVectorComponentsA):
            kStr += "%s*ldsWriteA_%u_%u%s = a_%u_%u%s;%s" \
                % (indent, para, perp, \
                (("_s%u"%s) if (writeTileDimComponentsA \
                or writeUnrollDimComponentsA) else "" ), \
                para, perp, \
                ((".%s"%self.vectorComponents[s]) if (writeTileDimComponentsA \
                or writeUnrollDimComponentsA) else "" ), \
                self.endLine)

      ########################################
      # Tail: lds write b
      for perp in range(0, kernel["NumLoadsPerpendicularB"]):
        for para in range(0, kernel["NumLoadsCoalescedB"]):
          for s in range(0, numWriteVectorComponentsB):
            kStr += "%s*ldsWriteB_%u_%u%s = b_%u_%u%s;%s" \
                % (indent, para, perp, \
                (("_s%u"%s) if (writeTileDimComponentsB \
                or writeUnrollDimComponentsB) else "" ), \
                para, perp, \
                ((".%s"%self.vectorComponents[s]) if (writeTileDimComponentsB \
                or writeUnrollDimComponentsB) else "" ), \
                self.endLine)

      kStr += indent + self.syncStr + self.endLine
      if self.language == "HIP":
        kStr += "#pragma clang diagnostic pop" + self.endLine

      # full end loop b/c local full of zeros
      kStr += self.endLine
      kStr += indent + "/* re-init lds read addresses */" + self.endLine
      kStr += "%sldsReadIterA = ldsReadA;%s" \
            % (indent, self.endLine)
      kStr += "%sldsReadIterB = ldsReadB;%s"\
            % (indent, self.endLine)

      kStr += self.endLine

      # begin loop
      kStr += "%ssumIter%s = (((size%s %% DEPTHU) + SPLITU - 1) / SPLITU);%s" % (indent, loopChar, loopChar, self.endLine)

      if kernel["LoopDoWhile"]:
        kStr += indent + "do {" + self.endLine
      else:
        kStr += indent + "while (sumIter%s-- > 0) {%s" \
            % (loopChar, self.endLine)

      indent += "  "

      ####################################
      # Tail: do macs
      kStr += self.endLine
      kStr += indent + "/* do macs */" + self.endLine
      kStr += indent + "SUMMATION_UNROLL" + self.endLine
      kStr += self.endLine

    ########################################################################
    # END UNROLL=1 LOOP
    ########################################################################


    ####################################
    # end loop
    for i in reversed(range(0,kernel["ProblemType"]["NumIndicesSummation"])):
      loopChar = indexChars[kernel["ProblemType"]["IndicesSummation"][i]]

      kStr += "%s/* increment global read addresses */%s" % (indent, self.endLine)
      ####################################
      # global read addresses a
      for perp in range(0, kernel["NumLoadsPerpendicularA"]):
        for para in range(0, kernel["NumLoadsCoalescedA"]):
          for s in range(0, numReadVectorComponentsA):
            kStr += "%sglobalReadA_%u_%u%s += (%s) (strideA%s%s)" \
                % (indent, para, perp, \
                (("_s%u"%s) if (readTileDimComponentsA \
                or readUnrollDimComponentsA) else ""), \
                self.int64Str, loopChar, "" if (readTileDimComponentsA \
                or readUnrollDimComponentsA) else "/VECTOR_WIDTH")
            if i==kernel["ProblemType"]["NumIndicesSummation"]-1:
              kStr += "*DEPTHU"
            else:
              for j in range(i+1, \
                  min(i+2, kernel["ProblemType"]["NumIndicesSummation"]) ):
                tmpChar = \
                    indexChars[kernel["ProblemType"]["IndicesSummation"][j]]
                kStr += " - strideA" + tmpChar + "*size" + tmpChar
            kStr += ";" + self.endLine

      ####################################
      # global read addresses b
      for perp in range(0, kernel["NumLoadsPerpendicularB"]):
        for para in range(0, kernel["NumLoadsCoalescedB"]):
          for s in range(0, numReadVectorComponentsB):
            kStr += "%sglobalReadB_%u_%u%s += (%s) (strideB%s%s)" \
                % (indent, para, perp, \
                (("_s%u"%s) if (readTileDimComponentsB \
                or readUnrollDimComponentsB) else ""), \
                self.int64Str, loopChar, "" if (readTileDimComponentsB \
                or readUnrollDimComponentsB) else "/VECTOR_WIDTH")
            if i==kernel["ProblemType"]["NumIndicesSummation"]-1:
              kStr += "*DEPTHU"
            else:
              for j in range(i+1,min(i+2, \
                  kernel["ProblemType"]["NumIndicesSummation"]) ):
                tmpChar = \
                    indexChars[kernel["ProblemType"]["IndicesSummation"][j]]
                kStr += " - strideB" + tmpChar + "*size" + tmpChar
            kStr += ";" + self.endLine



    ########################################
    # close loop
    indent = indent[2:]
    if kernel["LoopDoWhile"]:
      kStr += indent + "} while (--sumIter" + loopChar \
          + " > 0);" + self.endLine
    else:
      kStr += "%s}%s" % (indent, self.endLine)
    kStr += self.endLine

    ####################################
    # SplitU reduction
    ####################################
    if kernel["SplitU"] > 1:
      kStr += "  /***************************************/" + self.endLine
      kStr += "  /* SplitU Reduction                    */" + self.endLine
      kStr += "  /***************************************/" + self.endLine
      kStr += "  " + self.syncStr + self.endLine

      kStr += "  %sVECTOR_TYPE *ldsSplitU = (%sVECTOR_TYPE *)(lds);%s" \
          % (self.sharedPtrStr, self.sharedPtrStr, self.endLine)

      ####################################
      # SplitU: write to lds
      for j in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
        for i in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
          for s in range(0, kernel["VectorWidth"]):
            kStr += "  ldsSplitU[lr%s + %u*SG%s + (MT%s/VECTOR_WIDTH)*(lr%s*VECTOR_WIDTH + %u + SG%s*VECTOR_WIDTH*%u) + (MT%s*MT%s/VECTOR_WIDTH)*sgId] = rC[%u+%u*(TT%s/VECTOR_WIDTH)+%u*TT%s];%s" \
                % (tileChar0, i, tileChar0, tileChar0, tileChar1, \
                s, tileChar1, j, tileChar0, tileChar1, i, s, \
                tileChar0, j, tileChar0, self.endLine)
      kStr += "  " + self.syncStr + self.endLine + self.endLine

      ####################################
      # SplitU: C elements to store
      kStr += "  /* SplitU: new C elements to store */" + self.endLine
      for i in range(0, kernel["NumVectorsPerThread"]):
        kStr += "  rC[%3u] = ldsSplitU[serial+%u*NUM_THREADS];%s" \
            % (i, i, self.endLine)
      kStr += self.endLine

      ####################################
      # SplitU: reduction
      kStr += "  /* SplitU: reduction */" + self.endLine
      for s in range(1, kernel["SplitU"]):
        for i in range(0, kernel["NumVectorsPerThread"]):
          kStr += "  rC[%3u] += ldsSplitU[serial+%u*NUM_THREADS + %u*(MT%s*MT%s/VECTOR_WIDTH)];%s" \
              % (i, i, s, tileChar0, tileChar1, self.endLine)
        kStr += self.endLine

      ####################################
      # SplitU: which global Cij index
      kStr += "  /* which global Cij index */%s" % self.endLine
      for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
        kStr += "  unsigned int globalC%s = wg%s" \
            % (indexChars[i], indexChars[i])
        if i == kernel["ProblemType"]["Index0"]:
          kStr += "*MT%s + (serial %% (MT%s/VECTOR_WIDTH))*VECTOR_WIDTH" \
              % (tileChar0, tileChar0)
        if i == kernel["ProblemType"]["Index1"]:
          kStr += "*MT%s + (serial / (MT%s/VECTOR_WIDTH))" \
              % (tileChar1, tileChar0)
        kStr += ";" + self.endLine
      kStr += self.endLine


      ####################################
      # SplitU: Global Write
      ####################################
      kStr += "  /***************************************/" + self.endLine
      kStr += "  /* Global Write                        */" + self.endLine
      kStr += "  /***************************************/" + self.endLine
      if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
        kStr += "  float type_mac_tmp;" + self.endLine
      if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
        kStr += "  double type_mac_tmp;" + self.endLine

      for b in range(0, kernel["NumVectorsPerThread"]):
        for s in range(0, kernel["VectorWidth"]):
          if kernel["EdgeType"] != "None":

            kStr += "  if (globalC%s < size%s) {" \
                % (tileChar0, \
                tileChar0)

            kStr += "  if (globalC%s + %u*CPS < size%s) {" \
                % (tileChar1, b, tileChar1)

          kStr += "  TYPE_MAC_WRITE( C[ GLOBAL_C( (%s)" % self.uint64Str
          for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
            kStr += " globalC%s" % indexChars[i]
            if i == kernel["ProblemType"]["Index0"] and kernel["VectorWidth"]>1:
              kStr += " + %u" %s
            if i == kernel["ProblemType"]["Index1"]:
              kStr += " + %u*CPS" %b
            if i < kernel["ProblemType"]["NumIndicesC"]-1:
              kStr += ", (%s)" % self.uint64Str
          kStr += ") ]"
          kStr += ", alpha"
          kStr += ", rC[%d]%s" % (b, \
              ((".%s"%self.vectorComponents[s]) if kernel["VectorWidth"]>1 \
              else "") )

          if kernel["ProblemType"]["UseBeta"]:
            kStr += ", beta"
          kStr += ")"

          if kernel["EdgeType"] != "None":
            kStr += "} }"
          kStr += self.endLine



      ####################################
      # SplitU==1, write data straight
      ####################################
    else:

      ####################################
      # Global Write Addresses
      ####################################
      kStr += "  /***************************************/" + self.endLine
      kStr += "  /* Global Write                        */" + self.endLine
      kStr += "  /***************************************/" + self.endLine
      for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
        kStr += "  unsigned int globalC" + indexChars[i] \
            + " = wg" + indexChars[i]
        if i == kernel["ProblemType"]["Index0"]:
          kStr += "*MT%s + (serial %% SG%s)*VECTOR_WIDTH" % (tileChar0, tileChar0)
        if i == kernel["ProblemType"]["Index1"]:
          kStr += "*MT%s + (serial / SG%s)*VECTOR_WIDTH" % (tileChar1, tileChar0)
        kStr += ";" + self.endLine
      kStr += self.endLine

      """
      for b in range(0, kernel["ThreadTile1"]):
        for a in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
          for s in range(0, kernel["VectorWidth"]):
            kStr += "printf(\\\"T[%%02u]: %%.0f\\\\n\\\", serial, rC[(%d+%d*TT%s/VECTOR_WIDTH)]%s);%s" % (a, b, \
                tileChar0, \
                ((".%s"%self.vectorComponents[s]) if kernel["VectorWidth"]>1 \
                else ""), self.endLine )
      """

      if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
        kStr += "  float type_mac_tmp;" + self.endLine
      if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
        kStr += "  double type_mac_tmp;" + self.endLine

      for b in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
        for a in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
          for s1 in range(0, kernel["VectorWidth"]):
            for s0 in range(0, kernel["VectorWidth"]):
              if kernel["EdgeType"] != "None":
                kStr += "  if (globalC%s + (VECTOR_WIDTH-1) + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                    % (tileChar0, \
                    a, tileChar0, tileChar0)
                kStr += "  if (globalC%s + (VECTOR_WIDTH-1) + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                    % (tileChar1, \
                    b, tileChar1, tileChar1)

              kStr += "  TYPE_MAC_WRITE( C[ GLOBAL_C( (%s)" % self.uint64Str
              for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
                kStr += " globalC%s" % indexChars[i]
                if i == kernel["ProblemType"]["Index0"]:
                  kStr += "%s + %u*SG%s*VECTOR_WIDTH" % (\
                      ((" + %u"%s0) if kernel["VectorWidth"]>1 else ""), \
                      a, tileChar0)
                if i == kernel["ProblemType"]["Index1"]:
                  kStr += "%s + %u*SG%s*VECTOR_WIDTH" % (\
                      ((" + %u"%s1) if kernel["VectorWidth"]>1 else ""), \
                      b, tileChar1)
                if i < kernel["ProblemType"]["NumIndicesC"]-1:
                  kStr += ", (%s)" % self.uint64Str
              kStr += ") ]"
              kStr += ", alpha"
              kStr += ", rC[%d+%d*(TT%s/VECTOR_WIDTH)+%d*TT%s]%s" \
                  % (a, s1, tileChar0, b, tileChar0, \
                  ((".%s"%self.vectorComponents[s0]) if kernel["VectorWidth"]>1\
                  else "") )
              if kernel["ProblemType"]["UseBeta"]:
                kStr += ", beta"
              kStr += ")"

              if kernel["EdgeType"] != "None":
                kStr += " } }"
              kStr += self.endLine


    ####################################
    # end kernel body
    kStr += self.endLine
    kStr += "}" + self.endLine

    ####################################
    # undefine definitions if merged
    if globalParameters["MergeFiles"] and self.language == "HIP":
      kStr += "#undef UNROLL%s" % self.endLine
      kStr += "#undef SPLITU%s" % self.endLine
      kStr += "#undef DEPTHU%s" % self.endLine
      kStr += "#undef SG%s%s" % (tileChar0, self.endLine)
      kStr += "#undef SG%s%s" % (tileChar1, self.endLine)
      kStr += "#undef TT%s%s" % (tileChar0, self.endLine)
      kStr += "#undef TT%s%s" % (tileChar1, self.endLine)
      kStr += "#undef MT%s%s" % (tileChar0, self.endLine)
      kStr += "#undef MT%s%s" % (tileChar1, self.endLine)
      kStr += "#undef NLCA%s" % (self.endLine )
      kStr += "#undef NLCB%s" % (self.endLine )
      kStr += "#undef NLPA%s" % (self.endLine )
      kStr += "#undef NLPB%s" % (self.endLine )
      kStr += "#undef LSCA%s" % (self.endLine)
      kStr += "#undef LSPA%s" % (self.endLine)
      kStr += "#undef LSCB%s" % (self.endLine)
      kStr += "#undef LSPB%s" % (self.endLine)
      kStr += "#undef GLOBAL_C%s" % (self.endLine)
      kStr += "#undef GLOBAL_OFFSET_A%s" % (self.endLine)
      kStr += "#undef GLOBAL_OFFSET_B%s" % (self.endLine)
      kStr += "#undef DATA_TYPE%s" % (self.endLine)
      kStr += "#undef VECTOR_TYPE%s" % (self.endLine)
      kStr += "#undef SUMMATION_UNROLL%s" % (self.endLine)
      kStr += "#undef LDS_OFFSET_B%s" % (self.endLine)
      kStr += "#undef LDS_NUM_ELEMENTS%s" % (self.endLine)
      kStr += "#undef NUM_THREADS%s" % (self.endLine)
      kStr += "#undef WORK_GROUP_MAPPING%s" % (self.endLine)
      kStr += "#undef VECTOR_WIDTH%s" % (self.endLine)
      firstStride = 0
      if kernel["ProblemType"]["UseInitialStrides"]:
        lastStrideC = 0
        lastStrideA = 0
        lastStrideB = 0
      else:
        lastStrideC = 1
        lastStrideA = 1
        lastStrideB = 1
      for i in range(firstStride, lastStrideC):
        kStr += "#undef strideC" + indexChars[i] + self.endLine
      for i in range(firstStride, lastStrideA):
        kStr += "#undef strideA" \
            + indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]] \
            + self.endLine
      for i in range(firstStride, lastStrideB):
        kStr += "#undef strideB" \
            + indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]] \
            + self.endLine
      kStr += self.endLine + self.endLine

    return kStr

  ##############################################################################
  # source file string
  ##############################################################################
  def getSourceFileString(self, kernel):
    kernelName = self.getKernelName(kernel)
    fileString = "" # CHeader
    if not globalParameters["MergeFiles"]:
      fileString += "\n"
      fileString += "#include \"" + kernelName + ".h\"\n"
      fileString += "\n"

    # language pre
    fileString += "\n"
    if self.language == "OCL":
      fileString += "const char * const %s_src =\"" % (kernelName)

    # write kernel body
    fileString += self.getBody( kernel )

    # language post
    if self.language == "OCL":
      fileString += "\";\n"

    fileString  += "/* Kernel Parameters\n"
    fileString  += Solution.getParametersIndented(kernel, "  ")
    fileString  += "*/\n"
    fileString  += "\n"

    fileString += "\n"
    return fileString


  ##############################################################################
  # header file string
  ##############################################################################
  def getHeaderFileString(self, kernel):
    kernelName = self.getKernelName(kernel)
    fileString = "" # CHeader
    if not globalParameters["MergeFiles"]:
      fileString += "#pragma once\n\n"
      fileString += "\n"
      if self.language == "HIP":
        fileString += "#include <hip/hip_runtime.h>\n"
        fileString += "\n"
    if self.language == "OCL":
      fileString += "extern const char * const %s_src;\n" % kernelName
    else:
      fileString += self.getSignature(kernel)
      fileString += ";\n"

    return fileString



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
    self.backend = globalParameters["Backend"]
    self.kernelMinNaming = kernelMinNaming
    self.kernelSerialNaming = kernelSerialNaming

    if self.backend == "OCL":
      # everything escaped extra b/c string
      self.endLine = "\\n\"\n\""
      self.endLinePP = "\\\\" + self.endLine
    else:
      self.endLine = "\n"
      self.endLinePP =  "\\" + self.endLine

    if self.backend == "OCL":
      self.getGroupIdStr = "get_group_id"
      self.getNumGroupsStr = "get_num_groups"
      self.getLocalIdStr = "get_local_id"
      self.getGlobalIdStr = "get_global_id"
      self.sharedDeclStr = "__local "
      self.sharedPtrStr = "__local "
      self.syncStr = "barrier(CLK_LOCAL_MEM_FENCE);"
      self.fenceStr = "mem_fence(CLK_LOCAL_MEM_FENCE);"
      self.macFStr = "mad"
      self.macDStr = "mad"
      self.int64Str = "long"
      self.uint64Str = "unsigned long"
    else:
      self.getGroupIdStr = "hc_get_group_id"
      self.getNumGroupsStr = "hc_get_num_groups"
      self.getLocalIdStr = "hc_get_workitem_id"
      self.getGlobalIdStr = "hc_get_workitem_absolute_id"
      self.sharedDeclStr = "__shared__ "
      self.sharedPtrStr = ""
      self.syncStr = "__syncthreads();"
      self.fenceStr = self.syncStr
      self.macFStr = "fmaf"
      self.macDStr = "fma"
      self.int64Str = "int64_t"
      self.uint64Str = "uint64_t"

    self.returnOnly = False

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
    if self.backend == "OCL":
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
    if self.backend == "HIP":
      s += "  hipLaunchParm lp," + self.endLine
      globalStr = ""
    restrictStr = "restrict"
    if self.backend == "HIP":
      restrictStr = "__restrict__"
    s += "  " + globalStr + kernel["ProblemType"]["DataType"].toDevice(self.backend) \
        + "       *          C,"
    s += self.endLine
    s += "  " + globalStr + kernel["ProblemType"]["DataType"].toDevice(self.backend) \
        + " const * " + restrictStr + " A,"
    s += self.endLine
    s += "  " + globalStr + kernel["ProblemType"]["DataType"].toDevice(self.backend) \
        + " const * " + restrictStr + " B"

    # alpha & beta
    s += "," + self.endLine + "  " \
        + kernel["ProblemType"]["DataType"].toDevice(self.backend) + " const alpha"
    if kernel["ProblemType"]["UseBeta"]:
      s += "," + self.endLine + "  " \
          + kernel["ProblemType"]["DataType"].toDevice(self.backend) + " const beta"

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
    kStr += "/* %s */" % kernelName
    kStr += self.endLine

    ####################################
    # kernel preprocessor definitions
    kStr += self.endLine
    kStr += "/* tile parameters */" + self.endLine
    kStr += "#define NUM_THREADS %3d%s" \
        % (kernel["NumThreads"], self.endLine )
    kStr += "#define SPLITU %d%s" \
        % (kernel["SplitU"], self.endLine )
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
    kStr += "#define UNROLL %d%s" \
        % (kernel["LoopUnroll"], self.endLine )
    kStr += "#define DEPTHU (SPLITU*UNROLL)%s" % (self.endLine )
    kStr += "#define PAD %u%s" % (kernel["LdsPad"], self.endLine)
    kStr += "// columns per store%s" % self.endLine
    kStr += "#define CPS (NUM_THREADS / MT%s)%s" % (tileChar0, self.endLine)
    kStr += "#define WORK_GROUP_MAPPING %u%s" % (abs(kernel["WorkGroupMapping"]), self.endLine)
    kStr += self.endLine

    ####################################
    # load grid

    #totalLoadsA  = (kernel["SubGroup0"]*kernel["ThreadTile0"] \
    #    *kernel["LoopUnroll"]*kernel["SplitU"]) \
    #    / (kernel["SubGroup0"]*kernel["SubGroup1"])
    #totalLoadsB  = (kernel["SubGroup1"]*kernel["ThreadTile1"] \
    #    *kernel["LoopUnroll"]*kernel["SplitU"]) \
    #    / (kernel["SubGroup0"]*kernel["SubGroup1"])
    #numLoadsParaA = kernel["NumLoadsCoalescedA"]
    #numLoadsParaB = kernel["NumLoadsCoalescedB"]


    # num loads
    kStr += "/* num loads parallel and perpendicular to coalesced dimension */" + self.endLine
    kStr += "#define NLCA %d%s" % (kernel["NumLoadsCoalescedA"], self.endLine )
    kStr += "#define NLCB %d%s" % (kernel["NumLoadsCoalescedB"], \
        self.endLine )

    kStr += "#define NLPA %d%s" % (kernel["NumLoadsPerpendicularA"], \
        self.endLine )
    kStr += "#define NLPB %d%s" % (kernel["NumLoadsPerpendicularB"], \
        self.endLine )
    kStr += self.endLine

    # load size
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

    # lds buffer size
    #ldsAlign = 256 / kernel["ProblemType"]["DataType"].numRegisters()
    #ldsSizeA = kernel["DepthU"]*(kernel["MacroTile0"]+kernel["LdsPad"])
    #ldsSizeB = kernel["DepthU"]*(kernel["MacroTile1"]+kernel["LdsPad"])
    #ldsSizeAlignedA = ((ldsSizeA+ldsAlign-1)/ldsAlign)*ldsAlign
    #ldsSizeAlignedB = ((ldsSizeB+ldsAlign-1)/ldsAlign)*ldsAlign
    #ldsSizeReduction = 0 if (kernel["SplitU"] == 1) \
    #    else (kernel["MacroTile0"]*kernel["MacroTile1"])
    #ldsSize = max(ldsSizeAlignedA+ldsSizeB, ldsSizeReduction)
    kStr += "#define LDS_OFFSET_B %u%s" % (kernel["LdsOffsetB"], self.endLine)
    kStr += "#define LDS_NUM_ELEMENTS %u%s" % (kernel["LdsNumElements"], \
        self.endLine)


    ####################################
    # global memory indices
    kStr += self.endLine
    kStr += "/* global memory indices */" + self.endLine
    # C
    kStr += "#define GLOBAL_C(IDX" \
        + indexChars[0]
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      kStr += ", IDX" + indexChars[i]
    indexChar = indexChars[0]
    kStr += ") ( (IDX" + indexChar + ")*strideC" + indexChar
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      indexChar = indexChars[i]
      kStr += " + (IDX" + indexChar + ")*strideC" + indexChar
    kStr += " )" + self.endLine
    # A
    kStr += "#define GLOBAL_A(IDX" \
        + indexChars[kernel["ProblemType"]["IndexAssignmentsA"][0]]
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      kStr += ", IDX" + indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
    indexChar = indexChars[kernel["ProblemType"]["IndexAssignmentsA"][0]]
    kStr += ") ( (IDX" + indexChar + ")*strideA" + indexChar
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      indexChar = indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
      kStr += " + (IDX" + indexChar + ")*strideA" + indexChar
    kStr += " )" + self.endLine
    # B
    kStr += "#define GLOBAL_B(IDX" \
        + indexChars[kernel["ProblemType"]["IndexAssignmentsB"][0]]
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      kStr += ", IDX" + indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]
    indexChar = indexChars[kernel["ProblemType"]["IndexAssignmentsB"][0]]
    kStr += ") ( (IDX" + indexChar + ")*strideB" + indexChar
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      indexChar = indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]
      kStr += " + (IDX" + indexChar + ")*strideB" + indexChar
    kStr += " )" + self.endLine
    kStr += self.endLine

    ####################################
    # data types
    kStr += "/* data types */" + self.endLine
    kStr += "#define DATA_TYPE %s%s" \
        % (kernel["ProblemType"]["DataType"].toDevice(self.backend), \
        self.endLine)

    if self.backend == "OCL":
      kStr += "#define MAD(A,B,DST) mad(A,B,DST)"
    else:
      kStr += "#define MAD(A,B,DST) DST += A*B"
    kStr += self.endLine

    if self.backend == "HIP" and kernel["ProblemType"]["DataType"].isComplex():
      kStr += "#define s0 x" + self.endLine
      kStr += "#define s1 y" + self.endLine
    kStr += self.endLine

    ####################################
    # MADs
    kStr += "/* MADs */" + self.endLine
    if kernel["ProblemType"]["DataType"].isReal():
      # real data
      kStr += "#define TYPE_MAD(MULA,MULB,DST) " \
          + "DST = MAD(MULA,MULB,DST);" + self.endLine
      if kernel["ProblemType"]["UseBeta"]:
        # dst = alpha*reg + beta*dst
        kStr += "#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) " \
            + "DST = (ALPHA)*(REG) + (BETA)*(DST);" + self.endLine
      else:
        # dst = alpha*reg
        kStr += "#define TYPE_MAD_WRITE(DST,ALPHA,REG) " \
            + "DST = (ALPHA)*(REG);" + self.endLine
    else:
      # complex data
      if not kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]:
        # neither conjugate
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAD( -MULA.s1, MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]:
        # A conjugate (negate imaginary A.s1)
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s1, MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAD( -MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif not kernel["ProblemType"]["ComplexConjugateA"] and kernel["ProblemType"]["ComplexConjugateB"]:
        # B conjugate (negate imaginary B.s1)
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s0,  MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAD( -MULA.s1, -MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s0, -MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      else:
        # A & B conjugate (negate imaginary .s1)
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s0,  MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s1, -MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s0, -MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAD( -MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      if kernel["ProblemType"]["UseBeta"]:
        # dst = alpha*reg + beta*dst
        kStr += (
          "#define TYPE_MAD_WRITE( DST, ALPHA, REG, BETA ) "+self.endLinePP +
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
          "#define TYPE_MAD_WRITE( DST, ALPHA, REG ) "+self.endLinePP+
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

    kStr += "#define MICRO_TILE " + self.endLinePP
    for a in range(0, kernel["ThreadTile0"]):
      kStr += "  rA[%d] = localA[offA + %d*SG%s]; %s" \
          % (a, a, tileChar0, self.endLinePP)
    for b in range(0, kernel["ThreadTile1"]):
      kStr += "  rB[%d] = localB[offB + %d*SG%s]; %s" \
          % (b, b, tileChar1, self.endLinePP)
    kStr += "  offA += SPLITU*(MT" + tileChar0 + "+PAD); " + self.endLinePP
    kStr += "  offB += SPLITU*(MT" + tileChar1 + "+PAD); " + self.endLinePP
    for a in range(0, kernel["ThreadTile0"]):
      for b in range(0, kernel["ThreadTile1"]):
        kStr += "  TYPE_MAD(rA[%d],rB[%d],rC[%d+TT%s*%d]); %s" % (a, b, a, tileChar0, b, self.endLinePP)
    kStr += "  " + self.fenceStr + self.endLine
    kStr += self.endLine

    ####################################
    # preprocessor definitions of kernel arguments
    kStr += "/* preprocessor definitions of kernel arguments*/" + self.endLine
    firstStride = 0
    if kernel["ProblemType"]["UseInitialStrides"]:
      # no strides #defined
      lastStrideC = 0
      lastStrideA = 0
      lastStrideB = 0
    else:
      # #define initial stride
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
    kStr += self.endLine + self.endLine

    ####################################
    # function signature
    ####################################
    kStr += "/* kernel */" + self.endLine
    if self.backend == "HIP":
      kStr += "#pragma clang diagnostic push" + self.endLine
      kStr += "#pragma clang diagnostic ignored \"-Wunused-parameter\"" + self.endLine
    kStr += self.getSignature(kernel)
    kStr += " {" + self.endLine
    if self.backend == "HIP":
      kStr += "#pragma clang diagnostic pop" + self.endLine


    # debug printf - kernel args
    #kStr += "  if( " + self.getLocalIdStr + "(0) ==0 && " + self.getLocalIdStr + "(1) == 0) printf(\\\"alpha=%f, beta=%f, oC=%u, oA=%u, oB=%u, sCI=%u, sCJ=%u, sCK=%u, sAI=%u, sAK=%u, sAL=%u, sBI=%u, sBJ=%u, sBL=%u, sI=%u, sJ=%u, sK=%u, sL=%u\\\\n\\\", alpha, beta, offsetC, offsetA, offsetB, strideCI, strideC1J, strideC0K, strideAI, strideA0K, strideAL, strideBI, strideB1J, strideBL, sizeI, size1J, size0K, sizeL"
    #kStr += "  if( " + self.getLocalIdStr + "(0) ==0 && " + self.getLocalIdStr + "(1) == 0) printf(\\\"oC=%u, oA=%u, oB=%u, sCJ=%u, sAK=%u, sBK=%u, sI=%u, sJ=%u, sK=%u\\\\n\\\", offsetC, offsetA, offsetB, strideC1J, strideAK, strideBK, size0I, size1J, sizeK);" + self.endLine
    #kStr += "  if( " + self.getLocalIdStr + "(0) ==0 && " + self.getLocalIdStr + "(1) == 0) printf(\\\"%u, %u, %u, %u, %u, %u, %u, %u, %u,\\\\n\\\", offsetC, offsetA, offsetB, strideC1J, strideAK, strideBK, size0I, size1J, sizeK);" + self.endLine

    #kStr += "  printf(\\\"%u\\\\n\\\", sizeK);" + self.endLine
    # end debug printf

    # debug printf - tensor A, B
    # end debug printf

    # debug printf - tensor A, B
    #kStr += "unsigned int idx1 = " + self.getGlobalIdStr + "(0) + get_global_size(0)*(" + self.getGlobalIdStr + "(1)+get_global_size(1)*" + self.getGlobalIdStr + "(2));" + self.endLine
    #kStr += "unsigned int idx2 = idx1+(get_global_size(0)*get_global_size(1)*get_global_size(2));" + self.endLine
    #kStr += "printf(\"C[0] = %f, A[0] = %f; B[0] = %f\\n\", C[0], A[0], B[0]"
    #kStr += ");" + self.endLine
    # end debug printf


    ####################################
    # apply offsets
    kStr += self.endLine
    kStr += "  /* apply offsets */" + self.endLine
    kStr += ("  C += offsetC;" + self.endLine +
        "  A += offsetA;" + self.endLine +
        "  B += offsetB;" + self.endLine )

    ####################################
    # allocate registers
    kStr += self.endLine
    kStr += (
      "  /* allocate registers */" + self.endLine +
      "  DATA_TYPE rC[TT" + tileChar0 + "*TT" + tileChar1 + "] "
          + "= {0};" + self.endLine +
      "  DATA_TYPE rA[TT" + tileChar0 + "];" + self.endLine +
      "  DATA_TYPE rB[TT" + tileChar1 + "];" + self.endLine )


    ####################################
    # allocate local memory
    kStr += self.endLine
    kStr += "  /* allocate local memory */" + self.endLine
    kStr += "  %sDATA_TYPE lds[LDS_NUM_ELEMENTS];%s" \
        % (self.sharedDeclStr, self.endLine )
    kStr += "  %sDATA_TYPE *localA = lds;%s" \
        % (self.sharedPtrStr, self.endLine)
    kStr += "  %sDATA_TYPE *localB = lds + LDS_OFFSET_B;%s" \
        % (self.sharedPtrStr, self.endLine)

    ####################################
    # c indices
    ####################################

    kStr += "  /* c indices (group) */" + self.endLine
    if kernel["WorkGroupMapping"] == 1:
      kStr += "  unsigned int g" + tileChar0 + " = " \
          + self.getGroupIdStr + "(0);" + self.endLine
      kStr += "  unsigned int g" + tileChar1 + " = " \
          + self.getGroupIdStr + "(1);" + self.endLine
    else:
      dimCoal = (0 if kernel["WorkGroupMapping"] > 0 else 1)
      dimPerp = (1 if kernel["WorkGroupMapping"] > 0 else 0)

      # work-group free indices
      kStr += self.endLine
      kStr += "  unsigned int g%s, g%s;%s" % (tileChar0, tileChar1, self.endLine)
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
      kStr += "    g%s = (groupSerial/WORK_GROUP_MAPPING) %% %s(%s);%s" \
          % ((tileChar0 if kernel["WorkGroupMapping"] > 0 else tileChar1), \
          self.getNumGroupsStr, dimCoal, self.endLine)
      kStr += "    g%s = superGroup*WORK_GROUP_MAPPING + groupSerial %% WORK_GROUP_MAPPING;%s" \
          % ((tileChar1 if kernel["WorkGroupMapping"] > 0 else tileChar0), \
          self.endLine)

      # if in last super group
      kStr += "  } else {%s" % self.endLine
      kStr += "    g%s = (groupSerial-numWorkGroupsBeforeLastSuperGroup)/lastSuperGroupWidth;%s" \
          % ((tileChar0 if kernel["WorkGroupMapping"] > 0 else tileChar1), \
          self.endLine)
      kStr += "    g%s = superGroup*WORK_GROUP_MAPPING + groupSerial %% lastSuperGroupWidth;%s" \
          % ((tileChar1 if kernel["WorkGroupMapping"] > 0 else tileChar0), \
          self.endLine)

      # if in last super group
      kStr += "  }%s" % self.endLine


    # other free indices
    nonTileFreeIndices = range(0, kernel["ProblemType"]["NumIndicesC"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index0"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index1"])
    for i in range(0, len(nonTileFreeIndices)):
      index = nonTileFreeIndices[i]
      kStr += "  unsigned int g" + indexChars[index] \
          + " = ( " + self.getGroupIdStr + "(2)"
      for j in reversed( range( i+1, len(nonTileFreeIndices)) ):
        index2 = nonTileFreeIndices[j]
        kStr += " / size" + indexChars[index2]
      kStr += " ) % size" + indexChars[index] + ";" + self.endLine

    ####################################
    # local indices
    ####################################
    kStr += self.endLine
    kStr += "  /* c indices (local) */" + self.endLine
    kStr += "  unsigned int serial = %s(0);%s" \
        % (self.getLocalIdStr, self.endLine)
    kStr += "  unsigned int sgId = serial / (SG%s*SG%s);%s" \
        % (tileChar0, tileChar1, self.endLine)
    #kStr += "  unsigned int l" + tileChar0 \
    #    + " = " + self.getLocalIdStr + "(0); // d0" + self.endLine
    #kStr += "  unsigned int l" + tileChar1 \
    #    + " = " + self.getLocalIdStr + "(1); // d1" + self.endLine

    kStr += "  unsigned int a" + tileCharA + " = "
    if kernel["ProblemType"]["TLUA"]:
      kStr += "serial%LSCA;" + self.endLine
    else:
      kStr += "serial/LSCA;" + self.endLine

    kStr += "  unsigned int b" + tileCharB + " = "
    if not kernel["ProblemType"]["TLUB"]:
      kStr += "serial/LSCB;" + self.endLine
    else:
      kStr += "serial%LSCB;" + self.endLine
    kStr += self.endLine

    kStr += "  /* indices within subgroup */" + self.endLine
    kStr += "  unsigned int l%s = (serial %% SG%s);%s" \
        % (tileChar0, tileChar0, self.endLine)
    kStr += "  unsigned int l%s = (serial / SG%s) %% SG%s;%s" \
        % (tileChar1, tileChar0, tileChar1, self.endLine)
    kStr += self.endLine


    kStr += "  /* unrolled summation index */" + self.endLine
    kStr += "  unsigned int a" + unrollChar + " = "
    if kernel["ProblemType"]["TLUA"]:
      kStr += "serial/LSCA;" + self.endLine
    else:
      kStr += "serial%LSCA;" + self.endLine

    kStr += "  unsigned int b" + unrollChar + " = "
    if not kernel["ProblemType"]["TLUB"]:
      kStr += "serial%LSCB;" + self.endLine
    else:
      kStr += "serial/LSCB;" + self.endLine
    kStr += self.endLine

    # other non-unrolled summation indices
    kStr += "  /* other non-unrolled summation indices (all start at zero) */" + self.endLine
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
      index = i
      kStr += "#define a" + indexChars[index] + " 0" + self.endLine
      kStr += "#define b" + indexChars[index] + " 0" + self.endLine
    kStr += self.endLine



    ####################################
    # offset global pointers
    ####################################
    kStr += "  /* where will this thread read from global memory */" + self.endLine
    kStr += "  A += GLOBAL_A( (" + self.uint64Str + ")"
    for i in range(0, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      index = kernel["ProblemType"]["IndexAssignmentsA"][i]
      if index < kernel["ProblemType"]["NumIndicesC"]: # c index
        if index == kernel["ProblemType"]["TileA"]: # this index is A's tile index
          kStr += "a%s+g%s*MT%s" % (tileCharA, tileCharA, tileCharA)
        else: # just a group index
          kStr += "g" + indexChars[index]
      else: # summation index
        kStr += "a" + indexChars[index]
      if i < len(kernel["ProblemType"]["IndexAssignmentsA"])-1:
        kStr += ", (" + self.uint64Str + ")"
    kStr += " );" + self.endLine

    kStr += "  B += GLOBAL_B( (" + self.uint64Str + ")"
    for i in range(0, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      index = kernel["ProblemType"]["IndexAssignmentsB"][i]
      if index < kernel["ProblemType"]["NumIndicesC"]: # c index
        if index == kernel["ProblemType"]["TileB"]: # this index is B's tile index
          kStr += "b%s+g%s*MT%s" % (tileCharB, tileCharB, tileCharB)
        else: # just a group index
          kStr += "g" + indexChars[index]
      else: # summation index
        kStr += "b" + indexChars[index]
      if i < len(kernel["ProblemType"]["IndexAssignmentsB"])-1:
        kStr += ", (" + self.uint64Str + ")"
    kStr += " );" + self.endLine
    kStr += self.endLine

    # if udsgttdsA: tileCharA+g*MT, aUnroll
    # nt udsgttdsA: aUnroll, tileCharA+g*MT

    """
    if kernel["ProblemType"]["TLUA"]:
      kStr += "  A += GLOBAL_A( a%s+g%s*MT%s, a%s" \
          % (tileCharA, tileCharA, tileCharA, unrollChar)
    else:
      kStr += "  A += GLOBAL_A( a%s, a%s+g%s*MT%s" \
          % (unrollChar, tileCharA, tileCharA, tileCharA)
    for i in range(2, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      kStr += ", g%s" % indexChars[i]
    kStr += " );" + self.endLine

    if not kernel["ProblemType"]["TLUB"]:
      kStr += "  B += GLOBAL_B( b%s, b%s+g%s*MT%s" \
          % (unrollChar, tileCharB, tileCharB, tileCharB)
    else:
      kStr += "  B += GLOBAL_B( b%s+g%s*MT%s, b%s" \
          % (tileCharB, tileCharB, tileCharB, unrollChar)
    for i in range(2, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      kStr += ", g%s" % indexChars[i]
    kStr += " );" + self.endLine
    kStr += self.endLine
    """

    ####################################
    # offset local pointers
    ####################################
    kStr += "  /* where will this thread write to local memory */" + self.endLine
    kStr += "  %sDATA_TYPE *lA = localA + a%s + a%s*(MT%s+PAD);%s" \
        % (self.sharedPtrStr, tileCharA, unrollChar, tileCharA, self.endLine)
    kStr += "  %sDATA_TYPE *lB = localB + b%s + b%s*(MT%s+PAD);%s" \
        % (self.sharedPtrStr, tileCharB, unrollChar, tileCharB, self.endLine)
    kStr += self.endLine

    ####################################
    # global -> register branches
    ####################################
    if kernel["EdgeType"] == "Branch":
      kStr += "  /* conditionals to guard against loading A out-of-bounds */" + self.endLine
      for perp in range(0, kernel["NumLoadsPerpendicularA"]):
        for para in range(0, kernel["NumLoadsCoalescedA"]):
          kStr += "  bool condA_" + str(para) + "_" + str(perp) + " = "
          kStr += "( a%s+g%s*MT%s+" % ( tileCharA, tileCharA, tileCharA)
          if not kernel["ProblemType"]["TLUA"]:
            kStr += "%d*LSPA" % (perp)
          else:
            kStr += "%d*LSCA" % (para)
          kStr += " >= size%s);%s" %( tileCharA, self.endLine )
      kStr += self.endLine

    if kernel["EdgeType"] == "Branch":
      kStr += "  /* conditionals to guard against loading B out-of-bounds */" + self.endLine
      for perp in range(0, kernel["NumLoadsPerpendicularB"]):
        for para in range(0, kernel["NumLoadsCoalescedB"]):
          kStr += "  bool condB_" + str(para) + "_" + str(perp) + " = "
          kStr += "( b%s+g%s*MT%s+" % ( tileCharB, tileCharB, tileCharB)
          if not kernel["ProblemType"]["TLUB"]:
            kStr += "%d*LSPB" % (perp)
          else:
            kStr += "%d*LSCB" % (para)
          kStr += " >= size%s);%s" % (tileCharB, self.endLine )
      kStr += self.endLine

    kStr += "  /* registers used for global -> local loads */" + self.endLine
    kStr += "  DATA_TYPE "
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        kStr += "a_" + str(para) + "_" + str(perp)
        if para == kernel["NumLoadsCoalescedA"]-1 and perp == kernel["NumLoadsPerpendicularA"]-1:
          kStr += ";" + self.endLine
        else:
          kStr += ", "
    kStr += "  DATA_TYPE "
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        kStr += "b_" + str(para) + "_" + str(perp)
        if para == kernel["NumLoadsCoalescedB"]-1 and perp == kernel["NumLoadsPerpendicularB"]-1:
          kStr += ";" + self.endLine
        else:
          kStr += ", "
    kStr += self.endLine


    ####################################


    # debug printf - global data
    #kStr += "  printf(\\\"T[%u,%u] A[%u] = %f; B[%u] = %f\\\\n\\\", " + self.getLocalIdStr + "(0), " + self.getLocalIdStr + "(1), serial, A[serial], serial, B[serial]"
    #kStr += ");" + self.endLine
    # end debug printf


    # multi-dim if (kernel.order=="clblasColumnMajor")==(kernel.transA=="N"):
    #tensorAssignedToTileDim = []
    #if kernel["ProblemType"]["Tensor0"]:
    #  tensorAssignedToTileDim.append(kernel.problem.operation.
    #unrollStrideGreaterThanTileA


    # kStr += "  bool validC ="
    # if kernel.tile.branch[0]:
    #   kStr += " (globalC" \
    #       + tileChar0 + " + " \
    #       + str(a) + "*SG" + tileChar0 + "" + " < size" \
    #       + tileChar0 + ")"
    # if kernel.tile.branch[0] and kernel.tile.branch[1]:
    #   kStr += " &&"
    # if kernel.tile.branch[1]:
    #   kStr += " (globalC" \
    #       + tileChar1 + " + " \
    #       + str(b) + "*SG" + tileChar1 + "" + " < size" \
    #       + tileChar1 + ")"
    # kStr += ";" + self.endLine

    #if self.returnOnly:
    #  kStr += "return;" + self.endLine + "#if 0" + self.endLine


    ####################################
    # summations loops
    ####################################
    indent = "  "
    kStr += indent + "/* iterate over summation indice(s) */" + self.endLine
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
    kStr += self.endLine

    # 1st barrier
    kStr += indent + self.syncStr + self.endLine

    ####################################
    # load A
    ####################################
    kStr += indent + "/* load A global -> local */" + self.endLine

    #if kernel.loadRequiresFewerThreadsA():
    #  kStr += indent + "if ( serial < %d ) {%s" \
    #      % (kernel.loadSizeParaA*kernel.loadSizePerpA, self.endLine)
    #  indent += "  "
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        kStr += indent
        #condPara = (para==kernel["NumLoadsCoalescedA"]-1 and kernel.lastLoadRequiresGuardParaA())
        #condPerp = (perp==kernel["NumLoadsPerpendicularA"]-1 and kernel.lastLoadRequiresGuardPerpA())
        #if condPara or condPerp:
        #  kStr += "if ( "
        #  if condPara:
        #    kStr += "a%s < %d" % (unrollChar if not kernel["ProblemType"]["TLUA"] else tileCharA, kernel.totalLoadSizeParaA % kernel.loadSizeParaA )
        #  if condPerp:
        #    if condPara:
        #      kStr += " && "
        #    kStr += "a%s < %d" % (unrollChar if kernel["ProblemType"]["TLUA"] else tileCharA, kernel.totalLoadSizePerpA % kernel.loadSizePerpA )
        #  kStr += " ) { "

        kStr += "a_" + str(para) + "_" + str(perp) + " = "

        if kernel["EdgeType"] == "Branch":
          kStr += "( condA_%s_%s )" %( str(para), str(perp) )
          kStr += " ? %s : " %( kernel["ProblemType"]["DataType"].zeroString(self.backend) )

        kStr += "A[ %d*LSCA*strideA%s + %d*LSPA*strideA%s];" \
            % (para, unrollChar if not kernel["ProblemType"]["TLUA"] else tileCharA, perp, unrollChar if kernel["ProblemType"]["TLUA"] else tileCharA)
        #if condPara or condPerp:
        #  kStr += " }" + self.endLine
        kStr += self.endLine
    #if kernel.loadRequiresFewerThreadsA():
    #  indent = indent[2:]
    #  kStr += indent + "}" + self.endLine
    kStr += self.endLine

    ####################################
    # load B
    ####################################
    kStr += indent + "/* load B global -> local */" + self.endLine
    #if kernel.loadRequiresFewerThreadsB():
    #  kStr += indent + "if ( serial < %d ) {%s" \
    #      % (kernel.loadSizeParaB*kernel.loadSizePerpB, self.endLine)
    #  indent += "  "
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        kStr += indent
        #condPara = (para==kernel["NumLoadsCoalescedB"]-1 and kernel.lastLoadRequiresGuardParaB())
        #condPerp = (perp==kernel["NumLoadsPerpendicularB"]-1 and kernel.lastLoadRequiresGuardPerpB())
        #if condPara or condPerp:
        #  kStr += "if ( "
        #  if condPara:
        #        kStr += "b%s < %d" % (unrollChar if not kernel["ProblemType"]["TLUB"] else tileCharB, kernel.totalLoadSizeParaB % kernel.loadSizeParaB )
        #  if condPerp:
        #    if condPara:
        #      kStr += " && "
        #    kStr += "b%s < %d" % (unrollChar if kernel["ProblemType"]["TLUB"] else tileCharB, kernel.totalLoadSizePerpB % kernel.loadSizePerpB )
        #  kStr += " ) { "

        kStr += "b_" + str(para) + "_" + str(perp) + " = "

        if kernel["EdgeType"] == "Branch":
          kStr += "( condB_%s_%s )" % ( str(para), str(perp) )
          kStr += " ? %s : " % ( kernel["ProblemType"]["DataType"].zeroString(self.backend) )

        kStr += "B[ %d*LSCB*strideB%s + %d*LSPB*strideB%s];" \
            % (para, unrollChar if not kernel["ProblemType"]["TLUB"] else tileCharB, perp, unrollChar if kernel["ProblemType"]["TLUB"] else tileCharB)
        #if condPara or condPerp:
        #  kStr += " }" + self.endLine
        kStr += self.endLine
    #if kernel.loadRequiresFewerThreadsB():
    #  indent = indent[2:]
    #  kStr += indent + "}" + self.endLine
    kStr += self.endLine


    ########################################
    # store registers in lds
    ########################################
    if self.backend == "HIP":
      kStr += "#pragma clang diagnostic push" + self.endLine
      kStr += "#pragma clang diagnostic ignored \"-Wconditional-uninitialized\"" + self.endLine
    # if num threads
    #if kernel.loadRequiresFewerThreadsA():
    #  kStr += indent + "if ( serial < %d ) {%s" \
    #      % (kernel.loadSizeParaA*kernel.loadSizePerpA, self.endLine)
    #  indent += "  "
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        kStr += indent
        # if thread should be storing
        #condPara = (para==kernel["NumLoadsCoalescedA"]-1 and kernel.lastLoadRequiresGuardParaA())
        #condPerp = (perp==kernel["NumLoadsPerpendicularA"]-1 and kernel.lastLoadRequiresGuardPerpA())
        #if condPara or condPerp:
        #  kStr += "if ( "
        #  if condPara:
        #    kStr += "a%s < %d" % (unrollChar if not kernel["ProblemType"]["TLUA"] else tileCharA, kernel.totalLoadSizeParaA % kernel.loadSizeParaA )
        #  if condPerp:
        #    if condPara:
        #      kStr += " && "
        #    kStr += "a%s < %d" % (unrollChar if kernel["ProblemType"]["TLUA"] else tileCharA, kernel.totalLoadSizePerpA % kernel.loadSizePerpA )
        #  kStr += " ) { "
        # store
        kStr += "lA[ %d*LSCA" % para
        if not kernel["ProblemType"]["TLUA"]:
          kStr += "*(MT%s+PAD)" % tileCharA
        kStr += " + %d*LSPA" % perp
        if kernel["ProblemType"]["TLUA"]:
          kStr += "*(MT%s+PAD)" % tileCharA
        kStr += " ] = "
        kStr += "a_" + str(para) + "_" + str(perp) + ";"
        #if condPara or condPerp:
        #  kStr += " }"
        kStr += self.endLine
    #if kernel.loadRequiresFewerThreadsA():
    #  indent = indent[2:]
    #  kStr += indent + "}" + self.endLine
    kStr += self.endLine


    #if kernel.loadRequiresFewerThreadsB():
    #  kStr += indent + "if ( serial < %d ) {%s" \
    #      % (kernel.loadSizeParaB*kernel.loadSizePerpB, self.endLine)
    #  indent += "  "
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        kStr += indent
        # if thread should store
        #condPara = (para==kernel["NumLoadsCoalescedB"]-1 and kernel.lastLoadRequiresGuardParaB())
        #condPerp = (perp==kernel["NumLoadsPerpendicularB"]-1 and kernel.lastLoadRequiresGuardPerpB())
        #if condPara or condPerp:
        #  kStr += "if ( "
        #  if condPara:
        #        kStr += "b%s < %d" % (unrollChar if not kernel["ProblemType"]["TLUB"] else tileCharB, kernel.totalLoadSizeParaB % kernel.loadSizeParaB )
        #  if condPerp:
        #    if condPara:
        #      kStr += " && "
        #    kStr += "b%s < %d" % (unrollChar if kernel["ProblemType"]["TLUB"] else tileCharB, kernel.totalLoadSizePerpB % kernel.loadSizePerpB )
        #  kStr += " ) { "
        # store
        kStr += "lB[ %d*LSCB" % para
        if not kernel["ProblemType"]["TLUB"]:
          kStr += "*(MT%s+PAD)" % tileCharB
        kStr += " + %d*LSPB" % perp
        if kernel["ProblemType"]["TLUB"]:
          kStr += "*(MT%s+PAD)" % tileCharB
        kStr += " ] = "
        kStr += "b_" + str(para) + "_" + str(perp) + ";"
        #if condPara or condPerp:
        #  kStr += " }"
        kStr += self.endLine
    #if kernel.loadRequiresFewerThreadsB():
    #  indent = indent[2:]
    #  kStr += indent + "}" + self.endLine
    kStr += self.endLine
    # end store in lds
    if self.backend == "HIP":
      kStr += "#pragma clang diagnostic pop" + self.endLine

    # 2nd barrier
    kStr += indent + self.syncStr + self.endLine
    kStr += "%sunsigned int offA = l%s + sgId*(MT%s+PAD);%s" \
          % (indent, tileChar0, tileChar0, self.endLine)
    kStr += "%sunsigned int offB = l%s + sgId*(MT%s+PAD);%s"\
          % (indent, tileChar1, tileChar1, self.endLine)


    # # LDS state
    # kStr += indent + "/* print LDS state */" + self.endLine
    # kStr += indent + "if ( gJ==0 && gL==0 && g1K==0 && g0I==0 && serial == 0) {" + self.endLine
    # kStr += indent + "  for (unsigned int u = 0; u < UNROLL; u++) {" + self.endLine
    # kStr += indent + "    for (unsigned int i = 0; i < MT" + tileChar0 + "; i++) {" + self.endLine
    # kStr += indent + "      printf(\\\"[%u,%u,%u,%u][%u,%u,%u][%02u,%02u] a=%f; b=%f\\\\n\\\", gJ, gL, g1K, g0I, sumIterM, sumIterN, sumIterO, u, i, localA[i+u*(MT"+tileChar0+"+PAD)], localB[i+u*(MT"+tileChar0+"+PAD)] );" + self.endLine
    # # kStr += indent + "      printf(\\\"hi %u\\\\n\\\", size0I);" + self.endLine
    # # kStr += indent + "      printf(\\\"hi\\\\n\\\");" + self.endLine
    # kStr += indent + "    }" + self.endLine
    # kStr += indent + "  }" + self.endLine
    # kStr += indent + "}" + self.endLine
    # # [work-group id] idx=%i a=%f; b=%f


    ####################################
    # do macs
    kStr += self.endLine
    kStr += indent + "/* do macs */" + self.endLine
    for u in range(0, kernel["LoopUnroll"]):
      kStr += indent + "MICRO_TILE" + self.endLine
    kStr += self.endLine

    # debug printf - accumulation in registers
    # kStr += "  if (validC) printf(\\\"T[%u,%u] rC = %f g=%u\\\\n\\\", " + self.getLocalIdStr + "(0), " + self.getLocalIdStr + "(1), rC[0][0], GLOBAL_C(globalC0I, globalC1J) );" + self.endLine
    # end debug printf
    # kStr += indent + "if ( gJ==0 && gL==0 && g1K==0 && g0I==0 && serial == 0 ) printf(\\\"[%u,%u,%u,%u] m=%u, n=%u, o=%u, r[0][0]=%.0f\\\\n\\\", gJ, gL, g1K, g0I, sumIterM, sumIterN, sumIterO, rC[0][0] );" + self.endLine


    ########################################################################
    # BEGIN UNROLL=1 LOOP
    ########################################################################


    # if another loop, close current unrolled loops
    if kernel["LoopTail"]:
      loopChar = indexChars[kernel["ProblemType"]["IndicesSummation"][kernel["ProblemType"]["NumIndicesSummation"]-1]]
      # advance A, B along summation dimension
      kStr += indent + "A += (" + self.uint64Str + ")strideA" + loopChar + "*DEPTHU;" + self.endLine
      kStr += indent + "B += (" + self.uint64Str + ")strideB" + loopChar + "*DEPTHU;" + self.endLine
      indent = indent[2:]
      # close do-while loop
      if kernel["LoopDoWhile"]:
        kStr += indent + "} while (--sumIter" + loopChar \
            + " > 0);" + self.endLine
      else:
        kStr += "%s}%s" % (indent, self.endLine)
      kStr += self.endLine


      ####################################
      # summations loops
      #kStr += indent + "/* unroll=1 loop */" + self.endLine
      #kStr += indent + "sumIter" + loopChar + " = size" + loopChar + " % DEPTHU;" + self.endLine
      #kStr += self.endLine


      ####################################
      # load A single
      ####################################
      kStr += indent + "/* unroll=1 load A global -> local */" + self.endLine
      kStr += indent + self.syncStr + self.endLine
      #if kernel.loadRequiresFewerThreadsA():
      #  kStr += indent + "if ( serial < %d ) {%s" \
      #      % (kernel.loadSizeParaA*kernel.loadSizePerpA, self.endLine)
      #  indent += "  "
      for perp in range(0, kernel["NumLoadsPerpendicularA"]):
        for para in range(0, kernel["NumLoadsCoalescedA"]):
          kStr += indent
          #condPara = (para==kernel["NumLoadsCoalescedA"]-1 and kernel.lastLoadRequiresGuardParaA())
          #condPerp = (perp==kernel["NumLoadsPerpendicularA"]-1 and kernel.lastLoadRequiresGuardPerpA())
          #if condPara or condPerp:
          #  kStr += "if ( "
          #  if condPara:
          #    kStr += "a%s < %d" % (unrollChar if not kernel["ProblemType"]["TLUA"] else tileCharA, kernel.totalLoadSizeParaA % kernel.loadSizeParaA )
          #  if condPerp:
          #    if condPara:
          #      kStr += " && "
          #    kStr += "a%s < %d" % (unrollChar if kernel["ProblemType"]["TLUA"] else tileCharA, kernel.totalLoadSizePerpA % kernel.loadSizePerpA )
          #  kStr += " ) { "
          kStr += "lA[ %d*LSCA" % para
          if not kernel["ProblemType"]["TLUA"]:
            kStr += "*(MT%s+PAD)" % tileCharA
          kStr += " + %d*LSPA" % perp
          if kernel["ProblemType"]["TLUA"]:
            kStr += "*(MT%s+PAD)" % tileCharA
          kStr += " ] = "
          # guard around K
          kStr += "( a%s + " % (unrollChar)
          if kernel["ProblemType"]["TLUA"]:
            kStr += "%d*LSPA >= (size%s %% DEPTHU) )" % (perp, unrollChar)
          else:
            kStr += "%d*LSCA >= (size%s %% DEPTHU) )" % (para, unrollChar)
          # guard around branch
          if kernel["EdgeType"] == "Branch":
            kStr += " || "
            kStr += "( a%s+g%s*MT%s+" % ( tileCharA, tileCharA, tileCharA)
            if not kernel["ProblemType"]["TLUA"]:
              kStr += "%d*LSPA" % (perp)
            else:
              kStr += "%d*LSCA" % (para)
            kStr += " >= size%s)" %( tileCharA )
          kStr += " ? %s : " % kernel["ProblemType"]["DataType"].zeroString(self.backend)
          kStr += "A[ %d*LSCA*strideA%s + %d*LSPA*strideA%s];" \
              % (para, unrollChar if not kernel["ProblemType"]["TLUA"] else tileCharA, perp, unrollChar if kernel["ProblemType"]["TLUA"] else tileCharA)
          #if condPara or condPerp:
          #  kStr += " }" + self.endLine
          kStr += self.endLine
      #if kernel.loadRequiresFewerThreadsA():
      #  indent = indent[2:]
      #  kStr += indent + "}" + self.endLine


      ####################################
      # load B single
      ####################################
      #if kernel.loadRequiresFewerThreadsB():
      #  kStr += indent + "if ( serial < %d ) {%s" \
      #      % (kernel.loadSizeParaB*kernel.loadSizePerpB, self.endLine)
      #  indent += "  "
      for perp in range(0, kernel["NumLoadsPerpendicularB"]):
        for para in range(0, kernel["NumLoadsCoalescedB"]):
          kStr += indent
          #condPara = (para==kernel["NumLoadsCoalescedB"]-1 and kernel.lastLoadRequiresGuardParaB())
          #condPerp = (perp==kernel["NumLoadsPerpendicularB"]-1 and kernel.lastLoadRequiresGuardPerpB())
          #if condPara or condPerp:
          #  kStr += "if ( "
          #  if condPara:
          #        kStr += "b%s < %d" % (unrollChar if not kernel["ProblemType"]["TLUB"] else tileCharB, kernel.totalLoadSizeParaB % kernel.loadSizeParaB )
          #  if condPerp:
          #    if condPara:
          #      kStr += " && "
          #    kStr += "b%s < %d" % (unrollChar if kernel["ProblemType"]["TLUB"] else tileCharB, kernel.totalLoadSizePerpB % kernel.loadSizePerpB )
          #  kStr += " ) { "

          kStr += "lB[ %d*LSCB" % para
          if not kernel["ProblemType"]["TLUB"]:
            kStr += "*(MT%s+PAD)" % tileCharB
          kStr += " + %d*LSPB" % perp
          if kernel["ProblemType"]["TLUB"]:
            kStr += "*(MT%s+PAD)" % tileCharB
          kStr += " ] = "
          # guard around k
          kStr += "( b%s + " % (unrollChar)
          if kernel["ProblemType"]["TLUB"]:
            kStr += "%d*LSPB >= (size%s %% DEPTHU) )" % (perp, unrollChar)
          else:
            kStr += "%d*LSCB >= (size%s %% DEPTHU) )" % (para, unrollChar)
          # guard branch
          if kernel["EdgeType"] == "Branch":
            kStr += " || "
            kStr += "( b%s+g%s*MT%s+" % ( tileCharB, tileCharB, tileCharB)
            if not kernel["ProblemType"]["TLUB"]:
              kStr += "%d*LSPB" % (perp)
            else:
              kStr += "%d*LSCB" % (para)
            kStr += " >= size%s)" % (tileCharB )

          kStr += " ? %s : " % kernel["ProblemType"]["DataType"].zeroString(self.backend)
          kStr += "B[ %d*LSCB*strideB%s + %d*LSPB*strideB%s];" \
              % (para, unrollChar if not kernel["ProblemType"]["TLUB"] else tileCharB, perp, unrollChar if kernel["ProblemType"]["TLUB"] else tileCharB)
          #if condPara or condPerp:
          #  kStr += " }" + self.endLine
          kStr += self.endLine
      #if kernel.loadRequiresFewerThreadsB():
      #  indent = indent[2:]
      #  kStr += indent + "}" + self.endLine

      kStr += indent + self.syncStr + self.endLine

      # full end loop b/c local full of zeros
      kStr += self.endLine
      kStr += indent + "/* full unroll loop */" + self.endLine
      #kStr += indent + "sumIter" + indexChar + " = UNROLL;" + self.endLine
      #kStr += "#undef UNROLL" + self.endLine
      #kStr += "#define UNROLL 1" + self.endLine
      #kStr += self.endLine

      kStr += "%sunsigned int offA = l%s + sgId*(MT%s+PAD);%s" \
            % (indent, tileChar0, tileChar0, self.endLine)
      kStr += "%sunsigned int offB = l%s + sgId*(MT%s+PAD);%s"\
            % (indent, tileChar1, tileChar1, self.endLine)

      #kStr += indent + "unsigned int offA = l" + tileChar0 + "; // d0" + self.endLine
      #kStr += indent + "unsigned int offB = l" + tileChar1 + "; // d1" + self.endLine
      kStr += self.endLine

      # begin loop
      # kStr += "%ssumIter%s = UNROLL;%s" % (indent, loopChar, self.endLine)
      kStr += "%ssumIter%s = (((size%s %% DEPTHU) + SPLITU - 1) / SPLITU);%s" % (indent, loopChar, loopChar, self.endLine)

      if kernel["LoopDoWhile"]:
        kStr += indent + "do {" + self.endLine
      else:
        kStr += indent + "while (sumIter%s-- > 0) {%s" \
            % (loopChar, self.endLine)

      indent += "  "

      ####################################
      # do macs
      kStr += indent + "/* do macs */" + self.endLine
      kStr += indent + "MICRO_TILE" + self.endLine
      kStr += self.endLine

    ########################################################################
    # END UNROLL=1 LOOP
    ########################################################################


    ####################################
    # end loop
    for i in reversed(range(0,kernel["ProblemType"]["NumIndicesSummation"])):
      loopChar = indexChars[kernel["ProblemType"]["IndicesSummation"][i]]
      # advance A, B along summation dimension
      kStr += indent + "A += (" + self.int64Str + ") strideA" + loopChar
      if i==kernel["ProblemType"]["NumIndicesSummation"]-1:
        kStr += "*DEPTHU"
      else:
        for j in range(i+1,min(i+2, kernel["ProblemType"]["NumIndicesSummation"]) ):
          tmpChar = indexChars[kernel["ProblemType"]["IndicesSummation"][j]]
          kStr += " - strideA" + tmpChar + "*size" + tmpChar
      kStr += ";" + self.endLine

      kStr += indent + "B += (" + self.int64Str + ") strideB" + loopChar
      if i==kernel["ProblemType"]["NumIndicesSummation"]-1:
        kStr += "*DEPTHU"
      else:
        for j in range(i+1,min(i+2,kernel["ProblemType"]["NumIndicesSummation"]) ):
          tmpChar = indexChars[kernel["ProblemType"]["IndicesSummation"][j]]
          kStr += " - strideB" + tmpChar + "*size" + tmpChar
      kStr += ";" + self.endLine
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
      kStr += "  /* SplitU: store all rC's to LDS */" + self.endLine
      kStr += "  " + self.syncStr + self.endLine
      # assign initial
      for i in range(0, kernel["ThreadTile0"]):
        for j in range(0, kernel["ThreadTile1"]):
          kStr += "  lds[l%s + %u*SG%s + MT%s*(l%s + %u*SG%s) + MT%s*MT%s*sgId] = rC[%u+TT%s*%u];%s" \
              % (tileChar0, i, tileChar0, tileChar0, tileChar1, \
              j, tileChar1, tileChar0, tileChar1, i, tileChar0, j, self.endLine)
      kStr += "  " + self.syncStr + self.endLine + self.endLine

      ####################################
      # new C elements to store
      kStr += "  /* SplitU: new C elements to store */" + self.endLine
      for i in range(0, kernel["NumElementsPerThread"]):
        kStr += "  rC[%3u] = lds[serial+%u*NUM_THREADS];%s" \
            % (i, i, self.endLine)
      kStr += self.endLine

      ####################################
      # SplitU reduction
      kStr += "  /* SplitU: reduction */" + self.endLine
      for s in range(1, kernel["SplitU"]):
        for i in range(0, kernel["NumElementsPerThread"]):
          kStr += "  rC[%3u] += lds[serial+%u*NUM_THREADS + %u*MT%s*MT%s];%s" \
              % (i, i, s, tileChar0, tileChar1, self.endLine)
        kStr += self.endLine

      ####################################
      # which global Cij index
      kStr += "  /* which global Cij index */" + self.endLine
      for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
        kStr += "  unsigned int globalC" + indexChars[i] \
            + " = g" + indexChars[i]
        if i == kernel["ProblemType"]["Index0"]:
          kStr += "*MT%s + (serial %% MT%s)" % (tileChar0, tileChar0)
        if i == kernel["ProblemType"]["Index1"]:
          kStr += "*MT%s + (serial / MT%s)" % (tileChar1, tileChar0)
        kStr += ";" + self.endLine
      kStr += self.endLine


      ####################################
      # global store
      kStr += "  /* write global C */" + self.endLine
      if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
        kStr += "  float type_mac_tmp;" + self.endLine
      if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
        kStr += "  double type_mac_tmp;" + self.endLine

      for b in range(0, kernel["NumElementsPerThread"]):
        numEdges = 0
        #for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
        if kernel["EdgeType"] == "Branch":
          kStr += "  if (globalC" \
              + tileChar0 + " < size" \
              + tileChar0 + ") {"
          numEdges += 1
        if kernel["EdgeType"] == "Branch":
          kStr += "  if (globalC" \
              + tileChar1 + " + " \
              + str(b) + "*CPS < size" \
              + tileChar1 + ") {"
          numEdges += 1

        kStr += "  TYPE_MAD_WRITE( C[ GLOBAL_C( (" + self.uint64Str + ")"
        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          kStr += " globalC" + indexChars[i]
          if i == kernel["ProblemType"]["Index1"]:
            kStr += " + " + str(b) + "*CPS"
          if i < kernel["ProblemType"]["NumIndicesC"]-1:
            kStr += ", (" + self.uint64Str + ")"
        kStr += ") ]"
        kStr += ", alpha"
        kStr += ", rC[%d]" % (b)
        if kernel["ProblemType"]["UseBeta"]:
          kStr += ", beta"
        kStr += ")"

        for i in range(0,numEdges):
          kStr += " }"
        kStr += self.endLine








      ####################################
      # SplitU==1, write data straight
      ####################################
    else:

      ####################################
      # which global Cij index
      kStr += "  /* which global Cij index */" + self.endLine
      for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
        kStr += "  unsigned int globalC" + indexChars[i] \
            + " = g" + indexChars[i]
        if i == kernel["ProblemType"]["Index0"]:
          kStr += "*MT%s + (serial %% SG%s)" % (tileChar0, tileChar0)
        if i == kernel["ProblemType"]["Index1"]:
          kStr += "*MT%s + (serial / SG%s)" % (tileChar1, tileChar0)
        kStr += ";" + self.endLine
      kStr += self.endLine

      ####################################
      # write global Cij
      # debug printf
      #kStr += "  printf(\\\"T[%u,%u] global = %u, %u, %u size=%u, %u\\\\n\\\", " + self.getLocalIdStr + "(0), " + self.getLocalIdStr + "(1), global0I, global1J, globalCK, size0I, size1J);" + self.endLine
      # end debug
      # kStr += "  rC[0][0] = DATA_TYPE(1.23456789, -1.23456789);" + self.endLine

      # kStr += indent + "/* print LDS state */" + self.endLine
      # kStr += indent + "if ( gJ==0 && gL==0 && g1K==0 && g0I==0 && serial == 0) {" + self.endLine
      # kStr += indent + "  for (unsigned int u = 0; u < UNROLL; u++) {" + self.endLine
      # kStr += indent + "    for (unsigned int i = 0; i < MT" + tileChar0 + "; i++) {" + self.endLine
      # kStr += indent + "      printf(\\\"[%u,%u,%u,%u][%u,%u,%u][%02u,%02u] a=%f; b=%f\\\\n\\\", gJ, gL, g1K, g0I, sumIterM, sumIterN, sumIterO, u, i, localA[i+u*(MT"+tileChar0+"+PAD)], localB[i+u*(MT"+tileChar0+"+PAD)] );" + self.endLine
      # # kStr += indent + "      printf(\\\"hi %u\\\\n\\\", size0I);" + self.endLine
      # # kStr += indent + "      printf(\\\"hi\\\\n\\\");" + self.endLine
      # kStr += indent + "    }" + self.endLine
      # kStr += indent + "  }" + self.endLine
      # kStr += indent + "}" + self.endLine


      # kStr += indent + "  for (unsigned int i = 0; i < 8; i++) {" + self.endLine
      # kStr += indent + "    for (unsigned int j = 0; j < 8; j++) {" + self.endLine
      # kStr += indent + "      rC[i][j] = 75.0;" + self.endLine
      # kStr += indent + "    }" + self.endLine
      # kStr += indent + "  }" + self.endLine
      # kStr += self.endLine

      kStr += "  /* write global C */" + self.endLine
      if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
        kStr += "  float type_mac_tmp;" + self.endLine
      if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
        kStr += "  double type_mac_tmp;" + self.endLine

      for a in range(0, kernel["ThreadTile0"]):
        for b in range(0, kernel["ThreadTile1"]):
          numEdges = 0
          #for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          if kernel["EdgeType"] == "Branch":
            kStr += "  if (globalC" \
                + tileChar0 + " + " \
                + str(a) + "*SG" + tileChar0 + "" + " < size" \
                + tileChar0 + ") {"
            numEdges += 1
          if kernel["EdgeType"] == "Branch":
            kStr += "  if (globalC" \
                + tileChar1 + " + " \
                + str(b) + "*SG" + tileChar1 + "" + " < size" \
                + tileChar1 + ") {"
            numEdges += 1

          kStr += "  TYPE_MAD_WRITE( C[ GLOBAL_C( (" + self.uint64Str + ")"
          for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
            kStr += " globalC" + indexChars[i]
            if i == kernel["ProblemType"]["Index0"]:
              kStr += " + " + str(a) + "*SG" + tileChar0
            if i == kernel["ProblemType"]["Index1"]:
              kStr += " + " + str(b) + "*SG" + tileChar1
            if i < kernel["ProblemType"]["NumIndicesC"]-1:
              kStr += ", (" + self.uint64Str + ")"
          kStr += ") ]"
          kStr += ", alpha"
          kStr += ", rC[%d+TT%s*%d]" % (a, tileChar0, b)
          if kernel["ProblemType"]["UseBeta"]:
            kStr += ", beta"
          kStr += ")"
          # debug printf
          #kStr += " printf(\\\"T[%u,%u] Cijk = %f\\\\n\\\", " + self.getLocalIdStr + "(0), " + self.getLocalIdStr + "(1), rC[" + str(a) + "][" + str(b) + "] );"

          # debug printf
          # kStr += "  printf(\\\"T[%u,%u] writing C[%u] = %f, %f, %f\\\\n\\\", " + self.getLocalIdStr + "(0), " + self.getLocalIdStr + "(1), GLOBAL_C(globalC0I, globalC1J), alpha, beta, rC[0][0]"
          # kStr += ");" + self.endLine
          # end debug printf

          for i in range(0,numEdges):
            kStr += " }"
          kStr += self.endLine
          #kStr += "  if (serial < 24) printf(\\\"T[%u,%u]%u C[%u] = %f\\\\n\\\", " + self.getLocalIdStr + "(0), " + self.getLocalIdStr + "(1), globalCK, serial, C[serial]);"


    ####################################
    # end kernel body
    ####################################
    kStr += self.endLine
    kStr += "}" + self.endLine

    ####################################
    # undefine definitions if merged
    ####################################
    if globalParameters["MergeFiles"] and self.backend == "HIP":
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
      kStr += "#undef GLOBAL_A%s" % (self.endLine)
      kStr += "#undef GLOBAL_B%s" % (self.endLine)
      kStr += "#undef DATA_TYPE%s" % (self.endLine)
      kStr += "#undef MICRO_TILE%s" % (self.endLine)
      kStr += "#undef LDS_OFFSET_B%s" % (self.endLine)
      kStr += "#undef LDS_NUM_ELEMENTS%s" % (self.endLine)
      kStr += "#undef NUM_THREADS%s" % (self.endLine)
      kStr += "#undef WORK_GROUP_MAPPING%s" % (self.endLine)
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

    # backend pre
    fileString += "\n"
    if self.backend == "OCL":
      fileString += "const char * const %s_src =\"" % (kernelName)

    # write kernel body
    fileString += self.getBody( kernel )

    # backend post
    if self.backend == "OCL":
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
    #fileString += "#ifndef KERNEL_" + kernelName.upper() + "_H\n"
    #fileString += "#define KERNEL_" + kernelName.upper() + "_H\n"
    if not globalParameters["MergeFiles"]:
      fileString += "#pragma once\n\n"
      fileString += "\n"
      if self.backend == "HIP":
        fileString += "#include <hip/hip_runtime.h>\n"
        fileString += "\n"
    if self.backend == "OCL":
      fileString += "extern const char * const %s_src;\n" % kernelName
    else:
      fileString += self.getSignature(kernel)
      fileString += ";\n"

    #fileString += "#endif\n\n"
    return fileString







################################################################################
# Transpose Cases
################################################################################
# traditional GEMM as NN, NT... transpose cases which are different speeds.
# how do those map to new dimensions and strides
# in new terminology, we can do long/fast loads along d0,d1 (i,j) but only short slower loads along dU (k), so we prefer dimensions d0,d1 to be the ones with shortest strides (1 preferably). If ever dU of one of the tensors is the dimension with stride 1, that tensors will get read relatively slow.

# N*: read A fast b/c
# old: if (kernel.order=="clblasColumnMajor")==(kernel.transA=="N"):
# new: unrollDimStrideGreaterThanTileDimStrideA == true

# *T: read B fast b/c
# old: if (kernel.order=="clblasColumnMajor")==(kernel.transB=="T"):
# new: unrollDimStrideLessThanTileDimStrideB == true

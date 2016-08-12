import os
import sys
import argparse
import copy

import Structs


################################################################################
# Make OpenCL Kernel String
################################################################################
class KernelWriter:

  indexChars = [ "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", \
      "T", "U", "V", "W", "X", "Y", "Z" ]

  ##############################################################################
  # Make OpenCL Kernel String
  ##############################################################################
  def __init__( self, backend ):
    self.backend = backend

    if self.backend.isOpenCL():
      # everything escaped extra b/c string
      self.endLine = "\\n\"\n\""
      self.endLinePP = "\\\\" + self.endLine
    else:
      self.endLine = "\n"
      self.endLinePP =  "\\" + self.endLine

    if self.backend.isOpenCL():
      self.getGroupIdStr = "get_group_id"
      self.getNumGroupsStr = "get_num_groups"
      self.getLocalIdStr = "get_local_id"
      self.getGlobalIdStr = "get_global_id"
      self.sharedDeclStr = "__local"
      self.sharedPtrStr = "__local"
      self.syncStr = "barrier(CLK_LOCAL_MEM_FENCE);"
      self.fenceStr = "mem_fence(CLK_LOCAL_MEM_FENCE);"
      self.fmaFStr = "mad"
      self.fmaDStr = "mad"
    else:
      self.getGroupIdStr = "hc_get_group_id"
      self.getNumGroupsStr = "hc_get_num_groups"
      self.getLocalIdStr = "hc_get_workitem_id"
      self.getGlobalIdStr = "hc_get_workitem_absolute_id"
      self.sharedDeclStr = "__shared__"
      self.sharedPtrStr = ""
      self.syncStr = "__syncthreads();"
      self.fenceStr = self.syncStr
      self.fmaFStr = "fmaf"
      self.fmaDStr = "fma"

    self.returnOnly = False


    pass


  ##############################################################################
  # get kernel name from operation
  ##############################################################################
  def getNameOperation(self, kernel):
    kernelName = ""

    # operation type
    kernelName += str(kernel.problem.operation.type)
    kernelName += "_"

    # data dataTypes
    kernelName += kernel.dataTypeC.toChar().upper()
    kernelName += kernel.dataTypeA.toChar().upper()
    kernelName += kernel.dataTypeB.toChar().upper()

    # alpha
    # if kernel.problem.operation.useAlpha:
    kernelName += kernel.dataTypeAlpha.toChar().upper()
    # else:
    #   kernelName += "0"

    # beta
    # if kernel.problem.operation.useBeta():
    kernelName += kernel.dataTypeBeta.toChar().upper()
    # else:
    #   kernelName += "0"

    kernelName += "_"

    # C dimensions
    kernelName += "C"
    for i in range(0, len(kernel.indexOrderC)):
      #kernelName += self.indexChars[kernel.indexOrderC[i]].lower()
      kernelName += self.indexChars[i].lower()

    # summation indices
    kernelName += "_S"
    for i in range(0,len(kernel.indexOrderSummation)):
      kernelName += self.indexChars[len(kernel.indexOrderC) + kernel.indexOrderSummation[i]].lower()

    # A dimensions
    kernelName += "_A"
    for i in range(0, len(kernel.problem.operation.indexAssignmentsA)):
      kernelName += self.indexChars[kernel.problem.operation.indexAssignmentsA[i]].lower()

    # B dimensions
    kernelName += "_B"
    for i in range(0,len(kernel.problem.operation.indexAssignmentsB)):
      kernelName += self.indexChars[kernel.problem.operation.indexAssignmentsB[i]].lower()


    return kernelName

  ##############################################################################
  # get kernel name from tile
  ##############################################################################
  def getNameTile(self, kernel):
    kernelName = ""
    # tile dim A
    if kernel.tensorAssignedDim0 == 0: #A assigned d0
      #print "tensorA has d0"
      #print "d0=" + str(kernel.indexAssignmentDim0) + ", d1=" + str(kernel.indexAssignmentDim1)
      kernelName += self.indexChars[kernel.indexAssignmentDim0].lower()
      kernelName += str(kernel.tile.workGroup[0])
      kernelName += kernel.tile.branch[0].getChar()
      kernelName += str(kernel.tile.microTile[0])
    else:
      #print "tensorA has d1"
      #print "d0=" + str(kernel.indexAssignmentDim0) + ", d1=" + str(kernel.indexAssignmentDim1)
      kernelName += self.indexChars[kernel.indexAssignmentDim1].lower()
      kernelName += str(kernel.tile.workGroup[1])
      kernelName += kernel.tile.branch[1].getChar()
      kernelName += str(kernel.tile.microTile[1])
    if kernel.unrollDimStrideGreaterThanTileDimStrideA:
      kernelName += "f"
    else:
      kernelName += "s"
    kernelName += "_"

    # tile dim B
    if kernel.tensorAssignedDim0 == 1: #B assigned d0
      #print "tensorB has d0"
      #print "d0=" + str(kernel.indexAssignmentDim0) + ", d1=" + str(kernel.indexAssignmentDim1)
      kernelName += self.indexChars[kernel.indexAssignmentDim0].lower()
      kernelName += str(kernel.tile.workGroup[0])
      kernelName += kernel.tile.branch[0].getChar()
      kernelName += str(kernel.tile.microTile[0])
    else:
      #print "tensorB has d1"
      #print "d0=" + str(kernel.indexAssignmentDim0) + ", d1=" + str(kernel.indexAssignmentDim1)
      kernelName += self.indexChars[kernel.indexAssignmentDim1].lower()
      kernelName += str(kernel.tile.workGroup[1])
      kernelName += kernel.tile.branch[1].getChar()
      kernelName += str(kernel.tile.microTile[1])
    # "f"ast or "s"low
    if kernel.unrollDimStrideLessThanTileDimStrideB:
      kernelName += "s"
    else:
      kernelName += "f"
    kernelName += "_"

    # load pattern
    kernelName += "nl" + str(kernel.numLoadsParaA)
    kernelName += "x" + str(kernel.numLoadsParaB)
    kernelName += "_"

    # unroll
    kernelName += self.indexChars[len(kernel.indexOrderC)+len(kernel.indexOrderSummation)-1].lower()
    if len(kernel.unrolls)>0:
      kernelName += str(kernel.unrolls[0])
      for i in range(1,len(kernel.unrolls)):
        kernelName += "_" + str(kernel.unrolls[i])

    # optimization level
    ppdStr = ""
    if kernel.ppdOffsets and not kernel.ppdLeadingStride:
      ppdStr = "O1"
    elif not kernel.ppdOffsets and kernel.ppdLeadingStride:
      ppdStr = "O2"
    elif kernel.ppdOffsets and kernel.ppdLeadingStride and not kernel.ppdAll:
      ppdStr = "O3"
    elif kernel.ppdAll:
      ppdStr = "O4"
    else:
      ppdStr = "O0"
    kernelName += "_" + ppdStr

    #print kernelName
    return kernelName


  ##############################################################################
  # get kernel name - DONE
  ##############################################################################
  def getName(self, kernel):
    return self.getNameOperation(kernel) + "_" + self.getNameTile(kernel)


  ##############################################################################
  # get kernel signature - DONE
  ##############################################################################
  def getSignature(self, kernel ):

    # determine chars for fast access
    indexChars = copy.deepcopy(self.indexChars)
    indexChars[kernel.indexAssignmentDim0] \
        = "0" + indexChars[kernel.indexAssignmentDim0]
    indexChars[kernel.indexAssignmentDim1] \
        = "1" + indexChars[kernel.indexAssignmentDim1]
    unrollChar = indexChars[kernel.indexOrderSummation[ \
        len(kernel.indexOrderSummation)-1] + len(kernel.indexOrderC)]
    tileChar0 = indexChars[kernel.indexAssignmentDim0]
    tileChar1 = indexChars[kernel.indexAssignmentDim1]
    tileCharA = tileChar0 if (kernel.tensorAssignedDim0==0) else tileChar1
    tileCharB = tileChar0 if (kernel.tensorAssignedDim0==1) else tileChar1
    tensorChar0 = "A" if (kernel.tensorAssignedDim0==0) else "B"
    tensorChar1 = "A" if (kernel.tensorAssignedDim1==0) else "B"

    s = ""
    # kernel name
    if self.backend.isOpenCL():
      s += "__attribute__((reqd_work_group_size(WG_" \
          + tileChar0 + ",WG_" + tileChar1 + ",1)))"
      s += self.endLine
      s += "__kernel "
    else:
      s += "extern \"C\"\n"
      s += "__global__ "
    s += "void %s" % ( self.getName(kernel) )
    s += "(" + self.endLine
    # pointers
    globalStr = "__global "
    if self.backend.isHIP():
      s += "  hipLaunchParm lp," + self.endLine
      globalStr = ""
    restrictStr = "restrict"
    if self.backend.isHIP():
      restrictStr = "__restrict__"
    s += "  " + globalStr + kernel.dataTypeC.toDevice(self.backend) \
        + "       *          C,"
    s += self.endLine
    s += "  " + globalStr + kernel.dataTypeA.toDevice(self.backend) \
        + " const * " + restrictStr + " A,"
    s += self.endLine
    s += "  " + globalStr + kernel.dataTypeB.toDevice(self.backend) \
        + " const * " + restrictStr + " B"

    # alpha & beta
    if kernel.useAlpha():
      s += "," + self.endLine + "  " \
          + kernel.dataTypeC.toDevice(self.backend) + " const alpha"
    if kernel.useBeta():
      s += "," + self.endLine + "  " \
          + kernel.dataTypeC.toDevice(self.backend) + " const beta"

    # offsets
    if not kernel.ppdOffsets:
      s += ( "," + self.endLine + "  unsigned int const offsetC,"
        + self.endLine +
        "  unsigned int const offsetA," + self.endLine +
        "  unsigned int const offsetB" )

    # strides
    firstStride = 0
    if kernel.ppdLeadingStride:
      firstStride = 1
    lastStrideC = len(kernel.indexOrderC)
    lastStrideA = len(kernel.problem.operation.indexAssignmentsA)
    lastStrideB = len(kernel.problem.operation.indexAssignmentsB)
    if kernel.ppdAll:
      lastStrideC = firstStride
      lastStrideA = firstStride
      lastStrideB = firstStride
    for i in range(firstStride, lastStrideC):
      s += "," + self.endLine + "  unsigned int const strideC" + indexChars[i]
    for i in range(firstStride, lastStrideA):
      s += "," + self.endLine + "  unsigned int const strideA" \
          + indexChars[kernel.problem.operation.indexAssignmentsA[i]]
    for i in range(firstStride, lastStrideB):
      s += "," + self.endLine + "  unsigned int const strideB" \
          + indexChars[kernel.problem.operation.indexAssignmentsB[i]]

    # sizes
    if not kernel.ppdAll:
      for i in range(0, len(kernel.indexOrderC)+len(kernel.indexOrderSummation)):
        s += "," + self.endLine + "  unsigned int const size" + indexChars[i]

    s += " )"
    return s



  ##############################################################################
  # make kernel body
  ##############################################################################
  def getBody( self, kernel ):

    # determine chars for fast access
    indexChars = copy.deepcopy(self.indexChars)
    indexChars[kernel.indexAssignmentDim0] \
        = "0" + indexChars[kernel.indexAssignmentDim0]
    indexChars[kernel.indexAssignmentDim1] \
        = "1" + indexChars[kernel.indexAssignmentDim1]

    # determine indices
    unrollChar = indexChars[kernel.indexOrderSummation[ \
        len(kernel.indexOrderSummation)-1] + len(kernel.indexOrderC)]
    tileChar0 = indexChars[kernel.indexAssignmentDim0]
    tileChar1 = indexChars[kernel.indexAssignmentDim1]
    tileCharA = tileChar0 if (kernel.tensorAssignedDim0==0) else tileChar1
    tileCharB = tileChar0 if (kernel.tensorAssignedDim0==1) else tileChar1
    tensorChar0 = "A" if (kernel.tensorAssignedDim0==0) else "B"
    tensorChar1 = "A" if (kernel.tensorAssignedDim1==0) else "B"

    ####################################
    # initializations
    kStr = ""
    kStr += self.endLine
    kStr += "/* %s */" % self.getName(kernel)
    kStr += self.endLine

    ####################################
    # kernel preprocessor definitions
    kStr += self.endLine
    kStr += "/* tile parameters */" + self.endLine
    kStr += "#define WG_%s  %2d%s" \
        % (tileChar0, kernel.tile.workGroup[0], self.endLine )
    kStr += "#define WG_%s  %2d%s" \
        % (tileChar1, kernel.tile.workGroup[1], self.endLine )
    kStr += "#define UT_" + tileChar0 + "  %2d%s" \
        % (kernel.tile.microTile[0], self.endLine )
    kStr += "#define UT_" + tileChar1 + "  %2d%s" \
        % (kernel.tile.microTile[1], self.endLine )
    kStr += "#define MT_" + tileChar0 + "  %2d%s" \
        % ((kernel.tile.workGroup[0] * kernel.tile.microTile[0]), self.endLine )
    kStr += "#define MT_" + tileChar1 + "  %2d%s" \
        % ((kernel.tile.workGroup[1] * kernel.tile.microTile[1]), self.endLine )
    kStr += "#define UNROLL %2d%s" \
        % (kernel.unrolls[0], self.endLine )
    kStr += "#define PAD     1" + self.endLine
    kStr += self.endLine

    ####################################
    # load grid
    # totalLoadsA  = (kernel.tile.workGroup[0]*kernel.tile.microTile[0]*kernel.unrolls[0]) \
    #     / (kernel.tile.workGroup[0]*kernel.tile.workGroup[1])
    # totalLoadsB  = (kernel.tile.workGroup[1]*kernel.tile.microTile[1]*kernel.unrolls[0]) \
    #     / (kernel.tile.workGroup[0]*kernel.tile.workGroup[1])
    # numLoadsParaA = kernel.numLoadsA
    # numLoadsParaB = kernel.numLoadsB
    # numLoadsPerpA = totalLoadsA / numLoadsParaA
    # numLoadsPerpB = totalLoadsB / numLoadsParaB

    # kStr += "/* total num loads */" + self.endLine
    # kStr += "#define NL_A ((MT_%s*UNROLL)/(WG_%s*WG_%s))%s" \
    #     % (tileCharA, tileChar0, tileChar1, self.endLine)
    # kStr += "#define NL_B ((MT_%s*UNROLL)/(WG_%s*WG_%s))%s" \
    #     % (tileCharB, tileChar0, tileChar1, self.endLine)
    # kStr += self.endLine
    
    # num loads
    kStr += "/* num loads parallel and perpendicular to coalesced dimension */" + self.endLine
    kStr += "#define NL_PARA_A %d%s" % (kernel.numLoadsParaA, self.endLine )
    kStr += "#define NL_PARA_B %d%s" % (kernel.numLoadsParaB, self.endLine )
    kStr += "#define NL_PERP_A %d%s" % (kernel.numLoadsPerpA, self.endLine )
    kStr += "#define NL_PERP_B %d%s" % (kernel.numLoadsPerpB, self.endLine )
    kStr += self.endLine

    if False:
      # load size
      if kernel.unrollDimStrideGreaterThanTileDimStrideA:
        kStr += "#define LS_PARA_A (MT_%s/NL_PARA_A)%s" \
            % (tileCharA, self.endLine)
        kStr += "#define LS_PERP_A (UNROLL/NL_PERP_A)" + self.endLine
      else:
        kStr += "#define LS_PARA_A (UNROLL/NL_PARA_A)%s" \
            % (self.endLine)
        kStr += "#define LS_PERP_A (MT_%s/NL_PERP_A)%s" \
            % ( tileCharA, self.endLine)
      if not kernel.unrollDimStrideLessThanTileDimStrideB:
        kStr += "#define LS_PARA_B (MT_%s/NL_PARA_B)%s" \
            % (tileCharB, self.endLine)
        kStr += "#define LS_PERP_B (UNROLL/NL_PERP_B)" + self.endLine
      else:
        kStr += "#define LS_PARA_B (UNROLL/NL_PARA_B)%s" \
            % (self.endLine)
        kStr += "#define LS_PERP_B (MT_%s/NL_PERP_B)%s" % (tileCharB, self.endLine)
    if True:
      # load size
      kStr += "/* load size parallel and perpendicular to coalesced dimension */" + self.endLine
      kStr += "#define LS_PARA_A %d%s" % (kernel.loadSizeParaA, self.endLine)
      kStr += "#define LS_PERP_A %d%s" % (kernel.loadSizePerpA, self.endLine)
      kStr += "#define LS_PARA_B %d%s" % (kernel.loadSizeParaB, self.endLine)
      kStr += "#define LS_PERP_B %d%s" % (kernel.loadSizePerpB, self.endLine)


    ####################################
    # global memory indices
    kStr += self.endLine
    kStr += "/* global memory indices */" + self.endLine
    # C
    kStr += "#define GLOBAL_C(IDX" \
        + indexChars[0]
    for i in range(1, len(kernel.indexOrderC)):
      kStr += ", IDX" + indexChars[i]
    indexChar = indexChars[0]
    kStr += ") ( (IDX" + indexChar + ")*strideC" + indexChar
    for i in range(1, len(kernel.indexOrderC)):
      indexChar = indexChars[i]
      kStr += " + (IDX" + indexChar + ")*strideC" + indexChar
    kStr += " )" + self.endLine
    # A
    kStr += "#define GLOBAL_A(IDX" \
        + indexChars[kernel.problem.operation.indexAssignmentsA[0]]
    for i in range(1, len(kernel.problem.operation.indexAssignmentsA)):
      kStr += ", IDX" + indexChars[kernel.problem.operation.indexAssignmentsA[i]]
    indexChar = indexChars[kernel.problem.operation.indexAssignmentsA[0]]
    kStr += ") ( (IDX" + indexChar + ")*strideA" + indexChar
    for i in range(1, len(kernel.problem.operation.indexAssignmentsA)):
      indexChar = indexChars[kernel.problem.operation.indexAssignmentsA[i]]
      kStr += " + (IDX" + indexChar + ")*strideA" + indexChar
    kStr += " )" + self.endLine
    # B
    kStr += "#define GLOBAL_B(IDX" \
        + indexChars[kernel.problem.operation.indexAssignmentsB[0]]
    for i in range(1, len(kernel.problem.operation.indexAssignmentsB)):
      kStr += ", IDX" + indexChars[kernel.problem.operation.indexAssignmentsB[i]]
    indexChar = indexChars[kernel.problem.operation.indexAssignmentsB[0]]
    kStr += ") ( (IDX" + indexChar + ")*strideB" + indexChar
    for i in range(1, len(kernel.problem.operation.indexAssignmentsB)):
      indexChar = indexChars[kernel.problem.operation.indexAssignmentsB[i]]
      kStr += " + (IDX" + indexChar + ")*strideB" + indexChar
    kStr += " )" + self.endLine
    kStr += self.endLine

    ####################################
    # data types
    kStr += self.endLine
    kStr += "/* data types */" + self.endLine
    kStr += "#define TYPE_A     %s%s" \
        % (kernel.dataTypeA.toDevice(self.backend), self.endLine)
    kStr += "#define TYPE_B     %s%s" \
        % (kernel.dataTypeB.toDevice(self.backend), self.endLine)
    kStr += "#define TYPE_C     %s%s" \
        % (kernel.dataTypeC.toDevice(self.backend), self.endLine)
    kStr += "#define TYPE_ALPHA %s%s" \
        % (kernel.dataTypeC.toDevice(self.backend), self.endLine)
    kStr += "#define TYPE_BETA  %s%s" \
        % (kernel.dataTypeC.toDevice(self.backend), self.endLine)

    if self.backend.isOpenCL():
      kStr += "#define MAD(A,B,DST) mad(A,B,DST)"
    else:
      kStr += "#define MAD(A,B,DST) DST += A*B"
    kStr += self.endLine

    if self.backend.isHIP():
      kStr += "#define s0 x" + self.endLine
      kStr += "#define s1 y" + self.endLine
    kStr += self.endLine

    ####################################
    # MADs
    kStr += "/* MADs */" + self.endLine
    if kernel.dataTypeC.isReal():
      # real data
      kStr += "#define TYPE_MAD(MULA,MULB,DST) " \
          + "DST = MAD(MULA,MULB,DST);" + self.endLine
      if kernel.useAlpha():
        if kernel.useBeta():
          # dst = alpha*reg + beta*dst
          kStr += "#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) " \
              + "DST = (ALPHA)*(REG) + (BETA)*(DST);" + self.endLine
        else:
          # dst = alpha*reg
          kStr += "#define TYPE_MAD_WRITE(DST,ALPHA,REG) " \
              + "DST = (ALPHA)*(REG);" + self.endLine
      else:
        if kernel.useBeta():
          # dst = reg + beta*dst
          kStr += "#define TYPE_MAD_WRITE(DST,REG,BETA) " \
              + "DST = (REG) + (BETA)*(DST);" + self.endLine
        else:
          # dst = reg
          kStr += "#define TYPE_MAD_WRITE(DST,REG) " \
              + "DST = (REG);" + self.endLine
    else:
      # complex data
      if not kernel.dataTypeA.isConjugate() and not kernel.dataTypeB.isConjugate():
        # neither conjugate
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAD( -MULA.s1, MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif kernel.dataTypeA.isConjugate() and not kernel.dataTypeB.isConjugate():
        # A conjugate (negate imaginary A.s1)
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLinePP +
          "  DST.s0 = MAD(  MULA.s1, MULB.s1, DST.s0 ); " + self.endLinePP +
          "  DST.s1 = MAD(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLinePP +
          "  DST.s1 = MAD( -MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif not kernel.dataTypeA.isConjugate() and kernel.dataTypeB.isConjugate():
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
      if kernel.useAlpha():
        if kernel.useBeta():
          # dst = alpha*reg + beta*dst
          kStr += (
            "#define TYPE_MAD_WRITE( DST, ALPHA, REG, BETA ) "+self.endLinePP +
            "  /* (1) */ " + self.endLinePP +
            "  type_fma_tmp = REG.s0; " + self.endLinePP +
            "  REG.s0 *= ALPHA.s0; " + self.endLinePP +
            "  REG.s0 = MAD( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLinePP +
            "  REG.s1 *= ALPHA.s0; " + self.endLinePP +
            "  REG.s1 = MAD(  ALPHA.s1, type_fma_tmp, REG.s1 ); "+self.endLinePP+
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
            "  type_fma_tmp = REG.s0; " + self.endLinePP +
            "  REG.s0 *= ALPHA.s0; " + self.endLinePP +
            "  REG.s0 = MAD( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLinePP +
            "  REG.s1 *= ALPHA.s0; " + self.endLinePP +
            "  REG.s1 = MAD(  ALPHA.s1, type_fma_tmp, REG.s1 ); "+self.endLinePP+
            "  /* (3) */ " + self.endLinePP +
            "  DST = REG;" + self.endLine )
      else:
        if kernel.useBeta():
          # dst = reg + beta*dst
          kStr += (
            "#define TYPE_MAD_WRITE( DST, REG, BETA ) " + self.endLinePP +
            "  /* (2) */ " + self.endLinePP +
            "  REG.s0 = MAD(  BETA.s0, DST.s0, REG.s0 ); " + self.endLinePP +
            "  REG.s0 = MAD( -BETA.s1, DST.s1, REG.s0 ); " + self.endLinePP +
            "  REG.s1 = MAD(  BETA.s0, DST.s1, REG.s1 ); " + self.endLinePP +
            "  REG.s1 = MAD(  BETA.s1, DST.s0, REG.s1 ); " + self.endLinePP +
            "  /* (3) */ " + self.endLinePP +
            "  DST = REG;" + self.endLine )
        else:
          # dst = reg
          kStr += (
            "#define TYPE_MAD_WRITE( DST, REG ) " + self.endLinePP +
            "  /* (3) */ " + self.endLinePP +
            "  DST = REG;" + self.endLine )

    ####################################
    # micro-tile
    kStr += self.endLine
    kStr += "/* %dx%d micro-tile */%s" % (kernel.tile.microTile[0], kernel.tile.microTile[1], self.endLine)

    kStr += "#define MICRO_TILE " + self.endLinePP
    for a in range(0, kernel.tile.microTile[0]):
      kStr += "  rA[%d] = localA[offA + %d*WG_%s]; %s" \
          % (a, a, tileChar0, self.endLinePP)
    for b in range(0, kernel.tile.microTile[1]):
      kStr += "  rB[%d] = localB[offB + %d*WG_%s]; %s" \
          % (b, b, tileChar1, self.endLinePP)
    kStr += "  offA += (MT_" + tileChar0 + "+PAD); " + self.endLinePP
    kStr += "  offB += (MT_" + tileChar1 + "+PAD); " + self.endLinePP
    for a in range(0, kernel.tile.microTile[0]):
      for b in range(0, kernel.tile.microTile[1]):
        kStr += "  TYPE_MAD(rA[%d],rB[%d],rC[%d][%d]); %s" % (a, b, a, b, self.endLinePP)
    kStr += "  " + self.fenceStr + self.endLine
    kStr += self.endLine

    ####################################
    # preprocessor definitions of kernel arguments
    kStr += "/* preprocessor definitions of kernel arguments*/" + self.endLine
    firstStride = 0
    lastStrideC = len(kernel.indexOrderC)
    lastStrideA = len(kernel.problem.operation.indexAssignmentsA)
    lastStrideB = len(kernel.problem.operation.indexAssignmentsB)
    if kernel.ppdAll:
      #optimize all
      pass
    elif kernel.ppdLeadingStride:
      # optimize leading stride
      lastStrideC = 1
      lastStrideA = 1
      lastStrideB = 1
    else:
      # no optimization, do none
      lastStrideC = 0
      lastStrideA = 0
      lastStrideB = 0

    for i in range(firstStride, lastStrideC):
      kStr += "#define strideC" + indexChars[i] + " " + str(kernel.problem.tensorC.dimensions[i].stride) + self.endLine
    for i in range(firstStride, lastStrideA):
      kStr += "#define strideA" + indexChars[kernel.problem.operation.indexAssignmentsA[i]] + " " + str(kernel.problem.tensorA.dimensions[i].stride) + self.endLine
    for i in range(firstStride, lastStrideB):
      kStr += "#define strideB" + indexChars[kernel.problem.operation.indexAssignmentsB[i]] + " " + str(kernel.problem.tensorB.dimensions[i].stride) + self.endLine

    # sizes
    if kernel.ppdAll:
      for i in range(0, len(kernel.indexOrderC)+len(kernel.indexOrderSummation)):
        kStr += "#define size" + indexChars[i] + " "
        # which index of tensorA or B is assigned to that index; use its size
        size = -1
        for j in range(0, len(kernel.problem.operation.indexAssignmentsA)):
          index = kernel.problem.operation.indexAssignmentsA[j]
          if index == i:
            size = kernel.problem.tensorA.dimensions[j].size
            break
        for j in range(0, len(kernel.problem.operation.indexAssignmentsB)):
          index = kernel.problem.operation.indexAssignmentsB[j]
          if index == i:
            size = kernel.problem.tensorB.dimensions[j].size
            break
        kStr += str(size) + self.endLine
    kStr += self.endLine + self.endLine

    ####################################
    # function signature
    ####################################
    kStr += "/* kernel */" + self.endLine
    kStr += self.getSignature(kernel)
    kStr += " {" + self.endLine


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
    if not kernel.ppdOffsets:
      kStr += ("  C += offsetC;" + self.endLine +
        "  A += offsetA;" + self.endLine +
        "  B += offsetB;" + self.endLine )

    ####################################
    # allocate registers
    kStr += self.endLine
    kStr += (
      "  /* allocate registers */" + self.endLine +
      "  TYPE_C rC[UT_" + tileChar0 + "][UT_" + tileChar1 + "] "
          + "= {{0}};" + self.endLine +
      "  TYPE_A rA[UT_" + tileChar0 + "];" + self.endLine +
      "  TYPE_B rB[UT_" + tileChar1 + "];" + self.endLine )


    ####################################
    # allocate local memory
    kStr += self.endLine
    kStr += (
      "  /* allocate local memory */" + self.endLine +
      "  " + self.sharedDeclStr + " TYPE_A localA[UNROLL*(MT_" + tileChar0 + "+PAD)];" \
          + self.endLine +
      "  "+self.sharedDeclStr + " TYPE_B localB[UNROLL*(MT_" + tileChar1 + "+PAD)];" \
          + self.endLine )

    ####################################
    # c indices
    ####################################
    # kernel.indexOrderC - performance defined
    # kernel.indexOrderSummation - performance defined
    # kernel.indexAssignmentsA - user defined
    # kernel.indexAssignmentsB - user defined
    # convert self.getGroupIdStr(0) to however many c indices there are


    # work-group free indices
    kStr += self.endLine
    kStr += "  /* c indices (group) */" + self.endLine


    if kernel.transposeWorkGroupOrder:
      # swap order in which work-groups cover C
      kStr += "  unsigned int groupSerial = %s(0) * %s(1) + %s(1);%s" \
        % (self.getGroupIdStr, self.getNumGroupsStr, self.getGroupIdStr, self.endLine)
      kStr += "  unsigned int g%s = groupSerial %% %s(0);%s" % (tileChar0, self.getNumGroupsStr, self.endLine)
      kStr += "  unsigned int g%s = groupSerial / %s(0);%s" % (tileChar1, self.getNumGroupsStr, self.endLine)
    else:
      kStr += "  unsigned int g" + tileChar0 + " = " \
          + self.getGroupIdStr + "(0);" \
          + " // d0, tensor" + tensorChar0 + self.endLine
      kStr += "  unsigned int g" + tileChar1 + " = " \
          + self.getGroupIdStr + "(1);" \
          + " // d1, tensor" + tensorChar1 + self.endLine

    # other free indices
    nonTileFreeIndices = copy.deepcopy(kernel.indexOrderC)
    nonTileFreeIndices.remove(kernel.indexAssignmentDim0)
    nonTileFreeIndices.remove(kernel.indexAssignmentDim1)
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
    kStr += "  unsigned int l" + tileChar0 \
        + " = " + self.getLocalIdStr + "(0); // d0" + self.endLine
    kStr += "  unsigned int l" + tileChar1 \
        + " = " + self.getLocalIdStr + "(1); // d1" + self.endLine
    kStr += "  unsigned int loadSerial = l" + tileChar0 \
        + " + l" + tileChar1 + "*WG_" + tileChar0 \
        + ";" + self.endLine

    kStr += "  unsigned int a" + tileCharA + " = "
    if kernel.unrollDimStrideGreaterThanTileDimStrideA:
      kStr += "loadSerial%LS_PARA_A;" + self.endLine
    else:
      kStr += "loadSerial/LS_PARA_A;" + self.endLine

    kStr += "  unsigned int b" + tileCharB + " = "
    if kernel.unrollDimStrideLessThanTileDimStrideB:
      kStr += "loadSerial/LS_PARA_B;" + self.endLine
    else:
      kStr += "loadSerial%LS_PARA_B;" + self.endLine
    kStr += self.endLine

    kStr += "  /* unrolled summation index */" + self.endLine
    kStr += "  unsigned int a" + unrollChar + " = "
    if kernel.unrollDimStrideGreaterThanTileDimStrideA:
      kStr += "loadSerial/LS_PARA_A;" + self.endLine
    else:
      kStr += "loadSerial%LS_PARA_A;" + self.endLine

    kStr += "  unsigned int b" + unrollChar + " = "
    if kernel.unrollDimStrideLessThanTileDimStrideB:
      kStr += "loadSerial%LS_PARA_B;" + self.endLine
    else:
      kStr += "loadSerial/LS_PARA_B;" + self.endLine
    kStr += self.endLine

    # other non-unrolled summation indices
    kStr += "  /* other non-unrolled summation indices (all start at zero) */" + self.endLine
    for i in range(0,len(kernel.indexOrderSummation)-1):
      index = i + len(kernel.indexOrderC)
      kStr += "#define a" + indexChars[index] + " 0" + self.endLine
      kStr += "#define b" + indexChars[index] + " 0" + self.endLine
    kStr += self.endLine



    ####################################
    # offset global pointers
    ####################################
    kStr += "  /* where will this thread read from global memory */" + self.endLine
    kStr += "  A += GLOBAL_A( "
    for i in range(0, len(kernel.problem.operation.indexAssignmentsA)):
      index = kernel.problem.operation.indexAssignmentsA[i]
      if index < len(kernel.indexOrderC): # c index
        if index == kernel.indexAssignmentTileA[1]: # this index is A's tile index
          kStr += "a%s+g%s*MT_%s" % (tileCharA, tileCharA, tileCharA)
        else: # just a group index
          kStr += "g" + indexChars[index]
      else: # summation index
        kStr += "a" + indexChars[index]
      if i < len(kernel.problem.operation.indexAssignmentsA)-1:
        kStr += ", "
    kStr += " );" + self.endLine

    kStr += "  B += GLOBAL_B( "
    for i in range(0, len(kernel.problem.operation.indexAssignmentsB)):
      index = kernel.problem.operation.indexAssignmentsB[i]
      if index < len(kernel.indexOrderC): # c index
        if index == kernel.indexAssignmentTileB[1]: # this index is B's tile index
          kStr += "b%s+g%s*MT_%s" % (tileCharB, tileCharB, tileCharB)
        else: # just a group index
          kStr += "g" + indexChars[index]
      else: # summation index
        kStr += "b" + indexChars[index]
      if i < len(kernel.problem.operation.indexAssignmentsB)-1:
        kStr += ", "
    kStr += " );" + self.endLine
    kStr += self.endLine

    # if udsgttdsA: tileCharA+g*MT, aUnroll
    # nt udsgttdsA: aUnroll, tileCharA+g*MT

    """
    if kernel.unrollDimStrideGreaterThanTileDimStrideA:
      kStr += "  A += GLOBAL_A( a%s+g%s*MT_%s, a%s" \
          % (tileCharA, tileCharA, tileCharA, unrollChar)
    else:
      kStr += "  A += GLOBAL_A( a%s, a%s+g%s*MT_%s" \
          % (unrollChar, tileCharA, tileCharA, tileCharA)
    for i in range(2, len(kernel.problem.operation.indexAssignmentsA)):
      kStr += ", g%s" % indexChars[i]
    kStr += " );" + self.endLine

    if kernel.unrollDimStrideLessThanTileDimStrideB:
      kStr += "  B += GLOBAL_B( b%s, b%s+g%s*MT_%s" \
          % (unrollChar, tileCharB, tileCharB, tileCharB)
    else:
      kStr += "  B += GLOBAL_B( b%s+g%s*MT_%s, b%s" \
          % (tileCharB, tileCharB, tileCharB, unrollChar)
    for i in range(2, len(kernel.problem.operation.indexAssignmentsB)):
      kStr += ", g%s" % indexChars[i]
    kStr += " );" + self.endLine
    kStr += self.endLine
    """

    ####################################
    # offset local pointers
    ####################################
    kStr += "  /* where will this thread write to local memory */" + self.endLine
    kStr += "  %s TYPE_A *lA = localA + a%s + a%s*(MT_%s+PAD);%s" \
        % (self.sharedPtrStr, tileCharA, unrollChar, tileCharA, self.endLine)
    kStr += "  %s TYPE_B *lB = localB + b%s + b%s*(MT_%s+PAD);%s" \
        % (self.sharedPtrStr, tileCharB, unrollChar, tileCharB, self.endLine)
    kStr += self.endLine

    ####################################


    # debug printf - global data
    #kStr += "  printf(\\\"T[%u,%u] A[%u] = %f; B[%u] = %f\\\\n\\\", " + self.getLocalIdStr + "(0), " + self.getLocalIdStr + "(1), loadSerial, A[loadSerial], loadSerial, B[loadSerial]"
    #kStr += ");" + self.endLine
    # end debug printf


    # multidim if (kernel.order=="clblasColumnMajor")==(kernel.transA=="N"):
    #tensorAssignedToTileDim = []
    #if kernel.tensorAssignedDim0:
    #  tensorAssignedToTileDim.append(kernel.problem.operation.
    #unrollStrideGreaterThanTileA
    #kernel.unrollDimStrideGreaterThanTileDimStrideA = kernel.indexAssignmentDim0 \
     #   > kernel.indexOrderSummation[len(kernel.indexOrderSummation)-1]
    #kernel.unrollDimStrideLessThanTileDimStrideB = kernel.indexAssignmentDim1 \
    #    > kernel.indexOrderSummation[len(kernel.indexOrderSummation)-1]


    # kStr += "  bool validC ="
    # if kernel.tile.branch[0]:
    #   kStr += " (globalC" \
    #       + tileChar0 + " + " \
    #       + str(a) + "*WG_" + tileChar0 + "" + " < size" \
    #       + tileChar0 + ")"
    # if kernel.tile.branch[0] and kernel.tile.branch[1]:
    #   kStr += " &&"
    # if kernel.tile.branch[1]:
    #   kStr += " (globalC" \
    #       + tileChar1 + " + " \
    #       + str(b) + "*WG_" + tileChar1 + "" + " < size" \
    #       + tileChar1 + ")"
    # kStr += ";" + self.endLine

    #if self.returnOnly:
    #  kStr += "return;" + self.endLine + "#if 0" + self.endLine


    ####################################
    # summations loops
    ####################################
    indent = "  "
    kStr += indent + "/* iterate over summation indice(s) */" + self.endLine
    for i in range(0,len(kernel.indexOrderSummation)):
      indexChar = indexChars[kernel.indexOrderSummation[i] \
          + len(kernel.indexOrderC)]
      kStr += indent + "unsigned int sumIter" + indexChar \
          + " = size" + indexChar
      if i == len(kernel.indexOrderSummation)-1:
        kStr += " / UNROLL"
      kStr += ";" + self.endLine
      kStr += indent + "do {" + self.endLine
      indent += "  "
    kStr += self.endLine
    
    # 1st barrier
    kStr += indent + self.syncStr + self.endLine

    # zeroString for real and complex
    if self.backend.isOpenCL():
      zeroStringC = kernel.dataTypeC.zeroStringOpenCL()
      zeroStringA = kernel.dataTypeA.zeroStringOpenCL()
      zeroStringB = kernel.dataTypeB.zeroStringOpenCL()
    else:
      zeroStringC = kernel.dataTypeC.zeroStringHIP()
      zeroStringA = kernel.dataTypeA.zeroStringHIP()
      zeroStringB = kernel.dataTypeB.zeroStringHIP()
    
    ####################################
    # load A
    ####################################
    kStr += indent + "/* load A global -> local */" + self.endLine
    if kernel.loadRequiresFewerThreadsA():
      kStr += indent + "if ( loadSerial < %d ) {%s" \
          % (kernel.loadSizeParaA*kernel.loadSizePerpA, self.endLine)
      indent += "  "
    for perp in range(0, kernel.numLoadsPerpA):
      for para in range(0, kernel.numLoadsParaA):
        kStr += indent
        condPara = (para==kernel.numLoadsParaA-1 and kernel.lastLoadRequiresGuardParaA())
        condPerp = (perp==kernel.numLoadsPerpA-1 and kernel.lastLoadRequiresGuardPerpA())
        if condPara or condPerp:
          kStr += "if ( "
          if condPara:
            kStr += "a%s < %d" % (unrollChar if not kernel.unrollDimStrideGreaterThanTileDimStrideA else tileCharA, kernel.totalLoadSizeParaA % kernel.loadSizeParaA )
          if condPerp:
            if condPara:
              kStr += " && "
            kStr += "a%s < %d" % (unrollChar if kernel.unrollDimStrideGreaterThanTileDimStrideA else tileCharA, kernel.totalLoadSizePerpA % kernel.loadSizePerpA )
          kStr += " ) { "

        kStr += "lA[ %d*LS_PARA_A" % para
        if not kernel.unrollDimStrideGreaterThanTileDimStrideA:
          kStr += "*(MT_%s+PAD)" % tileCharA
        kStr += " + %d*LS_PERP_A" % perp
        if kernel.unrollDimStrideGreaterThanTileDimStrideA:
          kStr += "*(MT_%s+PAD)" % tileCharA
        kStr += " ] = "
        
        if not kernel.tile.branch[0].isNone():
          kStr += "( a%s+g%s*MT_%s+" % ( tileCharA, tileCharA, tileCharA)
          if not kernel.unrollDimStrideGreaterThanTileDimStrideA:
            kStr += "%d*LS_PERP_A" % (perp)
          else:
            kStr += "%d*LS_PARA_A" % (para)
          kStr += " >= size%s)" %( tileCharA )
          kStr += " ? %s : " %( zeroStringA )

        kStr += "A[ %d*LS_PARA_A + %d*LS_PERP_A*strideA%s];" \
            % (para, perp, unrollChar if kernel.unrollDimStrideGreaterThanTileDimStrideA else tileCharA)
        if condPara or condPerp:
          kStr += " }"
        kStr += self.endLine
    if kernel.loadRequiresFewerThreadsA():
      indent = indent[2:]
      kStr += indent + "}" + self.endLine
    kStr += self.endLine
    
    ####################################
    # load B
    ####################################
    kStr += indent + "/* load B global -> local */" + self.endLine
    if kernel.loadRequiresFewerThreadsB():
      kStr += indent + "if ( loadSerial < %d ) {%s" \
          % (kernel.loadSizeParaB*kernel.loadSizePerpB, self.endLine)
      indent += "  "
    for perp in range(0, kernel.numLoadsPerpB):
      for para in range(0, kernel.numLoadsParaB):
        kStr += indent
        condPara = (para==kernel.numLoadsParaB-1 and kernel.lastLoadRequiresGuardParaB())
        condPerp = (perp==kernel.numLoadsPerpB-1 and kernel.lastLoadRequiresGuardPerpB())
        if condPara or condPerp:
          kStr += "if ( "
          if condPara:
                kStr += "b%s < %d" % (unrollChar if kernel.unrollDimStrideLessThanTileDimStrideB else tileCharB, kernel.totalLoadSizeParaB % kernel.loadSizeParaB )
          if condPerp:
            if condPara:
              kStr += " && "
            kStr += "b%s < %d" % (unrollChar if not kernel.unrollDimStrideLessThanTileDimStrideB else tileCharB, kernel.totalLoadSizePerpB % kernel.loadSizePerpB )
          kStr += " ) { "

        kStr += "lB[ %d*LS_PARA_B" % para
        if kernel.unrollDimStrideLessThanTileDimStrideB:
          kStr += "*(MT_%s+PAD)" % tileCharB
        kStr += " + %d*LS_PERP_B" % perp
        if not kernel.unrollDimStrideLessThanTileDimStrideB:
          kStr += "*(MT_%s+PAD)" % tileCharB
        kStr += " ] = "

        if not kernel.tile.branch[1].isNone():
          kStr += "( b%s+g%s*MT_%s+" % ( tileCharB, tileCharB, tileCharB)
          if kernel.unrollDimStrideLessThanTileDimStrideB:
            kStr += "%d*LS_PERP_B" % (perp)
          else:
            kStr += "%d*LS_PARA_B" % (para)
          kStr += " >= size%s)" % (tileCharB )
          kStr += " ? %s : " % ( zeroStringB )
        
        kStr += "B[ %d*LS_PARA_B + %d*LS_PERP_B*strideB%s];" \
            % (para, perp, unrollChar if not kernel.unrollDimStrideLessThanTileDimStrideB else tileCharB)
        if condPara or condPerp:
          kStr += " }"
        kStr += self.endLine
    if kernel.loadRequiresFewerThreadsB():
      indent = indent[2:]
      kStr += indent + "}" + self.endLine
    kStr += self.endLine

    # 2nd barrier
    kStr += (
      indent + self.syncStr + self.endLine +
      indent + "unsigned int offA = l" + tileChar0 + "; // d0" + self.endLine +
      indent + "unsigned int offB = l" + tileChar1 + "; // d1" + self.endLine )


    # # LDS state
    # kStr += indent + "/* print LDS state */" + self.endLine
    # kStr += indent + "if ( gJ==0 && gL==0 && g1K==0 && g0I==0 && loadSerial == 0) {" + self.endLine
    # kStr += indent + "  for (unsigned int u = 0; u < UNROLL; u++) {" + self.endLine
    # kStr += indent + "    for (unsigned int i = 0; i < MT_" + tileChar0 + "; i++) {" + self.endLine
    # kStr += indent + "      printf(\\\"[%u,%u,%u,%u][%u,%u,%u][%02u,%02u] a=%f; b=%f\\\\n\\\", gJ, gL, g1K, g0I, sumIterM, sumIterN, sumIterO, u, i, localA[i+u*(MT_"+tileChar0+"+PAD)], localB[i+u*(MT_"+tileChar0+"+PAD)] );" + self.endLine
    # # kStr += indent + "      printf(\\\"hi %u\\\\n\\\", size0I);" + self.endLine
    # # kStr += indent + "      printf(\\\"hi\\\\n\\\");" + self.endLine
    # kStr += indent + "    }" + self.endLine
    # kStr += indent + "  }" + self.endLine
    # kStr += indent + "}" + self.endLine
    # # [work-group id] idx=%i a=%f; b=%f
   

    ####################################
    # do fmas
    kStr += self.endLine
    kStr += indent + "/* do fmas */" + self.endLine
    for u in range(0, kernel.unrolls[0]):
      kStr += indent + "MICRO_TILE" + self.endLine
    kStr += self.endLine

    # debug printf - accumulation in registers
    # kStr += "  if (validC) printf(\\\"T[%u,%u] rC = %f g=%u\\\\n\\\", " + self.getLocalIdStr + "(0), " + self.getLocalIdStr + "(1), rC[0][0], GLOBAL_C(globalC0I, globalC1J) );" + self.endLine
    # end debug printf
    # kStr += indent + "if ( gJ==0 && gL==0 && g1K==0 && g0I==0 && loadSerial == 0 ) printf(\\\"[%u,%u,%u,%u] m=%u, n=%u, o=%u, r[0][0]=%.0f\\\\n\\\", gJ, gL, g1K, g0I, sumIterM, sumIterN, sumIterO, rC[0][0] );" + self.endLine
    

    ########################################################################
    # BEGIN UNROLL=1 LOOP
    ########################################################################


    # if another loop, close current unrolled loops
    if len(kernel.unrolls) > 1:
      loopChar = indexChars[kernel.indexOrderSummation[len(kernel.indexOrderSummation)-1] \
          + len(kernel.indexOrderC)]
      # advance A, B along summation dimension
      kStr += indent + "A += strideA" + loopChar + "*UNROLL;" + self.endLine
      kStr += indent + "B += strideB" + loopChar + "*UNROLL;" + self.endLine
      indent = indent[2:]
      # close do-while loop
      kStr += indent + "} while (--sumIter" + loopChar + " > 0);" + self.endLine
      kStr += self.endLine
      kStr += indent + self.syncStr + self.endLine


      ####################################
      # summations loops
      kStr += indent + "/* unroll=1 loop */" + self.endLine
      kStr += indent + "sumIter" + indexChar + " = size" + indexChar + " % UNROLL;" + self.endLine
      kStr += self.endLine


      ####################################
      # load A single
      ####################################
      kStr += indent + "/* load A global -> local */" + self.endLine
      if kernel.loadRequiresFewerThreadsA():
        kStr += indent + "if ( loadSerial < %d ) {%s" \
            % (kernel.loadSizeParaA*kernel.loadSizePerpA, self.endLine)
        indent += "  "
      for perp in range(0, kernel.numLoadsPerpA):
        for para in range(0, kernel.numLoadsParaA):
          kStr += indent
          condPara = (para==kernel.numLoadsParaA-1 and kernel.lastLoadRequiresGuardParaA())
          condPerp = (perp==kernel.numLoadsPerpA-1 and kernel.lastLoadRequiresGuardPerpA())
          if condPara or condPerp:
            kStr += "if ( "
            if condPara:
              kStr += "a%s < %d" % (unrollChar if not kernel.unrollDimStrideGreaterThanTileDimStrideA else tileCharA, kernel.totalLoadSizeParaA % kernel.loadSizeParaA )
            if condPerp:
              if condPara:
                kStr += " && "
              kStr += "a%s < %d" % (unrollChar if kernel.unrollDimStrideGreaterThanTileDimStrideA else tileCharA, kernel.totalLoadSizePerpA % kernel.loadSizePerpA )
            kStr += " ) { "
          kStr += "lA[ %d*LS_PARA_A" % para
          if not kernel.unrollDimStrideGreaterThanTileDimStrideA:
            kStr += "*(MT_%s+PAD)" % tileCharA
          kStr += " + %d*LS_PERP_A" % perp
          if kernel.unrollDimStrideGreaterThanTileDimStrideA:
            kStr += "*(MT_%s+PAD)" % tileCharA
          kStr += " ] = "
          # guard around K
          kStr += " ( a%s + " % (unrollChar)
          if kernel.unrollDimStrideGreaterThanTileDimStrideA:
            kStr += "%d*LS_PERP_A >= sumIter%s )" % (perp, unrollChar)
          else:
            kStr += "%d*LS_PARA_A >= sumIter%s )" % (para, unrollChar)
          # guard around branch
          if not kernel.tile.branch[0].isNone():
            kStr += " || "
            kStr += "( a%s+g%s*MT_%s+" % ( tileCharA, tileCharA, tileCharA)
            if not kernel.unrollDimStrideGreaterThanTileDimStrideA:
              kStr += "%d*LS_PERP_A" % (perp)
            else:
              kStr += "%d*LS_PARA_A" % (para)
            kStr += " >= size%s)" %( tileCharA )
          kStr += " ? %s : " % zeroStringA
          kStr += "A[ %d*LS_PARA_A + %d*LS_PERP_A*strideA%s];" \
              % (para, perp, unrollChar if kernel.unrollDimStrideGreaterThanTileDimStrideA else tileCharA)
          if condPara or condPerp:
            kStr += " }"
          kStr += self.endLine
      if kernel.loadRequiresFewerThreadsA():
        indent = indent[2:]
        kStr += indent + "}" + self.endLine
      kStr += self.endLine


      ####################################
      # load B single
      ####################################
      kStr += indent + "/* load B global -> local */" + self.endLine
      if kernel.loadRequiresFewerThreadsB():
        kStr += indent + "if ( loadSerial < %d ) {%s" \
            % (kernel.loadSizeParaB*kernel.loadSizePerpB, self.endLine)
        indent += "  "
      for perp in range(0, kernel.numLoadsPerpB):
        for para in range(0, kernel.numLoadsParaB):
          kStr += indent
          condPara = (para==kernel.numLoadsParaB-1 and kernel.lastLoadRequiresGuardParaB())
          condPerp = (perp==kernel.numLoadsPerpB-1 and kernel.lastLoadRequiresGuardPerpB())
          if condPara or condPerp:
            kStr += "if ( "
            if condPara:
                  kStr += "b%s < %d" % (unrollChar if kernel.unrollDimStrideLessThanTileDimStrideB else tileCharB, kernel.totalLoadSizeParaB % kernel.loadSizeParaB )
            if condPerp:
              if condPara:
                kStr += " && "
              kStr += "b%s < %d" % (unrollChar if not kernel.unrollDimStrideLessThanTileDimStrideB else tileCharB, kernel.totalLoadSizePerpB % kernel.loadSizePerpB )
            kStr += " ) { "

          kStr += "lB[ %d*LS_PARA_B" % para
          if kernel.unrollDimStrideLessThanTileDimStrideB:
            kStr += "*(MT_%s+PAD)" % tileCharB
          kStr += " + %d*LS_PERP_B" % perp
          if not kernel.unrollDimStrideLessThanTileDimStrideB:
            kStr += "*(MT_%s+PAD)" % tileCharB
          kStr += " ] = "
          # guard around k
          kStr += "( b%s + " % (unrollChar)
          if not kernel.unrollDimStrideLessThanTileDimStrideB:
            kStr += "%d*LS_PERP_B >= sumIter%s )" % (perp, unrollChar)
          else:
            kStr += "%d*LS_PARA_B >= sumIter%s )" % (para, unrollChar)
          # guard branch
          if not kernel.tile.branch[1].isNone():
            kStr += " || "
            kStr += "( b%s+g%s*MT_%s+" % ( tileCharB, tileCharB, tileCharB)
            if kernel.unrollDimStrideLessThanTileDimStrideB:
              kStr += "%d*LS_PERP_B" % (perp)
            else:
              kStr += "%d*LS_PARA_B" % (para)
            kStr += " >= size%s) " % (tileCharB )
         
          kStr += " ? %s : " % zeroStringB
          kStr += "B[ %d*LS_PARA_B + %d*LS_PERP_B*strideB%s];" \
              % (para, perp, unrollChar if not kernel.unrollDimStrideLessThanTileDimStrideB else tileCharB)
          if condPara or condPerp:
            kStr += " }"
          kStr += self.endLine
      if kernel.loadRequiresFewerThreadsB():
        indent = indent[2:]
        kStr += indent + "}" + self.endLine
      kStr += self.endLine
      kStr += indent + self.syncStr + self.endLine

      # full end loop b/c local full of zeros
      kStr += indent + "/* full unroll loop */" + self.endLine
      #kStr += indent + "sumIter" + indexChar + " = UNROLL;" + self.endLine
      kStr += "#undef UNROLL" + self.endLine
      kStr += "#define UNROLL 1" + self.endLine
      kStr += self.endLine

      kStr += indent + "unsigned int offA = l" + tileChar0 + "; // d0" + self.endLine
      kStr += indent + "unsigned int offB = l" + tileChar1 + "; // d1" + self.endLine
      kStr += self.endLine

      # begin loop
      kStr += indent + "do {" + self.endLine
      indent += "  "

      ####################################
      # do fmas
      kStr += indent + "/* do fmas */" + self.endLine
      kStr += indent + "MICRO_TILE" + self.endLine
      kStr += self.endLine

    ########################################################################
    # END UNROLL=1 LOOP
    ########################################################################


    ####################################
    # end loop
    for i in reversed(range(0,len(kernel.indexOrderSummation))):
      loopChar = indexChars[kernel.indexOrderSummation[i] \
          + len(kernel.indexOrderC)]
      # advance A, B along summation dimension
      kStr += indent + "A += (long) strideA" + loopChar
      if i==len(kernel.indexOrderSummation)-1:
        kStr += "*UNROLL"
      else:
        for j in range(i+1,min(i+2, len(kernel.indexOrderSummation)) ):
          tmpChar = indexChars[kernel.indexOrderSummation[j] \
              + len(kernel.indexOrderC)]
          kStr += " - strideA" + tmpChar + "*size" + tmpChar
      kStr += ";" + self.endLine

      kStr += indent + "B += (long) strideB" + loopChar
      if i==len(kernel.indexOrderSummation)-1:
        kStr += "*UNROLL"
      else:
        for j in range(i+1,min(i+2,len(kernel.indexOrderSummation)) ):
          tmpChar = indexChars[kernel.indexOrderSummation[j] \
              + len(kernel.indexOrderC)]
          kStr += " - strideB" + tmpChar + "*size" + tmpChar
      kStr += ";" + self.endLine
      indent = indent[2:]
      kStr += indent + "} while (--sumIter" + loopChar + " > 0);" + self.endLine
      kStr += self.endLine



    ####################################
    # which global Cij index
    kStr += "  /* which global Cij index */" + self.endLine
    for i in range(0, len(kernel.indexOrderC)):
      index = kernel.indexOrderC[i]
      kStr += "  unsigned int globalC" + indexChars[index] \
          + " = g" + indexChars[index]
      if index == kernel.indexAssignmentDim0:
        kStr += "*MT_" + tileChar0 + " + l" + tileChar0
      if index == kernel.indexAssignmentDim1:
        kStr += "*MT_" + tileChar1 + " + l" + tileChar1
      kStr += ";" + self.endLine
    kStr += self.endLine

    ####################################
    # write global Cij
    # debug printf
    #kStr += "  printf(\\\"T[%u,%u] global = %u, %u, %u size=%u, %u\\\\n\\\", " + self.getLocalIdStr + "(0), " + self.getLocalIdStr + "(1), global0I, global1J, globalCK, size0I, size1J);" + self.endLine
    # end debug
    # kStr += "  rC[0][0] = TYPE_C(1.23456789, -1.23456789);" + self.endLine
    
    # kStr += indent + "/* print LDS state */" + self.endLine
    # kStr += indent + "if ( gJ==0 && gL==0 && g1K==0 && g0I==0 && loadSerial == 0) {" + self.endLine
    # kStr += indent + "  for (unsigned int u = 0; u < UNROLL; u++) {" + self.endLine
    # kStr += indent + "    for (unsigned int i = 0; i < MT_" + tileChar0 + "; i++) {" + self.endLine
    # kStr += indent + "      printf(\\\"[%u,%u,%u,%u][%u,%u,%u][%02u,%02u] a=%f; b=%f\\\\n\\\", gJ, gL, g1K, g0I, sumIterM, sumIterN, sumIterO, u, i, localA[i+u*(MT_"+tileChar0+"+PAD)], localB[i+u*(MT_"+tileChar0+"+PAD)] );" + self.endLine
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
    if kernel.dataTypeC.value == Structs.DataType.complexSingle or kernel.dataTypeC.value == Structs.DataType.complexConjugateSingle:
      kStr += "  float type_fma_tmp;" + self.endLine
    if kernel.dataTypeC.value == Structs.DataType.complexDouble or kernel.dataTypeC.value == Structs.DataType.complexConjugateDouble:
      kStr += "  double type_fma_tmp;" + self.endLine

    for a in range(0, kernel.tile.microTile[0]):
      for b in range(0, kernel.tile.microTile[1]):
        numEdges = 0
        #for i in range(0, len(kernel.indexOrderC)):
        if not kernel.tile.branch[0].isNone():
          kStr += "  if (globalC" \
              + tileChar0 + " + " \
              + str(a) + "*WG_" + tileChar0 + "" + " < size" \
              + tileChar0 + ") {"
          numEdges += 1
        if not kernel.tile.branch[1].isNone():
          kStr += "  if (globalC" \
              + tileChar1 + " + " \
              + str(b) + "*WG_" + tileChar1 + "" + " < size" \
              + tileChar1 + ") {"
          numEdges += 1

        kStr += "  TYPE_MAD_WRITE( C[ GLOBAL_C("
        for i in range(0, len(kernel.indexOrderC)):
          kStr += " globalC" + indexChars[i]
          if i == kernel.indexAssignmentDim0:
            kStr += " + " + str(a) + "*WG_" + tileChar0
          if i == kernel.indexAssignmentDim1:
            kStr += " + " + str(b) + "*WG_" + tileChar1
          if i < len(kernel.indexOrderC)-1:
            kStr += ","
        kStr += ") ]"
        if kernel.useAlpha():
          kStr += ", alpha"
        kStr += ", rC[%d][%d]" % (a, b)
        if kernel.useBeta():
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
        #kStr += "  if (loadSerial < 24) printf(\\\"T[%u,%u]%u C[%u] = %f\\\\n\\\", " + self.getLocalIdStr + "(0), " + self.getLocalIdStr + "(1), globalCK, loadSerial, C[loadSerial]);"


    ####################################
    # end kernel
    kStr += self.endLine
    kStr += "}" + self.endLine

    return kStr






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

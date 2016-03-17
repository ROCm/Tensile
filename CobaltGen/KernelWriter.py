import os
import sys
import argparse
import copy

import Structs


################################################################################
# Make OpenCL Kernel String
################################################################################
class KernelWriter:

  endLine = "\\n\"\n\""
  indexChars = [ "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", \
      "T", "U", "V", "W", "X", "Y", "Z" ]

  ##############################################################################
  # Make OpenCL Kernel String
  ##############################################################################
  def __init__( self, backend ):
    pass


  ##############################################################################
  # get kernel name from operation
  ##############################################################################
  def getNameOperation(self, kernel):
    kernelName = ""

    # operation type
    kernelName += str(kernel.operation.type)
    kernelName += "_"

    # data dataTypes
    kernelName += kernel.dataTypeC.toChar().upper()
    kernelName += kernel.dataTypeA.toChar().upper()
    kernelName += kernel.dataTypeB.toChar().upper()

    # alpha
    if kernel.operation.useAlpha:
      kernelName += kernel.operation.alphaType.toChar().upper()
    else:
      kernelName += "0"

    # beta
    if kernel.operation.useBeta:
      kernelName += kernel.operation.betaType.toChar().upper()
    else:
      kernelName += "0"

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
    for i in range(0, len(kernel.operation.indexAssignmentsA)):
      kernelName += self.indexChars[kernel.operation.indexAssignmentsA[i]].lower()

    # B dimensions
    kernelName += "_B"
    for i in range(0,len(kernel.operation.indexAssignmentsB)):
      kernelName += self.indexChars[kernel.operation.indexAssignmentsB[i]].lower()


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
      kernelName += "s"
    else:
      kernelName += "f"
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

    # unroll
    kernelName += self.indexChars[len(kernel.indexOrderC)+len(kernel.indexOrderSummation)-1].lower()
    if len(kernel.unrolls)>0:
      kernelName += str(kernel.unrolls[0])
      for i in range(1,len(kernel.unrolls)):
        kernelName += "_" + str(kernel.unrolls[i])
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
    s += "__attribute__((reqd_work_group_size(WG_DIM_" + tileChar0 + ",WG_DIM_" + tileChar1 + ",1)))"
    s += self.endLine
    s += "__kernel void %s" % ( self.getName(kernel) )
    s += "(" + self.endLine
    # pointers & offsets
    s += (
      "  __global DATA_TYPE_STR_C       *          C," + self.endLine +
      "  __global DATA_TYPE_STR_A const * restrict A," + self.endLine +
      "  __global DATA_TYPE_STR_B const * restrict B," + self.endLine +
      "  unsigned int const offsetC," + self.endLine +
      "  unsigned int const offsetA," + self.endLine +
      "  unsigned int const offsetB" )
    # strides
    for i in range(0, len(kernel.indexOrderC)):
      s += "," + self.endLine + "  unsigned int const strideC" + indexChars[i]
    for i in range(0, len(kernel.operation.indexAssignmentsA)):
      s += "," + self.endLine + "  unsigned int const strideA" \
          + indexChars[kernel.operation.indexAssignmentsA[i]]
    for i in range(0, len(kernel.operation.indexAssignmentsB)):
      s += "," + self.endLine + "  unsigned int const strideB" \
          + indexChars[kernel.operation.indexAssignmentsB[i]]
    # sizes
    for i in range(0, len(kernel.indexOrderC)+len(kernel.indexOrderSummation)):
      s += "," + self.endLine + "  unsigned int const size" + indexChars[i]

    # alpha & beta
    if kernel.operation.useAlpha:
      s += "," + self.endLine + "  DATA_TYPE_STR_C const alpha"
    if kernel.operation.useBeta:
      s += "," + self.endLine + "  DATA_TYPE_STR_C const beta"
    # TODO - if convolution, need stride and pad for each sum dim
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
    kStr += "#define WG_DIM_%s          %d%s" \
        % (tileChar0, kernel.tile.workGroup[0], self.endLine )
    kStr += "#define WG_DIM_%s          %d%s" \
        % (tileChar1, kernel.tile.workGroup[1], self.endLine )
    kStr += "#define MICRO_TILE_" + tileChar0 + "  %d%s" \
        % (kernel.tile.microTile[0], self.endLine )
    kStr += "#define MICRO_TILE_" + tileChar1 + "  %d%s" \
        % (kernel.tile.microTile[1], self.endLine )
    kStr += "#define MACRO_TILE_" + tileChar0 + "  %s%s" \
        % ((kernel.tile.workGroup[0] * kernel.tile.microTile[0]), self.endLine )
    kStr += "#define MACRO_TILE_" + tileChar1 + "  %s%s" \
        % ((kernel.tile.workGroup[1] * kernel.tile.microTile[1]), self.endLine )
    kStr += "#define NUM_UNROLL_ITER  %s%s" \
        % (kernel.unrolls[len(kernel.unrolls)-1], self.endLine )
    kStr += "" + self.endLine

    ####################################
    # global memory indices
    kStr += self.endLine
    kStr += "/* global memory indices */" + self.endLine
    # C
    kStr += "#define GET_GLOBAL_INDEX_C(IDX" \
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
    kStr += "#define GET_GLOBAL_INDEX_A(IDX" \
        + indexChars[kernel.operation.indexAssignmentsA[0]]
    for i in range(1, len(kernel.operation.indexAssignmentsA)):
      kStr += ", IDX" + indexChars[kernel.operation.indexAssignmentsA[i]]
    indexChar = indexChars[kernel.operation.indexAssignmentsA[0]]
    kStr += ") ( (IDX" + indexChar + ")*strideA" + indexChar
    for i in range(1, len(kernel.operation.indexAssignmentsA)):
      indexChar = indexChars[kernel.operation.indexAssignmentsA[i]]
      kStr += " + (IDX" + indexChar + ")*strideA" + indexChar
    kStr += " )" + self.endLine
    # B
    kStr += "#define GET_GLOBAL_INDEX_B(IDX" \
        + indexChars[kernel.operation.indexAssignmentsB[0]]
    for i in range(1, len(kernel.operation.indexAssignmentsB)):
      kStr += ", IDX" + indexChars[kernel.operation.indexAssignmentsB[i]]
    indexChar = indexChars[kernel.operation.indexAssignmentsB[0]]
    kStr += ") ( (IDX" + indexChar + ")*strideB" + indexChar
    for i in range(1, len(kernel.operation.indexAssignmentsB)):
      indexChar = indexChars[kernel.operation.indexAssignmentsB[i]]
      kStr += " + (IDX" + indexChar + ")*strideB" + indexChar
    kStr += " )" + self.endLine

    ####################################
    # global tile indices being loaded
    kStr += self.endLine
    kStr += "  /* global tile indices being loaded */" + self.endLine

    if not kernel.unrollDimStrideGreaterThanTileDimStrideA:
      kStr += "/* fast read */" + self.endLine
      kStr += "#define globalIdxA" + tileCharA + "(LID) (groupIdx" + tileChar0 \
          + "*MACRO_TILE_" + tileChar0 + " + (localSerial+(LID)*WG_DIM_" \
          + tileChar0 + "*WG_DIM_" + tileChar1 + ")%MACRO_TILE_" \
          + tileChar0 + ")" + self.endLine
      kStr += "#define globalIdxA" + unrollChar \
          + "(LID) ((localSerial+(LID)*WG_DIM_" + tileChar0 + "*WG_DIM_" \
          + tileChar1 + ")/MACRO_TILE_" + tileChar0 + ")" + self.endLine
    else:
      kStr += "/* slow read */" + self.endLine
      kStr += "#define globalIdxA" + tileCharA + "(LID) (groupIdx" \
          + tileChar0 + "*MACRO_TILE_" + tileChar0 \
          + " + (localSerial+(LID)*WG_DIM_" \
          + tileChar0 + "*WG_DIM_" + tileChar1 + ")/NUM_UNROLL_ITER)" \
          + self.endLine
      kStr += "#define globalIdxA" + unrollChar \
          + "(LID) ((localSerial+(LID)*WG_DIM_" + tileChar0 \
          + "*WG_DIM_" + tileChar1 + ")%NUM_UNROLL_ITER)" + self.endLine

    if kernel.unrollDimStrideLessThanTileDimStrideB:
      kStr += "/* slow read */" + self.endLine
      kStr += "#define globalIdxB" + unrollChar \
          + "(LID) ((localSerial+(LID)*WG_DIM_" + tileChar0 \
          + "*WG_DIM_" + tileChar1 + ")%NUM_UNROLL_ITER)" + self.endLine
      kStr += "#define globalIdxB" + tileCharB \
          + "(LID) (groupIdx" + tileChar1 + "*MACRO_TILE_" + tileChar1 \
          + " + (localSerial+(LID)*WG_DIM_" + tileChar0 + "*WG_DIM_" \
          + tileChar1 + ")/NUM_UNROLL_ITER)" + self.endLine
    else:
      kStr += "/* fast read */" + self.endLine
      kStr += "#define globalIdxB" + unrollChar \
          + "(LID) ((localSerial+(LID)*WG_DIM_" + tileChar0 \
          + "*WG_DIM_" + tileChar1 + ")/MACRO_TILE_" + tileChar1 + ")" \
          + self.endLine
      kStr += "#define globalIdxB" + tileCharB \
          + "(LID) (groupIdx" + tileChar1 \
          + "*MACRO_TILE_" + tileChar1 + " + (localSerial+(LID)*WG_DIM_" \
          + tileChar0 + "*WG_DIM_" + tileChar1 \
          + ")%MACRO_TILE_" + tileChar1 + ")" + self.endLine
    kStr += self.endLine

    ####################################
    # global non-tile indices being loaded (batch & outer summation)
    kStr += "  /* global non-tile indices being loaded */" + self.endLine
    # C free indices which don't belong to tile = groupIdx
    # C batch = groupIdx
    for indexC in kernel.indexOrderC:
      if indexC == kernel.indexAssignmentDim0 \
          or indexC == kernel.indexAssignmentDim1:
        continue
      if indexC in kernel.operation.indexAssignmentsA:
        kStr += "#define globalIdxA" + indexChars[indexC] \
            + "(LID) groupIdx" + indexChars[indexC] + self.endLine
      if indexC in kernel.operation.indexAssignmentsB:
        kStr += "#define globalIdxB" + indexChars[indexC] \
            + "(LID) groupIdx" + indexChars[indexC] + self.endLine
    # C outer summation indices which aren't unrolled = sumIdx
    for i in range(0,len(kernel.indexOrderSummation)-1):
      index = i + len(kernel.indexOrderC)
      kStr += "#define globalIdxA" + indexChars[index] \
            + "(LID) groupIdx" + indexChars[index] + self.endLine
      kStr += "#define globalIdxB" + indexChars[index] \
            + "(LID) groupIdx" + indexChars[index] + self.endLine
    kStr += self.endLine

    ####################################
    # local memory indices
    kStr += self.endLine
    kStr += "/* local memory indices */" + self.endLine
    kStr += "#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_" + tileChar0 + ") )" + self.endLine
    kStr += "#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_" + tileChar1 + ") )" + self.endLine

    ####################################
    # local indices being written
    # thoroughly verify by hand for 4 GEMM cases (after doing global)
    kStr += self.endLine
    kStr += "    /* local indices being written */" + self.endLine
# new indices will be localA_unroll and localA_tile, which gets assigned to row
    if not kernel.unrollDimStrideGreaterThanTileDimStrideA:
      kStr += "#define localA" + tileCharA \
          + " (localSerial % MACRO_TILE_" + tileChar0 + ")" + self.endLine \
          + "#define localA" + unrollChar \
          +  " (localSerial / MACRO_TILE_" + tileChar0 + ")" + self.endLine \
          + "#define localAStride (WG_DIM_" + tileChar0 + "*WG_DIM_" + tileChar1 + ")" + self.endLine
    else:
      kStr += "#define localA" + tileCharA \
          + " (localSerial / NUM_UNROLL_ITER)" + self.endLine \
          + "#define localA" + unrollChar \
          + " (localSerial % NUM_UNROLL_ITER)" + self.endLine \
          + "#define localAStride (WG_DIM_" + tileChar0 + "*WG_DIM_" + tileChar1 + "/NUM_UNROLL_ITER)" \
          + self.endLine

    if kernel.unrollDimStrideLessThanTileDimStrideB:
      kStr += "#define localB" + tileCharB \
          + " ( localSerial % NUM_UNROLL_ITER )" + self.endLine \
          + "#define localB" + unrollChar \
          + " ( localSerial / NUM_UNROLL_ITER )" + self.endLine \
          + "#define localBStride (WG_DIM_" + tileChar0 + "*WG_DIM_" + tileChar1 + "/NUM_UNROLL_ITER)" \
          + self.endLine
    else:
      kStr += "#define localB" + tileCharB \
          + " ( localSerial / MACRO_TILE_" + tileChar1 + " )" + self.endLine \
          + "#define localB" + unrollChar \
          + " ( localSerial % MACRO_TILE_" + tileChar1 + " )" + self.endLine \
          + "#define localBStride  (WG_DIM_" + tileChar0 + "*WG_DIM_" + tileChar1 + ")" + self.endLine


    ####################################
    # data types
    kStr += self.endLine
    kStr += "/* data types */" + self.endLine
    kStr += "#define DATA_TYPE_STR_A %s%s" \
        % (kernel.dataTypeA.toOpenCL(), self.endLine)
    kStr += "#define DATA_TYPE_STR_B %s%s" \
        % (kernel.dataTypeB.toOpenCL(), self.endLine)
    kStr += "#define DATA_TYPE_STR_C %s%s" \
        % (kernel.dataTypeC.toOpenCL(), self.endLine)

    ####################################
    # MADs
    # TODO - mix real/complex
    if kernel.dataTypeC.isReal():
      # real data
      kStr += "#define TYPE_MAD(MULA,MULB,DST) " \
          + "DST = mad(MULA,MULB,DST);" + self.endLine
      if kernel.operation.useAlpha:
        if kernel.operation.useBeta:
          # dst = alpha*reg + beta*dst
          kStr += "#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) " \
              + "DST = (ALPHA)*(REG) + (BETA)*(DST);" + self.endLine
        else:
          # dst = alpha*reg
          kStr += "#define TYPE_MAD_WRITE(DST,ALPHA,REG) " \
              + "DST = (ALPHA)*(REG);" + self.endLine
      else:
        if kernel.operation.useBeta:
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
          "#define TYPE_MAD(MULA,MULB,DST) \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s0, MULB.s0, DST.s0 ); \\\\" + self.endLine +
          "  DST.s0 = mad( -MULA.s1, MULB.s1, DST.s0 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s0, MULB.s1, DST.s1 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif kernel.dataTypeA.isConjugate() and not kernel.dataTypeB.isConjugate():
        # A conjugate (negate imaginary A.s1)
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s0, MULB.s0, DST.s0 ); \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s1, MULB.s1, DST.s0 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s0, MULB.s1, DST.s1 ); \\\\" + self.endLine +
          "  DST.s1 = mad( -MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif not kernel.dataTypeA.isConjugate() and kernel.dataTypeB.isConjugate():
        # B conjugate (negate imaginary B.s1)
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s0,  MULB.s0, DST.s0 ); \\\\" + self.endLine +
          "  DST.s0 = mad( -MULA.s1, -MULB.s1, DST.s0 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s0, -MULB.s1, DST.s1 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      else:
        # A & B conjugate (negate imaginary .s1)
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s0,  MULB.s0, DST.s0 ); \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s1, -MULB.s1, DST.s0 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s0, -MULB.s1, DST.s1 ); \\\\" + self.endLine +
          "  DST.s1 = mad( -MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      if kernel.operation.useAlpha:
        if kernel.operation.useBeta:
          # dst = alpha*reg + beta*dst
          kStr += (
            "#define TYPE_MAD_WRITE( DST, ALPHA, REG, BETA ) \\\\" + self.endLine +
            "  /* (1) */ \\\\" + self.endLine +
            "  type_mad_tmp = REG.s0; \\\\" + self.endLine +
            "  REG.s0 *= ALPHA.s0; \\\\" + self.endLine +
            "  REG.s0 = mad( -ALPHA.s1, REG.s1, REG.s0 ); \\\\" + self.endLine +
            "  REG.s1 *= ALPHA.s0; \\\\" + self.endLine +
            "  REG.s1 = mad(  ALPHA.s1, type_mad_tmp, REG.s1 ); \\\\"+endLine+
            "  /* (2) */ \\\\" + self.endLine +
            "  REG.s0 = mad(  BETA.s0, DST.s0, REG.s0 ); \\\\" + self.endLine +
            "  REG.s0 = mad( -BETA.s1, DST.s1, REG.s0 ); \\\\" + self.endLine +
            "  REG.s1 = mad(  BETA.s1, DST.s0, REG.s1 ); \\\\" + self.endLine +
            "  REG.s1 = mad(  BETA.s0, DST.s1, REG.s1 ); \\\\" + self.endLine +
            "  /* (3) */ \\\\" + self.endLine +
            "  DST = REG;" + self.endLine )
        else:
          # dst = alpha*reg
          kStr += (
            "#define TYPE_MAD_WRITE( DST, ALPHA, REG ) \\\\"+endLine+
            "  /* (1) */ \\\\" + self.endLine +
            "  type_mad_tmp = REG.s0; \\\\" + self.endLine +
            "  REG.s0 *= ALPHA.s0; \\\\" + self.endLine +
            "  REG.s0 = mad( -ALPHA.s1, REG.s1, REG.s0 ); \\\\" + self.endLine +
            "  REG.s1 *= ALPHA.s0; \\\\" + self.endLine +
            "  REG.s1 = mad(  ALPHA.s1, type_mad_tmp, REG.s1 ); \\\\"+endLine+
            "  /* (3) */ \\\\" + self.endLine +
            "  DST = REG;" + self.endLine )
      else:
        if kernel.operation.useBeta:
          # dst = reg + beta*dst
          kStr += (
            "#define TYPE_MAD_WRITE( DST, REG, BETA ) \\\\" + self.endLine +
            "  /* (2) */ \\\\" + self.endLine +
            "  REG.s0 = mad(  BETA.s0, DST.s0, REG.s0 ); \\\\" + self.endLine +
            "  REG.s0 = mad( -BETA.s1, DST.s1, REG.s0 ); \\\\" + self.endLine +
            "  REG.s1 = mad(  BETA.s0, DST.s1, REG.s1 ); \\\\" + self.endLine +
            "  REG.s1 = mad(  BETA.s1, DST.s0, REG.s1 ); \\\\" + self.endLine +
            "  /* (3) */ \\\\" + self.endLine +
            "  DST = REG;" + self.endLine )
        else:
          # dst = reg
          kStr += (
            "#define TYPE_MAD_WRITE( DST, REG ) \\\\" + self.endLine +
            "  /* (3) */ \\\\" + self.endLine +
            "  DST = REG;" + self.endLine )

    ####################################
    # micro-tile
    kStr += self.endLine
    kStr += "/* %dx%d micro-tile */%s" % (kernel.tile.microTile[0], kernel.tile.microTile[1], self.endLine)
    kStr += "#define MICRO_TILE \\\\" + self.endLine
    for a in range(0, kernel.tile.microTile[0]):
      kStr += "  rA[%d] = localA[offA + %d*WG_DIM_%s]; \\\\%s" \
          % (a, a, tileChar0, self.endLine)
    for b in range(0, kernel.tile.microTile[1]):
      kStr += "  rB[%d] = localB[offB + %d*WG_DIM_%s]; \\\\%s" \
          % (b, b, tileChar1, self.endLine)
    kStr += "  offA += MACRO_TILE_" + tileChar0 + "; \\\\" + self.endLine
    kStr += "  offB += MACRO_TILE_" + tileChar1 + "; \\\\" + self.endLine
    for a in range(0, kernel.tile.microTile[0]):
      for b in range(0, kernel.tile.microTile[1]):
        kStr += "  TYPE_MAD(rA[%d],rB[%d],rC[%d][%d]); \\\\%s" % (a, b, a, b, self.endLine)
    kStr += "  mem_fence(CLK_LOCAL_MEM_FENCE);" + self.endLine
    kStr += self.endLine

    ####################################
    # function signature
    ####################################
    kStr += self.getSignature(kernel)
    kStr += " {" + self.endLine

    # debug printf - kernel args
    #kStr += "  if( get_local_id(0) ==0 && get_local_id(1) == 0) printf(\\\"oC=%u, oA=%u, oB=%u, sCI=%u, sCJ=%u, sCK=%u, sAI=%u, sAK=%u, sAL=%u, sBI=%u, sBJ=%u, sBL=%u, sI=%u, sJ=%u, sK=%u, sL=%u\\\\n\\\", offsetC, offsetA, offsetB, strideCI, strideC1J, strideC0K, strideAI, strideA0K, strideAL, strideBI, strideB1J, strideBL, sizeI, size1J, size0K, sizeL"
    #kStr += ");" + self.endLine
    # end debug printf



    ####################################
    # apply offsets
    kStr += self.endLine
    kStr += (
      "  /* apply offsets */" + self.endLine +
      "  C += offsetC;" + self.endLine +
      "  A += offsetA;" + self.endLine +
      "  B += offsetB;" + self.endLine )

    ####################################
    # allocate registers
    kStr += self.endLine
    kStr += (
      "  /* allocate registers */" + self.endLine +
      "  DATA_TYPE_STR_C rC[MICRO_TILE_" + tileChar0 + "][MICRO_TILE_" + tileChar1 + "] "
          + "= {{0}};" + self.endLine +
      "  DATA_TYPE_STR_A rA[MICRO_TILE_" + tileChar0 + "];" + self.endLine +
      "  DATA_TYPE_STR_B rB[MICRO_TILE_" + tileChar1 + "];" + self.endLine )

    ####################################
    # allocate local memory
    kStr += self.endLine
    kStr += (
      "  /* allocate local memory */" + self.endLine +
      "  __local DATA_TYPE_STR_A localA[NUM_UNROLL_ITER*MACRO_TILE_" + tileChar0 + "];" \
          + self.endLine +
      "  __local DATA_TYPE_STR_B localB[NUM_UNROLL_ITER*MACRO_TILE_" + tileChar1 + "];" \
          + self.endLine )

    ####################################
    # c indices
    # kernel.indexOrderC - performance defined
    # kernel.indexOrderSummation - performance defined
    # kernel.indexAssignmentsA - user defined
    # kernel.indexAssignmentsB - user defined
    # convert get_group_id(0) to however many c indices there are


    kStr += self.endLine
    kStr += "  /* c indices */" + self.endLine
    kStr += "  unsigned int groupIdx" + tileChar0 + " = get_group_id(0);" \
        + " // d0, tensor" + tensorChar0 + self.endLine
    kStr += "  unsigned int groupIdx" + tileChar1 + " = get_group_id(1);" \
        + " // d1, tensor" + tensorChar1 + self.endLine

    # other free indices
    nonTileFreeIndices = copy.deepcopy(kernel.indexOrderC)
    nonTileFreeIndices.remove(kernel.indexAssignmentDim0)
    nonTileFreeIndices.remove(kernel.indexAssignmentDim1)

    for i in range(0, len(nonTileFreeIndices)):
      index = nonTileFreeIndices[i]
      kStr += "  unsigned int groupIdx" + indexChars[index] \
          + " = ( get_group_id(2)"
      for j in reversed( range( i+1, len(nonTileFreeIndices)) ):
        index2 = nonTileFreeIndices[j]
        kStr += " / size" + indexChars[index2]
      kStr += " ) % size" + indexChars[index] + ";" + self.endLine

    # local indices
    kStr += "  uint localIdx" + tileChar0 \
        + " = get_local_id(0); // d0" + self.endLine
    kStr += "  uint localIdx" + tileChar1 \
        + " = get_local_id(1); // d1" + self.endLine
    kStr += "  uint localSerial = localIdx" + tileChar0 \
        + " + localIdx" + tileChar1 + "*WG_DIM_" + tileChar0 \
        + ";" + self.endLine

    ####################################
    # which global Cij index
    kStr += self.endLine
    kStr += "  /* which global Cij index */" + self.endLine
    for i in range(0, len(kernel.indexOrderC)):
      index = kernel.indexOrderC[i]
      kStr += "  unsigned int globalIdxC" + indexChars[index] \
          + " = groupIdx" + indexChars[index]
      if index == kernel.indexAssignmentDim0:
        kStr += "*MACRO_TILE_" + tileChar0 + " + localIdx" + tileChar0
      if index == kernel.indexAssignmentDim1:
        kStr += "*MACRO_TILE_" + tileChar1 + " + localIdx" + tileChar1
      kStr += ";" + self.endLine


    # debug printf - global data
    #kStr += "  printf(\\\"T[%u,%u] A[%u] = %f; B[%u] = %f\\\\n\\\", get_local_id(0), get_local_id(1), localSerial, A[localSerial], localSerial, B[localSerial]"
    #kStr += ");" + self.endLine
    # end debug printf


    # multidim if (kernel.order=="clblasColumnMajor")==(kernel.transA=="N"):
    #tensorAssignedToTileDim = []
    #if kernel.tensorAssignedDim0:
    #  tensorAssignedToTileDim.append(kernel.operation.
    #unrollStrideGreaterThanTileA
    #kernel.unrollDimStrideGreaterThanTileDimStrideA = kernel.indexAssignmentDim0 \
     #   > kernel.indexOrderSummation[len(kernel.indexOrderSummation)-1]
    #kernel.unrollDimStrideLessThanTileDimStrideB = kernel.indexAssignmentDim1 \
    #    > kernel.indexOrderSummation[len(kernel.indexOrderSummation)-1]


    kStr += "  bool validC ="
    if kernel.tile.branch[0]:
      kStr += " (globalIdxC" \
          + tileChar0 + " + " \
          + str(a) + "*WG_DIM_" + tileChar0 + "" + " < size" \
          + tileChar0 + ")"
    if kernel.tile.branch[0] and kernel.tile.branch[1]:
      kStr += " &&"
    if kernel.tile.branch[1]:
      kStr += " (globalIdxC" \
          + tileChar1 + " + " \
          + str(b) + "*WG_DIM_" + tileChar1 + "" + " < size" \
          + tileChar1 + ")"
    kStr += ";" + self.endLine



    ####################################
    # summations loops
    indent = "  "
    kStr += indent + "/* iterate over all summation indices */" + self.endLine
    for i in range(0,len(kernel.indexOrderSummation)):
      indexChar = indexChars[kernel.indexOrderSummation[i] \
          + len(kernel.indexOrderC)]
      kStr += indent + "unsigned int sumIter" + indexChar \
          + " = size" + indexChar
      if i == len(kernel.indexOrderSummation)-1:
        kStr += " / NUM_UNROLL_ITER"
      kStr += ";" + self.endLine
      kStr += indent + "do {" + self.endLine
      indent += "  "

    kStr += indent + "__local DATA_TYPE_STR_A *lA = localA" \
        + " + GET_LOCAL_INDEX_A(localA" + tileCharA + ", localA" \
        + unrollChar + ");" + self.endLine \
        + indent + "__local DATA_TYPE_STR_B *lB = localB" \
        + " + GET_LOCAL_INDEX_B(localB" + tileCharB + ", localB" \
        + unrollChar + ");" + self.endLine \
        + indent + "barrier(CLK_LOCAL_MEM_FENCE);" + self.endLine

    # debug printf - LDS load offsets
    #kStr += "  printf(\\\"T[%u,%u] localIdx = %u, %u\\\\n\\\", get_local_id(0), get_local_id(1), "
    #kStr += "GET_LOCAL_INDEX_A(localA" + tileCharA + ", localA" \
    #    + unrollChar + "), "
    #kStr += "GET_LOCAL_INDEX_B(localB" + tileCharB + ", localB" \
    #    + unrollChar + ") "
    #kStr += ");" + self.endLine
    # end debug printf


    ####################################
    # how many elements to load global -> local
    # threads to do loading = (workGroup[0]*workGroup[1])
    # A elements to be loaded = workGroup[0]*microTile[0]*unroll
    # B elements to be loaded = workGroup[1]*microTile[1]*unroll
    kStr += self.endLine
    kStr += indent + "/* load global -> local */" + self.endLine
    numALoads  = (kernel.tile.workGroup[0]*kernel.tile.microTile[0]*kernel.unrolls[len(kernel.unrolls)-1]) \
        / (kernel.tile.workGroup[0]*kernel.tile.workGroup[1])
    numALoadsR = (kernel.tile.workGroup[0]*kernel.tile.microTile[0]*kernel.unrolls[len(kernel.unrolls)-1]) \
        % (kernel.tile.workGroup[0]*kernel.tile.workGroup[1])
    numBLoads  = (kernel.tile.workGroup[1]*kernel.tile.microTile[1]*kernel.unrolls[len(kernel.unrolls)-1]) \
        / (kernel.tile.workGroup[0]*kernel.tile.workGroup[1])
    numBLoadsR = (kernel.tile.workGroup[1]*kernel.tile.microTile[1]*kernel.unrolls[len(kernel.unrolls)-1]) \
        % (kernel.tile.workGroup[0]*kernel.tile.workGroup[1])

    # zeroString for real and complex
    zeroStringC = kernel.dataTypeC.zeroStringOpenCL()
    zeroStringA = kernel.dataTypeA.zeroStringOpenCL()
    zeroStringB = kernel.dataTypeB.zeroStringOpenCL()

    #if kernel.dataTypeA.value == Structs.DataType.complexSingle:
    #  zeroStringA = "(float2)(0.f, 0.f)"
    #elif kernel.dataTypeA.value == Structs.DataType.complexDouble:
    #  zeroStringA = "(double2)(0.0, 0.0)"
    #else:
    #  zeroStringA = "0.0"
    #if kernel.dataTypeB.value == Structs.DataType.complexSingle:
    #  zeroStringB = "(float2)(0.f, 0.f)"
    #elif kernel.dataTypeB.value == Structs.DataType.doubleComplex:
    #  zeroStringB = "(double2)(0.0, 0.0)"
    #else:
    #  zeroStringB = "0.0"
    #if kernel.dataTypeC.value == Structs.DataType.complexSingle:
    #  zeroStringC = "(float2)(0.f, 0.f)"
    #elif kernel.dataTypeC.value == Structs.DataType.doubleComplex:
    #  zeroStringC = "(double2)(0.0, 0.0)"
    #else:
    #  zeroStringC = "0.0"



    ####################################
    # load global -> local
    kStr += "    /* numALoads = " + str(numALoads) + " */" + self.endLine
    kStr += "    /* numALoadsR = " + str(numALoadsR) + " */" + self.endLine
    kStr += "    /* numBLoads = " + str(numBLoads) + " */" + self.endLine
    kStr += "    /* numBLoadsR = " + str(numBLoadsR) + " */" + self.endLine
    # load A whole WG
    for a in range(0, numALoads):
      kStr += indent + "lA[ %d*localAStride ] = " % a
      if kernel.tile.branch[0]:
        kStr += "( globalIdxA%s(%d) >= size%s) ? %s : " \
            % ( tileCharA, a, tileCharA, zeroStringA )
      kStr += "A[ GET_GLOBAL_INDEX_A( "
      kStr += "globalIdxA" + indexChars[ \
          kernel.operation.indexAssignmentsA[0]]  \
          + "(" + str(a) + ")"
      for i in range(1,len(kernel.operation.indexAssignmentsA)):
        kStr += ", globalIdxA" + indexChars[ \
            kernel.operation.indexAssignmentsA[i]]  \
            + "(" + str(a) + ")"
      kStr += " ) ];" + self.endLine

    # load A remainder
    if numALoadsR:
      kStr += indent + "if ( localSerial + " + str(numALoads) + \
          "*WG_DIM_" + tileChar0 + "*WG_DIM_" + tileChar1 + " < (WG_DIM_" + tileChar0 + "*MICRO_TILE_" + tileChar0 + "*NUM_UNROLL_ITER) ) {" \
          + self.endLine
      kStr += indent + "  lA[ %d*localAStride ] = " \
          % numALoads
      if kernel.tile.branch[0]:
        kStr += "( globalIdxA%s(%d) >= size%s) ? %s : " \
            % ( tileCharA, numALoads, tileCharA, zeroStringA )
      kStr += "A[ GET_GLOBAL_INDEX_A( "
      kStr += "globalIdxA" + indexChars[ \
          kernel.operation.indexAssignmentsA[0]]  \
          + "(" + str(numALoads) + ")"
      for i in range(1,len(kernel.operation.indexAssignmentsA)):
        kStr += ", globalIdxA" + indexChars[ \
            kernel.operation.indexAssignmentsA[i]]  \
            + "(" + str(numALoads) + ")"
      kStr += " ) ];" + self.endLine

      # debug printf - values loading into LDS
      """
      kStr += "    unsigned int tmpIdx = GET_GLOBAL_INDEX_A( "
      kStr += "globalIdxA" + indexChars[ \
          kernel.operation.indexAssignmentsA[0]]  \
          + "(" + str(numALoads) + ")"
      for i in range(1,len(kernel.operation.indexAssignmentsA)):
        kStr += ", globalIdxA" + indexChars[ \
            kernel.operation.indexAssignmentsA[i]]  \
            + "(" + str(numALoads) + ")"
      kStr += " );" + self.endLine
      kStr += "printf(\\\"T[%u,%u]%u:%u localA[%u] = %f <- globalA[%u] = %f; %u"
      for i in range(1,len(kernel.operation.indexAssignmentsA)):
        kStr += ", %u"

      kStr += "\\\\n\\\", get_local_id(0), get_local_id(1), globalIdxCK, sumIterL, GET_LOCAL_INDEX_A(localA" + tileCharA \
          + ", localA" + unrollChar +"), "
      kStr += " lA[0], tmpIdx, A[tmpIdx], "


      kStr += "globalIdxA" + indexChars[ \
          kernel.operation.indexAssignmentsA[0]]  \
          + "(" + str(numALoads) + ")"
      for i in range(1,len(kernel.operation.indexAssignmentsA)):
        kStr += ", globalIdxA" + indexChars[ \
            kernel.operation.indexAssignmentsA[i]]  \
            + "(" + str(numALoads) + ")"
      kStr += ");" + self.endLine
      """
      # end debug printf

      kStr += indent + "}" + self.endLine

    # load B whole WG
    for b in range(0, numBLoads):
      kStr += indent + "lB[ %d*localBStride ] = " % b
      if kernel.tile.branch[1]:
        kStr += "( globalIdxB%s(%d) >= size%s) ? %s : " \
            % ( tileCharB, b, tileCharB, zeroStringB )
      kStr += "B[ GET_GLOBAL_INDEX_B( "
      kStr += "globalIdxB" + indexChars[ \
          kernel.operation.indexAssignmentsB[0]]  \
          + "(" + str(b) + ")"
      for i in range(1,len(kernel.operation.indexAssignmentsB)):
        kStr += ", globalIdxB" + indexChars[ \
            kernel.operation.indexAssignmentsB[i]]  \
            + "(" + str(b) + ")"
      kStr += " ) ];" + self.endLine

    # load B remainder
    if numBLoadsR:
      kStr += indent + "if ( localSerial + " + str(numBLoads) + \
          "*WG_DIM_" + tileChar0 + "*WG_DIM_" + tileChar1 + " < (WG_DIM_" + tileChar1 + "*MICRO_TILE_" + tileChar1 + "*NUM_UNROLL_ITER) ) {" \
          + self.endLine
      kStr += indent + "  lB[ %d*localBStride ] = " % numBLoads
      if kernel.tile.branch[1]:
        kStr += "(globalIdxB%s(%d) >= size%s) ? %s : " \
            % ( tileCharB, numBLoads, tileCharB, zeroStringB )
      kStr += "B[ GET_GLOBAL_INDEX_B( "
      kStr += "globalIdxB" + indexChars[ \
          kernel.operation.indexAssignmentsB[0]]  \
          + "(" + str(numBLoads) + ")"
      for i in range(1,len(kernel.operation.indexAssignmentsB)):
        kStr += ", globalIdxB" + indexChars[ \
            kernel.operation.indexAssignmentsB[i]]  \
            + "(" + str(numBLoads) + ")"
      kStr += " ) ];" + self.endLine
      kStr += indent + "}" + self.endLine
    kStr += (
      indent + "barrier(CLK_LOCAL_MEM_FENCE);" + self.endLine +
      indent + "uint offA = localIdx" + tileChar0 + "; // d0" + self.endLine +
      indent + "uint offB = localIdx" + tileChar1 + "; // d1" + self.endLine )


    ####################################
    # do mads
    kStr += self.endLine
    kStr += indent + "/* do mads */" + self.endLine
    for u in range(0, kernel.unrolls[len(kernel.unrolls)-1]):
      kStr += indent + "MICRO_TILE" + self.endLine

    # debug printf - accumulation in registers
    #kStr += "  if (validC) printf(\\\"T[%u,%u]%u rC = %f gIdx=%u\\\\n\\\", get_local_id(0), get_local_id(1), globalIdxCK, rC[0][0], GET_GLOBAL_INDEX_C(globalIdxC0I, globalIdxC1J, globalIdxCK) );" + self.endLine
    # end debug printf














    ####################################
    # end loop
    for i in reversed(range(0,len(kernel.indexOrderSummation))):
      loopChar = indexChars[kernel.indexOrderSummation[i] \
          + len(kernel.indexOrderC)]
      # advance A, B along summation dimension
      kStr += indent + "A += strideA" + loopChar
      if i==len(kernel.indexOrderSummation)-1:
        kStr += "*NUM_UNROLL_ITER"
      else:
        for j in range(i+1,len(kernel.indexOrderSummation)):
          tmpChar = indexChars[kernel.indexOrderSummation[j] \
              + len(kernel.indexOrderC)]
          kStr += " - strideA" + tmpChar + "*size" + tmpChar
      kStr += ";" + self.endLine
      kStr += indent + "B += strideB" + loopChar
      if i==len(kernel.indexOrderSummation)-1:
        kStr += "*NUM_UNROLL_ITER"
      else:
        for j in range(i+1,len(kernel.indexOrderSummation)):
          tmpChar = indexChars[kernel.indexOrderSummation[j] \
              + len(kernel.indexOrderC)]
          kStr += " - strideB" + tmpChar + "*size" + tmpChar
      kStr += ";" + self.endLine
      indent = indent[2:]
      # close do-while loop
      kStr += indent + "} while (--sumIter" + loopChar + " > 0);" + self.endLine
    kStr += self.endLine


    ####################################
    # write global Cij
    kStr += self.endLine
    # debug printf
    #kStr += "  printf(\\\"T[%u,%u] global = %u, %u, %u size=%u, %u\\\\n\\\", get_local_id(0), get_local_id(1), globalIdx0I, globalIdx1J, globalIdxCK, size0I, size1J);" + self.endLine
    # end debug
    kStr += "  /* write global C */" + self.endLine
    if kernel.dataTypeC == Structs.DataType.complexSingle or kernel.dataTypeC == Structs.DataType.complexConjugateSingle:
      kStr += "  float type_mad_tmp;" + self.endLine
    if kernel.dataTypeC == Structs.DataType.complexDouble or kernel.dataTypeC == Structs.DataType.complexConjugateDouble:
      kStr += "  double type_mad_tmp;" + self.endLine

    for a in range(0, kernel.tile.microTile[0]):
      for b in range(0, kernel.tile.microTile[1]):
        numEdges = 0
        #for i in range(0, len(kernel.indexOrderC)):
        if kernel.tile.branch[0]:
          kStr += "  if (globalIdxC" \
              + tileChar0 + " + " \
              + str(a) + "*WG_DIM_" + tileChar0 + "" + " < size" \
              + tileChar0 + ") {"
          numEdges += 1
        if kernel.tile.branch[1]:
          kStr += "  if (globalIdxC" \
              + tileChar1 + " + " \
              + str(b) + "*WG_DIM_" + tileChar1 + "" + " < size" \
              + tileChar1 + ") {"
          numEdges += 1

        kStr += "  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C("
        for i in range(0, len(kernel.indexOrderC)):
          kStr += " globalIdxC" + indexChars[i]
          if i == kernel.indexAssignmentDim0:
            kStr += " + " + str(a) + "*WG_DIM_" + tileChar0
          if i == kernel.indexAssignmentDim1:
            kStr += " + " + str(b) + "*WG_DIM_" + tileChar1
          if i < len(kernel.indexOrderC)-1:
            kStr += ","
        kStr += ") ]"
        if kernel.operation.useAlpha:
          kStr += ", alpha"
        kStr += ", rC[%d][%d]" % (a, b)
        if kernel.operation.useBeta:
          kStr += ", beta"
        kStr += ")"
        # debug printf
        #kStr += " printf(\\\"T[%u,%u] Cijk = %f\\\\n\\\", get_local_id(0), get_local_id(1), rC[" + str(a) + "][" + str(b) + "] );"

        # debug printf
        #kStr += "  printf(\\\"T[%u,%u]%u writing C[%u] = %f\\\\n\\\", get_local_id(0), get_local_id(1), globalIdxCK, GET_GLOBAL_INDEX_C(globalIdxC1I, globalIdxC0J, globalIdxCK), rC[0][0]"
        #kStr += ");" + self.endLine
        # end debug printf

        for i in range(0,numEdges):
          kStr += " }"
        kStr += self.endLine
        #kStr += "  if (localSerial < 24) printf(\\\"T[%u,%u]%u C[%u] = %f\\\\n\\\", get_local_id(0), get_local_id(1), globalIdxCK, localSerial, C[localSerial]);"

    ####################################
    # end kernel
    kStr += self.endLine
    kStr += "}" + self.endLine

    return kStr


################################################################################
# Test GEMM
################################################################################
def testGEMM():
  print("Test GEMM Fast: C[ij] = Sum_k A[ki] * B[kj]")

  # kernel parameters
  dimensionsC = []
  dimensionsC.append( Structs.Dimension(    1, 1024 ) )
  dimensionsC.append( Structs.Dimension( 1024,  512 ) )
  tensorC = Structs.Tensor( \
      Structs.DataType(Structs.DataType.single),
      dimensionsC )

  dimensionsA = []
  dimensionsA.append( Structs.Dimension(   1,  256 ) )
  dimensionsA.append( Structs.Dimension( 256, 1024 ) )
  tensorA = Structs.Tensor( \
      Structs.DataType(Structs.DataType.single),
      dimensionsA )

  dimensionsB = []
  dimensionsB.append( Structs.Dimension(   1, 256 ) )
  dimensionsB.append( Structs.Dimension( 256, 512 ) )
  tensorB = Structs.Tensor( \
      Structs.DataType(Structs.DataType.single),
      dimensionsA )

  operationType = Structs.OperationType(Structs.OperationType.contraction)
  numFreeIndices = 2
  numIndicesBatch = 0
  numIndicesSummation = 1
  indexAssignmentsA = [2, 0]
  indexAssignmentsB = [2, 1]
  useAlpha = False
  useBeta = False
  operation = Structs.Operation( \
      operationType, \
      useAlpha, \
      useBeta, \
      numFreeIndices, \
      numIndicesBatch, \
      numIndicesSummation, \
      indexAssignmentsA, \
      indexAssignmentsB )

  kernel = Structs.Kernel(\
      operation, \
      tensorA, \
      tensorB, \
      tensorC )

  kernel.assignTile( 16, 16, 4, 4, 64, 64, 16 )

  print("\"GEMM\" Kernel Name: %s") % kernel.getName()
  backend = Structs.Backend(Structs.Backend.opencl)
  print("\"GEMM\" Kernel Body: %s") % kernel.getBody(backend)

def testAdvanced():
  print("Test Advanced: C[ijk] = Sum_lm A[mkli] * B[jlkm]")
  """
  dimension sizes
  i: 512
  j: 256
  k: 128
  l:  64
  m:  32

  *** C ***
  index stride size assignment dimorder
  0:    32,768  512  (i)  2
  1:         1  256  (j)  0
  2:       256  128  (k)  1

  *** A ***
  index stride size assignment dimorder
  0:        64   32  (m)  1
  1: 1,048,576  128  (k)  3
  2:         1   64  (l)  0
  2:     2,048  512  (i)  2

  *** B ***
  index stride size assignment dimorder
  0:        32  256  (j)  1
  1: 1,048,576   64  (l)  3
  2:     8,192  128  (k)  2
  2:         1   32  (m)  0

  """
  # tensor dimensions
  dimensionsC = []
  dimensionsC.append( Structs.Dimension(   32768, 512 ) )
  dimensionsC.append( Structs.Dimension(       1, 256 ) )
  dimensionsC.append( Structs.Dimension(     256, 128 ) )
  dimensionsA = []
  dimensionsA.append( Structs.Dimension(      64,  32 ) )
  dimensionsA.append( Structs.Dimension( 1048576, 128 ) )
  dimensionsA.append( Structs.Dimension(       1,  64 ) )
  dimensionsA.append( Structs.Dimension(    2048, 512 ) )
  dimensionsB = []
  dimensionsB.append( Structs.Dimension(      32, 256 ) )
  dimensionsB.append( Structs.Dimension( 1048576,  64 ) )
  dimensionsB.append( Structs.Dimension(    8192, 128 ) )
  dimensionsB.append( Structs.Dimension(       1,  32 ) )

  # tensor objects
  tensorC = Structs.Tensor( \
      Structs.DataType(Structs.DataType.single),
      dimensionsC )
  tensorA = Structs.Tensor( \
      Structs.DataType(Structs.DataType.single),
      dimensionsA )
  tensorB = Structs.Tensor( \
      Structs.DataType(Structs.DataType.single),
      dimensionsA )

  operationType = Structs.OperationType(Structs.OperationType.contraction)
  numFreeIndices = 2
  numIndicesBatch = 1
  numIndicesSummation = 2
  indexAssignmentsA = [ 4, 2, 3, 0 ]
  indexAssignmentsB = [ 1, 3, 2, 4 ]
  operation = Structs.Operation( \
      operationType, \
      numFreeIndices, \
      numIndicesBatch, \
      numIndicesSummation, \
      indexAssignmentsA, \
      indexAssignmentsB )
  useAlpha = False
  useBeta = False

  kernel = Kernel(\
      operation, \
      tensorA, \
      tensorB, \
      tensorC )

  kernel.assignTile( 16, 16, 4, 4, 64, 64, 16 )

  print("\"Advanced\" Kernel Name: %s") % kernel.getName()
  backend = Structs.Backend(Structs.Backend.opencl)
  print("\"Advanced\" Kernel Body: %s") % kernel.getBody(backend)

  pass

################################################################################
# Main
################################################################################
if __name__ == "__main__":
  testGEMM()
  print("\n\n\n")
  testAdvanced()



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

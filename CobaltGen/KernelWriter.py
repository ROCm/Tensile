import os
import sys
import argparse

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
  def __init__( self, kernel ):
    pass



  ##############################################################################
  # isEdge kernel along dimension
  # - macroTileDim0 == 1
  # - guards around gA -> lA
  # - guards around gC[gRow,:] = rC[row,:]
  ##############################################################################
  def isEdge(self, kernel, dim):
    if dim == kernel.indexAssignmentDim0:
      return kernel.tile.workGroupDim0 * kernel.tile.microTileDim0 != kernel.tile.macroTileDim0
    if dim == kernel.indexAssignmentDim1:
      return kernel.tile.workGroupDim1 * kernel.tile.microTileDim1 != kernel.tile.macroTileDim1
    return False


  ##############################################################################
  # get kernel name - DONE
  ##############################################################################
  def getName(self, kernel):
    kernelName = ""

    # operation type
    kernelName += str(kernel.operation.type)
    kernelName += "_"

    # data dataTypes
    kernelName += kernel.dataTypeA.toChar().upper()
    kernelName += kernel.dataTypeB.toChar().upper()
    kernelName += kernel.dataTypeC.toChar().upper()
    kernelName += "_"

    # C dimensions
    kernelName += "C"
    for i in range(0, len(kernel.indexOrderC)):
      kernelName += self.indexChars[kernel.indexOrderC[i]].lower()
    kernelName += "=S"

    # summation indices
    for i in range(0,len(kernel.indexOrderSummation)):
      kernelName += self.indexChars[kernel.indexOrderSummation[i]].lower()
    kernelName += "_"

    # A dimensions
    kernelName += "A"
    for i in range(0, len(kernel.operation.indexAssignmentsA)):
      kernelName += self.indexChars[kernel.operation.indexAssignmentsA[i]].lower()
    kernelName += "_"

    # B dimensions
    kernelName += "B"
    for i in range(0,len(kernel.operation.indexAssignmentsB)):
      kernelName += self.indexChars[kernel.operation.indexAssignmentsB[i]].lower()
    kernelName += "_"

    # alpha
    kernelName += "a"
    if kernel.operation.alpha:
      kernelName += "1"
    else:
      kernelName += "0"

    # beta
    kernelName += "b"
    if kernel.operation.beta:
      kernelName += "1"
    else:
      kernelName += "0"
    kernelName += "_"

    # tile dim 0
    kernelName += self.indexChars[kernel.indexAssignmentDim0].lower()
    kernelName += str(kernel.tile.workGroupDim0)
    if kernel.tile.macroTileDim0 == 1:
      kernelName += "y"
    else:
      kernelName += "x"
    kernelName += str(kernel.tile.microTileDim0)
    kernelName += "_"

    # tile dim 1
    kernelName += self.indexChars[kernel.indexAssignmentDim1].lower()
    kernelName += str(kernel.tile.workGroupDim1)
    if kernel.tile.macroTileDim1 == 1:
      kernelName += "y"
    else:
      kernelName += "x"
    kernelName += str(kernel.tile.microTileDim1)
    kernelName += "_"

    # unroll
    kernelName += self.indexChars[len(kernel.indexOrderC)+len(kernel.unrolls)-1].lower()
    kernelName += str(kernel.unrolls[0])
    for i in range(1,len(kernel.unrolls)):
      kernelName += "_" + str(kernel.unrolls[i])

    # c indices
    """for i in range(0,len(kernel.indexOrderC)):
      index = kernel.indexOrderC[i]
      if index == kernel.indexAssignmentDim0:
        kernelName= "xT0X" + str(kernel.tile.workGroupDim0) \
            + "x" + str(kernel.tile.microTileDim0)
      if index == kernel.indexAssignmentDim1:
        multipleStr = "xT1X" + str(kernel.tile.workGroupDim1) \
            + "x" + str(kernel.tile.microTileDim1)
      kernelName += self.indexChars[index].lower() + multipleStr
      kernelName += "_"
    """

    """# c indices
    for i in range(0,len(kernel.indexOrderC)):
      index = kernel.indexOrderC[i]
      multipleStr = "x1"
      if index == kernel.indexAssignmentDim0:
        multipleStr = "xT0X" + str(kernel.tile.workGroupDim0) \
            + "x" + str(kernel.tile.microTileDim0)
      if index == kernel.indexAssignmentDim1:
        multipleStr = "xT1X" + str(kernel.tile.workGroupDim1) \
            + "x" + str(kernel.tile.microTileDim1)
      kernelName += self.indexChars[index].lower() + multipleStr
      kernelName += "_" """

    # summation indices
    """
    for i in range(0,len(kernel.indexOrderSummation)):
      index = kernel.indexOrderSummation[i]
      multipleStr = "1"
      if index == len(kernel.indexOrderSummation)-1:
        multipleStr = str(kernel.unrolls[0])
        for j in range(1,len(kernel.unrolls)):
          multipleStr += "-" + str(kernel.unrolls[j])
      kernelName += self.indexChars[len(kernel.indexOrderC) \
          + index].lower() + "X" + multipleStr
      if i != len(kernel.indexOrderSummation)-1:
        kernelName += "_"
    """

    return kernelName


  ##############################################################################
  # get kernel signature - DONE
  ##############################################################################
  def getSignature(self, kernel ):
    s = ""
    # kernel name
    s += "__attribute__((reqd_work_group_size(WG_DIM1,WG_DIM0,1)))"
    s += self.endLine
    s += "__kernel void %s" % ( self.getName(kernel) )
    s += "(" + self.endLine
    # pointers & offsets
    s += (
      "  __global DATA_TYPE_STR_C       *          C," + self.endLine +
      "  __global DATA_TYPE_STR_A const * restrict A," + self.endLine +
      "  __global DATA_TYPE_STR_B const * restrict B," + self.endLine +
      "  size_t const offsetC," + self.endLine +
      "  size_t const offsetA," + self.endLine +
      "  size_t const offsetB," + self.endLine )
    # strides
    for i in range(0, len(kernel.indexOrderC)):
      s += "  size_t const strideC" + self.indexChars[i] + "," + self.endLine
    for i in range(0, len(kernel.operation.indexAssignmentsA)):
      s += "  size_t const strideA" \
          + self.indexChars[kernel.operation.indexAssignmentsA[i]] \
          + "," + self.endLine
    for i in range(0, len(kernel.operation.indexAssignmentsB)):
      s += "  size_t const strideB" \
          + self.indexChars[kernel.operation.indexAssignmentsB[i]] \
          + "," + self.endLine
    # sizes
    for i in range(0, len(kernel.indexOrderC)+len(kernel.indexOrderSummation)):
      s += "  size_t const size" + self.indexChars[i] + "," + self.endLine
    # alpha & beta
    if kernel.operation.alpha:
      s += "  DATA_TYPE_STR_C const alpha," + self.endLine
    if kernel.operation.beta:
      s += "  DATA_TYPE_STR_C const beta," + self.endLine
    # TODO - if convolution, need stride and pad for each sum dim
    s += " )"
    return s



  ##############################################################################
  # make kernel body
  ##############################################################################
  def getBody( self, kernel ):

    ####################################
    # initializations - DONE
    kStr = ""
    kStr += self.endLine
    kStr += "/* %s */" % self.getName(kernel)
    kStr += self.endLine

    ####################################
    # kernel preprocessor definitions - DONE
    kStr += self.endLine
    kStr += "/* kernel parameters */" + self.endLine
    kStr += "#define WG_DIM0          %d%s" \
        % (kernel.tile.workGroupDim0, self.endLine )
    kStr += "#define WG_DIM1          %d%s" \
        % (kernel.tile.workGroupDim1, self.endLine )
    kStr += "#define MICRO_TILE_DIM0  %d%s" \
        % (kernel.tile.microTileDim0, self.endLine )
    kStr += "#define MICRO_TILE_DIM1  %d%s" \
        % (kernel.tile.microTileDim1, self.endLine )
    kStr += "#define MACRO_TILE_DIM0  %s%s" \
        % ((kernel.tile.workGroupDim0 * kernel.tile.microTileDim0), self.endLine )
    kStr += "#define MACRO_TILE_DIM1  %s%s" \
        % ((kernel.tile.workGroupDim1 * kernel.tile.microTileDim1), self.endLine )
    kStr += "#define NUM_UNROLL_ITER  %s%s" \
        % (kernel.unrolls[len(kernel.unrolls)-1], self.endLine )
    kStr += "" + self.endLine

    ####################################
    # global memory indices - DONE
    kStr += self.endLine
    kStr += "/* global memory indices */" + self.endLine
    # C
    kStr += "#define GET_GLOBAL_INDEX_C(IDX" \
        + self.indexChars[0]
    for i in range(1, len(kernel.indexOrderC)):
      kStr += ", IDX" + self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") ( IDX" + indexChar + "*strideC" + indexChar
    for i in range(1, len(kernel.indexOrderC)):
      indexChar = self.indexChars[i]
      kStr += " + IDX" + indexChar + "*strideC" + indexChar
    kStr += " )" + self.endLine
    # A
    kStr += "#define GET_GLOBAL_INDEX_A(IDX" \
        + self.indexChars[kernel.operation.indexAssignmentsA[0]]
    for i in range(1, len(kernel.operation.indexAssignmentsA)):
      kStr += ", IDX" + self.indexChars[kernel.operation.indexAssignmentsA[i]]
    indexChar = self.indexChars[kernel.operation.indexAssignmentsA[0]]
    kStr += ") ( IDX" + indexChar + "*strideA" + indexChar
    for i in range(1, len(kernel.operation.indexAssignmentsA)):
      indexChar = self.indexChars[kernel.operation.indexAssignmentsA[i]]
      kStr += " + IDX" + indexChar + "*strideA" + indexChar
    kStr += " )" + self.endLine
    # B
    kStr += "#define GET_GLOBAL_INDEX_B(IDX" \
        + self.indexChars[kernel.operation.indexAssignmentsB[0]]
    for i in range(1, len(kernel.operation.indexAssignmentsB)):
      kStr += ", IDX" + self.indexChars[kernel.operation.indexAssignmentsB[i]]
    indexChar = self.indexChars[kernel.operation.indexAssignmentsB[0]]
    kStr += ") ( IDX" + indexChar + "*strideB" + indexChar
    for i in range(1, len(kernel.operation.indexAssignmentsB)):
      indexChar = self.indexChars[kernel.operation.indexAssignmentsB[i]]
      kStr += " + IDX" + indexChar + "*strideB" + indexChar
    kStr += " )" + self.endLine


    ####################################
    # local memory indices - TODO
    kStr += self.endLine
    kStr += "/* local memory indices */" + self.endLine
    kStr += "#define GET_LOCAL_INDEX_A(DIM0,DIM1) ((DIM0) + (DIM1)*(MACRO_TILE_DIM0) )" + self.endLine
    kStr += "#define GET_LOCAL_INDEX_B(DIM0,DIM1) ((DIM1) + (DIM0)*(MACRO_TILE_DIM1) )" + self.endLine

    ####################################
    # data types - DONE
    kStr += self.endLine
    kStr += "/* data types */" + self.endLine
    kStr += "#define DATA_TYPE_STR_A %s%s" \
        % (kernel.dataTypeA.toOpenCL(), self.endLine)
    kStr += "#define DATA_TYPE_STR_B %s%s" \
        % (kernel.dataTypeB.toOpenCL(), self.endLine)
    kStr += "#define DATA_TYPE_STR_C %s%s" \
        % (kernel.dataTypeC.toOpenCL(), self.endLine)

    ####################################
    # MADs - DONE
    # TODO - mix real/complex
    if kernel.dataTypeC.isReal():
      # real data
      kStr += "#define TYPE_MAD(MULA,MULB,DST) " \
          + "DST = mad(MULA,MULB,DST);" + self.endLine
      if kernel.operation.alpha:
        if kernel.operation.beta:
          # dst = alpha*reg + beta*dst
          kStr += "#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) " \
              + "DST = (ALPHA)*(REG) + (BETA)*(DST);" + self.endLine
        else:
          # dst = alpha*reg
          kStr += "#define TYPE_MAD_WRITE(DST,ALPHA,REG) " \
              + "DST = (ALPHA)*(REG);" + self.endLine
      else:
        if kernel.operation.beta:
          # dst = reg + beta*dst
          kStr += "#define TYPE_MAD_WRITE(DST,REG,BETA) " \
              + "DST = (REG) + (BETA)*(DST);" + self.endLine
        else:
          # dst = reg
          kStr += "#define TYPE_MAD_WRITE(DST,REG) " \
              + "DST = (REG);" + self.endLine
    else:
      # complex data
      if not kernel.conjugateA and not kernel.conjugateB:
        # neither conjugate
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s0, MULB.s0, DST.s0 ); \\\\" + self.endLine +
          "  DST.s0 = mad( -MULA.s1, MULB.s1, DST.s0 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s0, MULB.s1, DST.s1 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif kernel.conjugateA and not kernel.conjugateB:
        # A conjugate (negate imaginary A.s1)
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s0, MULB.s0, DST.s0 ); \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s1, MULB.s1, DST.s0 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s0, MULB.s1, DST.s1 ); \\\\" + self.endLine +
          "  DST.s1 = mad( -MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif not kernel.conjugateA and kernel.conjugateB:
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
      if kernel.operation.alpha:
        if kernel.operation.beta:
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
        if kernel.operation.beta:
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
    # micro-tile - DONE
    kStr += self.endLine
    kStr += "/* %dx%d micro-tile */%s" % (kernel.tile.microTileDim0, kernel.tile.microTileDim1, self.endLine)
    kStr += "#define MICRO_TILE \\\\" + self.endLine
    for a in range(0, kernel.tile.microTileDim0):
      kStr += "  rA[%d] = localA[offA + %d*WG_DIM0]; \\\\%s" % (a, a, self.endLine)
    for b in range(0, kernel.tile.microTileDim1):
      kStr += "  rB[%d] = localB[offB + %d*WG_DIM1]; \\\\%s" % (b, b, self.endLine)
    kStr += "  offA += MACRO_TILE_DIM0; \\\\" + self.endLine
    kStr += "  offB += MACRO_TILE_DIM1; \\\\" + self.endLine
    for a in range(0, kernel.tile.microTileDim0):
      for b in range(0, kernel.tile.microTileDim1):
        kStr += "  TYPE_MAD(rA[%d],rB[%d],rC[%d][%d]); \\\\%s" % (a, b, a, b, self.endLine)
    kStr += "  mem_fence(CLK_LOCAL_MEM_FENCE);" + self.endLine
    kStr += self.endLine

    ####################################
    # function signature - DONE
    ####################################
    kStr += self.getSignature(kernel)
    kStr += " {" + self.endLine

    ####################################
    # apply offsets - DONE
    kStr += self.endLine
    kStr += (
      "  /* apply offsets */" + self.endLine +
      "  C += offsetC;" + self.endLine +
      "  A += offsetA;" + self.endLine +
      "  B += offsetB;" + self.endLine )

    ####################################
    # allocate registers - DONE
    kStr += self.endLine
    kStr += (
      "  /* allocate registers */" + self.endLine +
      "  DATA_TYPE_STR_C rC[MICRO_TILE_DIM0][MICRO_TILE_DIM1] "
          + "= {{0}};" + self.endLine +
      "  DATA_TYPE_STR_A rA[MICRO_TILE_DIM0];" + self.endLine +
      "  DATA_TYPE_STR_B rB[MICRO_TILE_DIM1];" + self.endLine )

    ####################################
    # allocate local memory - DONE
    kStr += self.endLine
    kStr += (
      "  /* allocate local memory */" + self.endLine +
      "  __local DATA_TYPE_STR localA[NUM_UNROLL_ITER*MACRO_TILE_DIM0];" \
          + self.endLine +
      "  __local DATA_TYPE_STR localB[NUM_UNROLL_ITER*MACRO_TILE_DIM1];" \
          + self.endLine )

    ####################################
    # c indices - DONE
    # kernel.indexOrderC - performance defined
    # kernel.indexOrderSummation - performance defined
    # kernel.indexAssignmentsA - user defined
    # kernel.indexAssignmentsB - user defined
    # convert get_group_id(0) to however many c indices there are
    kStr += self.endLine
    kStr += "  /* c indices */" + self.endLine
    for i in range(0, len(kernel.indexOrderC)):
      index = kernel.indexOrderC[i]
      kStr += "  size_t groupIdx" + self.indexChars[index] \
          + " = ( get_group_id(0)"
      for j in reversed( range( i+1, len(kernel.indexOrderC)) ):
        index2 = kernel.indexOrderC[j]
        kStr += " / size" + self.indexChars[index2]
      kStr += " ) % size" + self.indexChars[index] + ";" + self.endLine

    kStr += (
      "  uint localIdx0 = get_local_id(0);" + self.endLine +
      "  uint localIdx1 = get_local_id(1);" + self.endLine +
      "  uint localSerial = localIdx0 + localIdx1*WG_DIM0;" + self.endLine )

    # multidim if (kernel.order=="clblasColumnMajor")==(kernel.transA=="N"):
    tileIdxLaterA = kernel.indexAssignmentDim0 \
        > kernel.indexOrderSummation[len(kernel.indexOrderSummation)-1]
    tileIdxLaterB = kernel.indexAssignmentDim1 \
        > kernel.indexOrderSummation[len(kernel.indexOrderSummation)-1]
    unrollChar = self.indexChars[kernel.indexOrderSummation[ \
        len(kernel.indexOrderSummation)-1] + len(kernel.indexOrderC)]
    tile0Char = self.indexChars[kernel.indexAssignmentDim0]
    tile1Char = self.indexChars[kernel.indexAssignmentDim1]

    ####################################
    # global indices being loaded - TODO
    kStr += self.endLine
    kStr += "  /* global indices being loaded */" + self.endLine

    if tileIdxLaterA:
      kStr += (
        "#define globalIdxA" + tile0Char + "(LID) (groupIdx" + tile0Char + "*MACRO_TILE_DIM0 + (localSerial+(LID)*WG_DIM0*WG_DIM1)%MACRO_TILE_DIM0)" + self.endLine +
        "#define globalIdxA" + unrollChar + "(LID) ((localSerial+(LID)*WG_DIM0*WG_DIM1)/MACRO_TILE_DIM0)" + self.endLine )
    else:
      kStr += (
        "#define globalIdxA" + unrollChar + "(LID) (groupIdx" + tile1Char + "*MACRO_TILE_DIM0 + (localSerial+(LID)*WG_DIM0*WG_DIM1)/NUM_UNROLL_ITER)" + self.endLine +
        "#define globalIdxA" + tile0Char + "(LID) ((localSerial+(LID)*WG_DIM0*WG_DIM1)%NUM_UNROLL_ITER)" + self.endLine )

    if tileIdxLaterB:
      kStr += (
        "#define globalIdxB" + tile1Char + "(LID) ((localSerial+(LID)*WG_DIM0*WG_DIM1)%NUM_UNROLL_ITER)" + self.endLine +
        "#define globalIdxB" + unrollChar + "(LID) (groupCol*MACRO_TILE_DIM1 + (localSerial+(LID)*WG_DIM0*WG_DIM1)/NUM_UNROLL_ITER)" + self.endLine )
    else:
      kStr += (
        "#define globalIdxB" + unrollChar + "(LID) ((localSerial+(LID)*WG_DIM0*WG_DIM1)/MACRO_TILE_DIM1)" + self.endLine +
        "#define globalIdxB" + tile1Char + "(LID) (groupCol*MACRO_TILE_DIM1 + (localSerial+(LID)*WG_DIM0*WG_DIM1)%MACRO_TILE_DIM1)" + self.endLine )

    #kStr += (
    #  "  A += GET_GLOBAL_INDEX_A( globalARow, globalACol );" + self.endLine +
    #  "  B += GET_GLOBAL_INDEX_B( globalBRow, globalBCol );" + self.endLine )


    ####################################
    # summations loops - DONE
    indent = "  "
    kStr += indent + "/* iterate over all summation indices */" + self.endLine
    for i in range(0,len(kernel.indexOrderSummation)):
      indexChar = self.indexChars[kernel.indexOrderSummation[i] \
          + len(kernel.indexOrderC)]
      kStr += indent + "size_t sumIter" + indexChar \
          + " = size" + indexChar
      if i == len(kernel.indexOrderSummation)-1:
        kStr += " / NUM_UNROLL_ITER"
      kStr += ";" + self.endLine
      kStr += indent + "do {" + self.endLine
      indent += "  "

    ####################################
    # local indices being written
    # thoroughly verify by hand for 4 GEMM cases (after doing global) - TODO
    kStr += self.endLine
    kStr += "    /* local indices being written */" + self.endLine
# new indices will be localA_unroll and localA_tile, which gets assigned to row
    if tileIdxLaterA:
      kStr += "#define localA" + tile0Char \
          + " (localSerial % MACRO_TILE_DIM0)" + self.endLine \
          + "#define localA" + unrollChar \
          +  " (localSerial / MACRO_TILE_DIM0)" + self.endLine \
          + "#define localAStride (WG_DIM0*WG_DIM1)" + self.endLine
    else:
      kStr += "#define localA" + tile0Char \
          + "(localSerial / NUM_UNROLL_ITER)" + self.endLine \
          + "#define localA" + unrollChar \
          + " (localSerial % NUM_UNROLL_ITER)" + self.endLine \
          + "#define localAStride (WG_DIM0*WG_DIM1/NUM_UNROLL_ITER)" \
          + self.endLine

    if tileIdxLaterB:
      kStr += "#define localB" + tile1Char \
          + "( localSerial % NUM_UNROLL_ITER )" + self.endLine \
          + "#define localB" + unrollChar \
          + "( localSerial / NUM_UNROLL_ITER )" + self.endLine \
          + "#define localBStride (WG_DIM0*WG_DIM1/NUM_UNROLL_ITER)" \
          + self.endLine
    else:
      kStr += "#define localB" + tile1Char \
          + "( localSerial / MACRO_TILE_DIM1 )" + self.endLine \
          + "#define localB" + unrollChar \
          + "( localSerial % MACRO_TILE_DIM1 )" + self.endLine \
          + "#define localBStride  (WG_DIM0*WG_DIM1)" + self.endLine

    kStr += indent + "__local DATA_TYPE_STR *lA = localA" \
        + " + GET_LOCAL_INDEX_A(localA" + tile0Char + ", localA" \
        + unrollChar + ");" + self.endLine \
        + indent + "__local DATA_TYPE_STR *lB = localB" \
        + " + GET_LOCAL_INDEX_B(localB" + tile1Char + ", localB" \
        + unrollChar + ");" + self.endLine \
        + indent + "barrier(CLK_LOCAL_MEM_FENCE);" + self.endLine

    ####################################
    # how many elements to load global -> local - DONE
    # threads to do loading = (workGroupDim0*workGroupDim1)
    # A elements to be loaded = workGroupDim0*microTileDim0*unroll
    # B elements to be loaded = workGroupDim1*microTileDim1*unroll
    kStr += self.endLine
    kStr += indent + "/* load global -> local */" + self.endLine
    numALoads  = (kernel.tile.workGroupDim0*kernel.tile.microTileDim0*kernel.unrolls[len(kernel.unrolls)-1]) \
        / (kernel.tile.workGroupDim0*kernel.tile.workGroupDim1)
    numALoadsR = (kernel.tile.workGroupDim0*kernel.tile.microTileDim0*kernel.unrolls[len(kernel.unrolls)-1]) \
        % (kernel.tile.workGroupDim0*kernel.tile.workGroupDim1)
    numBLoads  = (kernel.tile.workGroupDim1*kernel.tile.microTileDim1*kernel.unrolls[len(kernel.unrolls)-1]) \
        / (kernel.tile.workGroupDim0*kernel.tile.workGroupDim1)
    numBLoadsR = (kernel.tile.workGroupDim1*kernel.tile.microTileDim1*kernel.unrolls[len(kernel.unrolls)-1]) \
        % (kernel.tile.workGroupDim0*kernel.tile.workGroupDim1)

    # zeroString for real and complex
    if kernel.dataTypeA.value == Structs.DataType.singleComplex:
      zeroStringA = "(float2)(0.f, 0.f)"
    elif kernel.dataTypeA.value == Structs.DataType.doubleComplex:
      zeroStringA = "(double2)(0.0, 0.0)"
    else:
      zeroStringA = "0.0"
    if kernel.dataTypeB.value == Structs.DataType.singleComplex:
      zeroStringB = "(float2)(0.f, 0.f)"
    elif kernel.dataTypeB.value == Structs.DataType.doubleComplex:
      zeroStringB = "(double2)(0.0, 0.0)"
    else:
      zeroStringB = "0.0"
    if kernel.dataTypeC.value == Structs.DataType.singleComplex:
      zeroStringC = "(float2)(0.f, 0.f)"
    elif kernel.dataTypeC.value == Structs.DataType.doubleComplex:
      zeroStringC = "(double2)(0.0, 0.0)"
    else:
      zeroStringC = "0.0"



    ####################################
    # load global -> local - DONE
    for a in range(0, numALoads):
      kStr += indent + "lA[ %d*localAStride ] = " % a
      if self.isEdge(kernel, 0):
        kStr += "( globalARow(%d) >= M) ? %s : " % ( a, zeroStringA )
      kStr += "A[ GET_GLOBAL_INDEX_A( "
      kStr += "globalIdxA" + self.indexChars[ \
          kernel.operation.indexAssignmentsA[0]]  \
          + "(" + str(a) + ")"
      for i in range(1,len(kernel.operation.indexAssignmentsA)):
        kStr += ", globalIdxA" + self.indexChars[ \
            kernel.operation.indexAssignmentsA[i]]  \
            + "(" + str(a) + ")"
      kStr += " ) ];" + self.endLine

    if numALoadsR:
      kStr += indent + "if ( localSerial + " + str(numALoads) + "*WG_DIM0*WG_DIM1 < (WG_DIM0*MICRO_TILE_DIM0*NUM_UNROLL_ITER) ) {" + self.endLine
      kStr += indent + "  lA[ %d*localAStride ] = " % numALoads
      if self.isEdge(kernel,0):
        kStr += "( globalARow(%d) >= M) ? %s : " % ( numALoads, zeroStringA )
      kStr += "A[ GET_GLOBAL_INDEX_A( "
      kStr += "globalIdxA" + self.indexChars[ \
          kernel.operation.indexAssignmentsA[0]]  \
          + "(" + str(a) + ")"
      for i in range(1,len(kernel.operation.indexAssignmentsA)):
        kStr += ", globalIdxA" + self.indexChars[ \
            kernel.operation.indexAssignmentsA[i]]  \
            + "(" + str(a) + ")"
      kStr += " ) ];" + self.endLine
      kStr += indent + "}" + self.endLine

    for b in range(0, numBLoads):
      kStr += indent + "lB[ %d*localBStride ] = " % b
      if self.isEdge(kernel,1):
        kStr += "( globalBCol(%d) >= N) ? %s : " % ( b, zeroStringB )
      kStr += "B[ GET_GLOBAL_INDEX_B( "
      kStr += "globalIdxB" + self.indexChars[ \
          kernel.operation.indexAssignmentsB[0]]  \
          + "(" + str(b) + ")"
      for i in range(1,len(kernel.operation.indexAssignmentsB)):
        kStr += ", globalIdxB" + self.indexChars[ \
            kernel.operation.indexAssignmentsB[i]]  \
            + "(" + str(b) + ")"
      kStr += " ) ];" + self.endLine

    if numBLoadsR:
      kStr += indent + "if ( localSerial + " + str(numBLoads) + "*WG_DIM0*WG_DIM1 < (WG_DIM1*MICRO_TILE_DIM1*NUM_UNROLL_ITER) ) {" + self.endLine
      kStr += indent + "  lB[ %d*localBStride ] = " % numBLoads
      if self.isEdge(kernel,1):
        kStr += "(globalBCol(%d) >= N) ? %s : " % ( numBLoads, zeroStringB )
      kStr += "B[ GET_GLOBAL_INDEX_B( "
      kStr += "globalIdxB" + self.indexChars[ \
          kernel.operation.indexAssignmentsB[0]]  \
          + "(" + str(b) + ")"
      for i in range(1,len(kernel.operation.indexAssignmentsB)):
        kStr += ", globalIdxB" + self.indexChars[ \
            kernel.operation.indexAssignmentsB[i]]  \
            + "(" + str(b) + ")"
      kStr += " ) ];" + self.endLine
      kStr += indent + "}" + self.endLine
    kStr += (
      indent + "barrier(CLK_LOCAL_MEM_FENCE);" + self.endLine +
      indent + "uint offA = localIdx0;" + self.endLine +
      indent + "uint offB = localIdx1;" + self.endLine )

    ####################################
    # do mads - DONE
    kStr += self.endLine
    kStr += indent + "/* do mads */" + self.endLine
    for u in range(0, kernel.unrolls[len(kernel.unrolls)-1]):
      kStr += indent + "MICRO_TILE" + self.endLine

    ####################################
    # end loop - DONE
    for i in reversed(range(0,len(kernel.indexOrderSummation))):
      loopChar = self.indexChars[kernel.indexOrderSummation[i] \
          + len(kernel.indexOrderC)]
      # advance A, B along summation dimension
      kStr += indent + "A += strideA" + loopChar
      if i==len(kernel.indexOrderSummation)-1:
        kStr += "*NUM_UNROLL_ITER"
      else:
        for j in range(i+1,len(kernel.indexOrderSummation)):
          tmpChar = self.indexChars[kernel.indexOrderSummation[j] \
              + len(kernel.indexOrderC)]
          kStr += " - strideA" + tmpChar + "*size" + tmpChar
      kStr += ";" + self.endLine
      kStr += indent + "B += strideB" + loopChar
      if i==len(kernel.indexOrderSummation)-1:
        kStr += "*NUM_UNROLL_ITER"
      else:
        for j in range(i+1,len(kernel.indexOrderSummation)):
          tmpChar = self.indexChars[kernel.indexOrderSummation[j] \
              + len(kernel.indexOrderC)]
          kStr += " - strideB" + tmpChar + "*size" + tmpChar
      kStr += ";" + self.endLine
      indent = indent[2:]
      # close do-while loop
      kStr += indent + "} while (--sumIter" + loopChar + " > 0);" + self.endLine
    kStr += self.endLine

    ####################################
    # which global Cij index - DONE
    kStr += self.endLine
    kStr += "  /* which global Cij index */" + self.endLine
    for i in range(0, len(kernel.indexOrderC)):
      index = kernel.indexOrderC[i]
      kStr += "  size_t globalIdx" + self.indexChars[index] \
          + " = groupIdx" + self.indexChars[index]
      if index == kernel.indexAssignmentDim0:
        kStr += "*MACRO_TILE_DIM0 + localIdx0"
      if index == kernel.indexAssignmentDim1:
        kStr += "*MACRO_TILE_DIM1 + localIdx1"
      kStr += ";" + self.endLine

    ####################################
    # write global Cij - DONE
    kStr += self.endLine
    kStr += "  /* write global C */" + self.endLine
    if kernel.dataTypeC == Structs.DataType.singleComplex:
      kStr += "  float type_mad_tmp;" + self.endLine
    if kernel.dataTypeC == Structs.DataType.doubleComplex:
      kStr += "  double type_mad_tmp;" + self.endLine

    for a in range(0, kernel.tile.microTileDim0):
      for b in range(0, kernel.tile.microTileDim1):
        numEdges = 0
        for i in range(0, len(kernel.indexOrderC)):
          if self.isEdge(kernel,i):
            kStr += "  if (globalIdx" + self.indexChars[i]
            if i == kernel.indexAssignmentDim0:
              kStr += " + " + str(a) + "*WG_DIM0"
            if i == kernel.indexAssignmentDim1:
              kStr += " + " + str(b) + "*WG_DIM1"
            kStr += " < size" + self.indexChars[i] + ") {"
            numEdges += 1

        kStr += "  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C("
        for i in range(0, len(kernel.indexOrderC)):
          kStr += " globalIdx" + self.indexChars[i]
          if i == kernel.indexAssignmentDim0:
            kStr += " + " + str(a) + "*WG_DIM0"
          if i == kernel.indexAssignmentDim1:
            kStr += " + " + str(b) + "*WG_DIM1"
          if i < len(kernel.indexOrderC)-1:
            kStr += ","
        kStr += ") ]"
        if kernel.operation.alpha:
          kStr += ", alpha"
        kStr += ", rC[%d][%d]" % (a, b)
        if kernel.operation.beta:
          kStr += ", beta"
        kStr += ")"
        for i in range(0,numEdges):
          kStr += " }"
        kStr += self.endLine

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
  alpha = False
  beta = False
  operation = Structs.Operation( \
      operationType, \
      alpha, \
      beta, \
      numFreeIndices, \
      numIndicesBatch, \
      numIndicesSummation, \
      indexAssignmentsA, \
      indexAssignmentsB )

  kernel = Kernel(\
      operation, \
      tensorA, \
      tensorB, \
      tensorC, \
      alpha, \
      beta )

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
  alpha = False
  beta = False

  kernel = Kernel(\
      operation, \
      tensorA, \
      tensorB, \
      tensorC, \
      alpha, \
      beta )

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

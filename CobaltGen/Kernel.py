import os
import sys
import copy
import argparse

import Structs


################################################################################
# Make OpenCL Kernel String
################################################################################
class Kernel:

  endLine = "\\n\"\n\""
  indexChars = [ "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", \
      "T", "U", "V", "W", "X", "Y", "Z" ]

  ##############################################################################
  # Make OpenCL Kernel String
  ##############################################################################
  def __init__(self, \
      operation, \
      dataTypeA, \
      dataTypeB, \
      dataTypeC, \
      alpha, \
      beta, \
      ):

    # operation
    self.operation = operation
    self.freeIndexAssignments = []
    self.summationIndexAssignments = []
    self.tileIndexAssignment0 = -1
    self.tileIndexAssignment1 = -1
    # self.unrollIndexAssignment = -1 always last
    self.indexAssignmentsMade = False

    # non-tile
    self.dataTypeA = dataTypeA
    self.dataTypeB = dataTypeB
    self.dataTypeC = dataTypeC
    self.alpha = alpha
    self.beta = beta

    # tile
    self.workGroupDim0 = -1
    self.workGroupDim1 = -1
    self.microTileDim0 = -1
    self.microTileDim1 = -1
    self.macroTileDim0 = -1
    self.macroTileDim1 = -1
    self.unroll        = -1

  ##############################################################################
  # Row Kernel
  # - macroTileDim0 == 1
  # - guards around gA -> lA
  # - guards around gC[gRow,:] = rC[row,:]
  ##############################################################################
  def isRow(self):
    return self.workGroupDim0 * self.microTileDim0 \
        != self.macroTileDim0

  ##############################################################################
  # Col Kernel
  # - macroTileDim1 == 1
  # - guards around gB -> lB
  # - guards around gC[:,gCol] = rC[:,col]
  ##############################################################################
  def isCol(self):
    return self.workGroupDim1 * self.microTileDim1 \
        != self.macroTileDim1

  ##############################################################################
  # Corner Kernel
  # - macroTileDim0,Cols == 1
  # - guards around gA -> lA, gB -> lB
  # - guards around gC[gRow,:] = rC[row,:], gC[:,gCol] = rC[:,col]
  ##############################################################################
  def isCor(self):
    return self.isRow() and self.isCol()

  ##############################################################################
  # Make Index Assignments - TODO
  ##############################################################################
  def makeIndexAssignments(self):
    self.freeIndexAssignments = []
    for i in range(0,self.operation.numFreeIndices):
      self.freeIndexAssignments.append( i )
    self.summationIndexAssignments = []
    for i in range(0,self.operation.numSummationIndices):
      self.summationIndexAssignments.append( i )

    self.indexAssignmentsMade = True

  ##############################################################################
  # get kernel name - DONE
  ##############################################################################
  def getName( self ):
    if not self.indexAssignmentsMade:
      self.makeIndexAssignments()

    kernelName = ""

    # operation type
    kernelName += self.operation.type.toString()
    kernelName += "_"

    # data dataTypes
    kernelName += self.dataTypeA.toChar().upper()
    kernelName += self.dataTypeB.toChar().upper()
    kernelName += self.dataTypeC.toChar().upper()
    kernelName += "_"

    # C dimensions
    kernelName += "C"
    kernelName += str(self.operation.numFreeIndices)
    kernelName += "_"

    # A dimensions
    for i in range(0, len(self.operation.indexAssignmentsA)):
      kernelName += str(self.operation.indexAssignmentsA[i])
    kernelName += "_"

    # B dimensions
    for i in range(0,len(self.operation.indexAssignmentsB)):
      kernelName += str(self.operation.indexAssignmentsB[i])
    kernelName += "_"

    # alpha
    kernelName += "a"
    if self.alpha:
      kernelName += "1"
    else:
      kernelName += "0"
    kernelName += "_"

    # beta
    kernelName += "b"
    if self.beta:
      kernelName += "1"
    else:
      kernelName += "0"
    kernelName += "_"

    # free indices
    for i in range(0,len(self.freeIndexAssignments)):
      index = self.freeIndexAssignments[i]
      multipleStr = "X1"
      if index == self.tileIndexAssignment0:
        multipleStr = "X" + str(self.workGroupDim0) \
            + "x" + str(self.microTileDim0)
      if index == self.tileIndexAssignment1:
        multipleStr = "X" + str(self.workGroupDim1) \
            + "x" + str(self.microTileDim1)
      kernelName += str(index) + multipleStr
      kernelName += "_"

    # summation indices
    for i in range(0,len(self.summationIndexAssignments)):
      index = self.summationIndexAssignments[i]
      multiple = 1
      if index == len(self.summationIndexAssignments)-1:
        multiple = self.unroll
      kernelName += str(index) + ":" + str(multiple)
      kernelName += "_"
    return kernelName


  ##############################################################################
  # get kernel signature - DONE
  ##############################################################################
  def getSignature(self, backend):
    s = ""
    # kernel name
    s += "__attribute__((reqd_work_group_size(WG_DIM1,WG_DIM0,1)))"
    s += self.endLine
    s += "__self void %s" % ( self.getName() )
    s += "(" + self.endLine
    # tensor data
    s += (
      "  __global DATA_TYPE_STR_C       *          C," + self.endLine +
      "  __global DATA_TYPE_STR_A const * restrict A," + self.endLine +
      "  __global DATA_TYPE_STR_B const * restrict B," + self.endLine +
      "  size_t const offsetC," + self.endLine +
      "  size_t const offsetA," + self.endLine +
      "  size_t const offsetB," + self.endLine )
    # TODO - if convolution, need stride and pad for each sum dim
    if self.alpha:
      s += "  DATA_TYPE_STR_C const alpha," + self.endLine
    if self.beta:
      s += "  DATA_TYPE_STR_C const beta," + self.endLine
    # tensor C dimensions
    for i in range(0, self.operation.numFreeIndices):
      s += "  size_t const strideC" + str(i) + "," + self.endLine
    for i in range(0, self.operation.numFreeIndices):
      s += "  size_t const sizeC" + str(i) + "," + self.endLine
    # tensor A dimensions
    for i in range(0, len(self.operation.indexAssignmentsA)):
      s += "  size_t const strideA" + str(i) + "," + self.endLine
    # tensor B dimensions
    for i in range(0, len(self.operation.indexAssignmentsB)):
      s += "  size_t const strideB" + str(i) + "," + self.endLine
    # summation dimensions
    for i in range(0, len(self.operation.indexAssignmentsA)):
      s += "  size_t const sumSize" + str(i)
      if i < len(self.operation.indexAssignmentsA):
        s += "," + self.endLine
    s += " )"
    return s



  ##############################################################################
  # make kernel body
  ##############################################################################
  def getBody( self, backend):

    if not self.indexAssignmentsMade:
      self.makeIndexAssignments()

    numDimensionsA = len(self.operation.indexAssignmentsA)
    numDimensionsB = len(self.operation.indexAssignmentsB)
    numDimensionsC = self.operation.numFreeIndices

    ####################################
    # initializations - DONE
    kStr = ""
    kStr += self.endLine
    kStr += "/* %s */" % self.getName()
    kStr += self.endLine

    ####################################
    # kernel preprocessor definitions - DONE
    kStr += self.endLine
    kStr += "/* kernel parameters */" + self.endLine
    kStr += "#define WG_DIM0          %d%s" \
        % (self.workGroupDim0, self.endLine )
    kStr += "#define WG_DIM1          %d%s" \
        % (self.workGroupDim1, self.endLine )
    kStr += "#define MICRO_TILE_DIM0  %d%s" \
        % (self.microTileDim0, self.endLine )
    kStr += "#define MICRO_TILE_DIM1  %d%s" \
        % (self.microTileDim1, self.endLine )
    kStr += "#define MACRO_TILE_DIM0  %s%s" \
        % ((self.workGroupDim0 * self.microTileDim0), self.endLine )
    kStr += "#define MACRO_TILE_DIM1  %s%s" \
        % ((self.workGroupDim1 * self.microTileDim1), self.endLine )
    kStr += "#define NUM_UNROLL_ITER      %s%s" \
        % (self.unroll, self.endLine )
    kStr += "" + self.endLine

    ####################################
    # global memory indices
    kStr += self.endLine
    kStr += "/* global memory indices */" + self.endLine
    kStr += "#define GET_GLOBAL_INDEX_A(IDX_" \
        + self.indexChars[self.operation.indexAssignmentsA[0]]
    for i in range(1, numDimensionsA):
      kStr += ", IDX_" + self.indexChars[self.operation.indexAssignmentsA[i]]
    kStr += ") ( IDX_" + self.indexChars[self.operation.indexAssignmentsA[0]] \
        + "*strideA0"
    for i in range(1, numDimensionsA):
      kStr += " + IDX_" + self.indexChars[self.operation.indexAssignmentsA[i]] \
          + "*strideA" + str(i)
    kStr += " )" + self.endLine
    kStr += "#define GET_GLOBAL_INDEX_B(IDX_" \
        + self.indexChars[self.operation.indexAssignmentsB[0]]
    for i in range(1, numDimensionsB):
      kStr += ", IDX_" + self.indexChars[self.operation.indexAssignmentsB[i]]
    kStr += ") ( IDX_" + self.indexChars[self.operation.indexAssignmentsB[0]] \
        + "*strideB0"
    for i in range(1, numDimensionsB):
      kStr += " + IDX_" + self.indexChars[self.operation.indexAssignmentsB[i]] \
          + "*strideB" + str(i)
    kStr += " )" + self.endLine
    kStr += "#define GET_GLOBAL_INDEX_C(IDX_" \
        + self.indexChars[0]
    for i in range(1, numDimensionsC):
      kStr += ", IDX_" + self.indexChars[i]
    kStr += ") ( IDX_" + self.indexChars[0] \
        + "*strideC0"
    for i in range(1, numDimensionsC):
      kStr += " + IDX_" + self.indexChars[i] \
          + "*strideC" + str(i)
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
        % (self.dataTypeA.toOpenCL(), self.endLine)
    kStr += "#define DATA_TYPE_STR_B %s%s" \
        % (self.dataTypeB.toOpenCL(), self.endLine)
    kStr += "#define DATA_TYPE_STR_C %s%s" \
        % (self.dataTypeC.toOpenCL(), self.endLine)

    ####################################
    # MADs - DONE
    # TODO - mix real/complex
    if self.dataTypeC.isReal():
      # real data
      kStr += "#define TYPE_MAD(MULA,MULB,DST) " \
          + "DST = mad(MULA,MULB,DST);" + self.endLine
      if self.alpha:
        if self.beta:
          # dst = alpha*reg + beta*dst
          kStr += "#define TYPE_MAD_WRITE(DST,ALPHA,REG,BETA) " \
              + "DST = (ALPHA)*(REG) + (BETA)*(DST);" + self.endLine
        else:
          # dst = alpha*reg
          kStr += "#define TYPE_MAD_WRITE(DST,ALPHA,REG) " \
              + "DST = (ALPHA)*(REG);" + self.endLine
      else:
        if self.beta:
          # dst = reg + beta*dst
          kStr += "#define TYPE_MAD_WRITE(DST,REG,BETA) " \
              + "DST = (REG) + (BETA)*(DST);" + self.endLine
        else:
          # dst = reg
          kStr += "#define TYPE_MAD_WRITE(DST,REG) " \
              + "DST = (REG);" + self.endLine
    else:
      # complex data
      if not self.conjugateA and not self.conjugateB:
        # neither conjugate
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s0, MULB.s0, DST.s0 ); \\\\" + self.endLine +
          "  DST.s0 = mad( -MULA.s1, MULB.s1, DST.s0 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s0, MULB.s1, DST.s1 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif self.conjugateA and not self.conjugateB:
        # A conjugate (negate imaginary A.s1)
        kStr += (
          "#define TYPE_MAD(MULA,MULB,DST) \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s0, MULB.s0, DST.s0 ); \\\\" + self.endLine +
          "  DST.s0 = mad(  MULA.s1, MULB.s1, DST.s0 ); \\\\" + self.endLine +
          "  DST.s1 = mad(  MULA.s0, MULB.s1, DST.s1 ); \\\\" + self.endLine +
          "  DST.s1 = mad( -MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif not self.conjugateA and self.conjugateB:
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
      if self.alpha:
        if self.beta:
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
        if self.beta:
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
    kStr += "/* %dx%d micro-tile */%s" % (self.microTileDim0, self.microTileDim1, self.endLine)
    kStr += "#define MICRO_TILE \\\\" + self.endLine
    for a in range(0, self.microTileDim0):
      kStr += "  rA[%d] = localA[offA + %d*WG_DIM0]; \\\\%s" % (a, a, self.endLine)
    for b in range(0, self.microTileDim1):
      kStr += "  rB[%d] = localB[offB + %d*WG_DIM1]; \\\\%s" % (b, b, self.endLine)
    kStr += "  offA += (MACRO_TILE_DIM0+LOCAL_COL_PAD); \\\\" + self.endLine
    kStr += "  offB += (MACRO_TILE_DIM1+LOCAL_ROW_PAD); \\\\" + self.endLine
    for a in range(0, self.microTileDim0):
      for b in range(0, self.microTileDim1):
        kStr += "  TYPE_MAD(rA[%d],rB[%d],rC[%d][%d]); \\\\%s" % (a, b, a, b, self.endLine)
    kStr += "  mem_fence(CLK_LOCAL_MEM_FENCE);" + self.endLine
    kStr += self.endLine

    ####################################
    # function signature - DONE
    ####################################
    kStr += self.getSignature(backend)
    kStr += " {" + self.endLine

    ####################################
    # apply offsets - DONE
    kStr += self.endLine
    kStr += (
      "  /* apply offsets */" + self.endLine +
      "  A += offsetA;" + self.endLine +
      "  B += offsetB;" + self.endLine +
      "  C += offsetC;" + self.endLine )

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
    # free indices - DONE
    # self.freeIndexAssignments - performance defined
    # self.summationIndexAssignments - performance defined
    # self.indexAssignmentsA - user defined
    # self.indexAssignmentsB - user defined
    # convert get_group_id(0) to however many free indices there are
    kStr += self.endLine
    kStr += "  /* free indices */" + self.endLine
    for i in range(0, self.operation.numFreeIndices):
      index = self.freeIndexAssignments[i]
      kStr += "  size_t freeIdx" + self.indexChars[i] \
          + " = ( get_group_id(0)"
      for j in range( i, self.operation.numFreeIndices):
        index2 = self.freeIndexAssignments[j]
        kStr += " / sizeC" + str(index2)
      kStr += " ) % sizeC" + str(index) + ";" + self.endLine

    kStr += (
      "  uint localIdx0 = get_local_id(0);" + self.endLine +
      "  uint localIdx1 = get_local_id(1);" + self.endLine +
      "  uint localSerial = localIdx0 + localIdx1*WG_DIM0;" + self.endLine )

    ####################################
    # global indices being loaded - TODO
    kStr += self.endLine
    kStr += "  /* global indices being loaded */" + self.endLine

    """
    if (self.order=="clblasColumnMajor")==(self.transA=="N"):
      kStr += (
        "#define globalARow(LID) (groupRow*MACRO_TILE_DIM0 + (localSerial+(LID)*WG_DIM0*WG_DIM1)%MACRO_TILE_DIM0)" + self.endLine +
        "#define globalACol(LID) ((localSerial+(LID)*WG_DIM0*WG_DIM1)/MACRO_TILE_DIM0)" + self.endLine )
    else:
      kStr += (
        "#define globalARow(LID) (groupRow*MACRO_TILE_DIM0 + (localSerial+(LID)*WG_DIM0*WG_DIM1)/NUM_UNROLL_ITER)" + self.endLine +
        "#define globalACol(LID) ((localSerial+(LID)*WG_DIM0*WG_DIM1)%NUM_UNROLL_ITER)" + self.endLine )

    if (self.order=="clblasColumnMajor")==(self.transB=="N"):
      kStr += (
        "#define globalBRow(LID) ((localSerial+(LID)*WG_DIM0*WG_DIM1)%NUM_UNROLL_ITER)" + self.endLine +
        "#define globalBCol(LID) (groupCol*MACRO_TILE_DIM1 + (localSerial+(LID)*WG_DIM0*WG_DIM1)/NUM_UNROLL_ITER)" + self.endLine )
    else:
      kStr += (
        "#define globalBRow(LID) ((localSerial+(LID)*WG_DIM0*WG_DIM1)/MACRO_TILE_DIM1)" + self.endLine +
        "#define globalBCol(LID) (groupCol*MACRO_TILE_DIM1 + (localSerial+(LID)*WG_DIM0*WG_DIM1)%MACRO_TILE_DIM1)" + self.endLine )
    """

    #kStr += (
    #  "  A += GET_GLOBAL_INDEX_A( globalARow, globalACol );" + self.endLine +
    #  "  B += GET_GLOBAL_INDEX_B( globalBRow, globalBCol );" + self.endLine )


    ####################################
    # summations loops - DONE
    indent = "  "
    for i in range(0,len(self.summationIndexAssignments)):
      indexChar = self.indexChars[self.summationIndexAssignments[i] \
          + self.operation.numFreeIndices]
      kStr += indent + "size_t sumIdx" + indexChar \
          + " = sumSize" + indexChar
      if i == len(self.summationIndexAssignments):
        kStr += " / NUM_UNROLL_ITER"
      kStr += ";" + self.endLine
      kStr += indent + "do {" + self.endLine

    ####################################
    # local indices being written
    kStr += self.endLine
    """
    kStr += "    /* local indices being written */" + self.endLine
    if (self.order=="clblasColumnMajor")==(self.transA=="N"):
      kStr += (
        "#define localARow (localSerial % MACRO_TILE_DIM0)" + self.endLine +
        "#define localACol (localSerial / MACRO_TILE_DIM0)" + self.endLine +
        "#define localAStride (WG_DIM0*WG_DIM1)" + self.endLine )
    else:
      kStr += (
        "#define localARow (localSerial / NUM_UNROLL_ITER)" + self.endLine +
        "#define localACol (localSerial % NUM_UNROLL_ITER)" + self.endLine +
        "#define localAStride (WG_DIM0*WG_DIM1/NUM_UNROLL_ITER)" + self.endLine )

    if (self.order=="clblasColumnMajor")==(self.transB=="N"):
      kStr += (
        "#define localBRow ( localSerial % NUM_UNROLL_ITER )" + self.endLine +
        "#define localBCol ( localSerial / NUM_UNROLL_ITER )" + self.endLine +
        "#define localBStride (WG_DIM0*WG_DIM1/NUM_UNROLL_ITER)" + self.endLine )
    else:
      kStr += (
        "#define localBRow ( localSerial / MACRO_TILE_DIM1 )" + self.endLine +
        "#define localBCol ( localSerial % MACRO_TILE_DIM1 )" + self.endLine +
        "#define localBStride  (WG_DIM0*WG_DIM1)" + self.endLine )
    """

    kStr += (
      "    __local DATA_TYPE_STR *lA = localA + GET_LOCAL_INDEX_A(localARow, localACol);" + self.endLine +
      "    __local DATA_TYPE_STR *lB = localB + GET_LOCAL_INDEX_B(localBRow, localBCol);" + self.endLine +
      "    barrier(CLK_LOCAL_MEM_FENCE);" + self.endLine )


    ####################################
    # how many elements to load global -> local
    # threads to do loading = (workGroupDim0*workGroupDim1)
    # A elements to be loaded = workGroupDim0*microTileDim0*unroll
    # B elements to be loaded = workGroupDim1*microTileDim1*unroll
    kStr += self.endLine
    kStr += "    /* load global -> local */" + self.endLine
    numALoads  = (self.workGroupDim0*self.microTileDim0*self.unroll) \
        / (self.workGroupDim0*self.workGroupDim1)
    numALoadsR = (self.workGroupDim0*self.microTileDim0*self.unroll) \
        % (self.workGroupDim0*self.workGroupDim1)
    numBLoads  = (self.workGroupDim1*self.microTileDim1*self.unroll) \
        / (self.workGroupDim0*self.workGroupDim1)
    numBLoadsR = (self.workGroupDim1*self.microTileDim1*self.unroll) \
        % (self.workGroupDim0*self.workGroupDim1)

    # zeroString for real and complex
    if self.dataTypeA.value == Structs.DataType.singleComplex:
      zeroStringA = "(float2)(0.f, 0.f)"
    elif self.dataTypeA.value == Structs.DataType.doubleComplex:
      zeroStringA = "(double2)(0.0, 0.0)"
    else:
      zeroStringA = "0.0"
    if self.dataTypeB.value == Structs.DataType.singleComplex:
      zeroStringB = "(float2)(0.f, 0.f)"
    elif self.dataTypeB.value == Structs.DataType.doubleComplex:
      zeroStringB = "(double2)(0.0, 0.0)"
    else:
      zeroStringB = "0.0"
    if self.dataTypeC.value == Structs.DataType.singleComplex:
      zeroStringC = "(float2)(0.f, 0.f)"
    elif self.dataTypeC.value == Structs.DataType.doubleComplex:
      zeroStringC = "(double2)(0.0, 0.0)"
    else:
      zeroStringC = "0.0"



    ####################################
    # load global -> local
    for a in range(0, numALoads):
      kStr += "    lA[ %d*localAStride ] = " % a
      if self.isRow():
        kStr += "( globalARow(%d) >= M) ? %s : " % ( a, zeroStringA )
      kStr += "A[ GET_GLOBAL_INDEX_A( globalARow(%d), globalACol(%d) ) ];%s" % (a, a, self.endLine)
    if numALoadsR:
      kStr += "    if ( localSerial + " + str(numALoads) + "*WG_DIM0*WG_DIM1 < (WG_DIM0*MICRO_TILE_DIM0*NUM_UNROLL_ITER) ) {" + self.endLine
      kStr += "      lA[ %d*localAStride ] = " % numALoads
      if self.isRow():
        kStr += "( globalARow(%d) >= M) ? %s : " % ( numALoads, zeroStringA )
      kStr += "A[ GET_GLOBAL_INDEX_A( globalARow(%d), globalACol(%d) ) ];%s" % (numALoads, numALoads, self.endLine)
      kStr += "    }" + self.endLine

    for b in range(0, numBLoads):
      kStr += "    lB[ %d*localBStride ] = " % b
      if self.isCol():
        kStr += "( globalBCol(%d) >= N) ? %s : " % ( b, zeroStringB )
      kStr += "B[ GET_GLOBAL_INDEX_B( globalBRow(%d), globalBCol(%d) ) ];%s" % (b, b, self.endLine)
    if numBLoadsR:
      kStr += "    if ( localSerial + " + str(numBLoads) + "*WG_DIM0*WG_DIM1 < (WG_DIM1*MICRO_TILE_DIM1*NUM_UNROLL_ITER) ) {" + self.endLine
      kStr += "      lB[ %d*localBStride ] = " % numBLoads
      if self.isCol():
        kStr += "(globalBCol(%d) >= N) ? %s : " % ( numBLoads, zeroStringB )
      kStr += "B[ GET_GLOBAL_INDEX_B( globalBRow(%d), globalBCol(%d) ) ];%s" % (numBLoads, numBLoads, self.endLine)
      kStr += "    }" + self.endLine
    kStr += (
      "    barrier(CLK_LOCAL_MEM_FENCE);" + self.endLine +
      "    uint offA = localRow;" + self.endLine +
      "    uint offB = localCol;" + self.endLine )

    ####################################
    # do mads
    kStr += self.endLine
    kStr += "    /* do mads */" + self.endLine
    for u in range(0, self.unroll):
      kStr += "    MICRO_TILE" + self.endLine

    ####################################
    # shift to next k block
    kStr += self.endLine
    kStr += "    /* shift to next k block */" + self.endLine
    """
    if (self.order=="clblasColumnMajor")==(self.transA=="N"):
      kStr += "    A += lda*NUM_UNROLL_ITER;" + self.endLine
    else:
      kStr += "    A += NUM_UNROLL_ITER;" + self.endLine
    if (self.order=="clblasColumnMajor")==(self.transB=="N"):
      kStr += "    B += NUM_UNROLL_ITER;" + self.endLine
    else:
      kStr += "    B += ldb*NUM_UNROLL_ITER;" + self.endLine
    """

    ####################################
    # end loop - DONE
    for i in reversed(range(0,len(self.summationIndexAssignments))):
      indexChar = self.indexChars[self.summationIndexAssignments[i] + self.operation.numFreeIndices]
      kStr += indent + "} while (--sumIdx" + indexChar + " > 0);" + self.endLine
    kStr += self.endLine

    ####################################
    # which global Cij index
    kStr += self.endLine
    kStr += "  /* which global Cij index */" + self.endLine
    kStr += "  uint globalCRow = groupRow * MACRO_TILE_DIM0 + localRow;" + self.endLine
    kStr += "  uint globalCCol = groupCol * MACRO_TILE_DIM1 + localCol;" + self.endLine

    ####################################
    # write global Cij
    kStr += self.endLine
    kStr += "  /* write global Cij */" + self.endLine
    if self.dataTypeC == "c":
      kStr += "  float type_mad_tmp;" + self.endLine
    if self.dataTypeC == "z":
      kStr += "  double type_mad_tmp;" + self.endLine

    for a in range(0, self.microTileDim0):
      for b in range(0, self.microTileDim1):
        if self.isRow():
          kStr += "  if (globalCRow+%d*WG_DIM0 < M)" % a
        if self.isCol():
          kStr += "  if (globalCCol+%d*WG_DIM1 < N)" % b
        if self.isRow() or self.isCol():
          kStr += "{"
        kStr += "  TYPE_MAD_WRITE( C[ GET_GLOBAL_INDEX_C( globalCRow+%d*WG_DIM0, globalCCol+%d*WG_DIM1) ], alpha, rC[%d][%d], beta )" % (a, b, a, b)
        if self.isRow() or self.isCol():
          kStr += "}"
        kStr += self.endLine

    ####################################
    # end kernel
    kStr += self.endLine
    kStr += "}" + self.endLine

    return kStr


##############################################################################
# get source file string
##############################################################################
def getSourceFileString( kernel, backend):
  kernelName = kernel.getName()
  fileString = ""
  fileString += Common.getFileHeader()
  fileString += "#ifndef KERNEL_" + kernelName.upper() + "_INL\n"
  fileString += "#define KERNEL_" + kernelName.upper() + "_INL\n"
  fileString += "\n"
  fileString += "const unsigned int %s_workGroupDim0 = %u;\n" \
      % (kernelName, kernel.workGroupDim0 )
  fileString += "const unsigned int %s_workGroupDim1 = %u;\n" \
      % (kernelName(), kernel.workGroupDim1 )
  fileString += "const unsigned int %s_microTileDim0 = %u;\n" \
      % (kernelName(), kernel.microTileDim0 )
  fileString += "const unsigned int %s_microTileDim1 = %u;\n" \
      % (kernelName(), kernel.microTileDim1 )
  fileString += "const unsigned int %s_unroll = %u;\n" \
      % (kernelName(), kernel.unroll)
  fileString += "\n"
  fileString += "const char * const %s_src =\"" % (kernelName)
  fileString += getKernelString( kernel, backend)
  fileString += "\";\n"
  fileString += "\n"
  fileString += "#else\n"
  fileString += "#pragma message(\"%s was overriden by user kernel.\")\n" % kernelName()
  fileString += "#endif\n"
  return fileString


##############################################################################
# get header file string
##############################################################################
def getHeaderFileString( kernel, backend):
  kernelName = kernel.getName()
  fileString = ""
  fileString += Common.getFileHeader()
  fileString += "#ifndef KERNEL_" + kernelName.upper() + "_H\n"
  fileString += "#define KERNEL_" + kernelName.upper() + "_H\n"
  fileString += "\n"
  fileString += "extern const unsigned int %s_workGroupDim0;\n" % kernelName
  fileString += "extern const unsigned int %s_workGroupDim1;\n" % kernelName
  fileString += "extern const unsigned int %s_microTileDim0;\n" % kernelName
  fileString += "extern const unsigned int %s_microTileDim1;\n" % kernelName
  fileString += "extern const unsigned int %s_unroll;\n" % kernelName
  fileString += "extern const char * const %s_src;\n" % kernelName
  fileString += "#endif\n"

################################################################################
# Test GEMM
################################################################################
def testGEMM():
  print("Test GEMM")
  # kernel parameters
  operationType = Structs.OperationType(Structs.OperationType.contraction)
  numFreeIndices = 2
  numBatchIndices = 0
  numSummationIndices = 1
  indexAssignmentsA = [0, 2, 3]
  indexAssignmentsB = [1, 2, 3]
  operation = Structs.Operation( \
      operationType, \
      numFreeIndices, \
      numBatchIndices, \
      numSummationIndices, \
      indexAssignmentsA, \
      indexAssignmentsB )
  dataTypeA = Structs.DataType(Structs.DataType.single)
  dataTypeB = Structs.DataType(Structs.DataType.single)
  dataTypeC = Structs.DataType(Structs.DataType.single)
  alpha = False
  beta = False

  kernel = Kernel(\
      operation, \
      dataTypeA, \
      dataTypeB, \
      dataTypeC, \
      alpha, \
      beta )

  kernel.workGroupDim0 = 16
  kernel.workGroupDim1 = 16
  kernel.microTileDim0 = 6
  kernel.microTileDim1 = 6
  kernel.macroTileDim0 = 96
  kernel.macroTileDim1 = 96
  kernel.unroll        = 16

  print("Kernel Name: %s") % kernel.getName()
  backend = Structs.Backend(Structs.Backend.opencl)
  print("Kernel Body: %s") % kernel.getBody(backend)

def testAdvanced():
  pass

################################################################################
# Main
################################################################################
if __name__ == "__main__":
  testGEMM()

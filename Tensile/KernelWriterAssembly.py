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


from SolutionStructs import DataType
from Common import globalParameters
from KernelWriter import KernelWriter

################################################################################
# Assembly Kernel
################################################################################
class KernelWriterAssembly(KernelWriter):

  ##############################################################################
  # Init
  ##############################################################################
  def __init__( self, kernelMinNaming, kernelSerialNaming ):
    super(KernelWriterAssembly, self).__init__( \
        kernelMinNaming, kernelSerialNaming)

    self.versionMajor = self.language[3]
    self.versionMinor = self.language[4]
    self.versionPatch = self.language[5]
    print1("Generating assembly for gfx-%s:%s:%s\n" \
        % (self.versionMajor, self.versionMinor, self.versionPatch) )

    ########################################
    # available memory instructions in order of preference
    # name,
    # num addresses,
    # num offsets,
    # offset multiplier,
    # bit width
    self.memoryArchitecture = {
        ["LocalRead"]: [
          ["ds_read_b128",        1, 1,   4, 128],
          ["ds_read2st64_b64",    1, 2, 128,  64],
          ["ds_read2_b64",        1, 2,   2,  64],
          ["ds_read_b64",         1, 1,   2,  64],
          ["ds_read2st64_b32",    1, 2,  64,  32],
          ["ds_read2_b32",        1, 2,   1,  32],
          ["ds_read_b32",         1, 1,   1,  32] ],
        ["LocalWrite"]: [
          ["ds_write_b128",       1, 1,   4, 128],
          ["ds_write2st64_b64",   1, 2, 128,  64],
          ["ds_write2_b64",       1, 2,   2,  64],
          ["ds_write_b64",        1, 1,   2,  64],
          ["ds_write2st64_b32",   1, 2,  64,  32],
          ["ds_write2_b32",       1, 2,   1,  32],
          ["ds_write_b32",        1, 1,   1,  32] ],
        ["GlobalRead"]: [
          ["flat_load_dwordx4",   1, 1,   0, 128],
          ["flat_load_dwordx2",   1, 1,   0,  64],
          ["flat_load_dword",     1, 1,   0,  32] ],
        ["GlobalWrite"]: [
          ["flat_store_dwordx4",  1, 1,   0, 128],
          ["flat_store_dwordx2",  1, 1,   0,  64],
          ["flat_store_dword",    1, 1,   0,  32] ]
        }

    # Supported AMD Graphics Architectures
    # gfx701 - Hawaii
    # gfx801 - Carrizo
    # gfx802 - Tonga
    # gfx803 - Fiji
    """
    for b64, offset is how many b64's to skip, now how many b32's to skip
ds_read_b32: read single word, 16-bit offset = 65536
ds_read_b64: read 2 adjacent words, 16-bit offset = 65536*2
ds_read2_b32: read 2 words diff addr, 8-it offsets = 256
ds_read2_b64: read 2x2 words diff addr, 8-it offsets = 256*2 (128*unroll4)
ds_read2st64_b32: read 2 words diff addr, 8-it offsets = 256
ds_read2st64_b64: read 2x2 words diff addr, 8-it offsets = 256*2 (128*unroll4)
    """

    self.endLine = "\n"
    self.syncStr = "s_barrier"
    self.commentPrefix = "/*"
    self.commentSuffix = "*/"
    self.commentHR = "*"*40
    self.indent = ""


  ##############################################################################
  #
  #   Functions to Write Kernel Segments
  #
  ##############################################################################

  ##############################################################################
  # Open String - DONE
  ##############################################################################
  def openString(self, kernel):
    return ""

  ##############################################################################
  # Close String - DONE
  ##############################################################################
  def closeString(self, kernel):
    return ""

  ##############################################################################
  # Function Prefix
  ##############################################################################
  def functionPrefix(self, kernel):
    kStr = ""

    ####################################
    # register allocation
    # choosing instructions
    ####################################
    self.instructions = {}

    # registers per element
    self.rpe = kernel["ProblemType"]["DataType"].numRegisters()
    #self.bpe = self.rpe * 32
    #self.bpv = self.bpe * kernel["VectorWidth"] # this is how many bits
    # wide each read/write would be without read2
    #print1("BytesPerVector: %u"% self.pbv)

    # registers per global address
    self.rpga = 2 # 64-bit
    # registers per local address
    self.rpla = 1 # 32-bit

    ########################################
    # multiply-accumulate
    numRegC = kernel["ThreadTile0"]*kernel["ThreadTile1"] * self.rpe
    numRegValuA = kernel["ThreadTile0"] * self.rpe
    numRegValuB = kernel["ThreadTile0"] * self.rpe
    numRegValuBlkA = numRegValuA if kernel["PrefetchLocalRead"] else 0
    numRegValuBlkB = numRegValuB if kernel["PrefetchLocalRead"] else 0

    ########################################
    # local read a
    self.numLocalReadVectorsA = kernel["ThreadTile0"] / kernel["VectorWidth"]
    self.localReadVectorStrideA = kernel["MacroTile0"] / kernel["VectorWidth"]
    # within the same unroll

    # CombineAllowed = (nLRVA > 1 and LocalRead2=-1,1)
    combineA = self.numLocalReadVectorsA > 1 and kernel["LocalRead2"] != 0

"""
strategy:
  what is the offset in regs from one read address to the next same iter
  what is the max offset in regs we need (last reg address in last iter - first reg address in first iter)
for each read instruction
  what is the max offset in number (maxOffsetRegs/4 for b128)
  what is the max offset supported by instruction
  numAddresses = maxOffsetNumber / maxOffsetSupported
  if it succeeds
    read instruction name
    how many different addresses will we need
"""

    # attempt ds_read_b128
    if "ds_read_b128" in self.architecture and self.bpv >= 128:
      self.instructions["LocalReadA"] = "ds_read_b32"
      self.localReadBitsA = 128
      self.localRead2A = 16

  # if bpv >= 128 and instruction exists
  # 4+x4+ microtile of float4
  # 2+x2+ microtile of double2
  # 2+x2+ microtile of CS2
  # 1+x1+ microtile of CD2

# attempt ds_read2_b64
  # if (bpv > 64 or (bpv=64 and CombineAllowed)) and exists and not prior
  # 2+x2+ microtile of floats

# attempt ds_read_b64
  # if bpv >= 64 and exists and not prior
  # 1x1 microtile of doubles
  # 1x1 microtile of ComplexSingles
  # 2x2 microtile of float2's
  # 2+x2+ microtile of doubles but LR2=0
  # 2+x2+ microtile of ComplexSingles but LR2=0
  # 4+x4+ microtile of float2's but LR2=0

# attempt ds_read2_b32
  # if (bpv > 32 or ( bpv=32 and CombineAllowed)) and exists and not prior
  # 2+x2+ microtile of floats

# attept ds_read_b32
  # if exists and nor prior
  # 1x1 microtile of floats
  # 2+x2+ microtile of floats and LR2=0

# if none assigned, FAIL


    if kernel["LocalRead2"] and self.numLocalReadVectorsA > 1:
      # if read stride is multiple of 64
      if self.localReadVectorStrideA % 64 == 0:
        if (kernel["VectorWidth"] == 1 \
            and "ds_read2st64_b32" in self.architecture ):
          self.instructions["LocalReadA"] = "ds_read2st64_b32"
          self.numVectorsPerLocalReadA = 2
        if (kernel["VectorWidth"] == 2 \
            and "ds_read2st64_b64" in self.architecture ):
          self.instructions["LocalReadA"] = "ds_read2st64_b64"
          self.numVectorsPerLocalReadA = 2
        if (kernel["VectorWidth"] == 4 \
            and "ds_read2st64_b128" in self.architecture ):
          self.instructions["LocalReadA"] = "ds_read2st64_b128"
          self.numVectorsPerLocalReadA = 2
      else:
        # if read stride is not multiple of 64
        if (kernel["VectorWidth"] == 1 \
            and "ds_read2_b32" in self.architecture ):
          self.instructions["LocalReadA"] = "ds_read2_b32"
          self.numVectorsPerLocalReadA = 2
        if (kernel["VectorWidth"] == 2 \
            and "ds_read2_b64" in self.architecture ):
          self.instructions["LocalReadA"] = "ds_read2_b64"
          self.numVectorsPerLocalReadA = 2
        if (kernel["VectorWidth"] == 4 \
            and "ds_read2_b128" in self.architecture ):
          self.instructions["LocalReadA"] = "ds_read2_b128"
          self.numVectorsPerLocalReadA = 2
    if "LocalReadA" not in self.instructions:
      if (kernel["VectorWidth"] == 1 \
          and "ds_read_b32" in self.architecture ):
        self.instructions["LocalReadA"] = "ds_read_b32"
        self.numVectorsPerLocalReadA = 1
      if (kernel["VectorWidth"] == 2 \
          and "ds_read_b64" in self.architecture ):
        self.instructions["LocalReadA"] = "ds_read_b64"
        self.numVectorsPerLocalReadA = 1
      if kernel["VectorWidth"] == 4:
        if "ds_read_b128" in self.architecture:
          self.instructions["LocalReadA"] = "ds_read_b128"
          self.numVectorsPerLocalReadA = 1
        elif "ds_read2_b64" in self.architecture:
          self.instructions["LocalReadA"] = "ds_read2_b64"
          self.numVectorsPerLocalReadA = 1
    if "LocalReadA" not in self.instructions:
      printWarning("LocalReadA: no suitable instruction found")
    numLocalReadInstructionsA = numLocalReadVectorsA / numVectorsPerLocalReadA
    numRegLocalReadAddressesA = numLocalReadInstructionsA * self.rpla

    numLocalReadAddressesA = 1 # all offsets hardcoded
    numLocalReadAddressRegsA = numLocalReadAddressesA * self.rpla

LocalRead2A = -1, 0, 1
LocalReadIncrementA = False, True
# TODO: option: when offset bits aren't sufficient, do we use VALU to
# increment address or do we use extra registers to store addresses?
# (1) read1 and aways have sufficient offset bits
# (2) read2 and if insufficient offset bits then IncrementAndReset
# (3) read2 and if insufficient offset bits then AllocateAdditionalAddresses

    ########################################
    # local read b
    numLocalReadAddressesB = 1 # all offsets hardcoded
    numLocalReadAddressRegsB = numLocalReadAddressesB * self.rpla

    ########################################
    # TODO registers used for local read increments, resets
    # in case I don't have enough offset bits when read2
    # do I need any, or can it be completely hardcoded
    # hardcode the mod value, or in sgpr

    ########################################
    # local write a
# TODO: option: when offset bits aren't sufficient, do we use VALU to
# increment address or do we use extra registers to store addresses?
# (1) write1 and aways have sufficient offset bits
# (2) write2 and if insufficient offset bits then IncrementAndReset
# (3) write2 and if insufficient offset bits then AllocateAdditionalAddresses

    # CombineAllowed = (nLRVA > 1 and LocalRead2=-1,1)
    combineA = self.numLocalReadVectorsA > 1 and kernel["LocalRead2"] != 0

    # attempt ds_read_b128
    if "ds_read_b128" in self.architecture and self.bpv >= 128:
      self.instructions["LocalReadA"] = "ds_read_b32"
      self.localReadBits = 128

  # if bpv >= 128 and instruction exists
  # 4+x4+ microtile of float4
  # 2+x2+ microtile of double2
  # 2+x2+ microtile of CS2
  # 1+x1+ microtile of CD2

# attempt ds_read2_b64
  # if (bpv > 64 or (bpv=64 and CombineAllowed)) and exists and not prior
  # 2+x2+ microtile of floats

# attempt ds_read_b64
  # if bpv >= 64 and exists and not prior
  # 1x1 microtile of doubles
  # 1x1 microtile of ComplexSingles
  # 2x2 microtile of float2's
  # 2+x2+ microtile of doubles but LR2=0
  # 2+x2+ microtile of ComplexSingles but LR2=0
  # 4+x4+ microtile of float2's but LR2=0

# attempt ds_read2_b32
  # if (bpv > 32 or ( bpv=32 and CombineAllowed)) and exists and not prior
  # 2+x2+ microtile of floats

# attept ds_read_b32
  # if exists and nor prior
  # 1x1 microtile of floats
  # 2+x2+ microtile of floats and LR2=0

# if none assigned, FAIL










    numLocalWriteVectorsA = kernel["NumLoadsPerpendicularA"] \
        * kernel["NumLoadsCoalescedA"] * NumWriteVectorComponentsA * self.rpla
    if kernel["LocalWrite2"] < 0:
      # combine local writes in tile dimension
      self.localWriteVectorTileStrideA = -1
      pass
    elif kernel["LocalWrite2"] > 0:
      # combine local writes in unroll dimension
      self.localWriteVectorUnrollStrideA = -1
      pass
    else:
      # don't combine local writes
      if (kernel["VectorWidth"] == 1 \
          and "ds_write_b32" in self.architecture):
        self.instructions["LocalWriteA"] = "ds_write_b32"
      if (kernel["VectorWidth"] == 2 \
          and "ds_write_b64" in self.architecture):
        self.instructions["LocalWriteA"] = "ds_write_b64"
      numVectorsPerLocalWriteA = 1

    numLocalWriteInstructionsA = numLocalWriteVectorsB \
        / numVectorsPerLocalWriteA
    numRegLocalWriteAddressesA = numLocalWriteInstructionsA * self.rpla

    # registers used for global load increments

    # registers used for global load elements

    # registers used for global load addresses

    kStr += ".set vC 0%s" % self.endLine
    kStr += ".set vA %s%s" % (numReg, self.endLine)


    ####################################
    # kernel preprocessor definitions
    kStr += self.comment("tile parameters")
    kStr += ".set NUM_THREADS %3d%s" \
        % (kernel["NumThreads"], self.endLine )
    kStr += ".set SG%s %d%s" \
        % (self.tileChar0, kernel["SubGroup0"], self.endLine )
    kStr += ".set SG%s %d%s" \
        % (self.tileChar1, kernel["SubGroup1"], self.endLine )
    kStr += ".set TT%s %d%s" \
        % (self.tileChar0, kernel["ThreadTile0"], self.endLine )
    kStr += ".set TT%s %d%s" \
        % (self.tileChar1, kernel["ThreadTile1"], self.endLine )
    kStr += ".set MT%s (SG%s*TT%s)%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine )
    kStr += ".set MT%s (SG%s*TT%s)%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine )
    kStr += self.comment("DepthU parameters")
    kStr += ".set CPS (NUM_THREADS / MT%s * VECTOR_WIDTH)%s" \
        % (self.tileChar0, self.endLine)
    kStr += "#define SPLITU %d%s" \
        % (kernel["LocalSplitU"], self.endLine )
    kStr += "#define UNROLL %d%s" \
        % (kernel["LoopUnroll"], self.endLine )
    kStr += ".set DEPTHU (SPLITU*UNROLL)%s" % (self.endLine )
    kStr += self.comment("other")
    kStr += ".set PAD %u%s" % (kernel["LdsPad"], self.endLine)
    kStr += ".set WORK_GROUP_MAPPING %u%s" % (abs(kernel["WorkGroupMapping"]), self.endLine)
    kStr += ".set VECTOR_WIDTH %u%s" % (kernel["VectorWidth"], self.endLine)

    ####################################
    # num loads
    kStr += self.comment("num loads parallel and perpendicular to coalesced")
    kStr += ".set NLCA %d%s" % (kernel["NumLoadsCoalescedA"], self.endLine )
    kStr += ".set NLCB %d%s" % (kernel["NumLoadsCoalescedB"], \
        self.endLine )

    kStr += ".set NLPA %d%s" % (kernel["NumLoadsPerpendicularA"], \
        self.endLine )
    kStr += ".set NLPB %d%s" % (kernel["NumLoadsPerpendicularB"], \
        self.endLine )

    ####################################
    # load sizes
    kStr += self.comment("load sizes parallel and perpendicular to coalesced")
    if kernel["ProblemType"]["TLUA"]:
      kStr += ".set LSCA (MT%s/NLCA)%s" \
          % (self.tileCharA, self.endLine)
      kStr += ".set LSPA (DEPTHU/NLPA)" + self.endLine
    else:
      kStr += ".set LSCA (DEPTHU/NLCA)%s" \
          % (self.endLine)
      kStr += ".set LSPA (MT%s/NLPA)%s" \
          % ( self.tileCharA, self.endLine)
    if kernel["ProblemType"]["TLUB"]:
      kStr += ".set LSCB (MT%s/NLCB)%s" \
          % (self.tileCharB, self.endLine)
      kStr += ".set LSPB (DEPTHU/NLPB)" + self.endLine
    else:
      kStr += ".set LSCB (DEPTHU/NLCB)%s" \
          % (self.endLine)
      kStr += ".set LSPB (MT%s/NLPB)%s" % (self.tileCharB, self.endLine)
    kStr += ".set LVCA (LSCA/VECTOR_WIDTH)%s" % (self.endLine)
    kStr += ".set LVCB (LSCB/VECTOR_WIDTH)%s" % (self.endLine)
    kStr += ".set LVPA (LSPA/VECTOR_WIDTH)%s" % (self.endLine)
    kStr += ".set LVPB (LSPB/VECTOR_WIDTH)%s" % (self.endLine)

    # local buffer size
    kStr += ".set LDS_OFFSET_B %u%s" % (kernel["LdsOffsetB"], self.endLine)
    kStr += ".set LDS_NUM_ELEMENTS %u%s" % (kernel["LdsNumElements"], \
        self.endLine)

    # prefetch local buffer offsets
    # layout is redA, redB, blkA, blkB
    if kernel["PrefetchGlobalRead"]:
      kStr += ".set LDS_OFFSET_BLK %u%s" \
          % (kernel["LdsOffsetA_Blk"], self.endLine)

    ####################################
    # global memory indices
    kStr += self.comment("global memory indices")
    # C
    kStr += ".set GLOBAL_C(IDX%s" % self.indexChars[0]
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      kStr += ", IDX%s" % self.indexChars[i]
    indexChar = self.indexChars[0]
    kStr += ") (( (IDX%s)*strideC%s" % (indexChar, indexChar)
    for i in range(1, kernel["ProblemType"]["NumIndicesC"]):
      indexChar = self.indexChars[i]
      kStr += " + (IDX%s)*strideC%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine
    # A non-vector
    kStr += ".set GLOBAL_OFFSET_A(IDX%s" \
        % self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][0]]
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      kStr += ", IDX%s" \
          % self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
    indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][0]]
    kStr += ") (( (IDX%s)*strideA%s" % (indexChar, indexChar)
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsA"])):
      indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
      kStr += " + (IDX%s)*strideA%s" % (indexChar, indexChar)
    kStr += " ))%s" % self.endLine
    # B non-vector
    kStr += ".set GLOBAL_OFFSET_B(IDX%s" \
        % self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][0]]
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      kStr += ", IDX%s" \
          % self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]
    indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][0]]
    kStr += ") (( (IDX%s)*strideB%s" % (indexChar, indexChar)
    for i in range(1, len(kernel["ProblemType"]["IndexAssignmentsB"])):
      indexChar = self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]
      kStr += " + (IDX%s)*strideB%s" % (indexChar, indexChar)
    kStr += " ))" + self.endLine

    ####################################
    # data types
    kStr += self.comment("data types")
    kStr += ".set DATA_TYPE %s%s" \
        % (kernel["ProblemType"]["DataType"].toDevice(self.language), \
        self.endLine)
    vecStr = kernel["ProblemType"]["DataType"].toDevice(self.language)
    if kernel["VectorWidth"] > 1:
      vecStr += str(kernel["VectorWidth"])
    kStr += ".set VECTOR_TYPE %s%s" % (vecStr, self.endLine)

    if self.language == "OCL":
      kStr += ".set MAD(A,B,DST) mad(A,B,DST)"
    else:
      kStr += ".set MAD(A,B,DST) DST += A*B"
    kStr += self.endLine

    ####################################
    # MACs
    """
    kStr += self.comment("MAC's")
    if kernel["ProblemType"]["DataType"].isReal():
      # real data
      kStr += ".set TYPE_MAC(MULA,MULB,DST) " \
          + "DST = MAD(MULA,MULB,DST);" + self.endLine
      if kernel["ProblemType"]["UseBeta"]:
        # dst = alpha*reg + beta*dst
        kStr += ".set TYPE_MAC_WRITE(DST,ALPHA,REG,BETA) " \
            + "DST = (ALPHA)*(REG) + (BETA)*(DST);" + self.endLine
      else:
        # dst = alpha*reg
        kStr += ".set TYPE_MAC_WRITE(DST,ALPHA,REG) " \
            + "DST = (ALPHA)*(REG);" + self.endLine
    else:
      # complex data
      if not kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]:
        # neither conjugate
        kStr += (
          ".set TYPE_MAC(MULA,MULB,DST) " + self.endLine +
          "  DST.s0 = MAD(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLine +
          "  DST.s0 = MAD( -MULA.s1, MULB.s1, DST.s0 ); " + self.endLine +
          "  DST.s1 = MAD(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLine +
          "  DST.s1 = MAD(  MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]:
        # A conjugate (negate imaginary A.s1)
        kStr += (
          ".set TYPE_MAC(MULA,MULB,DST) " + self.endLine +
          "  DST.s0 = MAD(  MULA.s0, MULB.s0, DST.s0 ); " + self.endLine +
          "  DST.s0 = MAD(  MULA.s1, MULB.s1, DST.s0 ); " + self.endLine +
          "  DST.s1 = MAD(  MULA.s0, MULB.s1, DST.s1 ); " + self.endLine +
          "  DST.s1 = MAD( -MULA.s1, MULB.s0, DST.s1 );" + self.endLine )
      elif not kernel["ProblemType"]["ComplexConjugateA"] and kernel["ProblemType"]["ComplexConjugateB"]:
        # B conjugate (negate imaginary B.s1)
        kStr += (
          ".set TYPE_MAC(MULA,MULB,DST) " + self.endLine +
          "  DST.s0 = MAD(  MULA.s0,  MULB.s0, DST.s0 ); " + self.endLine +
          "  DST.s0 = MAD( -MULA.s1, -MULB.s1, DST.s0 ); " + self.endLine +
          "  DST.s1 = MAD(  MULA.s0, -MULB.s1, DST.s1 ); " + self.endLine +
          "  DST.s1 = MAD(  MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      else:
        # A & B conjugate (negate imaginary .s1)
        kStr += (
          ".set TYPE_MAC(MULA,MULB,DST) " + self.endLine +
          "  DST.s0 = MAD(  MULA.s0,  MULB.s0, DST.s0 ); " + self.endLine +
          "  DST.s0 = MAD(  MULA.s1, -MULB.s1, DST.s0 ); " + self.endLine +
          "  DST.s1 = MAD(  MULA.s0, -MULB.s1, DST.s1 ); " + self.endLine +
          "  DST.s1 = MAD( -MULA.s1,  MULB.s0, DST.s1 );" + self.endLine )
      if kernel["ProblemType"]["UseBeta"]:
        # dst = alpha*reg + beta*dst
        kStr += (
          ".set TYPE_MAC_WRITE( DST, ALPHA, REG, BETA ) "+self.endLine +
          "  /* (1) */ " + self.endLine +
          "  type_mac_tmp = REG.s0; " + self.endLine +
          "  REG.s0 *= ALPHA.s0; " + self.endLine +
          "  REG.s0 = MAD( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLine +
          "  REG.s1 *= ALPHA.s0; " + self.endLine +
          "  REG.s1 = MAD(  ALPHA.s1, type_mac_tmp, REG.s1 ); "+self.endLine+
          "  /* (2) */ " + self.endLine +
          "  REG.s0 = MAD(  BETA.s0, DST.s0, REG.s0 ); " + self.endLine +
          "  REG.s0 = MAD( -BETA.s1, DST.s1, REG.s0 ); " + self.endLine +
          "  REG.s1 = MAD(  BETA.s1, DST.s0, REG.s1 ); " + self.endLine +
          "  REG.s1 = MAD(  BETA.s0, DST.s1, REG.s1 ); " + self.endLine +
          "  /* (3) */ " + self.endLine +
          "  DST = REG;" + self.endLine )
      else:
        # dst = alpha*reg
        kStr += (
          ".set TYPE_MAC_WRITE( DST, ALPHA, REG ) "+self.endLine+
          "  /* (1) */ " + self.endLine +
          "  type_mac_tmp = REG.s0; " + self.endLine +
          "  REG.s0 *= ALPHA.s0; " + self.endLine +
          "  REG.s0 = MAD( -ALPHA.s1, REG.s1, REG.s0 ); " + self.endLine +
          "  REG.s1 *= ALPHA.s0; " + self.endLine +
          "  REG.s1 = MAD(  ALPHA.s1, type_mac_tmp, REG.s1 ); "+self.endLine+
          "  /* (3) */ " + self.endLine +
          "  DST = REG;" + self.endLine )
    """

    ####################################
    # sumation unroll
    kStr += self.comment("%dx%d micro-tile" \
        % (kernel["ThreadTile0"], kernel["ThreadTile1"]) )
    numMacs = 2 if kernel["PrefetchLocalRead"] else 1

    for m in range(0, numMacs):
      kStr += ".set MAC_%ux%u" \
          % (kernel["ThreadTile0"], kernel["ThreadTile1"])
      if kernel["PrefetchLocalRead"]:
        kStr += ("" if m==0 else "_BLK")
      kStr += self.endLine

      """
    if False:
      if kernel["VectorWidth"] == 1:
        kStr += "  printf(\\\"MAC: T[%%02u]: %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f; %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f, %%.0f\\\\n\\\", serial, rA[0], rA[1], rA[2], rA[3], rA[4], rA[5], rA[6], rA[7], rB[0], rB[1], rB[2], rB[3], rB[4], rB[5], rB[6], rB[7]); %s" % (self.endLine)
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
            self.endLine)
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
            self.endLine)
      """

      """
      for b in range(0, kernel["ThreadTile1"]):
        for a in range(0, kernel["ThreadTile0"]):
          # a
          vecA = a / kernel["VectorWidth"]
          elemA = a % kernel["VectorWidth"]
          strA = "rA[%d%s]" % (vecA, ("+TT%s/VECTOR_WIDTH"%self.tileCharA) \
              if m>0 else "")
          if kernel["VectorWidth"] > 1:
            strA += ".%s" % self.vectorComponents[elemA]
          # b
          vecB = b / kernel["VectorWidth"]
          elemB = b % kernel["VectorWidth"]
          strB = "rB[%d%s]" % (vecB, ("+TT%s/VECTOR_WIDTH"%self.tileCharB) \
              if m>0 else "")
          if kernel["VectorWidth"] > 1:
            strB += ".%s" % self.vectorComponents[elemB]
          # c
          strC = "rC[%d+%d*TT%s/VECTOR_WIDTH]" % (vecA, b, self.tileChar0 )
          elemC = elemA
          if kernel["VectorWidth"] > 1:
            strC += ".%s" % self.vectorComponents[elemC]
          """
          kStr += "  printf(\\\"T[%%u,%u,%u]: %s:%%.0f += %s:%%.0f * %s:%%.0f\\\\n\\\", serial, %s, %s, %s); %s" % (a, b, strC, strA, strB, strC, strA, strB, self.endLinePP)
          """
          kStr += "  TYPE_MAC(%s,%s,%s); %s" % (strA, strB, strC, \
              self.endLine)
      kStr += "  " + self.fenceStr + self.endLine
    kStr += self.endLine
      """

    ####################################
    # preprocessor definitions of kernel arguments
    firstStride = 0
    if kernel["ProblemType"]["UseInitialStrides"]:
      # no strides .setd
      lastStrideC = 0
      lastStrideA = 0
      lastStrideB = 0
    else:
      # .set initial stride
      kStr += self.comment("hard-coded initial strides")
      lastStrideC = 1
      lastStrideA = 1
      lastStrideB = 1

    for i in range(firstStride, lastStrideC):
      kStr += ".set strideC" + self.indexChars[i] + " 1" + self.endLine
    for i in range(firstStride, lastStrideA):
      kStr += ".set strideA" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]] \
          + " 1" + self.endLine
    for i in range(firstStride, lastStrideB):
      kStr += ".set strideB" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]] \
          + " 1" + self.endLine
    kStr += self.endLine
    return kStr


  ##############################################################################
  # Function Signature Prefix
  ##############################################################################
  def functionSignaturePrefix(self, kernel):
    return ""
    s = ""
    if self.language == "HIP":
      s += "#pragma clang diagnostic push" + self.endLine
      s += "#pragma clang diagnostic ignored \"-Wunused-parameter\"" + self.endLine
    return s


  ##############################################################################
  # Function Signature
  ##############################################################################
  def functionSignature(self, kernel ):
    return ""
    kernelName = self.getKernelName(kernel)

    # determine chars for fast access
    self.indexChars = []
    for i in range(0, len(globalParameters["IndexChars"])):
      self.indexChars.append(globalParameters["IndexChars"][i])
    self.indexChars[kernel["ProblemType"]["Index0"]] \
        = "0" + self.indexChars[kernel["ProblemType"]["Index0"]]
    self.indexChars[kernel["ProblemType"]["Index1"]] \
        = "1" + self.indexChars[kernel["ProblemType"]["Index1"]]
    self.tileChar0 = self.indexChars[kernel["ProblemType"]["Index0"]]
    self.tileChar1 = self.indexChars[kernel["ProblemType"]["Index1"]]

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
      s += "," + self.endLine + "  unsigned int const strideC" + self.indexChars[i]
    for i in range(firstStride, lastStrideA):
      s += "," + self.endLine + "  unsigned int const strideA" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]]
    for i in range(firstStride, lastStrideB):
      s += "," + self.endLine + "  unsigned int const strideB" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]]

    # sizes
    for i in range(0, kernel["ProblemType"]["TotalIndices"]):
      s += "," + self.endLine + "  unsigned int const size" + self.indexChars[i]
    s += " )"
    return s

  ##############################################################################
  # Function Signature Suffix
  ##############################################################################
  def functionSignatureSuffix(self, kernel):
    return ""
    s = ""
    if self.language == "HIP":
      s += "#pragma clang diagnostic pop" + self.endLine
    return s

  ##############################################################################
  # Function Begin
  ##############################################################################
  def functionBegin(self, kernel):
    return ""
    s = ""
    s += " {" + self.endLine
    return s

  ##############################################################################
  # Allocate Resources
  ##############################################################################
  def allocateResources(self, kernel):
    return ""
    kStr = ""
    kStr += "  VECTOR_TYPE rC[TT%s*TT%s/VECTOR_WIDTH] = {0};%s" \
        % (self.tileChar0, self.tileChar1, self.endLine )
    kStr += "  VECTOR_TYPE rA[TT%s/VECTOR_WIDTH%s];%s" \
        % (self.tileChar0, ("*2" if kernel["PrefetchLocalRead"] else ""), \
        self.endLine)
    kStr += "  VECTOR_TYPE rB[TT%s/VECTOR_WIDTH%s];%s" \
        % (self.tileChar1, ("*2" if kernel["PrefetchLocalRead"] else ""), \
        self.endLine)

    ####################################
    # registers for global -> local load
    kStr += "  VECTOR_TYPE "
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        kStr += "a_" + str(para) + "_" + str(perp)
        if para == kernel["NumLoadsCoalescedA"]-1 \
            and perp == kernel["NumLoadsPerpendicularA"]-1:
          kStr += ";" + self.endLine
        else:
          kStr += ", "
    kStr += "  VECTOR_TYPE "
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        kStr += "b_" + str(para) + "_" + str(perp)
        if para == kernel["NumLoadsCoalescedB"]-1 \
            and perp == kernel["NumLoadsPerpendicularB"]-1:
          kStr += ";" + self.endLine
        else:
          kStr += ", "

    ####################################
    # allocate local memory
    kStr += "  %sDATA_TYPE localMemory[LDS_NUM_ELEMENTS];%s" \
        % (self.sharedDeclStr, self.endLine )
    return kStr

  ##############################################################################
  # Global Read Addresses: Work-Group
  ##############################################################################
  def graWorkGroup(self, kernel):
    return ""
    kStr = ""
    if kernel["WorkGroupMapping"] == 1:
      kStr += "  unsigned int wg" + self.tileChar0 + " = " \
          + self.getGroupIdStr + "(0);" + self.endLine
      kStr += "  unsigned int wg" + self.tileChar1 + " = " \
          + self.getGroupIdStr + "(1);" + self.endLine
    else:
      dimCoal = (0 if kernel["WorkGroupMapping"] > 0 else 1)
      dimPerp = (1 if kernel["WorkGroupMapping"] > 0 else 0)

      # work-group free indices
      kStr += self.endLine
      kStr += "  unsigned int wg%s, wg%s;%s" % (self.tileChar0, self.tileChar1, self.endLine)
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
          % ((self.tileChar0 if kernel["WorkGroupMapping"] > 0 else self.tileChar1), \
          self.getNumGroupsStr, dimCoal, self.endLine)
      kStr += "    wg%s = superGroup*WORK_GROUP_MAPPING + groupSerial %% WORK_GROUP_MAPPING;%s" \
          % ((self.tileChar1 if kernel["WorkGroupMapping"] > 0 else self.tileChar0), \
          self.endLine)

      # if in last super group
      kStr += "  } else {%s" % self.endLine
      kStr += "    wg%s = (groupSerial-numWorkGroupsBeforeLastSuperGroup)/lastSuperGroupWidth;%s" \
          % ((self.tileChar0 if kernel["WorkGroupMapping"] > 0 else self.tileChar1), \
          self.endLine)
      kStr += "    wg%s = superGroup*WORK_GROUP_MAPPING + groupSerial %% lastSuperGroupWidth;%s" \
          % ((self.tileChar1 if kernel["WorkGroupMapping"] > 0 else self.tileChar0), \
          self.endLine)

      # if in last super group
      kStr += "  }%s" % self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Subgroup
  ##############################################################################
  def graSubgroup(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int serial = %s(0);%s" \
        % (self.getLocalIdStr, self.endLine)
    kStr += "  unsigned int sgId = serial / (SG%s*SG%s);%s" \
        % (self.tileChar0, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Assignment A
  ##############################################################################
  def graTileAssignmentA(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int globalReadOffsetA%s = (serial%s" \
        % (self.tileCharA, ("%" if self.globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "/") )
    if self.globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorA"] == kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += " + (wg%s*MT%s);%s" \
        % (self.tileCharA, self.tileCharA, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Assignment B
  ##############################################################################
  def graTileAssignmentB(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int globalReadOffsetB%s = (serial%s" \
        % (self.tileCharB, ("%" if self.globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "/") )
    if self.globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] == kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += " + (wg%s*MT%s);%s" \
        % (self.tileCharB, self.tileCharB, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Assignment A
  ##############################################################################
  def graUnrollAssignmentA(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int globalReadOffsetA%s = (serial%s" \
        % (self.unrollChar, ("/" if self.globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "%") )
    if self.globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorA"] != kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Assignment B
  ##############################################################################
  def graUnrollAssignmentB(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int globalReadOffsetB%s = (serial%s" \
        % (self.unrollChar, ("/" if self.globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "%") )
    if self.globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] != kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Other Free Assignments
  ##############################################################################
  def graOtherFreeAssignments(self, kernel):
    return ""
    kStr = ""
    nonTileFreeIndices = range(0, kernel["ProblemType"]["NumIndicesC"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index0"])
    nonTileFreeIndices.remove(kernel["ProblemType"]["Index1"])
    for i in range(0, len(nonTileFreeIndices)):
      index = nonTileFreeIndices[i]
      kStr += "  unsigned int wg" + self.indexChars[index] \
          + " = ( " + self.getGroupIdStr + "(2)"
      for j in reversed( range( i+1, len(nonTileFreeIndices)) ):
        index2 = nonTileFreeIndices[j]
        kStr += " / size" + self.indexChars[index2]
      kStr += " ) % size" + self.indexChars[index] + ";" + self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments
  ##############################################################################
  def graOtherSummationAssignments(self, kernel):
    return ""
    kStr = ""
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
      index = i
      kStr += ".set globalReadOffsetA%s 0%s" \
          % (self.indexChars[index], self.endLine)
      kStr += ".set globalReadOffsetB%s 0%s" \
          % (self.indexChars[index], self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets A
  ##############################################################################
  def graTileOffsetsA(self, kernel):
    return ""
    kStr = ""
    for l in range(0, self.numReadsTileA):
      if self.readTileDimComponentsA:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetA%s_%u_s%u = globalReadOffsetA%s + %u + %d*%s;%s" \
              % (self.tileCharA, l, s, self.tileCharA, s, l, \
              ("LSCA" if kernel["ProblemType"]["TLUA"] else "LSPA"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetA%s_%u = globalReadOffsetA%s + %d*%s;%s" \
            % (self.tileCharA, l, self.tileCharA, l, \
            ("LSCA" if kernel["ProblemType"]["TLUA"] else "LSPA"), \
            self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets B
  ##############################################################################
  def graTileOffsetsB(self, kernel):
    return ""
    kStr = ""
    for l in range(0, self.numReadsTileB):
      if self.readTileDimComponentsB:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetB%s_%u_s%u = globalReadOffsetB%s + %u + %d*%s;%s" \
              % (self.tileCharB, l, s, self.tileCharB, s, l, \
              ("LSCB" if kernel["ProblemType"]["TLUB"] else "LSPB"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetB%s_%u = globalReadOffsetB%s + %d*%s;%s" \
            % (self.tileCharB, l, self.tileCharB, l, \
            ("LSCB" if kernel["ProblemType"]["TLUB"] else "LSPB"), \
            self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A
  ##############################################################################
  def graUnrollOffsetsA(self, kernel):
    return ""
    kStr = ""
    for l in range(0, self.numReadsUnrollA):
      if self.readUnrollDimComponentsA:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetA%s_%u_s%u = globalReadOffsetA%s + %u + %d*%s;%s" \
              % (self.unrollChar, l, s, self.unrollChar, s, l, \
              ("LSPA" if kernel["ProblemType"]["TLUA"] else "LSCA"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetA%s_%u = globalReadOffsetA%s + %d*%s;%s" \
            % (self.unrollChar, l, self.unrollChar, l, \
            ("LSPA" if kernel["ProblemType"]["TLUA"] else "LSCA"), \
            self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Offsets B
  ##############################################################################
  def graUnrollOffsetsB(self, kernel):
    return ""
    kStr = ""
    for l in range(0, self.numReadsUnrollB):
      if self.readUnrollDimComponentsB:
        for s in range(0, kernel["VectorWidth"]):
          kStr += "  unsigned int globalReadOffsetB%s_%u_s%u = globalReadOffsetB%s + %u + %d*%s;%s" \
              % (self.unrollChar, l, s, self.unrollChar, s, l, \
              ("LSPB" if kernel["ProblemType"]["TLUB"] else "LSCB"), \
              self.endLine)
      else:
        kStr += "  unsigned int globalReadOffsetB%s_%u = globalReadOffsetB%s + %d*%s;%s" \
            % (self.unrollChar, l, self.unrollChar, l, \
            ("LSPB" if kernel["ProblemType"]["TLUB"] else "LSCB"), \
            self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Branch A
  ##############################################################################
  def graBranchA(self, kernel):
    return ""
    kStr = ""
    for l in range(0, self.numReadsTileA):
      gro = "(globalReadOffsetA%s_%u%s)" % (self.tileCharA, l, \
          ("_s0 + (VECTOR_WIDTH-1)" if self.readTileDimComponentsA else "") )
      limit = "size%s" % (self.tileCharA)
      kStr += "  bool inBoundsA_%u = %s < %s;%s" \
          % (l, gro, \
          limit, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Branch B
  ##############################################################################
  def graBranchB(self, kernel):
    return ""
    kStr = ""
    for l in range(0, self.numReadsTileB):
        gro = "(globalReadOffsetB%s_%u%s)" % (self.tileCharB, l, \
            ("_s0 + (VECTOR_WIDTH-1)" if self.readTileDimComponentsB else ""))
        limit = "size%s" % self.tileCharB
        kStr += "  bool inBoundsB_%u = %s < %s;%s" \
            % (l, gro, \
            limit, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Shift A
  ##############################################################################
  def graShiftA(self, kernel):
    return ""
    kStr = ""
    for l in range(0, self.numReadsTileA):
      gro = "globalReadOffsetA%s_%u%s" % (self.tileCharA, l, \
          ("_s0" if self.readTileDimComponentsA else "") )
      limit = "(size%s-%s)" % (self.tileCharA, \
          ("VECTOR_WIDTH" if self.readTileDimVectorA else "1") )
      kStr += "  %s = (%s > %s) ? %s : %s;%s" \
          % (gro, gro, limit, limit, gro, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Shift B
  ##############################################################################
  def graShiftB(self, kernel):
    return ""
    kStr = ""
    for l in range(0, self.numReadsTileB):
      gro = "globalReadOffsetB%s_%u%s" % (self.tileCharB, l, \
          ("_s0" if self.readTileDimComponentsB else ""))
      limit = "(size%s-%s)" % (self.tileCharB, \
          ("VECTOR_WIDTH" if self.readTileDimVectorB else "1") )
      kStr += "  %s = (%s > %s) ? %s : %s;%s" \
          % (gro, gro, limit, limit, gro, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets A
  ##############################################################################
  def graFinalOffsetsA(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numReadVectorComponentsA):
          kStr += "  %s globalReadOffsetA_%u_%u%s = GLOBAL_OFFSET_A( " \
              % (self.uint64Str, para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsA \
              or self.readUnrollDimComponentsA) else ""))
          for i in range(0, len(kernel["ProblemType"]["IndexAssignmentsA"])):
            index = kernel["ProblemType"]["IndexAssignmentsA"][i]
            if index < kernel["ProblemType"]["NumIndicesC"]:
              if index == kernel["ProblemType"]["TileA"]:
                kStr += "globalReadOffsetA%s_%u%s" \
                    % (self.tileCharA, \
                    (para if kernel["ProblemType"]["TLUA"] else perp), \
                    (("_s%u"%s) if self.readTileDimComponentsA else "") )
              else: # just a group index
                kStr += "wg" + self.indexChars[index]
            else: # summation index
              if index == kernel["ProblemType"]["IndexUnroll"]:
                kStr += "globalReadOffsetA%s_%u%s" \
                    % (self.unrollChar, \
                    (perp if kernel["ProblemType"]["TLUA"] else para), \
                    (("_s%u"%s) if self.readUnrollDimComponentsA else "") )
              else:
                kStr += "globalReadOffsetA%s" % self.indexChars[index]
            if i < len(kernel["ProblemType"]["IndexAssignmentsA"])-1:
              kStr += ", "
          kStr += " );%s" % self.endLine
          """
          kStr += "  printf(\\\"GRA T[%%02u] gROA_%u_%u%s = %%4u\\\\n\\\", serial, globalReadOffsetA_%u_%u%s);%s" \
              % (para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsA \
              or self.readUnrollDimComponentsA) else ""), \
              para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsA \
              or self.readUnrollDimComponentsA) else ""), \
              self.endLine )
          """
    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets B
  ##############################################################################
  def graFinalOffsetsB(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numReadVectorComponentsB):
          kStr += "  %s globalReadOffsetB_%u_%u%s = GLOBAL_OFFSET_B( " \
              % (self.uint64Str, para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsB \
              or self.readUnrollDimComponentsB) else ""))
          for i in range(0, len(kernel["ProblemType"]["IndexAssignmentsB"])):
            index = kernel["ProblemType"]["IndexAssignmentsB"][i]
            if index < kernel["ProblemType"]["NumIndicesC"]:
              if index == kernel["ProblemType"]["TileB"]:
                kStr += "globalReadOffsetB%s_%u%s" \
                    % (self.tileCharB, \
                    (para if kernel["ProblemType"]["TLUB"] else perp), \
                    (("_s%u"%s) if self.readTileDimComponentsB else "") )
              else: # just a group index
                kStr += "wg" + self.indexChars[index]
            else: # summation index
              if index == kernel["ProblemType"]["IndexUnroll"]:
                kStr += "globalReadOffsetB%s_%u%s" \
                    % (self.unrollChar, \
                    (perp if kernel["ProblemType"]["TLUB"] else para), \
                    (("_s%u"%s) if self.readUnrollDimComponentsB else "") )
              else:
                kStr += "globalReadOffsetB%s" % self.indexChars[index]
            if i < len(kernel["ProblemType"]["IndexAssignmentsB"])-1:
              kStr += ", "
          kStr += " );%s" % self.endLine
          """
          kStr += "  printf(\\\"GRB T[%%02u] gROB_%u_%u%s = %%4u\\\\n\\\", serial, globalReadOffsetB_%u_%u%s);%s" \
              % (para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsB \
              or self.readUnrollDimComponentsB) else ""), \
              para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsB \
              or self.readUnrollDimComponentsB) else ""), \
              self.endLine )
          """
    return kStr

  ##############################################################################
  # Global Read Addresses: Apply User Offsets
  ##############################################################################
  def graApplyUserOffsets(self, kernel):
    return ""
    kStr = ""
    kStr += "  C += offsetC;%s" % self.endLine
    kStr += "  A += offsetA;%s" % self.endLine
    kStr += "  B += offsetB;%s" % self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses A
  ##############################################################################
  def graAddressesA(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        if self.readTileDimComponentsA or self.readUnrollDimComponentsA:
          for s in range(0, self.numReadVectorComponentsA):
            kStr += "  %sDATA_TYPE const *globalReadA_%u_%u%s = A + globalReadOffsetA_%u_%u%s;%s" \
                % (self.globalPtrStr, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else ""), \
                para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else ""), \
                self.endLine)
        else:
            kStr += "  %sVECTOR_TYPE const *globalReadA_%u_%u = (%sVECTOR_TYPE const *)(A + globalReadOffsetA_%u_%u);%s" \
                % (self.globalPtrStr, para, perp, self.globalPtrStr, \
                para, perp, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses B
  ##############################################################################
  def graAddressesB(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        if self.readTileDimComponentsB or self.readUnrollDimComponentsB:
          for s in range(0, self.numReadVectorComponentsB):
            kStr += "  %sDATA_TYPE const *globalReadB_%u_%u%s = B + globalReadOffsetB_%u_%u%s;%s" \
                % (self.globalPtrStr, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else ""), \
                para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else ""), self.endLine)
        else:
            kStr += "  %sVECTOR_TYPE const *globalReadB_%u_%u = (%sVECTOR_TYPE const *)(B + globalReadOffsetB_%u_%u);%s" \
                % (self.globalPtrStr, para, perp, self.globalPtrStr, \
                para, perp, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Increments A
  ##############################################################################
  def graIncrementsA(self, kernel, loopIdx):
    return ""
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    kStr += "%s%s globalReadIncA%s = (%s)strideA%s" \
        % (self.indent, self.int64Str, loopChar, \
        self.int64Str, loopChar)
    if loopIdx==kernel["ProblemType"]["NumIndicesSummation"]-1:
      kStr += "*DEPTHU"
    else:
      for j in range(loopIdx+1, \
          min(loopIdx+2,kernel["ProblemType"]["NumIndicesSummation"]) ):
        tmpChar = self.indexChars[ \
            kernel["ProblemType"]["IndicesSummation"][j]]
        kStr += " - strideA%s*size%s" % (tmpChar, tmpChar)
    kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # Global Read Addresses: Increments B
  ##############################################################################
  def graIncrementsB(self, kernel, loopIdx):
    return ""
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    kStr += "%s%s globalReadIncB%s = (%s)strideB%s" \
        % (self.indent, self.int64Str, loopChar, \
        self.int64Str, loopChar)
    if loopIdx==kernel["ProblemType"]["NumIndicesSummation"]-1:
      kStr += "*DEPTHU"
    else:
      for j in range(loopIdx+1, \
          min(loopIdx+2,kernel["ProblemType"]["NumIndicesSummation"]) ):
        tmpChar = self.indexChars[ \
            kernel["ProblemType"]["IndicesSummation"][j]]
        kStr += " - strideB%s*size%s" % (tmpChar, tmpChar)
    kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment A
  ##############################################################################
  def lwaTileAssignmentA(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int lwA%s = (serial%s" \
        % (self.tileCharA, ("%" if self.globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "/") )
    if self.globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")";
    if kernel["GlobalReadCoalesceVectorA"] == kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment B
  ##############################################################################
  def lwaTileAssignmentB(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int lwB%s = (serial%s" \
        % (self.tileCharB, ("%" if self.globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "/") )
    if self.globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] == kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A
  ##############################################################################
  def lwaUnrollAssignmentA(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int lwA%s = (serial%s" \
        % (self.unrollChar, ("/" if self.globalReadCoalesceGroupA \
        == kernel["ProblemType"]["TLUA"] else "%") )
    if self.globalReadCoalesceGroupA:
      kStr += ("LVCA" if kernel["GlobalReadCoalesceVectorA"] else "LSCA")
    else:
      kStr += ("LSPA" if kernel["GlobalReadCoalesceVectorA"] else "LVPA")
    kStr += ")";
    if kernel["GlobalReadCoalesceVectorA"] != kernel["ProblemType"]["TLUA"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: Unroll Assignment B
  ##############################################################################
  def lwaUnrollAssignmentB(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int lwB%s = (serial%s" \
        % (self.unrollChar, ("/" if self.globalReadCoalesceGroupB \
        == kernel["ProblemType"]["TLUB"] else "%") )
    if self.globalReadCoalesceGroupB:
      kStr += ("LVCB" if kernel["GlobalReadCoalesceVectorB"] else "LSCB")
    else:
      kStr += ("LSPB" if kernel["GlobalReadCoalesceVectorB"] else "LVPB")
    kStr += ")"
    if kernel["GlobalReadCoalesceVectorB"] != kernel["ProblemType"]["TLUB"]:
      kStr += "*VECTOR_WIDTH"
    kStr += ";%s" % self.endLine
    return kStr

  ##############################################################################
  # Local Write Addresses: First Offset A
  ##############################################################################
  def lwaFirstOffsetA(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int localWriteFirstOffsetA = lwA%s + lwA%s*(MT%s+PAD);%s" \
        % (self.tileCharA, self.unrollChar, self.tileCharA, self.endLine)
    return kStr

  ##############################################################################
  # Local Write Addresses: First Offset B
  ##############################################################################
  def lwaFirstOffsetB(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int localWriteFirstOffsetB = lwB%s + lwB%s*(MT%s+PAD) + LDS_OFFSET_B;%s" \
        % (self.tileCharB, self.unrollChar, self.tileCharB, self.endLine)
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A
  ##############################################################################
  def lwaFinalOffsetsA(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "  unsigned int localWriteOffsetA_%u_%u%s = localWriteFirstOffsetA + (%s%d*%s)" \
              % (para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              (("%u + "%s) if self.writeTileDimComponentsA else ""), \
              para, ("LSCA" if not kernel["ProblemType"]["TLUA"] else "LSCA") )
          if not kernel["ProblemType"]["TLUA"]:
            kStr += "*(MT%s+PAD)" % (self.tileCharA)
          kStr += " + (%s%d*%s)" % (
              (("%u + "%s) if self.writeUnrollDimComponentsA else ""), perp, \
              ("LSPA" if kernel["ProblemType"]["TLUA"] else "LSPA") )
          if kernel["ProblemType"]["TLUA"]:
            kStr += "*(MT%s+PAD)" % (self.tileCharA)
          kStr += ";%s" % self.endLine
          """
          kStr += "  printf(\\\"LWA T[%%02u] lWOA_%u_%u%s = %%4u\\\\n\\\", serial, localWriteOffsetA_%u_%u%s);%s" \
              % (para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              self.endLine )
          """
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets B
  ##############################################################################
  def lwaFinalOffsetsB(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "  unsigned int localWriteOffsetB_%u_%u%s = localWriteFirstOffsetB + (%s%d*%s)" \
              % (para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              (("%u + "%s) if self.writeTileDimComponentsB else ""), para, \
              ("LSCB" if not kernel["ProblemType"]["TLUB"] else "LSCB") )
          if not kernel["ProblemType"]["TLUB"]:
            kStr += "*(MT%s+PAD)" % (self.tileCharB)
          kStr += " + (%s%d*%s)" % ( \
              (("%u + "%s) if self.writeUnrollDimComponentsB else ""), perp, \
              ("LSPB" if not kernel["ProblemType"]["TLUB"] else "LSPB") )
          if kernel["ProblemType"]["TLUB"]:
            kStr += "*(MT%s+PAD)" % (self.tileCharB)
          kStr += ";%s" % self.endLine
          """
          kStr += "  printf(\\\"LWB T[%%02u] lWOB_%u_%u%s = %%4u\\\\n\\\", serial, localWriteOffsetB_%u_%u%s);%s" \
             % (para, perp,
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              self.endLine )
          """
    return kStr

  ##############################################################################
  # Local Write Addresses: Declare Addresses A
  ##############################################################################
  def lwaDeclareAddressesA(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "  %s%s *localWriteA_%u_%u%s;%s"\
              % (self.sharedPtrStr, \
              ("DATA_TYPE" if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), self.endLine )
    return kStr

  ##############################################################################
  # Local Write Addresses: Declare Addresses B
  ##############################################################################
  def lwaDeclareAddressesB(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "  %s%s *localWriteB_%u_%u%s;%s"\
              % (self.sharedPtrStr, ("DATA_TYPE" \
              if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), self.endLine )
    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment A
  ##############################################################################
  def lraTileAssignmentA(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int lr%s = (serial %% SG%s);%s" \
        % (self.tileChar0, self.tileChar0, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment B
  ##############################################################################
  def lraTileAssignmentB(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int lr%s = (serial / SG%s) %% SG%s;%s" \
        % (self.tileChar1, self.tileChar0, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset A
  ##############################################################################
  def lraFinalOffsetA(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int localReadOffsetA = lr%s*VECTOR_WIDTH + sgId*(MT%s+PAD);%s" \
        % ( self.tileChar0, self.tileChar0, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset B
  ##############################################################################
  def lraFinalOffsetB(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int localReadOffsetB = lr%s*VECTOR_WIDTH + sgId*(MT%s+PAD) + LDS_OFFSET_B;%s" \
        % (self.tileChar1, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Declare Addresses A
  ##############################################################################
  def lraDeclareAddressesA(self, kernel):
    return ""
    kStr = ""
    kStr += "  %sVECTOR_TYPE *localReadA;%s" % (self.sharedPtrStr, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Declare Addresses B
  ##############################################################################
  def lraDeclareAddressesB(self, kernel):
    return ""
    kStr = ""
    kStr += "  %sVECTOR_TYPE *localReadB;%s" % (self.sharedPtrStr, self.endLine)
    return kStr

  ##############################################################################
  # Declare Loop Num Iterations
  ##############################################################################
  def declareLoopNumIterators(self, kernel):
    kStr = ""
    for loopIdx in kernel["ProblemType"]["IndicesSummation"]:
      loopChar = self.indexChars[loopIdx]
      kStr += "%sunsigned int sumIter%s;%s" \
          % (self.indent, loopChar, self.endLine)
    return kStr

  ##############################################################################
  # Calculate Loop Num Iter
  ##############################################################################
  def calculateLoopNumIter(self, kernel, loopIdx):
    return ""

  ##############################################################################
  # Open Loop
  ##############################################################################
  def openLoop(self, kernel, loopIdx):
    return ""
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx

    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if tailLoop:
      kStr += "%ssumIter%s = (((size%s %% DEPTHU) + SPLITU - 1) / SPLITU);%s" \
          % (self.indent, self.unrollChar, self.unrollChar, self.endLine)
    else:
      kStr += "%ssumIter%s = size%s%s;%s" \
          % (self.indent, loopChar, loopChar, \
          (" / DEPTHU" if loopIdx == self.unrollIdx else ""), self.endLine)
    if kernel["LoopDoWhile"]:
      kStr += "%sdo {%s" % (self.indent, self.endLine)
    else:
      kStr += "%swhile (sumIter%s-- > %u) {%s" \
          % (self.indent, loopChar, \
          (1 if (kernel["PrefetchGlobalRead"] and loopIdx == self.unrollIdx \
          and not tailLoop) else 0), self.endLine)
    self.indent += "  "
    return kStr

  ##############################################################################
  # Close Loop
  ##############################################################################
  def closeLoop(self, kernel, loopIdx):
    return ""
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    self.indent = self.indent[2:]
    if kernel["LoopDoWhile"]:
      kStr += "%s} while (--sumIter%s > %u);%s" \
          % (self.indent, loopChar, \
          (1 if kernel["PrefetchGlobalRead"] else 0), self.endLine )
    else:
      kStr += "%s}%s" % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # MAC Iteration
  ##############################################################################
  def macIter(self, kernel, black):
    return ""
    kStr = ""
    kStr += "%sMAC_%ux%u" % (self.indent, \
        kernel["ThreadTile0"],kernel["ThreadTile1"])
    if black:
      kStr += "_BLK"
    kStr += self.endLine
    return kStr

  ##############################################################################
  # At Least 1 Unroll
  ##############################################################################
  def openSumAtLeastUnroll(self, kernel):
    return ""
    kStr = ""
    kStr += "%sif (size%s >= DEPTHU) {%s" \
        % (self.indent, self.unrollChar, self.endLine)
    self.indent += "  "
    return kStr
  def closeSumAtLeastUnroll(self, kernel):
    return ""
    kStr = ""
    self.indent = self.indent[2:]
    kStr += "%s}%s" % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Tail Loop: Num Iter
  ##############################################################################
  def tailLoopNumIter(self, kernel):
    return ""
    kStr = ""
    kStr += "%ssumIter%s = (((size%s %% DEPTHU) + SPLITU - 1) / SPLITU);%s" \
          % (self.indent, self.unrollChar, self.unrollChar, self.endLine)
    return kStr

  ##############################################################################
  # Global Read: Increment A
  ##############################################################################
  def globalReadIncrementA(self, kernel, loopIdx):
    return ""
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numReadVectorComponentsA):
          if self.readTileDimVectorA or self.readUnrollDimVectorA:
            kStr += "%sglobalReadA_%u_%u%s = (%sVECTOR_TYPE const *)( ((%sDATA_TYPE const *)globalReadA_%u_%u%s) + globalReadIncA%s);%s" \
                % (self.indent, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else ""), \
                self.globalPtrStr, self.globalPtrStr, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else ""), \
                loopChar, self.endLine)
          else:
            kStr += "%sglobalReadA_%u_%u%s += globalReadIncA%s%s;%s" \
                % (self.indent, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else ""), \
                loopChar, "" if (self.readTileDimComponentsA \
                or self.readUnrollDimComponentsA) else "/VECTOR_WIDTH", \
                self.endLine)
    return kStr

  ##############################################################################
  # Global Read: Increment B
  ##############################################################################
  def globalReadIncrementB(self, kernel, loopIdx):
    return ""
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numReadVectorComponentsB):
          if self.readTileDimVectorB or self.readUnrollDimVectorB:
            kStr += "%sglobalReadB_%u_%u%s = (%sVECTOR_TYPE const *)( ((%sDATA_TYPE const *)globalReadB_%u_%u%s) + globalReadIncB%s);%s" \
                % (self.indent, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else ""), \
                self.globalPtrStr, self.globalPtrStr, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else ""), \
                loopChar, self.endLine )
          else:
            kStr += "%sglobalReadB_%u_%u%s += globalReadIncB%s%s;%s" \
                % (self.indent, para, perp, \
                (("_s%u"%s) if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else ""), \
                loopChar, "" if (self.readTileDimComponentsB \
                or self.readUnrollDimComponentsB) else "/VECTOR_WIDTH", \
                self.endLine)
    return kStr

  ##############################################################################
  # Global Read: Do It A
  ##############################################################################
  def globalReadDoA(self, kernel, guardK):
    return ""
    kStr = ""
    return kStr
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numReadVectorComponentsA):
          kStr += "%sa_%u_%u%s = " % (self.indent, para, perp, \
              ((".%s"%self.vectorComponents[s]) if (self.readTileDimComponentsA\
              or self.readUnrollDimComponentsA) else "") )
          # guard around K
          if guardK:
            kStr += "( globalReadOffsetA%s_%u%s >= (size%s %% DEPTHU) )" \
                % (self.unrollChar, \
                (perp if kernel["ProblemType"]["TLUA"] else para), \
                (("_s%u"%s) if self.readUnrollDimComponentsA else ""), \
                self.unrollChar)
          # guard around edge
          if kernel["EdgeType"] == "Branch":
            if guardK:
              kStr += " || "
            kStr += "( !inBoundsA_%u )" % ( \
                (para if kernel["ProblemType"]["TLUA"] else perp) )
          if kernel["EdgeType"] == "Branch" or guardK:
            kStr += " ? %s : " % \
               kernel["ProblemType"]["DataType"].zeroString(self.language, kernel["VectorWidth"])
          kStr += "*globalReadA_%u_%u%s;%s" % (para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsA \
              or self.readUnrollDimComponentsA) else ""), self.endLine)
    return kStr

  ##############################################################################
  # Global Gead: Do It B
  ##############################################################################
  def globalReadDoB(self, kernel, guardK):
    return ""
    kStr = ""
    return kStr
    # global read B
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numReadVectorComponentsB):
          kStr += "%sb_%u_%u%s = " % (self.indent, para, perp, \
              ((".%s"%self.vectorComponents[s]) if (self.readTileDimComponentsB\
              or self.readUnrollDimComponentsB) \
              else "") )
          # guard around k
          if guardK:
            kStr += "( globalReadOffsetB%s_%u%s >= (size%s %% DEPTHU) )" \
                % (self.unrollChar, \
                (perp if kernel["ProblemType"]["TLUB"] else para), \
                (("_s%u"%s) if self.readUnrollDimComponentsB else ""), \
                self.unrollChar)
          # guard around edge
          if kernel["EdgeType"] == "Branch":
            if guardK:
              kStr += " || "
            kStr += "( !inBoundsB_%u )" % ( \
                (para if kernel["ProblemType"]["TLUB"] else perp) )
          if kernel["EdgeType"] == "Branch" or guardK:
            kStr += " ? %s : " % \
                kernel["ProblemType"]["DataType"].zeroString(self.language, kernel["VectorWidth"])
          kStr += "*globalReadB_%u_%u%s;%s" \
              % (para, perp, \
              (("_s%u"%s) if (self.readTileDimComponentsB \
              or self.readUnrollDimComponentsB) else ""), self.endLine)
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets A
  ##############################################################################
  def localWriteSwapOffsetsA(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "%slocalWriteOffsetA_%u_%u%s = (localWriteOffsetA_%u_%u%s + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
              % (self.indent, \
              para, perp, (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              para, perp, (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), self.endLine )
          """
          kStr += "%slocalWriteA_%u_%u%s = (%s%s *)(localMemory + localWriteOffsetA_%u_%u%s);%s"\
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              self.sharedPtrStr, ("DATA_TYPE" if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              self.endLine)
          """
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets B
  ##############################################################################
  def localWriteSwapOffsetsB(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "%slocalWriteOffsetB_%u_%u%s = (localWriteOffsetB_%u_%u%s + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              para, perp, (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), self.endLine )
          """
          kStr += "%slocalWriteB_%u_%u%s = (%s%s *)(localMemory + localWriteOffsetB_%u_%u%s);%s"\
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              self.sharedPtrStr, ("DATA_TYPE" if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              self.endLine)
          """
    return kStr

  ##############################################################################
  # Local Write: Reset Offsets A
  ##############################################################################
  def localWriteResetOffsetsA(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "%slocalWriteOffsetA_%u_%u%s %%= LDS_OFFSET_BLK;%s" \
              % (self.indent, \
              para, perp, (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), self.endLine )
    return kStr

  ##############################################################################
  # Local Write: Reset Offsets B
  ##############################################################################
  def localWriteResetOffsetsB(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "%slocalWriteOffsetB_%u_%u%s %%= LDS_OFFSET_BLK;%s" \
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), self.endLine )
    return kStr



  ##############################################################################
  # Local Write: Init Pointers A
  ##############################################################################
  def localWriteInitPointersA(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "%slocalWriteA_%u_%u%s = (%s%s *)(localMemory + localWriteOffsetA_%u_%u%s);%s"\
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              self.sharedPtrStr, ("DATA_TYPE" if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              self.endLine)
    return kStr

  ##############################################################################
  # Local Write: Init Pointers B
  ##############################################################################
  def localWriteInitPointersB(self, kernel):
    return ""
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "%slocalWriteB_%u_%u%s = (%s%s *)(localMemory + localWriteOffsetB_%u_%u%s);%s"\
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              self.sharedPtrStr, ("DATA_TYPE" if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else "VECTOR_TYPE"), \
              para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              self.endLine)
    return kStr



  ##############################################################################
  # Local Write: Do It A
  ##############################################################################
  def localWriteDoA(self, kernel):
    return ""
    kStr = ""
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic push" + self.endLine
      kStr += "#pragma clang diagnostic ignored \"-Wconditional-uninitialized\"" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          kStr += "%s*localWriteA_%u_%u%s = a_%u_%u%s;%s" \
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "" ), \
              para, perp, \
              ((".%s"%self.vectorComponents[s]) \
              if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "" ), \
              self.endLine)
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic pop" + self.endLine
    return kStr

  ##############################################################################
  # Local Write: Do It B
  ##############################################################################
  def localWriteDoB(self, kernel):
    return ""
    kStr = ""
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic push" + self.endLine
      kStr += "#pragma clang diagnostic ignored \"-Wconditional-uninitialized\"" + self.endLine
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          kStr += "%s*localWriteB_%u_%u%s = b_%u_%u%s;%s" \
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else "" ), \
              para, perp, \
              ((".%s"%self.vectorComponents[s]) \
              if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else "" ), \
              self.endLine)
    if self.language == "HIP":
      kStr += "#pragma clang diagnostic pop" + self.endLine
    return kStr

  ##############################################################################
  # Local Read: Swap Offsets A
  ##############################################################################
  def localReadSwapOffsetsA(self, kernel):
    return ""
    kStr = ""
    kStr += "%slocalReadOffsetA = (localReadOffsetA + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
        % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Wwap Offsets B
  ##############################################################################
  def localReadSwapOffsetsB(self, kernel):
    return ""
    kStr = ""
    kStr += "%slocalReadOffsetB = (localReadOffsetB + LDS_OFFSET_BLK)%%(LDS_OFFSET_BLK*2);%s" \
        % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets A
  ##############################################################################
  def localReadResetOffsetsA(self, kernel):
    return ""
    kStr = ""
    kStr += "%slocalReadOffsetA %%= LDS_OFFSET_BLK;%s" \
        % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets B
  ##############################################################################
  def localReadResetOffsetsB(self, kernel):
    return ""
    kStr = ""
    kStr += "%slocalReadOffsetB %%= LDS_OFFSET_BLK;%s" \
        % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Init Pointers A
  ##############################################################################
  def localReadInitPointersA(self, kernel):
    return ""
    kStr = ""
    kStr += "%slocalReadA = (%sVECTOR_TYPE *)(localMemory + localReadOffsetA);%s" \
        % (self.indent, self.sharedPtrStr, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Init Pointers B
  ##############################################################################
  def localReadInitPointersB(self, kernel):
    return ""
    kStr = ""
    kStr += "%slocalReadB = (%sVECTOR_TYPE *)(localMemory + localReadOffsetB);%s" \
        % (self.indent, self.sharedPtrStr, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Increment A
  ##############################################################################
  def localReadIncA(self, kernel):
    return ""
    kStr = ""
    kStr += "%slocalReadA += SPLITU*(MT%s/VECTOR_WIDTH+PAD);%s" \
        % (self.indent, self.tileChar0, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Increment B
  ##############################################################################
  def localReadIncB(self, kernel):
    return ""
    kStr = ""
    kStr += "%slocalReadB += SPLITU*(MT%s/VECTOR_WIDTH+PAD);%s" \
        % (self.indent, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Do It A
  ##############################################################################
  def localReadDoA(self, kernel, black):
    return ""
    kStr = ""
    for a in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
      kStr += "%srA[%d%s] = localReadA[%d*SG%s]; %s" \
          % (self.indent, a, \
          (("+TT%s/VECTOR_WIDTH"%self.tileCharA) if black else ""), \
          a, self.tileChar0, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Do It B
  ##############################################################################
  def localReadDoB(self, kernel, black):
    return ""
    kStr = ""
    for b in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      kStr += "%srB[%d%s] = localReadB[%d*SG%s]; %s" \
          % (self.indent, b, \
          (("+TT%s/VECTOR_WIDTH"%self.tileCharB) if black else ""), \
          b, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Shift Vector Components d0
  ##############################################################################
  def shiftVectorComponents0(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int wgMT%s = size%s - wg%s*MT%s;%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar0, \
        self.tileChar0, self.endLine)
    kStr += "  if (wgMT%s > MT%s) wgMT%s = MT%s;%s" \
        %(self.tileChar0, self.tileChar0, self.tileChar0, \
        self.tileChar0, self.endLine)
    kStr += "  unsigned int r%s = wgMT%s %% VECTOR_WIDTH;%s" \
        % (self.tileChar0, self.tileChar0, self.endLine)
    kStr += "  if (r%s > 0 && ((wgMT%s/VECTOR_WIDTH)%%SG%s) == serial %% SG%s ) {%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar0, \
        self.tileChar0, self.endLine)
    kStr += "    unsigned int s%s = (wgMT%s/VECTOR_WIDTH)/SG%s;%s" \
        % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
    for r0 in range(1, kernel["VectorWidth"]):
      kStr += "    if (r%s == %u) {%s" % (self.tileChar0, r0, self.endLine)
      for tt1 in range(0, kernel["ThreadTile1"]):
        for s in range(0, r0):
          kStr += "      rC[s%s+%u*(TT%s/VECTOR_WIDTH)].%s = rC[s%s+%u*(TT%s/VECTOR_WIDTH)].%s;%s" \
            % (self.tileChar0, tt1, self.tileChar0, self.vectorComponents[s],  \
            self.tileChar0, tt1, self.tileChar0, \
            self.vectorComponents[s+kernel["VectorWidth"]-r0], self.endLine)
      kStr += "    }%s" % self.endLine
    kStr += "  }%s" % self.endLine
    return kStr

  ##############################################################################
  # Shift Vectors Components d1
  ##############################################################################
  def shiftVectorComponents1(self, kernel):
    return ""
    kStr = ""
    kStr += "  unsigned int wgMT%s = size%s - wg%s*MT%s;%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, \
        self.tileChar1, self.endLine)
    kStr += "  if (wgMT%s > MT%s) wgMT%s = MT%s;%s" \
        %(self.tileChar1, self.tileChar1, self.tileChar1, \
        self.tileChar1, self.endLine)
    kStr += "  unsigned int r%s = wgMT%s %% VECTOR_WIDTH;%s" \
        % (self.tileChar1, self.tileChar1, self.endLine)
    kStr += "  if (r%s > 0 && ((wgMT%s/VECTOR_WIDTH) %% SG%s) == ((serial / SG%s) %% SG%s) ) {%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, \
        self.tileChar0, self.tileChar1, \
        self.endLine)
    kStr += "    unsigned int s%s = (wgMT%s/VECTOR_WIDTH)/SG%s;%s" \
        % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)
    for r1 in range(1, kernel["VectorWidth"]):
      kStr += "    if (r%s == %u) {%s" % (self.tileChar1, r1, self.endLine)
      for tt0 in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for s in range(0, r1):
          kStr += "      rC[%u+s%s*(TT%s/VECTOR_WIDTH)*(VECTOR_WIDTH) + %u*(TT%s/VECTOR_WIDTH)] = rC[%u+s%s*(TT%s/VECTOR_WIDTH)*(VECTOR_WIDTH) + %u*(TT%s/VECTOR_WIDTH)];%s" \
            % (tt0, self.tileChar1, self.tileChar0, s, self.tileChar0, \
            tt0, self.tileChar1, self.tileChar0, \
            s+kernel["VectorWidth"]-r1, self.tileChar0, self.endLine)
      kStr += "    }%s" % self.endLine
    kStr += "  }%s" % self.endLine
    return kStr

  ##############################################################################
  # Complex Declare Tmp Registers
  ##############################################################################
  def complexDeclareTmpRegisters(self, kernel):
    return ""
    kStr = ""
    if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
      kStr += "  float type_mac_tmp;" + self.endLine
    if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
      kStr += "  double type_mac_tmp;" + self.endLine
    return kStr


  ##############################################################################
  # LocalSplitU: Local Write
  ##############################################################################
  def localSplitULocalWrite(self, kernel):
    kStr = ""
    kStr += "  %sVECTOR_TYPE *localLocalSplitU = (%sVECTOR_TYPE *)(localMemory);%s" \
        % (self.sharedPtrStr, self.sharedPtrStr, self.endLine)
    for j in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for i in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for s in range(0, kernel["VectorWidth"]):
          kStr += "%slocalLocalSplitU[lr%s + %u*SG%s + (MT%s/VECTOR_WIDTH)*(lr%s*VECTOR_WIDTH + %u + SG%s*VECTOR_WIDTH*%u) + (MT%s*MT%s/VECTOR_WIDTH)*sgId] = rC[%u+%u*(TT%s/VECTOR_WIDTH)+%u*TT%s];%s" \
              % (self.indent, self.tileChar0, i, self.tileChar0, \
              self.tileChar0, self.tileChar1, \
              s, self.tileChar1, j, self.tileChar0, self.tileChar1, i, s, \
              self.tileChar0, j, self.tileChar0, self.endLine)
    kStr += self.indent + self.syncStr + self.endLine
    """
    kStr += "    /* print Local state */" + self.endLine
    kStr += "    for (unsigned int i = serial; i < MT0I*MT1J*SPLITU; i+=NUM_THREADS) {%s" % self.endLine
    kStr += "      printf(\\\"localLocalSplitU[%%06u] = %%10.0f, %%10.0f\\\\n\\\", i, localLocalSplitU[i], localLocalSplitU[i]);%s" \
        % self.endLine
    kStr += "    }" + self.endLine
    """
    return kStr

  ##############################################################################
  # LocalSplitU: Local Read
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    kStr = ""
    for i in range(0, kernel["NumVectorsPerThread"]):
      kStr += "  rC[%3u] = localLocalSplitU[serial+%u*NUM_THREADS];%s" \
          % (i, i, self.endLine)
    kStr += self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Reduction
  ##############################################################################
  def localSplitUReduction(self, kernel):
    kStr = ""
    for s in range(1, kernel["LocalSplitU"]):
      for i in range(0, kernel["NumVectorsPerThread"]):
        kStr += "  rC[%3u] += localLocalSplitU[serial+%u*NUM_THREADS + %u*(MT%s*MT%s/VECTOR_WIDTH)];%s" \
            % (i, i, s, self.tileChar0, self.tileChar1, self.endLine)
      kStr += self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write Indices
  ##############################################################################
  def localSplitUGlobalWriteIndices(self, kernel):
    kStr = ""
    kStr += "  unsigned int localC%s = (serial %% (MT%s/VECTOR_WIDTH))*VECTOR_WIDTH;%s" \
        % (self.tileChar0, self.tileChar0, self.endLine)
    kStr += "  unsigned int localC%s = serial / (MT%s/VECTOR_WIDTH);%s" \
        % (self.tileChar1, self.tileChar0, self.endLine)
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int globalC%s = wg%s" \
          % (self.indexChars[i], self.indexChars[i])
      if i == kernel["ProblemType"]["Index0"]:
        kStr += "*MT%s + localC%s" \
            % (self.tileChar0, self.tileChar0)
      if i == kernel["ProblemType"]["Index1"]:
        kStr += "*MT%s + localC%s" \
            % (self.tileChar1, self.tileChar1)
      kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Global Write
  ##############################################################################
  def localSplitUGlobalWrite(self, kernel):
    kStr = ""
    if kernel["ProblemType"]["DataType"].value == DataType.complexSingle:
      kStr += "  float type_mac_tmp;" + self.endLine
    if kernel["ProblemType"]["DataType"].value == DataType.complexDouble:
      kStr += "  double type_mac_tmp;" + self.endLine

    for b in range(0, kernel["NumVectorsPerThread"]):
      for s in range(0, kernel["VectorWidth"]):
        if kernel["EdgeType"] != "None":
          kStr += "  if (globalC%s%s < size%s) {" \
              % (self.tileChar0, \
              ((" + %u" %s) if kernel["VectorWidth"]>1 else ""), \
              self.tileChar0)
          kStr += "  if (globalC%s + %u*CPS < size%s) {" \
              % (self.tileChar1, b, self.tileChar1)

        kStr += "  TYPE_MAC_WRITE( C[ GLOBAL_C( (%s)" % self.uint64Str
        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          kStr += " globalC%s" % self.indexChars[i]
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
    return kStr

  ##############################################################################
  # Not LocalSplitU: Global Write Indices
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    kStr = ""
    for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
      kStr += "  unsigned int globalC" + self.indexChars[i] \
          + " = wg" + self.indexChars[i]
      if i == kernel["ProblemType"]["Index0"]:
        kStr += "*MT%s + (serial %% SG%s)*VECTOR_WIDTH" \
            % (self.tileChar0, self.tileChar0)
      if i == kernel["ProblemType"]["Index1"]:
        kStr += "*MT%s + (serial / SG%s)*VECTOR_WIDTH" \
            % (self.tileChar1, self.tileChar0)
      kStr += ";" + self.endLine
    return kStr

  ##############################################################################
  # Not LocalSplitU: Global Write
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel):
    kStr = ""
    for b in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for a in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for s1 in range(0, kernel["VectorWidth"]):
          for s0 in range(0, kernel["VectorWidth"]):
            if kernel["EdgeType"] == "Branch":
              kStr += "  if (globalC%s + (VECTOR_WIDTH-1) + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar0, a, self.tileChar0, self.tileChar0)
              kStr += "  if (globalC%s + (VECTOR_WIDTH-1) + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar1, b, self.tileChar1, self.tileChar1)
            elif kernel["EdgeType"] == "ShiftPtr":
              kStr += "  if (globalC%s%s + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar0, \
                  ((" + %u"%s0) if kernel["VectorWidth"]>1 else ""), \
                  a, self.tileChar0, self.tileChar0)
              kStr += "  if (globalC%s%s + %u*SG%s*VECTOR_WIDTH < size%s) {" \
                  % (self.tileChar1, \
                  ((" + %u"%s1) if kernel["VectorWidth"]>1 else ""), \
                  b, self.tileChar1, self.tileChar1)

            kStr += "  TYPE_MAC_WRITE( C[ GLOBAL_C( (%s)" % self.uint64Str
            for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
              kStr += " globalC%s" % self.indexChars[i]
              if i == kernel["ProblemType"]["Index0"]:
                kStr += "%s + %u*SG%s*VECTOR_WIDTH" % (\
                    ((" + %u"%s0) if kernel["VectorWidth"]>1 else ""), \
                    a, self.tileChar0)
              if i == kernel["ProblemType"]["Index1"]:
                kStr += "%s + %u*SG%s*VECTOR_WIDTH" % (\
                    ((" + %u"%s1) if kernel["VectorWidth"]>1 else ""), \
                    b, self.tileChar1)
              if i < kernel["ProblemType"]["NumIndicesC"]-1:
                kStr += ", (%s)" % self.uint64Str
            kStr += ") ]"
            kStr += ", alpha"
            kStr += ", rC[%d+%d*(TT%s/VECTOR_WIDTH)+%d*TT%s]%s" \
                % (a, s1, self.tileChar0, b, self.tileChar0, \
                ((".%s"%self.vectorComponents[s0]) if kernel["VectorWidth"]>1\
                else "") )
            if kernel["ProblemType"]["UseBeta"]:
              kStr += ", beta"
            kStr += ")"

            if kernel["EdgeType"] != "None":
              kStr += " } }"
            kStr += self.endLine
    return kStr

  ##############################################################################
  # Function End
  ##############################################################################
  def functionEnd(self, kernel):
    return ""
    kStr = ""
    kStr += self.endLine
    kStr += "}" + self.endLine
    return kStr

  ##############################################################################
  # Function Suffix
  ##############################################################################
  def functionSuffix(self, kernel):
    return ""
    kStr = ""
    if globalParameters["MergeFiles"] and self.language == "HIP":
      kStr += "#undef UNROLL%s" % self.endLine
      kStr += "#undef SPLITU%s" % self.endLine
      kStr += "#undef DEPTHU%s" % self.endLine
      kStr += "#undef SG%s%s" % (self.tileChar0, self.endLine)
      kStr += "#undef SG%s%s" % (self.tileChar1, self.endLine)
      kStr += "#undef TT%s%s" % (self.tileChar0, self.endLine)
      kStr += "#undef TT%s%s" % (self.tileChar1, self.endLine)
      kStr += "#undef MT%s%s" % (self.tileChar0, self.endLine)
      kStr += "#undef MT%s%s" % (self.tileChar1, self.endLine)
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
      kStr += "#undef LDS_OFFSET_B%s" % (self.endLine)
      kStr += "#undef LDS_OFFSET_BLK%s" % (self.endLine)
      kStr += "#undef LDS_NUM_ELEMENTS%s" % (self.endLine)
      kStr += "#undef NUM_THREADS%s" % (self.endLine)
      kStr += "#undef WORK_GROUP_MAPPING%s" % (self.endLine)
      kStr += "#undef VECTOR_WIDTH%s" % (self.endLine)
      kStr += "#undef TYPE_MAC%s" % (self.endLine)
      kStr += "#undef TYPE_MAC_WRITE%s" % (self.endLine)

      numMacs = 2 if kernel["PrefetchLocalRead"] else 1
      for m in range(0, numMacs):
        kStr += "#undef MAC_%ux%u" \
            % (kernel["ThreadTile0"], kernel["ThreadTile1"])
        if kernel["PrefetchLocalRead"]:
          kStr += ("" if m==0 else "_BLK")
        kStr += self.endLine

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
        kStr += "#undef strideC" + self.indexChars[i] + self.endLine
      for i in range(firstStride, lastStrideA):
        kStr += "#undef strideA" \
            + self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]] \
            + self.endLine
      for i in range(firstStride, lastStrideB):
        kStr += "#undef strideB" \
            + self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]] \
            + self.endLine
      kStr += self.endLine + self.endLine
    return kStr

  ##############################################################################
  # Kernel Body Prefix
  ##############################################################################
  def kernelBodyPrefix(self, kernel):
    return ""
    s = ""
    kernelName = self.getKernelName(kernel)
    if not globalParameters["MergeFiles"]:
      s += "\n"
      s += "#include \"%s.h\"\n" % kernelName
      s += "\n"

    return s

  ##############################################################################
  # Kernel Body Suffix
  ##############################################################################
  def kernelBodySuffix(self, kernel):
    return ""
    s = ""
    kernelName = self.getKernelName(kernel)

    if self.language == "OCL":
      s += "std::string %s_src_concatenated = \n  %s_src_0" \
          % (kernelName, kernelName)
      for i in range(1, self.stringIdx):
        s += "\n  + %s_src_%u" % (kernelName, i)
      s += ";\n"
      s += "const char * const %s_src = %s_src_concatenated.c_str();" \
          % (kernelName, kernelName)

    s += "\n"
    return s


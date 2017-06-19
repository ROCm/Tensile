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

# TODO
# (1) explicit and implicit incrementing of global/local offsets to handle
# to handle when implicit offset spills, attach offsets to a register, when
# we increment that offset and it spills, then it automatically triggers an
# explicit increment and the implicit increment resets
# Address class, request explicit or implicit increment, request reset
#  32,64-bit, which registers, pass into MemoryInstruction class
# (2) MemoryPool to handle all allocations
# (3) Dictionary of gpr names?
# (4) Will I need to use fewer sgprs?
# (5) Divide-only, Remainder-only, Divide&Remainder
# (6) Add and multiply functions, compile time known, handles carrying, only need to debug in one place


from SolutionStructs import DataType
from Common import globalParameters, kernelLanguageIsSource, print1
from KernelWriter import KernelWriter
from math import log
import abc

################################################################################
# Memory Instruction
################################################################################
class MemoryInstruction:
  def __init__(self, name, numAddresses, numOffsets, \
      offsetMultiplier, blockWidth, formatting):
    self.name = name
    self.formatting = formatting
    self.numAddresses = numAddresses
    self.numOffsets = numOffsets
    self.offsetMultiplier = offsetMultiplier
    self.blockWidth = blockWidth
    self.numBlocks = 2 if self.numAddresses > 1 or self.numOffsets > 1 else 1
    self.totalWidth = self.blockWidth * self.numBlocks
    self.endLine = "\n"
  ########################################
  # write in assembly format
  def toString(self, params, comment):
    instStr = "%s %s" % (self.name, (self.formatting % params) )
    line = "%-50s // %s%s" % (instStr, comment, self.endLine)
    return line
  def __str__(self):
    return self.name

################################################################################
# ScratchRegisters
################################################################################
class ScratchRegisters:
  def __init__(self, start, size):
    self.start = start
    self.size = size
    self.available = [True]*self.size
    self.checkOutSize = {}

  ########################################
  # Check Out
  def checkOut(self, size):
    found = -1
    for i in range(0, self.size):
      valid = True
      for j in range(0, size):
        if not self.available[i+j]:
          valid = False
          i = j+1
          break
      if valid:
        found = i
        break
      else:
        continue

    if found > -1:
      for i in range(found, found+size):
        self.available[i] = False
      self.checkOutSize[found] = size
      return (found+self.start)
    else:
      printExit("Ran out of scratch registers.")

  ########################################
  # Check Out
  def checkIn(self, start):
    start -= self.start
    if start in self.checkOutSize:
      size = self.checkOutSize[start]
      for i in range(start, start+size):
        self.available[i] = True
    else:
      printExit("Checking in registers @ %i that weren't checked out"%start)




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

    # ISA version, such as 803
    self.version = int(self.language[3:])
    self.versionMajor = int(self.language[3])
    self.versionMinor = int(self.language[4])
    self.versionPatch = int(self.language[5])
    print1("KernelWriterAsssembly for gfx%u\n" % self.version )

    ########################################
    # Available Memory Instructions
    ########################################

    ########################################
    # Local Read
    ds_read_b128 = MemoryInstruction("ds_read_b128",  1, 1, 4, 4, \
        "%s, %s offset:%s" )
    ds_read2_b64 = MemoryInstruction("ds_read2_b64",  1, 2, 2, 2, \
        "%s, %s offset0:%s, offset1:%s" )
    ds_read_b64 = MemoryInstruction("ds_read_b64",    1, 1, 2, 2, \
        "%s, %s offset:%s" )
    ds_read2_b32 = MemoryInstruction("ds_read2_b32",  1, 2, 1, 1, \
        "%s, %s offset0:%s offset1:%s" )
    ds_read_b32 = MemoryInstruction("ds_read_b32",    1, 1, 1, 1, \
        "%s, %s offset:%s" )
    ########################################
    # Local Write
    ds_write_b128 = MemoryInstruction("ds_write_b128",  1, 1, 4, 4, \
        "%s, %s offset:%s" )
    ds_write2_b64 = MemoryInstruction("ds_write2_b64",  1, 2, 2, 2, \
        "%s, %s, %s offset0:%s, offset1:%s" )
    ds_write_b64 = MemoryInstruction("ds_write_b64",    1, 1, 2, 2, \
        "%s, %s offset:%s" )
    ds_write2_b32 = MemoryInstruction("ds_write2_b32",  1, 2, 1, 1, \
        "%s, %s, %s offset0:%s offset1:%s" )
    ds_write_b32 = MemoryInstruction("ds_write_b32",    1, 1, 1, 1, \
        "%s, %s offset:%s" )
    ########################################
    # Global Read
    flat_load_dwordx4 = MemoryInstruction("flat_load_dwordx4",  1, 0, 0, 4, \
        "%s, %s" )
    flat_load_dwordx2 = MemoryInstruction("flat_load_dwordx2",  1, 0, 0, 2, \
        "%s, %s" )
    flat_load_dword = MemoryInstruction("flat_load_dword",      1, 0, 0, 1, \
        "%s, %s" )
    ########################################
    # Global Write
    flat_store_dwordx4 = MemoryInstruction("flat_store_dwordx4",  1, 0, 0, 4, \
        "%s, %s" )
    flat_store_dwordx2 = MemoryInstruction("flat_store_dwordx2",  1, 0, 0, 2, \
        "%s, %s" )
    flat_store_dword = MemoryInstruction("flat_store_dword",      1, 0, 0, 1, \
        "%s, %s" )

    ########################################
    # Available Memory Instructions per Architecture
    # gfx701 "Hawaii"
    # gfx801 "Carrizo"
    # gfx802 "Tonga"
    # gfx803 "Fiji"
    # gfx900
    ########################################
    self.memoryInstructions = {
        803: { # Fiji
          "GlobalRead": [ flat_load_dwordx4, flat_load_dwordx2,
            flat_load_dword ],
          "GlobalWrite": [ flat_store_dwordx4, flat_store_dwordx2,
            flat_store_dword ],
          "LocalRead": [ ds_read_b128, ds_read2_b64,
            ds_read_b64, ds_read2_b32, ds_read_b32 ],
          "LocalWrite": [ ds_write_b128, ds_write2_b64,
            ds_write_b64, ds_write2_b32, ds_write_b32 ]
          } # 803
        }

    self.endLine = "\n"
    self.syncStr = "s_barrier"
    self.commentPrefix = "/*"
    self.commentSuffix = "*/"
    self.commentHR = "*"*40
    self.indent = ""
    self.labels = {}
    self.localReadOffsetA = 0
    self.localReadOffsetB = 0


  ########################################
  # Get Label
  def getLabel(self, name):
    if name not in self.labels:
      self.labels[name] = len(self.labels)
    return self.labels[name]

  ##############################################################################
  # Find Memory Instruction For Width and Stride
  ##############################################################################
  def findMemoryInstructionForWidthStride(self, width, strides, combine, \
      instructions):
    for i in range(0, len(instructions)):
      instruction = instructions[i]
      name = instruction.name
      numAddresses = instruction.numAddresses
      numOffsets = instruction.numOffsets
      offsetMultiplier = instruction.offsetMultiplier
      blockWidth = instruction.blockWidth
      valid = True
      if width < blockWidth:
        valid = False
      if combine: # try to combine ops
        if numOffsets > 0: # if inst combines using offsets
          for stride in strides:
            if stride % offsetMultiplier != 0:
              valid = False
      else: # don't try to combine ops
        if numOffsets > 1 or numAddresses > 1:
          valid = False
      if valid:
        return i
      else:
        continue
    return len(instructions)


  ##############################################################################
  # Select Memory Instruction
  # when selecting instruction, need to support stride in both dims
  ##############################################################################
  def selectMemoryInstruction(self,
      operation, # ReadGlobal, WriteLocal, ReadLocal
      width, # num registers 1 chunk
      write2, # Para, Perp, None
      para2, # NumLoadsPara >= 2
      perp2, # NumLoadsPerp >= 2
      strides ):
    #instructions = self.memoryArchitecture[operation]
    instructions = self.memoryInstructions[self.version][operation]
    # try to combine
    if (write2 == "Coalesced" and para2) \
        or (write2 == "Perpendicular" and perp2):
      instructionIdx = self.findMemoryInstructionForWidthStride( \
          width, strides, True, instructions)
      if instructionIdx < len(instructions): # found combined
        return instructionIdx

    # don't or can't combine
    return self.findMemoryInstructionForWidthStride( \
        width, strides, False, instructions)

# TODO: option: when offset bits aren't sufficient, do we use VALU to
# increment address or do we use extra registers to store addresses?
# (1) write1 and aways have sufficient offset bits
# (2) write2 and if insufficient offset bits then IncrementAndReset
# (3) write2 and if insufficient offset bits then AllocateAdditionalAddresses

  ##############################################################################
  #
  #   Functions to Write Kernel Segments
  #
  ##############################################################################

  ##############################################################################
  # Init Kernel
  ##############################################################################
  def initKernel(self, kernel):
    super(KernelWriterAssembly, self).initKernel(kernel)
    self.kernelName = self.getKernelName(kernel)

    # registers per element
    self.rpe = kernel["ProblemType"]["DataType"].numRegisters()
    self.bpe = self.rpe * 4
    # registers per global address
    self.rpga = 2 # 64-bit
    # registers per local address
    self.rpla = 1 # 32-bit

    ####################################
    # choose memory instructions
    ####################################

    ########################################
    # globalReadA instruction; no flat_load2_*
    #globalReadStrideTile = 0
    #globalReadStrideUnroll = 0
    self.globalReadWidthA = kernel["VectorWidth"] if self.readTileDimVectorA \
        else 1
    self.globalReadWidthA *= self.rpe
    self.globalRead2CoalescedA = kernel["NumLoadsCoalescedA"]>1 \
        or self.readCoalescedComponentsA
    self.globalRead2PerpendicularA = kernel["NumLoadsPerpendicularA"]>1 \
        or self.readPerpendicularComponentsA
    self.globalReadInstructionIdxA = \
        self.selectMemoryInstruction("GlobalRead", self.globalReadWidthA, \
        kernel["GlobalRead2A"], \
        self.globalRead2CoalescedA, self.globalRead2PerpendicularA, [] )

    ########################################
    # globalReadB instruction; no flat_load2_
    self.globalReadWidthB = kernel["VectorWidth"] if self.readTileDimVectorB  \
        else 1
    self.globalReadWidthB *= self.rpe
    self.globalRead2CoalescedB = kernel["NumLoadsCoalescedB"]>1 \
        or self.readCoalescedComponentsB
    self.globalRead2PerpendicularB = kernel["NumLoadsPerpendicularB"]>1 \
        or self.readPerpendicularComponentsB
    self.globalReadInstructionIdxB = \
        self.selectMemoryInstruction("GlobalRead", self.globalReadWidthB, \
        kernel["GlobalRead2B"], \
        self.globalRead2CoalescedB, self.globalRead2PerpendicularB, [] )

    ########################################
    # localWriteA instruction
    # for local, tile->para, unroll->perp
    self.localWriteWidthA = 1 if (self.writeTileDimComponentsA \
        or self.writeUnrollDimComponentsA) else kernel["VectorWidth"]
    self.localWriteWidthA *= self.rpe
    self.localWrite2CoalescedA = self.numWritesCoalescedA>1 \
        or self.writeTileDimComponentsA
    self.localWrite2PerpendicularA = self.numWritesPerpendicularA>1 \
        or self.writeUnrollDimComponentsA
    # localWriteA stride tile
    if kernel["ProblemType"]["TLUA"]:
      if self.writeTileDimComponentsA:
        self.localWriteStrideTileA = 1
        self.localWriteJoinTileA = "Components"
      else:
        self.localWriteStrideTileA = kernel["LSCA"]
        self.localWriteJoinTileA = "Coalesced"
    else:
      if self.writeUnrollDimComponentsA:
        self.localWriteStrideTileA = 1
        self.localWriteJoinTileA = "Components"
      else:
        self.localWriteStrideTileA = kernel["LSPA"]
        self.localWriteJoinTileA = "Perpendicular"
    self.localWriteStrideTileA *= self.rpe
    # localWriteA stride unroll
    if kernel["ProblemType"]["TLUA"]:
      if self.writeUnrollDimComponentsA:
        self.localWriteStrideUnrollA = 1*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Components"
      else:
        self.localWriteStrideUnrollA = kernel["LSCA"]*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Perpendicular"
    else:
      if self.writeTileDimComponentsA:
        self.localWriteStrideUnrollA = 1*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Components"
      else:
        self.localWriteStrideUnrollA = kernel["LSCA"]*kernel["MacroTileA"]
        self.localWriteJoinUnrollA = "Coalesced"
    self.localWriteStrideUnrollA *= self.rpe
    self.localWriteInstructionIdxA = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthA, \
        kernel["LocalWrite2A"], \
        self.localWrite2CoalescedA, self.localWrite2PerpendicularA,
        [self.localWriteStrideTileA, self.localWriteStrideUnrollA] )

    ########################################
    # localWriteB instruction
    # for local, tile->para, unroll->perp
    self.localWriteWidthB = 1 if (self.writeTileDimComponentsB \
        or self.writeUnrollDimComponentsB) else kernel["VectorWidth"]
    self.localWriteWidthB *= self.rpe
    self.localWrite2CoalescedB = self.numWritesCoalescedB>1 \
        or self.writeTileDimComponentsB
    self.localWrite2PerpendicularB = self.numWritesPerpendicularB>1 \
        or self.writeUnrollDimComponentsB
    # localWriteB stride tile
    if kernel["ProblemType"]["TLUB"]:
      if self.writeTileDimComponentsB:
        self.localWriteStrideTileB = 1
        self.localWriteJoinTileB = "Components"
      else:
        self.localWriteStrideTileB = kernel["LSCB"]
        self.localWriteJoinTileB = "Coalesced"
    else:
      if self.writeUnrollDimComponentsB:
        self.localWriteStrideTileB = 1
        self.localWriteJoinTileB = "Components"
      else:
        self.localWriteStrideTileB = kernel["LSPB"]
        self.localWriteJoinTileB = "Perpendicular"
    self.localWriteStrideTileB *= self.rpe
    # localWriteB stride unroll
    if kernel["ProblemType"]["TLUB"]:
      if self.writeUnrollDimComponentsB:
        self.localWriteStrideUnrollB = 1*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Components"
      else:
        self.localWriteStrideUnrollB = kernel["LSCB"]*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Perpendicular"
    else:
      if self.writeTileDimComponentsB:
        self.localWriteStrideUnrollB = 1*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Components"
      else:
        self.localWriteStrideUnrollB = kernel["LSCB"]*kernel["MacroTileB"]
        self.localWriteJoinUnrollB = "Coalesced"
    self.localWriteStrideUnrollB *= self.rpe
    self.localWriteInstructionIdxB = \
        self.selectMemoryInstruction("LocalWrite", self.localWriteWidthB, \
        kernel["LocalWrite2B"], \
        self.localWrite2CoalescedB, self.localWrite2PerpendicularB,
        [self.localWriteStrideTileB, self.localWriteStrideUnrollB] )

    ########################################
    # localRead A
    self.localReadWidth = kernel["VectorWidth"] * self.rpe
    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedA = kernel["ThreadTile0"] * self.rpe
    self.localRead2CoalescedA = kernel["ThreadTile0"]/kernel["VectorWidth"] > 1
    self.localReadInstructionIdxA = \
        self.selectMemoryInstruction("LocalRead", self.localReadWidth, \
        kernel["LocalRead2A"], \
        self.localRead2CoalescedA, localRead2Perpendicular,
        [self.localReadStrideCoalescedA] )

    ########################################
    # localRead B
    self.localReadWidth = kernel["VectorWidth"] * self.rpe
    #localReadStridePerpendicular = 0
    localRead2Perpendicular = False
    self.localReadStrideCoalescedB = kernel["ThreadTile1"] * self.rpe
    self.localRead2CoalescedB = kernel["ThreadTile1"]/kernel["VectorWidth"] > 1
    self.localReadInstructionIdxB = \
        self.selectMemoryInstruction("LocalRead", self.localReadWidth, \
        kernel["LocalRead2B"], \
        self.localRead2CoalescedB, localRead2Perpendicular,
        [self.localReadStrideCoalescedB] )

    instructions = self.memoryInstructions[self.version]
    self.globalReadInstructionA = instructions["GlobalRead"][ \
        self.globalReadInstructionIdxA]
    self.globalReadInstructionB = instructions["GlobalRead"][ \
        self.globalReadInstructionIdxB]
    self.localWriteInstructionA = instructions["LocalWrite"][ \
        self.localWriteInstructionIdxA]
    self.localWriteInstructionB = instructions["LocalWrite"][ \
        self.localWriteInstructionIdxB]
    self.localReadInstructionA = instructions["LocalRead"][ \
        self.localReadInstructionIdxA]
    self.localReadInstructionB = instructions["LocalRead"][ \
        self.localReadInstructionIdxB]
    print self.getKernelName(kernel)
    """
    print "\n"
    print self.getKernelName(kernel)
    print "GlobalReadInstructionA", self.globalReadInstructionA
    print "GlobalReadInstructionB", self.globalReadInstructionB
    print "LocalWriteInstructionA", self.localWriteInstructionA
    print "LocalWriteInstructionB", self.localWriteInstructionB
    print "LocalReadInstructionA ", self.localReadInstructionA
    print "LocalReadInstructionB ", self.localReadInstructionB
    """

    ####################################
    # VGPR Allocation
    ####################################

    ####################################
    # num vgprs: valu
    self.numVgprValuC = kernel["ThreadTile0"]*kernel["ThreadTile1"]*self.rpe
    numVgprValuA = kernel["ThreadTileA"]*self.rpe
    numVgprValuB = kernel["ThreadTileB"]*self.rpe
    numVgprValuBlkA = numVgprValuA if kernel["PrefetchLocalRead"] else 0
    numVgprValuBlkB = numVgprValuB if kernel["PrefetchLocalRead"] else 0

    ####################################
    # num vgprs: global -> local elements
    numVgprG2LA = kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["VectorWidth"] * self.rpe
    numVgprG2LB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["VectorWidth"] * self.rpe

    ####################################
    # num vgprs: local read addresses
    numVgprLocalReadAddressesA = 1 * self.rpla
    numVgprLocalReadAddressesB = 1 * self.rpla

    ####################################
    # num vgprs: local write addresses
    #numLocalWritesA = kernel["NumLoadsCoalescedA"] \
    #    * kernel["NumLoadsPerpendicularA"] * self.numWriteVectorComponentsA
    #numLocalWriteInstructionsA = numLocalWritesA \
    #    / self.localWriteInstructionA[self.instructionIdxNumOffsets]
    numVgprLocalWriteAddressesA = 1 * self.rpla

    #numLocalWritesB = kernel["NumLoadsCoalescedB"] \
    #    * kernel["NumLoadsPerpendicularB"] * self.numWriteVectorComponentsB
    #numLocalWriteInstructionsB = numLocalWritesB \
    #    / self.localWriteInstructionB[self.instructionIdxNumOffsets]
    numVgprLocalWriteAddressesB = 1 * self.rpla

    ####################################
    # num vgprs: global read addresses
    numGlobalReadsA = kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["VectorWidth"] \
        * self.numReadVectorComponentsA
    numGlobalReadInstructionsA = numGlobalReadsA \
        / self.globalReadInstructionA.blockWidth
    numVgprGlobalReadAddressesA = numGlobalReadInstructionsA * self.rpga

    numGlobalReadsB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["VectorWidth"] \
        * self.numReadVectorComponentsB
    numGlobalReadInstructionsB = numGlobalReadsB \
        / self.globalReadInstructionB.blockWidth
    numVgprGlobalReadAddressesB = numGlobalReadInstructionsB * self.rpga
    numVgprSerial = 1

    numVgprAddressD = 1 * self.rpga

    ####################################
    # num vgprs: c write address
    # 1 address where to write first value
    # 1 tmp address where to write current value


    ####################################
    # VGPR Assignment
    ####################################
    vgprIdx = 0
    self.startVgprValuC = vgprIdx; vgprIdx += self.numVgprValuC

    self.startVgprValuA = vgprIdx; vgprIdx += numVgprValuA
    self.startVgprValuBlkA = vgprIdx; vgprIdx += numVgprValuBlkA
    if kernel["PrefetchGlobalRead"]:
      self.startVgprG2LA = vgprIdx; vgprIdx += numVgprG2LA
    else: # g2l can overlap valu
      self.startVgprG2LA = self.startVgprValuA
      vgprIdx = self.startVgprValuA \
          + max(numVgprValuA+numVgprValuBlkA, numVgprG2LA)

    self.startVgprValuB = vgprIdx; vgprIdx += numVgprValuB
    self.startVgprValuBlkB = vgprIdx; vgprIdx += numVgprValuBlkB
    if kernel["PrefetchGlobalRead"]:
      self.startVgprG2LB = vgprIdx; vgprIdx += numVgprG2LB
    else: # g2l can overlap valu
      self.startVgprG2LB = self.startVgprValuB
      vgprIdx = self.startVgprValuB \
          + max(numVgprValuB+numVgprValuBlkB, numVgprG2LB)

    self.startVgprLocalReadAddressesA = vgprIdx
    vgprIdx += numVgprLocalReadAddressesA
    self.startVgprLocalReadAddressesB = vgprIdx
    vgprIdx += numVgprLocalReadAddressesB
    self.startVgprLocalWriteAddressesA = vgprIdx
    vgprIdx += numVgprLocalWriteAddressesA
    self.startVgprLocalWriteAddressesB = vgprIdx
    vgprIdx += numVgprLocalWriteAddressesB
    self.startVgprGlobalReadAddressesA = vgprIdx
    vgprIdx += numVgprGlobalReadAddressesA
    self.startVgprGlobalReadAddressesB = vgprIdx
    vgprIdx += numVgprGlobalReadAddressesB
    self.startVgprAddressD = vgprIdx
    vgprIdx += numVgprAddressD
    self.startVgprSerial = vgprIdx
    vgprIdx += numVgprSerial
    print1("%3u vgprs <- %s" % (vgprIdx, self.kernelName) )
    self.startVgprTmp = vgprIdx
    vgprPerCU = 65536
    vgprPerThreadPerOccupancy = vgprPerCU / kernel["NumThreads"]
    numWorkGroupsPerCU = vgprPerThreadPerOccupancy / self.startVgprTmp
    numWavesPerWorkGroup = kernel["NumThreads"] / 64
    numWavesPerCU = numWorkGroupsPerCU * numWavesPerWorkGroup
    self.numWavesPerSimd = numWavesPerCU / 4
    maxVgprSameOccupancy = vgprPerThreadPerOccupancy / numWorkGroupsPerCU
    self.numVgprTmp = maxVgprSameOccupancy - self.startVgprTmp
    self.totalVgprs = maxVgprSameOccupancy
    self.globalWriteAddrC = self.totalVgprs-4 # match macro

    ########################################
    # Pre Loop Scratch Vgprs
    ########################################
    self.vgprScratch = ScratchRegisters(self.startVgprValuC, \
        self.startVgprLocalReadAddressesA - self.startVgprValuC)

    ########################################
    # SGPR Allocation
    ########################################

    ####################################
    # num sgprs: initial kernel state
    numSgprKernArgAddress = self.rpga
    numSgprWorkGroup0 = 1
    numSgprWorkGroup1 = 1
    numSgprWorkGroup2 = 1 # assume batched gemm at least
    numSgprAddressC = self.rpga # til end
    numSgprAddressA = self.rpga # til read offsets
    numSgprAddressB = self.rpga # til read offsets
    numSgprOffsetC = 1
    numSgprOffsetA = 1
    numSgprOffsetB = 1
    numSgprAlpha = 1
    numSgprBeta = 1 if kernel["ProblemType"]["UseBeta"] else 0
    self.numSgprStridesC = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprStridesA = len(kernel["ProblemType"]["IndexAssignmentsA"])
    self.numSgprStridesB = len(kernel["ProblemType"]["IndexAssignmentsB"])
    if not kernel["ProblemType"]["UseInitialStrides"]:
      self.numSgprStridesC -= 1
      self.numSgprStridesA -= 1
      self.numSgprStridesB -= 1
    self.numSgprSizesSum = kernel["ProblemType"]["NumIndicesSummation"]
    self.numSgprSizesFree = kernel["ProblemType"]["NumIndicesC"]
    self.numSgprAddressD = self.rpga

    ####################################
    # num sgprs: global read increments
    numSgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] \
        * self.rpga
    numSgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] \
        * self.rpga
    numSgprLoopCounters = 1 * kernel["ProblemType"]["NumIndicesSummation"]

    numSgprLoopCountersAndIncrements = numSgprGlobalReadIncsA \
        + numSgprGlobalReadIncsB + numSgprLoopCounters
    numSgprFreedBeforeLoops = self.numSgprStridesA + self.numSgprStridesB \
        + self.numSgprSizesFree + numSgprAddressA + numSgprAddressB \
        + numSgprOffsetC + numSgprOffsetA + numSgprOffsetB
    numSgprLoopPadding = max(0, numSgprFreedBeforeLoops  \
        - numSgprLoopCountersAndIncrements)

    ########################################
    # SGPR Assignment
    ########################################
    sgprIdx = 0
    self.startSgprKernArgAddress = sgprIdx; sgprIdx += numSgprKernArgAddress
    self.startSgprWorkGroup0 = sgprIdx; sgprIdx += numSgprWorkGroup0
    self.startSgprWorkGroup1 = sgprIdx; sgprIdx += numSgprWorkGroup1
    self.startSgprWorkGroup2 = sgprIdx; sgprIdx += numSgprWorkGroup2
    self.startSgprAddressC = sgprIdx; sgprIdx += numSgprAddressC
    self.startSgprStridesC = sgprIdx; sgprIdx += self.numSgprStridesC
    self.startSgprAlpha = sgprIdx; sgprIdx += numSgprAlpha
    self.startSgprBeta = sgprIdx; sgprIdx += numSgprBeta
    self.startSgprSizesSum = sgprIdx; sgprIdx += self.numSgprSizesSum
    self.startSgprLoopPadding = sgprIdx; sgprIdx += numSgprLoopPadding # overlap
    self.startSgprStridesA = sgprIdx; sgprIdx += self.numSgprStridesA
    self.startSgprStridesB = sgprIdx; sgprIdx += self.numSgprStridesB
    self.startSgprSizesFree = sgprIdx; sgprIdx += self.numSgprSizesFree
    self.startSgprAddressA = sgprIdx; sgprIdx += numSgprAddressA
    self.startSgprAddressB = sgprIdx; sgprIdx += numSgprAddressB
    self.startSgprOffsetC = sgprIdx; sgprIdx += numSgprOffsetC
    self.startSgprOffsetA = sgprIdx; sgprIdx += numSgprOffsetA
    self.startSgprOffsetB = sgprIdx; sgprIdx += numSgprOffsetB
    self.startSgprAddressD = sgprIdx; sgprIdx += self.numSgprAddressD
    self.totalSgprs = sgprIdx

    # assign loop sgprs which overlap above assignments
    sgprIdx = self.startSgprLoopPadding
    self.startSgprGlobalReadIncsA = sgprIdx; sgprIdx += numSgprGlobalReadIncsA
    self.startSgprGlobalReadIncsB = sgprIdx; sgprIdx += numSgprGlobalReadIncsB
    self.startSgprLoopCounters = sgprIdx

    # TODO - what occupancy does this numSgpr limit to;
    # it probably wouldn't matter but good to calculate and print warning
    # if it is more limiting than vgpr limitation,
    # also print LDS occupancy limitation even though it is explicit



  ##############################################################################
  # format macro
  def macroRegister(self, name, value):
    return ".set %s, %s%s" % (name, value, self.endLine)

  ##############################################################################
  # Function Prefix - DONE
  ##############################################################################
  def functionPrefix(self, kernel):
    kStr = ""

    ########################################
    # VGPR Macros
    ########################################
    kStr += self.comment3("VGPR Assignments")
    kStr += self.macroRegister("vgprValuC", self.startVgprValuC)
    kStr += self.macroRegister("vgprValuA", self.startVgprValuA)
    kStr += self.macroRegister("vgprValuBlkA", self.startVgprValuBlkA)
    kStr += self.macroRegister("vgprG2LA", self.startVgprG2LA)
    kStr += self.macroRegister("vgprValuB", self.startVgprValuB)
    kStr += self.macroRegister("vgprValuBlkB", self.startVgprValuBlkB)
    kStr += self.macroRegister("vgprG2LB", self.startVgprG2LB)
    kStr += self.macroRegister("vgprLocalReadAddrA", \
        self.startVgprLocalReadAddressesA)
    kStr += self.macroRegister("vgprLocalReadAddrB", \
        self.startVgprLocalReadAddressesB)
    kStr += self.macroRegister("vgprLocalWriteAddrA", \
        self.startVgprLocalWriteAddressesA)
    kStr += self.macroRegister("vgprLocalWriteAddrB", \
        self.startVgprLocalWriteAddressesB)
    kStr += self.macroRegister("vgprGlobalReadAddrA", \
        self.startVgprGlobalReadAddressesA)
    kStr += self.macroRegister("vgprGlobalReadAddrB", \
        self.startVgprGlobalReadAddressesB)
    kStr += self.macroRegister("vgprAddressD", \
        self.startVgprAddressD)
    kStr += self.macroRegister("vgprSerial", \
        self.startVgprSerial)
    kStr += self.comment1("VGPRs: %u + %u = %u" \
        % (self.startVgprTmp, self.numVgprTmp, self.totalVgprs) )
    kStr += self.comment1("Occu: %u waves/simd" % self.numWavesPerSimd )


    ########################################
    # SGPR Macros
    ########################################
    kStr += self.comment3("SGPR Assignments")
    kStr += self.macroRegister("sgprKernArgAddress", \
        self.startSgprKernArgAddress)
    kStr += self.macroRegister("sgprWorkGroup0", self.startSgprWorkGroup0)
    kStr += self.macroRegister("sgprWorkGroup1", self.startSgprWorkGroup1)
    kStr += self.macroRegister("sgprAddressC", self.startSgprAddressC)
    kStr += self.macroRegister("sgprStridesC", self.startSgprStridesC)
    kStr += self.macroRegister("sgprAlpha", self.startSgprAlpha)
    if kernel["ProblemType"]["UseBeta"]:
      kStr += self.macroRegister("sgprBeta", self.startSgprBeta)
    kStr += self.macroRegister("sgprSizesSum", self.startSgprSizesSum)
    kStr += self.macroRegister("sgprLoopPadding", self.startSgprLoopPadding)
    kStr += self.macroRegister("sgprStridesA", self.startSgprStridesA)
    kStr += self.macroRegister("sgprStridesB", self.startSgprStridesB)
    kStr += self.macroRegister("sgprSizesFree", self.startSgprSizesFree)
    kStr += self.macroRegister("sgprAddressA", self.startSgprAddressA)
    kStr += self.macroRegister("sgprAddressB", self.startSgprAddressB)
    kStr += self.macroRegister("sgprOffsetC", self.startSgprOffsetC)
    kStr += self.macroRegister("sgprOffsetA", self.startSgprOffsetA)
    kStr += self.macroRegister("sgprOffsetB", self.startSgprOffsetB)
    kStr += self.macroRegister("sgprAddressD", self.startSgprAddressD)
    kStr += self.macroRegister("sgprGlobalReadIncsA", \
        self.startSgprGlobalReadIncsA)
    kStr += self.macroRegister("sgprGlobalReadIncsB", \
        self.startSgprGlobalReadIncsB)
    kStr += self.macroRegister("sgprLoopCounters", self.startSgprLoopCounters)
    kStr += self.comment1("SGPR: %u" % self.totalSgprs)

    ########################################
    # Global Offsets
    ########################################
    for (tensorChar, indices) in [ \
        ("C", range(0, kernel["ProblemType"]["NumDimensionsC"])), \
        ("A", kernel["ProblemType"]["IndexAssignmentsA"]), \
        ("B", kernel["ProblemType"]["IndexAssignmentsB"]) ]:
      kStr += self.comment("Global Offset %s"%tensorChar)
      numDim = len(indices)
      idxChars = []
      for i in indices:
        idxChars.append(self.indexChars[i])
      kStr += ".macro GLOBAL_OFFSET_%s vgprAddr"%tensorChar
      for i in range(0, numDim):
        kStr += " vgprOffset%s" % idxChars[i]
      kStr += " vgprTmp%s" % self.endLine
      # d0
      kStr += inst("v_mov_b32", "v[\\vgprAddr+0]", "v[\\vgprOffset%s]" \
          % idxChars[0], "d0 lower")
      kStr += inst("v_mov_b32", "v[\\vgprAddr+1]", hex(0), "d0 upper" )
      #kStr += dump("v[\\vgprOffset%s]"%idxChars[0])
      #kStr += dump("v[\\vgprOffset%s]"%idxChars[1])
      #kStr += "s_endpgm\n"
      # d1+
      for i in range(1, numDim):
        kStr += inst("v_mul_lo_u32", \
            "v[\\vgprTmp+0]", \
            "v[\\vgprOffset%s]" % idxChars[i],  \
            sgpr("Strides%s+%u"%(tensorChar,i-1)), \
            "mul d%u lower"%i)
        #kStr += dump("v[\\vgprTmp+0]")

        kStr += inst("v_mov_b32", \
            "v[\\vgprTmp+2]", \
            hex(0),  \
            "mul d%u upper"%i)
        kStr += inst("v_addc_u32", \
            "v[\\vgprTmp+1]", \
            "vcc",  \
            hex(0), \
            "v[\\vgprTmp+2]", \
            "vcc",  \
            "mul d%u upper"%i)
        #kStr += dump("v[\\vgprTmp+1]")

        kStr += inst("v_add_i32", \
            "v[\\vgprAddr+0]", \
            "vcc", \
            "v[\\vgprAddr+0]",  \
            "v[\\vgprTmp+0]", \
            "accumulate d%u lower"%i)
        #kStr += dump("v[\\vgprAddr+0]")
        kStr += inst("v_addc_u32", \
            "v[\\vgprAddr+1]", \
            "vcc", \
            "v[\\vgprAddr+1]",  \
            hex(0), \
            "vcc", \
            "accumulate d%u lower"%i)
      kStr += inst("v_lshlrev_b64", \
          "v[\\vgprAddr+0:\\vgprAddr+1]", \
          hex(log2(self.bpe)), \
          "v[\\vgprAddr+0:\\vgprAddr+1]", \
          "offset *= bytes/element")
      #kStr += "s_endpgm\n"
      kStr += ".endm%s" % self.endLine

    ####################################
    # Global Write Macro
    ####################################
    kStr += self.comment3("Global Write")
    kStr += ".macro GLOBAL_WRITE vc0 vc1 d0 d1%s" % self.endLine
    kStr += ".set idx, %u + \\vc0 + \\d0*%u + \\vc1*%u + \\d1*%u*%u %s" \
        % (self.startVgprValuC, \
        kernel["VectorWidth"], \
        kernel["ThreadTile0"], \
        kernel["VectorWidth"], kernel["ThreadTile0"], \
        self.endLine )
    kStr += ".set vgprTmp, %u%s" % ( self.totalVgprs-8, self.endLine)
    #kStr += dump(vgpr(self.globalWriteAddrC+0))
    #kStr += dump(vgpr(self.globalWriteAddrC+1))
    #kStr += inst("v_mov_b32", vgpr(65), sgpr("Alpha"), "")
    #kStr += dump(vgpr(65))
    #kStr += inst("v_mov_b32", vgpr(65), sgpr("Beta"), "")
    #kStr += dump(vgpr(65))
    #kStr += dump("v[idx]")
    # static tmps b/c 
    vgprAddr = self.totalVgprs-6
    vgprValue = self.totalVgprs-7
    kStr += inst("v_mov_b32", vgpr(vgprAddr), sgpr("StridesC"), \
        "%s = StridesC"%vgpr(vgprAddr))
    kStr += inst("v_mov_b32", vgpr(vgprAddr+1), hex(0x0), \
        "%s = 0"%vgpr(vgprAddr+1) )
    #addr += (d1*strideC + d0*VW + vc)*4bytes

    # tmp1 = strideC*(vc1 + d1*16*vw)
    kStr += inst("v_lshlrev_b32", vgpr("Tmp+1"), \
        hex(log2(kernel["SubGroup1"]*kernel["VectorWidth"])), \
        "\\d1", "tmp1 = d1*sg1*VW" )
    kStr += inst("v_add_u32", vgpr("Tmp+1"), "vcc", "\\vc1", vgpr("Tmp+1"), \
        "tmp1 = vc1 + d1*sg1*VW")
    kStr += inst("v_mul_u32_u24", vgpr("Tmp+1"), vgpr("Tmp+1"), vgpr(vgprAddr), \
        "%s = StridesC*(vc1+d1*sg1*VW)"%vgpr("Tmp+1") )

    # tmp0 = c0 + d0*16*vw
    kStr += inst("v_lshlrev_b32", vgpr("Tmp+0"), \
        hex(log2(kernel["SubGroup0"]*kernel["VectorWidth"])), "\\d0", \
        "tmp0 = d0*sg0*VW" )
    kStr += inst("v_add_u32", vgpr("Tmp+0"), "vcc", "\\vc0", vgpr("Tmp+0"), \
        "tmp0 = vc0 + d0*sg0*VW")
    #kStr += dump(vgpr("Tmp"))
    #kStr += "s_endpgm\n"
    kStr += inst("v_add_u32", vgpr(vgprAddr), "vcc", vgpr("Tmp+0"), vgpr("Tmp+1"), \
        "%s = vc0 + d0*sg0*VW + StridesC*(vc1+d1*sg1*VW)"%vgpr(vgprAddr) )

    #kStr += inst("v_lshlrev_b64", vgpr(vgprAddr,2), hex(log2(kernel["SubGroup0"])), \
    #    vgpr(vgprAddr,2), "%s = 16*(strideC1J*(d1*VW+vc1)+d0*VW)"%vgpr(vgprAddr) )

    #kStr += inst("v_add_u32", vgpr(vgprAddr), "vcc", "\\vc0", vgpr(vgprAddr), \
    #    "%s = 16*(StridesC1J*(d1*VW+vc1)+d0*VW)+vc0"%vgpr(vgprAddr) )

    kStr += inst("v_lshlrev_b64", vgpr(vgprAddr,2), hex(log2(self.bpe)), \
        vgpr(vgprAddr,2), "%s = 4*(vc0 + d0*sg0*VW + StridesC*(vc1+d1*sg1*VW) )"%vgpr(vgprAddr) )

    kStr += inst("v_add_u32", vgpr(vgprAddr), "vcc", vgpr(self.globalWriteAddrC+0), \
        vgpr(vgprAddr), "%s = base + (16*(strideC1J*(d1*VW+vc1)+d0*VW)+vc0)*4"%vgpr(vgprAddr) )
    kStr += inst("v_addc_u32", vgpr(vgprAddr+1), "vcc", vgpr(self.globalWriteAddrC+1), \
        vgpr(vgprAddr+1), "vcc", "%s = base + (16*(strideC1J*(d1*VW+vc1)+d0*VW)+vc0)*4"%vgpr(vgprAddr+1))
    kStr += inst("flat_load_dword", vgpr(vgprValue), vgpr(vgprAddr,2), \
        "load C" )
    kStr += inst("s_waitcnt", "vmcnt(0) & lgkmcnt(0)", "wait C" )
    #kStr += dump(vgpr(vgprValue))
    kStr += inst("v_mul_f32", vgpr(vgprValue), sgpr("Beta"), vgpr(vgprValue), \
        "%s = C*beta"%vgpr(vgprValue) )
    #kStr += dump(vgpr(vgprValue))

    kStr += inst("v_mul_f32", "v[idx]", sgpr("Alpha"), "v[idx]", "*= alpha" )
    #kStr += dump("v[idx]")
    kStr += inst("v_add_f32", "v[idx]", vgpr(vgprValue), "v[idx]", \
        "v[idx] = sum*alpha + C*beta" )
    #kStr += dump(vgpr(vgprAddr+0))
    #kStr += dump(vgpr(vgprAddr+1))
    #kStr += dump("v[idx]")
    #kStr += "s_endpgm\n"

    kStr += inst("flat_store_dword", vgpr(vgprAddr,2), "v[idx]", "store C" )
    kStr += ".endm%s"%self.endLine


    ########################################
    # MACs
    kStr += self.comment3("%dx%d thread-tile" \
        % (kernel["ThreadTile0"], kernel["ThreadTile1"]) )
    numMacs = 2 if kernel["PrefetchLocalRead"] else 1
    for m in range(0, numMacs):
      kStr += ".macro MAC_%ux%u" \
          % (kernel["ThreadTile0"], kernel["ThreadTile1"])
      if kernel["PrefetchLocalRead"]:
        kStr += ("" if m==0 else "_BLK")
      kStr += self.endLine
      for b in range(0, kernel["ThreadTile1"]):
        for a in range(0, kernel["ThreadTile0"]):
          cStr = "v[%s+%u+%u*%u]" % ("vgprValuC", a, b, kernel["ThreadTile0"])
          aStr = "v[%s+%u]" \
              % ("vgprValuA" if m==0 else "vgprValuBlkA", a)
          bStr = "v[%s+%u]" \
              % ("vgprValuB" if m==0 else "vgprValuBlkB", b)
          kStr += "v_mac_f32 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine)
          #if a==0 and b==2:
          #  kStr += dump(cStr)
      kStr += ".endm%s" % self.endLine

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
      kStr += ".set strideC" + self.indexChars[i] + ", 1" + self.endLine
    for i in range(firstStride, lastStrideA):
      kStr += ".set strideA" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsA"][i]] \
          + ", 1" + self.endLine
    for i in range(firstStride, lastStrideB):
      kStr += ".set strideB" \
          + self.indexChars[kernel["ProblemType"]["IndexAssignmentsB"][i]] \
          + ", 1" + self.endLine
    kStr += self.endLine

    ####################################
    # scalar macros
    kStr += self.comment("kernel parameter macros")
    vw = kernel["VectorWidth"]
    mt0 = kernel["MacroTile0"]
    mt1 = kernel["MacroTile1"]
    du = kernel["DepthU"]
    if kernel["ProblemType"]["Tensor0"]==0:
      mtA = mt0
      mtB = mt1
    else:
      mtA = mt1
      mtB = mt0
    nlca = kernel["NumLoadsCoalescedA"]
    nlcb = kernel["NumLoadsCoalescedB"]
    nlpa = kernel["NumLoadsPerpendicularA"]
    nlpb = kernel["NumLoadsPerpendicularB"]
    kStr += ".set NLCA, %u%s" % (nlca, self.endLine)
    kStr += ".set NLCB, %u%s" % (nlcb, self.endLine)
    kStr += ".set NLPA, %u%s" % (nlpa, self.endLine)
    kStr += ".set NLPB, %u%s" % (nlpb, self.endLine)

    if kernel["ProblemType"]["TLUA"]:
      lsca = mtA/nlpa
      lspa = du/nlpa
    else:
      lsca = du/nlpa
      lspa = mtA/nlpa
    if kernel["ProblemType"]["TLUB"]:
      lscb = mtB/nlpb
      lspb = du/nlpb
    else:
      lscb = du/nlpb
      lspb = mtB/nlpb

    kStr += ".set LSCA, %u%s" % (lsca, self.endLine)
    kStr += ".set LSPA, %u%s" % (lspa, self.endLine)
    kStr += ".set LSCB, %u%s" % (lscb, self.endLine)
    kStr += ".set LSPB, %u%s" % (lspb, self.endLine)

    kStr += ".set LVCA, %u%s" % (lsca/vw, self.endLine)
    kStr += ".set LVCB, %u%s" % (lscb/vw, self.endLine)
    kStr += ".set LVPA, %u%s" % (lspa/vw, self.endLine)
    kStr += ".set LVPB, %u%s" % (lspb/vw, self.endLine)

    return kStr


  ##############################################################################
  # Function Signature Prefix - DONE
  ##############################################################################
  def functionSignaturePrefix(self, kernel):
    return ""


  ##############################################################################
  # Function Signature - DONE
  ##############################################################################
  def functionSignature(self, kernel ):
    kStr = ""

    kStr += ".hsa_code_object_version 2,0%s" % self.endLine
    kStr += ".hsa_code_object_isa 8, 0, 3, \"AMD\", \"AMDGPU\" %s" % self.endLine
    kStr += ".text%s" % self.endLine
    kStr += ".p2align 8%s" % self.endLine
    kStr += ".amdgpu_hsa_kernel %s%s" % (self.kernelName, self.endLine)
    kStr += "%s:%s" % (self.kernelName, self.endLine)
    kStr += ".amd_kernel_code_t%s" % self.endLine

    kStr += "  is_ptr64 = 1%s" % self.endLine
    kStr += "  enable_sgpr_kernarg_segment_ptr = 1%s" % self.endLine


    kernArgReg = 0
    kernArgReg += 3*self.rpga
    kernArgReg += 1 # alpha
    if kernel["ProblemType"]["UseBeta"]:
      kernArgReg += 1 # beta
    kernArgReg += 3 # offsets
    kernArgReg += kernel["ProblemType"]["NumIndicesC"] # strides
    kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsA"]) # strides
    kernArgReg += len(kernel["ProblemType"]["IndexAssignmentsB"]) # strides
    if not kernel["ProblemType"]["UseInitialStrides"]:
      kernArgReg -= 3 # strides
    kernArgReg += kernel["ProblemType"]["NumIndicesSummation"]
    kernArgReg += kernel["ProblemType"]["NumIndicesC"]
    kernArgReg += self.rpga # debug buffer

    kernArgBytes = kernArgReg * 4 # bytes/reg
    kStr += "  kernarg_segment_byte_size = %u // bytes of kern args%s" \
        % (kernArgBytes, self.endLine)
    # register allocation
    kStr += "  workitem_vgpr_count = %u // vgprs%s" \
        % (self.totalVgprs, self.endLine)
    kStr += "  wavefront_sgpr_count = %u // sgprs%s" \
        % (self.totalSgprs, self.endLine)
    kStr += "  compute_pgm_rsrc1_vgprs = %u // floor((%u-1)/4)%s" \
        % ( (self.totalVgprs-1)/4, self.totalVgprs, self.endLine)
    kStr += "  compute_pgm_rsrc1_sgprs = %u // floor((%u-1)/8)%s" \
        % ( (self.totalSgprs-1)/8, self.totalSgprs, self.endLine)
    # work-group dimensions
    kStr += "  compute_pgm_rsrc2_user_sgpr = 2 // ?%s" % self.endLine
    kStr += "  compute_pgm_rsrc2_tidig_comp_cnt = 0 // 1D wg%s" % self.endLine
    kStr += "  compute_pgm_rsrc2_tgid_x_en = 1 // wg.x%s" % self.endLine
    kStr += "  compute_pgm_rsrc2_tgid_y_en = 1 // wg.y%s" % self.endLine
    if kernel["ProblemType"]["NumIndicesC"] > 2:
      kStr += "  compute_pgm_rsrc2_tgid_z_en = %u // wg.z%s" % (1 if kernel["ProblemType"]["NumIndicesC"] > 2 else 0, self.endLine)
    kStr += "  compute_pgm_rsrc2_lds_size = 1 // ?%s" % self.endLine
    kStr += "  workgroup_group_segment_byte_size = %u // lds bytes%s" \
        % ( kernel["LdsNumElements"] \
        * self.bpe, self.endLine )
    kStr += "  kernarg_segment_alignment = 4%s" % self.endLine
    kStr += "  group_segment_alignment = 4%s" % self.endLine
    kStr += "  private_segment_alignment = 4%s" % self.endLine
    kStr += ".end_amd_kernel_code_t%s" % self.endLine

    return kStr


  ##############################################################################
  # Function Beginning
  ##############################################################################
  def functionSignaturePrefix(self, kernel): return ""
  def functionSignatureSuffix(self, kernel): return ""
  def functionBegin(self, kernel): return ""
  def allocateResources(self, kernel):
    kStr = ""

    # set m0
    kStr += inst("s_mov_b32", "m0", hex(kernel["LdsNumElements"] \
        * self.bpe), "LDS clamp at %u bytes" \
        %(kernel["LdsNumElements"] * self.bpe) )
        
    kStr += inst("v_mov_b32", vgpr("Serial"), vgpr(0), "thread serial id")

    ########################################
    # load kernel args
    # TODO revert to s_load_dwordx2
    kStr += self.comment("Load Kernel Args")
    kernArgOffset = 0
    kStr += inst("s_load_dword", sgpr("AddressD"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr debug" )
    kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("AddressD+1"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr debug" )
    kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("AddressC"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr c" )
    kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("AddressC+1"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr c" )
    kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("AddressA"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr a" )
    kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("AddressA+1"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr a" )
    kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("AddressB"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr b" )
    kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("AddressB+1"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr b" )
    kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("Alpha"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load alpha" )
    kernArgOffset += 1*4
    if kernel["ProblemType"]["UseBeta"]:
      kStr += inst("s_load_dword", sgpr("Beta"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load beta" )
      kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("OffsetC"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset c" )
    kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("OffsetA"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset a" )
    kernArgOffset += 1*4
    kStr += inst("s_load_dword", sgpr("OffsetB"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset b" )
    kernArgOffset += 1*4
    for i in range(0, self.numSgprStridesC):
      kStr += inst("s_load_dword", sgpr("StridesC+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride c %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprStridesA):
      kStr += inst("s_load_dword", sgpr("StridesA+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride a %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprStridesB):
      kStr += inst("s_load_dword", sgpr("StridesB+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride b %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprSizesFree):
      kStr += inst("s_load_dword", sgpr("SizesFree+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load size free %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprSizesSum):
      kStr += inst("s_load_dword", sgpr("SizesSum+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load size free %u"%i )
      kernArgOffset += 1*4
    kStr += inst("s_waitcnt", "lgkmcnt(0)", \
        "wait for %u bytes of kern args" % kernArgOffset )

    # test debug buffer
    #v = self.vgprScratch.checkOut(3)
    #kStr += inst("v_mov_b32", vgpr(v), sgpr("AddressC"), "" )
    #kStr += inst("v_mov_b32", vgpr(v+1), sgpr("AddressC+1"), "" )
    #kStr += inst("v_mov_b32", vgpr(v+2), hex(3), "" )
    #kStr += inst("flat_store_dword", vgpr(v, 2), vgpr(v+2), "debug serial" )
    #kStr += "s_endpgm\n"

    ########################################
    # Debug Buffer
    kStr += self.comment("Debug Buffer")
    nt_log2 = log2(kernel["NumThreads"])
    # TODO: read nwg0 from sgpr
    nwg0 = 1 # num work-groups 0
    self.nipt = 8 # num integers per thread
    v = self.vgprScratch.checkOut(3)
    kStr += inst("v_mov_b32", vgpr(v), "s2", "%s=wg0"%vgpr(v) )
    kStr += inst("v_mov_b32", vgpr(v+1), "s3", "%s=wg1"%vgpr(v+1) )
    kStr += inst("v_mul_lo_u32", vgpr(v+1), vgpr(v+1), hex(nwg0), \
        "%s=wg1*nwg0"%vgpr(v+1) )
    kStr += inst("v_add_i32", vgpr(v), "vcc", vgpr(v), vgpr(v+1), \
        "%s=wg1*nwg0+wg0"%vgpr(v) )
    kStr += inst("v_lshlrev_b32", vgpr(v), nt_log2, vgpr(v), \
        "%s=NT*(wg1*nwg0+wg0)"%vgpr(v) )
    kStr += inst("v_add_i32", vgpr(v), "vcc", vgpr(v), vgpr("Serial"), \
        "%s=tid+NT*(wg1*nwg0+wg0)=serial"%vgpr(v) )
    kStr += inst("v_mul_lo_u32", vgpr(v), hex(self.nipt*4), vgpr(v), \
        "%s=serial*nipt*4"%vgpr(v) )
    kStr += inst("v_mov_b32", vgpr(v+1), 0, "")
    kStr += inst("v_add_i32", vgpr("AddressD"), "vcc", sgpr("AddressD"), \
        vgpr(v), "%s=AddrD* + serial*nipt*4"%vgpr("AddressD") )
    kStr += inst("v_mov_b32", vgpr(v+2), sgpr("AddressD+1"), "%s=AddressD1"%vgpr(v+2) )
    kStr += inst("v_addc_u32", vgpr("AddressD+1"), "vcc", vgpr(v+2), \
        vgpr(v+1), "vcc", "%s=AddrD* + serial*nipt*4"%vgpr("AddressD") )
    self.vgprScratch.checkIn(v)
    #kStr += dump(vgpr("Serial"))
    #kStr += "s_endpgm\n"

    ########################################
    # Apply User Offsets
    kStr += self.comment("User Offsets")
    kStr += inst("s_add_u32", sgpr("AddressC"), sgpr("OffsetC"), \
        sgpr("AddressC"), "addrC += offsetC" )
    kStr += inst("s_mov_b32", sgpr("OffsetC"), 0, "")
    kStr += inst("s_addc_u32", sgpr("AddressC"), sgpr("OffsetC"),\
        sgpr("AddressC"), "addrC += offsetC carry" )
    kStr += inst("s_add_u32", sgpr("AddressA"), sgpr("OffsetA"), \
        sgpr("AddressA"), "addrA += offsetA" )
    kStr += inst("s_mov_b32", sgpr("OffsetA"), 0, "")
    kStr += inst("s_addc_u32", sgpr("AddressA"), sgpr("OffsetA"),\
        sgpr("AddressA"), "addrA += offsetA carry" )
    kStr += inst("s_add_u32", sgpr("AddressB"), sgpr("OffsetB"), \
        sgpr("AddressB"), "addrB += offsetB" )
    kStr += inst("s_mov_b32", sgpr("OffsetB"), 0, "")
    kStr += inst("s_addc_u32", sgpr("AddressB"), sgpr("OffsetB"),\
        sgpr("AddressB"), "addrB += offsetB carry" )
    # now sgpr OffsetC,A,B are freed up for arithmetic
    #kStr += dump(vgpr("Serial"))
    #kStr += "s_endpgm\n"


    return kStr

  ##############################################################################
  # Global Read Addresses: Work-Group - LATER
  ##############################################################################
  def graWorkGroup(self, kernel):
    return self.comment1("  N/A")

  ##############################################################################
  # Global Read Addresses: Subgroup - DONE
  ##############################################################################
  def graSubgroup(self, kernel):
    return self.comment1("  not needed until local read addresses")

  ##############################################################################
  # Global Read Addresses: Tile Assignment A - DONE
  # stores to v1,2
  ##############################################################################
  def graTileAssignmentA(self, kernel):
    kStr = ""
    #kStr += "  unsigned int globalReadOffsetA%s = (serial" % self.tileCharA
    # what register to store these values into
    if self.globalReadCoalesceGroupA:
      if kernel["GlobalReadCoalesceVectorA"]:
        divisorName = "LVCA"
      else:
        divisorName = "LSCA"
    else:
      if kernel["GlobalReadCoalesceVectorA"]:
        divisorName = "LSPA"
      else:
        divisorName = "LVPA"
    divisor = kernel[divisorName]

    if self.globalReadCoalesceGroupA == kernel["ProblemType"]["TLUA"]:
      rReg = self.vgprScratch.checkOut(1) # groA-tile = serial%divisor
      qReg = self.vgprScratch.checkOut(1) # groA-unroll = serial/divisor
      tReg = rReg
      uReg = qReg
      tOpStr = "%"
      uOpStr = "/"
    else:
      qReg = self.vgprScratch.checkOut(1) # groA-tile = serial/divisor
      rReg = self.vgprScratch.checkOut(1) # groA-unroll = serial%divisor
      tReg = qReg
      uReg = rReg
      tOpStr = "/"
      uOpStr = "%"
    tReg2 = self.vgprScratch.checkOut(1)
    kStr += self.comment1("%s = groA-tile = serial%s%s + (wgA*MTA);" \
        % (vgpr(tReg2), tOpStr, divisorName) )
    kStr += self.comment1("%s = groA-unroll = serial%s%s;" \
        % (vgpr(uReg), uOpStr, divisorName) )
    dividendReg = "Serial" # local serial
    tmpVgpr = self.vgprScratch.checkOut(1)
    tmpSgpr = self.startSgprOffsetC
    kStr += staticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)

    if kernel["VectorWidth"] > 1:
      if kernel["GlobalReadCoalesceVectorA"] == kernel["ProblemType"]["TLUA"]:
        kStr += inst("v_lshlrev_b32", vgpr(tReg), log2(kernel["VectorWidth"]), \
            vgpr(tReg), "%s *= VW"%vgpr(tReg) )
      else:
        kStr += inst("v_lshlrev_b32", vgpr(uReg), log2(kernel["VectorWidth"]), \
            vgpr(lReg), "%s *= VW"%vgpr(uReg) )
    kStr += inst("v_lshlrev_b32", vgpr(tmpVgpr), log2(kernel["MacroTileA"]), \
        sgpr("WorkGroup0"), "%s = wgA * MTA"%vgpr(tmpVgpr) )
    #kStr += dump(vgpr(tmpVgpr))
    kStr += inst("v_add_u32", vgpr(tReg2), "vcc", vgpr(tmpVgpr), \
        vgpr(tReg), "groA-tile = serial%s%s*VW + (wgA*MTA)" \
        % (tOpStr, divisorName) )
    self.lwoTA = tReg
    self.tRegA = tReg2
    self.uRegA = uReg
    self.vgprScratch.checkIn(tmpVgpr)
    #kStr += dump(vgpr(tReg2))
    #kStr += dump(vgpr(uReg))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Assignment B - DONE
  # stores to v3,4
  ##############################################################################
  def graTileAssignmentB(self, kernel):
    #kStr += "  unsigned int globalReadOffsetB%s = (serial" % self.tileCharB
    # what register to store these values into
    kStr = ""
    if self.globalReadCoalesceGroupB:
      if kernel["GlobalReadCoalesceVectorB"]:
        divisorName = "LVCB"
      else:
        divisorName = "LSCB"
    else:
      if kernel["GlobalReadCoalesceVectorB"]:
        divisorName = "LSPB"
      else:
        divisorName = "LVPB"
    divisor = kernel[divisorName]

    if self.globalReadCoalesceGroupB == kernel["ProblemType"]["TLUB"]:
      rReg = self.vgprScratch.checkOut(1) # groB-tile = serial%divisor
      qReg = self.vgprScratch.checkOut(1) # groB-unroll = serial/divisor
      tReg = rReg
      uReg = qReg
      tOpStr = "%"
      uOpStr = "/"
    else:
      qReg = self.vgprScratch.checkOut(1) # groB-tile = serial/divisor
      rReg = self.vgprScratch.checkOut(1) # groB-unroll = serial%divisor
      tReg = qReg
      uReg = rReg
      tOpStr = "/"
      uOpStr = "%"
    tReg2 = self.vgprScratch.checkOut(1)
    kStr += self.comment1("%s = groB-tile = serial%s%s + (wgB*MTB);" \
        % (vgpr(tReg2), tOpStr, divisorName) )
    kStr += self.comment1("%s = groB-unroll = serial%s%s;" \
        % (vgpr(uReg), uOpStr, divisorName) )
    dividendReg = "Serial" # local serial
    tmpVgpr = self.vgprScratch.checkOut(1)
    tmpSgpr = self.startSgprOffsetC
    kStr += staticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)

    if kernel["VectorWidth"] > 1:
      if kernel["GlobalReadCoalesceVectorB"] == kernel["ProblemType"]["TLUB"]:
        kStr += inst("v_lshlrev_b32", vgpr(tReg), log2(kernel["VectorWidth"]), \
            vgpr(tReg), "%s *= VW"%vgpr(tReg) )
      else:
        kStr += inst("v_lshlrev_b32", vgpr(uReg), log2(kernel["VectorWidth"]), \
            vgpr(uReg), "%s *= VW"%vgpr(uReg) )
    kStr += inst("v_lshlrev_b32", vgpr(tmpVgpr), log2(kernel["MacroTileB"]), \
        sgpr("WorkGroup1"), "%s = wgB * MTB"%vgpr(tmpVgpr) )
    #kStr += dump(vgpr(tmpVgpr))
    kStr += inst("v_add_u32", vgpr(tReg2), "vcc", vgpr(tmpVgpr), \
        vgpr(tReg), "groB-tile = serial%s%s*VW + (wgB*MTB)" \
        % (tOpStr, divisorName) )
    self.lwoTB = tReg
    self.tRegB = tReg2
    self.uRegB = uReg
    self.vgprScratch.checkIn(tmpVgpr)
    #kStr += dump(vgpr("Serial"))
    #kStr += dump(vgpr(uReg))
    #kStr += "s_endpgm\n"
    return kStr


  ##############################################################################
  # Global Read Addresses: Unroll Assignment A - DONE
  ##############################################################################
  def graUnrollAssignmentA(self, kernel):
    return self.comment1(vgpr(self.uRegA))

  ##############################################################################
  # Global Read Addresses: Unroll Assignment B - DONE
  ##############################################################################
  def graUnrollAssignmentB(self, kernel):
    return self.comment1(vgpr(self.uRegB))

  ##############################################################################
  # Global Read Addresses: Other Free Assignments - LATER
  ##############################################################################
  def graOtherFreeAssignments(self, kernel):
    # LATER: support more dimensions that just batched
    return self.comment1(sgpr("WorkGroup2"))

  ##############################################################################
  # Global Read Addresses: Other Summation Assignments - DONE
  ##############################################################################
  def graOtherSummationAssignments(self, kernel):
    kStr = ""
    for i in range(0,kernel["ProblemType"]["NumIndicesSummation"]-1):
      index = i
      kStr += ".set globalReadOffsetA%s 0%s" \
          % (self.indexChars[index], self.endLine)
      kStr += ".set globalReadOffsetB%s 0%s" \
          % (self.indexChars[index], self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets A - DONE
  ##############################################################################
  def graTileOffsetsA(self, kernel):
    numTileOffsetsA = self.numReadsTileA
    if self.readTileDimComponentsA:
      numTileOffsetsA *= kernel["VectorWidth"]
    self.vgprTileOffsetsA = self.vgprScratch.checkOut(numTileOffsetsA)
    v = self.vgprTileOffsetsA
    stride = "LSCA" if kernel["ProblemType"]["TLUA"] else "LSPA"
    kStr = ""
    if self.readTileDimComponentsA:
      # l=0, s=0
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(self.tRegA), "groA%s_%u_s%u"%(self.tileCharA, 0, 0) )
      # l=0, s>0
      for s in range(1, kernel["VectorWidth"]):
        kStr += inst("v_add_u32", vgpr(v+s), "vcc", 1, \
            vgpr(v+s-1), "groA%s_%u_s%u"%(self.tileCharA, 0, s) )
      for l in range(1, self.numReadsTileA):
        # l>0, s=0
        kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]), "vcc", stride, \
            vgpr(v+(l-1)*kernel["VectorWidth"]), \
            "groA%s_%u_s%u"%(self.tileCharA, l, 0) )
        # l>0, s>0
        for s in range(0, kernel["VectorWidth"]):
          kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]+s), "vcc", \
              1, vgpr(v+l*kernel["VectorWidth"]+(s-1)), \
              "groA%s_%u_s%u"%(self.tileCharA, 0, s) )
    else:
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(self.tRegA), "groA%s_%u"%(self.tileCharA, 0) )
      for l in range(1, self.numReadsTileA):
        kStr += inst("v_add_u32", vgpr(v+l), "vcc", stride, \
            vgpr(v+l-1), "groA%s_%u"%(self.tileCharA, l) )
    return kStr

  ##############################################################################
  # Global Read Addresses: Tile Offsets B - DONE
  ##############################################################################
  def graTileOffsetsB(self, kernel):
    numTileOffsetsB = self.numReadsTileB
    if self.readTileDimComponentsB:
      numTileOffsetsB *= kernel["VectorWidth"]
    self.vgprTileOffsetsB = self.vgprScratch.checkOut(numTileOffsetsB)
    v = self.vgprTileOffsetsB
    stride = "LSCB" if kernel["ProblemType"]["TLUB"] else "LSPB"
    kStr = ""
    if self.readTileDimComponentsB:
      # l=0, s=0
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(self.tRegB), "groB%s_%u_s%u"%(self.tileCharB, 0, 0) )
      # l=0, s>0
      for s in range(1, kernel["VectorWidth"]):
        kStr += inst("v_add_u32", vgpr(v+s), "vcc", 1, \
            vgpr(v+s-1), "groB%s_%u_s%u"%(self.tileCharB, 0, s) )
      for l in range(1, self.numReadsTileB):
        # l>0, s=0
        kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]), "vcc", stride, \
            vgpr(v+(l-1)*kernel["VectorWidth"]), \
            "groB%s_%u_s%u"%(self.tileCharB, l, 0) )
        # l>0, s>0
        for s in range(0, kernel["VectorWidth"]):
          kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]+s), "vcc", \
              1, vgpr(v+l*kernel["VectorWidth"]+(s-1)), \
              "groB%s_%u_s%u"%(self.tileCharB, 0, s) )
    else:
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(self.tRegB), "groB%s_%u"%(self.tileCharB, 0) )
      for l in range(1, self.numReadsTileB):
        kStr += inst("v_add_u32", vgpr(v+l), "vcc", stride, \
            vgpr(v+l-1), "groB%s_%u"%(self.tileCharB, l) )
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Offsets A - DONE
  ##############################################################################
  def graUnrollOffsetsA(self, kernel):
    numUnrollOffsetsA = self.numReadsUnrollA
    if self.readUnrollDimComponentsA:
      numUnrollOffsetsA *= kernel["VectorWidth"]
    self.vgprUnrollOffsetsA = self.vgprScratch.checkOut(numUnrollOffsetsA)
    v = self.vgprUnrollOffsetsA
    kStr = ""
    stride = ("LSPA" if kernel["ProblemType"]["TLUA"] else "LSCA")
    if self.readUnrollDimComponentsA:
      # l=0, s=0
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(self.uRegA), "groA%s_%u_s%u"%(self.unrollChar, 0, 0) )
      # l=0, s>0
      for s in range(1, kernel["VectorWidth"]):
        kStr += inst("v_add_u32", vgpr(v+s), "vcc", 1, \
            vgpr(v+s-1), "groA%s_%u_s%u"%(self.unrollChar, 0, s) )
      for l in range(1, self.numReadsUnrollA):
        # l>0, s=0
        kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]), "vcc", stride, \
            vgpr(v+(l-1)*kernel["VectorWidth"]), \
            "groA%s_%u_s%u"%(self.unrollChar, l, 0) )
        # l>0, s>0
        for s in range(0, kernel["VectorWidth"]):
          kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]+s), "vcc", \
              1, vgpr(v+l*kernel["VectorWidth"]+(s-1)), \
              "groA%s_%u_s%u"%(self.unrollChar, 0, s) )
    else:
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(self.uRegA), "groA%s_%u"%(self.unrollChar, 0) )
      for l in range(1, self.numReadsUnrollA):
        kStr += inst("v_add_u32", vgpr(v+l), "vcc", stride, \
            vgpr(v+l-1), "groA%s_%u"%(self.unrollChar, l) )
    return kStr


  ##############################################################################
  # Global Read Addresses: Unroll Offsets B - DONE
  ##############################################################################
  def graUnrollOffsetsB(self, kernel):
    numUnrollOffsetsB = self.numReadsUnrollB
    if self.readUnrollDimComponentsB:
      numUnrollOffsetsB *= kernel["VectorWidth"]
    self.vgprUnrollOffsetsB = self.vgprScratch.checkOut(numUnrollOffsetsB)
    v = self.vgprUnrollOffsetsB
    kStr = ""
    stride = ("LSPB" if kernel["ProblemType"]["TLUB"] else "LSCB")
    if self.readUnrollDimComponentsB:
      # l=0, s=0
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(self.uRegB), "groB%s_%u_s%u"%(self.unrollChar, 0, 0) )
      # l=0, s>0
      for s in range(1, kernel["VectorWidth"]):
        kStr += inst("v_add_u32", vgpr(v+s), "vcc", 1, \
            vgpr(v+s-1), "groB%s_%u_s%u"%(self.unrollChar, 0, s) )
      for l in range(1, self.numReadsUnrollB):
        # l>0, s=0
        kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]), "vcc", stride, \
            vgpr(v+(l-1)*kernel["VectorWidth"]), \
            "groB%s_%u_s%u"%(self.unrollChar, l, 0) )
        # l>0, s>0
        for s in range(0, kernel["VectorWidth"]):
          kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]+s), "vcc", \
              1, vgpr(v+l*kernel["VectorWidth"]+(s-1)), \
              "groB%s_%u_s%u"%(self.unrollChar, 0, s) )
    else:
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(self.uRegB), "groB%s_%u"%(self.unrollChar, 0) )
      for l in range(1, self.numReadsUnrollB):
        kStr += inst("v_add_u32", vgpr(v+l), "vcc", stride, \
            vgpr(v+l-1), "groB%s_%u"%(self.unrollChar, l) )
    return kStr

  ##############################################################################
  # Global Read Addresses: Branch A - SKIP
  ##############################################################################
  def graBranchA(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Branch B - SKIP
  ##############################################################################
  def graBranchB(self, kernel):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A - SKIP
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
  # Global Read Addresses: Shift B - SKIP
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
  # Global Read Addresses: Final Offsets A - DONE
  ##############################################################################
  def graFinalOffsetsA(self, kernel):
    tVW = 1
    tVS = 0
    uVW = 1
    uVS = 0
    if self.readTileDimComponentsA:
      tVW = kernel["VectorWidth"]
      tVS = 1
    elif self.readUnrollDimComponentsA:
      uVW = kernel["VectorWidth"]
      uVS = 1
    tmp = self.vgprScratch.checkOut(2)
    kStr = ""
    graIdxA = 0
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numReadVectorComponentsA):
          # vgpr assignments
          if kernel["ProblemType"]["TLUA"]:
            vgprTile   = self.vgprTileOffsetsA   + para*tVW + s*tVS
            vgprUnroll = self.vgprUnrollOffsetsA + perp*uVW + s*uVS
          else:
            vgprTile   = self.vgprTileOffsetsA   + perp*tVW + s*tVS
            vgprUnroll = self.vgprUnrollOffsetsA + para*uVW + s*uVS
          # global offset macro
          kStr += "GLOBAL_OFFSET_A vgprGlobalReadAddrA+%u"%graIdxA
          for i in kernel["ProblemType"]["IndexAssignmentsA"]:
            if i < kernel["ProblemType"]["NumIndicesC"]:
              if i == kernel["ProblemType"]["TileA"]:
                kStr += ", %2u" % vgprTile
              else: # just a group index
                kStr += ", %s" % sgpr("WorkGroup+%u"%i)
            else: # summation index
              if i == kernel["ProblemType"]["IndexUnroll"]:
                kStr += ", %2u" % vgprUnroll
              else:
                kStr += "globalReadOffsetA%s" % self.indexChars[i]
          kStr += ", %u // gROA_%u_%u%s%s" % (tmp, para, perp, \
              "_%u"%s if self.numReadVectorComponentsA>1 else "", self.endLine)
          graIdxA += self.rpga
    #kStr += dump(vgpr("GlobalReadAddrA+0"))
    #kStr += "s_endpgm\n"

    self.vgprScratch.checkIn(self.vgprTileOffsetsA)
    self.vgprScratch.checkIn(self.vgprUnrollOffsetsA)
    self.vgprScratch.checkIn(tmp)
    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets B - DONE
  ##############################################################################
  def graFinalOffsetsB(self, kernel):
    tVW = 1
    tVS = 0
    uVW = 1
    uVS = 0
    if self.readTileDimComponentsB:
      tVW = kernel["VectorWidth"]
      tVS = 1
    elif self.readUnrollDimComponentsB:
      uVW = kernel["VectorWidth"]
      uVS = 1
    tmp = self.vgprScratch.checkOut(2)
    kStr = ""
    graIdxB = 0
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numReadVectorComponentsB):
          # vgpr assignments
          if kernel["ProblemType"]["TLUB"]:
            vgprTile   = self.vgprTileOffsetsB   + para*tVW + s*tVS
            vgprUnroll = self.vgprUnrollOffsetsB + perp*uVW + s*uVS
          else:
            vgprTile   = self.vgprTileOffsetsB   + perp*tVW + s*tVS
            vgprUnroll = self.vgprUnrollOffsetsB + para*uVW + s*uVS
          # global offset macro
          kStr += "GLOBAL_OFFSET_B vgprGlobalReadAddrB+%u"%graIdxB
          for i in kernel["ProblemType"]["IndexAssignmentsB"]:
            if i < kernel["ProblemType"]["NumIndicesC"]:
              if i == kernel["ProblemType"]["TileB"]:
                kStr += ", %2u" % vgprTile
              else: # just a group index
                kStr += ", %s" % sgpr("WorkGroup+%u"%i)
            else: # summation index
              if i == kernel["ProblemType"]["IndexUnroll"]:
                kStr += ", %2u" % vgprUnroll
              else:
                kStr += "globalReadOffsetB%s" % self.indexChars[i]
          kStr += ", %u // gROB_%u_%u%s%s" % (tmp, para, perp, \
              "_%u"%s if self.numReadVectorComponentsB>1 else "", self.endLine)
          graIdxB += self.rpga

    #kStr += dump(vgpr("GlobalReadAddrB+0"))
    #kStr += "s_endpgm\n"
    self.vgprScratch.checkIn(self.vgprTileOffsetsB)
    self.vgprScratch.checkIn(self.vgprUnrollOffsetsB)
    self.vgprScratch.checkIn(tmp)
    return kStr

  ##############################################################################
  # Global Read Addresses: Apply User Offsets - DONE
  ##############################################################################
  def graApplyUserOffsets(self, kernel):
    kStr = ""
    kStr += self.comment1("moved earlier")
    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses A - DONE
  ##############################################################################
  def graAddressesA(self, kernel):
    kStr = ""
    graIdxA = 0
    tmp = self.vgprScratch.checkOut(2)
    kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("AddressA+0"), "" )
    kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("AddressA+1"), "" )
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numReadVectorComponentsA):

          comment = "gRAA_%u_%u%s = addrA+grOA_%u_%u%s" % (para, perp, \
              "_%u"%s if self.numReadVectorComponentsA>1 else "", para, perp, \
              "_%u"%s if self.numReadVectorComponentsA>1 else "", )
          #kStr += dump(vgpr("GlobalReadAddrA+0"))
          #kStr += dump(vgpr("GlobalReadAddrA+1"))
          #kStr += dump(vgpr(tmp+0))
          #kStr += dump(vgpr(tmp+1))

          kStr += inst("v_add_i32", \
              vgpr("GlobalReadAddrA+%u+0"%graIdxA), \
              "vcc", \
              vgpr("GlobalReadAddrA+%u+0"%graIdxA),  \
              vgpr(tmp+0), \
              comment+" (lower)")
          kStr += inst("v_addc_u32", \
              vgpr("GlobalReadAddrA+%u+1"%graIdxA), \
              "vcc", \
              vgpr("GlobalReadAddrA+%u+1"%graIdxA), \
              vgpr(tmp+1), \
              "vcc", \
              comment+" (upper)")
          #kStr += dump(vgpr("GlobalReadAddrA+0"))
          #kStr += dump(vgpr("GlobalReadAddrA+1"))
        graIdxA += self.rpga
    #kStr += "s_endpgm\n"
    self.vgprScratch.checkIn(tmp)
    return kStr

  ##############################################################################
  # Global Read Addresses: Addresses B - DONE
  ##############################################################################
  def graAddressesB(self, kernel):
    kStr = ""
    graIdxB = 0
    tmp = self.vgprScratch.checkOut(2)
    kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("AddressB+0"), "" )
    kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("AddressB+1"), "" )
    #kStr += dump(vgpr(tmp+0))
    #kStr += dump(vgpr(tmp+1))
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numReadVectorComponentsB):

          comment = "gRAB_%u_%u%s = addrB+grOB_%u_%u%s" % (para, perp, \
              "_%u"%s if self.numReadVectorComponentsB>1 else "", para, perp, \
              "_%u"%s if self.numReadVectorComponentsB>1 else "", )
          #kStr += dump(vgpr("GlobalReadAddrB+0"))
          #kStr += dump(vgpr("GlobalReadAddrB+1"))
          #kStr += dump(vgpr(tmp+0))
          #kStr += dump(vgpr(tmp+1))

          kStr += inst("v_add_i32 ", \
              vgpr("GlobalReadAddrB+%u+0"%graIdxB), \
              "vcc", \
              vgpr("GlobalReadAddrB+%u+0"%graIdxB),  \
              vgpr(tmp+0), \
              comment+" (lower)")
          kStr += inst("v_addc_u32", \
              vgpr("GlobalReadAddrB+%u+1"%graIdxB), \
              "vcc", \
              vgpr("GlobalReadAddrB+%u+1"%graIdxB), \
              vgpr(tmp+1), \
              "vcc", \
              comment+" (upper)")
          #kStr += dump(vgpr("GlobalReadAddrB+0"))
          #kStr += dump(vgpr("GlobalReadAddrB+1"))
        graIdxB += self.rpga
    #kStr += "s_endpgm\n"
    self.vgprScratch.checkIn(tmp)
    return kStr

  ##############################################################################
  # Global Read Addresses: Increments A - DONE
  ##############################################################################
  def graIncrementsA(self, kernel, loopIdx):
    kStr = ""
    if loopIdx==kernel["ProblemType"]["NumIndicesSummation"]-1:
      kStr += inst("s_mul_i32", sgpr("GlobalReadIncsA+0"), \
          hex(kernel["DepthU"]*4), sgpr("StridesA"), \
          "incr = stride*%u*4bytes"%kernel["DepthU"] )
      kStr += inst("s_addc_u32", \
          sgpr("GlobalReadIncsA+1"), \
          hex(0), \
          hex(0), \
          "(carry)")
    else:
      printExit("NumIndicesSummation=%u not yet supported in assembly" \
          % kernel["ProblemType"]["NumIndicesSummation"] )
    #tmp = self.vgprScratch.checkOut(2)
    #kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("GlobalReadIncsA+0"), "" )
    #kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("GlobalReadIncsA+1"), "" )
    #kStr += dump(vgpr(tmp+0))
    #kStr += dump(vgpr(tmp+1))
    #self.vgprScratch.checkIn(tmp)
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Global Read Addresses: Increments B - DONE
  ##############################################################################
  def graIncrementsB(self, kernel, loopIdx):
    kStr = ""
    if loopIdx==kernel["ProblemType"]["NumIndicesSummation"]-1:
      kStr += inst("s_mul_i32", sgpr("GlobalReadIncsB+0"), \
          hex(kernel["DepthU"]*4), sgpr("StridesB"), \
          "incr = stride*%u*4bytes"%kernel["DepthU"] )
      kStr += inst("s_addc_u32", \
          sgpr("GlobalReadIncsB+1"), \
          hex(0), \
          hex(0), \
          "(carry)")
    else:
      printExit("NumIndicesSummation=%u not yet supported in assembly" \
          % kernel["ProblemType"]["NumIndicesSummation"] )
    #tmp = self.vgprScratch.checkOut(2)
    #kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("GlobalReadIncsB+0"), "" )
    #kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("GlobalReadIncsB+1"), "" )
    #kStr += dump(vgpr(tmp+0))
    #kStr += dump(vgpr(tmp+1))
    #self.vgprScratch.checkIn(tmp)
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment A - DONE
  ##############################################################################
  def lwaTileAssignmentA(self, kernel):
    return self.comment1("lwaTileA = %s" % vgpr(self.lwoTA))

  ##############################################################################
  # Local Write Addresses: Tile Assignment B - DONE
  ##############################################################################
  def lwaTileAssignmentB(self, kernel):
    return self.comment1("lwaTileB = %s" % vgpr(self.lwoTB))

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A - DONE
  ##############################################################################
  def lwaUnrollAssignmentA(self, kernel):
    return self.comment1("lwaUnrollA = %s" % vgpr(self.uRegA))

  ##############################################################################
  # Local Write Addresses: Unroll Assignment B - DONE
  ##############################################################################
  def lwaUnrollAssignmentB(self, kernel):
    return self.comment1("lwaUnrollB = %s" % vgpr(self.uRegB))

  ##############################################################################
  # Local Write Addresses: First Offset A - DONE
  ##############################################################################
  def lwaFirstOffsetA(self, kernel):
    kStr = ""
    "lwFOA = lwA%s + lwA%s*MT%s" \
        % (self.tileCharA, self.unrollChar, self.tileCharA)
    kStr += inst("v_mul_u32_u24", \
        vgpr("LocalWriteAddrA"), \
        hex(kernel["MacroTileA"]), \
        vgpr(self.uRegA), \
        "lwA%s*MTA"%self.unrollChar)
    #kStr += dump(vgpr("LocalWriteAddrA"))
    kStr += inst("v_add_u32", \
        vgpr("LocalWriteAddrA"), \
        "vcc", \
        vgpr(self.lwoTA), \
        vgpr("LocalWriteAddrA"), \
        "lwFOA = lwA%s + lwA%s*MT%s" \
        % (self.tileCharA, self.unrollChar, self.tileCharA) )
    kStr += inst("v_lshlrev_b32", \
        vgpr("LocalWriteAddrA"), \
        hex(log2(self.bpe)), \
        vgpr("LocalWriteAddrA"), \
        " *= bytes/element" )
    #kStr += dump(vgpr("LocalWriteAddrA"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Write Addresses: First Offset B - DONE
  ##############################################################################
  def lwaFirstOffsetB(self, kernel):
    kStr = ""
    "lwFOB = lwB%s + lwB%s*MT%s" \
        % (self.tileCharB, self.unrollChar, self.tileCharB)
    kStr += inst("v_mul_u32_u24", \
        vgpr("LocalWriteAddrB"), \
        hex(kernel["MacroTileB"]), \
        vgpr(self.uRegB), \
        "lwB%s*MTB"%self.unrollChar)
    #kStr += dump(vgpr("LocalWriteAddrB"))
    kStr += inst("v_add_u32", \
        vgpr("LocalWriteAddrB"), \
        "vcc", \
        vgpr(self.lwoTB), \
        vgpr("LocalWriteAddrB"), \
        "lwFOB = lwB%s + lwB%s*MT%s" \
        % (self.tileCharB, self.unrollChar, self.tileCharB) )
    kStr += inst("v_lshlrev_b32", \
        vgpr("LocalWriteAddrB"), \
        hex(log2(self.bpe)), \
        vgpr("LocalWriteAddrB"), \
        " *= bytes/element" )
    #kStr += dump(vgpr("LocalWriteAddrB"))
    kStr += inst("v_add_u32", \
        vgpr("LocalWriteAddrB"), \
        "vcc", \
        hex(kernel["LdsOffsetB"]*self.bpe), \
        vgpr("LocalWriteAddrB"), \
        "lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_B=%u*%u" % (self.tileCharB, \
        self.unrollChar, self.tileCharB, kernel["LdsOffsetB"], \
        self.bpe) )
    #kStr += dump(vgpr("LocalWriteAddrB"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A - DONE
  # initially assume write offsets fit into 8-bits
  ##############################################################################
  def lwaFinalOffsetsA(self, kernel):
    self.localWriteOffsetsA = []
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numWriteVectorComponentsA):
          lsca = para * kernel["LSCA"]
          lspa = perp * kernel["LSPA"]
          if self.writeTileDimComponentsA:
            lsca += s
          elif self.writeUnrollDimComponentsA:
            lspa += s
          if kernel["ProblemType"]["TLUA"]:
            lspa *= kernel["MacroTileA"]
          else:
            lsca *= kernel["MacroTileA"]
          offset = lspa + lsca
          offset /= self.localWriteInstructionA.offsetMultiplier
          self.localWriteOffsetsA.append(offset)

          kStr += "%slwoA_%u_%u%s = (%s%d*%s)" \
              % (self.commentPrefix, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else ""), \
              (("%u + "%s) if self.writeTileDimComponentsA else ""), \
              para, "LSCA" )
          if not kernel["ProblemType"]["TLUA"]:
            kStr += "*MT%s" % (self.tileCharA)
          kStr += " + (%s%d*%s)" % (
              (("%u + "%s) if self.writeUnrollDimComponentsA else ""), perp, \
              "LSPA")
          if kernel["ProblemType"]["TLUA"]:
            kStr += "*MT%s" % (self.tileCharA)
          kStr += " = %u%s%s" % (offset, self.commentSuffix, self.endLine)
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets B - DONE
  # initially assume write offsets fit into 8-bits
  ##############################################################################
  def lwaFinalOffsetsB(self, kernel):
    self.localWriteOffsetsB = []
    kStr = ""
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numWriteVectorComponentsB):
          lscb = para * kernel["LSCB"]
          lspb = perp * kernel["LSPB"]
          if self.writeTileDimComponentsB:
            lscb += s
          elif self.writeUnrollDimComponentsB:
            lspb += s
          if kernel["ProblemType"]["TLUB"]:
            lspb *= kernel["MacroTileB"]
          else:
            lscb *= kernel["MacroTileB"]
          offset = lspb + lscb
          offset /= self.localWriteInstructionB.offsetMultiplier
          self.localWriteOffsetsB.append(offset)

          kStr += "%slwoB_%u_%u%s = (%s%d*%s)" \
              % (self.commentPrefix, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsB \
              or self.writeUnrollDimComponentsB) else ""), \
              (("%u + "%s) if self.writeTileDimComponentsB else ""), \
              para, "LSCB" )
          if not kernel["ProblemType"]["TLUB"]:
            kStr += "*MT%s" % (self.tileCharB)
          kStr += " + (%s%d*%s)" % (
              (("%u + "%s) if self.writeUnrollDimComponentsB else ""), perp, \
              "LSPB")
          if kernel["ProblemType"]["TLUB"]:
            kStr += "*MT%s" % (self.tileCharB)
          kStr += " = %u%s%s" % (offset, self.commentSuffix, self.endLine)
    return kStr

  ##############################################################################
  # Local Write Addresses: Declare Addresses A - DONE
  ##############################################################################
  def lwaDeclareAddressesA(self, kernel):
    return self.comment1("N/A")

  ##############################################################################
  # Local Write Addresses: Declare Addresses B - DONE
  ##############################################################################
  def lwaDeclareAddressesB(self, kernel):
    return self.comment1("N/A")

  ##############################################################################
  # Local Read Addresses: Tile Assignment A - DONE
  ##############################################################################
  def lraTileAssignmentA(self, kernel):
    kStr = ""
    kStr += "%slr%s = serial %% SG%s%s%s" \
        % (self.commentPrefix, self.tileChar0, self.tileChar0, \
        self.commentSuffix, self.endLine)

    divisor = kernel["SubGroup0"]
    qReg = self.vgprScratch.checkOut(1) # quotient
    rReg = self.vgprScratch.checkOut(1) # remainder
    dividendReg = "Serial" # local serial
    tmpVgpr = self.vgprScratch.checkOut(1)
    tmpSgpr = self.startSgprOffsetC
    kStr += staticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    #kStr += dump(vgpr(rReg))
    #kStr += dump(vgpr(qReg))
    self.lroA = rReg
    #kStr += dump(vgpr(self.lroA))
    self.lroB = qReg
    self.vgprScratch.checkIn(tmpVgpr)
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment B - DONE
  ##############################################################################
  def lraTileAssignmentB(self, kernel):
    kStr = ""
    kStr += "%slr%s = (serial / SG%s) %% SG%s%s%s" \
        % (self.commentPrefix, self.tileChar1, self.tileChar0, \
        self.tileChar1, self.commentSuffix, self.endLine)
    divisor = kernel["SubGroup1"]
    qReg = self.vgprScratch.checkOut(1) # quotient
    rReg = self.vgprScratch.checkOut(1) # remainder
    dividendReg = self.lroB
    tmpVgpr = self.vgprScratch.checkOut(1)
    tmpSgpr = self.startSgprOffsetC
    kStr += staticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    #kStr += dump(vgpr(rReg))
    #kStr += dump(vgpr(qReg))
    #kStr += "s_endpgm\n"
    self.vgprScratch.checkIn(self.lroB) # old
    self.lroB = rReg
    #kStr += dump(vgpr(self.lroB))
    self.vgprScratch.checkIn(qReg)
    self.vgprScratch.checkIn(tmpVgpr)
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset A - DONE
  ##############################################################################
  def lraFinalOffsetA(self, kernel):
    kStr = ""
    divisor = kernel["NumThreads"]
    qReg = self.vgprScratch.checkOut(1) # quotient
    rReg = self.vgprScratch.checkOut(1) # remainder
    dividendReg = 0
    tmpVgpr = self.vgprScratch.checkOut(1)
    tmpSgpr = self.startSgprOffsetC
    kStr += staticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    sgid = qReg
    #kStr += dump(vgpr(self.lroA))
    kStr += inst("s_mov_b32", \
        sgpr(tmpSgpr), \
        hex(kernel["MacroTile0"]), \
        "MT0" )
    kStr += inst("v_mul_lo_u32", \
        vgpr(sgid), \
        vgpr(sgid), \
        sgpr(tmpSgpr), \
        "sgid*sgid*MT0" )
    #kStr += dump(vgpr(sgid))
    if kernel["VectorWidth"] > 1:
      kStr += inst("v_lshlrev_b32", \
          vgpr(self.lroA), \
          log2(kernel["VectorWidth"]), \
          vgpr(self.lroA), \
          "lroA *= VW" )
      #kStr += dump(vgpr(self.lroA))
    kStr += inst("v_add_u32", \
        vgpr("LocalReadAddrA"), \
        "vcc", \
        vgpr(sgid), \
        vgpr(self.lroA), \
        "o = lroA*VW+sgid*MT0" )
    kStr += inst("v_lshlrev_b32", \
        vgpr("LocalReadAddrA"), \
        hex(log2(self.bpe)), \
        vgpr("LocalReadAddrA"), \
        "*= bytes/element" )
    #kStr += dump(vgpr("LocalReadAddrA"))
    #kStr += "s_endpgm\n"

    self.vgprScratch.checkIn(tmpVgpr)
    self.vgprScratch.checkIn(qReg)
    self.vgprScratch.checkIn(rReg)
    #kStr += "  unsigned int localReadOffsetA = lr%s*VECTOR_WIDTH + sgId*MT%s;%s" \
    #    % ( self.tileChar0, self.tileChar0, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset B - DONE
  ##############################################################################
  def lraFinalOffsetB(self, kernel):
    kStr = ""
    divisor = kernel["NumThreads"]
    qReg = self.vgprScratch.checkOut(1) # quotient
    rReg = self.vgprScratch.checkOut(1) # remainder
    dividendReg = 0
    tmpVgpr = self.vgprScratch.checkOut(1)
    tmpSgpr = self.startSgprOffsetC
    kStr += staticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    sgid = qReg
    #kStr += dump(vgpr(self.lroB))
    kStr += inst("s_mov_b32", \
        sgpr(tmpSgpr), \
        hex(kernel["MacroTile1"]), \
        "MT1" )
    kStr += inst("v_mul_lo_u32", \
        vgpr(sgid), \
        vgpr(sgid), \
        sgpr(tmpSgpr), \
        "sgid*sgid*MT1" )
    #kStr += dump(vgpr(sgid))
    if kernel["VectorWidth"] > 1:
      kStr += inst("v_lshlrev_b32", \
          vgpr(self.lroB), \
          log2(kernel["VectorWidth"]), \
          vgpr(self.lroB), \
          "lroB *= VW" )
      #kStr += dump(vgpr(self.lroB))
    kStr += inst("v_add_u32", \
        vgpr("LocalReadAddrB"), \
        "vcc", \
        vgpr(sgid), \
        vgpr(self.lroB), \
        "o = lroB*VW+sgid*MT1" )
    kStr += inst("v_lshlrev_b32", \
        vgpr("LocalReadAddrB"), \
        hex(log2(self.bpe)), \
        vgpr("LocalReadAddrB"), \
        "*= bytes/element" )
    #kStr += dump(vgpr("LocalReadAddrB"))
    #kStr += "s_endpgm\n"
    self.vgprScratch.checkIn(tmpVgpr)
    self.vgprScratch.checkIn(qReg)
    self.vgprScratch.checkIn(rReg)
    #kStr += "  unsigned int localReadOffsetB = lr%s*VECTOR_WIDTH + sgId*(MT%s+PAD) + LDS_OFFSET_B;%s" \
    #    % (self.tileChar1, self.tileChar1, self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Declare Addresses A - DONE
  ##############################################################################
  def lraDeclareAddressesA(self, kernel):
    return self.comment1("N/A")

  ##############################################################################
  # Local Read Addresses: Declare Addresses B - DONE
  ##############################################################################
  def lraDeclareAddressesB(self, kernel):
    kStr = ""
    kStr += inst("v_add_u32", \
        vgpr("LocalReadAddrB+0"), \
        "vcc", \
        hex(kernel["LdsOffsetB"]*self.bpe), \
        vgpr("LocalReadAddrB+0"), \
        " += LdsOffsetB (lower)")
    #kStr += dump(vgpr("LocalReadAddrB"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Declare Loop Num Iterations - DONE
  ##############################################################################
  def declareLoopNumIter(self, kernel):
    kStr = ""
    for i in range(0, self.numVgprValuC):
      kStr += inst("v_mov_b32", vgpr("ValuC+%u"%i), hex(0), "")
    return kStr


  ##############################################################################
  # Calculate Loop Num Iter - DONE
  ##############################################################################
  def calculateLoopNumIter(self, kernel, loopIdx):
    kStr = ""
    tmp = self.vgprScratch.checkOut(1)
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if tailLoop:
      kStr += "%snumIter%s = (((size%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU);%s" \
          % (self.indent, self.unrollChar, self.unrollChar, self.endLine)
      if kernel["GlobalSplitU"] > 1:
        # SKIP
        printExit("Asm GSU>1 not yet supported")
        kStr += "%sif (gsuSumIdx != numIterPerWgRemainder) {%s" \
            % (self.indent, self.endLine)
        kStr += "%s  numIter%s = 0;%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%s}%s" % (self.indent, self.endLine)
    else:
      if loopIdx == self.unrollIdx:
        #kStr += inst("v_mov_b32", vgpr(tmp), \
        #    sgpr("SizesSum+0"), "" )
        #kStr += dump(vgpr(tmp))

        kStr += inst("s_lshr_b32", \
            sgpr("LoopCounters+%u"%loopIdx), \
            sgpr("SizesSum+%u"%loopIdx), \
            log2(kernel["DepthU"]), \
            "numIter%s = size%s / DU"%(loopChar, loopChar) )

        #kStr += inst("v_mov_b32", vgpr(tmp), \
        #    sgpr("LoopCounters+0"), "" )
        #kStr += dump(vgpr(tmp))

        kStr += inst("s_sub_u32", \
            sgpr("LoopCounters+%u"%loopIdx), \
            hex(0), \
            sgpr("LoopCounters+%u"%loopIdx), \
            "counter%s = -size%s"%(loopChar, loopChar) )
      else:
        # SKIP
        printExit("Asm GSU>1 not yet supported")
        kStr += "%snumIter%s = size%s;" \
            % (self.indent, loopChar, loopChar)

      if loopIdx == self.unrollIdx and kernel["GlobalSplitU"] > 1:
        # SKIP
        printExit("Asm GSU>1 not yet supported")
        kStr += "%sunsigned int numIterMyWg = numIter%s / GLOBAL_SPLITU;%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%sunsigned int numIterPerWgRemainder = numIter%s %% GLOBAL_SPLITU;%s" \
            % (self.indent, self.unrollChar, self.endLine)
        kStr += "%sif (gsuSumIdx < numIterPerWgRemainder) {%s" \
            % (self.indent, self.endLine)
        kStr += "%s  numIterMyWg++;%s" % (self.indent, self.endLine)
        kStr += "%s}%s" % (self.indent, self.endLine)
        kStr += "%snumIter%s = numIterMyWg;%s" \
            % (self.indent, self.unrollChar, self.endLine)

    kStr += inst("v_mov_b32", vgpr(tmp), \
        sgpr("LoopCounters+0"), "" )
    #kStr += dump(vgpr(tmp))

    #kStr += "s_endpgm\n"
    self.vgprScratch.checkIn(tmp)
    return kStr

  ##############################################################################
  # Open Loop - DONE
  ##############################################################################
  def openLoop(self, kernel, loopIdx):
    kStr = ""
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    loopLabelBegin = self.getLabel("LoopBegin%s"%loopChar)
    loopLabelEnd = self.getLabel("LoopEnd%s"%loopChar)
    kStr += "label_%04u:%s" % (loopLabelBegin, self.endLine)
    return kStr


  ##############################################################################
  # Close Loop - DONE
  ##############################################################################
  def closeLoop(self, kernel, loopIdx):
    kStr = ""
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    loopLabelBegin = self.getLabel("LoopBegin%s"%loopChar)
    loopLabelEnd = self.getLabel("LoopEnd%s"%loopChar)
    endCounter = -1 if kernel["PrefetchGlobalRead"] else 0

    kStr += inst("s_add_u32", \
        sgpr("LoopCounters+%u"%loopIdx), \
        sgpr("LoopCounters+%u"%loopIdx), \
        hex(1), \
        "counter%s++"%(loopChar) )
    kStr += inst("s_cmp_eq_i32", \
        sgpr("LoopCounters+%u"%loopIdx), \
        hex(endCounter), \
        "counter%s==0"%(loopChar) )
    kStr += inst("s_cbranch_scc1 label_%04u"%loopLabelEnd, \
        "exit Loop%s"%loopChar )
    kStr += inst("s_branch label_%04u"%loopLabelBegin, \
        "restart Loop%s"%loopChar )
    kStr += "label_%04u:%s" % (loopLabelEnd, self.endLine)
    return kStr

  ##############################################################################
  # End Summation
  ##############################################################################
  def endSummation(self):
    self.vgprScratch = ScratchRegisters(self.startVgprValuA, \
        self.startVgprSerial - self.startVgprValuA)
    return ""

  ##############################################################################
  # MAC Iteration - DONE
  ##############################################################################
  def macIter(self, kernel, black):
    kStr = ""
    kStr += "MAC_%ux%u" % (kernel["ThreadTile0"],kernel["ThreadTile1"])
    if black:
      kStr += "_BLK"
    kStr += self.endLine
    return kStr

  ##############################################################################
  # At Least 1 Unroll - SKIP
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
  # Global Read: Increment A - DONE
  ##############################################################################
  def globalReadIncrementA(self, kernel, loopIdx):
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    graIdxA = 0
    tmp = self.vgprScratch.checkOut(1)
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numReadVectorComponentsA):
          kStr += inst("v_mov_b32 ", \
              vgpr(tmp), \
              sgpr("GlobalReadIncsA+%u+0"%graIdxA), \
              "vgpr GlobalReadIncsA")
          kStr += inst("v_add_i32 ", \
              vgpr("GlobalReadAddrA+%u+0"%graIdxA), \
              "vcc", \
              vgpr("GlobalReadAddrA+%u+0"%graIdxA),  \
              vgpr(tmp), \
              "gra += incA%s (lower)"%loopChar)
          kStr += inst("v_mov_b32 ", \
              vgpr(tmp), \
              sgpr("GlobalReadIncsA+%u+1"%graIdxA), \
              "vgpr GlobalReadIncsA")
          kStr += inst("v_addc_u32", \
              vgpr("GlobalReadAddrA+%u+1"%graIdxA), \
              "vcc", \
              vgpr("GlobalReadAddrA+%u+1"%graIdxA), \
              vgpr(tmp), \
              "vcc", \
              "gra += incA%s (upper)"%loopChar)
          graIdxA += self.rpga
    self.vgprScratch.checkIn(tmp)
    #kStr += dump(vgpr("GlobalReadAddrA+0"))
    #kStr += dump(vgpr("GlobalReadAddrA+1"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Global Read: Increment B - DONE
  ##############################################################################
  def globalReadIncrementB(self, kernel, loopIdx):
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    graIdxB = 0
    tmp = self.vgprScratch.checkOut(1)
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numReadVectorComponentsB):
          kStr += inst("v_mov_b32 ", \
              vgpr(tmp), \
              sgpr("GlobalReadIncsB+%u+0"%graIdxB), \
              "vgpr GlobalReadIncsB")
          kStr += inst("v_add_i32 ", \
              vgpr("GlobalReadAddrB+%u+0"%graIdxB), \
              "vcc", \
              vgpr("GlobalReadAddrB+%u+0"%graIdxB),  \
              vgpr(tmp), \
              "gra += incB%s (lower)"%loopChar)
          kStr += inst("v_mov_b32 ", \
              vgpr(tmp), \
              sgpr("GlobalReadIncsB+%u+1"%graIdxB), \
              "vgpr GlobalReadIncsB")
          kStr += inst("v_addc_u32", \
              vgpr("GlobalReadAddrB+%u+1"%graIdxB), \
              "vcc", \
              vgpr("GlobalReadAddrB+%u+1"%graIdxB), \
              vgpr(tmp), \
              "vcc", \
              "gra += incB%s (upper)"%loopChar)
          graIdxB += self.rpga
    self.vgprScratch.checkIn(tmp)
    #kStr += dump(vgpr("GlobalReadAddrB+0"))
    #kStr += dump(vgpr("GlobalReadAddrB+1"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Global Read: Do It A - DONE
  ##############################################################################
  def globalReadDoA(self, kernel, guardK):
    kStr = ""
    graIdxA = 0
    g2lIdxA = 0
    loadWidth = self.globalReadInstructionA.totalWidth
    #kStr += dump(vgpr("GlobalReadAddrA+0"))
    #kStr += dump(vgpr("GlobalReadAddrA+1"))
    #kStr += "s_endpgm\n"
    for perp in range(0, kernel["NumLoadsPerpendicularA"]):
      for para in range(0, kernel["NumLoadsCoalescedA"]):
        for s in range(0, self.numReadVectorComponentsA):
          kStr += self.globalReadInstructionA.toString( \
              (vgpr("G2LA+%u"%g2lIdxA, loadWidth), \
              vgpr("GlobalReadAddrA+%u"%graIdxA,2)), \
              "G -> Reg %u_%u%s"%(para, perp, \
              "_%u"%s if self.numReadVectorComponentsA>1 else "") )
          #kStr += "s_endpgm\n"
          graIdxA += self.rpga
          g2lIdxA += loadWidth
    kStr += self.wait(0,0)
    #kStr += dump(vgpr("GlobalReadAddrA+0"))
    #kStr += dump(vgpr("GlobalReadAddrA+1"))
    #kStr += dump(vgpr("G2LA+0"))
    #kStr += dump(vgpr("G2LA+1"))
    #kStr += "s_endpgm\n"
    return kStr
    # SKIP branches
    """
          #kStr += "%sa_%u_%u%s = " % (self.indent, para, perp, \
          #    ((".%s"%self.vectorComponents[s]) if (self.readTileDimComponentsA\
          #    or self.readUnrollDimComponentsA) else "") )
          # SKIP branches and tail loop guarded loads
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
          #kStr += "*globalReadA_%u_%u%s;%s" % (para, perp, \
          #    (("_s%u"%s) if (self.readTileDimComponentsA \
          #    or self.readUnrollDimComponentsA) else ""), self.endLine)
    """

  ##############################################################################
  # Global Gead: Do It B - DONE
  ##############################################################################
  def globalReadDoB(self, kernel, guardK):
    kStr = ""
    graIdxB = 0
    g2lIdxB = 0
    loadWidth = self.globalReadInstructionB.totalWidth
    #kStr += dump(vgpr("GlobalReadAddrB+0"))
    #kStr += dump(vgpr("GlobalReadAddrB+1"))
    #kStr += "s_endpgm\n"
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numReadVectorComponentsB):
          kStr += self.globalReadInstructionB.toString( \
              (vgpr("G2LB+%u"%g2lIdxB, loadWidth), \
              vgpr("GlobalReadAddrB+%u"%graIdxB, 2)), \
              "G -> Reg %u_%u%s"%(para, perp, \
              "_%u"%s if self.numReadVectorComponentsB>1 else "") )
          graIdxB += self.rpga
          g2lIdxB += loadWidth
    #kStr += dump(vgpr("GlobalReadAddrB+0"))
    #kStr += "s_endpgm\n"
    # RESUME
    # SKIP branches
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets A - DONE
  ##############################################################################
  def localWriteSwapOffsetsA(self, kernel):
    kStr = ""
    kStr += inst("v_xor_b32", \
        vgpr("LocalWriteAddrA"), \
        hex(kernel["LdsOffsetA_Blk"]*self.bpe), \
        vgpr("LocalWriteAddrA"), \
        "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets B - DONE
  ##############################################################################
  def localWriteSwapOffsetsB(self, kernel):
    kStr = ""
    kStr += inst("v_xor_b32", \
        vgpr("LocalWriteAddrB"), \
        hex(kernel["LdsOffsetA_Blk"]*self.bpe), \
        vgpr("LocalWriteAddrB"), \
        "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Write: Reset Offsets A - SKIP
  # used for global-read + tail-loop to reset to writing in red
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
  # Local Write: Reset Offsets B - SKIP
  # used for global-read + tail-loop to reset to writing in red
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
  # Local Write: Init Pointers A - DONE
  ##############################################################################
  def localWriteInitPointersA(self, kernel):
    return self.comment1("N/A")

  ##############################################################################
  # Local Write: Init Pointers B - DONE
  ##############################################################################
  def localWriteInitPointersB(self, kernel):
    return self.comment1("N/A")



  ##############################################################################
  # Local Write: Do It A - DONE
  ##############################################################################
  def localWriteDoA(self, kernel):
    kStr = ""
    instruction = self.localWriteInstructionA
    numBlocks = instruction.numBlocks
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    totalWrites = len(self.localWriteOffsetsA)/numOffsets
    g2lIdx = 0
    graIdx = 0

    for graIdx in range(0, totalWrites):

      paramList = []
      paramList.append(vgpr("LocalWriteAddrA"))
      for blockIdx in range(0, numBlocks):
        if blockWidth == 1:
          paramList.append(vgpr("G2LA+%u"%g2lIdx))
        else:
          paramList.append( vgpr("G2LA+%u"%g2lIdx,blockWidth))
      for oIdx in range(0, numOffsets):
        paramList.append(self.localWriteOffsetsA[graIdx*numOffsets+oIdx])

      paramTuple = tuple(paramList)
      comment = "Reg -> L %u"%graIdx
      kStr += self.localWriteInstructionA.toString(paramTuple, comment)
      graIdx += 1
      g2lIdx += blockWidth

    #kStr += dump(vgpr("LocalWriteAddrA"))
    #kStr += dump(vgpr("G2LA+0"))
    #kStr += dump(vgpr("G2LA+1"))
    #kStr += "s_endpgm\n"


    """
          kStr += "%s*localWriteA_%u_%u%s = a_%u_%u%s;%s" \
              % (self.indent, para, perp, \
              (("_s%u"%s) if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "" ), \
              para, perp, \
              ((".%s"%self.vectorComponents[s]) \
              if (self.writeTileDimComponentsA \
              or self.writeUnrollDimComponentsA) else "" ), \
              self.endLine)
    """


    """
    loadWidth = self.globalReadInstructionB.totalWidth
    for perp in range(0, kernel["NumLoadsPerpendicularB"]):
      for para in range(0, kernel["NumLoadsCoalescedB"]):
        for s in range(0, self.numReadVectorComponentsB):
          kStr += self.globalReadInstructionB.toString( \
              (vgpr("G2LB+%u:G2LB+%u"%(g2lIdxB, g2lIdxB+loadWidth-1)), \
              vgpr("GlobalReadAddrB+%u+0:GlobalReadAddrB+%u+1" \
              % (graIdxB,graIdxB))), \
              "G -> Reg %u_%u%s"%(para, perp, \
              "_%u"%s if self.numReadVectorComponentsB>1 else "") )
          graIdxB += self.rpga
          g2lIdxB += loadWidth
    """




    return kStr

  ##############################################################################
  # Local Write: Do It B - DONE
  ##############################################################################
  def localWriteDoB(self, kernel):
    kStr = ""
    instruction = self.localWriteInstructionB
    numBlocks = instruction.numBlocks
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    totalWrites = len(self.localWriteOffsetsB)/numOffsets
    g2lIdx = 0
    graIdx = 0

    for graIdx in range(0, totalWrites):

      paramList = []
      paramList.append(vgpr("LocalWriteAddrB"))
      for blockIdx in range(0, numBlocks):
        if blockWidth == 1:
          paramList.append(vgpr("G2LB+%u"%g2lIdx))
        else:
          paramList.append( vgpr("G2LB+%u"%g2lIdx,blockWidth))
      for oIdx in range(0, numOffsets):
        paramList.append(self.localWriteOffsetsB[graIdx*numOffsets+oIdx])

      paramTuple = tuple(paramList)
      comment = "Reg -> L %u"%graIdx
      kStr += self.localWriteInstructionB.toString(paramTuple, comment)
      graIdx += 1
      g2lIdx += blockWidth
    #kStr += dump(vgpr("LocalWriteAddrB"))
    #kStr += dump(vgpr("G2LB+0"))
    #kStr += dump(vgpr("G2LB+1"))

    ########################################
    # dump lds state
    if False:
      kStr += self.comment("dump lds state")
      kStr += inst("s_barrier", "" )
      kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
      tmp = self.vgprScratch.checkOut(1)
      tmpAddr = self.vgprScratch.checkOut(1)
      kStr += inst("v_lshlrev_b32", \
          vgpr(tmpAddr), \
          hex(log2(self.bpe)), \
          vgpr("Serial"), \
          "dump lds")
      kStr += inst("ds_read_b32", \
          vgpr(tmp), \
          vgpr(tmpAddr) + " offset:%u"%(256*4), \
          "dump lds")
      kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
      kStr += dump(vgpr(tmp))
      kStr += "s_endpgm\n"

    return kStr
    """
    return ""
    kStr = ""
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
    return kStr
    """

  ##############################################################################
  # Local Read: Swap Offsets A - DONE
  ##############################################################################
  def localReadSwapOffsetsA(self, kernel):
    kStr = ""
    kStr += inst("v_xor_b32", \
        vgpr("LocalReadAddrA"), \
        hex(kernel["LdsOffsetA_Blk"]*self.bpe), \
        vgpr("LocalReadAddrA"), \
        "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Read: Wwap Offsets B - DONE
  ##############################################################################
  def localReadSwapOffsetsB(self, kernel):
    kStr = ""
    kStr += inst("v_xor_b32", \
        vgpr("LocalReadAddrB"), \
        hex(kernel["LdsOffsetA_Blk"]*self.bpe), \
        vgpr("LocalReadAddrB"), \
        "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets A - DONE
  # x % n == n & (n-1) for n power of 2
  ##############################################################################
  def localReadResetOffsetsA(self, kernel):
    kStr = ""
    if self.localReadInstructionA.numOffsets == 1:
      self.localReadOffsetA = 0
      kStr += self.comment1("N/A")
    else:
      kStr += inst("v_and_b32", \
          vgpr("LocalReadAddrA"), \
          hex(kernel["LdsOffsetA_Blk"]*self.bpe-1), \
          vgpr("LocalReadAddrA"), \
          "reset Red,Blk -> Red")
    #kStr += "%slocalReadOffsetA %%= LDS_OFFSET_BLK;%s" \
    #    % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets B - DONE
  ##############################################################################
  def localReadResetOffsetsB(self, kernel):
    kStr = ""
    if self.localReadInstructionB.numOffsets == 1:
      self.localReadOffsetB = 0
      kStr += self.comment1("N/A")
    else:
      kStr += inst("v_and_b32", \
          vgpr("LocalReadAddrB"), \
          hex(kernel["LdsOffsetA_Blk"]*self.bpe-1), \
          vgpr("LocalReadAddrB"), \
          "reset Red,Blk -> Red")
    return kStr

  ##############################################################################
  # Local Read: Init Pointers A - DONE
  ##############################################################################
  def localReadInitPointersA(self, kernel):
    kStr = ""
    if self.localReadInstructionA.numOffsets == 1:
      self.localReadOffsetA = 0
      kStr += self.comment1("N/A")
    else:
      kStr += inst("v_and_b32", \
          vgpr("LocalReadAddrA"), \
          hex(kernel["LdsOffsetA_Blk"]*self.bpe-1), \
          vgpr("LocalReadAddrA"), \
          "reset Red,Blk -> Red")
    #kStr += "%slocalReadOffsetA %%= LDS_OFFSET_BLK;%s" \
    #    % (self.indent, self.endLine)
    return kStr

  ##############################################################################
  # Local Read: Init Pointers B - DONE
  ##############################################################################
  def localReadInitPointersB(self, kernel):
    kStr = ""
    if self.localReadInstructionB.numOffsets == 1:
      self.localReadOffsetB = 0
      kStr += self.comment1("N/A")
    else:
      kStr += inst("v_and_b32", \
          vgpr("LocalReadAddrB"), \
          hex(kernel["LdsOffsetA_Blk"]*self.bpe-1), \
          vgpr("LocalReadAddrB"), \
          "reset Red,Blk -> Red")
    return kStr

  ##############################################################################
  # Local Read: Increment A - DONE
  ##############################################################################
  def localReadIncA(self, kernel):
    kStr = ""
    if self.localReadInstructionA.numOffsets == 1:
      self.localReadOffsetA += kernel["LocalSplitU"]*kernel["MacroTile0"]
      kStr += self.comment1("N/A")
    else:
      inc = kernel["LocalSplitU"]*kernel["MacroTile0"]
      kStr += inst("v_add_i32", \
          vgpr("LocalReadAddrA"), \
          "vcc", \
          vgpr("LocalReadAddrA"), \
          hex(inc), \
          "lrA += %u"%inc )
    return kStr

  ##############################################################################
  # Local Read: Increment B - DONE
  ##############################################################################
  def localReadIncB(self, kernel):
    kStr = ""
    if self.localReadInstructionB.numOffsets == 1:
      self.localReadOffsetB += kernel["LocalSplitU"]*kernel["MacroTile1"]
      kStr += self.comment1("N/A")
    else:
      inc = kernel["LocalSplitU"]*kernel["MacroTile1"]
      kStr += inst("v_add_i32", \
          vgpr("LocalReadAddrB"), \
          "vcc", \
          vgpr("LocalReadAddrB"), \
          hex(inc), \
          "lrB += %u"%inc )
    return kStr

  ##############################################################################
  # Local Read: Do It A - DONE
  ##############################################################################
  def localReadDoA(self, kernel, black):
    kStr = ""
    instruction = self.localReadInstructionA
    numBlocks = instruction.numBlocks
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    offsetMultiplier = 1 # instruction.offsetMultiplier
    print "lrda oM: %u" % offsetMultiplier
    totalReads = (kernel["ThreadTile0"]/kernel["VectorWidth"]) / numOffsets
    valuIdx = 0
    for lrIdx in range(0, totalReads):
      paramList = []
      if blockWidth == 1:
        paramList.append(vgpr("Valu%sA+%u"%("Blk" if black else "", valuIdx)))
      else:
        paramList.append( vgpr("Valu%sA+%u"%("Blk" if black else "", valuIdx), \
            blockWidth))
      paramList.append(vgpr("LocalReadAddrA"))
      for oIdx in range(0, numOffsets):
        paramList.append((kernel["SubGroup0"]*(lrIdx*numOffsets+oIdx)*kernel["VectorWidth"] \
            + self.localReadOffsetA)*self.bpe/offsetMultiplier)
      paramTuple = tuple(paramList)
      comment = "L -> Reg %u"%lrIdx
      kStr += instruction.toString(paramTuple, comment)
      valuIdx += blockWidth
    return kStr

  ##############################################################################
  # Local Read: Do It B - DONE
  ##############################################################################
  def localReadDoB(self, kernel, black):
    kStr = ""
    instruction = self.localReadInstructionB
    numBlocks = instruction.numBlocks
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    offsetMultiplier = 1 # instruction.offsetMultiplier
    totalReads = (kernel["ThreadTile1"]/kernel["VectorWidth"]) / numOffsets
    valuIdx = 0
    for lrIdx in range(0, totalReads):
      paramList = []
      if blockWidth == 1:
        paramList.append(vgpr("Valu%sB+%u"%("Blk" if black else "", valuIdx)))
      else:
        paramList.append( vgpr("Valu%sB+%u"%("Blk" if black else "", valuIdx), \
            blockWidth))
      paramList.append(vgpr("LocalReadAddrB"))
      for oIdx in range(0, numOffsets):
        paramList.append((kernel["SubGroup1"]*(lrIdx*numOffsets+oIdx)*kernel["VectorWidth"] \
            + self.localReadOffsetB)*self.bpe/offsetMultiplier)
      paramTuple = tuple(paramList)
      comment = "L -> Reg %u"%lrIdx
      kStr += instruction.toString(paramTuple, comment)
      valuIdx += blockWidth
    return kStr


  ##############################################################################
  # Shift Vector Components d0 - SKIP
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
  # Shift Vectors Components d1 - SKIP
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
  # Complex Declare Tmp Registers - SKIP
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
  # LocalSplitU: Local Write - SKIP
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
  # LocalSplitU: Local Read - SKIP
  ##############################################################################
  def localSplitULocalRead(self, kernel):
    kStr = ""
    for i in range(0, kernel["NumVectorsPerThread"]):
      kStr += "  rC[%3u] = localLocalSplitU[serial+%u*NUM_THREADS];%s" \
          % (i, i, self.endLine)
    kStr += self.endLine
    return kStr

  ##############################################################################
  # LocalSplitU: Reduction - SKIP
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
  # LocalSplitU: Global Write Indices - SKIP
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
  # LocalSplitU: Global Write - SKIP
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
          kStr += "  if (globalC%s + %u*CPSV < size%s) {" \
              % (self.tileChar1, b, self.tileChar1)

        kStr += "  TYPE_MAC_WRITE( C[ GLOBAL_C( (%s)" % self.uint64Str
        for i in range(0, kernel["ProblemType"]["NumIndicesC"]):
          kStr += " globalC%s" % self.indexChars[i]
          if i == kernel["ProblemType"]["Index0"] and kernel["VectorWidth"]>1:
            kStr += " + %u" %s
          if i == kernel["ProblemType"]["Index1"]:
            kStr += " + %u*CPSV" %b
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
  # Not LocalSplitU: Global Write Indices - DONE
  ##############################################################################
  def notLocalSplitUGlobalWriteIndices(self, kernel):
    kStr = ""
    tmp = self.vgprScratch.checkOut(2)
    kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("AddressC+0"), "" )
    kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("AddressC+1"), "" )
    #kStr += dump(vgpr(tmp+0))
    #kStr += dump(vgpr(tmp+1))
    # 3 scratch sgprs
    s0 = self.startSgprAddressA
    s1 = s0+1
    s2 = self.startSgprAddressB
    s3 = s2+1
    #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("WorkGroup1"), "" )
    #kStr += dump(vgpr(tmp))

    # work-group offset
    kStr += inst("s_mul_i32", \
        sgpr(s0), \
        hex(kernel["MacroTile0"]*4), \
        sgpr("WorkGroup0"), \
        "%s = wg0*MT0*4"%sgpr(s0))
    kStr += inst("s_mul_i32", \
        sgpr(s1), \
        hex(kernel["MacroTile1"]*4), \
        sgpr("WorkGroup1"), \
        "%s = wg1*MT1"%sgpr(s1))
    kStr += inst("s_mul_i32", \
        sgpr(s1), \
        sgpr("StridesC+0"), \
        sgpr(s1), \
        "%s = wg1*MT1*4*StrideC0"%sgpr(s1))

    #kStr += inst("v_mov_b32", vgpr(tmp), sgpr(s1), "" )
    #kStr += dump(vgpr(tmp))

    if False: # add address C here
      kStr += inst("s_add_u32", \
          sgpr(s2), \
          sgpr("AddressC+0"), \
          sgpr(s0), \
          "%s = C + wg0*MT0*4 (lower)"%sgpr(s2))
      kStr += inst("s_addc_u32", \
          sgpr(s3), \
          hex(0), \
          sgpr("AddressC+1"), \
          "%s = C + wg0*MT0*4 (upper)"%sgpr(s3))
      kStr += inst("s_add_u32", \
          sgpr(s2), \
          sgpr("AddressC+0"), \
          sgpr(s1), \
          "%s = C + wg0*MT0*4 + wg1*MT1*4*StrideC0 (lower)"%sgpr(s2))
      kStr += inst("s_addc_u32", \
          sgpr(s3), \
          hex(0), \
          sgpr("AddressC+1"), \
          "%s = C + wg0*MT0*4 + wg1*MT1*4*StrideC0 (upper)"%sgpr(s3))
    else: # add address C later
      kStr += inst("s_add_u32", \
          sgpr(s2), \
          sgpr(s1), \
          sgpr(s0), \
          "%s = wg0*MT0*4 + wg1*MT1*4*StrideC0 (lower)"%sgpr(s2))
      kStr += inst("s_addc_u32", \
          sgpr(s3), \
          hex(0), \
          hex(0), \
          "%s = wg0*MT0*4 + wg1*MT1*4*StrideC0 (upper)"%sgpr(s3))

    # thread offset = work-group offset
    kStr += inst("v_mov_b32", \
        vgpr(self.globalWriteAddrC+0), \
        sgpr(s2), \
        "move work-group offset into vgpr (lower)")
    kStr += inst("v_mov_b32", \
        vgpr(self.globalWriteAddrC+1), \
        sgpr(s3), \
        "move work-group offset into vgpr (upper)")
    #kStr += dump(vgpr(self.globalWriteAddrC+0))
    #kStr += dump(vgpr(self.globalWriteAddrC+1))

    # thread id 0,1
    self.local01 = self.vgprScratch.checkOut(2)
    dividendReg = "Serial" # local serial
    rReg = self.local01 + 0
    qReg = self.local01 + 1
    tmpVgpr = self.vgprScratch.checkOut(1)
    tmpSgpr = s0
    divisor = kernel["SubGroup0"]
    kStr += staticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    tid0 = rReg
    tid1 = qReg
    kStr += inst("v_lshlrev_b32", \
        vgpr(tid0), \
        log2(self.bpe*kernel["VectorWidth"]), \
        vgpr(tid0), \
        "*= %u VW*bytes/element"%(self.bpe*kernel["VectorWidth"]))
    kStr += inst("v_lshlrev_b32", \
        vgpr(tid1), \
        log2(self.bpe*kernel["VectorWidth"]), \
        vgpr(tid1), \
        "*= %u bytes/element"%(self.bpe*kernel["VectorWidth"]))

    kStr += inst("v_mul_lo_u32", \
        vgpr(tid1), \
        sgpr("StridesC+0"), \
        vgpr(tid1), \
        "%s = tid1*StridesC"%vgpr(tid1))
    kStr += inst("v_add_u32", \
        vgpr(self.globalWriteAddrC+0), \
        "vcc", \
        vgpr(tid1), \
        vgpr(self.globalWriteAddrC+0), \
        "C += tid1*StridesC (lower)")
    kStr += inst("v_addc_u32", \
        vgpr(self.globalWriteAddrC+1), \
        "vcc", \
        hex(0), \
        vgpr(self.globalWriteAddrC+1), \
        "vcc", \
        "C += tid1*StridesC (upper)")
    kStr += inst("v_add_u32", \
        vgpr(self.globalWriteAddrC+0), \
        "vcc", \
        vgpr(tid0), \
        vgpr(self.globalWriteAddrC+0), \
        "C += tid0 (lower)")
    kStr += inst("v_addc_u32", \
        vgpr(self.globalWriteAddrC+1), \
        "vcc", \
        hex(0), \
        vgpr(self.globalWriteAddrC+1), \
        "vcc", \
        "C += tid0 (upper)")
    #kStr += dump(vgpr(self.globalWriteAddrC+0))
    #kStr += dump(vgpr(self.globalWriteAddrC+1))

    # thread offset += addressC
    kStr += inst("v_add_u32", \
        vgpr(self.globalWriteAddrC+0), \
        "vcc", \
        sgpr("AddressC+0"), \
        vgpr(self.globalWriteAddrC+0), \
        "%s += C (lower)"%vgpr(self.globalWriteAddrC))
    kStr += inst("v_mov_b32", vgpr(tmp), sgpr("AddressC+1"), "" )
    kStr += inst("v_addc_u32", \
        vgpr(self.globalWriteAddrC+1), \
        "vcc", \
        vgpr(tmp), \
        vgpr(self.globalWriteAddrC+1), \
        "vcc", \
        "%s += C (upper)"%vgpr(self.globalWriteAddrC+1))
    #kStr += dump(vgpr(self.globalWriteAddrC+0))
    #kStr += dump(vgpr(self.globalWriteAddrC+1))
    #kStr += "s_endpgm\n"

    """
    kStr += "GLOBAL_OFFSET_C( vgprGlobalReadAddrA+%u"%graIdxA
      for i in kernel["ProblemType"]["NumIndicesC"]:
        if i == kernel["ProblemType"]["Dim0"]:
          kStr += ", %2u" % vgprDim0
        if i == kernel["ProblemType"]["Dim1"]:
          kStr += ", %2u" % vgprDim1
        else: # just a group index
          kStr += ", %s" % sgpr("WorkGroup+%u"%i)
        kStr += ")%s" % self.endLine
    """


    """
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
    """
    return kStr

  ##############################################################################
  # Not LocalSplitU: Global Write - DONE
  ##############################################################################
  def notLocalSplitUGlobalWrite(self, kernel):
    kStr = ""
    kStr += self.comment1("GLOBAL_WRITE vc0 vc1 tt0 tt1%s" % (self.endLine) )
    #kStr += "GLOBAL_WRITE 0 0 1 1%s" % (self.endLine)
    #kStr += "s_endpgm\n"
    for tt1 in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for tt0 in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for vc1 in range(0, kernel["VectorWidth"]):
          for vc0 in range(0, kernel["VectorWidth"]):
            kStr += "GLOBAL_WRITE %u %u %u %u%s" % (vc0, vc1, tt0, tt1, self.endLine)
    return kStr

  ##############################################################################
  # Function End - DONE
  ##############################################################################
  def functionEnd(self, kernel):
    return inst("s_endpgm", "End Kernel")

  ##############################################################################
  # Function Suffix - DONE
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
  # Kernel Body Prefix - DONE
  ##############################################################################
  def kernelBodyPrefix(self, kernel):
    return ""

  ##############################################################################
  # Kernel Body Suffix - DONE
  ##############################################################################
  def kernelBodySuffix(self, kernel):
    return ""

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
  # WaitCnt- DONE
  ##############################################################################
  def wait(self, lgkmcnt, vmcnt):
    kStr = ""
    kStr += "s_waitcnt "
    if lgkmcnt >= 0:
      kStr += "lgkmcnt(%u)"%lgkmcnt
    if lgkmcnt >= 0 and vmcnt >= 0:
      kStr += " & "
    if vmcnt >= 0:
      kStr += "vmcnt(%u)"%vmcnt
    kStr += self.endLine
    return kStr

  ##############################################################################
  #
  #   Beta-Only Kernel
  #
  ##############################################################################

  ##############################################################################
  # Function Signature - SKIP
  ##############################################################################
  def functionSignatureBetaOnly(self, kernel):
    kStr = ""
    return kStr

  ##############################################################################
  # Kernel Body Beta-Only - SKIP
  ##############################################################################
  def kernelBodyBetaOnly(self, kernel):
    kStr = ""
    return kStr
  


################################################################################
# Helper Functions
################################################################################

########################################
# Store to Debug Buffer - DONE
########################################
def dump(vgprStore):
  kStr = ""
  kStr += inst("flat_store_dword", vgpr("AddressD", 2), vgprStore, "debug store" )
  kStr += inst("v_add_i32", vgpr("AddressD"), "vcc", vgpr("AddressD"), \
      hex(4), "debug inc" )
  return kStr

########################################
# Format Instruction - DONE
########################################
def inst(*args):
  params = args[0:len(args)-1]
  comment = args[len(args)-1]
  formatting = "%s"
  if len(params) > 1:
    formatting += " %s"
  for i in range(0, len(params)-2):
    formatting += ", %s"
  instStr = formatting % (params)
  line = "%-50s // %s\n" % (instStr, comment)
  return line

########################################
# Format GPRs - DONE
########################################
def gpr(*args):
  gprType = args[0]
  args = args[1]
  if isinstance(args[0], int):
    if len(args) == 1:
      return "%s%u"%(gprType, args[0])
    elif len(args) == 2:
      if args[1] == 1:
        return "%s%u"%(gprType, args[0])
      else:
        return "%s[%u:%u]"%(gprType, args[0], args[0]+args[1]-1)
  if isinstance(args[0], str):
    if len(args) == 1:
      return "%s[%sgpr%s]"%(gprType, gprType, args[0])
    elif len(args) == 2:
      if args[1] == 1:
        return "%s[%sgpr%s]"%(gprType, gprType, args[0])
      else:
        return "%s[%sgpr%s:%sgpr%s+%u]"%(gprType, gprType, args[0], \
            gprType, args[0], args[1]-1)
def vgpr(*args):
  return gpr("v", args)
def sgpr(*args):
  return gpr("s", args)

########################################
# Log 2 - DONE
########################################
def log2(x):
  return int(log(x, 2) + 0.5)

########################################
# Divide & Remainder
# quotient register, remainder register, dividend register, divisor, tmps
########################################
def staticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, \
    doRemainder=True):
  kStr = ""
  if ((divisor & (divisor - 1)) == 0): # pow of 2
    divisor_log2 = log2(divisor)
    kStr += inst("v_lshrrev_b32", vgpr(qReg), divisor_log2, vgpr(dReg), \
        "%s = %s / %u"%(vgpr(qReg), vgpr(dReg), divisor) )
    #kStr += dump(vgpr(qReg))
    if doRemainder:
      kStr += inst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), \
          "%s = %s %% %u"%(vgpr(rReg), vgpr(dReg), divisor) )
      #kStr += dump(vgpr(rReg))
  elif (((divisor/3) & ((divisor/3) - 1)) == 0): # 3 * pow of 2
    tmp = 32 + log2(divisor/3)
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), "0xaaaaaaab", "")
    kStr += inst("v_mul_hi_u32", vgpr(tmpVgpr+1), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+0), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("v_lshrrev_b64", vgpr(tmpVgpr,2), tmp, vgpr(tmpVgpr,2), "")
    kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr), vgpr(tmpVgpr), divisor, "")
    if doRemainder:
      kStr += inst("v_sub_u32", vgpr(rReg), "vcc", vgpr(dReg), vgpr(tmpVgpr), "")
  else:
    printExit("KernelWriterAssembly::divmod doesn't support %u" % divisor)
  return kStr

def staticDivide(qReg, dReg, divisor, tmpVgpr, tmpSgpr):
  rReg = -1 # unused
  kStr = staticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, False)
  return kStr

def staticRemainder(rReg, dReg, divisor, tmpVgpr, tmpSgpr):
  qReg = self.vgprScratch.checkOut(1)
  kStr = staticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr)
  self.vgprScratch.checkIn(qReg)
  return kStr

  """
# mod 3 v0 -> v0
s_mov_b32  s1 0xaaaaaaab
v_mul_hi_u32  v2 v0 s1
v_mul_lo_u32  v1 v0 s1
v_lshrrev_b64  v[1:2] 33 v[1:2]
v_mul_lo_u32  v1 v1 3
v_sub_u32  v0 vcc v0 v1
# mod 6
s_mov_b32  s1 0xaaaaaaab
v_mul_hi_u32  v2 v0 s1
v_mul_lo_u32  v1 v0 s1
v_lshrrev_b64  v[1:2] 34 v[1:2]
v_mul_lo_u32  v1 v1 6
v_sub_u32  v0 vcc v0 v1
# mod 12
s_mov_b32  s1 0xaaaaaaab
v_mul_hi_u32  v2 v0 s1
v_mul_lo_u32  v1 v0 s1
v_lshrrev_b64  v[1:2] 35 v[1:2]
v_mul_lo_u32  v1 v1 12
v_sub_u32  v0 vcc v0 v1
# mod 2
V_AND_B32  v0 1 v0
# mod 4
V_AND_B32  v0 3 v0
# mod 8
V_AND_B32  v0 7 v0
# mod 16
V_AND_B32  v0 15 v0

    else:
      kStr += "/"
# div 2
V_LSHLREV_B32  v0 1 v0
# div 2
V_LSHLREV_B32  v0 2 v0
# div 8
V_LSHLREV_B32  v0 3 v0
# div 16
V_LSHLREV_B32  v0 4 v0
# div 3
s_mov_b32  s0 0xaaaaaaab
v_mul_hi_u32  v3 v0 s0
v_mul_lo_u32  v2 v0 s0
v_lshrrev_b64  v[2:3] 33 v[2:3]
# div 6
s_mov_b32  s0 0xaaaaaaab
v_mul_hi_u32  v3 v0 s0
v_mul_lo_u32  v2 v0 s0
v_lshrrev_b64  v[2:3] 34 v[2:3]
# div 12
s_mov_b32  s0 0xaaaaaaab
v_mul_hi_u32  v3 v0 s0
v_mul_lo_u32  v2 v0 s0
v_lshrrev_b64  v[2:3] 35 v[2:3]
  """

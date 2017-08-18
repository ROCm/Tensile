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
# (6) Add and multiply functions, compile time known, handles carrying, only need to debug in one place


from SolutionStructs import DataType
from Common import globalParameters, kernelLanguageIsSource, print1, print2, printExit, printWarning
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
    self.overflowedState = False

  ########################################
  # Check Out
  def checkOut(self, size):
    if self.overflowedState: return self.start

    found = -1
    for i in range(0, self.size):
      valid = True
      if i + size > self.size:
        valid = False
        break
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
      #print "checkOut(%u,%u)"% (found+self.start, size)
      return (found+self.start)
    else:
      self.overflowedState = True
      printWarning("Scratch register overflow!")
      #for a in self.available:
      #  print a
      return self.start

  ########################################
  # Check In
  def checkIn(self, start):
    if self.overflowedState: return

    start -= self.start
    if start in self.checkOutSize:
      size = self.checkOutSize[start]
      #print "checkIn(%u,%u)"% (start+self.start, size)
      for i in range(start, start+size):
        self.available[i] = True
      self.checkOutSize.pop(start)
    else:
      printWarning("Checking in registers @ %i that weren't checked out"%start)

  ########################################
  # Overflowed ?
  def overflowed(self):
    return self.overflowedState




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

    self.do = {}

    self.do["PreLoop"]    = True
    self.do["GlobalRead"] = True
    self.do["GlobalInc"]  = True
    self.do["LocalWrite"] = True
    self.do["LocalRead"]  = True
    self.do["Wait"]       = True
    self.do["Sync"]       = True
    self.do["MAC"]        = True
    self.do["PostLoop"]   = True
    self.globalReadIncsUseVgpr = True # slightly fewer [v_mov] instructions but extra registers

    # ISA version, such as 803
    self.version = int(self.language[3:])
    self.versionMajor = int(self.language[3])
    self.versionMinor = int(self.language[4])
    self.versionStep  = int(self.language[5])
    print1("KernelWriterAsssembly targeting gfx%u\n" % self.version )

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
          }, # 803
        900: {
          "GlobalRead": [ flat_load_dwordx4, flat_load_dwordx2,
            flat_load_dword ],
          "GlobalWrite": [ flat_store_dwordx4, flat_store_dwordx2,
            flat_store_dword ],
          "LocalRead": [ ds_read_b128, ds_read2_b64,
            ds_read_b64, ds_read2_b32, ds_read_b32 ],
          "LocalWrite": [ ds_write_b128, ds_write2_b64,
            ds_write_b64, ds_write2_b32, ds_write_b32 ]
          } # 900
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
  def initKernel(self, kernel, tPA, tPB ):
    super(KernelWriterAssembly, self).initKernel(kernel, tPA, tPB)
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
    self.globalReadWidthA = tPA["nrcv"]
    #self.globalReadWidthA = kernel["VectorWidth"] if (self.readTileDimVectorA \
    #    or self.readUnrollDimVectorA) else 1
    self.globalReadWidthA *= self.rpe
    self.globalRead2CoalescedA = kernel["NumLoadsCoalescedA"]>1 \
        or self.readCoalescedComponentsA
    self.globalRead2PerpendicularA = kernel["NumLoadsPerpendicularA"] > 1 \
        or self.readPerpendicularComponentsA
    self.globalReadInstructionIdxA = \
        self.selectMemoryInstruction("GlobalRead", self.globalReadWidthA, \
        kernel["GlobalRead2A"], \
        self.globalRead2CoalescedA, self.globalRead2PerpendicularA, [] )

    ########################################
    # globalReadB instruction; no flat_load2_
    self.globalReadWidthB = tPB["nrcv"]
    #self.globalReadWidthB = kernel["VectorWidth"] if (self.readTileDimVectorB \
    #    or self.readUnrollDimVectorB) else 1
    self.globalReadWidthB *= self.rpe
    self.globalRead2CoalescedB = kernel["NumLoadsCoalescedB"]>1 \
        or self.readCoalescedComponentsB
    self.globalRead2PerpendicularB = kernel["NumLoadsPerpendicularB"] > 1 \
        or self.readPerpendicularComponentsB
    self.globalReadInstructionIdxB = \
        self.selectMemoryInstruction("GlobalRead", self.globalReadWidthB, \
        kernel["GlobalRead2B"], \
        self.globalRead2CoalescedB, self.globalRead2PerpendicularB, [] )

    ########################################
    # localWriteA instruction
    # for local, tile->para, unroll->perp
    #self.localWriteWidthA = 1 if (self.writeTileDimComponentsA \
    #    or self.writeUnrollDimComponentsA) else kernel["VectorWidth"]
    self.localWriteWidthA = tPA["nwcv"]
    self.localWriteWidthA *= self.rpe
    self.localWrite2CoalescedA = tPA["nrc"]>1 \
        or self.writeTileDimComponentsA
    self.localWrite2PerpendicularA = tPA["nrp"]>1 \
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
    #self.localWriteWidthB = 1 if (self.writeTileDimComponentsB \
    #    or self.writeUnrollDimComponentsB) else kernel["VectorWidth"]
    self.localWriteWidthB = tPA["nwcv"]
    print "lwwb", self.localWriteWidthB
    self.localWriteWidthB *= self.rpe
    self.localWrite2CoalescedB = tPB["nrc"]>1 \
        or self.writeTileDimComponentsB
    self.localWrite2PerpendicularB = tPB["nrp"]>1 \
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
    print "localWrite", self.localWriteInstructionA
    # global reads per instruction
    tPA["nrcvpi"] = self.globalReadInstructionA.totalWidth / self.rpe
    tPB["nrcvpi"] = self.globalReadInstructionB.totalWidth / self.rpe
    tPA["nwcvpi"] = self.localWriteInstructionA.totalWidth / self.rpe
    tPB["nwcvpi"] = self.localWriteInstructionB.totalWidth / self.rpe
    print "nwcvpi", tPA["nwcvpi"] 
    #print self.getKernelName(kernel)
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
    #    * nlp * self.numWriteVectorComponentsA
    #numLocalWriteInstructionsA = numLocalWritesA \
    #    / self.localWriteInstructionA[self.instructionIdxNumOffsets]
    numVgprLocalWriteAddressesA = 1 * self.rpla

    #numLocalWritesB = kernel["NumLoadsCoalescedB"] \
    #    * nlp * self.numWriteVectorComponentsB
    #numLocalWriteInstructionsB = numLocalWritesB \
    #    / self.localWriteInstructionB[self.instructionIdxNumOffsets]
    numVgprLocalWriteAddressesB = 1 * self.rpla

    ####################################
    # num vgprs: global read addresses
    numGlobalReadsA = kernel["NumLoadsCoalescedA"] \
        * kernel["NumLoadsPerpendicularA"] * kernel["VectorWidth"] \
        * self.numReadVectorComponentsA
    print "numGlobalReadsA", numGlobalReadsA
    numGlobalReadInstructionsA = numGlobalReadsA \
        / self.globalReadInstructionA.blockWidth
    print "numGlobalReadInstructionsA", numGlobalReadInstructionsA
    numVgprGlobalReadAddressesA = numGlobalReadInstructionsA * self.rpga

    numGlobalReadsB = kernel["NumLoadsCoalescedB"] \
        * kernel["NumLoadsPerpendicularB"] * kernel["VectorWidth"] \
        * self.numReadVectorComponentsB
    numGlobalReadInstructionsB = numGlobalReadsB \
        / self.globalReadInstructionB.blockWidth
    numVgprGlobalReadAddressesB = numGlobalReadInstructionsB * self.rpga
    numVgprSerial = 1
    if self.globalReadIncsUseVgpr:
      numVgprGlobalReadIncsA = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpga
      numVgprGlobalReadIncsB = kernel["ProblemType"]["NumIndicesSummation"] \
          * self.rpga
    else:
      numVgprGlobalReadIncsA = 0
      numVgprGlobalReadIncsB = 0

    numVgprAddressD = self.rpga if globalParameters["DebugKernel"] else 0

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
    self.startVgprGlobalReadIncsA = vgprIdx
    vgprIdx += numVgprGlobalReadIncsA
    self.startVgprGlobalReadIncsB = vgprIdx
    vgprIdx += numVgprGlobalReadIncsB
    self.startVgprAddressD = vgprIdx
    vgprIdx += numVgprAddressD
    self.startVgprSerial = vgprIdx
    vgprIdx += numVgprSerial
    print2("%3u vgprs <- %s" % (vgprIdx, self.kernelName) )
    self.startVgprTmp = vgprIdx
    vgprPerCU = 65536
    vgprPerThreadPerOccupancy = vgprPerCU / kernel["NumThreads"]
    numWorkGroupsPerCU = vgprPerThreadPerOccupancy / self.startVgprTmp
    numWavesPerWorkGroup = kernel["NumThreads"] / 64
    numWavesPerCU = numWorkGroupsPerCU * numWavesPerWorkGroup
    self.numWavesPerSimd = numWavesPerCU / 4
    self.spills = self.numWavesPerSimd < 1
    #if self.numWavesPerSimd < 1:
    #  printExit("waves/simd: %u; %u vgprs" % (self.numWavesPerSimd, self.startVgprTmp) )
    maxVgprSameOccupancy = vgprPerThreadPerOccupancy / numWorkGroupsPerCU
    self.numVgprTmp = maxVgprSameOccupancy - self.startVgprTmp
    self.totalVgprs = maxVgprSameOccupancy

    self.startVgprSerial = self.totalVgprs-1
    #self.globalWriteAddrC = self.totalVgprs-4 # match macro
    self.globalWriteAddrC = self.startVgprSerial-4 # match macro
    if self.spills:
      self.do["PreLoop"]    = False
      self.do["GlobalRead"] = False
      self.do["GlobalInc"]  = False
      self.do["LocalWrite"] = False
      self.do["LocalRead"]  = False
      self.do["Wait"]       = False
      self.do["Sync"]       = False
      self.do["MAC"]        = False
      self.do["PostLoop"]   = False
      self.totalVgprs = 1

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
    self.numSgprAddressD = self.rpga if globalParameters["DebugKernel"] else 0

    ####################################
    # num sgprs: global read increments
    if self.globalReadIncsUseVgpr:
      numSgprGlobalReadIncsA = 0
      numSgprGlobalReadIncsB = 0
    else:
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
    # SGPR Assignment according to AMDGPU-ABI
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

    # place any of these gpr inst values into tPA, tPB for later reference
    tPA["localWriteOffsets"] = []
    tPA["globalReadInstruction"] = self.globalReadInstructionA
    tPA["localWriteInstruction"] = self.localWriteInstructionA
    tPA["localReadInstruction"] = self.localReadInstructionA
    tPA["gpr"] = {}

    tPB["localWriteOffsets"] = []
    tPB["globalReadInstruction"] = self.globalReadInstructionB
    tPB["localWriteInstruction"] = self.localWriteInstructionB
    tPB["localReadInstruction"] = self.localReadInstructionB
    tPB["gpr"] = {}



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
    if self.globalReadIncsUseVgpr:
      kStr += self.macroRegister("vgprGlobalReadIncsA", \
          self.startVgprGlobalReadIncsA)
      kStr += self.macroRegister("vgprGlobalReadIncsB", \
          self.startVgprGlobalReadIncsB)
    if globalParameters["DebugKernel"]:
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
    kStr += self.macroRegister("sgprWorkGroup%u"%(0 if kernel["WorkGroupMapping"]>0 else 1), self.startSgprWorkGroup0)
    kStr += self.macroRegister("sgprWorkGroup%u"%(1 if kernel["WorkGroupMapping"]>0 else 0), self.startSgprWorkGroup1)
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
    if globalParameters["DebugKernel"]:
      kStr += self.macroRegister("sgprAddressD", self.startSgprAddressD)
    if not self.globalReadIncsUseVgpr:
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
    kStr += ".macro GLOBAL_WRITE vc0 vc1 d0 d1 base tmp%s"%self.endLine

    kStr += ".set idx, %u + \\vc0 + \\d0*%u + \\vc1*%u + \\d1*%u*%u %s" \
        % (self.startVgprValuC, \
        kernel["VectorWidth"], \
        kernel["ThreadTile0"], \
        kernel["VectorWidth"], kernel["ThreadTile0"], \
        self.endLine )
    kStr += ".set addr, \\tmp+2%s" % self.endLine
    #kStr += ".set vgprTmp, %u%s" % ( self.totalVgprs-8, self.endLine)
    #kStr += dump(vgpr(self.globalWriteAddrC+0))
    #kStr += dump(vgpr(self.globalWriteAddrC+1))
    #kStr += inst("v_mov_b32", vgpr(65), sgpr("Alpha"), "")
    #kStr += dump(vgpr(65))
    #kStr += inst("v_mov_b32", vgpr(65), sgpr("Beta"), "")
    #kStr += dump(vgpr(65))
    #kStr += dump("v[idx]")
    # static tmps b/c
    #vgprAddr = self.totalVgprs-6
    #vgprValue = self.totalVgprs-7
    kStr += inst("v_mov_b32", "v[addr+0]", sgpr("StridesC"), \
        "%s = StridesC"%"v[addr+0]")
    kStr += inst("v_mov_b32", "v[addr+1]", hex(0x0), \
        "%s = 0"%"v[addr+1]" )
    #addr += (d1*strideC + d0*VW + vc)*4bytes

    # tmp1 = strideC*(vc1 + d1*16*vw)
    #kStr += inst("v_lshlrev_b32", "v[\\tmp+1]", \
    #    hex(log2(kernel["SubGroup1"]*kernel["VectorWidth"])), \
    #    "\\d1", "tmp1 = d1*sg1*VW" )
    kStr += staticMultiply("v[\\tmp+1]", "\\d1", (kernel["SubGroup1"]*kernel["VectorWidth"]))

    kStr += inst("v_add_u32", "v[\\tmp+1]", "vcc", "\\vc1", "v[\\tmp+1]", \
        "tmp1 = vc1 + d1*sg1*VW")
    kStr += inst("v_mul_u32_u24", "v[\\tmp+1]", "v[\\tmp+1]", "v[addr+0]", \
        "%s = StridesC*(vc1+d1*sg1*VW)"%"v[\\tmp+1]" )

    # tmp0 = c0 + d0*16*vw
    #kStr += inst("v_lshlrev_b32", "v[\\tmp+0]", \
    #    hex(log2(kernel["SubGroup0"]*kernel["VectorWidth"])), "\\d0", \
    #    "tmp0 = d0*sg0*VW" )
    kStr += staticMultiply("v[\\tmp+0]", "\\d0", (kernel["SubGroup0"]*kernel["VectorWidth"]))
    kStr += inst("v_add_u32", "v[\\tmp+0]", "vcc", "\\vc0", "v[\\tmp+0]", \
        "tmp0 = vc0 + d0*sg0*VW")
    #kStr += dump(vgpr("Tmp"))
    #kStr += "s_endpgm\n"
    kStr += inst("v_add_u32", "v[addr+0]", "vcc", "v[\\tmp+0]", "v[\\tmp+1]", \
        "%s = vc0 + d0*sg0*VW + StridesC*(vc1+d1*sg1*VW)"%"v[addr+0]" )

    #kStr += inst("v_lshlrev_b64", vgpr(vgprAddr,2), hex(log2(kernel["SubGroup0"])), \
    #    vgpr(vgprAddr,2), "%s = 16*(strideC1J*(d1*VW+vc1)+d0*VW)"%vgpr(vgprAddr) )

    #kStr += inst("v_add_u32", vgpr(vgprAddr), "vcc", "\\vc0", vgpr(vgprAddr), \
    #    "%s = 16*(StridesC1J*(d1*VW+vc1)+d0*VW)+vc0"%vgpr(vgprAddr) )

    kStr += inst("v_lshlrev_b64", "v[addr:addr+1]", hex(log2(self.bpe)), \
        "v[addr:addr+1]", "%s = 4*(vc0 + d0*sg0*VW + StridesC*(vc1+d1*sg1*VW) )"%"v[addr]" )

    kStr += inst("v_add_u32", "v[addr]", "vcc", "v[\\base+0]", \
        "v[addr]", "%s = base + (16*(strideC1J*(d1*VW+vc1)+d0*VW)+vc0)*4"%"v[addr]" )
    kStr += inst("v_addc_u32", "v[addr+1]", "vcc", "v[\\base+1]", \
        "v[addr+1]", "vcc", "%s = base + (16*(strideC1J*(d1*VW+vc1)+d0*VW)+vc0)*4"%"v[addr+1]")
    kStr += inst("v_mul_f32", "v[idx]", sgpr("Alpha"), "v[idx]", "*= alpha" )
    #kStr += dump("v[idx]")
    if kernel["ProblemType"]["UseBeta"]:
      kStr += inst("flat_load_dword", "v[\\tmp]", "v[addr:addr+1]", \
          "load C" )
      kStr += inst("s_waitcnt", "vmcnt(0) & lgkmcnt(0)", "wait C" )
      #kStr += dump("v[\\tmp]")
      kStr += inst("v_mul_f32", "v[\\tmp]", sgpr("Beta"), "v[\\tmp]", \
          "%s = C*beta"%"v[\\tmp]" )
      #kStr += dump(vgpr("v[\\tmp]")
      kStr += inst("v_add_f32", "v[idx]", "v[\\tmp]", "v[idx]", \
          "v[idx] = sum*alpha + C*beta" )
    #kStr += dump("v[addr+0]")
    #kStr += dump("v[addr+1]")
    #kStr += dump("v[idx]")
    #kStr += "s_endpgm\n"

    kStr += inst("flat_store_dword", "v[addr:addr+1]", "v[idx]", "store C" )
    kStr += ".endm%s"%self.endLine

    ########################################
    # Dynamic Scalar Divide
    kStr += self.comment3("Dynamic Scalar Divide: vQuotient=vDividend/vDivisor; vRemainder=vDividend%vDivisor;")
    kStr += ".macro DYNAMIC_VECTOR_DIVIDE vQuotient vRemainder vDividend vDivisor vTmp0 vTmp1 sTmp%s" % self.endLine
    kStr += inst("v_cvt_f32_u32", "v[\\vQuotient]",  "v[\\vDivisor]",  "" )
    kStr += inst("v_rcp_f32",     "v[\\vQuotient]",  "v[\\vQuotient]", "" )
    kStr += inst("v_mul_f32",     "v[\\vQuotient]",  "0x4f800000",     "v[\\vQuotient]", "" )
    kStr += inst("v_cvt_u32_f32", "v[\\vQuotient]",  "v[\\vQuotient]", "" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vDivisor]", "v[\\vQuotient]", "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vTmp0]",      "v[\\vDivisor]", "v[\\vQuotient]", "" )
    kStr += inst("v_sub_u32",     "v[\\vTmp1]",      "vcc", hex(0),    "v[\\vRemainder]", "" )
    kStr += inst("v_cmp_ne_i32",  "s[\\sTmp:\\sTmp+1]", hex(0),        "v[\\vTmp0]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vRemainder]", "v[\\vTmp1]",     "v[\\vRemainder]", "s[\\sTmp:\\sTmp+1]", "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vRemainder]", "v[\\vRemainder]", "v[\\vQuotient]", "" )
    kStr += inst("v_sub_u32",     "v[\\vTmp0]",      "vcc",            "v[\\vQuotient]", "v[\\vRemainder]", "" )
    kStr += inst("V_add_u32",     "v[\\vQuotient]",  "vcc",            "v[\\vQuotient]", "v[\\vRemainder]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vTmp0]", "s[\\sTmp:\\sTmp+1]", "" )
    kStr += inst("v_mul_hi_u32",  "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vDividend]", "" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vQuotient]", "v[\\vDivisor]", "" )
    kStr += inst("v_sub_u32",     "v[\\vTmp0]",      "vcc",            "v[\\vDividend]", "v[\\vRemainder]", "" )
    kStr += inst("v_cmp_ge_u32",  "s[\\sTmp:\\sTmp+1]", "v[\\vDividend]", "v[\\vRemainder]", "" )
    kStr += inst("V_add_u32",     "v[\\vRemainder]", "vcc",            hex(1), "v[\\vQuotient]", "" )
    kStr += inst("V_add_u32",     "v[\\vTmp1]",      "vcc", -1,        "v[\\vQuotient]", "" )
    kStr += inst("v_cmp_le_u32",  "vcc",             "v[\\vDivisor]", "v[\\vTmp0]", "" )
    kStr += inst("s_and_b64", "vcc", "s[\\sTmp:\\sTmp+1]",                "vcc", "" ) # FIXME
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vQuotient]", "v[\\vRemainder]", "vcc", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  "v[\\vTmp1]",     "v[\\vQuotient]", "s[\\sTmp:\\sTmp+1]", "" )
    kStr += inst("v_cmp_ne_i32",  "vcc", hex(0),     "v[\\vDivisor]", "" )
    kStr += inst("v_cndmask_b32", "v[\\vQuotient]",  -1, "v[\\vQuotient]", "vcc", "final result" )
    kStr += inst("v_mul_lo_u32",  "v[\\vRemainder]", "v[\\vQuotient]", "v[\\vDivisor]", "" )
    kStr += inst("v_sub_u32",     "v[\\vRemainder]", "vcc",            "v[\\vDividend]", "v[\\vRemainder]", "final result" )
    kStr += ".endm%s" % self.endLine

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
          #if a==0 and b==0:
          #  kStr += dump(aStr)
          kStr += "v_mac_f32 %s, %s, %s%s" % (cStr, aStr, bStr, self.endLine)
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

    # begin kernel descriptor
    kStr += ".hsa_code_object_version 2,0%s" % self.endLine
    kStr += ".hsa_code_object_isa %u, %u, %u, \"AMD\", \"AMDGPU\" %s" \
        % (self.versionMajor, self.versionMinor, self.versionStep, self.endLine)
    kStr += ".text%s" % self.endLine
    kStr += ".p2align 8%s" % self.endLine
    kStr += ".amdgpu_hsa_kernel %s%s" % (self.kernelName, self.endLine)
    kStr += "%s:%s" % (self.kernelName, self.endLine)
    kStr += ".amd_kernel_code_t%s" % self.endLine
    kStr += "  is_ptr64 = 1%s" % self.endLine
    kStr += "  enable_sgpr_kernarg_segment_ptr = 1%s" % self.endLine

    # kern arg size
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
    if globalParameters["DebugKernel"]:
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
        % ( 1+(self.totalSgprs-1)/8, self.totalSgprs, self.endLine)

    # work-group dimensions
    kStr += "  compute_pgm_rsrc2_tidig_comp_cnt = 0 // 1D wg%s" % self.endLine

    # grid dimensions
    kStr += "  compute_pgm_rsrc2_tgid_x_en = 1 // wg.x%s" % self.endLine
    kStr += "  compute_pgm_rsrc2_tgid_y_en = 1 // wg.y%s" % self.endLine
    if kernel["ProblemType"]["NumIndicesC"] > 2:
      kStr += "  compute_pgm_rsrc2_tgid_z_en = %u // wg.z%s" % (1 if kernel["ProblemType"]["NumIndicesC"] > 2 else 0, self.endLine)
    #if abs(kernel["WorkGroupMapping"]) > 1:
    #  kStr += "  enable_sgpr_grid_workgroup_count_x = 1 // nwg0%s" % self.endLine
    #  kStr += "  enable_sgpr_grid_workgroup_count_y = 1 // nwg1%s" % self.endLine

    # lds size
    kStr += "  compute_pgm_rsrc2_lds_size = 1 // ?%s" % self.endLine
    kStr += "  workgroup_group_segment_byte_size = %u // lds bytes%s" \
        % ( kernel["LdsNumElements"] \
        * self.bpe, self.endLine )

    # other
    kStr += "  compute_pgm_rsrc2_user_sgpr = 2 // vcc%s" % self.endLine
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
    if self.do["PreLoop"]: kStr += inst("s_mov_b32", "m0", hex(kernel["LdsNumElements"] \
        * self.bpe), "LDS clamp at %u bytes" \
        %(kernel["LdsNumElements"] * self.bpe) )

    if self.do["PreLoop"]: kStr += inst("v_mov_b32", vgpr("Serial"), vgpr(0), "thread serial id")

    ########################################
    # load kernel args
    # TODO revert to s_load_dwordx2
    kStr += self.comment("Load Kernel Args")
    kernArgOffset = 0
    if globalParameters["DebugKernel"]:
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressD"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr debug" )
      kernArgOffset += 1*4
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressD+1"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr debug" )
      kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressC"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr c" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressC+1"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr c" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressA"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr a" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressA+1"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr a" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressB"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr b" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("AddressB+1"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load addr b" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("Alpha"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load alpha" )
    kernArgOffset += 1*4
    if kernel["ProblemType"]["UseBeta"]:
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("Beta"), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load beta" )
      kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("OffsetC"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset c" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("OffsetA"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset a" )
    kernArgOffset += 1*4
    if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("OffsetB"), \
        sgpr("KernArgAddress",2), hex(kernArgOffset), "load offset b" )
    kernArgOffset += 1*4
    for i in range(0, self.numSgprStridesC):
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("StridesC+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride c %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprStridesA):
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("StridesA+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride a %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprStridesB):
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("StridesB+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load stride b %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprSizesFree):
      if self.do["PreLoop"]: kStr += inst("s_load_dword", sgpr("SizesFree+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load size free %u"%i )
      kernArgOffset += 1*4
    for i in range(0, self.numSgprSizesSum):
      kStr += inst("s_load_dword", sgpr("SizesSum+%u"%i), \
          sgpr("KernArgAddress",2), hex(kernArgOffset), "load size free %u"%i )
      kernArgOffset += 1*4
    kStr += inst("s_waitcnt", "lgkmcnt(0)", \
        "wait for %u bytes of kern args" % kernArgOffset )
    if not self.do["PreLoop"]:
      kStr += ".if 0\n"

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

    # test debug buffer
    #v = self.vgprScratch.checkOut(3)
    #kStr += inst("v_mov_b32", vgpr(v), sgpr("AddressC"), "" )
    #kStr += inst("v_mov_b32", vgpr(v+1), sgpr("AddressC+1"), "" )
    #kStr += inst("v_mov_b32", vgpr(v+2), hex(3), "" )
    #kStr += inst("flat_store_dword", vgpr(v, 2), vgpr(v+2), "debug serial" )
    #kStr += "s_endpgm\n"

    ########################################
    # Debug Buffer
    if globalParameters["DebugKernel"]:
      kStr += self.comment("Debug Buffer")
      nt_log2 = log2(kernel["NumThreads"])

      # nwg0
      nwg0 = self.vgprScratch.checkOut(1)
      tmpVgpr = self.vgprScratch.checkOut(1)
      tmpSgpr = self.startSgprOffsetC
      kStr += "// nwg0 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
      kStr += inst("v_add_u32", vgpr(nwg0), "vcc", sgpr("SizesFree+0"), \
          hex(kernel["MacroTile0"]-1), "%s = size0+MT0-1"%vgpr(nwg0))
      kStr += vectorStaticDivide(nwg0, nwg0, kernel["MacroTile0"], tmpVgpr, tmpSgpr)
      tmpVgpr = self.vgprScratch.checkIn(tmpVgpr)
      self.nipt = 16 # num integers per thread
      v = self.vgprScratch.checkOut(3)
      if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
      kStr += inst("v_mov_b32", vgpr(v), sgpr("WorkGroup0"), "%s=wg0"%vgpr(v) )
      kStr += inst("v_mov_b32", vgpr(v+1), sgpr("WorkGroup1"), "%s=wg1"%vgpr(v+1) )
      kStr += inst("v_mul_u32_u24", vgpr(v+1), vgpr(v+1), vgpr(nwg0), \
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

    return kStr

  ##############################################################################
  # Global Read Addresses: Work-Group - LATER
  ##############################################################################
  def graWorkGroup(self, kernel):
    kStr = ""
    tmpVgpr = self.vgprScratch.checkOut(2)

    ########################################
    # Blocked rows or columns
    if kernel["WorkGroupMappingType"] == "B" and abs(kernel["WorkGroupMapping"]) > 1:
      # nwg0
      nwg0 = self.vgprScratch.checkOut(1)
      tmpSgpr = self.startSgprOffsetC
      kStr += "// nwg0 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar0, self.tileChar0, self.tileChar0, self.endLine)
      kStr += inst("v_mov_b32", vgpr(nwg0), sgpr("SizesFree+0"), "")
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile0"]-1), "")
      kStr += inst("v_add_u32", vgpr(nwg0), "vcc", vgpr(nwg0), \
          sgpr(tmpSgpr), "%s = size0+MT0-1"%vgpr(nwg0))
      kStr += vectorStaticDivide(nwg0, nwg0, kernel["MacroTile0"], tmpVgpr, tmpSgpr)
      #kStr += dump(vgpr(nwg0))

      # nwg1
      nwg1 = self.vgprScratch.checkOut(1)
      kStr += "// nwg1 = (size%s + MT%s - 1) / MT%s;%s" \
          % (self.tileChar1, self.tileChar1, self.tileChar1, self.endLine)
      kStr += inst("v_mov_b32", vgpr(nwg1), sgpr("SizesFree+1"), "")
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(kernel["MacroTile1"]-1), "")
      kStr += inst("v_add_u32", vgpr(nwg1), "vcc", vgpr(nwg1), \
          sgpr(tmpSgpr), "%s = size1+MT1-1"%vgpr(nwg1))
      kStr += vectorStaticDivide(nwg1, nwg1, kernel["MacroTile1"], tmpVgpr, tmpSgpr)
      #kStr += dump(vgpr(nwg1))

      # blockId and serial within block
      blockId = self.vgprScratch.checkOut(1)
      wgSerial = self.vgprScratch.checkOut(1)
      wg1 = self.vgprScratch.checkOut(1)
      kStr += inst("v_mov_b32", vgpr(wg1), sgpr("WorkGroup1"), "wg1")

      #kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr("WorkGroup0"), "wg0")
      #kStr += dump(vgpr(tmpVgpr))
      #kStr += dump(vgpr(wg1))

      kStr += vectorStaticDivideAndRemainder(blockId, wgSerial, wg1, \
          abs(kernel["WorkGroupMapping"]), tmpVgpr, tmpSgpr)
      #kStr += dump(vgpr(wgSerial))
      kStr += inst("v_mul_u32_u24", vgpr(wgSerial), vgpr(wgSerial), \
          vgpr(nwg0), "(wg1 % WGM)*nwg0")
      self.vgprScratch.checkIn(nwg0)
      #kStr += dump(vgpr(wgSerial))
      kStr += inst("v_add_u32", vgpr(wgSerial), "vcc", vgpr(wgSerial), \
          sgpr("WorkGroup0"), "wgSerial = wg0 + (wg1 % WGM)*nwg0")
      #kStr += "s_endpgm\n"
      #return kStr

      # num full blocks
      numFullBlocks = self.vgprScratch.checkOut(1)
      kStr += "// numFullBlocks = (nwg1) / WGM;%s" % (self.endLine)
      blockRemainder = self.vgprScratch.checkOut(1)
      kStr += vectorStaticDivideAndRemainder(numFullBlocks, blockRemainder, \
          nwg1, abs(kernel["WorkGroupMapping"]), tmpVgpr, tmpSgpr)
      self.vgprScratch.checkIn(nwg1)

      #kStr += dump(vgpr(blockId))
      #kStr += dump(vgpr(numFullBlocks))
      #kStr += dump(vgpr(wgSerial))
      #kStr += dump(vgpr(blockRemainder))
      # lastBlockWidth = blockRemainder

      # my block's width
      #kStr += inst("v_mov_b32", vgpr(tmpVgpr), hex(111), "")
      #kStr += dump(vgpr(tmpVgpr))
      kStr += inst("v_cmp_lt_u32", sgpr(tmpSgpr,2), vgpr(blockId), vgpr(numFullBlocks), "blockId < numFullBlocks" )
      self.vgprScratch.checkIn(numFullBlocks)
      blockWidth = self.vgprScratch.checkOut(1)
      kStr += inst("v_cndmask_b32", vgpr(blockWidth), vgpr(blockRemainder), hex(abs(kernel["WorkGroupMapping"])), sgpr(tmpSgpr,2), "blockWidth = (blockId < numFullBlocks) ? WGM : remainder" )
      self.vgprScratch.checkIn(blockRemainder)
      #kStr += dump(vgpr(blockWidth))

      # dynamic divide and remainder
      # wg0 = wgSerialInBlock / myBlockWidth
      # wg1 = wgSerialInBlock % myBlockWidth + blockId*WGM
      wg0 = self.vgprScratch.checkOut(1)
      kStr += "DYNAMIC_VECTOR_DIVIDE %s %s %s %s %s %s %s%s" % ( wg0, wg1, wgSerial, blockWidth, tmpVgpr, tmpVgpr+1, tmpSgpr, self.endLine )
      kStr += inst("v_mul_u32_u24", vgpr(blockId), vgpr(blockId), \
          abs(kernel["WorkGroupMapping"]), "blockId * WGM")
      kStr += inst("v_add_u32", vgpr(wg1), "vcc", vgpr(wg1), \
          vgpr(blockId), "wg1 += blockId * WGM")

      # move wg0,1 in vgprs into sgprs
      kStr += inst("v_readfirstlane_b32", sgpr("WorkGroup0"), vgpr(wg0), "")
      kStr += inst("v_readfirstlane_b32", sgpr("WorkGroup1"), vgpr(wg1), "")

      # checkin scratch registers
      self.vgprScratch.checkIn(wg0)
      self.vgprScratch.checkIn(wg1)
      self.vgprScratch.checkIn(blockWidth)
      self.vgprScratch.checkIn(wgSerial)
      self.vgprScratch.checkIn(blockId)

      #kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr("WorkGroup0"), "")
      #kStr += dump(vgpr(tmpVgpr))
      #kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr("WorkGroup1"), "")
      #kStr += dump(vgpr(tmpVgpr))
      self.vgprScratch.checkIn(tmpVgpr)
      #kStr += "s_endpgm\n"

    return kStr

  ##############################################################################
  # Global Read Addresses: Subgroup - DONE
  ##############################################################################
  def graSubgroup(self, kernel):
    return self.comment1("  not needed until local read addresses")


  ##############################################################################
  # Global Read Addresses: Tile Assignment A/B
  # stores to v1,2
  ##############################################################################
  def graTileAssignment(self, kernel, tP):
    kStr = ""
    if tP["grcg"]:
      if tP["grcv"]:
        divisorName = tP["lvc"]
      else:
        divisorName = tP["lsc"]
    else:
      if tP["grcv"]:
        divisorName = tP["lsp"]
      else:
        divisorName = lvp
    divisor = kernel[divisorName]

    if tP["grcg"] == tP["tlu"]:
      rReg = self.vgprScratch.checkOut(1) # gro-tile = serial%divisor
      if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
      qReg = self.vgprScratch.checkOut(1) # gro-unroll = serial/divisor
      if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
      tReg = rReg
      uReg = qReg
      tOpStr = "%"
      uOpStr = "/"
    else:
      qReg = self.vgprScratch.checkOut(1) # gro-tile = serial/divisor
      if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
      rReg = self.vgprScratch.checkOut(1) # gro-unroll = serial%divisor
      if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
      tReg = qReg
      uReg = rReg
      tOpStr = "/"
      uOpStr = "%"
    tReg2 = self.vgprScratch.checkOut(1)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    kStr += self.comment1("%s = gro%s-tile = serial%s%s + (wg%s*MT%s);" \
        % (vgpr(tReg2), tP["tensorChar"], tOpStr, divisorName, tP["tensorChar"], tP["tensorChar"]) )
    kStr += self.comment1("%s = gro%s-unroll = serial%s%s;" \
        % (vgpr(uReg), tP["tensorChar"], uOpStr, divisorName) )
    dividendReg = "Serial" # local serial
    tmpVgpr = self.vgprScratch.checkOut(1)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    tmpSgpr = self.startSgprOffsetC
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)

    if kernel["VectorWidth"] > 1:
      if tP["grcv"] == tP["tlu"]:
        kStr += staticMultiply(vgpr(tReg), vgpr(tReg), kernel["VectorWidth"])
      else:
        kStr += staticMultiply(vgpr(uReg), vgpr(uReg), kernel["VectorWidth"])
    kStr += staticMultiply(vgpr(tmpVgpr), sgpr(tP["wg"]), kernel[tP["mt"]])
    kStr += inst("v_add_u32", vgpr(tReg2), "vcc", vgpr(tmpVgpr), \
        vgpr(tReg), "gro%s-tile = serial%s%s*VW + (wg%s*MT%s)" \
        % (tP["tensorChar"], tOpStr, divisorName, tP["tensorChar"], tP["tensorChar"]) )
    tP["gpr"]["lwoT"] = tReg
    tP["gpr"]["tReg"] = tReg2
    tP["gpr"]["uReg"] = uReg
    self.vgprScratch.checkIn(tmpVgpr)
    #kStr += dump(vgpr("Serial"))
    #kStr += dump(vgpr(tReg2))
    #kStr += dump(vgpr(uReg))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Global Read Addresses: Unroll Assignment A/B
  ##############################################################################
  def graUnrollAssignment(self, kernel, tP):
    return self.comment1(vgpr(tP["gpr"]["uReg"]))

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
  # Global Read Addresses: Tile Offsets A/B
  ##############################################################################
  def graTileOffsets(self, kernel, tP):
    kStr = ""
    numTileOffsets = tP["nrt"]
    if tP["rtc"]:
      numTileOffsets *= kernel["VectorWidth"]
    tP["vgprTileOffsets"] = self.vgprScratch.checkOut(numTileOffsets)
    v = tP["vgprTileOffsets"]
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    stride = tP["lsc"] if tP["tlu"] else tP["lsp"]
    stride = kernel[stride]
    if tP["rtc"]:
      # l=0, s=0
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(tP["gpr"]["tReg"]), "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], 0, 0) )
      # l=0, s>0
      for s in range(1, kernel["VectorWidth"]):
        kStr += inst("v_add_u32", vgpr(v+s), "vcc", 1, \
            vgpr(v+s-1), "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], 0, s) )
      for l in range(1, tP["nrt"]):
        # l>0, s=0
        kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]), "vcc", stride, \
            vgpr(v+(l-1)*kernel["VectorWidth"]), \
            "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], l, 0) )
        # l>0, s>0
        for s in range(1, kernel["VectorWidth"]):
          kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]+s), "vcc", \
              1, vgpr(v+l*kernel["VectorWidth"]+(s-1)), \
              "gro%s%s_%u_s%u"%(tP["tensorChar"], tP["tileChar"], l, s) )
    else:
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(tP["gpr"]["tReg"]), "gro%s%s_%u"%(tP["tensorChar"], tP["tileChar"], 0) )
      for l in range(1, tP["nrt"]):
        kStr += inst("v_add_u32", vgpr(v+l), "vcc", stride, \
            vgpr(v+l-1), "gro%s%s_%u"%(tP["tensorChar"], tP["tileChar"], l) )
    self.vgprScratch.checkIn(tP["gpr"]["tReg"])
    return kStr


  ##############################################################################
  # Global Read Addresses: Unroll Offsets A/B
  ##############################################################################
  def graUnrollOffsets(self, kernel, tP):
    kStr = ""
    numUnrollOffsets = tP["nru"]
    if tP["ruc"]:
      numUnrollOffsets *= kernel["VectorWidth"]
    tP["gpr"]["unrollOffsets"] = self.vgprScratch.checkOut(numUnrollOffsets)
    v = tP["gpr"]["unrollOffsets"]
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    stride = (tP["lsp"] if tP["tlu"] else tP["lsc"])
    stride = kernel[stride]
    if tP["ruc"]:
      # l=0, s=0
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(tP["gpr"]["uReg"]), "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, 0) )
      # l=0, s>0
      for s in range(1, kernel["VectorWidth"]):
        kStr += inst("v_add_u32", vgpr(v+s), "vcc", 1, \
            vgpr(v+s-1), "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, s) )
      for l in range(1, tP["nru"]):
        # l>0, s=0
        kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]), "vcc", stride, \
            vgpr(v+(l-1)*kernel["VectorWidth"]), \
            "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, l, 0) )
        # l>0, s>0
        for s in range(0, kernel["VectorWidth"]):
          kStr += inst("v_add_u32", vgpr(v+l*kernel["VectorWidth"]+s), "vcc", \
              1, vgpr(v+l*kernel["VectorWidth"]+(s-1)), \
              "gro%s%s_%u_s%u"%(tP["tensorChar"], self.unrollChar, 0, s) )
    else:
      kStr += inst("v_mov_b32", vgpr(v), \
          vgpr(tP["gpr"]["uReg"]), "gro%s%s_%u"%(tP["tensorChar"], self.unrollChar, 0) )
      for l in range(1, tP["nru"]):
        kStr += inst("v_add_u32", vgpr(v+l), "vcc", stride, \
            vgpr(v+l-1), "gro%s%s_%u"%(tP["tensorChar"], self.unrollChar, l) )
    #self.vgprScratch.checkIn(tP["gpr"]["uReg"])
    return kStr


  ##############################################################################
  # Global Read Addresses: Branch A/B - SKIP
  ##############################################################################
  def graBranch(self, kernel, tP):
    return ""

  ##############################################################################
  # Global Read Addresses: Shift A/B - SKIP
  ##############################################################################
  def graShift(self, kernel, tP):
    return ""
    kStr = ""
    for l in range(0, tP["nrt"]):
      gro = "globalReadOffset%s%s_%u%s" % (tP["tensorChar"], tP["tileChar"], l, \
          ("_s0" if tP["rtc"] else "") )
      limit = "(size%s-%s)" % (tP["tileChar"], \
          ("VECTOR_WIDTH" if tP["rtv"] else "1") )
      kStr += "  %s = (%s > %s) ? %s : %s;%s" \
          % (gro, gro, limit, limit, gro, self.endLine)
    return kStr

  ##############################################################################
  # Global Read Addresses: Final Offsets A/B
  ##############################################################################
  def graFinalOffsets(self, kernel, tP):
    kStr = ""
    tVW = 1
    tVS = 0
    uVW = 1
    uVS = 0
    if tP["rtc"]:
      tVW = kernel["VectorWidth"]
      tVS = 1
    elif tP["ruc"]:
      uVW = kernel["VectorWidth"]
      uVS = 1
    tileOffsets = tP["vgprTileOffsets"]
    unrollOffsets = tP["gpr"]["unrollOffsets"]
    tmp = self.vgprScratch.checkOut(3)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    graIdx = 0
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]/tP["nrcvpi"]):
            # vgpr assignments
            if tP["tlu"]:
              vgprTile   = tileOffsets   + para*tVW + sPara*tVS
              vgprUnroll = unrollOffsets + perp*uVW + sPerp*uVS
            else:
              vgprTile   = tileOffsets   + perp*tVW + sPara*tVS
              vgprUnroll = unrollOffsets + para*uVW + sPerp*uVS
            # global offset macro
            kStr += "GLOBAL_OFFSET_%s vgprGlobalReadAddr%s+%u"%(tP["tensorChar"], tP["tensorChar"], graIdx)
            for i in tP["ia"]:
              if i < kernel["ProblemType"]["NumIndicesC"]:
                if i == tP["tileIdx"]:
                  kStr += ", %2u" % vgprTile
                else: # just a group index
                  kStr += ", %s" % sgpr("WorkGroup+%u"%i)
              else: # summation index
                if i == kernel["ProblemType"]["IndexUnroll"]:
                  kStr += ", %2u" % vgprUnroll
                else:
                  kStr += "globalReadOffset%s%s" % (tP["tensorChar"], self.indexChars[i] )
            kStr += ", %u // gRO%s_%u_%u_%u_%u%s" % (tmp, tP["tensorChar"], \
                para, sPara, perp, sPerp, self.endLine)
            graIdx += self.rpga
    if False:
      kStr += dump(vgpr("GlobalReadAddrA+0"))
      #kStr += dump(vgpr("GlobalReadAddrA+2"))
      #kStr += dump(vgpr("GlobalReadAddrA+4"))
      #kStr += dump(vgpr("GlobalReadAddrA+6"))
      #kStr += dump(vgpr("GlobalReadAddrA+8"))
      #kStr += dump(vgpr("GlobalReadAddrA+10"))
      #kStr += dump(vgpr("GlobalReadAddrA+12"))
      #kStr += dump(vgpr("GlobalReadAddrA+14"))

    self.vgprScratch.checkIn(tileOffsets)
    self.vgprScratch.checkIn(unrollOffsets)
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
  # Global Read Addresses: Addresses A/B - DONE
  ##############################################################################
  def graAddresses(self, kernel, tP):
    kStr = ""
    graIdx = 0
    tmp = self.vgprScratch.checkOut(2)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("Address%s+0"%tP["tensorChar"]), "" )
    kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("Address%s+1"%tP["tensorChar"]), "" )
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]/tP["nrcvpi"]):

            comment = "gRA%s_%u_%u_%u_%u = addr%s+grO%s_%u_%u_%u_%u" \
                % (tP["tensorChar"], para, sPara, perp, sPerp, \
                tP["tensorChar"], tP["tensorChar"], \
                para, sPara, perp, sPerp )
            #kStr += dump(vgpr("GlobalReadAddrA+0"))
            #kStr += dump(vgpr("GlobalReadAddrA+1"))
            #kStr += dump(vgpr(tmp+0))
            #kStr += dump(vgpr(tmp+1))

            kStr += inst("v_add_i32", \
                vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                "vcc", \
                vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                vgpr(tmp+0), \
                comment+" (lower)")
            kStr += inst("v_addc_u32", \
                vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                "vcc", \
                vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                vgpr(tmp+1), \
                "vcc", \
                comment+" (upper)")
            #kStr += dump(vgpr("GlobalReadAddrA+%u+0"%graIdx))
            #kStr += dump(vgpr("GlobalReadAddrA+%u+1"%graIdx))
            graIdx += self.rpga
    #kStr += "s_endpgm\n"
    self.vgprScratch.checkIn(tmp)
    return kStr

  ##############################################################################
  # Global Read Addresses: Increments A/B - DONE
  ##############################################################################
  def graIncrements(self, kernel, loopIdx, tP):
    kStr = ""
    if loopIdx==kernel["ProblemType"]["NumIndicesSummation"]-1:
      if tP["tlu"]:
        if self.globalReadIncsUseVgpr:
          tmpSgpr = self.startSgprOffsetC
          kStr += inst("s_mul_i32", sgpr(tmpSgpr+0), \
              hex(kernel["DepthU"]*4), sgpr("Strides%s"%tP["tensorChar"]), \
              "incr = stride*%u*4bytes"%kernel["DepthU"] )
          """
          kStr += inst("s_addc_u32", \
              sgpr(tmpSgpr+1), \
              hex(0), \
              hex(0), \
              "(carry)")
          """
          kStr += inst("s_mov_b32", \
              sgpr(tmpSgpr+1), \
              hex(0), \
              "(carry)")
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              sgpr(tmpSgpr+0), \
              "" )
          kStr += inst("v_mov_b32", \
              vgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              sgpr(tmpSgpr+1), \
              "" )
        else:
          kStr += inst("s_mul_i32", sgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(kernel["DepthU"]*4), sgpr("Strides%s"%tP["tensorChar"]), \
              "incr = stride*%u*4bytes"%kernel["DepthU"] )
          """
          kStr += inst("s_addc_u32", \
              sgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), \
              hex(0), \
              "(carry)")
          """
          kStr += inst("s_mov_b32", \
              sgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), \
              "(carry)")
        #tmp = self.vgprScratch.checkOut(2)
        #kStr += inst("v_mov_b32", vgpr(tmp+0), sgpr("GlobalReadIncsA+0"), "" )
        #kStr += inst("v_mov_b32", vgpr(tmp+1), sgpr("GlobalReadIncsA+1"), "" )
        #kStr += dump(vgpr(tmp+0))
        #kStr += dump(vgpr(tmp+1))
        #self.vgprScratch.checkIn(tmp)
      else: # transposed
        if self.globalReadIncsUseVgpr:
          kStr += inst("v_mov_b32", vgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(kernel["DepthU"]*4), \
              "incr = %u*4bytes"%kernel["DepthU"] )
          kStr += inst("v_mov_b32", vgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), "incr = %u*4bytes (upper)"%kernel["DepthU"] )
        else:
          kStr += inst("s_mov_b32", sgpr("GlobalReadIncs%s+0"%tP["tensorChar"]), \
              hex(kernel["DepthU"]*4), \
              "incr = %u*4bytes"%kernel["DepthU"] )
          kStr += inst("s_mov_b32", sgpr("GlobalReadIncs%s+1"%tP["tensorChar"]), \
              hex(0), "incr = %u*4bytes (upper)"%kernel["DepthU"] )
    else:
      printExit("NumIndicesSummation=%u not yet supported in assembly" \
          % kernel["ProblemType"]["NumIndicesSummation"] )
    #kStr += dump(vgpr("GlobalReadIncsA+0"))
    #kStr += dump(vgpr("GlobalReadIncsA+1"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Write Addresses: Tile Assignment A/B - DONE
  ##############################################################################
  def lwaTileAssignment(self, kernel, tP):
    return self.comment1("lwaTile%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["lwoT"])))

  ##############################################################################
  # Local Write Addresses: Unroll Assignment A/B - DONE
  ##############################################################################
  def lwaUnrollAssignment(self, kernel, tP):
    return self.comment1("lwaUnroll%s = %s" % (tP["tensorChar"], \
        vgpr(tP["gpr"]["uReg"])))

  ##############################################################################
  # Local Write Addresses: First Offset A/B - DONE
  ##############################################################################
  def lwaFirstOffset(self, kernel, tP):
    kStr = ""
    #"lwFOA = lwA%s + lwA%s*MT%s" \
    #    % (tP["tileChar"], self.unrollChar, tP["tileChar"])

    kStr += inst("v_mul_u32_u24", \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        hex(kernel["MacroTile%s"%tP["tensorChar"]]), \
        vgpr(tP["gpr"]["uReg"]), \
        "lw%s%s*MT%s"%(tP["tensorChar"], self.unrollChar, tP["tensorChar"]))
    #kStr += dump(vgpr("LocalWriteAddrA"))
    kStr += inst("v_add_u32", \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        "vcc", \
        vgpr(tP["gpr"]["lwoT"]), \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        "lwFO%s = lw%s%s + lw%s%s*MT%s" \
        % (tP["tensorChar"], tP["tensorChar"], tP["tileChar"], \
        tP["tensorChar"], self.unrollChar, tP["tileChar"]) )
    kStr += inst("v_lshlrev_b32", \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        hex(log2(self.bpe)), \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        " *= bytes/element" )
    if tP["isB"]:
      kStr += inst("v_add_u32", \
          vgpr("LocalWriteAddrB"), \
          "vcc", \
          hex(kernel["LdsOffsetB"]*self.bpe), \
          vgpr("LocalWriteAddrB"), \
          "lwFOB = lwB%s + lwB%s*MT%s + LDS_OFFSET_B=%u*%u" % (tP["tileChar"], \
          self.unrollChar, tP["tileChar"], kernel["LdsOffsetB"], self.bpe) )
    self.vgprScratch.checkIn(tP["gpr"]["lwoT"])
    self.vgprScratch.checkIn(tP["gpr"]["uReg"])
    #kStr += dump(vgpr("LocalWriteAddrA"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Write Addresses: Final Offsets A - DONE
  # initially assume write offsets fit into 8-bits
  ##############################################################################
  def lwaFinalOffsets(self, kernel, tP):
    kStr = ""
    for perp in range(0, tP["nrp"]):
      for para in range(0, tP["nrc"]):
        for s in range(0, max(tP["nwcv"],tP["nwpv"])/tP["nwcvpi"]):
          lscaOffset = para * kernel[tP["lsc"]]
          lspaOffset = perp * kernel[tP["lsp"]]
          sPara = 1
          sPerp = 1
          if tP["wtc"]:
            sPerp = s
            lscaOffset += s
          elif tP["wuc"]:
            sPara = s
            lspaOffset += s # * VW could go here, check transpose options
          if tP["tlu"]:
            lspaOffset *= kernel[tP["mt"]]
            #lspa *= kernel["VectorWidth"]
          else:
            lscaOffset *= kernel[tP["mt"]]
          if tP["tlu"] == tP["grcv"]:
            lspaOffset *= kernel["VectorWidth"]
          offset = lspaOffset + lscaOffset
          offset *= self.bpe
          offset /= tP["localWriteInstruction"].offsetMultiplier
          tP["localWriteOffsets"].append(offset)
          kStr += "%slwo%s_%u_%u_%u_%u = (%s%d*%s)" \
              % (self.commentPrefix, tP["tensorChar"], \
              para, sPara, perp, sPerp, \
              (("%u + "%sPara) if tP["wtc"] else ""), \
              para, tP["lsc"] )
          if not tP["tlu"]:
            kStr += "*MT%s" % (tP["tileChar"])
          kStr += " + (%s%d*%s)" % (
              (("%u + "%sPerp) if tP["wuc"] else ""), perp, \
              tP["lsp"])
          if tP["tlu"]:
            kStr += "*MT%s" % (tP["tileChar"])
          kStr += " = %u%s%s" % (offset, self.commentSuffix, self.endLine)
    return kStr

  ##############################################################################
  # Local Write Addresses: Declare Addresses A/B - DONE
  ##############################################################################
  def lwaDeclareAddresses(self, kernel, tP):
    return self.comment1("N/A")

  ##############################################################################
  # Local Read Addresses: Tile Assignment A - DONE
  ##############################################################################
  def lraTileAssignmentA(self, kernel, tP):
    kStr = ""
    kStr += "%slr%s = serial %% SG%s%s%s" \
        % (self.commentPrefix, tP["tileChar"], tP["tileChar"], \
        self.commentSuffix, self.endLine)

    divisor = kernel["SubGroup0"]
    qReg = self.vgprScratch.checkOut(1) # quotient
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    rReg = self.vgprScratch.checkOut(1) # remainder
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    dividendReg = "Serial" # local serial
    tmpVgpr = self.vgprScratch.checkOut(1)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    tmpSgpr = self.startSgprOffsetC
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    #kStr += dump(vgpr(rReg))
    #kStr += dump(vgpr(qReg))
    tP["gpr"]["lro"] = rReg
    #kStr += dump(vgpr(tP["gpr"]["lro"]))
    self.tmplroB = qReg
    self.vgprScratch.checkIn(tmpVgpr)
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Read Addresses: Tile Assignment B - DONE
  ##############################################################################
  def lraTileAssignmentB(self, kernel, tP):
    kStr = ""
    kStr += "%slr%s = (serial / SG%s) %% SG%s%s%s" \
        % (self.commentPrefix, tP["tileChar"], tP["tileChar"], \
        tP["tileChar"], self.commentSuffix, self.endLine)
    divisor = kernel["SubGroup1"]
    qReg = self.vgprScratch.checkOut(1) # quotient
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    rReg = self.vgprScratch.checkOut(1) # remainder
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    dividendReg = self.tmplroB
    tmpVgpr = self.vgprScratch.checkOut(1)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    tmpSgpr = self.startSgprOffsetC
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    #kStr += dump(vgpr(rReg))
    #kStr += dump(vgpr(qReg))
    #kStr += "s_endpgm\n"
    self.vgprScratch.checkIn(self.tmplroB) # old
    tP["gpr"]["lro"] = rReg
    #kStr += dump(vgpr(tP["gpr"]["lro"]))
    self.vgprScratch.checkIn(qReg)
    self.vgprScratch.checkIn(tmpVgpr)
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Local Read Addresses: Final Offset A/B - DONE
  ##############################################################################
  def lraFinalOffset(self, kernel, tP):
    kStr = ""
    divisor = kernel["NumThreads"]
    qReg = self.vgprScratch.checkOut(1) # quotient
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    rReg = self.vgprScratch.checkOut(1) # remainder
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    dividendReg = 0
    tmpVgpr = self.vgprScratch.checkOut(1)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    tmpSgpr = self.startSgprOffsetC
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    sgid = qReg
    #kStr += dump(vgpr(tP["gpr"]["lro"]))
    kStr += inst("s_mov_b32", \
        sgpr(tmpSgpr), \
        hex(kernel["MacroTile%u"%tP["tensorIdx"]]), \
        "MT%u"%tP["tensorIdx"] )
    kStr += inst("v_mul_lo_u32", \
        vgpr(sgid), \
        vgpr(sgid), \
        sgpr(tmpSgpr), \
        "sgid*sgid*MT%u"%tP["tensorIdx"] )
    #kStr += dump(vgpr(sgid))
    if kernel["VectorWidth"] > 1:
      #kStr += inst("v_lshlrev_b32", \
      #    vgpr(tP["gpr"]["lro"]), \
      #    log2(kernel["VectorWidth"]), \
      #    vgpr(tP["gpr"]["lro"]), \
      #    "lroA *= VW" )
      kStr += staticMultiply(vgpr(tP["gpr"]["lro"]), vgpr(tP["gpr"]["lro"]), kernel["VectorWidth"])
      #kStr += dump(vgpr(tP["gpr"]["lro"]))
    kStr += inst("v_add_u32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "vcc", \
        vgpr(sgid), \
        vgpr(tP["gpr"]["lro"]), \
        "o = lro%s*VW+sgid*MT%u"%(tP["tensorChar"], tP["tensorIdx"]) )
    kStr += inst("v_lshlrev_b32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        hex(log2(self.bpe)), \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "*= bytes/element" )
    """
    if tP["isB"]:
      kStr += inst("v_lshlrev_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(log2(self.bpe)), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "*= bytes/element" )
    """
    #kStr += dump(vgpr("LocalReadAddrA"))
    #kStr += "s_endpgm\n"

    self.vgprScratch.checkIn(tmpVgpr)
    self.vgprScratch.checkIn(qReg)
    self.vgprScratch.checkIn(rReg)
    self.vgprScratch.checkIn(tP["gpr"]["lro"])
    #kStr += "  unsigned int localReadOffsetA = lr%s*VECTOR_WIDTH + sgId*MT%s;%s" \
    #    % ( tP["tileChar"], tP["tileChar"], self.endLine)
    return kStr

  ##############################################################################
  # Local Read Addresses: Declare Addresses A/B - DONE
  ##############################################################################
  def lraDeclareAddresses(self, kernel, tP):
    if tP["isA"]:
      return self.comment1("N/A")
    else:
      return inst("v_add_u32", \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          "vcc", \
          hex(kernel["LdsOffset%s"%tP["tensorChar"]]*self.bpe), \
          vgpr("LocalReadAddr%s+0"%tP["tensorChar"]), \
          " += LdsOffset%s (lower)"%tP["tensorChar"])

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
    if not self.do["PreLoop"]: kStr += ".endif\n"

    tmp = self.vgprScratch.checkOut(1)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    if tailLoop:
      kStr += "%s//numIter%s = (((size%s %% LOCAL_DEPTHU) + LOCAL_SPLITU - 1) / LOCAL_SPLITU);%s" \
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

        # TODO doesn't support DU non-pow2
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

    #kStr += inst("v_mov_b32", vgpr(tmp), \
    #    sgpr("LoopCounters+0"), "" )
    #kStr += dump(vgpr(tmp))

    #kStr += "s_endpgm\n"
    self.vgprScratch.checkIn(tmp)
    return kStr

  ##############################################################################
  # Open Loop - DONE
  ##############################################################################
  def openLoop(self, kernel, loopIdx):
    kStr = ""
    #kStr += "s_endpgm\n"
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    loopLabelBegin = self.getLabel("%sLoopBegin%s"%("Tail" if tailLoop else "", loopChar) )
    loopLabelEnd = self.getLabel("%sLoopEnd%s"%("Tail" if tailLoop else "", loopChar) )
    kStr += "label_%04u:%s" % (loopLabelBegin, self.endLine)
    #kStr += self.indent + self.syncStr + self.endLine
    return kStr


  ##############################################################################
  # Close Loop - DONE
  ##############################################################################
  def closeLoop(self, kernel, loopIdx):
    kStr = ""
    #kStr += self.indent + self.syncStr + self.endLine
    #kStr += "s_endpgm\n"
    tailLoop = loopIdx < 0
    if tailLoop:
      loopIdx = self.unrollIdx
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    loopLabelBegin = self.getLabel("%sLoopBegin%s"%("Tail" if tailLoop else "", loopChar) )
    loopLabelEnd = self.getLabel("%sLoopEnd%s"%("Tail" if tailLoop else "", loopChar) )
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
    #kStr += "s_endpgm\n"
    #kStr += self.dumpLds(8, 8)
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
    if not self.do["MAC"]: return ""
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
  # Global Read: Increment A/B - DONE
  ##############################################################################
  def globalReadIncrement(self, kernel, loopIdx, tP):
    if not self.do["GlobalInc"]: return ""
    kStr = ""
    loopChar = self.indexChars[ \
        kernel["ProblemType"]["IndicesSummation"][loopIdx]]
    graIdx = 0
    tmp = self.startVgprSerial - 1
    #for perp in range(0, tP["nrp"]):
    #  for para in range(0, tP["nrc"]):
    #    for s in range(0, tP["nrcv"]):
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]/tP["nrcvpi"]):
            if self.globalReadIncsUseVgpr:
              kStr += inst("v_add_i32 ", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                  "vcc", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                  vgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], loopIdx)), \
                  "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
              kStr += inst("v_addc_u32", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  "vcc", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  vgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], loopIdx)), \
                  "vcc", \
                  "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
            else:
              kStr += inst("v_add_i32 ", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)), \
                  "vcc", \
                  vgpr("GlobalReadAddr%s+%u+0"%(tP["tensorChar"], graIdx)),  \
                  sgpr("GlobalReadIncs%s+%u+0"%(tP["tensorChar"], loopIdx)), \
                  "gra += inc%s%s (lower)"%(tP["tensorChar"], loopChar))
              kStr += inst("v_mov_b32 ", \
                  vgpr(tmp), \
                  sgpr("GlobalReadIncs%s+%u+1"%(tP["tensorChar"], loopIdx)), \
                  "vgpr GlobalReadIncs%s"%tP["tensorChar"] )
              kStr += inst("v_addc_u32", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  "vcc", \
                  vgpr("GlobalReadAddr%s+%u+1"%(tP["tensorChar"], graIdx)), \
                  vgpr(tmp), \
                  "vcc", \
                  "gra += inc%s%s (upper)"%(tP["tensorChar"], loopChar))
            graIdx += self.rpga
    #kStr += dump(vgpr("GlobalReadAddrA+0"))
    #kStr += dump(vgpr("GlobalReadAddrA+1"))
    #kStr += "s_endpgm\n"
    return kStr

  ##############################################################################
  # Global Read: Do It A/B - DONE
  ##############################################################################
  def globalReadDo(self, kernel, guardK, tP):
    if not self.do["GlobalRead"]: return ""
    kStr = ""
    graIdx = 0
    g2lIdx = 0
    loadWidth = tP["globalReadInstruction"].totalWidth
    #for perp in range(0, tP["nrp"]):
    #  for para in range(0, tP["nrc"]):
    #    for s in range(0, tP["nrcv"]):
    for perp in range(0, tP["nrp"]):
      for sPerp in range(0, tP["nrpv"]):
        for para in range(0, tP["nrc"]):
          for sPara in range(0, tP["nrcv"]/tP["nrcvpi"]):
            kStr += tP["globalReadInstruction"].toString( \
                (vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), loadWidth), \
                vgpr("GlobalReadAddr%s+%u"%(tP["tensorChar"], graIdx),2)), \
                "G -> Reg %u_%u_%u_%u"%(para, sPara, perp, sPerp ) )
            #kStr += "s_endpgm\n"
            graIdx += self.rpga
            g2lIdx += loadWidth
    return kStr

  ##############################################################################
  # Local Write: Swap Offsets A/B - DONE
  ##############################################################################
  def localWriteSwapOffsets(self, kernel, tP):
    if not self.do["LocalWrite"]: return ""
    kStr = ""
    kStr += inst("v_xor_b32", \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        hex(kernel["LdsOffsetA_Blk"]*self.bpe), \
        vgpr("LocalWriteAddr%s"%tP["tensorChar"]), \
        "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Write: Reset Offsets A/B - SKIP
  # used for global-read + tail-loop to reset to writing in red
  ##############################################################################
  def localWriteResetOffsets(self, kernel, tP):
    return ""

  ##############################################################################
  # Local Write: Init Pointers A/B - DONE
  ##############################################################################
  def localWriteInitPointers(self, kernel, tP):
    return self.comment1("N/A")

  ##############################################################################
  # Local Write: Do It A/B - DONE
  ##############################################################################
  def localWriteDo(self, kernel, tP):
    if not self.do["LocalWrite"]: return ""
    kStr = ""
    instruction = tP["localWriteInstruction"]
    numBlocks = instruction.numBlocks
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    totalWrites = len(tP["localWriteOffsets"])/numOffsets
    g2lIdx = 0
    graIdx = 0
    for graIdx in range(0, totalWrites):
      paramList = []
      paramList.append(vgpr("LocalWriteAddr%s"%tP["tensorChar"]))
      for blockIdx in range(0, numBlocks):
        if blockWidth == 1:
          paramList.append(vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx)))
        else:
          paramList.append( vgpr("G2L%s+%u"%(tP["tensorChar"], g2lIdx), \
              blockWidth))
      for oIdx in range(0, numOffsets):
        paramList.append(tP["localWriteOffsets"][graIdx*numOffsets+oIdx])

      paramTuple = tuple(paramList)
      comment = "Reg -> L %u"%graIdx
      kStr += tP["localWriteInstruction"].toString(paramTuple, comment)
      graIdx += 1
      g2lIdx += blockWidth
    return kStr

  ##############################################################################
  # Local Read: Swap Offsets A/B - DONE
  ##############################################################################
  def localReadSwapOffsets(self, kernel, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    kStr += inst("v_xor_b32", \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        hex(kernel["LdsOffsetA_Blk"]*self.bpe), \
        vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
        "swap Red Blk")
    return kStr

  ##############################################################################
  # Local Read: Reset Offsets A/B - DONE
  # x % n == n & (n-1) for n power of 2
  ##############################################################################
  def localReadResetOffsets(self, kernel, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    if tP["localReadInstruction"].numOffsets == 1:
      tP["localReadOffset"] = 0
      kStr += self.comment1("N/A")
    else:
      kStr += inst("v_and_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffset%s_Blk"%tP["tensorChar"]]*self.bpe-1), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "reset Red,Blk -> Red")
    return kStr

  ##############################################################################
  # Local Read: Init Pointers A/B - DONE
  ##############################################################################
  def localReadInitPointers(self, kernel, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    if self.localReadInstructionA.numOffsets == 1:
      tP["localReadOffset"] = 0
      kStr += self.comment1("N/A")
    else:
      kStr += inst("v_and_b32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(kernel["LdsOffset%s_Blk"%tP["tensorChar"]]*self.bpe-1), \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "reset Red,Blk -> Red")
    return kStr

  ##############################################################################
  # Local Read: Increment A/B - DONE
  ##############################################################################
  def localReadInc(self, kernel, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    if tP["localReadInstruction"].numOffsets == 1:
      tP["localReadOffset"] += kernel["LocalSplitU"]*kernel["MacroTile%u"%tP["tensorIdx"]]
      kStr += self.comment1("N/A")
    else:
      inc = kernel["LocalSplitU"]*kernel["MacroTile%u"%tP["tensorIdx"]]
      kStr += inst("v_add_i32", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          "vcc", \
          vgpr("LocalReadAddr%s"%tP["tensorChar"]), \
          hex(inc), \
          "lr%s += %u"%(tP["tensorChar"], inc) )
    return kStr

  ##############################################################################
  # Local Read: Do It A/B - DONE
  ##############################################################################
  def localReadDo(self, kernel, black, tP):
    if not self.do["LocalRead"]: return ""
    kStr = ""
    instruction = tP["localReadInstruction"]
    numBlocks = instruction.numBlocks
    numOffsets = instruction.numOffsets
    blockWidth = instruction.blockWidth
    offsetMultiplier = 1 # instruction.offsetMultiplier
    totalReads = (kernel["ThreadTile%u"%tP["tensorIdx"]]/kernel["VectorWidth"]) / numOffsets
    valuIdx = 0
    for lrIdx in range(0, totalReads):
      paramList = []
      if blockWidth == 1:
        paramList.append(vgpr("Valu%s%s+%u"%( \
            "Blk" if black else "", tP["tensorChar"], valuIdx)))
      else:
        paramList.append( vgpr("Valu%s%s+%u"%( \
            "Blk" if black else "", tP["tensorChar"],valuIdx), \
            blockWidth))
      paramList.append(vgpr("LocalReadAddr%s"%tP["tensorChar"]))
      for oIdx in range(0, numOffsets):
        paramList.append((kernel["SubGroup%u"%tP["tensorIdx"]]*(lrIdx*numOffsets+oIdx)*kernel["VectorWidth"] \
            + tP["localReadOffset"])*self.bpe/offsetMultiplier)
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
    #print "GlobalWriteIndices"
    if not self.do["PostLoop"]: return ""
    kStr = ""
    tmp = self.vgprScratch.checkOut(2)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
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
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    dividendReg = "Serial" # local serial
    rReg = self.local01 + 0
    qReg = self.local01 + 1
    tmpVgpr = self.vgprScratch.checkOut(1)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    tmpSgpr = s0
    divisor = kernel["SubGroup0"]
    kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, \
        tmpVgpr, tmpSgpr)
    tid0 = rReg
    tid1 = qReg
    #kStr += inst("v_lshlrev_b32", \
    #    vgpr(tid0), \
    #    log2(self.bpe*kernel["VectorWidth"]), \
    #    vgpr(tid0), \
    #    "*= %u VW*bytes/element"%(self.bpe*kernel["VectorWidth"]))
    kStr += staticMultiply(vgpr(tid0), vgpr(tid0), (self.bpe*kernel["VectorWidth"]))
    #kStr += inst("v_lshlrev_b32", \
    #    vgpr(tid1), \
    #    log2(self.bpe*kernel["VectorWidth"]), \
    #    vgpr(tid1), \
    #    "*= %u VW*bytes/element"%(self.bpe*kernel["VectorWidth"]))
    kStr += staticMultiply(vgpr(tid1), vgpr(tid1), (self.bpe*kernel["VectorWidth"]))

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
    #kStr += inst("v_mov_b32", vgpr(tmp), sgpr("AddressC+1"), "" )
    kStr += inst("v_addc_u32", \
        vgpr(self.globalWriteAddrC+1), \
        "vcc", \
        vgpr(tmp+1), \
        vgpr(self.globalWriteAddrC+1), \
        "vcc", \
        "%s += C (upper)"%vgpr(self.globalWriteAddrC+1))

    #kStr += dump(vgpr(self.globalWriteAddrC+0))
    #kStr += dump(vgpr(self.globalWriteAddrC+1))
    #kStr += "s_endpgm\n"

    self.vgprScratch.checkIn(tmpVgpr)
    self.vgprScratch.checkIn(tmp)
    self.vgprScratch.checkIn(self.local01)
    self.globalWriteTmp = self.vgprScratch.checkOut(4)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"

    """
    kStr += "GLOBAL_OFFSET_C( vgprGlobalReadAddrA+%u"%graIdx
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
    if not self.do["PostLoop"]: return ""
    kStr = ""
    kStr += self.comment1("GLOBAL_WRITE vc0 vc1 tt0 tt1%s" % (self.endLine) )
    #kStr += "s_endpgm\n"
    #kStr += "GLOBAL_WRITE 0 0 0 0%s" % (self.endLine)
    #kStr += "s_endpgm\n"
    for tt1 in range(0, kernel["ThreadTile1"]/kernel["VectorWidth"]):
      for tt0 in range(0, kernel["ThreadTile0"]/kernel["VectorWidth"]):
        for vc1 in range(0, kernel["VectorWidth"]):
          for vc0 in range(0, kernel["VectorWidth"]):
            kStr += "GLOBAL_WRITE %u %u %u %u %u %u%s" % (vc0, vc1, tt0, tt1, \
                self.globalWriteAddrC, self.globalWriteTmp, self.endLine)
    self.vgprScratch.checkIn(self.globalWriteTmp)
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
  def kernelBodyPrefix(self, kernel, tPA, tPB ):
    return ""

  ##############################################################################
  # Kernel Body Suffix - DONE
  ##############################################################################
  def kernelBodySuffix(self, kernel, tPA, tPB ):
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
  def wait(self, kernel, tPA, tPB, skipGlobalRead, skipLocalWrite, \
      skipLocalRead, comment):
    if not self.do["Wait"]: return ""
    # skip = -1 -> ignore
    # skip =  n -> waitcnt(n*num)

    lgkmcnt = 0 if skipLocalWrite > -1 or skipLocalRead > -1 else -1

    if skipLocalWrite > -1 or skipLocalRead > -1:
      if skipLocalWrite > -1:
        numA = len(tPA["localWriteOffsets"]) \
            / tPA["localWriteInstruction"].numOffsets
        numB = len(tPB["localWriteOffsets"]) \
            / tPB["localWriteInstruction"].numOffsets
        lgkmcnt += skipLocalWrite * (numA + numB)
      if skipLocalRead > -1:
        numA = (kernel["ThreadTile0"] / kernel["VectorWidth"]) \
            / self.localReadInstructionA.numOffsets
        numB = (kernel["ThreadTile1"] / kernel["VectorWidth"]) \
            / self.localReadInstructionB.numOffsets
        lgkmcnt += skipLocalRead * (numA + numB)

    vmcnt = 0 if skipGlobalRead > -1 else -1
    if skipGlobalRead > -1:
      numA = kernel["NumLoadsPerpendicularA"] * kernel["NumLoadsCoalescedA"] \
          * self.numReadVectorComponentsA
      numB = kernel["NumLoadsPerpendicularB"] * kernel["NumLoadsCoalescedB"] \
          * self.numReadVectorComponentsB
      vmcnt += skipGlobalRead * (numA + numB)
      if lgkmcnt > -1:
        lgkmcnt += skipGlobalRead * (numA + numB)

    if False:
      return "s_waitcnt lgkmcnt(0) & vmcnt(0) // debug%s" % self.endLine

    lgkmcnt = min(lgkmcnt, 15)
    vmcnt = min(vmcnt, 15)

    kStr = ""
    kStr += "s_waitcnt "
    if lgkmcnt >= 0:
      kStr += "lgkmcnt(%u)"%lgkmcnt
    #if lgkmcnt >= 0 and vmcnt >= 0:
    #  kStr += " & "
    elif vmcnt >= 0:
      kStr += "vmcnt(%u)"%vmcnt
    kStr += " // %s" % comment
    kStr += self.endLine
    return kStr

  ##############################################################################
  # SyncThreads
  ##############################################################################
  def syncThreads(self, kernel):
    if kernel["NumThreads"] > 64 and self.do["Sync"]:
      return self.indent + self.syncStr + self.endLine
    else:
      return ""


  ########################################
  # dump lds state
  ########################################
  def dumpLds(self, kernel, startU, numU):
    kStr = self.comment("dump lds state")
    kStr += inst("s_barrier", "" )
    kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "" )
    tmp = self.vgprScratch.checkOut(1)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    tmpAddr = self.vgprScratch.checkOut(1)
    if self.vgprScratch.overflowed(): kStr += "s_endpgm\n"
    kStr += inst("v_lshlrev_b32", \
        vgpr(tmpAddr), \
        hex(log2(self.bpe)), \
        vgpr("Serial"), \
        "dump lds")
    for i in range(startU, startU+numU):
      kStr += inst("ds_read_b32", vgpr(tmp), \
          vgpr(tmpAddr) + " offset:%u"%(i*kernel["NumThreads"]*4), "dump lds")
      kStr += inst("s_waitcnt", "lgkmcnt(0) & vmcnt(0)", "dump" )
      kStr += dump(vgpr(tmp))
    self.vgprScratch.checkIn(tmp)
    self.vgprScratch.checkIn(tmpAddr)
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
  if globalParameters["DebugKernel"]:
    kStr += inst("flat_store_dword", vgpr("AddressD", 2), \
        vgprStore, "debug dump store" )
    kStr += inst("v_add_i32", vgpr("AddressD"), "vcc", vgpr("AddressD"), \
        hex(4), "debug dump inc" )
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
# quotient register, remainder register, dividend register, divisor, tmpVgprx2, tmpSgpr
########################################
def vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, \
    doRemainder=True):
  kStr = ""
  if ((divisor & (divisor - 1)) == 0): # pow of 2
    divisor_log2 = log2(divisor)
    kStr += inst("v_lshrrev_b32", vgpr(qReg), divisor_log2, vgpr(dReg), \
        "%s = %s / %u"%(vgpr(qReg), vgpr(dReg), divisor) )
    if doRemainder:
      kStr += inst("v_and_b32", vgpr(rReg), (divisor-1), vgpr(dReg), \
          "%s = %s %% %u"%(vgpr(rReg), vgpr(dReg), divisor) )

  elif (((divisor/3) & ((divisor/3) - 1)) == 0): # 3 * pow of 2
    shift = 33 + log2(divisor/3)
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), "0xaaaaaaab", "")
    kStr += inst("v_mul_hi_u32", vgpr(tmpVgpr+1), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr+0), vgpr(dReg), sgpr(tmpSgpr), "")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(shift), "")
    kStr += inst("v_lshrrev_b64", vgpr(tmpVgpr,2), sgpr(tmpSgpr), vgpr(tmpVgpr,2), "")
    kStr += inst("v_mov_b32", vgpr(qReg), vgpr(tmpVgpr), "quotient")
    #kStr += dump(vgpr(qReg))
    if doRemainder:
      kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), "divisor")
      #kStr += inst("v_mov_b32", vgpr(tmpVgpr), sgpr(tmpSgpr), "")
      #kStr += dump(vgpr(tmpVgpr))
      kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr), vgpr(qReg), sgpr(tmpSgpr), "product = quotient * divisor")
      #kStr += dump(vgpr(tmpVgpr))
      kStr += inst("v_sub_u32", vgpr(rReg), "vcc", vgpr(dReg), vgpr(tmpVgpr), "remainder = dividend - product")
      #kStr += dump(vgpr(rReg))

  else:
    printExit("KernelWriterAssembly::divmod doesn't support %u" % divisor)
  return kStr

def vectorStaticDivide(qReg, dReg, divisor, tmpVgpr, tmpSgpr):
  rReg = -1 # unused
  kStr = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr, False)
  return kStr

def scalarStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpSgpr, \
    doRemainder=True):
  kStr = ""
  if ((divisor & (divisor - 1)) == 0): # pow of 2
    divisor_log2 = log2(divisor)
    kStr += inst("s_lshr_b32", sgpr(qReg), sgpr(dReg), divisor_log2, \
        "%s = %s / %u"%(sgpr(qReg), sgpr(dReg), divisor) )
    #kStr += dump(sgpr(qReg))
    if doRemainder:
      kStr += inst("s_and_b32", sgpr(rReg), (divisor-1), sgpr(dReg), \
          "%s = %s %% %u"%(sgpr(rReg), sgpr(dReg), divisor) )
      #kStr += dump(sgpr(rReg))
  elif (((divisor/3) & ((divisor/3) - 1)) == 0): # 3 * pow of 2 TODO FIXME
    printExit("KernelWriterAssembly::scalarStaticDivide doesn't support %u" % divisor)
    shift = 32 + log2(divisor/3)
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), "0xaaaaaaab", "tmp = magic")
    kStr += inst("s_mul_i32", sgpr(tmpSgpr+1), sgpr(dReg), sgpr(tmpSgpr), "tmp1 = dividend * magic")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(shift), "tmp = shift")
    kStr += inst("s_lshr_b32", sgpr(tmpSgpr+1), sgpr(tmpSgpr+1), tmpSgpr, "tmp1 = (dividend * magic) << shift")
    kStr += inst("s_mov_b32", sgpr(tmpSgpr), hex(divisor), "tmp = divisor")
    kStr += inst("v_mul_i32", sgpr(qReg), sgpr(tmpSgpr+1), sgpr(tmpSgpr), "qReg = ( (dividend*magic)<<shift )*divisor")
    if doRemainder:
      kStr += inst("v_sub_u32", sgpr(rReg), "vcc", sgpr(dReg), sgpr(tmpSgpr), "rReg = dividend - divisor")
  else:
    printExit("KernelWriterAssembly::divmod doesn't support %u" % divisor)
  return kStr

#def staticRemainder(rReg, dReg, divisor, tmpVgpr, tmpSgpr):
#  qReg = self.vgprScratch.checkOut(1)
#  kStr = vectorStaticDivideAndRemainder(qReg, rReg, dReg, divisor, tmpVgpr, tmpSgpr)
#  self.vgprScratch.checkIn(qReg)
#  return kStr

########################################
# Multiply
# product register, operand register, multiplier
# TODO clarify support of sgpr and constant operand/multipliers
########################################
def staticMultiply(product, operand, multiplier):
  if ((multiplier & (multiplier - 1)) == 0): # pow of 2
    multiplier_log2 = log2(multiplier)
    return inst("v_lshlrev_b32", product, multiplier_log2, operand, \
        "%s = %s * %u"%(product, operand, multiplier) )
  else:
    if True: # operand.startswith("s["):
      kStr = ""
      kStr += inst("v_mov_b32", product, multiplier, \
        "%s = %u"%(product, multiplier) )
      kStr += inst("v_mul_lo_u32", product, product, operand, \
        "%s *= %s"%(product, operand) )
      return kStr
    else:
      return inst("v_mul_u32_u24", product, operand, hex(multiplier), \
          "%s = %s * %u"%(product, operand, multiplier) )
    """
    VOP3 (cannot use literal constant)
    v_mul_lo_u32
    v_mul_hi_u32
    v_mul_hi_i32
    VOP2
    v_mul_i32_i24
    v_mul_u32_u24
    v_mul_hi_i32_i24
    v_mul_hi_u32_u24

    """

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

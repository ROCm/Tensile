################################################################################
#
# Copyright (C) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

from ..Component import LocalRead
from .. import Code
from ..AsmUtils import vgpr, sgpr
from math import ceil

class LocalReadVALU(LocalRead):
    kernel = {"EnableMatrixInstruction": False}

    """
    Local Read: Do It A/B
    iui = Inner Unroll Idx
    epsi = expand pointer swap index. Only used for PAP
    """
    def __call__(self, writer, bufferIdx, iui, epsi, tP):
        kernel = writer.kernel

        writer.localReadDoCnt += 1

        tc                = tP["tensorChar"]
        tile01            = tP["tile01Idx"]
        imod              = Code.Module("LocalReadDo%s_I%s"%(tc,iui))
        pack              = Code.Module("pack%s_I%s"%(tc,iui))
        instruction       = tP["localReadInstruction"]
        numOffsets        = instruction.numOffsets
        blockWidth        = instruction.blockWidth
        offsetMultiplier  = 1 # instruction.offsetMultiplier
        valuIdx           = 0
        numVectorsPerTile = (kernel["ThreadTile%u"%tile01]//kernel["VectorWidth"])
        numReadsPerVector = (kernel["VectorWidth"] * tP["bpe"]) // (blockWidth*4) # bytes/register
    
        for vIdx in range(0, numVectorsPerTile):
            for rIdx in range(0, numReadsPerVector):
                localReadCode = imod.addCode (Code.Module("LocalRead%s Valu%u"%(tc,valuIdx)))
                paramList     = []
                destVgpr      = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuIdx), blockWidth)

                paramList.append(destVgpr)
                paramList.append(vgpr("LocalReadAddr%s"%tc))

                for oIdx in range(0, numOffsets):
                    paramList.append(((rIdx*blockWidth + kernel["SubGroup%u"%tile01] * (vIdx*numOffsets+oIdx)*kernel["VectorWidth"] \
                      + tP["localReadOffset"]) * tP["bpe"] + tP["localReadSwapByteOffset"]) // offsetMultiplier)
                    # print("Debug: Matrix{}, rIdx offset {}, vIdx offset {}, bpe {}, net offset {}".format( \
                    #     tP["tensorChar"], \
                    #     rIdx * blockWidth, \
                    #     kernel["SubGroup%u" % tP["tensorIdx"]] * (vIdx * numOffsets + oIdx) * kernel["VectorWidth"] + tP["localReadOffset"], \
                    #     tP["bpe"], \
                    #     paramList[-1]))
                paramTuple = tuple(paramList)
                comment = "L -> Reg lro=%d swapByteOffset=%u ti=%u vIdx=%u rIdx=%u oIdx=%u buffer=%u iui=%u"\
                    %(tP["localReadOffset"],tP["localReadSwapByteOffset"],kernel["SubGroup%u"%tile01], vIdx, rIdx, oIdx, bufferIdx, iui)
                localReadCode.addCode(Code.LocalReadInst(instruction.IssueLatency,False,instruction.toCodeInst(paramTuple), comment))
                valuIdx += blockWidth

                # TODO - handle vector-load
                tmpSgpr = writer.getTmpSgpr(1).idx()
                if writer.db["CheckValue1%s" % tc]:
                    dbgVgpr = destVgpr
                    dbgVgprList = destVgpr.split("v[")
                    if len(dbgVgprList) == 1: # vIdx, no []
                        dbgVgpr = dbgVgprList[0]
                    else:
                        # We only check the first one now
                        # TODO: Handle vector, but need to take care the last one
                        dbgVgprList = (dbgVgprList[1].split("]")[0]).split(':')
                        dbgVgpr = "v[%s]"%dbgVgprList[0]

                    localReadCode.addInst("s_waitcnt lgkmcnt(0)", "CheckValue1 wait for LDS read")
                    if writer.archCaps["SeparateVscnt"]:
                        localReadCode.addInst( "s_waitcnt_vscnt", "null", "0", "")

                    if kernel["ProblemType"]["DataType"].isHalf():
                        localReadCode.addInst("s_mov_b32", sgpr(tmpSgpr), hex(0x3c003c00),"CheckValue1: FP16")   # packed 1s
                        localReadCode.addCode(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                    elif kernel["ProblemType"]["DataType"].isBFloat16():
                        localReadCode.addInst("s_mov_b32", sgpr(tmpSgpr), hex(0x3f803f80),"CheckValue1: BF16")   # packed 1s
                        localReadCode.addCode(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                    # TODO - Check if this works
                    if kernel["ProblemType"]["DataType"].isInt8():
                        localReadCode.addInst("s_mov_b32", sgpr(tmpSgpr), hex(0x01010101),"CheckValue1: INT8")   # packed 1s
                        localReadCode.addCode(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                    # TODO - Check if this works
                    elif kernel["ProblemType"]["DataType"].isInt8x4():
                        localReadCode.addCode(writer.assert_eq( dbgVgpr, 1))

                    elif kernel["ProblemType"]["DataType"].isSingle():
                        localReadCode.addCode(writer.assert_eq( dbgVgpr, 1.0) )
    
        return imod, pack


class LocalReadMFMA(LocalRead):
    kernel = {"EnableMatrixInstruction": True}

    """
    Local Read: Do It A/B
    iui = Inner Unroll Idx
    epsi = expand pointer swap index. Only used for PAP
    """
    def __call__(self, writer, bufferIdx, iui, epsi, tP):
        kernel = writer.kernel

        writer.localReadDoCnt += 1

        tc               = tP["tensorChar"]
        imod = Code.Module("LocalReadDo%s_I%s" % (tc,iui))

        if tc == "A":
            lrvw = writer.lrvwA
            writer.localReadDoCntA += 1
        else:
            lrvw = writer.lrvwB
            writer.localReadDoCntB += 1
        tile01           = tP["tile01Idx"]
        instruction      = tP["localReadInstruction"]

        numOffsets       = instruction.numOffsets
        blockWidth       = instruction.blockWidth
        vectorWidthA     = kernel["VectorWidth"] if kernel["SourceSwap"] else 1 # TODO: nonSwap VectorWidth
        vectorWidthB     = writer.VectorWidthB
        MIWaveGroupShape = [ kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0] * vectorWidthA, \
                             kernel["MatrixInstN"] * kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] * vectorWidthB]

        LdsPad           = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
        tileStride       = 1
        UnrollStride     = kernel["MacroTile%s" % tc] + LdsPad
        if kernel["UnrollMajorLDS%s" % tc]:
            tileStride   = kernel["_DepthULds"] + LdsPad
            UnrollStride = 1

        vectorWidth          = vectorWidthA if (tile01 == 0) else vectorWidthB
        #numReadPerTileVector = (vectorWidth * tP["bpe"]) // int(blockWidth * 4) # bytes/register
        numReadPerTileVector = vectorWidth if (tile01 == 0) else 1
        numVectorsPerTile    = kernel["MIWaveTile"][tile01] // vectorWidth
        # overloading numReadsPerUnroll for DirectToLds x2/x4 case when blockWidth of instruction < LocalReadVectorWidth
        # fp64 TLU=1 reading 0.5element/lane/read..
        # for TLU=0 case, blockWidth and LRVW should match
        numReadsPerUnroll = tP["bpe"] * lrvw // int(blockWidth * 4) # bytes/register
        numVgpr  = int(ceil(blockWidth))
        numElementPerRead = int(blockWidth * 4) // tP['bpe']

        # pack register
        pack     = Code.Module("pack%s_I%s"%(tc,iui))
        hasEccHalf = writer.archCaps["HasEccHalf"]
        needPack = (blockWidth < 1) if hasEccHalf else (blockWidth == 0.25)
        if needPack and (not kernel["VgprForLocalReadPacking"]):
            # allcate tmp vgpr only for no VgprForLocalReadPacking case
            # No ECC 0.25: pack one time 0x00ff00ff | (0x00ff00ff << 8)
            packTimesPerVgpr = (int(1/blockWidth) - 1) if writer.archCaps["HasEccHalf"] else 1
            tmpVgprIdx = writer.vgprPool.checkOut(writer.numVgprValuAPerBlock*writer.numReadsIterCoalescedA*packTimesPerVgpr if tc == 'A' \
                else writer.numVgprValuBPerBlock*writer.numReadsIterCoalescedB*packTimesPerVgpr, "local read pack")
            pack.addTempVgpr(tmpVgprIdx) # important, add to pack Module for later CheckIn

        valufIdx = 0
        for vIdx in range(0, numVectorsPerTile):
            for eIdx in range(0, numReadPerTileVector):
                valuiIdx = int(valufIdx)
                localReadCode = imod.addCode (Code.Module("LocalRead%s Valu%u"%(tc,valuiIdx)))
                if needPack:
                    packCode = pack.addCode (Code.Module("packCode"))
                for rIdx in range(0, numReadsPerUnroll):
                    valuiIdx = int(valufIdx)
                    baseLRVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx), numVgpr)
                    destVgpr = baseLRVgpr

                    highBitsForHalf = (blockWidth == 0.5) and ((rIdx % 2) == 1) # rIdx = 1
                    isHigh8Bits  = (blockWidth == 0.25) and ( ((rIdx % 4) % 2) == 1) # 1,3
                    isHigh16Bits = (blockWidth == 0.25) and ( ((rIdx % 4) //2) == 1) # 2,3

                    if needPack:
                        if kernel["VgprForLocalReadPacking"]:
                            # use allocated vgpr
                            highVgprBase = "Valu%s_X%u_I%u"%(tc, bufferIdx, iui)
                            highVgprHalf = vgpr(highVgprBase + "_D%u+%u"%(rIdx%2, valuiIdx), numVgpr)
                            highVgpr8Bits = vgpr(highVgprBase + "_D%u+%u"%(rIdx%4, valuiIdx), numVgpr)
                            lowVgpr = vgpr(highVgprBase + "_D%u+%u"%((rIdx%4)-1, valuiIdx), numVgpr) if isHigh16Bits else baseLRVgpr
                        else:
                            highVgprHalf = vgpr(tmpVgprIdx)
                            highVgpr8Bits = vgpr(tmpVgprIdx)
                            lowVgpr = vgpr(tmpVgprIdx-1) if isHigh16Bits else baseLRVgpr

                        if hasEccHalf: # ECC pack
                            # pack for ECC blockWidth 0.5 type
                            if needPack and highBitsForHalf:
                                highVgpr = highVgprHalf
                                packCode.addInst("v_or_b32", destVgpr, destVgpr, highVgpr, "pack two half Vgpr to one Vgpr")
                                destVgpr = highVgpr

                            # pack for ECC blockwidth 0.25 type
                            if isHigh8Bits or isHigh16Bits:
                                highVgpr = highVgpr8Bits
                                destVgpr = highVgpr
                            if isHigh8Bits:
                                packCode.addInst("_v_lshl_or_b32", lowVgpr, highVgpr, "0x8", lowVgpr, "pack two int8 Vgpr to one half Vgpr")
                                if isHigh16Bits:
                                    packCode.addInst("v_or_b32", baseLRVgpr, baseLRVgpr, lowVgpr, "pack two half Vgpr to one Vgpr")

                            if (not kernel["VgprForLocalReadPacking"]) and (highBitsForHalf or isHigh8Bits or isHigh16Bits):
                                tmpVgprIdx += 1

                        else: # no ECC pack
                            # pack for No ECC blockwidth 0.25 type
                            if isHigh8Bits:
                                highVgpr = vgpr(tmpVgprIdx)
                                destVgpr = highVgpr
                            if isHigh8Bits and isHigh16Bits:
                                packCode.addInst("_v_lshl_or_b32", baseLRVgpr, highVgpr, "0x8", baseLRVgpr, "pack two int8x2 Vgpr to one Vgpr")
                                tmpVgprIdx += 1

                    valufIdx += blockWidth

                    # load read instrution
                    paramList = []
                    paramList.append(destVgpr)
                    paramList.append(vgpr("LocalReadAddr%s"%tc))

                    for oIdx in range(0, numOffsets):
                        localReadOffset = tP["localReadOffset"]
                        tileStride2 = tileStride
                        localReadOffsetDiv = 0
                        if kernel["DirectToLds%s" % tc] and kernel["ThreadSeparateGlobalRead%c"%tc] and kernel["UnrollMajorLDS%s" % tc]:
                          ldsWidth = tileStride
                          ldsWidth -= LdsPad # do not include LdsPad
                          lrdOffsetMod = ldsWidth // (kernel["ThreadSeparateGlobalRead%c"%tc]*2)
                          tileStride2 //= (kernel["ThreadSeparateGlobalRead%c"%tc]*2) # change tileStride only for eIdx
                          localReadOffsetDiv = localReadOffset // lrdOffsetMod
                          localReadOffset = localReadOffset % lrdOffsetMod # keep only mod of lrdOffsetMod
                        offset_val = eIdx * tileStride2 + (vIdx * numOffsets+oIdx) * MIWaveGroupShape[tile01] * tileStride
                        if (kernel["DirectToLds%s" % tc] and  \
                            kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"] > 4 and \
                            (tP["bpe"] > 4 or tP["tlu"] == False and tP["bpe"] * lrvw > writer.bpr)):
                          # directToLds special case (local read before conversion is larger than bpr)
                          # In this case, we compose the original single local read(b64 or b128) with multiple local reads (b32)
                          # divVal represents the number of b32 local reads to compose single original b64 or b128 local read
                          # rIdxMod represents the location of split local read
                          # A original wide local read loads consecutive 2 (for b64) or 4 (for b128) values from LDS.
                          # rIdxDiv represents the original rIdx before conversion
                          divVal = tP["bpe"] * lrvw // writer.bpr
                          rIdxMod = rIdx % divVal
                          rIdxDiv = rIdx // divVal
                          offset_val = (rIdxDiv * UnrollStride + offset_val + localReadOffset) * tP["bpe"] * blockWidth + rIdxMod * writer.bpr
                        else:
                          # normal case
                          offset_val = (rIdx * numElementPerRead * UnrollStride + offset_val + localReadOffset) * tP["bpe"]
                        if localReadOffsetDiv > 0:
                          # TSGR special conversion
                          # Multiply BlockSize for each lrdOffsetMod
                          MblockSizePerLoad = (kernel["WavefrontSize"] * kernel["GlobalLoadVectorWidth%c"%tc]) // ldsWidth
                          offset_val += ((MblockSizePerLoad * lrdOffsetMod * tP["bpe"]) * localReadOffsetDiv)
                        if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                            offset_val = offset_val + (offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
                        offset_val = offset_val + tP["localReadSwapByteOffset"]
                        if kernel["DirectToLds%s" % tc]:
                          # another address conversion for DirectToLds + NumLoadsCoalesced > 1
                          dummy, offset_val = writer.lraOffsetConversionForDTLandNLC(kernel, tP, offset_val)

                          if kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"] > 4:
                            # magic offset conversion for DirectToLds
                            # TLU=0 case, modify bit3-6 of offset_val as follows
                            # (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
                            bit2 = offset_val & 4
                            bit3 = offset_val & 8
                            bit4 = offset_val & 16
                            bit5 = offset_val & 32
                            if (kernel["GlobalLoadVectorWidth%s"%tc] * tP["bpe"] == 8):
                              # dword_x2 case
                              # (bit2<<3) | (bit3 >>1) | (bit4>>1) | (bit5>>1)
                              newVal = (bit2<<3) | (bit3 >>1) | (bit4>>1) | (bit5>>1)
                            else:  #if (kernel["GlobalLoadVectorWidth%s"%tc] * tP["bpe"] == 16):  # most preferred case
                              # dword_x4 case
                              # (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
                              newVal = (bit2<<3) | (bit3 <<1) | (bit4>>2) | (bit5>>2)
                            offset_val = offset_val & (~0x3c)
                            offset_val = offset_val | newVal

                        paramList.append(int(offset_val))

                    paramTuple = tuple(paramList)
                    comment = "L -> Reg lro=%d swapByteOffset=%u ti=%u vIdx=%u rIdx=%u oIdx=%u buffer=%u iui=%u" \
                            % (tP["localReadOffset"], tP["localReadSwapByteOffset"], MIWaveGroupShape[tile01], vIdx, rIdx, oIdx, bufferIdx, iui)

                    highBits = highBitsForHalf or isHigh16Bits
                    readToTempVgpr = ((highBitsForHalf or isHigh8Bits or isHigh16Bits) if writer.archCaps["HasEccHalf"] else isHigh8Bits) and (not kernel["VgprForLocalReadPacking"])
                    localReadCode.addCode(Code.LocalReadInst(instruction.IssueLatency,readToTempVgpr,instruction.toCodeInst(paramTuple, 0, highBits), comment))

                    # TODO - handle vector-load
                    tmpSgpr = writer.getTmpSgpr(1).idx()
                    if writer.db["CheckValue1%s"%tc] and not writer.inTailLoop:
                        dbgVgpr = destVgpr
                        dbgVgprList = destVgpr.split("v[")
                        if len(dbgVgprList) == 1: # vIdx, no []
                            dbgVgpr = dbgVgprList[0]
                        else:
                            # We only check the first one now
                            # TODO: Handle vector, but need to take care the last one
                            dbgVgprList = (dbgVgprList[1].split("]")[0]).split(':')
                            dbgVgpr = "v[%s]"%dbgVgprList[0]

                        localReadCode.addInst("s_waitcnt lgkmcnt(0)", "CheckValue1 wait for LDS read")
                        if writer.archCaps["SeparateVscnt"]:
                            localReadCode.addInst( "s_waitcnt_vscnt", "null", "0", "")

                        if kernel["ProblemType"]["DataType"].isHalf():
                            hexValue = hex(0x3c003c00)     # packed 1s
                            if needPack:
                                hexValue = hex(0x3c000000) if highBitsForHalf else hex(0x00003c00)
                            localReadCode.addInst("s_mov_b32", sgpr(tmpSgpr), hexValue,"CheckValue1: FP16")
                            localReadCode.addCode(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                        elif kernel["ProblemType"]["DataType"].isBFloat16():
                            hexValue = hex(0x3f803f80)     # packed 1s
                            if needPack:
                                hexValue = hex(0x3f800000) if highBitsForHalf else hex(0x00003f80)
                            localReadCode.addInst("s_mov_b32", sgpr(tmpSgpr), hexValue,"CheckValue1: BF16")
                            localReadCode.addCode(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                        if kernel["ProblemType"]["DataType"].isInt8():
                            if needPack:
                                hexValue = hex(0x00010000) if isHigh16Bits else hex(0x00000001)
                                localReadCode.addInst("s_mov_b32", sgpr(tmpSgpr), hexValue,"CheckValue1: INT8")
                                localReadCode.addCode(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                        # TODO - Check if this works. But need this? MFMA would use INT8
                        elif kernel["ProblemType"]["DataType"].isInt8x4():
                            localReadCode.addInst("s_mov_b32", sgpr(tmpSgpr), hex(0x01010101),"CheckValue1: INT8x4")
                            localReadCode.addCode(writer.assert_eq( dbgVgpr, sgpr(tmpSgpr)))

                        elif kernel["ProblemType"]["DataType"].isSingle():
                            localReadCode.addCode(writer.assert_eq( dbgVgpr, 1.0) )

        return imod, pack

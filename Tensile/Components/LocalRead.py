################################################################################
# Copyright 2021 Advanced Micro Devices, Inc. All rights reserved.
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
                localReadCode.addCode(Code.LocalReadInst(instruction.IssueLatency,instruction.toCodeInst(paramTuple), comment))
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

        imod = Code.Module("LocalReadDo%s_I%s" % (tP["tensorChar"],iui))

        tc               = tP["tensorChar"]
        if tc == "A":
            writer.localReadDoCntA += 1
        else:
            writer.localReadDoCntB += 1
        tile01           = tP["tile01Idx"]
        instruction      = tP["localReadInstruction"]

        numOffsets       = instruction.numOffsets
        blockWidth       = instruction.blockWidth
        MIWaveGropuShape = [ kernel["MatrixInstM"] * kernel["MatrixInstBM"] * kernel["MIWaveGroup"][0], \
                             kernel["MatrixInstN"] * kernel["MatrixInstBN"] * kernel["MIWaveGroup"][1] ]

        LdsPad           = kernel["LdsPad%s"%tc] if kernel["LdsBlockSizePerPad%s"%tc] == 0 else 0
        tileStride       = 1
        UnrollStride     = kernel["MacroTile%s" % tP["tensorChar"]] + LdsPad
        if kernel["UnrollMajorLDS%s" % tP["tensorChar"]]:
            tileStride   = kernel["_DepthULds"] + LdsPad
            UnrollStride = 1

        numVectorsPerTile = kernel["MIWaveTile"][tile01]
        if tc == "A":
            numReadsPerVector = tP["bpe"] * writer.lrvwA // int(blockWidth * 4) # bytes/register
        else:
            numReadsPerVector = tP["bpe"] * writer.lrvwB // int(blockWidth * 4) # bytes/register
        numVgpr  = int(ceil(blockWidth))

        # pack register
        needPack = blockWidth < 1
        pack     = Code.Module("pack%s_I%s"%(tc,iui))
        if needPack:
            packTimesPerVgpr = int(1/blockWidth) - 1 # 0.5->pack once (16->32) / 0.25->pack three times (8->16, 8->16, 16->32)
            tmpVgprIdx = writer.vgprPool.checkOut(writer.numVgprValuAPerBlock*writer.numReadsIterCoalescedA*packTimesPerVgpr if tc == 'A' \
                else writer.numVgprValuBPerBlock*writer.numReadsIterCoalescedB*packTimesPerVgpr)
            pack.addTempVgpr(tmpVgprIdx) # important, add to pack Module for later CheckIn

        valufIdx = 0
        for vIdx in range(0, numVectorsPerTile):
            valuiIdx = int(valufIdx)
            localReadCode = imod.addCode (Code.Module("LocalRead%s Valu%u"%(tc,valuiIdx)))
            if needPack:
                packCode = pack.addCode (Code.Module("packCode"))

            for rIdx in range(0, numReadsPerVector):
                valuiIdx = int(valufIdx)
                baseLRVgpr = vgpr("Valu%s_X%u_I%u+%u"%(tc, bufferIdx, iui, valuiIdx), numVgpr)
                destVgpr = baseLRVgpr

                # pack for blockWidth 0.5 type
                highBitsForHalf = (blockWidth == 0.5) and ((rIdx % 2) == 1) # rIdx = 1
                if needPack and highBitsForHalf:
                    # highVgpr = vgpr(tmpVgprIdx + valuiIdx)
                    highVgpr = vgpr(tmpVgprIdx)
                    tmpVgprIdx += 1
                    packCode.addInst("v_or_b32", destVgpr, destVgpr, highVgpr, "pack two half Vgpr to one Vgpr")
                    destVgpr = highVgpr

                isHigh8Bits  = (blockWidth == 0.25) and ( ((rIdx % 4) % 2) == 1) # 1,3
                isHigh16Bits = (blockWidth == 0.25) and ( ((rIdx % 4) //2) == 1) # 2,3
                if needPack:
                    if isHigh8Bits or isHigh16Bits:
                        highVgpr = vgpr(tmpVgprIdx)
                        destVgpr = highVgpr
                    if isHigh8Bits:
                        lowVgpr = vgpr(tmpVgprIdx-1) if isHigh16Bits else baseLRVgpr
                        packCode.addInst("_v_lshl_or_b32", lowVgpr, highVgpr, "0x8", lowVgpr, "pack two int8 Vgpr to one half Vgpr")
                        if isHigh16Bits:
                            packCode.addInst("v_or_b32", baseLRVgpr, baseLRVgpr, lowVgpr, "pack two half Vgpr to one Vgpr")
                    if isHigh8Bits or isHigh16Bits:
                        tmpVgprIdx += 1

                valufIdx += blockWidth

                # load read instrution
                paramList = []
                paramList.append(destVgpr)
                paramList.append(vgpr("LocalReadAddr%s"%tc))

                for oIdx in range(0, numOffsets):
                    offset_val = (vIdx * numOffsets+oIdx) * MIWaveGropuShape[tile01] * tileStride
                    offset_val = (rIdx * UnrollStride + offset_val + tP["localReadOffset"]) * tP["bpe"]
                    if (kernel["LdsBlockSizePerPad%s"%tc] != 0) and (kernel["LdsPad%s"%tc] != 0):
                        offset_val = offset_val + (offset_val // kernel["LdsBlockSizePerPad%s"%tc]) * kernel["LdsPad%s"%tc] * tP["bpe"]
                    offset_val = offset_val + tP["localReadSwapByteOffset"]
                    paramList.append(int(offset_val))

                paramTuple = tuple(paramList)
                comment = "L -> Reg lro=%d swapByteOffset=%u ti=%u vIdx=%u rIdx=%u oIdx=%u buffer=%u iui=%u" \
                        % (tP["localReadOffset"], tP["localReadSwapByteOffset"], MIWaveGropuShape[tile01], vIdx, rIdx, oIdx, bufferIdx, iui)

                highBits = highBitsForHalf or isHigh16Bits
                localReadCode.addCode(Code.LocalReadInst(instruction.IssueLatency,instruction.toCodeInst(paramTuple, 0, highBits), comment))

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

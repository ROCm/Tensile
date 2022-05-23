################################################################################
# Copyright 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import LraTileAssignment
from ..AsmUtils import inst, vgpr, sgpr, vectorStaticDivideAndRemainder, vectorStaticDivide, staticMultiply, vectorStaticRemainder

class LraTileAssignmentVALU(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": False}

    """
    Local Read Addresses: Tile Assignment
    """
    def __call__(self, writer, kernel, tP):
        kStr = ""

        # allocate resources
        qReg    = writer.vgprPool.checkOut(1,"qReg") # quotient
        rReg    = writer.vgprPool.checkOut(1,"rReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        tmpSgpr = writer.getTmpSgpr(1).idx()

        if tP["tileIdx"] == 0:
            kStr += "%slr%s = serial %% SG%s%s%s" \
                    % (writer.commentPrefix, tP["tileChar"], tP["tileChar"], \
                    writer.commentSuffix, writer.endLine)

            # constant
            dividendReg = "Serial" # local serial
            divisor = kernel["SubGroup0"]

            # generate instruction
            kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgpr, tmpSgpr)

            # release and return resource
            tP["gpr"]["lro"] = rReg
            writer.tmplro = qReg
        else:
            kStr += "%slr%s = (serial / SG%s) %% SG%s%s%s" \
                    % (writer.commentPrefix, tP["tileChar"], tP["tileChar"], \
                    tP["tileChar"], writer.commentSuffix, writer.endLine)

            # constant
            divisor = kernel["SubGroup1"]
            dividendReg = writer.tmplro

            # generate instruction
            kStr += vectorStaticDivideAndRemainder(qReg, rReg, dividendReg, divisor, tmpVgpr, tmpSgpr)

            # release and return resource
            tP["gpr"]["lro"] = rReg

            writer.vgprPool.checkIn(writer.tmplro) # old
            writer.vgprPool.checkIn(qReg)

        writer.vgprPool.checkIn(tmpVgpr)

        return kStr


class LraTileAssignmentMFMA(LraTileAssignment):
    kernel = {"EnableMatrixInstruction": True}

    """
    Local Read Addresses: Tile Assignment A/B
    """
    def __call__(self, writer, kernel, tP):
        kStr = ""

        kStr += "%slr%s%s%s" \
                % (writer.commentPrefix, tP["tileChar"], writer.commentSuffix, writer.endLine)

        # alloc vgpr
        wReg    = writer.vgprPool.checkOut(1,"wReg") # quotient
        tReg    = writer.vgprPool.checkOut(1,"tReg") # remainder
        kReg    = writer.vgprPool.checkOut(1,"kReg") # remainder
        mReg    = writer.vgprPool.checkOut(1,"mReg") # remainder
        mReg1    = writer.vgprPool.checkOut(1,"mReg") # remainder
        mReg2    = writer.vgprPool.checkOut(1,"mReg") # remainder
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        ldsVgpr = writer.vgprPool.checkOut(1,"ldsVgpr")
        ldsVgpr1 = writer.vgprPool.checkOut(1,"ldsVgpr1")
        dummy   = writer.vgprPool.checkOut(1,"dummy")

         # alloc sgpr
        tmpSgpr = writer.getTmpSgpr(1).idx()

        # get constant parameter
        tc               = tP["tensorChar"]
        tile01           = tP["tile01Idx"]
        waveWidth        = writer.kernel["WavefrontSize"]
        inputPerThread   = max(writer.lrvwA,writer.lrvwB)
        if kernel["DirectToVgprB"]:
          # DirectToVgprB case, ignore lrvwB
          inputPerThread = writer.lrvwA
        LdsPad           = kernel["LdsPad%s" % tc] if kernel["LdsBlockSizePerPad%s" % tc] == 0 else 0

        # parameter for get each type index
        dividendForKId   = kernel["MatrixInstM"] * kernel["MatrixInstB"]
        num1DBlocks      = kernel["MatrixInstBM"] if (tile01 == 0) else kernel["MatrixInstBN"]
        num1DWaves       = kernel["MIWaveGroup"][0] if (tile01 == 0) else kernel["MIWaveGroup"][1]
        if kernel["SourceSwap"]:
            dividedForBlkId  = kernel["MatrixInstM"] if (tile01 == 0) else (kernel["MatrixInstM"] * kernel["MatrixInstBM"])
        else:
            dividedForBlkId  = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (tile01 == 0) else kernel["MatrixInstN"]
        dividedForWaveId = waveWidth if (tile01 == 0) else (waveWidth * kernel["MIWaveGroup"][0])
        vectorWidth      = kernel["VectorWidth"] if ((tile01 == 0) and kernel["SourceSwap"]) else 1 # TODO: nonSwap VectorWidth
        if writer.allowLRVWforTLUandMI:
          lrvw = writer.lrvwA if tP["isA"] else writer.lrvwB
          if lrvw > 1:
            vectorWidth = lrvw
          inputPerThread = 1

        # strider for each type of index
        umlds            = kernel["UnrollMajorLDS%s" % tc]
        mt               = kernel["MacroTile%u" % tile01]
        strideTile       = kernel["_DepthULds"] + LdsPad if umlds else 1
        if kernel["ThreadSeparateGlobalRead%c"%tc]:
          #WSGR splits global fetch 2Dtile MblockxdepthU into (WSPR *2)xMblockxdepthU/(WSPR*2)  (Mblock = waveWidth * glvw  / depthU)
          #LDS layout stored as 3Dtile K1xMblockxK0
          #Padding is not allowed in directToLds
          MblockSizePerLoad = (waveWidth * kernel["GlobalLoadVectorWidth%c"%tc]) // kernel["_DepthULds"]
        strideK          = inputPerThread if umlds else (mt + LdsPad) * inputPerThread
        strideBlock      = kernel["MatrixInstM"] * strideTile
        strideWave       = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth

        # tile offset
        kStr += vectorStaticRemainder(dummy, kReg, "Serial", waveWidth, tmpVgpr, tmpSgpr, \
            "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth)
        kStr += vectorStaticRemainder(dummy, tReg, kReg, kernel["MatrixInstN"], tmpVgpr, tmpSgpr, \
            "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"])
        if kernel["ThreadSeparateGlobalRead%c"%tc]:
          kStr += vectorStaticDivide(mReg, tReg, MblockSizePerLoad, tmpVgpr, tmpSgpr, \
              "1. M offset: mIdx_upper = mIdx >> MblockSizePerLoad")
          kStr += staticMultiply(vgpr(mReg), vgpr(mReg), MblockSizePerLoad, sgpr(tmpSgpr), \
              "1. M offset: mIdxlower_offset = nIdxlower * nStride(%u)" % strideTile)
          #kStr += inst("v_not_b32",vgpr(mReg1),vgpr(mReg), \
          #    "1. N offset: mIdxlower = mIdx % MblockSizePerLoad")
          kStr += staticMultiply(vgpr(mReg), vgpr(mReg), kernel["_DepthULds"], sgpr(tmpSgpr), \
              "1. N offset: mIdxlower_offset = nIdxlower * nStride(%u)" % strideTile)
          KelementsPerMFrag = kernel["_DepthULds"]//(kernel["ThreadSeparateGlobalRead%c"%tc]*2)
          KlanesPerMFrag = KelementsPerMFrag // kernel["GlobalLoadVectorWidth%c"%tc] 
          MidxScale = KelementsPerMFrag
          MidxRemainder = MblockSizePerLoad
          if ((KlanesPerMFrag == 8 and (kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]) == 8) or 
              (KlanesPerMFrag == 4 and (kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]) == 16)):
            numElementsPerLane = kernel["GlobalLoadVectorWidth%c"%tc]
            kStr += inst("v_and_b32",vgpr(mReg1),(MidxRemainder-1),vgpr(tReg), \
              "1. N offset: mIdxlower = mIdx and MblockSizePerLoad")
            kStr += staticMultiply(vgpr(mReg1), vgpr(mReg1), (numElementsPerLane*KlanesPerMFrag), sgpr(tmpSgpr), \
              "1. N offset: mIdxlower_offset = nIdxlower * nStride(%u)" % strideTile)
          elif ((KlanesPerMFrag == 4 and (kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]) == 8) or
                (KlanesPerMFrag == 2 and (kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]) == 8)):
            numMidxPer8Ldslanes =  8 // KlanesPerMFrag
            MidxScale = KlanesPerMFrag * 4 // tP["bpe"]
            numElementsPerLane = kernel["GlobalLoadVectorWidth%c"%tc]
            kStr += inst("v_and_b32",vgpr(tReg),(MidxRemainder-1),vgpr(tReg), \
              "1. N offset: mIdxlower = mIdx and MblockSizePerLoad")
            kStr += inst("v_and_b32",vgpr(mReg1),(numMidxPer8Ldslanes-1),vgpr(tReg), \
              "1. N offset: mIdxlower = mIdx and MblockSizePerLoad")
            kStr += staticMultiply(vgpr(mReg1), vgpr(mReg1), (MidxScale), sgpr(tmpSgpr), \
              "1. N offset: mIdxlower_offset = nIdxlower * nStride(%u)" % strideTile)
            kStr += vectorStaticDivide(mReg2, tReg, numMidxPer8Ldslanes, tmpVgpr, tmpSgpr, \
            "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"])
            kStr += staticMultiply(vgpr(mReg2), vgpr(mReg2), (numElementsPerLane*numMidxPer8Ldslanes*KlanesPerMFrag), sgpr(tmpSgpr), \
              "1. N offset: mIdxlower_offset = nIdxlower * nStride(%u)" % strideTile)
            kStr += inst("_v_add_u32", vgpr(mReg1), vgpr(mReg1), vgpr(mReg2), \
                  "1. N offset: mOffset =  mIdxlower_stride + m_idx")
          elif (KlanesPerMFrag == 2 and (kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]) == 16):
            numMidxPer4LdsLanes =  4 // KlanesPerMFrag
            MidxScale = KlanesPerMFrag * 4 // tP["bpe"]
            numElementsPerLane = kernel["GlobalLoadVectorWidth%c"%tc]
            kStr += inst("v_and_b32",vgpr(tReg),(MidxRemainder-1),vgpr(tReg), \
              "1. N offset: mIdxlower = mIdx and MblockSizePerLoad")
            kStr += inst("v_and_b32",vgpr(mReg1),(numMidxPer4LdsLanes-1),vgpr(tReg), \
              "1. N offset: mIdxlower = mIdx and MblockSizePerLoad")
            kStr += staticMultiply(vgpr(mReg1), vgpr(mReg1), MidxScale, sgpr(tmpSgpr), \
              "1. N offset: mIdxlower_offset = nIdxlower * nStride(%u)" % strideTile)
            kStr += vectorStaticDivide(mReg2, tReg, numMidxPer4LdsLanes, tmpVgpr, tmpSgpr, \
            "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"])
            kStr += staticMultiply(vgpr(mReg2), vgpr(mReg2), (numElementsPerLane*numMidxPer4LdsLanes*KlanesPerMFrag), sgpr(tmpSgpr), \
              "1. N offset: mIdxlower_offset = nIdxlower * nStride(%u)" % strideTile)
            kStr += inst("_v_add_u32", vgpr(mReg1), vgpr(mReg1), vgpr(mReg2), \
                  "1. N offset: mOffset =  mIdxlower_stride + m_idx")

          kStr += inst("_v_add_u32", vgpr(tReg), vgpr(mReg), vgpr(mReg1), \
                  "1. N offset: mOffset =  mIdxlower_stride + m_idx")
        else:
          kStr += staticMultiply(vgpr(tReg), vgpr(tReg), strideTile, sgpr(tmpSgpr), \
            "1. N offset: nOffset = nIdx * nStride(%u)" % strideTile)
        # block offset
        kStr += vectorStaticDivide(wReg, kReg, dividedForBlkId, tmpVgpr, tmpSgpr, \
            "2. block offset: bnIdx = wtid / dividedForBlkId(%u)" % dividedForBlkId)
        kStr += vectorStaticRemainder(dummy, wReg, wReg, num1DBlocks, tmpVgpr, tmpSgpr, \
            "2. block offset: bnIdx = bnIdx %% num1DBlocks(%u)" % num1DBlocks)
        kStr += staticMultiply(vgpr(wReg), vgpr(wReg), strideBlock, sgpr(tmpSgpr), \
            "2. block offset: bnOffset = bnIdx * strideBlock(%u)" % strideBlock)
        kStr += inst("_v_add_u32", vgpr(tReg), vgpr(wReg), vgpr(tReg), \
            "3. add N and block offset: bnOffset = block and N offset")
        kStr += staticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, sgpr(tmpSgpr), \
            "3. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth)

        #check DirectToLds TLU=0 ?? skip Koffset  fix TLU=1 case later

        if not (kernel["DirectToLds%s" % tc] and not kernel["ProblemType"]["TLU%c"%tc]):
          # unroll offset
          kStr += vectorStaticDivide(kReg, kReg, dividendForKId, tmpVgpr, tmpSgpr, \
              "4. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))" % (kernel["MatrixInstN"], kernel["MatrixInstB"]))
          kStr += staticMultiply(vgpr(kReg), vgpr(kReg), strideK, sgpr(tmpSgpr), \
              "4. K offset: lrKOffset = kIdx * mStride(%u)" % strideK)

          kStr += inst("_v_add_u32", vgpr(tReg), vgpr(kReg), vgpr(tReg), \
              "5. offset in wave: lrOffset = bnOffset + lrKOffset")

        # wave offset
        if num1DWaves > 1:
            kStr += vectorStaticDivide(wReg, "Serial", dividedForWaveId, tmpVgpr, tmpSgpr, \
                "6. wave offset in N dimen: wtid = tid / dividedForWaveId(%u)" % dividedForWaveId)
            kStr += vectorStaticRemainder(dummy, wReg, wReg, num1DWaves, tmpVgpr, tmpSgpr, \
                "6. wave offset in M dimen: wtid0 = wtid / num1DWaves(%u)" % num1DWaves)
            kStr += staticMultiply(vgpr(wReg), vgpr(wReg), strideWave, sgpr(tmpSgpr), \
                "6. wave offset in M dimen: wOffset = wtid0 * W0Stride(%u)" % strideWave)
            kStr += inst("_v_add_u32", vgpr(tReg), vgpr(wReg), vgpr(tReg), \
                "7. final local read offset: flrOffset = lrOffset + WOffset")

        # release register
        tP["gpr"]["lro"] = tReg
        writer.vgprPool.checkIn(wReg)
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(mReg)
        writer.vgprPool.checkIn(mReg1)
        writer.vgprPool.checkIn(mReg2)
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(ldsVgpr)
        writer.vgprPool.checkIn(ldsVgpr1)
        writer.vgprPool.checkIn(dummy)

        return kStr

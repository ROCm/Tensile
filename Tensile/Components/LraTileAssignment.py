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

from ..Component import LraTileAssignment
from ..AsmUtils import inst, vgpr, sgpr, log2, vectorStaticDivideAndRemainder, vectorStaticDivide, staticMultiply, vectorStaticRemainder

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

        # strider for each type of index
        umlds            = kernel["UnrollMajorLDS%s" % tP["tensorChar"]]
        mt               = kernel["MacroTile%u" % tile01]
        strideTile       = kernel["_DepthULds"] + LdsPad if umlds else 1
        strideK          = inputPerThread if umlds else (mt + LdsPad) * inputPerThread
        strideBlock      = kernel["MatrixInstM"] * strideTile
        strideWave       = kernel["MatrixInstM"] * num1DBlocks * strideTile * vectorWidth

        # tile offset
        kStr += vectorStaticRemainder(dummy, kReg, "Serial", waveWidth, tmpVgpr, tmpSgpr, \
            "0. thread id in wave: wtid = tid %% wavelength(%u)" % waveWidth)
        kStr += vectorStaticRemainder(dummy, tReg, kReg, kernel["MatrixInstN"], tmpVgpr, tmpSgpr, \
            "1. N offset: nIdx = wtid %% MI_N(%u)" % kernel["MatrixInstN"])

        # offset calculation for TLU=1 when glvw * bpe * wavefrontsize > 256
        if (kernel["DirectToLds%s" % tP["tensorChar"]] and kernel["ProblemType"]["TLU%s" % tP["tensorChar"]] and (kernel["GlobalLoadVectorWidth%c"%tP["tensorChar"]] * tP["bpe"] * kernel["WavefrontSize"] > 256)):
          # x2/x4 directToLds stores 8/16 bytes into LDS like below
          # address offset in LDS in bytes
          # DWORD# written by LDS_DMA
          #  address offset in LDS (byte offseet)    
          #  0    4    8    12    16   20   24   28   32   36   40   44    48    52   56   60 
          #  data dword#:                            
          #  0    4    8    12    2    6    10   14    1   5    9    13     3    7    11   15
          #  Noffset calculation for VW =1 (BPe=8) / VW =2 (BPE=4)
          #  use direcToLds for best VW and GRVW case; other cases requires bit more lane manipulation..
          #  offset calculation  for B might benefit from some optimization. 
          #  offset calcualtion for x2/x4  is basically manipulation lane offset based on layout
          
          if (kernel["VectorWidth"] * tP["bpe"] == 8):
            kStr += inst("v_lshrrev_b32", vgpr(ldsVgpr),  hex(3), vgpr(tReg),        "1. magic  offset calc")
            kStr += inst("v_lshlrev_b32", vgpr(ldsVgpr),  hex(6), vgpr(ldsVgpr),     "1. magic  offset calc")
            kStr += inst("v_lshrrev_b32", vgpr(ldsVgpr1), hex(log2(16//(kernel["VectorWidth"] * tP["bpe"]))), vgpr(tReg), "1.magic  offset calc")
            kStr += inst("v_lshlrev_b32", vgpr(ldsVgpr1), hex(2),  vgpr(ldsVgpr1),   "1. magic  offset calc")
            kStr += inst("_v_add_u32",    vgpr(ldsVgpr1), hex(16), vgpr(ldsVgpr1),   "1. magic  offset calc")
            kStr += inst("_v_add_u32",    vgpr(tReg), vgpr(ldsVgpr1), vgpr(ldsVgpr), "1. Noffset:NIdx = magic_func(vw,bpe,grvw)")
          elif (kernel["VectorWidth"] * tP["bpe"] == 16):   # most prefered case
              if (tP["isA"]):
                kStr += inst("v_lshrrev_b32", vgpr(ldsVgpr),  hex(2), vgpr(tReg),    "1. magic  offset calc")
                kStr += inst("v_lshlrev_b32", vgpr(ldsVgpr),  hex(6), vgpr(ldsVgpr), "1. magic  offset calc")
                kStr += inst("v_and_b32", vgpr(ldsVgpr1), hex(3), vgpr(tReg),        "1. magic  offset calc")
                kStr += inst("v_lshlrev_b32", vgpr(ldsVgpr1), hex(2),  vgpr(ldsVgpr1),    "1.magic  offset calc")
                kStr += inst("_v_add_u32",    vgpr(tReg), vgpr(ldsVgpr1), vgpr(ldsVgpr),  "1.Noffset:NIdx = magic_func(vw,bpe,grvw)")
              else:
                kStr += inst("v_lshrrev_b32", vgpr(ldsVgpr),  hex(3), vgpr(tReg),        "1. magic  offset calc")
                kStr += inst("v_lshlrev_b32", vgpr(ldsVgpr),  hex(6), vgpr(ldsVgpr),     "1. magic  offset calc")
                kStr += inst("v_and_b32",     vgpr(ldsVgpr1), hex(7), vgpr(tReg),        "1. magic  offset calc")
                kStr += inst("v_lshrrev_b32", vgpr(ldsVgpr1), hex(1), vgpr(ldsVgpr1),    "1. magic  offset calc")
                kStr += inst("v_lshlrev_b32", vgpr(ldsVgpr1), hex(2), vgpr(ldsVgpr1),    "1. magic  offset calc")
                kStr += inst("_v_add_u32",    vgpr(ldsVgpr), vgpr(ldsVgpr), vgpr(ldsVgpr1),   "1. magic  offset calc")
                kStr += inst("v_and_b32",     vgpr(ldsVgpr1), hex(1), vgpr(tReg),        "1. magic  offset calc")
                kStr += inst("v_lshlrev_b32", vgpr(ldsVgpr1),  hex(4), vgpr(ldsVgpr1),   "1. magic  offset calc")
                kStr += inst("_v_add_u32",    vgpr(tReg), vgpr(ldsVgpr1), vgpr(ldsVgpr), "1. Noffset:NIdx = magic_func(vw,bpe,grvw)")
          else:
            kStr += inst("v_lshrrev_b32", vgpr(ldsVgpr1), hex(1),  vgpr(ldsVgpr),       "1.magic  offset calc")
            kStr += inst("v_lshlrev_b32", vgpr(ldsVgpr1), hex(4), vgpr(ldsVgpr1),       "1.magic  offset calc")
            kStr += inst("_v_add_u32",    vgpr(ldsVgpr1), hex(16), vgpr(ldsVgpr1),      "1.magic  offset calc")
            kStr += inst("v_and_b32",     vgpr(ldsVgpr),  hex(1), vgpr(tReg),           "1.magic  offset calc")
            kStr += inst("v_and_b32",     vgpr(ldsVgpr),  hex(32), vgpr(ldsVgpr),       "1.magic  offset calc")
            kStr += inst("_v_add_u32",    vgpr(ldsVgpr), vgpr(ldsVgpr1), vgpr(ldsVgpr), "1.magic offset calc ") 
            kStr += inst("v_lshrrev_b32", vgpr(ldsVgpr),  hex(2), vgpr(tReg),           "1.magic  offset calc")
            kStr += inst("v_lshlrev_b32", vgpr(ldsVgpr),  hex(6), vgpr(ldsVgpr),        "1.magic  offset calc")
            kStr += inst("_v_add_u32",    vgpr(tReg), vgpr(ldsVgpr1), vgpr(ldsVgpr),    "1 Noffset:NIdx = magic_func(vw,bpe,grvw)")
        #else: # TLU=0 case for should work fine mostly (
               # addition of summation index partial accumulation should satisfy associative property
               # TODO (re-check for different MFMA_MXNXK instructions
               

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
        if not (kernel["DirectToLds%s" % tP["tensorChar"]] and  \
                kernel["GlobalLoadVectorWidth%c"%tP["tensorChar"]] * tP["bpe"] > 4 and  \
                kernel["ProblemType"]["TLU%s" % tP["tensorChar"]]):
          kStr += staticMultiply(vgpr(tReg), vgpr(tReg), vectorWidth, sgpr(tmpSgpr), \
            "3. apply VectorWidth: bnOffset = bnOffset * vw(%u)" % vectorWidth)

        # unroll offset
        kStr += vectorStaticDivide(kReg, kReg, dividendForKId, tmpVgpr, tmpSgpr, \
            "4. K offset: kIdx = wtid / (MIN(%u) * MIBB(%u))" % (kernel["MatrixInstN"], kernel["MatrixInstB"]))
        kStr += staticMultiply(vgpr(kReg), vgpr(kReg), strideK, sgpr(tmpSgpr), \
            "4. K offset: lrKOffset = kIdx * mStride(%u)" % strideK)
        if (kernel["DirectToLds%s" % tP["tensorChar"]] and  \
            kernel["GlobalLoadVectorWidth%c"%tP["tensorChar"]] * tP["bpe"] > 4 and  \
            kernel["ProblemType"]["TLU%s" % tP["tensorChar"]]):
          kStr += inst("v_lshlrev_b32", vgpr(kReg), hex(log2(tP["bpe"])), vgpr(kReg), \
            "4. lrKoffset = lrkOffset * bpe")
          if tP["nrc"] > 1:
            # DirectToLds + above conditions, swap offset_val bits to adjust LDS offset
            waveDiff = 1
            scale = tP["nrc"]
            scaleShift = log2(scale) # assuming scale is power of 2
            scaleShift += int(log2(waveDiff))
            ldsLineSize = kernel["WavefrontSize"] * kernel["GlobalLoadVectorWidth%c"%tc] * tP["bpe"]
            ldsLineSize //= scale
            maskBitsLow = (scale - 1) * ldsLineSize
            maskBitsHigh = maskBitsLow * scale * waveDiff
            maskBitsAll = (maskBitsLow | maskBitsHigh)
            tmp1    = writer.vgprPool.checkOut(1,"tmp1")
            tmp2    = writer.vgprPool.checkOut(1,"tmp2")
            tmpSgpr2 = writer.getTmpSgpr(1).idx()
            kStr += inst("v_and_b32", vgpr(tmp1), hex(maskBitsLow), vgpr(kReg), \
              "4. Offset adjustment (swap row and col index) for DirectToLds + %s > 1"%tP["lsc"])
            kStr += inst("v_and_b32", vgpr(tmp2), hex(maskBitsHigh), vgpr(kReg), "")
            kStr += inst("v_lshlrev_b32", vgpr(tmp1), hex(scaleShift), vgpr(tmp1), "")
            kStr += inst("v_lshrrev_b32", vgpr(tmp2), hex(scaleShift), vgpr(tmp2), "")
            kStr += inst("v_or_b32", vgpr(tmp1), vgpr(tmp1), vgpr(tmp2), "")
            kStr += inst("s_mov_b32", sgpr(tmpSgpr2), hex(maskBitsAll), "")
            kStr += inst("v_not_b32", vgpr(tmp2), sgpr(tmpSgpr2), "")
            kStr += inst("v_and_b32", vgpr(kReg), vgpr(tmp2), vgpr(kReg), "")
            kStr += inst("v_or_b32", vgpr(kReg), vgpr(tmp1), vgpr(kReg), "")
            writer.vgprPool.checkIn(tmp1)
            writer.vgprPool.checkIn(tmp2)
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
            #TODO fix bpe scaling for LDSDma
            if (kernel["DirectToLds%s" % tP["tensorChar"]] and \
                kernel["GlobalLoadVectorWidth%c"%tP["tensorChar"]] * tP["bpe"] > 4 and  \
                kernel["ProblemType"]["TLU%s" % tP["tensorChar"]]):
              kStr += inst("v_lshlrev_b32", vgpr(wReg), hex(log2(tP["bpe"])), vgpr(wReg), \
                "6. wave offset in M dimen: wOffset = wOffset * bpe")
            kStr += inst("_v_add_u32", vgpr(tReg), vgpr(wReg), vgpr(tReg), \
                "7. final local read offset: flrOffset = lrOffset + WOffset")

        # release register
        tP["gpr"]["lro"] = tReg
        writer.vgprPool.checkIn(wReg)
        writer.vgprPool.checkIn(kReg)
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(ldsVgpr)
        writer.vgprPool.checkIn(ldsVgpr1)
        writer.vgprPool.checkIn(dummy)

        return kStr

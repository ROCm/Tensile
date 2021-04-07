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

from ..Component import ComputeStoreVgprs
from ..AsmUtils import vectorStaticDivideAndRemainder, staticMultiply, vgpr, sgpr, inst, vectorStaticDivide, vectorStaticRemainder

class ComputeStoreVgprsVALU(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstruction": False}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        kStr = ""

        tmpS0 = writer.getTmpSgpr(3).idx()
        tmpS1 = tmpS0+1
        wgMT1 = tmpS0+2

        if writer.prefetchAcrossPersistent:
            wg0="PrevWorkGroup0"
            wg1="PrevWorkGroup1"
        else:
            wg0="WorkGroup0"
            wg1="WorkGroup1"

        # tid0, tid1: element offsets from the start of macroTile in 0 and 1 direction
        # These will live for entire GlobalWrite loop - allocate before tmps
        # to avoid fragmentation
        tid0 = writer.vgprPool.checkOut(1, "tid0")
        tid1 = writer.vgprPool.checkOut(1, "tid1")

        packedC1 = kernel["PackedC1IndicesX"]

        if kernel["BufferStore"]:
            writer.cinRowPtr    = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.coutRowPtr = writer.vgprPool.checkOut(1, "coutRowPtr")

        tmpV0 = writer.vgprPool.checkOutAligned(2,2)
        kStr += vectorStaticDivideAndRemainder(tid1, tid0, "Serial", divisor, tmpV0, tmpS0)
        kStr += staticMultiply(vgpr(tid0), vgpr(tid0), tid0Scale, sgpr(tmpS1))
        if tid1Scale != 1:
            kStr += staticMultiply(vgpr(tid1), vgpr(tid1), tid1Scale, sgpr(tmpS1))
        writer.vgprPool.checkIn(tmpV0)

        if kernel["BufferStore"]:
            # compute rowStart- this is just tid1 scaled by appropriate stride.
            # rowPtr is offset from the beginning of the tile/SRD not the tensor base
            # when incremented, it moves in units of (col) Stride to get to a new row
            # it is used for address computation, not element range detection.
            # rowPtr is in the element space and must be scaled by bpe if bytes are required.
            # Do this before code below which overwries the tid1:
            # TODO-packed
            # Eventually need to modify if supporting packed coord1, to start just assert if that case is detected
            #--
            strideC1 = "StrideC%s" % (writer.indexChars[packedC1[0]])
            kStr += inst("v_mul_lo_u32", vgpr(writer.cinRowPtr),
                         vgpr(tid1), sgpr(strideC1), \
                         "rowStart vgpr")
            strideD1 = "StrideD%s" % (writer.indexChars[packedC1[0]])
            kStr += inst("v_mul_lo_u32", vgpr(writer.coutRowPtr),
                         vgpr(tid1), sgpr(strideD1), \
                         "rowStart vgpr")
            kStr += "\n"

            #kStr += writer.assert_ne(sgpr("WorkGroup1"),1)

        # Compute coord0 and coord1
        # These are element offsets from the beginning of the tensor.
        # These are 'flattened' meaning they span packed tensor dims.
        # They need to be preserved so can use in comparisons against
        # product-of-packed sizes to determine OOB cases. (for Edge tiles only)
        kStr += inst("s_mul_i32", \
                sgpr(tmpS0), \
                hex(kernel["MacroTile0"]), \
                sgpr(wg0), \
                "%s = wg0*MT0"%sgpr(tmpS0))

        # coord = tid*VW + workgroup offset
        kStr += inst("_v_add_co_u32", \
                vgpr(tid0), \
                writer.vcc, \
                sgpr(tmpS0), \
                vgpr(tid0), \
                "coord0 = tid0*VW + wg0*MT0")
        kStr += inst("s_mul_i32", \
                sgpr(wgMT1), \
                hex(kernel["MacroTile1"]), \
                sgpr(wg1), \
                "<- wg1*MT1")
        kStr += inst("_v_add_co_u32", \
                vgpr(tid1), \
                writer.vcc, \
                sgpr(wgMT1), \
                vgpr(tid1), \
                "coord1 = tid1*VW + wg1*MT1")

        if len(packedC1) > 1:
          kStr += writer.extractPackedCoord1ToRowStart(kernel, packedC1, tid1, 'D')

        writer.coord0 = tid0
        writer.coord1 = tid1

        return kStr

class ComputeStoreVgprsMFMA(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstruction": True,
              "SourceSwap": False}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        # writer.coord0
        # writer.coord1
        # writer.cinRowPtr  : C buffer coulmn offset
        # writer.coutRowPtr : D buffer coulmn offset

        # alloc resources
        tid0 = writer.vgprPool.checkOut(1)
        tid1 = writer.vgprPool.checkOut(1)
        if kernel["BufferStore"]:
            writer.cinRowPtr    = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.coutRowPtr = writer.vgprPool.checkOut(1, "coutRowPtr")

        wave_id = writer.vgprPool.checkOut(1)

        tmpVgpr0 = writer.vgprPool.checkOut(1,"tmpVgpr0")
        tmpVgpr1 = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr1")
        dummy    = writer.vgprPool.checkOut(1,"dummy")
        tmpSgpr  = writer.getTmpSgpr(1).idx()

        # constant
        MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
        MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

        kStr = ""

        # coord 1 : wave part
        kStr += vectorStaticDivide(wave_id, "Serial", writer.kernel["WavefrontSize"], tmpVgpr1, tmpSgpr)
        kStr += vectorStaticDivide(tid1, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1, tmpSgpr)
        kStr += inst("v_mul_lo_u32", vgpr(tid1), hex(MIBShape1), vgpr(tid1), "wave coordination offset 1")

        # coord 1 : thread part
        kStr += vectorStaticRemainder(dummy, tmpVgpr0, "Serial", kernel["MatrixInstN"], tmpVgpr1, tmpSgpr)
        kStr += inst("_v_add_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), "coordination 1 = wave_id1 + tid1")


        if kernel["MatrixInstM"] == 4:
            remainder = writer.kernel["WavefrontSize"]
            divisor =  kernel["MatrixInstN"] * kernel["MatrixInstBM"]
            if kernel["ProblemType"]["DataType"].isDouble():
                divisor *= 4
                if kernel["MatrixInstBM"] < 4:
                    remainder = 16
                    divisor = 8
            kStr += vectorStaticRemainder(dummy, tmpVgpr0, "Serial", remainder, tmpVgpr1, tmpSgpr)
            kStr += vectorStaticDivide(tmpVgpr0, tmpVgpr0, divisor, tmpVgpr1, tmpSgpr)
            kStr   += staticMultiply(vgpr(tmpVgpr0), vgpr(tmpVgpr0), kernel["MatrixInstN"], sgpr(tmpSgpr))
            kStr   += inst("_v_add_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), "coordination 1 = wave_id1 + tid1")

        # coord 1 : offset part
        packedC1 = kernel["PackedC1IndicesX"]
        strideC1 = "StrideC%s" % (writer.indexChars[packedC1[0]])
        strideD1 = "StrideD%s" % (writer.indexChars[packedC1[0]])
        kStr += inst("v_mul_lo_u32", vgpr(writer.cinRowPtr), vgpr(tid1), sgpr(strideC1), " offset 1")
        kStr += inst("v_mul_lo_u32", vgpr(writer.coutRowPtr), vgpr(tid1), sgpr(strideD1), " offset 1")

        # coord 0 : wave part
        kStr += vectorStaticRemainder(dummy, tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1, tmpSgpr)
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr0), hex(MIBShape0), vgpr(tmpVgpr0), "wave coordination offset 0")
        if kernel["MatrixInstM"] == 4 and kernel["ProblemType"]["DataType"].isDouble():
            divisor =  kernel["MatrixInstN"] * kernel["MatrixInstM"]
            kStr += vectorStaticDivide(tid0, "Serial", divisor, tmpVgpr1, tmpSgpr)
            kStr += vectorStaticRemainder(dummy, tid0, tid0, kernel["MatrixInstN"], tmpVgpr1, tmpSgpr)
            kStr += inst("_v_add_u32", vgpr(tmpVgpr0), vgpr(tmpVgpr0), vgpr(tid0), "WAAA")

        # coord 0 : thread part
        kStr += vectorStaticRemainder(dummy, tid0, "Serial", writer.kernel["WavefrontSize"], tmpVgpr1, tmpSgpr)
        kStr += vectorStaticDivide(tid0, tid0, kernel["MatrixInstM"], tmpVgpr1, tmpSgpr)
        if kernel["MatrixInstM"] == 4:
            kStr += vectorStaticRemainder(dummy, tid0, tid0, kernel["MatrixInstBM"], tmpVgpr1, tmpSgpr)
        if kernel["MatrixInstM"] == 4 or not kernel["ProblemType"]["DataType"].isDouble():
            kStr += inst("v_lshlrev_b32", vgpr(tid0), hex(2), vgpr(tid0), "thread0 * 4 : mfma output 4 continuous outputs")
        kStr += inst("_v_add_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), "coordination 0 = wave_id0 + tid0")

        if writer.prefetchAcrossPersistent:
            wg0="PrevWorkGroup0"
            wg1="PrevWorkGroup1"
        else:
            wg0="WorkGroup0"
            wg1="WorkGroup1"

        # macro tile 0 part
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile0"], sgpr(wg0), "wgp0 * MT0")
        kStr += inst("_v_add_u32", vgpr(tid0), sgpr(tmpSgpr), vgpr(tid0), "coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0")

        # macro tile 1 part
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile1"], sgpr(wg1), "wgp1 * MT1")
        kStr += inst("_v_add_u32", vgpr(tid1), sgpr(tmpSgpr), vgpr(tid1), "coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1")

        # extract packed rowStart vgpr
        if len(packedC1) > 1:
          kStr += writer.extractPackedCoord1ToRowStart(kernel, packedC1, tid1, 'D')

        # release resource
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(tmpVgpr1)
        writer.vgprPool.checkIn(tmpVgpr0)
        writer.vgprPool.checkIn(wave_id)

        # StoreRemap: calculate
        # 1. local read address
        # 2. local write address
        # 3. global write coord0 and coord1
        if kernel["StoreRemapVectorWidth"]:
            kStr += writer.storeRemapComputeStoreVgprs(kernel)

        writer.coord0 = tid0
        writer.coord1 = tid1

        return kStr

class ComputeStoreVgprsMFMASwap(ComputeStoreVgprs):
    kernel = {"EnableMatrixInstruction": True,
              "SourceSwap": True}

    """
    computeStoreVgprs
    Compute workitem/TT offsets in VGPRS
    and coord0/coord1
    tid0Scale specifies the number of output elements in 0/coalesced dim
    that should be written by each work-item in each batch element.
    """
    def __call__(self, writer, kernel, divisor, tid0Scale, tid1Scale):

        # writer.coord0
        # writer.coord1
        # writer.cinRowPtr  : C buffer coulmn offset
        # writer.coutRowPtr : D buffer coulmn offset

        # alloc resources
        tid0 = writer.vgprPool.checkOut(1)
        tid1 = writer.vgprPool.checkOut(1)
        if kernel["BufferStore"]:
            writer.cinRowPtr    = writer.vgprPool.checkOut(1, "cinRowPtr")
            writer.coutRowPtr = writer.vgprPool.checkOut(1, "coutRowPtr")

        wave_id = writer.vgprPool.checkOut(1)

        tmpVgpr0 = writer.vgprPool.checkOut(1,"tmpVgpr0")
        tmpVgpr1 = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr1")
        dummy    = writer.vgprPool.checkOut(1,"dummy")
        tmpSgpr  = writer.getTmpSgpr(1).idx()

        # constant
        MIBShape0 = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
        MIBShape1 = kernel["MatrixInstN"] * kernel["MatrixInstBN"]

        kStr = ""

        kStr += vectorStaticDivide(wave_id, "Serial", writer.kernel["WavefrontSize"], tmpVgpr1, tmpSgpr)

        # coord 1 : wave part
        kStr += vectorStaticDivide(tmpVgpr0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1, tmpSgpr)
        kStr += inst("v_mul_lo_u32", vgpr(tmpVgpr0), hex(MIBShape1), vgpr(tmpVgpr0), "wave coordination offset 1")
        # if kernel["MatrixInstM"] == 4 and kernel["ProblemType"]["DataType"].isDouble():
        #     divisor =  kernel["MatrixInstN"] * kernel["MatrixInstM"]
        #     kStr += vectorStaticDivide(tid1, "Serial", divisor, tmpVgpr1, tmpSgpr)
        #     kStr += vectorStaticRemainder(dummy, tid1, tid1, kernel["MatrixInstN"], tmpVgpr1, tmpSgpr)
        #     kStr += inst("v_add_u32", vgpr(tmpVgpr0), vgpr(tmpVgpr0), vgpr(tid1), "WAAA")

        # coord 1 : thread part
        kStr += vectorStaticRemainder(dummy, tid1, "Serial", writer.kernel["WavefrontSize"], tmpVgpr1, tmpSgpr)
        kStr += vectorStaticDivide(tid1, tid1, kernel["MatrixInstM"], tmpVgpr1, tmpSgpr)
        if kernel["MatrixInstM"] == 4:
            kStr += vectorStaticRemainder(dummy, tid1, tid1, kernel["MatrixInstBM"], tmpVgpr1, tmpSgpr)
        if kernel["MatrixInstM"] == 4 or not kernel["ProblemType"]["DataType"].isDouble():
            kStr += inst("v_lshlrev_b32", vgpr(tid1), hex(2), vgpr(tid1), "thread0 * 4 : mfma output 4 continuous outputs")
        kStr += inst("v_add_u32", vgpr(tid1), vgpr(tmpVgpr0), vgpr(tid1), "coordination 1 = wave_id1 + tid1")

        # coord 1 : offset part
        packedC1 = kernel["PackedC1IndicesX"]
        strideC1 = "StrideC%s" % (writer.indexChars[packedC1[0]])
        strideD1 = "StrideD%s" % (writer.indexChars[packedC1[0]])
        kStr += inst("v_mul_lo_u32", vgpr(writer.cinRowPtr), vgpr(tid1), sgpr(strideC1), " offset 1")
        kStr += inst("v_mul_lo_u32", vgpr(writer.coutRowPtr), vgpr(tid1), sgpr(strideD1), " offset 1")

        # coord 0 : wave part
        kStr += vectorStaticRemainder(dummy, tid0, wave_id, kernel["MIWaveGroup"][0], tmpVgpr1, tmpSgpr)
        kStr += inst("v_mul_lo_u32", vgpr(tid0), hex(MIBShape0), vgpr(tid0), "wave coordination offset 0")

        # coord 0 : thread part
        kStr += vectorStaticRemainder(dummy, tmpVgpr0, "Serial", kernel["MatrixInstN"], tmpVgpr1, tmpSgpr)
        kStr += inst("v_add_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), "coordination 0 = wave_id0 + tid0")


        # if kernel["MatrixInstM"] == 4:
        #     remainder = writer.kernel["WavefrontSize"]
        #     divisor =  kernel["MatrixInstN"] * kernel["MatrixInstBM"]
        #     if kernel["ProblemType"]["DataType"].isDouble():
        #         divisor *= 4
        #         if kernel["MatrixInstBM"] < 4:
        #             remainder = 16
        #             divisor = 8
        #     kStr += vectorStaticRemainder(dummy, tmpVgpr0, "Serial", remainder, tmpVgpr1, tmpSgpr)
        #     kStr   += vectorStaticDivide(tmpVgpr0, tmpVgpr0, divisor, tmpVgpr1, tmpSgpr)
        #     kStr   += staticMultiply(vgpr(tmpVgpr0), vgpr(tmpVgpr0), kernel["MatrixInstN"], sgpr(tmpSgpr))
        #     kStr   += inst("v_add_u32", vgpr(tid0), vgpr(tmpVgpr0), vgpr(tid0), "coordination 1 = wave_id1 + tid1")

        if writer.prefetchAcrossPersistent:
            wg0="PrevWorkGroup0"
            wg1="PrevWorkGroup1"
        else:
            wg0="WorkGroup0"
            wg1="WorkGroup1"

        # macro tile 0 part
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile0"], sgpr(wg0), "wgp0 * MT0")
        kStr += inst("v_add_u32", vgpr(tid0), sgpr(tmpSgpr), vgpr(tid0), "coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0")

        # macro tile 1 part
        kStr += inst("s_mul_i32", sgpr(tmpSgpr), kernel["MacroTile1"], sgpr(wg1), "wgp1 * MT1")
        kStr += inst("v_add_u32", vgpr(tid1), sgpr(tmpSgpr), vgpr(tid1), "coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1")

        # release resource
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(tmpVgpr1)
        writer.vgprPool.checkIn(tmpVgpr0)
        writer.vgprPool.checkIn(wave_id)

        # StoreRemap: calculate
        # 1. local read address
        # 2. local write address
        # 3. global write coord0 and coord1
        if kernel["StoreRemapVectorWidth"]:
            kStr += writer.storeRemapComputeStoreVgprs(kernel)

        writer.coord0 = tid0
        writer.coord1 = tid1

        return kStr

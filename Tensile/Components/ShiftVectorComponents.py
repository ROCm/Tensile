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

from ..Component import ShiftVectorComponents
from ..AsmUtils import inst, vgpr, sgpr, accvgpr, staticMultiply, vectorStaticDivide, vectorStaticRemainder, vectorStaticDivideAndRemainder, log2

class ShiftVectorComponentsVALU(ShiftVectorComponents):
    kernel = {"EnableMatrixInstruction": False}

    """
    Shift Vector Components d0,1
    """
    def __call__(self, writer, kernel, tP):
        kStr = ""

        # glvw
        vw = tP["glvw"]
        numVectors = kernel[tP["tt"]]//vw
        # labels
        svrLabels = []
        sviLabels = []
        for i in range(0, vw):
            r = (i+1) % vw
            label = writer.getLabelNum("ShiftVectorComponents%u_R%u"%(tP["idx"], r) )
            svrLabels.append(label)
            tmpLabels = []
            for v in range(0, numVectors):
                label = writer.getLabelNum("ShiftVectorComponents%u_R%u_V%u"%(tP["idx"], r, v) )
                tmpLabels.append(label)
            sviLabels.append(tmpLabels)

        wg = tP["prevWg"] if writer.prefetchAcrossPersistent else tP["wg"]
        # wgMT value
        tmpSgpr = writer.getTmpSgpr(writer.laneSGPRCount).idx()
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2,"tmpVgpr")
        wgMT = writer.vgprPool.checkOut(1,"wgMT")
        kStr += inst("v_mov_b32", vgpr(wgMT), sgpr(wg), "")
        kStr += inst("v_mul_i32_i24", vgpr(wgMT), hex(-kernel[tP["mt"]]), vgpr(wgMT), \
                "wg*MT")
        kStr += inst("_v_add_co_u32", vgpr(wgMT), writer.vcc, sgpr("SizesFree+%u"%tP["idx"]), \
                vgpr(wgMT), "wgMT = Size - wg*MT")
        kStr += inst("v_mov_b32", vgpr(tmpVgpr), hex(kernel[tP["mt"]]), "MT")
        kStr += inst("v_cmp_lt_u32", sgpr(tmpSgpr,writer.laneSGPRCount), vgpr(wgMT), \
                vgpr(tmpVgpr), "wgMT < MT" )
        kStr += inst("v_cndmask_b32", vgpr(wgMT), vgpr(tmpVgpr), \
                vgpr(wgMT), sgpr(tmpSgpr,writer.laneSGPRCount), "wgMT = (wgMT < MT) ? wgMT : MT" )
        dummy = writer.vgprPool.checkOut(1,"dummy")

        # qReg
        qReg = writer.vgprPool.checkOut(1,"qReg")
        divisor = kernel["VectorWidth"] # vw
        kStr += vectorStaticDivide(qReg, wgMT, divisor, \
                tmpVgpr, tmpSgpr)

        # rReg
        rReg = writer.vgprPool.checkOut(1,"rReg")
        divisor = vw
        kStr += vectorStaticRemainder(dummy, rReg, wgMT, divisor, \
                tmpVgpr, tmpSgpr)

        # qReg %/ SG
        sReg = writer.vgprPool.checkOut(1,"sReg")
        eReg = writer.vgprPool.checkOut(1,"eReg")
        divisor = kernel[tP["sg"]]
        kStr += vectorStaticDivideAndRemainder(sReg, eReg, qReg, divisor, \
                tmpVgpr, tmpSgpr)

        if tP["isA"]:
            # thread = serial % SG0
            thread = writer.vgprPool.checkOut(1,"thread")
            divisor = kernel["SubGroup0"]
            kStr += vectorStaticRemainder(dummy, thread, "Serial", divisor, \
                    tmpVgpr, tmpSgpr)
            #kStr += dump(vgpr(thread))
            #kStr += dump(vgpr(thread))
        else:
            # thread = (serial / SG0) % SG1
            sd0 = writer.vgprPool.checkOut(1,"sd0")
            divisor = kernel["SubGroup0"]
            kStr += vectorStaticDivide(sd0, "Serial", divisor, \
                    tmpVgpr, tmpSgpr) # thread = serial / SG0
            divisor = kernel["SubGroup1"]
            thread = writer.vgprPool.checkOut(1,"thread")
            kStr += vectorStaticRemainder(dummy, thread, sd0, divisor, \
                    tmpVgpr, tmpSgpr) # thread = (serial / SG0) % SG1
            writer.vgprPool.checkIn(sd0)

        # which glvw vector of thread to shift? wgMT / (SG0*VW) -> (wgMT%VW) / glvw
        # (wgMT/(WG0*VW))*(VW/glvw) + (wgMT%VW) / glvw
        if True:#tP["tensorIdx"] > kernel["VectorWidth"]:
            mvReg = writer.vgprPool.checkOut(1,"mvReg")
            divisor = kernel[tP["sg"]]*kernel["VectorWidth"]
            kStr += vectorStaticDivide(mvReg, wgMT, divisor, \
                    tmpVgpr, tmpSgpr)
            if vw < kernel["VectorWidth"]:
                kStr += inst("v_lshlrev_b32", vgpr(mvReg), hex(log2(kernel["VectorWidth"]//vw)), vgpr(mvReg), "vId *= VW/glvw")
        #kStr += dump(vgpr(mvReg))

        vReg = writer.vgprPool.checkOut(1,"vReg")
        divisor = kernel["VectorWidth"]
        kStr += vectorStaticRemainder(dummy, vReg, wgMT, divisor, \
                tmpVgpr, tmpSgpr)
        vRegD = writer.vgprPool.checkOut(1,"vRegD")
        kStr += inst("v_mov_b32", vgpr(vRegD), vgpr(vReg), "duplicate")
        divisor = vw
        kStr += vectorStaticDivide(vReg, vRegD, divisor, \
                tmpVgpr, tmpSgpr)
        #kStr += dump(vgpr(vReg))

        if True:#tP["tensorIdx"] > kernel["VectorWidth"]:
            kStr += inst("_v_add_co_u32", vgpr(vReg), writer.vcc, vgpr(mvReg), vgpr(vReg), "vId = 2 components")
            writer.vgprPool.checkIn(mvReg)
            writer.vgprPool.checkIn(vRegD)

        # for each remainder, jump
        for r in range(1, vw):
            kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(rReg), \
                    hex(r), "wgMT%%VW == %u"%r )
            kStr += inst("s_cbranch_vccnz label_%04u"\
                    % svrLabels[(r-1)%vw], \
                    "shift d%u r=%u"%(tP["idx"], r))
            #kStr += inst("s_mov_b32", sgpr(sgprLoc), hex(location), \
            #        "location=%u"%location) location *= 2
            #kStr += inst("v_or_b32", vgpr(vgprPath), sgpr(sgprLoc), \
            #        vgpr(vgprPath), "path+=location")
        kStr += inst("s_branch label_%04u"%svrLabels[vw-1], \
                "no shifting" )

        # code blocks for shifting
        for r in range(1, vw):
            kStr += writer.comment3("shift d%u r=%u"%(tP["idx"], r))
            kStr += "label_%04u:%s" % (svrLabels[r-1], writer.endLine)
            #if r==3 and tP["isA"]:
            #    kStr += dump(vgpr("Serial"))

            # for each vector index, jump
            for vectorIdx in range(0, numVectors):
                kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(vReg), \
                        hex(vectorIdx), "wgMT/(SG*VW) == %u"%vectorIdx )
                kStr += inst("s_cbranch_vccnz label_%04u"\
                        % sviLabels[(r-1)%vw][vectorIdx], \
                        "shift d%u, r=%u, v=%u"%(tP["idx"], r, vectorIdx))

            # code blocks for shifting vector
            for vectorIdx in range(0, numVectors):
                kStr += writer.comment("shift d%u r=%u v=%u"%(tP["idx"], r, vectorIdx))
                kStr += "label_%04u:%s" % (sviLabels[r-1][vectorIdx], writer.endLine)
                # mask if last thread in thread#-tile column
                kStr += inst("_v_cmpx_eq_u32", sgpr(tmpSgpr,writer.laneSGPRCount), vgpr(thread), \
                    vgpr(eReg), "serial % SG == (wgMT/VECTOR_WIDTH)%SG" )
                tto = kernel["ThreadTile%u"%((tP["idx"]+1)%2)] # thread tile orthogonal
                for tt in range(0, tto):
                    for s in range(0, r):
                        if tP["isA"]: # shift d0
                            dst = (s) \
                                + vectorIdx * vw + tt * kernel["ThreadTile0"]
                            src = (s+vw-r) \
                                + vectorIdx * vw + tt * kernel["ThreadTile0"]
                            comment = "rC[%u+%u*VW+%u*TT%s] = rC[%u+%u*VW+%u*TT%s]" \
                                % (s, vectorIdx, tt, writer.tileChar0, \
                                s+vw-r, vectorIdx, tt, writer.tileChar0 )
                        else: # shift d1
                            dst = (tt) \
                                + vectorIdx*vw*kernel["ThreadTile0"] + s * kernel["ThreadTile0"]
                            src = (tt) \
                                + vectorIdx * vw*kernel["ThreadTile0"] + (s+vw-r) * kernel["ThreadTile0"]
                            comment = "rC[%u+%u*TT%s*VW+%u*TT%s] = rC[%u+%u*TT%s*VW+%u*TT%s]" \
                                % (tt, vectorIdx, writer.tileChar0, s, writer.tileChar0, \
                                tt, vectorIdx, writer.tileChar0, \
                                s+vw-r, writer.tileChar0)

                        kStr += "// src=%u, dst=%u\n" % (src,dst)

                        # f16
                        #jgolds I think this should be bpeCinternal
                        if writer.bpeCinternal == 2:
                            srcVgpr = writer.startVgprValuC+src*writer.bpeCinternal//writer.bpr
                            dstVgpr = writer.startVgprValuC+dst*writer.bpeCinternal//writer.bpr
                            kStr += "// %u, %u, %u, %u, %u, %u\n" % (r, vectorIdx, tt, s, dst, src)
                            if tP["isA"]: # f16 d0
                                if r % 2 == 0: # even shift can use mov_b32
                                    if s % 2 == 0:
                                        kStr += inst("v_mov_b32", vgpr(dstVgpr), \
                                                vgpr(srcVgpr), comment)
                                    else:
                                        pass # above command performs two moves
                                else: # odd shift
                                    srcLo = src % 2 == 0 # even
                                    dstLo = dst % 2 == 0 # even
                                    kStr += "// srcLo=%u, dstLo=%u\n" % (srcLo,dstLo)
                                    if dstLo: # hi src to lo dst; can clobber hi bits
                                        kStr += inst("v_lshrrev_b32", vgpr(dstVgpr), \
                                                hex(16), vgpr(srcVgpr), "hi16 -> lo16")
                                    else: # dstHi; cannot clobber lo bits
                                        tmpSrcVgpr = writer.vgprPool.checkOut(1,"tmpSrcVgpr")
                                        # zero out dst hi bits -> dst
                                        kStr += inst("v_and_b32", vgpr(dstVgpr), \
                                                "0x0000FFFF", vgpr(dstVgpr), "zero out dst hi16")
                                        if srcLo: # lo src to hi dst
                                            # left shift src 16 bits -> tmpSrc
                                            kStr += inst("v_lshlrev_b32", vgpr(tmpSrcVgpr), \
                                                    hex(16), vgpr(srcVgpr), "left shift src 16 bits")
                                        else: # hi src to hi dst
                                            # zero out src lo bits -> tmpSrc
                                            kStr += inst("v_and_b32", vgpr(srcVgpr), \
                                                    "0xFFFF0000", vgpr(tmpSrcVgpr), "zero out src lo16")
                                        # dst = tmpSrc | dst
                                        kStr += inst("v_or_b32", vgpr(dstVgpr), \
                                                vgpr(tmpSrcVgpr), vgpr(dstVgpr), "dst = tmpSrc | dst")
                                        writer.vgprPool.checkIn(tmpSrcVgpr)

                            else: # f16 d1
                                if tt%2==0:
                                    kStr += inst("v_mov_b32", vgpr(dstVgpr), \
                                            vgpr(srcVgpr), comment)
                                else:
                                    pass # above shift moves two f16

                        # f32 or larger
                        else:
                            for i in range(0, writer.bpeCinternal//writer.bpr):
                                kStr += inst("v_mov_b32", vgpr(writer.startVgprValuC+dst*writer.bpeCinternal//writer.bpr+i), \
                                        vgpr(writer.startVgprValuC+src*writer.bpeCinternal//writer.bpr+i), comment)

                # end shift reset mask and jump out
                all1mask = "0xFFFFFFFF" if (kernel["WavefrontSize"] == 32) else "0xFFFFFFFFFFFFFFFF"
                kStr += inst("s_mov_b{}".format(kernel["WavefrontSize"]), sgpr(tmpSgpr,writer.laneSGPRCount), \
                        all1mask, "to restore all threads active")
                kStr += inst("s_or_saveexec_b{}".format(kernel["WavefrontSize"]), writer.vcc, sgpr(tmpSgpr,writer.laneSGPRCount), \
                        "all threads active")
                kStr += inst("s_branch label_%04u"%svrLabels[vw-1], \
                        "done shifting" )
        kStr += "label_%04u: // end shift0%s" % (svrLabels[vw-1], writer.endLine)

        # checkin scratch vgprs
        writer.vgprPool.checkIn(wgMT)
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(qReg)
        writer.vgprPool.checkIn(rReg)
        writer.vgprPool.checkIn(sReg)
        writer.vgprPool.checkIn(eReg)
        writer.vgprPool.checkIn(thread)
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(vReg)
        return kStr

class ShiftVectorComponentsMFMA(ShiftVectorComponents):
    kernel = {"EnableMatrixInstruction": True}

    """
    Shift Vector Components d0,1
    """
    def __call__(self, writer, kernel, tP):
        """ when we enable shift ptr with vectorwidth(2), we shift global read on edge block when size % vectorwidth != 0.
            For example if M size == 3 vector width == 2, we want to do global read for [0-1] and [2-3].
            But 3 is not in memory object, so we shift to do global read [0-1] and [1-2].
            So buffer become [0, 1, 1, 2], assume result in register is same as input [0, 1, 1, 2]
            We need to shift it back to [0, 1, 2].

            In MFMA outputs, We have numContinuousOutput(4) for each thread.
            We have numThreadInWave(64) threads.
            number of thread in N is sames as kernel["MatrixInstN"] (32)
            number of thread in M is numThreadInWave/numOutputThreads1 = 2
            stride of continous output for each thread (numSubOutputPerWave0) is numOutputThreads0 * numContinuousOutput, (8).
            we have numSubOutputGroupsPerWave0 which is 4 (kernel[tP["mt"]](64) // numSubOutputPerWave0(8))

            So we do shift back by below alorithm.
            1. check if M_size % vectorwidth != 0, return if == 0
            2. decide which subgroup we need to shift, M_size(3) means 3/8 = group 0
            3. decide which thread we need to shift, we have different groups of thread, (0-31) for first group, (32-63) for second group.
            4. decide which shift block (subTile1) we want to shift. for ex [0-1], [1-2], we want to shift second subtile
        """

        kStr = ""

        glvw                       = tP["glvw"]
        numThreadInWave            = writer.kernel["WavefrontSize"]
        MIBShape0                  = kernel["MatrixInstM"] * kernel["MatrixInstBM"]
        numContinuousOutput        = kernel["MIOutputVectorWidth"]
        numOutputThreads1          = kernel["MatrixInstN"]
        numOutputThreads0          = kernel["MatrixInstBM"] if (kernel["MatrixInstM"] == 4) else (numThreadInWave // numOutputThreads1)
        numSubOutputPerWave0       = numOutputThreads0 * numContinuousOutput
        numSubOutputGroupsPerWave0 = MIBShape0 // numSubOutputPerWave0
        numShiftBlock              = numContinuousOutput // glvw
        if kernel["ProblemType"]["DataType"].isDouble():
            numShiftBlock *= 2
        numOutputElements          = numSubOutputGroupsPerWave0 * numContinuousOutput * kernel["MIWaveTile"][0]
        subTile1                   = kernel["MIWaveTile"][1] if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBN"] * kernel["MIWaveTile"][1]

        # labels for reminder of vectorwidth
        svrLabels = []
        # label for reminder of subgroup
        sviLabels = []
        # label for reminder of shift block(subtile)
        svoLabels = []

        for i in range(0, glvw):
            r = (i+1) % glvw    # r = [1,2,3,...,glvw-1, 0], the last one svrLabels[glvw-1] stores for r=0 -> no shift
            label = writer.getLabelNum("ShiftVectorComponents%u_R%u" % (tP["idx"], r) )
            svrLabels.append(label)
            tmpLabels = []
            tmp2Labels = []
            for wt in range(0, kernel["MIWaveTile"][0]):
                for v in range(0, numSubOutputGroupsPerWave0):
                    label = writer.getLabelNum("ShiftVectorComponents%u_R%u_WT%u_V%u" % (tP["idx"], r, wt, v) )
                    tmpLabels.append(label)
                    tmp2Labels2 = []
                    for o in range(0, numShiftBlock):
                        label = writer.getLabelNum("ShiftVectorComponents%u_R%u_Wt%u_V%u_O%u" % (tP["idx"], r, wt, v, o) )
                        tmp2Labels2.append(label)
                    tmp2Labels.append(tmp2Labels2)
            sviLabels.append(tmpLabels)
            svoLabels.append(tmp2Labels)

        # wgMT value
        tmpSgpr = writer.getTmpSgpr(writer.laneSGPRCount).idx()
        tmpVgpr = writer.vgprPool.checkOutAligned(2,2)
        dummy   = writer.vgprPool.checkOut(1)
        wgMT    = writer.vgprPool.checkOut(1)
        wg      = tP["prevWg"] if writer.prefetchAcrossPersistent else tP["wg"]

        # get M size of edge block
        mtReg = writer.vgprPool.checkOut(1)
        kStr += inst("v_mov_b32"    , vgpr(wgMT), sgpr(wg), "")
        kStr += inst("v_mul_i32_i24", vgpr(wgMT), hex(-kernel[tP["mt"]]), vgpr(wgMT), "wg*MT")
        kStr += inst("_v_add_co_u32", vgpr(wgMT), writer.vcc, sgpr("SizesFree+%u"%tP["idx"]), vgpr(wgMT), "wgMT = Size - wg*MT")
        kStr += inst("v_mov_b32"    , vgpr(mtReg), hex(kernel[tP["mt"]]), "MT")
        kStr += inst("v_cmp_lt_u32" , sgpr(tmpSgpr,writer.laneSGPRCount), vgpr(wgMT), vgpr(mtReg), "wgMT < MT" )
        kStr += inst("v_cndmask_b32", vgpr(wgMT), vgpr(mtReg), vgpr(wgMT), sgpr(tmpSgpr,writer.laneSGPRCount), "wgMT = (wgMT < MT) ? wgMT : MT" )

        wReg = writer.vgprPool.checkOut(1)
        kStr += vectorStaticDivide(wReg, "Serial", writer.kernel["WavefrontSize"], tmpVgpr, tmpSgpr)
        kStr += vectorStaticRemainder(dummy, wReg, wReg, kernel["MIWaveGroup"][0], tmpVgpr, tmpSgpr)
        sReg = writer.vgprPool.checkOut(1)
        kStr += vectorStaticDivide(sReg, wgMT, MIBShape0, tmpVgpr, tmpSgpr)
        kStr += vectorStaticRemainder(dummy, sReg, sReg, kernel["MIWaveGroup"][0], tmpVgpr, tmpSgpr)
        kStr += inst("v_cmp_eq_u32" , sgpr(tmpSgpr,writer.laneSGPRCount), vgpr(sReg), vgpr(wReg), "wave_id0 == block_belong_to_wave0?" )
        kStr += inst("v_cndmask_b32", vgpr(wgMT), vgpr(mtReg), vgpr(wgMT), sgpr(tmpSgpr,writer.laneSGPRCount), "wgMT = (wgMT < MT) ? wgMT : MT" )
        writer.vgprPool.checkIn(mtReg)
        writer.vgprPool.checkIn(sReg)

        # gReg : group id of numSubOutputGroupsPerWave0
        kStr += writer.comment("gReg : group id of numSubOutputGroupsPerWave0")
        gReg = writer.vgprPool.checkOut(1)
        kStr += staticMultiply(vgpr(wReg), vgpr(wReg), MIBShape0 // numSubOutputPerWave0, sgpr(tmpSgpr))
        kStr += vectorStaticDivide(gReg, wgMT, numSubOutputPerWave0, tmpVgpr, tmpSgpr)
        kStr += inst("_v_sub_u32", vgpr(gReg), vgpr(gReg), vgpr(wReg), "")
        writer.vgprPool.checkIn(wReg)

        # eReg : use to disguish which shift block (sub-tile) we need to deal with
        kStr += writer.comment("eReg : use to disguish which shift block (sub-tile) we need to deal with")
        eReg = writer.vgprPool.checkOut(1)
        if kernel["ProblemType"]["DataType"].isDouble():
            kStr += vectorStaticDivide(eReg, wgMT, numContinuousOutput, tmpVgpr, tmpSgpr)
            kStr += vectorStaticRemainder(dummy, eReg, eReg, numContinuousOutput, tmpVgpr, tmpSgpr)
        else:
            kStr += vectorStaticRemainder(dummy, eReg, wgMT, numContinuousOutput, tmpVgpr, tmpSgpr)

        # mReg : decide which thread have to deal with this M-size
        kStr += writer.comment("mReg : decide which thread have to deal with this M-size")
        mReg = writer.vgprPool.checkOut(1)
        if kernel["ProblemType"]["DataType"].isDouble():
            if kernel["SourceSwap"]:
                kStr += vectorStaticRemainder(dummy, mReg, wgMT, MIBShape0, tmpVgpr, tmpSgpr)
                kStr += vectorStaticDivide(mReg, mReg, glvw, tmpVgpr, tmpSgpr)
            else:
                kStr += vectorStaticRemainder(dummy, mReg, wgMT, numContinuousOutput, tmpVgpr, tmpSgpr)
                kStr += vectorStaticDivide(mReg, mReg, numContinuousOutput // 2, tmpVgpr, tmpSgpr)
        else:
            kStr += vectorStaticDivide(mReg, wgMT, numContinuousOutput, tmpVgpr, tmpSgpr)
            kStr += vectorStaticRemainder(dummy, mReg, mReg, numOutputThreads0, tmpVgpr, tmpSgpr)

        # tReg : thread group id [0-31] or [32-63] for mfma 32x32x2
        kStr += writer.comment("tReg : thread group id [0-31] or [32-63] for mfma 32x32x2")
        tReg = writer.vgprPool.checkOut(1)
        if kernel["ProblemType"]["DataType"].isDouble():
            if kernel["SourceSwap"]:
                kStr += vectorStaticDivide(tReg, "Serial", glvw, tmpVgpr, tmpSgpr)
                kStr += vectorStaticRemainder(dummy, tReg, tReg, numOutputThreads1 // 2, tmpVgpr, tmpSgpr)
            else:
                kStr += vectorStaticDivide(tReg, "Serial", kernel["MatrixInstN"] * 2, tmpVgpr, tmpSgpr)
                kStr += vectorStaticRemainder(dummy, tReg, tReg, numOutputThreads0 // 2, tmpVgpr, tmpSgpr)
        else:
            kStr += vectorStaticDivide(tReg, "Serial", kernel["MatrixInstN"], tmpVgpr, tmpSgpr)
            kStr += vectorStaticRemainder(dummy, tReg, tReg, numOutputThreads0, tmpVgpr, tmpSgpr)

        # rReg : reminder of M_size % vectorwidth
        # decide to jump to block which handle this case, M_size % vector width
        kStr += writer.comment("rReg : reminder of M_size % vectorwidth")
        rReg = writer.vgprPool.checkOut(1)
        kStr += vectorStaticRemainder(dummy, rReg, wgMT, glvw, tmpVgpr, tmpSgpr)
        for r in range(1, glvw):
            kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(rReg), hex(r), "wgMT%%VW == %u"%r )
            kStr += inst("s_cbranch_vccnz label_%04u" % svrLabels[(r-1)], "branch to shift d%u r=%u"%(tP["idx"], r))
        kStr += inst("s_branch label_%04u"%svrLabels[glvw-1], "no shifting" )
        writer.vgprPool.checkIn(rReg)

        _, arch2acc = writer.AccToArchMapper(kernel)

        # blocks for handle M_size % vector width
        for r in range(1, glvw):
            kStr += writer.comment3("shift d%u r=%u"%(tP["idx"], r))
            kStr += "label_%04u:%s" % (svrLabels[r-1], writer.endLine)

            for wt in range(0, kernel["MIWaveTile"][0]):
                # decide to jump to block wich handle sub group id for numSubOutputGroupsPerWave0
                # we have 8 blocks for MT-M 64 with mfma 32x32x2. 64/2(thread group)/4(continous output)
                for ot in range(0, numSubOutputGroupsPerWave0):
                    packIdx = wt * numSubOutputGroupsPerWave0 + ot
                    grpVal  = wt * numSubOutputGroupsPerWave0 * kernel["MIWaveGroup"][0] + ot
                    kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(gReg), hex(grpVal), "wgMT/8 == %u" % packIdx )
                    kStr += inst("s_cbranch_vccnz label_%04u" % sviLabels[(r-1)][packIdx], "branch to shift d%u, r=%u, v=%u" % (tP["idx"], r, packIdx))

            for wt in range(0, kernel["MIWaveTile"][0]):
                # blocks for handle sub group id for numSubOutputGroupsPerWave0
                for ot in range(0, numSubOutputGroupsPerWave0):
                    packIdx = wt * numSubOutputGroupsPerWave0 + ot
                    kStr += writer.comment("shift d%u r=%u v=%u" % (tP["idx"], r, packIdx))
                    kStr += "label_%04u:%s" % (sviLabels[r-1][packIdx], writer.endLine)

                    cmt = "(serial % 64) / 32 == (wgMT/4)%2"
                    if kernel["SourceSwap"]:
                        cmt = "(serial / glvw) % wt0 == (wgMT % mib0) / glvw"
                    kStr += inst("v_cmpx_eq_u32", sgpr(tmpSgpr,writer.laneSGPRCount), vgpr(tReg), vgpr(mReg), cmt )

                    if not kernel["SourceSwap"]:
                        # decide to jump to block wich handle element of shfit block (subtile)
                        # for vector widht 2 with continuous 4, we have 1, 3 case to handle
                        for outIdx in range(0, numShiftBlock):
                            if kernel["ProblemType"]["DataType"].isDouble():
                                kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(eReg), hex(outIdx), "wgMT %% 4 == %u" % (outIdx) )
                            else:
                                kStr += inst("v_cmp_eq_u32", writer.vcc, vgpr(eReg), hex(outIdx*glvw+r), "wgMT %% 4 == %u" % (outIdx*2+1) )
                            kStr += inst("s_cbranch_vccnz label_%04u" % svoLabels[(r-1)][packIdx][outIdx], "branch to shift d%u, r=%u, v=%u, o=%u" % (tP["idx"], r, packIdx, outIdx))

                    # blocks to handle shfiting
                    for outIdx in range(0, numShiftBlock):
                        kStr += "label_%04u:%s" % (svoLabels[(r-1)][packIdx][outIdx], writer.endLine)
                        for subTile1Idx in range(0, subTile1):
                            for shiftIdx in range(0, r):
                                if kernel["ProblemType"]["DataType"].isDouble():
                                    tmpVgpr2 = writer.vgprPool.checkOutAligned(2,2)

                                    dstVgpr = 2 * (subTile1Idx * numOutputElements + packIdx * numContinuousOutput + outIdx + shiftIdx)
                                    if kernel["SourceSwap"]:
                                        dstVgpr = 2 * (subTile1Idx * numOutputElements + outIdx * kernel["MIWaveTile"][0] + packIdx + shiftIdx)

                                    swapSize = 1 if kernel["SourceSwap"] else 16
                                    kStr += inst("v_accvgpr_read_b32", vgpr(tmpVgpr), accvgpr(arch2acc[dstVgpr]), "")
                                    kStr += inst("s_nop", "1", "v_accvgpr read vgpr after write vgpr: 2 wait states")
                                    kStr += inst("ds_swizzle_b32", vgpr(tmpVgpr2), vgpr(tmpVgpr), "offset:swizzle(SWAP, {})".format(swapSize), "swizzle edge values")
                                    kStr += inst("s_waitcnt", "0", "wait for swizzle operation")
                                    kStr += inst("v_accvgpr_write_b32", accvgpr(arch2acc[dstVgpr]), vgpr(tmpVgpr2), "")

                                    kStr += inst("v_accvgpr_read_b32", vgpr(tmpVgpr), accvgpr(arch2acc[dstVgpr]+1), "")
                                    kStr += inst("s_nop", "1", "v_accvgpr read vgpr after write vgpr: 2 wait states")
                                    kStr += inst("ds_swizzle_b32", vgpr(tmpVgpr2), vgpr(tmpVgpr), "offset:swizzle(SWAP, {})".format(swapSize), "swizzle edge values")
                                    kStr += inst("s_waitcnt", "0", "wait for swizzle operation")
                                    kStr += inst("v_accvgpr_write_b32", accvgpr(arch2acc[dstVgpr]+1), vgpr(tmpVgpr2), "")

                                    writer.vgprPool.checkIn(tmpVgpr2)
                                else:
                                    dstVgpr = subTile1Idx * numOutputElements + packIdx * numContinuousOutput + outIdx * glvw + shiftIdx
                                    srcVgpr = subTile1Idx * numOutputElements + packIdx * numContinuousOutput + outIdx * glvw + shiftIdx + (glvw - r)
                                    if writer.serializedStore:
                                        kStr += inst("v_accvgpr_read_b32", vgpr(tmpVgpr), accvgpr(arch2acc[srcVgpr]), "")
                                        kStr += inst("s_nop", "1", "v_accvgpr read vgpr after write vgpr: 2 wait states")
                                        kStr += inst("v_accvgpr_write_b32", accvgpr(arch2acc[dstVgpr]), vgpr(tmpVgpr), "acc%u = acc%u"%(arch2acc[dstVgpr], arch2acc[srcVgpr]))
                                        if writer.agprMultiplier == 2:
                                            accImOffset = writer.AccVgprImagNumOffset(kernel)
                                            kStr += inst("v_accvgpr_read_b32", vgpr(tmpVgpr), accvgpr(arch2acc[srcVgpr]+accImOffset), "")
                                            kStr += inst("s_nop", "1", "v_accvgpr read vgpr after write vgpr: 2 wait states")
                                            kStr += inst("v_accvgpr_write_b32", accvgpr(arch2acc[dstVgpr]+accImOffset), vgpr(tmpVgpr), "acc%u (imag)= acc%u (imag)"%(arch2acc[dstVgpr] + accImOffset, arch2acc[srcVgpr] + accImOffset))
                                    else:
                                        kStr += inst("v_mov_b32", vgpr(dstVgpr), vgpr(srcVgpr), "")

                    # end shift reset mask and jump out
                    all1mask = "0xFFFFFFFF" if (kernel["WavefrontSize"] == 32) else "0xFFFFFFFFFFFFFFFF"
                    kStr += inst("s_mov_b{}".format(kernel["WavefrontSize"]), sgpr(tmpSgpr,writer.laneSGPRCount), all1mask, "to restore all threads active")
                    kStr += inst("s_or_saveexec_b{}".format(kernel["WavefrontSize"]), writer.vcc, sgpr(tmpSgpr,writer.laneSGPRCount), "all threads active")
                    kStr += inst("s_branch label_%04u" % svrLabels[glvw-1], "done shifting" )

        kStr += "label_%04u: // end shift0%s" % (svrLabels[glvw-1], writer.endLine)

        # checkin scratch vgprs
        writer.vgprPool.checkIn(tmpVgpr)
        writer.vgprPool.checkIn(wgMT)
        writer.vgprPool.checkIn(dummy)
        writer.vgprPool.checkIn(gReg)
        writer.vgprPool.checkIn(eReg)
        writer.vgprPool.checkIn(mReg)
        writer.vgprPool.checkIn(tReg)
        return kStr

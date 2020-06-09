################################################################################
# Copyright 2020 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import MAC
from ..DataType import DataType

class FMA_NonPacked(MAC):
    asmCaps = {"v_fma_f16": True,
               "v_pk_fma_f16": False}
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel

        kStr = self.commentHeader()
        beAggressive = kernel["AggressivePerfMode"]

        doOnce = False

        for blockB in range(0, kernel["ThreadTile1"]//2):
            for blockA in range(0, kernel["ThreadTile0"]//2):
                for iui in range(0, innerUnroll):
                    cStr1 = "v[%s+%u+%u*%u+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # /2 b/c of 2 f16's per 32-bit vgpr
                    cStr2 = "v[%s+%u+%u*%u+%u]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                    aStr = "v[%s+%u]" % ("vgprValuA_X%u_I%u"%(m,iui), blockA)
                    bStr = "v[%s+%u]" % ("vgprValuB_X%u_I%u"%(m,iui), blockB)
                    kStr += "v_fma_f16 %s, %s, %s, %s op_sel:[0,0,0,0] %s" % (cStr1, aStr, bStr, cStr1, writer.endLine)
                    kStr += "v_fma_f16 %s, %s, %s, %s op_sel:[0,1,0,0] %s" % (cStr2, aStr, bStr, cStr2, writer.endLine)
                    if beAggressive and not doOnce:
                        kStr += "s_setprio 1 // Raise priority while processing macs%s" % writer.endLine
                        doOnce = True
                    kStr += "v_fma_f16 %s, %s, %s, %s op_sel:[1,0,1,1] %s" % (cStr1, aStr, bStr, cStr1, writer.endLine)
                    kStr += "v_fma_f16 %s, %s, %s, %s op_sel:[1,1,1,1] %s" % (cStr2, aStr, bStr, cStr2, writer.endLine)

        if beAggressive:
            kStr += "s_setprio 0 // Reset priority after macs %s" % writer.endLine

        return kStr

class FMA_HPA_MAD_MIX(MAC):
    asmCaps = {"v_mad_mix_f32": True}
    #archCaps = {}
    kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                              "HighPrecisionAccumulate": True},
              "LocalDotLayout": lambda ldl: ldl > 1
             }

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel

        kStr = self.commentHeader()
        beAggressive = kernel["AggressivePerfMode"]

        doOnce = False

        for blockB in range(0, kernel["ThreadTile1"]//2):
            for blockA in range(0, kernel["ThreadTile0"]//2):
                # we treat HighPrecisionAccumulate as expanded packed math
                if kernel["LocalDotLayout"] > 1 and innerUnroll == 2:    # Only supports LocalDotLayout == 2 for now
                    cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                    cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 0
                    aStr = "v[%s+%u]" % ("vgprValuA_X%u_I0"%m, blockA)
                    bStr = "v[%s+%u]" % ("vgprValuB_X%u_I0"%m, blockB)
                    kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, writer.endLine)
                    if beAggressive and not doOnce:
                        kStr += "s_setprio 1 // Raise priority while processing macs%s" % writer.endLine
                        doOnce = True
                    kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u] %s" % (cStr, aStr, bStr, cStr, cidx, writer.endLine)
                    cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + 1
                    aStr = "v[%s+%u]" \
                        % ("vgprValuA_X%u_I1"%m, blockA)
                    bStr = "v[%s+%u]" \
                        % ("vgprValuB_X%u_I0"%m, blockB)
                    cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"]) # *2 b/c of fp32
                    kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, writer.endLine)
                    kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, writer.endLine)
                    cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                    aStr = "v[%s+%u]" \
                        % ("vgprValuA_X%u_I0"%m, blockA)
                    bStr = "v[%s+%u]" \
                        % ("vgprValuB_X%u_I1"%m, blockB)
                    cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                    kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, writer.endLine)
                    kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, writer.endLine)
                    cidx = blockA*2 + blockB*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                    aStr = "v[%s+%u]" \
                        % ("vgprValuA_X%u_I1"%m, blockA)
                    bStr = "v[%s+%u]" \
                        % ("vgprValuB_X%u_I1"%m, blockB)
                    cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", blockA, blockB, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                    kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[0,0,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, writer.endLine)
                    kStr += "v_mad_mix_f32 %s, %s, %s, %s op_sel:[1,1,0] op_sel_hi:[1,1,0] //valuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, writer.endLine)
                    #kStr += writer.bomb(-13)
                    """
                    ignore this, not quite correct for mixed precision
                    D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                    D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                    C[0] = A[0]*B[0]+D[0]
                    C[1] = A[1]*B[1]+D[1]
                    """

        return kStr

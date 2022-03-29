################################################################################
# Copyright 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from ..Component import Component, MAC
from ..DataType import DataType

class FMA_BF16_HPA(MAC):
    asmCaps = {"v_fma_f32": True}
    kernel = {"ProblemType": {"DataType": DataType(DataType.bfloat16),
                              "HighPrecisionAccumulate": False}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel
        kStr = self.commentHeader()
        priority = Component.Priority.find(writer)
        for iui in range(0, innerUnroll):
            for blockA in range(kernel["ThreadTileA"]//2-1, -1, -1):
                kStr += "v_and_b32         v[vgprValuA_X%u_I%u+%u], 0xffff0000, v[vgprValuA_X%u_I%u+%u]%s" % (m, iui, blockA*2+1, m, iui, blockA, self.endLine)
                kStr += "v_lshlrev_b32 v[vgprValuA_X%u_I%u+%u], 16,                 v[vgprValuA_X%u_I%u+%u]%s" % (m, iui, blockA*2,     m, iui, blockA, self.endLine)

            for blockB in range(kernel["ThreadTileB"]//2-1, -1, -1):
                kStr += "v_and_b32         v[vgprValuB_X%u_I%u+%u], 0xffff0000, v[vgprValuB_X%u_I%u+%u]%s" % (m, iui, blockB*2+1, m, iui, blockB, self.endLine)
                kStr += "v_lshlrev_b32 v[vgprValuB_X%u_I%u+%u], 16,                 v[vgprValuB_X%u_I%u+%u]%s" % (m, iui, blockB*2,     m, iui, blockB, self.endLine)

        for block1 in range(0, kernel["ThreadTile1"]//2):
            for block0 in range(0, kernel["ThreadTile0"]//2):
                if kernel["ProblemType"]["HighPrecisionAccumulate"]:
                    # we treat HighPrecisionAccumulate as expanded packed math
                    for iui in range(0, innerUnroll):

                        blockA = block0 if self.tPB["tile01Idx"] else block1
                        blockB = block1 if self.tPB["tile01Idx"] else block0

                        aStr0 = "v[%s+%u]" % ("vgprValuA_X%u_I%u"%(m,iui), blockA*2+0)
                        aStr1 = "v[%s+%u]" % ("vgprValuA_X%u_I%u"%(m,iui), blockA*2+1)
                        bStr0 = "v[%s+%u]" % ("vgprValuB_X%u_I%u"%(m,iui), blockB*2+0)
                        bStr1 = "v[%s+%u]" % ("vgprValuB_X%u_I%u"%(m,iui), blockB*2+1)

                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + 0
                        cStr = "v[%s+%u*2+%u*%u*2+0*2+0]" % ("vgprValuC", block0, block1, kernel["ThreadTile0"]) # *2 b/c of fp32
                        kStr += "v_fma_f32 %s, %s, %s, %s //ValuC[%u]%s" % (cStr, aStr0, bStr0, cStr, cidx, self.endLine)

                        kStr += priority(writer, 1, "Raise priority while processing macs")

                        aStr = aStr1 if self.tPB["tile01Idx"] else aStr0
                        bStr = bStr0 if self.tPB["tile01Idx"] else bStr1
                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + 1
                        cStr = "v[%s+%u*2+%u*%u*2+0*2+1]" % ("vgprValuC", block0, block1, kernel["ThreadTile0"]) # *2 b/c of fp32
                        kStr += "v_fma_f32 %s, %s, %s, %s //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)

                        aStr = aStr0 if self.tPB["tile01Idx"] else aStr1
                        bStr = bStr1 if self.tPB["tile01Idx"] else bStr0
                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 0
                        cStr = "v[%s+%u*2+%u*%u*2+%u*2+0]" % ("vgprValuC", block0, block1, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                        kStr += "v_fma_f32 %s, %s, %s, %s //ValuC[%u]%s" % (cStr, aStr, bStr, cStr, cidx, self.endLine)

                        cidx = block0*2 + block1*kernel["ThreadTile0"]*2 + kernel["ThreadTile0"] + 1
                        cStr = "v[%s+%u*2+%u*%u*2+%u*2+1]" % ("vgprValuC", block0, block1, kernel["ThreadTile0"], kernel["ThreadTile0"]//2)
                        kStr += "v_fma_f32 %s, %s, %s, %s //valuC[%u]%s" % (cStr, aStr1, bStr1, cStr, cidx, self.endLine)
                        """
                        ignore this, not quite correct for mixed precision
                        D.f[31:16] = S0.f[31:16] * S1.f[31:16] + S2.f[31:16]
                        D.f[15:00] = S0.f[15:00] * S1.f[15:00] + S2.f[15:00]
                        C[0] = A[0]*B[0]+D[0]
                        C[1] = A[1]*B[1]+D[1]
                        """
                        #kStr += self.bomb(-13)

        kStr += priority(writer, 0, "Reset priority after macs")
        return kStr

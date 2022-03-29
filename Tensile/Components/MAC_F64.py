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

class MAC_F64_Plain(MAC):
    asmCaps = {"v_fma_f64": True}
    kernel = {"ProblemType": {"DataType": DataType(DataType.double)}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel
        kStr = self.commentHeader()
        priority = Component.Priority.find(writer)

        for b in range(0, kernel["ThreadTile1"]):
            for a in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    cStr = "v[%s+(%u+%u*%u)*2:(%s+%u+%u*%u)*2+1]" % ("vgprValuC", a, b, kernel["ThreadTile0"], "vgprValuC", a, b, kernel["ThreadTile0"])
                    aStr = "v[%s+%u*2:%s+%u*2+1]" \
                            % ("vgprValuA_X%u_I%u"%(m,iui) , a, "vgprValuA_X%u_I%u"%(m,iui), a)
                    bStr = "v[%s+%u*2:%s+%u*2+1]" \
                            % ("vgprValuB_X%u_I%u"%(m,iui) , b, "vgprValuB_X%u_I%u"%(m,iui), b)
                    kStr += "v_fma_f64 %s, %s, %s, %s%s" % (cStr, aStr, bStr, cStr, writer.endLine)
                    kStr += priority(writer, 1, "Raise priority while processing macs")

        kStr += priority(writer, 0, "Reset priority after macs")
        return kStr

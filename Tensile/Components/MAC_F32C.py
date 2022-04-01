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

class MAC_F32C_Plain(MAC):
    kernel = {"ProblemType": {"DataType": DataType(DataType.complexSingle)}}

    def __call__(self, writer, m, innerUnroll):
        kernel = writer.kernel
        kStr = self.commentHeader()
        priority = Component.Priority.find(writer)

        vars = {}
        vars["endLine"] = writer.endLine
        vars["m"] = m
        vars["ThreadTile0"] = kernel["ThreadTile0"]

        for b in range(0, kernel["ThreadTile1"]):
            for a in range(0, kernel["ThreadTile0"]):
                for iui in range(0, innerUnroll):
                    vars["a"] = a
                    vars["b"] = b
                    vars["iui"] = iui

                    vars["cStr"] = "v[vgprValuC+({a}+{b}*{ThreadTile0})*2]".format_map(vars)
                    vars["aStr"] = "v[vgprValuA_X{m}_I{iui}+{a}*2]".format_map(vars)
                    vars["bStr"] = "v[vgprValuB_X{m}_I{iui}+{b}*2]".format_map(vars)
                    kStr += "_v_mac_f32 {cStr}, {aStr}, {bStr}{endLine}".format_map(vars)

                    vars["cStr"] = "v[vgprValuC+({a}+{b}*{ThreadTile0})*2]".format_map(vars)
                    vars["aStr"] = "v[vgprValuA_X{m}_I{iui}+{a}*2+1]".format_map(vars)
                    vars["bStr"] = "v[vgprValuB_X{m}_I{iui}+{b}*2+1]".format_map(vars)
                    vars["sign"] = "-" if (not kernel["ProblemType"]["ComplexConjugateA"] and not kernel["ProblemType"]["ComplexConjugateB"]) or \
                            (kernel["ProblemType"]["ComplexConjugateA"] and kernel["ProblemType"]["ComplexConjugateB"]) else ""
                    kStr += "_v_mac_f32 {cStr}, {sign}{aStr}, {bStr}{endLine}".format_map(vars)

                    vars["cStr"] = "v[vgprValuC+({a}+{b}*{ThreadTile0})*2+1]".format_map(vars)
                    vars["aStr"] = "v[vgprValuA_X{m}_I{iui}+{a}*2]".format_map(vars)
                    vars["bStr"] = "v[vgprValuB_X{m}_I{iui}+{b}*2+1]".format_map(vars)
                    vars["sign"] = "-" if kernel["ProblemType"]["ComplexConjugateB"] else ""
                    kStr += "_v_mac_f32 {cStr}, {aStr}, {sign}{bStr}{endLine}".format_map(vars)

                    vars["cStr"] = "v[vgprValuC+({a}+{b}*{ThreadTile0})*2+1]".format_map(vars)
                    vars["aStr"] = "v[vgprValuA_X{m}_I{iui}+{a}*2+1]".format_map(vars)
                    vars["bStr"] = "v[vgprValuB_X{m}_I{iui}+{b}*2]".format_map(vars)
                    vars["sign"] = "-" if kernel["ProblemType"]["ComplexConjugateA"] else ""
                    kStr += "_v_mac_f32 {cStr}, {sign}{aStr}, {bStr}{endLine}".format_map(vars)

                    kStr += priority(writer, 1, "Raise priority while processing macs")
        kStr += priority(writer, 0, "Reset priority after macs")
        return kStr

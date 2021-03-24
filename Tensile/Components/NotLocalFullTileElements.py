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

from ..Component import NotLocalFullTileElements
from ..Common import globalParameters

class NotLocalFullTileElementsVALU(NotLocalFullTileElements):
    kernel = {"EnableMatrixInstruction": False}

    """
    Partition thread-tile into writeElements for store code
    This function creates the writeElement mapping for full tiles
    (ie non-edge cases)
    """
    def __call__(self, writer, kernel, edge):
        elements        = []
        vectorwidth = 0

        if edge:
            vectorwidth = kernel["VectorWidth"] if kernel["_VectorStore"] else 1
            vectorwidth = min(vectorwidth, writer.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
            assert(kernel["VectorWidth"] % vectorwidth == 0)
        else:
            vectorwidth = kernel["VectorWidth"] if kernel["_VectorStore"] else 1
            vectorwidth = min(vectorwidth, writer.maxGwvw(kernel))

        # Full tile loop:
        for tt1 in range(0, kernel["ThreadTile1"]//kernel["VectorWidth"]):
            for vc1 in range(0, kernel["VectorWidth"]):
                for tt0 in range(0, kernel["ThreadTile0"]//kernel["VectorWidth"]):
                    for vc0 in range(0, kernel["VectorWidth"], vectorwidth): # note step by fullVw
                        element = (tt1, tt0, vc1, vc0)
                        elements.append(element)

        return (vectorwidth, elements)

class NotLocalFullTileElementsMFMA(NotLocalFullTileElements):
    kernel = {"EnableMatrixInstruction": True,
              "SourceSwap": False}

    """
    Partition thread-tile into writeElements for store code
    This function creates the writeElement mapping for full tiles
    (ie non-edge cases)
    """
    def __call__(self, writer, kernel, edge):
        elements        = []
        vectorwidth = 0

        if edge:
            vectorwidth = kernel["StoreVectorWidth"] if kernel["_VectorStore"] else 1
            vectorwidth = min(vectorwidth, writer.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
        else:
            vectorwidth = kernel["StoreVectorWidth"] if kernel["_VectorStore"] else 1
            vectorwidth = min(vectorwidth, writer.maxGwvw(kernel))

        MFMAcontinoutsOuptut = kernel["MIOutputVectorWidth"]

        if kernel["MatrixInstM"] == 4:
            totalTT0            = kernel["MIWaveTile"][0] * MFMAcontinoutsOuptut
            totalTT1            = kernel["MIWaveTile"][1]
        else:
            outputsPerThread    = kernel["MatrixInstM"] * kernel["MatrixInstN"] // globalParameters["WavefrontWidth"]
            totalTT0            = kernel["MatrixInstBM"] * kernel["MIWaveTile"][0] * outputsPerThread
            totalTT1            = kernel["MatrixInstBN"] * kernel["MIWaveTile"][1]

        for tt1 in range(0, totalTT1):
            for vc1 in range(0, 1):
                for tt0 in range(0, totalTT0 // MFMAcontinoutsOuptut):
                    for vc0 in range(0, MFMAcontinoutsOuptut, vectorwidth): # note step by vectorwidth
                        element = (tt1, tt0, vc1, vc0)
                        elements.append(element)

        return (vectorwidth, elements)

class NotLocalFullTileElementsMFMASwap(NotLocalFullTileElements):
    kernel = {"EnableMatrixInstruction": True,
              "SourceSwap": True}

    """
    Partition thread-tile into writeElements for store code
    This function creates the writeElement mapping for full tiles
    (ie non-edge cases)
    """
    def __call__(self, writer, kernel, edge):
        elements        = []
        vectorwidth = 0

        if edge:
            vectorwidth = kernel["StoreVectorWidth"] if kernel["_VectorStore"] else 1
            vectorwidth = min(vectorwidth, writer.maxGwvw(kernel), kernel["AssertFree0ElementMultiple"])
        else:
            vectorwidth = kernel["StoreVectorWidth"] if kernel["_VectorStore"] else 1
            vectorwidth = min(vectorwidth, writer.maxGwvw(kernel))

        MFMAcontinoutsOuptut = kernel["MIOutputVectorWidth"]

        if kernel["MatrixInstM"] == 4:
            totalTT0            = kernel["MIWaveTile"][0] * MFMAcontinoutsOuptut
            totalTT1            = kernel["MIWaveTile"][1]
        else:
            outputsPerThread    = kernel["MatrixInstM"] * kernel["MatrixInstN"] // globalParameters["WavefrontWidth"]
            totalTT0            = kernel["MatrixInstBM"] * kernel["MIWaveTile"][0] * outputsPerThread
            totalTT1            = kernel["MatrixInstBN"] * kernel["MIWaveTile"][1]

        for tt1 in range(0, totalTT1):
            for vc1 in range(0, 1):
                for vc0 in range(0, MFMAcontinoutsOuptut, vectorwidth): # note step by vectorwidth
                    for tt0 in range(0, totalTT0 // MFMAcontinoutsOuptut):
                        element = (tt1, tt0, vc1, vc0)
                        elements.append(element)


        return (vectorwidth, elements)

################################################################################
# Copyright (C) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

import pytest
import yaml

from Tensile.SolutionStructs import ProblemSizeRange, ProblemSizeRangeOld, sizeRange

def test_indexRange_Simple():
    assert list(sizeRange(2)) == [2]
    assert list(sizeRange(2,5)) == list(range(2,6))
    assert list(sizeRange(1,10,100)) == list(range(1,101,10))

    assert list(sizeRange(1,1,1,10)) == [1, 2, 4, 7]
    assert list(sizeRange(1,1,2,10)) == [1, 2, 5, 10]

@pytest.mark.parametrize('cls', [ProblemSizeRangeOld, ProblemSizeRange])
def test_Simple(cls):

    problemType = {"IndexAssignmentsA": [0,3],
                   "IndexAssignmentsB": [1,3],
                   "NumIndicesLD": 4,
                   "NumIndicesC": 3}

    config = [[3,2,6], 0, [1,2], [1,1,1,7]]

    psr = cls(problemType, config)

    expectedSizes = [(3, 3, 1, 1, 0, 0, 0, 0),
                     (5, 5, 1, 1, 0, 0, 0, 0),
                     (3, 3, 2, 1, 0, 0, 0, 0),
                     (5, 5, 2, 1, 0, 0, 0, 0),
                     (3, 3, 1, 2, 0, 0, 0, 0),
                     (5, 5, 1, 2, 0, 0, 0, 0),
                     (3, 3, 2, 2, 0, 0, 0, 0),
                     (5, 5, 2, 2, 0, 0, 0, 0),
                     (3, 3, 1, 4, 0, 0, 0, 0),
                     (5, 5, 1, 4, 0, 0, 0, 0),
                     (3, 3, 2, 4, 0, 0, 0, 0),
                     (5, 5, 2, 4, 0, 0, 0, 0),
                     (3, 3, 1, 7, 0, 0, 0, 0),
                     (5, 5, 1, 7, 0, 0, 0, 0),
                     (3, 3, 2, 7, 0, 0, 0, 0),
                     (5, 5, 2, 7, 0, 0, 0, 0)]

    assert psr.problemSizes == expectedSizes

    psrStr = str(psr)
    newConfig = yaml.load(psrStr)

    psrNew = cls(problemType, newConfig)

    assert psrNew.problemSizes == psr.problemSizes




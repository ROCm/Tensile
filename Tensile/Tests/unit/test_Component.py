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

import Tensile.Component  as Component
import Tensile.Components as Components
from Tensile.DataType import DataType

def test_PartialMatch():
    a = {'foo': True,
         'bar': 25,
         'baz': {'Enabled': True,
                 'Debug': False}}

    b = {'foo': True}

    assert Component.PartialMatch(b, a)
    assert not Component.PartialMatch(a, b)

    assert not Component.PartialMatch({'foo': False}, a)

    assert not Component.PartialMatch({'baz': {"Enabled": False}}, a)
    assert Component.PartialMatch({'baz': {"Enabled": True}},  a)

    assert not Component.PartialMatch({'baz': {"Error": True}}, a)

    assert not Component.PartialMatch({'bar': lambda x: x < 10}, a)
    assert Component.PartialMatch({'bar': lambda x: x > 10}, a)

class MockWriter:
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

navi = MockWriter(asmCaps = {'v_fma_f16': True,
                             'v_pk_fma_f16': False},
                  kernel = {"ProblemType": {"DataType": DataType(DataType.half),
                                            "HighPrecisionAccumulate": False},
                            "AggressivePerfMode": True,
                            "ThreadTile0": 4,
                            "ThreadTile1": 4},
                  endLine = '\n')

def test_find():
    found = Component.MAC.find(navi)
    assert isinstance(found, Components.MAC_F16.FMA_NonPacked)

def test_find2():
    found = Component.Component.find(navi)
    assert isinstance(found, Components.MAC_F16.FMA_NonPacked)

def test_MAC_F16_FMA_NonPacked():
    found = Component.MAC.find(navi)
    kernelText = found(navi, 2, 4)
    print(kernelText)

def test_componentPath():
    assert Components.MAC_F16.FMA_NonPacked.componentPath() == ["Component", "MAC", "FMA_NonPacked"]

################################################################################
# Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.
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
import itertools
import copy
from Tensile.Hardware import HardwarePredicate
from Tensile.SolutionLibrary import PredicateLibrary

def test_hardware_predicate_comparison():
    a = HardwarePredicate.FromISA((9,0,0))
    b = HardwarePredicate.FromISA((9,0,6))
    c = HardwarePredicate("TruePred")

    assert a < b
    assert a < c
    assert b < c

    assert not b < a
    assert not c < a
    assert not c < b

    assert not a < a
    assert not b < b
    assert not c < c

def predicate_library_objects():
    objs = [PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,0))}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,6))}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate.FromISA((9,0,8))}]),
            PredicateLibrary('Hardware', [{'predicate': HardwarePredicate('TruePred')}])
    ]

    return [copy.deepcopy(libs) for libs in itertools.permutations(objs)]

@pytest.mark.parametrize("libraries", predicate_library_objects())
def test_predicate_library_merge(libraries):
    lib = libraries[0]
    for lib2 in libraries[1:]:
        lib.merge(lib2)

    assert lib.rows[-1]['predicate'] == HardwarePredicate('TruePred')
    for r in lib.rows[:-1]:
        assert r['predicate'] != HardwarePredicate('TruePred')

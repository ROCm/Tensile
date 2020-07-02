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

import pytest
import os

from Tensile.ReplacementKernels import ReplacementKernels

def test_DefaultInstance():
    assert ReplacementKernels.Get("asdf") is None

    myReplacement = ReplacementKernels.Get("Cijk_Ailk_Bjlk_DB_MT48x64x4_SE_APM1_AF0EM1_AF1EM1_AMAS3_ASBE01_ASEM1_BL1_DTL0_DVO0_EPS1_FL1_GRVW2_GSU1_ISA906_IU1_K1_KLA_LBSPP0_LPA0_LPB0_LDL1_MTSM64_NLCA1_NLCB1_ONLL1_PBD0_PK0_PGR1_PLR0_RK1_SIA1_SU0_SUM0_SUS256_SRVW0_SVW4_SNLL0_TT6_4_TLDS0_USFGRO0_VAW1_VS1_VW2_WSGRA0_WSGRB0_WG8_16_1_WGM4")

    assert os.path.isfile(myReplacement)
    assert os.path.isabs(myReplacement)

def replacementDir(dirname):
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(scriptDir, 'replacement', dirname)

def test_BadFile():
    with pytest.raises(RuntimeError):
        obj = ReplacementKernels(replacementDir('bad_file'), 'V3')
        obj.get("asdf")

def test_DuplicateKernel():
    with pytest.raises(RuntimeError):
        obj = ReplacementKernels(replacementDir('duplicate_kernel'), 'V3')
        obj.get("asdf")

goodObjs = [ReplacementKernels(replacementDir('known_kernels_v2'), "V2"),
            ReplacementKernels(replacementDir('known_kernels_v3'), "V3")]

@pytest.mark.parametrize("obj", goodObjs)
def test_foo(obj):
    foo = obj.get('foo')
    assert os.path.isfile(foo)
    assert os.path.isabs(foo)
    assert foo.endswith('kernel_named_foo.txt')

@pytest.mark.parametrize("obj", goodObjs)
def test_bar(obj):
    bar = obj.get('bar')
    assert os.path.isfile(bar)
    assert os.path.isabs(bar)
    assert bar.endswith('kernel_named_bar.txt')

@pytest.mark.parametrize("obj", goodObjs)
def test_baz(obj):
    baz = obj.get('baz')
    assert os.path.isfile(baz)
    assert os.path.isabs(baz)
    assert baz.endswith('baz.s.txt')

@pytest.mark.parametrize("obj", goodObjs)
def test_unknown(obj):
    assert obj.get('asdfds') is None

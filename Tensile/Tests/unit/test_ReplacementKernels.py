################################################################################
#
# Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import pytest
import os

from Tensile.ReplacementKernels import ReplacementKernels

def test_DefaultInstance():
    assert ReplacementKernels.Get("asdf") is None

    myReplacement = ReplacementKernels.Get("Cijk_Alik_Bljk_SB_MT64x128x32_SE_1LDSB0_APM1_ABV0_ACED0_AF0EM8_AF1EM1_AMAS3_ASAE01_ASCE01_ASEM8_AAC0_BL1_DTL0_DVO0_EPS1_FL0_GRVW4_GSU1_GSUASB_GLS0_ISA908_IU1_K1_KLA_LBSPP0_LPA0_LPB0_LDL1_LRVW4_MAC_MIAV0_MDA2_NTC0_NTD0_NEPBS0_NLCA1_NLCB1_ONLL1_OPLV0_PK0_PAP0_PGR1_PLR1_RK1_SIA1_SS0_SU32_SUM0_SUS256_SCIUI1_SPO0_SRVW0_SSO0_SVW4_SNLL0_TT4_4_TLDS0_USFGRO1_VAW1_VS1_VW4_WSGRA0_WSGRB0_WS64_WG16_32_1_WGM8")

    assert os.path.isfile(myReplacement)
    assert os.path.isabs(myReplacement)

def replacementDir(dirname):
    scriptDir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(scriptDir, 'replacement', dirname)

def test_BadFile():
    with pytest.raises(RuntimeError):
        obj = ReplacementKernels(replacementDir('bad_file'), 'default')
        obj.get("asdf")

def test_DuplicateKernel():
    with pytest.raises(RuntimeError):
        obj = ReplacementKernels(replacementDir('duplicate_kernel'), 'default')
        obj.get("asdf")

goodObjs = [ReplacementKernels(replacementDir('known_kernels_v3'), "default")]

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

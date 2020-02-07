import pytest
from Tensile.SolutionStructs import Convolution


@pytest.mark.parametrize("targetStr,replStr1,replStr2,expected",[
        ('NCHW', 'C,N,ZYX', 'K,C,DHW', 'CKYX'),
        ('NCHW', 'C', 'N', 'CNHW'),
        ('NCHW', '', '', 'NCHW'),
        ('NCHW', 'x', 'x', 'NCHW'),
        ('KCYX', 'C,K,YX', 'N,C,HW', 'CNHW'),
        ('KCYX', 'CKYX', 'NCHW', 'CNHW'),
        ])
def test_string_swap(targetStr,replStr1,replStr2,expected):
    assert(Convolution.swap(targetStr,replStr1,replStr2) == expected)

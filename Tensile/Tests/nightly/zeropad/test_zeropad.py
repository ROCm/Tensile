import Tensile.Tensile as Tensile

def test_zeropad_3x2(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/zeropad/test_zeropad_3x2.yaml"), tmpdir.strpath])

def test_zeropad_simple_2x0(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/zeropad/test_zeropad_simple_2x0.yaml"), tmpdir.strpath])

def test_zeropad_simple_0x2(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/zeropad/test_zeropad_simple_0x2.yaml"), tmpdir.strpath])

def test_zeropad_simple_2x3_tt8x8(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/zeropad/test_zeropad_simple_2x3_tt8x8.yaml"), tmpdir.strpath])

def test_zeropad_simple_3x2_sum5(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/zeropad/test_zeropad_simple_3x2_sum5.yaml"), tmpdir.strpath])

def test_zeropad_simple_2x3(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/zeropad/test_zeropad_simple_2x3.yaml"), tmpdir.strpath])


import Tensile.Tensile as Tensile

def test_simple_forward_cnhw_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/simple_forward_cnhw_defaults.yaml"), tmpdir.strpath])

def test_forward_nchw_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/forward_nchw_defaults.yaml"), tmpdir.strpath])

def test_simple_forward_nchw_3x1(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/simple_forward_nchw_3x1.yaml"), tmpdir.strpath])

def test_forward_nchw_kcyx_cnhw(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/forward_nchw_kcyx_cnhw.yaml"), tmpdir.strpath])

def test_simple_forward_nchw_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/simple_forward_nchw_defaults.yaml"), tmpdir.strpath])


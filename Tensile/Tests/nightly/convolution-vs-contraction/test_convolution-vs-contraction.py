import Tensile.Tensile as Tensile

def test_simple_forward_nchw_ckyx_3xN(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/simple_forward_nchw_ckyx_3xN.yaml"), tmpdir.strpath])

def test_simple_forward_cnhw_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/simple_forward_cnhw_defaults.yaml"), tmpdir.strpath])

def test_example_0(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/example_0.yaml"), tmpdir.strpath])

def test_forward_nchw_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/forward_nchw_defaults.yaml"), tmpdir.strpath])

def test_simple_forward_nchw_ckyx_3x4(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/simple_forward_nchw_ckyx_3x4.yaml"), tmpdir.strpath])

def test_simple_forward_nchw_3x1(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/simple_forward_nchw_3x1.yaml"), tmpdir.strpath])

def test_simple_strides(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/simple_strides.yaml"), tmpdir.strpath])

def test_forward_nchw_kcyx_cnhw(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/forward_nchw_kcyx_cnhw.yaml"), tmpdir.strpath])

def test_simple_forward_nchw_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution-vs-contraction/simple_forward_nchw_defaults.yaml"), tmpdir.strpath])


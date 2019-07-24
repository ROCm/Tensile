import Tensile.Tensile as Tensile

def test_conv_act1d_filter3d_simple(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution/test_conv_act1d_filter3d_simple.yaml"), tmpdir.strpath])

def test_conv_act1d_filter1d(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution/test_conv_act1d_filter1d.yaml"), tmpdir.strpath])

def test_conv_act1d_filter2d_simple(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution/test_conv_act1d_filter2d_simple.yaml"), tmpdir.strpath])

def test_conv_act1d_filter1d_simple(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution/test_conv_act1d_filter1d_simple.yaml"), tmpdir.strpath])

def test_conv_act2d_filter1d(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution/test_conv_act2d_filter1d.yaml"), tmpdir.strpath])

def test_conv_act2d_filter1d_simple(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution/test_conv_act2d_filter1d_simple.yaml"), tmpdir.strpath])

def test_conv_act1d_filter5d_simple(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/convolution/test_conv_act1d_filter5d_simple.yaml"), tmpdir.strpath])


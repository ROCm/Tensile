import Tensile.Tensile as Tensile

def test_2(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/use_initial_strides/test_2.yaml"), tmpdir.strpath])

def test_1(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/use_initial_strides/test_1.yaml"), tmpdir.strpath])

def test_strides(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/use_initial_strides/test_strides.yaml"), tmpdir.strpath])

def test_strides1(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/use_initial_strides/test_strides1.yaml"), tmpdir.strpath])

def test_simple_use_initial_strides_1(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/use_initial_strides/simple_use_initial_strides_1.yaml"), tmpdir.strpath])


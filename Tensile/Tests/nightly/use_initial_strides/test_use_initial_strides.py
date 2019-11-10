import Tensile.Tensile as Tensile

def test_simple_use_initial_strides_1(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("use_initial_strides/simple_use_initial_strides_1.yaml"), tmpdir.strpath])


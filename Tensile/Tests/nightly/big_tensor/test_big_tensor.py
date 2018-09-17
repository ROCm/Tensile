import Tensile.Tensile as Tensile

def test_bigskinny_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/big_tensor/bigskinny_nt.yaml"), tmpdir.strpath])


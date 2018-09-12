import Tensile.Tensile as Tensile

def test_hgemm_asem2_asm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/assertions/test_hgemm_asem2_asm.yaml"), tmpdir.strpath])


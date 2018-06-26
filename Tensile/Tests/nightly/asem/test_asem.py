import Tensile.Tensile as Tensile

def test_hgemm_asem2_asm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/asem/test_hgemm_asem2_asm.yaml"), tmpdir.strpath])


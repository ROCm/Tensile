import Tensile.Tensile as Tensile

def test_hgemm_asm_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/direct_to_lds/hgemm_asm_nn.yaml"), tmpdir.strpath])


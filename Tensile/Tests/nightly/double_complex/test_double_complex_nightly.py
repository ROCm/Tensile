import Tensile.Tensile as Tensile


def test_zgemm_asm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_asm.yaml"), tmpdir.strpath])


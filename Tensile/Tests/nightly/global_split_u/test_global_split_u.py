import Tensile.Tensile as Tensile

def test_hgemm_gsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/global_split_u/hgemm_gsu.yaml"), tmpdir.strpath])

def test_sgemm_gsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/global_split_u/sgemm_gsu.yaml"), tmpdir.strpath])


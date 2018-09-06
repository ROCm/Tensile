import Tensile.Tensile as Tensile

def test_sgemm_lsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/local_split_u/sgemm_lsu.yaml"), tmpdir.strpath])

def test_dgemm_lsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/local_split_u/dgemm_lsu.yaml"), tmpdir.strpath])

def test_hgemm_lsu_grvw2(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/local_split_u/hgemm_lsu_grvw2.yaml"), tmpdir.strpath])

def test_hgemm_lsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/local_split_u/hgemm_lsu.yaml"), tmpdir.strpath])


import Tensile.Tensile as Tensile

def test_dgemm_lsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/local_split_u/test_dgemm_lsu.yaml"), tmpdir.strpath])

def test_hgemm_lsu_grvw1(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/local_split_u/test_hgemm_lsu_grvw1.yaml"), tmpdir.strpath])

def test_hgemm_lsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/local_split_u/test_hgemm_lsu.yaml"), tmpdir.strpath])

def test_sgemm_lsu(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/local_split_u/test_sgemm_lsu.yaml"), tmpdir.strpath])


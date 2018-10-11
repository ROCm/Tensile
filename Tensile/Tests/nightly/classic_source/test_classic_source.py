import Tensile.Tensile as Tensile

def test_hgemm_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_hgemm_nn.yaml"), tmpdir.strpath])

def test_hgemm_tn_tt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_hgemm_tn_tt.yaml"), tmpdir.strpath])

def test_sgemm_vectors(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_sgemm_vectors.yaml"), tmpdir.strpath])

def test_hgemm_vectors(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_hgemm_vectors.yaml"), tmpdir.strpath])

def test_dgemm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_dgemm.yaml"), tmpdir.strpath])

def test_hgemm_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_hgemm_nt.yaml"), tmpdir.strpath])

def test_sgemm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_sgemm.yaml"), tmpdir.strpath])


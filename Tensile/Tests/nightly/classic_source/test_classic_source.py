import Tensile.Tensile as Tensile

def test_dgemm_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_dgemm_defaults.yaml"), tmpdir.strpath])

def test_hgemm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_hgemm.yaml"), tmpdir.strpath])

def test_hgemm_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_hgemm_defaults.yaml"), tmpdir.strpath])

def test_sgemm_vectors(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_sgemm_vectors.yaml"), tmpdir.strpath])

def test_hgemm_vectors(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_hgemm_vectors.yaml"), tmpdir.strpath])

def test_sgemm_defaults(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_sgemm_defaults.yaml"), tmpdir.strpath])

def test_sgemm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/classic_source/test_sgemm.yaml"), tmpdir.strpath])


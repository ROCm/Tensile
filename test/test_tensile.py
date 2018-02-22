import Tensile.Tensile as Tensile

# defaults
def test_hgemm_defaults(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm_defaults.yaml"), tmpdir.strpath])
def test_sgemm_defaults(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_sgemm_defaults.yaml"), tmpdir.strpath])
def test_dgemm_defaults(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_dgemm_defaults.yaml"), tmpdir.strpath])

# thorough tests
def test_hgemm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm.yaml"), tmpdir.strpath])
def test_sgemm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_sgemm.yaml"), tmpdir.strpath])

# vectors
def test_hgemm_vectors(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_hgemm_vectors.yaml"), tmpdir.strpath])
def test_sgemm_vectors(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_sgemm_vectors.yaml"), tmpdir.strpath])

# tensor contractions
def test_tensor_contraction(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_tensor_contraction.yaml"), tmpdir.strpath])

# assembly
def test_sgemm_asm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_sgemm_asm.yaml"), tmpdir.strpath])
def test_dgemm_asm(tmpdir):
    Tensile.Tensile([Tensile.TensileConfigPath("test_dgemm_asm.yaml"), tmpdir.strpath])

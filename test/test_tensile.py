import Tensile.Tensile as Tensile

def test_rocblas_sgemm(tmpdir):
  Tensile.Tensile([Tensile.TensileConfigPath("rocblas_sgemm.yaml"), tmpdir.strpath])


def test_rocblas_dgemm(tmpdir):
  Tensile.Tensile([Tensile.TensileConfigPath("rocblas_dgemm.yaml"), tmpdir.strpath])


def test_rocblas_cgemm(tmpdir):
  Tensile.Tensile([Tensile.TensileConfigPath("rocblas_cgemm.yaml"), tmpdir.strpath])


def test_rocblas_zgemm(tmpdir):
  Tensile.Tensile([Tensile.TensileConfigPath("rocblas_zgemm.yaml"), tmpdir.strpath])


def test_sgemm_5760(tmpdir):
  Tensile.Tensile([Tensile.TensileConfigPath("sgemm_5760.yaml"), tmpdir.strpath])



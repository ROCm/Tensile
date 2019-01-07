import Tensile.Tensile as Tensile

def test_igemm_tt(tmpdir):
  Tensile.Tensile([Tensile.TensileTestPath("special/igemm/igemm_hpa_hip_tt.yaml"), tmpdir.strpath])

def test_igemm_nn(tmpdir):
  Tensile.Tensile([Tensile.TensileTestPath("special/igemm/igemm_hpa_hip_nn.yaml"), tmpdir.strpath])

def test_igemm_lsu(tmpdir):
  Tensile.Tensile([Tensile.TensileTestPath("special/igemm/igemm_hpa_hip_lsu.yaml"), tmpdir.strpath])



import Tensile.Tensile as Tensile

def test_igemm_hpa_hip_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/igemm_hpa_vega20_nn.yaml"), tmpdir.strpath])

def test_igemm_hpa_vega20_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/igemm_hpa_vega20_nt.yaml"), tmpdir.strpath])

def test_igemm_hpa_vega20_tt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/igemm_hpa_vega20_tt.yaml"), tmpdir.strpath])

def test_igemm_hpa_vega20_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/igemm_hpa_vega20_tn.yaml"), tmpdir.strpath])

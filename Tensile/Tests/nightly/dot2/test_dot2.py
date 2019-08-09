import Tensile.Tensile as Tensile

def test_hgemm_hpa_dot2_tn_2(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/dot2/hgemm_hpa_dot2_tn_2.yaml"), tmpdir.strpath])

def test_hgemm_hpa_dot2_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/dot2/hgemm_hpa_dot2_tn.yaml"), tmpdir.strpath])

def test_hgemm_hpa_dot2_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/dot2/hgemm_hpa_dot2_nn.yaml"), tmpdir.strpath])


import Tensile.Tensile as Tensile

def test_bfloat8_source_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/bfloat8/bfloat8_hpa_source_nn.yaml"), tmpdir.strpath])

def test_bfloat8_source_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/bfloat8/bfloat8_hpa_source_nt.yaml"), tmpdir.strpath])

def test_bfloat8_source_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/bfloat8/bfloat8_hpa_source_tn.yaml"), tmpdir.strpath])

def test_bfloat8_source_tt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/bfloat8/bfloat8_hpa_source_tt.yaml"), tmpdir.strpath])


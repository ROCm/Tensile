import Tensile.Tensile as Tensile

def test_sgemm_asm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_sgemm_asm.yaml"), tmpdir.strpath])

def test_hgemm_asm_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_asm_nn.yaml"), tmpdir.strpath])

def test_dgemm_asm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_dgemm_asm.yaml"), tmpdir.strpath])

def test_hgemm_hpa_asm_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_hpa_asm_tn.yaml"), tmpdir.strpath])

def test_hgemm_hpa_asm_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_hpa_asm_nn.yaml"), tmpdir.strpath])

def test_hgemm_asm_tt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_asm_tt.yaml"), tmpdir.strpath])

def test_hgemm_hpa_asm_tt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_hpa_asm_tt.yaml"), tmpdir.strpath])

def test_hgemm_asm_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_asm_nt.yaml"), tmpdir.strpath])

def test_hgemm_asm_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_asm_tn.yaml"), tmpdir.strpath])

def test_hgemm_hpa_asm_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_hpa_asm_nt.yaml"), tmpdir.strpath])


def test_hgemm_hpa_src_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_hpa_src_tn.yaml"), tmpdir.strpath])

def test_hgemm_hpa_src_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_hpa_src_nn.yaml"), tmpdir.strpath])

def test_hgemm_hpa_src_tt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_hpa_src_tt.yaml"), tmpdir.strpath])

def test_hgemm_hpa_src_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("pre_checkin/test_hgemm_hpa_src_nt.yaml"), tmpdir.strpath])


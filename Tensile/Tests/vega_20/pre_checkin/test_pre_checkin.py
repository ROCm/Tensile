import Tensile.Tensile as Tensile

def test_igemm_asm_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/pre_checkin/igemm_asm_nn.yaml"), tmpdir.strpath])

def test_igemm_asm_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/pre_checkin/igemm_asm_nt.yaml"), tmpdir.strpath])

def test_igemm_asm_tt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/pre_checkin/igemm_asm_tt.yaml"), tmpdir.strpath])

def test_igemm_asm_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/pre_checkin/igemm_asm_tn.yaml"), tmpdir.strpath])

def test_gemm_ldc_equals_ldd(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("vega_20/pre_checkin/gemm_ldc_equals_ldd.yaml"), tmpdir.strpath])
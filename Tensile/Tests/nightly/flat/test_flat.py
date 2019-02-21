import Tensile.Tensile as Tensile

def test_sgemm_asm_flat_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/flat/test_sgemm_asm_flat_tn.yaml"), tmpdir.strpath])

def test_dgemm_asm_flat(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/flat/test_dgemm_asm_flat.yaml"), tmpdir.strpath])

def test_sgemm_asm_flat_tt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/flat/test_sgemm_asm_flat_tt.yaml"), tmpdir.strpath])

def test_sgemm_asm_flat_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/flat/test_sgemm_asm_flat_nt.yaml"), tmpdir.strpath])

def test_sgemm_asm_flat(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/flat/test_sgemm_asm_flat.yaml"), tmpdir.strpath])


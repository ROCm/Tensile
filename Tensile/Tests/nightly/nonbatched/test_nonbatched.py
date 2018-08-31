import Tensile.Tensile as Tensile

def test_sgemm_asm_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/nonbatched/sgemm_asm_nt.yaml"), tmpdir.strpath])

def test_sgemm_asm_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/nonbatched/sgemm_asm_nn.yaml"), tmpdir.strpath])

def test_sgemm_asm_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/nonbatched/sgemm_asm_tn.yaml"), tmpdir.strpath])

def test_sgemm_asm_tt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/nonbatched/sgemm_asm_tt.yaml"), tmpdir.strpath])


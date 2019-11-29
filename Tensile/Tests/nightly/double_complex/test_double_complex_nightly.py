import Tensile.Tensile as Tensile

def test_zgemm_hip_source_nn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_hip_source_nn.yaml"), tmpdir.strpath])

def test_zgemm_hip_source_nt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_hip_source_nt.yaml"), tmpdir.strpath])

def test_zgemm_hip_source_nc(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_hip_source_nc.yaml"), tmpdir.strpath])

def test_zgemm_hip_source_cn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_hip_source_cn.yaml"), tmpdir.strpath])

def test_zgemm_hip_source_tc(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_hip_source_tc.yaml"), tmpdir.strpath])

def test_zgemm_asm(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_asm.yaml"), tmpdir.strpath])

def test_zgemm_hip_source_ct(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_hip_source_ct.yaml"), tmpdir.strpath])

def test_zgemm_hip_source_tn(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_hip_source_tn.yaml"), tmpdir.strpath])

def test_zgemm_hip_source_tt(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_hip_source_tt.yaml"), tmpdir.strpath])

def test_zgemm_hip_source_cc(tmpdir):
 Tensile.Tensile([Tensile.TensileTestPath("nightly/double_complex/zgemm_hip_source_cc.yaml"), tmpdir.strpath])


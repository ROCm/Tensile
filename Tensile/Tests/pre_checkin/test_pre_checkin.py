import Tensile.Tensile as Tensile


def test_dgemm_asm(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/dgemm_asm.yaml"), tmpdir.strpath
    ])


def test_sgemm_asm_tn_bigk(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/sgemm_asm_tn_bigk.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_hpa_iu2_asm_tt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_hpa_iu2_asm_tt.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_hpa_iu2_asm_nn(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_hpa_iu2_asm_nn.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_hpa_asm_tn(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_hpa_asm_tn.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_asm_nt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_asm_nt.yaml"),
        tmpdir.strpath
    ])


def test_igemm_hpa_hip_tn(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/igemm_hpa_hip_tn.yaml"),
        tmpdir.strpath
    ])


def test_sgemm_asm_nt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/sgemm_asm_nt.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_hpa_iu2_asm_tn(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_hpa_iu2_asm_tn.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_hpa_iu2_asm_nt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_hpa_iu2_asm_nt.yaml"),
        tmpdir.strpath
    ])


def test_sgemm_asm_nn(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/sgemm_asm_nn.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_asm_nn(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_asm_nn.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_asm_tt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_asm_tt.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_asm_tn(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_asm_tn.yaml"),
        tmpdir.strpath
    ])


def test_igemm_hpa_hip_nn(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/igemm_hpa_hip_nn.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_hpa_asm_nt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_hpa_asm_nt.yaml"),
        tmpdir.strpath
    ])


def test_igemm_hpa_hip_tt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/igemm_hpa_hip_tt.yaml"),
        tmpdir.strpath
    ])


def test_sgemm_asm_tn(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/sgemm_asm_tn.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_hpa_asm_tt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_hpa_asm_tt.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_hpa_asm_nn(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/hgemm_hpa_asm_nn.yaml"),
        tmpdir.strpath
    ])


def test_sgemm_asm_tt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/sgemm_asm_tt.yaml"),
        tmpdir.strpath
    ])


def test_igemm_hpa_hip_nt(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("pre_checkin/igemm_hpa_hip_nt.yaml"),
        tmpdir.strpath
    ])

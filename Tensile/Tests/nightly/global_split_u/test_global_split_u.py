import Tensile.Tensile as Tensile


def test_hgemm_gsu(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("nightly/global_split_u/hgemm_gsu.yaml"),
        tmpdir.strpath
    ])


def test_sgemm_gsu_beta2(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("nightly/global_split_u/sgemm_gsu_beta2.yaml"),
        tmpdir.strpath
    ])


def test_sgemm_gsu_beta1(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("nightly/global_split_u/sgemm_gsu_beta1.yaml"),
        tmpdir.strpath
    ])


def test_sgemm_gsu_beta0(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath("nightly/global_split_u/sgemm_gsu_beta0.yaml"),
        tmpdir.strpath
    ])

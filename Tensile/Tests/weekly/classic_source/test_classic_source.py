import Tensile.Tensile as Tensile


def test_hgemm_vectors(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath(
            "weekly/classic_source/test_hgemm_vectors.yaml"), tmpdir.strpath
    ])


def test_sgemm_vectors(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath(
            "weekly/classic_source/test_sgemm_vectors.yaml"), tmpdir.strpath
    ])

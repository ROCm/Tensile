import Tensile.Tensile as Tensile


def test_dgemm_fractional_tile_sweep(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath(
            "nightly/fractional/test_dgemm_fractional_tile_sweep.yaml"),
        tmpdir.strpath
    ])


def test_hgemm_fractional_tile_sweep(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath(
            "nightly/fractional/test_hgemm_fractional_tile_sweep.yaml"),
        tmpdir.strpath
    ])


def test_sgemm_fractional_edge(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath(
            "nightly/fractional/test_sgemm_fractional_edge.yaml"),
        tmpdir.strpath
    ])


def test_sgemm_fractional_tile_sweep(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath(
            "nightly/fractional/test_sgemm_fractional_tile_sweep.yaml"),
        tmpdir.strpath
    ])

import Tensile.Tensile as Tensile


def test_igemm_lsu(tmpdir):
    Tensile.Tensile([
        Tensile.TensileTestPath(
            "vega_20/nightly/local_split_u/igemm_lsu.yaml"), tmpdir.strpath
    ])
